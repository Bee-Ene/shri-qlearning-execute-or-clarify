"""
SHRI Decision Node 
ROS2 Humble + Gazebo Classic 11 + Unitree Go2

"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import numpy as np
import os
import time
import threading
from enum import Enum

from shri_decision.shri_state import (
    build_state, decode_state, state_to_string,
    parse_command, generate_clarification_question,
    Feasibility, is_tts_echo, BLOCKED_DISTANCE_THRESHOLD
)


class Phase(Enum):
    WAITING    = 0
    DECIDING   = 1
    MOVING     = 2
    FEEDBACK   = 3
    CLAR       = 4
    RECOVERING = 5


class SHRIDecisionNode(Node):

    def __init__(self):
        super().__init__("shri_decision_node")

        # ── Load Q-table ──────────────────────────────────────────────
        q_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "q_table.npy")
        if not os.path.exists(q_path):
            self.get_logger().error("Q-table not found at: " + q_path)
            raise FileNotFoundError(q_path)
        self.Q = np.load(q_path)
        self.get_logger().info(
            "Q-table loaded: shape=" + str(self.Q.shape))

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            String,    "/command_text",   self.command_callback,  10)
        self.create_subscription(
            LaserScan, "/scan",           self.scan_callback,     10)
        self.create_subscription(
            String,    "/human_feedback", self.feedback_callback, 10)

        # ── Publishers ────────────────────────────────────────────────
        self.pub_vel  = self.create_publisher(Twist,  "/cmd_vel",             10)
        self.pub_clar = self.create_publisher(String, "/clarification_query", 10)
        self.pub_stat = self.create_publisher(String, "/shri_status",         10)

        # ── Episode state ─────────────────────────────────────────────
        self.phase   = Phase.WAITING
        self.command = None
        self.state   = None   # int 0-17, None when no episode active
        self.n_clar  = 0
        self.MAX_CLAR = 3

        # ── Sensor state ──────────────────────────────────────────────
        self.path_clear      = True
        self.obs_dist        = float("inf")
        self.perc_conf       = 0.88
        self.target_visible  = True
        self.n_targets       = 1
        self._scan_ready     = False
        self._start_time     = time.time()
        self.SCAN_DELAY      = 2.0        # ignore LiDAR first 2s

        # ── Startup brake ─────────────────────────────────────────────
        # Publishes zero velocity to prevent diff_drive drift at spawn.
        # Stops the moment a real command arrives (_brake_active flag).
        self._brake_active = True
        threading.Thread(target=self._run_brake, daemon=True).start()

        # ── Timer state ───────────────────────────────────────────────
        self._stop_timer           = None
        self._stop_timer_fired     = False
        self._recovery_timer       = None
        self._recovery_timer_fired = False
        self._redecide_timer       = None
        self._redecide_timer_fired = False

        self.get_logger().info("=" * 60)
        self.get_logger().info("  SHRI Decision Node — FINAL VERSION")
        self.get_logger().info("  Q-table: " + str(self.Q.shape[0]) +
                               " states × " + str(self.Q.shape[1]) + " actions")
        self.get_logger().info("  Startup brake active for 3 seconds...")
        self.get_logger().info("=" * 60)

    # ─────────────────────────────────────────────────────────────────
    # STARTUP BRAKE
    # ─────────────────────────────────────────────────────────────────

    def _run_brake(self):
        """Publish zero velocity until brake flag cleared or 3s elapsed."""
        time.sleep(0.5)   # let node finish initialising
        stop     = Twist()
        deadline = time.time() + 3.0
        while time.time() < deadline and self._brake_active and rclpy.ok():
            try:
                self.pub_vel.publish(stop)
            except Exception:
                pass
            time.sleep(0.1)
        self._brake_active = False
        self.get_logger().info("Startup brake done. Ready for commands.")
        self._publish_status("READY - waiting for command")

    # ─────────────────────────────────────────────────────────────────
    # SENSOR
    # ─────────────────────────────────────────────────────────────────

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR. Ignore first SCAN_DELAY seconds (startup settling)."""
        if not self._scan_ready:
            if time.time() - self._start_time < self.SCAN_DELAY:
                return
            self._scan_ready = True
            self.get_logger().info("LiDAR active.")

        ranges = [r for r in msg.ranges
                  if not (np.isinf(r) or np.isnan(r))]
        if not ranges:
            return

        n = len(msg.ranges)
        # Front 60-degree arc (robot faces X+, so front = centre of scan)
        front = [
            msg.ranges[i]
            for i in range(int(n * 0.42), int(n * 0.58))
            if not (np.isinf(msg.ranges[i]) or np.isnan(msg.ranges[i]))
        ]
        if front:
            self.obs_dist   = round(min(front), 3)
            self.path_clear = (self.obs_dist > BLOCKED_DISTANCE_THRESHOLD)

        valid_ratio   = len(ranges) / max(n, 1)
        self.perc_conf = float(np.clip(0.55 + 0.43 * valid_ratio, 0.10, 0.98))

    # ─────────────────────────────────────────────────────────────────
    # COMMAND CALLBACK
    # ─────────────────────────────────────────────────────────────────

    def command_callback(self, msg: String):
        """Receives a new locomotion command and starts an episode."""
        # Stop startup brake immediately
        self._brake_active = False

        if self.phase != Phase.WAITING:
            self.get_logger().warn(
                "Command received while busy (" + self.phase.name +
                ") — ignored. Wait for current episode to finish.")
            return

        raw = msg.data.strip()
        if not raw:
            return

        # Reject robot's own TTS output picked up by microphone
        if is_tts_echo(raw):
            self.get_logger().info("TTS echo rejected: '" + raw + "'")
            return

        self.get_logger().info("=" * 60)
        self.get_logger().info("  NEW COMMAND: '" + raw + "'")
        self.get_logger().info("=" * 60)

        self.command = parse_command(raw)
        self.state   = None    # will be set in _decide()
        self.n_clar  = 0
        self.phase   = Phase.DECIDING

        if self.command["intent"] == "UNKNOWN":
            self.get_logger().warn(
                "Could not parse intent from: '" + raw + "'")
            self._clarify_unknown()
            return

        self.get_logger().info(
            "  intent="   + str(self.command["intent"]) +
            "  dir="    + str(self.command["direction"]) +
            "  target=" + str(self.command["target"]) +
            "  dist="   + str(self.command["distance"]) +
            "  conf="   + str(round(self.command["parse_confidence"], 2)) +
            "  missing=" + str(self.command["missing_params"]))

        self._decide()

    def _clarify_unknown(self):
        """Ask human to rephrase an unrecognised command."""
        question = (
            "I did not understand that command. "
            "Please try: move forward, move backward, "
            "turn left, turn right, or go to the chair.")
        self._publish_clarification(question)
        self.phase = Phase.CLAR
        self._publish_status("CLARIFYING - please rephrase your command")

    # ─────────────────────────────────────────────────────────────────
    # DECISION
    # ─────────────────────────────────────────────────────────────────

    def _decide(self):
        """Build state from current command + sensor readings, apply Q-policy."""
        if self.command is None:
            return

        # Build discrete state
        state = build_state(
            intent               = self.command["intent"],
            present_params       = self.command["present_params"],
            missing_params       = self.command["missing_params"],
            parse_confidence     = self.command["parse_confidence"],
            path_clear           = self.path_clear,
            target_visible       = self.target_visible,
            num_target_matches   = self.n_targets,
            perception_confidence= self.perc_conf,
        )
        self.state = state   # always set before any decode_state() call

        # Greedy Q-policy
        action = int(np.argmax(self.Q[state]))
        c, cl, f = decode_state(state)

        self.get_logger().info(
            "  State [" + str(state) + "] " + state_to_string(state) +
            "  obs=" + str(self.obs_dist) + "m" +
            "  clear=" + str(self.path_clear))
        self.get_logger().info(
            "  Q(EXECUTE)=" + str(round(self.Q[state, 0], 3)) +
            "  Q(CLARIFY)=" + str(round(self.Q[state, 1], 3)) +
            "  → " + ("EXECUTE" if action == 0 else "CLARIFY"))

        if action == 0:
            self._execute()
        else:
            self._clarify()

    # ─────────────────────────────────────────────────────────────────
    # EXECUTE
    # ─────────────────────────────────────────────────────────────────

    def _execute(self):
        """Translate parsed command into Twist and publish to /cmd_vel."""
        cmd     = self.command
        twist   = Twist()
        LINEAR  = 0.3    # m/s
        ANGULAR = 0.5    # rad/s (~57 degrees in 2 seconds)

        intent = cmd["intent"]
        dirn   = cmd["direction"]
        dist   = cmd["distance"] or 1.0

        if intent == "MOVE":
            if   dirn == "forward":  twist.linear.x  =  LINEAR
            elif dirn == "backward": twist.linear.x  = -LINEAR
            elif dirn == "left":     twist.angular.z =  ANGULAR
            elif dirn == "right":    twist.angular.z = -ANGULAR
        elif intent == "TURN":
            if   dirn == "left":  twist.angular.z =  ANGULAR
            elif dirn == "right": twist.angular.z = -ANGULAR
        elif intent == "NAVIGATE_TO":
            twist.linear.x = LINEAR
        # STOP / SIT / STAND: zero twist; robot stays still

        # Duration
        if twist.linear.x != 0:
            duration = dist / LINEAR
        elif twist.angular.z != 0:
            duration = 2.0
        else:
            duration = 0.5

        # Human-readable action description
        if intent == "MOVE":
            if   dirn == "forward":  desc = "Moving FORWARD"
            elif dirn == "backward": desc = "Moving BACKWARD"
            elif dirn == "left":     desc = "Rotating LEFT"
            elif dirn == "right":    desc = "Rotating RIGHT"
            else:                    desc = "Moving"
        elif intent == "TURN":
            desc = "Turning " + str(dirn).upper()
        elif intent == "NAVIGATE_TO":
            desc = "Moving toward " + str(cmd["target"])
        else:
            desc = str(intent)

        self.pub_vel.publish(twist)
        self.phase = Phase.MOVING

        self.get_logger().info("  EXECUTING: " + desc +
            "  lin.x=" + str(round(twist.linear.x, 2)) +
            "  ang.z=" + str(round(twist.angular.z, 2)) +
            "  for "   + str(round(duration, 1)) + "s")
        self.get_logger().info(
            "  → Type or say YES if correct, NO if wrong.")
        self._publish_status(
            "EXECUTING: " + desc +
            " | Say YES if correct, NO if wrong")

        self._stop_timer_fired = False
        self._stop_timer = self.create_timer(duration, self._on_stop_timer)

    def _on_stop_timer(self):
        if self._stop_timer_fired:
            return
        self._stop_timer_fired = True
        if self._stop_timer:
            self._stop_timer.cancel()
            self._stop_timer = None
        self.pub_vel.publish(Twist())   # explicit brake
        self.get_logger().info("  Movement complete. Robot stopped.")
        self.get_logger().info("  → Type or say YES if correct, NO if wrong.")
        self.phase = Phase.FEEDBACK
        self._publish_status("WAITING FOR FEEDBACK | Say YES or NO")

    # ─────────────────────────────────────────────────────────────────
    # CLARIFY
    # ─────────────────────────────────────────────────────────────────

    def _clarify(self):
        """Ask the human a targeted question to resolve the ambiguity."""
        if self.n_clar >= self.MAX_CLAR:
            self.get_logger().warn(
                "  Max clarifications (" + str(self.MAX_CLAR) +
                ") reached. Resetting episode.")
            self._publish_status(
                "EPISODE FAILED - too many clarifications. Give new command.")
            self._reset()
            return

        # self.state is guaranteed set by _decide() before _clarify() is called
        if self.state is None:
            self.get_logger().error(
                "  _clarify called with state=None. Resetting.")
            self._reset()
            return

        self.n_clar += 1
        c, cl, f = decode_state(self.state)
        question  = generate_clarification_question(
            self.command["intent"],
            self.command["missing_params"],
            f)

        self._publish_clarification(question)
        self.phase = Phase.CLAR

        self.get_logger().info(
            "  CLARIFY (" + str(self.n_clar) + "/" +
            str(self.MAX_CLAR) + "): " + question)
        self._publish_status(
            "CLARIFYING (" + str(self.n_clar) + "/" +
            str(self.MAX_CLAR) + ") | " + question)

    # ─────────────────────────────────────────────────────────────────
    # FEEDBACK
    # ─────────────────────────────────────────────────────────────────

    def feedback_callback(self, msg: String):
        """Handles human response to execution or clarification."""
        response = msg.data.strip().lower()
        if not response:
            return
        if is_tts_echo(response):
            return

        self.get_logger().info("  Feedback: '" + response + "'")

        if self.phase == Phase.FEEDBACK:
            self._handle_exec_feedback(response)
        elif self.phase == Phase.CLAR:
            self._handle_clar_feedback(response)
        elif self.phase == Phase.RECOVERING:
            self.get_logger().info(
                "  Robot is recovering. Please wait.")
        else:
            self.get_logger().warn(
                "  Feedback received in phase " +
                self.phase.name + " — ignored.")

    def _handle_exec_feedback(self, response: str):
        """YES = success, NO = failure."""
        pos = any(w in response for w in
                  ["yes", "good", "correct", "great", "ok",
                   "done", "perfect", "right", "nice", "yeah"])
        neg = any(w in response for w in
                  ["no", "wrong", "stop", "bad", "incorrect",
                   "cancel", "not", "nope"])
        if pos:
            self.get_logger().info("  ✓ Confirmed. Episode complete.")
            self._publish_status("SUCCESS ✓ | Ready for next command")
            self._reset()
        elif neg:
            self.get_logger().info("  ✗ Rejected. Stopping robot.")
            self.pub_vel.publish(Twist())
            self._publish_status("FAILED ✗ | Ready for next command")
            self._reset()
        else:
            self.get_logger().info(
                "  Not understood. Please say YES or NO.")
            self._publish_status(
                "WAITING FOR FEEDBACK | Please say YES or NO")

    def _handle_clar_feedback(self, response: str):
        """Process human answer to clarification question."""
        cmd = self.command

        # Cancel words
        if any(w in response for w in
               ["cancel", "abort", "nevermind", "forget", "quit",
                "never mind"]):
            self.get_logger().info("  Command cancelled.")
            self.pub_vel.publish(Twist())
            self._publish_status("CANCELLED | Ready for next command")
            self._reset()
            return

        if response.strip() in ("stop", "stop.", "halt", "halt."):
            self.get_logger().info("  Stopped.")
            self.pub_vel.publish(Twist())
            self._publish_status("STOPPED | Ready for next command")
            self._reset()
            return

        # Detect direction word in response
        from shri_decision.shri_state import DIRECTIONS, DIRECTION_MAP
        recovery_dir = None
        words = response.split()
        for d in DIRECTIONS:
            if d in words:
                recovery_dir = DIRECTION_MAP.get(d, d)
                break

        # If BLOCKED and direction given → physical recovery move
        if self.state is not None:
            c, cl, f = decode_state(self.state)
            if f == Feasibility.BLOCKED and recovery_dir is not None:
                self.get_logger().info(
                    "  BLOCKED. Recovery move: " + recovery_dir)
                self._execute_recovery(recovery_dir)
                return

        # Update command direction from answer
        if recovery_dir is not None:
            cmd["direction"] = recovery_dir
            if "direction" in cmd["missing_params"]:
                cmd["missing_params"].remove("direction")
            if "direction" not in cmd["present_params"]:
                cmd["present_params"].append("direction")
            cmd["parse_confidence"] = min(
                cmd["parse_confidence"] + 0.20, 0.95)
            self.get_logger().info(
                "  Direction updated: " + str(cmd["direction"]))

        # Update target from answer
        for t in ["chair", "door", "table", "person", "marker"]:
            if t in response:
                cmd["target"] = "the " + t
                if "target" in cmd["missing_params"]:
                    cmd["missing_params"].remove("target")
                if "target" not in cmd["present_params"]:
                    cmd["present_params"].append("target")
                cmd["parse_confidence"] = min(
                    cmd["parse_confidence"] + 0.20, 0.95)
                self.get_logger().info(
                    "  Target updated: " + str(cmd["target"]))
                break

        # Affirmative words boost confidence
        if any(w in response for w in
               ["yes", "go", "proceed", "clear", "safe", "ok", "yeah"]):
            cmd["parse_confidence"] = min(
                cmd["parse_confidence"] + 0.25, 0.95)
            if any(w in response for w in ["clear", "safe"]):
                self.path_clear = True

        self.command = cmd
        self.phase   = Phase.DECIDING
        self.get_logger().info("  Re-evaluating with updated command...")
        self._decide()

    # ─────────────────────────────────────────────────────────────────
    # RECOVERY
    # ─────────────────────────────────────────────────────────────────

    def _execute_recovery(self, direction: str):
        """Move robot to clear obstacle, then re-evaluate."""
        twist   = Twist()
        LINEAR  = 0.3
        ANGULAR = 0.6

        if   direction == "forward":  twist.linear.x  =  LINEAR
        elif direction == "backward": twist.linear.x  = -LINEAR
        elif direction == "left":     twist.angular.z =  ANGULAR
        elif direction == "right":    twist.angular.z = -ANGULAR

        self.pub_vel.publish(twist)
        self.phase = Phase.RECOVERING
        self._publish_status(
            "RECOVERING : moving " + direction.upper() +
            " to clear obstacle...")

        self._recovery_timer_fired = False
        self._recovery_timer = self.create_timer(
            1.5, self._on_recovery_done)

    def _on_recovery_done(self):
        if self._recovery_timer_fired:
            return
        self._recovery_timer_fired = True
        if self._recovery_timer:
            self._recovery_timer.cancel()
            self._recovery_timer = None

        self.pub_vel.publish(Twist())   # brake
        self.get_logger().info(
            "  Recovery done. obs_dist=" + str(self.obs_dist) +
            "m  path_clear=" + str(self.path_clear))

        self._redecide_timer_fired = False
        self._redecide_timer = self.create_timer(
            0.5, self._on_redecide)

    def _on_redecide(self):
        if self._redecide_timer_fired:
            return
        self._redecide_timer_fired = True
        if self._redecide_timer:
            self._redecide_timer.cancel()
            self._redecide_timer = None

        if not self.path_clear:
            self.get_logger().warn(
                "  Path still BLOCKED after recovery "
                "(obs=" + str(self.obs_dist) + "m). "
                "Resetting episode.")
            self._publish_status(
                "FAILED : obstacle not cleared | Give new command")
            self._reset()
            return

        self.get_logger().info(
            "  Path clear after recovery. Re-deciding.")
        self.n_clar = 0          # fresh clarification budget
        self.phase  = Phase.DECIDING
        self._decide()

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _reset(self):
        """Reset all episode state. Robot ready for next command."""
        self.command = None
        self.state   = None
        self.n_clar  = 0
        self.phase   = Phase.WAITING
        self.get_logger().info("─" * 60)
        self.get_logger().info("  Episode complete. Ready for next command.")
        self.get_logger().info("─" * 60)
        self._publish_status("READY : waiting for command")

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_stat.publish(msg)

    def _publish_clarification(self, question: str):
        msg = String()
        msg.data = question
        self.pub_clar.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SHRIDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
