"""
Session Logger Node — SHRI Decision System

Automatically logs every interaction to a CSV file per user session.
This data feeds directly into the user study analysis and thesis charts.

Subscribes to all relevant topics and reconstructs episode records.

Usage:
    ros2 run shri_decision session_logger_node --ros-args -p user_id:=U01

Each session produces: session_U01_YYYY-MM-DD_HH-MM-SS.csv
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

import csv
import os
import time
import threading
import numpy as np
from datetime import datetime


LOG_DIR = os.path.expanduser("~/Documents/ros2_learning/session_logs")


class SessionLoggerNode(Node):

    def __init__(self):
        super().__init__("session_logger_node")

        # ── User ID parameter ─────────────────────────────────────────
        self.declare_parameter("user_id", "U01")
        self.declare_parameter("robotics_experience", "none")  # none/some/expert
        self.declare_parameter("age", 0)
        self.declare_parameter("gender", "unspecified")

        self.user_id     = self.get_parameter("user_id").value
        self.experience  = self.get_parameter("robotics_experience").value
        self.age         = self.get_parameter("age").value
        self.gender      = self.get_parameter("gender").value

        # ── CSV setup ─────────────────────────────────────────────────
        os.makedirs(LOG_DIR, exist_ok=True)
        ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"session_{self.user_id}_{ts}.csv"
        self.csv_path = os.path.join(LOG_DIR, filename)

        self.csv_fields = [
            "user_id", "age", "gender", "robotics_experience",
            "episode_number", "command_text", "intent_parsed",
            "decision", "n_clarifications", "clarification_questions",
            "human_feedback_given", "episode_outcome",
            "episode_duration_sec", "obs_dist_at_decision",
            "state_id", "state_name", "q_execute", "q_clarify",
            "timestamp_start", "timestamp_end",
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writeheader()

        self.get_logger().info("Session log: " + self.csv_path)

        # ── Subscriptions ─────────────────────────────────────────────
        self.create_subscription(String, "/command_text",
                                 self._cmd_cb,    10)
        self.create_subscription(String, "/shri_status",
                                 self._status_cb, 10)
        self.create_subscription(String, "/clarification_query",
                                 self._clar_cb,   10)
        self.create_subscription(String, "/human_feedback",
                                 self._feedback_cb, 10)
        self.create_subscription(LaserScan, "/scan",
                                 self._scan_cb,   10)

        # ── Episode state ─────────────────────────────────────────────
        self._ep_num         = 0
        self._ep_start       = None
        self._cmd_text       = ""
        self._intent         = ""
        self._decision       = ""
        self._n_clar         = 0
        self._clar_questions = []
        self._feedbacks      = []
        self._outcome        = ""
        self._obs_dist       = float("inf")
        self._state_id       = -1
        self._state_name     = ""
        self._q_exec         = 0.0
        self._q_clar         = 0.0
        self._in_episode     = False
        self._lock           = threading.Lock()

        self.get_logger().info("=" * 55)
        self.get_logger().info(
            f"  Session Logger started — User: {self.user_id}")
        self.get_logger().info("=" * 55)

    # ── Sensor callback ───────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        ranges = [r for r in msg.ranges
                  if not (np.isinf(r) or np.isnan(r))]
        if ranges:
            n = len(msg.ranges)
            front = [msg.ranges[i]
                     for i in range(int(n * 0.42), int(n * 0.58))
                     if not (np.isinf(msg.ranges[i]) or
                             np.isnan(msg.ranges[i]))]
            if front:
                self._obs_dist = round(min(front), 3)

    # ── Command received — new episode starts ─────────────────────────
    def _cmd_cb(self, msg: String):
        with self._lock:
            self._ep_num    += 1
            self._ep_start   = time.time()
            self._cmd_text   = msg.data.strip()
            self._intent     = ""
            self._decision   = ""
            self._n_clar     = 0
            self._clar_questions = []
            self._feedbacks  = []
            self._outcome    = ""
            self._in_episode = True
        self.get_logger().info(
            f"  Logging episode {self._ep_num}: '{self._cmd_text}'")

    # ── Status updates — capture decision and outcome ─────────────────
    def _status_cb(self, msg: String):
        status = msg.data.strip().upper()
        with self._lock:
            if not self._in_episode:
                return

            # Parse decision from first action status
            if "EXECUTING" in status and not self._decision:
                self._decision = "EXECUTE"

            elif "CLARIFYING" in status and not self._decision:
                self._decision = "CLARIFY"

            # Parse outcome
            if "SUCCESS" in status:
                self._outcome = "SUCCESS"
                self._close_episode()

            elif "FAILED" in status:
                if "TOO MANY" in status or "DISENGAGED" in status:
                    self._outcome = "FAILED_EXCESSIVE_CLARIF"
                else:
                    self._outcome = "FAILED_EXECUTION"
                self._close_episode()

    # ── Clarification question logged ─────────────────────────────────
    def _clar_cb(self, msg: String):
        with self._lock:
            if self._in_episode:
                self._n_clar += 1
                self._clar_questions.append(msg.data.strip())

    # ── Human feedback logged ─────────────────────────────────────────
    def _feedback_cb(self, msg: String):
        with self._lock:
            if self._in_episode:
                self._feedbacks.append(msg.data.strip())

    # ── Close and write episode record ────────────────────────────────
    def _close_episode(self):
        """Called within _lock. Writes completed episode to CSV."""
        if not self._in_episode:
            return
        self._in_episode = False
        duration = round(time.time() - self._ep_start, 2) \
            if self._ep_start else 0.0

        row = {
            "user_id":                self.user_id,
            "age":                    self.age,
            "gender":                 self.gender,
            "robotics_experience":    self.experience,
            "episode_number":         self._ep_num,
            "command_text":           self._cmd_text,
            "intent_parsed":          self._intent,
            "decision":               self._decision,
            "n_clarifications":       self._n_clar,
            "clarification_questions": " | ".join(self._clar_questions),
            "human_feedback_given":   " | ".join(self._feedbacks),
            "episode_outcome":        self._outcome,
            "episode_duration_sec":   duration,
            "obs_dist_at_decision":   self._obs_dist,
            "state_id":               self._state_id,
            "state_name":             self._state_name,
            "q_execute":              self._q_exec,
            "q_clarify":              self._q_clar,
            "timestamp_start":        datetime.fromtimestamp(
                self._ep_start).strftime("%H:%M:%S")
                if self._ep_start else "",
            "timestamp_end":          datetime.now().strftime("%H:%M:%S"),
        }

        try:
            with open(self.csv_path, "a", newline="",
                      encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                writer.writerow(row)
            self.get_logger().info(
                f"  Episode {self._ep_num} logged: "
                f"{self._outcome} | "
                f"{self._n_clar} clarif | "
                f"{duration}s")
        except Exception as e:
            self.get_logger().error("CSV write error: " + str(e))


def main(args=None):
    rclpy.init(args=args)
    node = SessionLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
