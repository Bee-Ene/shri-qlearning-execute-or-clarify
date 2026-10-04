"""
Combined Human Input Node, for User Study
Replaces whisper_stt_node + command_input_node + feedback_input_node.
ONE terminal for the human operator. Robot speaks via TTS.

Flow:
  - Robot speaks status and questions aloud (macOS 'say' command)
  - Human TYPES commands and answers at the prompt
  - Everything is published to the correct ROS2 topics
  - Clean display shows exactly what is happening

For user study: the participant sits at this terminal.
They type commands and responses. The robot speaks back.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import threading
import subprocess
import time
import sys


class CombinedInputNode(Node):

    def __init__(self):
        super().__init__("combined_input_node")

        # Publishers
        self.pub_cmd = self.create_publisher(String, "/command_text",   10)
        self.pub_fb  = self.create_publisher(String, "/human_feedback", 10)

        # Subscriptions : show robot messages to human
        self.create_subscription(
            String, "/shri_status",
            self._status_cb, 10)
        self.create_subscription(
            String, "/clarification_query",
            self._clar_cb, 10)

        # State
        self._phase         = "ready"
        self._last_status   = ""
        self._waiting_for   = "command"   # "command" or "feedback" or "answer"
        self._lock          = threading.Lock()

        # Speech queue
        self._speech_queue  = []
        self._speech_lock   = threading.Lock()
        self._speech_thread = threading.Thread(
            target=self._speech_loop, daemon=True)
        self._speech_thread.start()

        # Input thread
        self._input_thread = threading.Thread(
            target=self._input_loop, daemon=True)
        self._input_thread.start()

        self._print_header()
        self._speak("SHRI system ready. Please type your first command.")

    # ── Display and speech ─────────────────────────────────────────────

    def _print_header(self):
        print("\n" + "=" * 60)
        print("  SHRI SYSTEM — Human Interaction Terminal")
        print("=" * 60)
        print("  Type locomotion commands and press Enter.")
        print("  The robot will speak its questions aloud.")
        print("  Type YES or NO to confirm/reject execution.")
        print("  Type CANCEL to cancel any episode.")
        print("=" * 60)
        print()

    def _print_status(self, text: str):
        """Print status in a clear, prominent way."""
        print("\n  [ROBOT STATUS] " + text)

    def _print_question(self, text: str):
        """Print clarification question prominently."""
        print("\n" + "─" * 60)
        print("  ROBOT ASKS: " + text)
        print("─" * 60)

    def _speak(self, text: str):
        """Queue text for TTS via macOS 'say' command."""
        with self._speech_lock:
            self._speech_queue.append(text)

    def _speech_loop(self):
        """Background thread for TTS."""
        while rclpy.ok():
            text = None
            with self._speech_lock:
                if self._speech_queue:
                    text = self._speech_queue.pop(0)
            if text:
                try:
                    subprocess.run(
                        ["say", "-r", "170", "-v", "Samantha", text],
                        timeout=20, check=False)
                    time.sleep(0.5)
                except Exception:
                    pass
            else:
                time.sleep(0.1)

    # ── Status and clarification callbacks ─────────────────────────────

    def _status_cb(self, msg: String):
        status = msg.data.strip()
        if status == self._last_status:
            return
        self._last_status = status
        upper = status.upper()

        with self._lock:
            if "READY" in upper:
                self._phase       = "ready"
                self._waiting_for = "command"
                self._print_status("Ready for your command.")
                self._print_prompt()
            elif "EXECUTING" in upper:
                self._phase       = "executing"
                self._waiting_for = "feedback"
                action = status.replace("EXECUTING:", "").replace(
                    "| Say YES or NO if correct", "").strip()
                self._print_status("Executing: " + action)
                self._speak("Executing now.")
            elif "FEEDBACK" in upper:
                self._phase       = "feedback"
                self._waiting_for = "feedback"
                self._print_status("Movement complete.")
                print("  → Type YES if correct, NO if wrong:")
            elif "SUCCESS" in upper:
                self._phase       = "ready"
                self._waiting_for = "command"
                self._print_status("✓ Success! Episode complete.")
                self._speak("Task complete.")
                self._print_prompt()
            elif "FAILED" in upper or "CANCELLED" in upper:
                self._phase       = "ready"
                self._waiting_for = "command"
                self._print_status("Episode ended. Ready for next command.")
                self._speak("Ready for next command.")
                self._print_prompt()
            elif "RECOVERING" in upper:
                self._print_status("Recovering — moving to clear obstacle...")
                self._speak("Moving to clear the obstacle.")
            elif "CLARIFYING" in upper:
                self._phase       = "clarifying"
                self._waiting_for = "answer"

    def _clar_cb(self, msg: String):
        question = msg.data.strip()
        if not question:
            return
        self._print_question(question)
        self._speak(question)
        print("  → Type your answer:")

    def _print_prompt(self):
        print("\n  Command > ", end="", flush=True)

    # ── Input loop ─────────────────────────────────────────────────────

    def _input_loop(self):
        """Main input loop runs in background thread."""
        # Wait for system to start
        time.sleep(2.0)
        print("\n  Command > ", end="", flush=True)

        while rclpy.ok():
            try:
                text = input("").strip()
            except EOFError:
                break

            if not text:
                print("  Command > ", end="", flush=True)
                continue

            with self._lock:
                waiting = self._waiting_for

            if waiting == "command":
                # New locomotion command
                msg = String()
                msg.data = text
                self.pub_cmd.publish(msg)
                self.get_logger().info("Command sent: '" + text + "'")

            elif waiting in ("feedback", "answer"):
                # Response to execution or clarification
                msg = String()
                msg.data = text
                self.pub_fb.publish(msg)
                self.get_logger().info("Feedback sent: '" + text + "'")

            else:
                # System busy — queue as feedback anyway
                msg = String()
                msg.data = text
                self.pub_fb.publish(msg)

            time.sleep(0.1)
            print("  > ", end="", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = CombinedInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
