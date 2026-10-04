"""
TTS Node — macOS 'say' command (zero dependencies)
Uses subprocess to call the built-in macOS speech engine directly.
No pip install needed. Works offline always.

Speaks:
  - ALL status events (ready, executing, success, failed)
  - ALL clarification questions
  - Recovery announcements

Publishes /tts_speaking so Whisper pauses during speech.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import subprocess
import threading
import time


# Status messages to speak (maps status keyword → spoken phrase)
STATUS_SPEECH = {
    "READY":      "Ready. Please give your command.",
    "EXECUTING":  "Understood. Executing now.",
    "SUCCESS":    "Task complete. Ready for next command.",
    "FAILED":     "Episode failed. Ready for next command.",
    "CANCELLED":  "Command cancelled. Ready for next command.",
    "RECOVERING": "Moving to clear the obstacle.",
    "CLARIFYING": None,   # handled separately via /clarification_query
}


class TTSNode(Node):

    def __init__(self):
        super().__init__("gtts_tts_node")

        # Subscriptions
        self.create_subscription(
            String, "/clarification_query",
            self._clar_cb, 10)
        self.create_subscription(
            String, "/shri_status",
            self._status_cb, 10)

        # Publisher — signals Whisper to pause while speaking
        self.pub_busy = self.create_publisher(
            String, "/tts_speaking", 10)

        # Speech queue and lock
        self._queue = []
        self._lock  = threading.Lock()
        self._last_status = ""

        # Background speech thread
        self._thread = threading.Thread(
            target=self._loop, daemon=True)
        self._thread.start()

        self.get_logger().info("=" * 55)
        self.get_logger().info("  TTS Node ready (macOS built-in speech).")
        self.get_logger().info("  Robot will speak all events aloud.")
        self.get_logger().info("=" * 55)

        # Announce startup
        self._enqueue("SHRI system ready. Please give your command.")

    # ── Subscriptions ──────────────────────────────────────────────────

    def _clar_cb(self, msg: String):
        """Speak clarification question immediately."""
        text = msg.data.strip()
        if text:
            self.get_logger().info("Speaking question: " + text)
            self._enqueue(text, priority=True)

    def _status_cb(self, msg: String):
        """Speak status announcements on transitions only."""
        status = msg.data.strip()
        if status == self._last_status:
            return
        self._last_status = status
        status_upper = status.upper()

        phrase = None
        for key, speech in STATUS_SPEECH.items():
            if key in status_upper and speech is not None:
                phrase = speech
                break

        if phrase:
            self.get_logger().info("Speaking status: " + phrase)
            self._enqueue(phrase)

    # ── Speech queue ───────────────────────────────────────────────────

    def _enqueue(self, text: str, priority: bool = False):
        with self._lock:
            if priority:
                self._queue.insert(0, text)
            else:
                self._queue.append(text)

    def _loop(self):
        """Background thread processes speech queue."""
        while rclpy.ok():
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.pop(0)

            if text:
                self._speak(text)
            else:
                time.sleep(0.1)

    def _speak(self, text: str):
        """
        Uses macOS built-in 'say' command.
        Zero dependencies — works on every Mac with no pip install.
        Same voice engine as Siri and the Terminal 'say' command.
        """
        try:
            # Signal Whisper to pause
            busy = String()
            busy.data = "speaking"
            self.pub_busy.publish(busy)

            # Speak using macOS built-in engine
            # -r 175 = rate (words per minute), natural conversational speed
            # -v Samantha = clear US English voice (best for demos)
            subprocess.run(
                ["say", "-r", "175", "-v", "Samantha", text],
                timeout=30,
                check=False
            )

            # Buffer: wait 1.5s after speech before Whisper listens again
            time.sleep(1.5)

        except FileNotFoundError:
            # 'say' command not found (non-Mac system)
            self.get_logger().error(
                "macOS 'say' command not found. "
                "This node requires macOS.")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("TTS timeout for: " + text)
        except Exception as e:
            self.get_logger().error("TTS error: " + str(e))
        finally:
            # Signal Whisper it can listen again
            done = String()
            done.data = "done"
            self.pub_busy.publish(done)


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
