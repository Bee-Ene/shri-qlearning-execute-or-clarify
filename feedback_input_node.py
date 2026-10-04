"""
Feedback Input Node

Lets you type human feedback in the terminal.
Publishes to /human_feedback for the SHRI decision node.

For execution feedback: type 'yes' (correct) or 'no' (wrong).
For clarification feedback: type the answer (e.g., 'forward', 'the chair').

In the real system, this is replaced by a speech recognition node
or a physical button interface. The decision node receives the same
message format either way.

Usage: ros2 run shri_decision feedback_input_node
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading


class FeedbackInputNode(Node):
    def __init__(self):
        super().__init__("feedback_input_node")
        self.pub = self.create_publisher(String, "/human_feedback", 10)

        # Also subscribe to clarification queries to show them here
        self.sub_clar = self.create_subscription(
            String, "/clarification_query", self.clar_callback, 10)
        self.sub_status = self.create_subscription(
            String, "/shri_status", self.status_callback, 10)

        self.get_logger().info("Feedback Input Node ready.")
        self.get_logger().info("=" * 55)
        self.get_logger().info("  Type feedback here when the robot asks.")
        self.get_logger().info("  After execution: 'yes' (correct) or 'no' (wrong)")
        self.get_logger().info("  After clarification question: type your answer")
        self.get_logger().info("    e.g., 'forward', 'the chair', 'yes', 'cancel'")
        self.get_logger().info("=" * 55)

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

    def clar_callback(self, msg: String):
        print(f"\n  [ROBOT ASKS]: {msg.data}")
        print("  Feedback > ", end="", flush=True)

    def status_callback(self, msg: String):
        print(f"\n  [STATUS]: {msg.data}")
        if "READY" in msg.data:
            print("  (Waiting for next command in command terminal)")

    def _input_loop(self):
        while rclpy.ok():
            try:
                text = input("\nFeedback > ").strip()
            except EOFError:
                break
            if not text:
                continue
            msg = String()
            msg.data = text
            self.pub.publish(msg)
            self.get_logger().info(f"Published feedback: '{text}'")


def main(args=None):
    rclpy.init(args=args)
    node = FeedbackInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
