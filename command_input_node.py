"""
Command Input Node

Lets you type locomotive commands in the terminal.
Publishes them to /command_text for the SHRI decision node.

In the real system, this is replaced by a speech recognition node
(e.g., using Whisper) that publishes transcribed speech to /command_text.
The decision node receives the same message format either way.

Usage: ros2 run shri_decision command_input_node
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import threading


class CommandInputNode(Node):
    def __init__(self):
        super().__init__("command_input_node")
        self.pub = self.create_publisher(String, "/command_text", 10)
        self.get_logger().info("Command Input Node ready.")
        self.get_logger().info("=" * 55)
        self.get_logger().info("  Type locomotive commands and press Enter.")
        self.get_logger().info("  Examples:")
        self.get_logger().info("    move forward")
        self.get_logger().info("    move forward 2 meters")
        self.get_logger().info("    turn left")
        self.get_logger().info("    go to the chair")
        self.get_logger().info("    move            (incomplete — will ask direction)")
        self.get_logger().info("    stop")
        self.get_logger().info("    sit down")
        self.get_logger().info("=" * 55)

        # Run input in a separate thread so ROS2 spin is not blocked
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

    def _input_loop(self):
        while rclpy.ok():
            try:
                text = input("\nCommand > ").strip()
            except EOFError:
                break
            if not text:
                continue
            msg = String()
            msg.data = text
            self.pub.publish(msg)
            self.get_logger().info(f"Published command: '{text}'")


def main(args=None):
    rclpy.init(args=args)
    node = CommandInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
