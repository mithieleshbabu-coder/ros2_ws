import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Counter(Node):
    def __init__(self):
        super().__init__('counter')
        self.pub = self.create_publisher(String, 'count', 10)
        self.timer = self.create_timer(0.5, self.tick)
        self.n = 0

    def tick(self):
        msg = String()
        msg.data = f'count {self.n}'
        self.pub.publish(msg)
        self.get_logger().info(f'publishing: {msg.data}')
        self.n += 1


def main():
    rclpy.init()
    rclpy.spin(Counter())


if __name__ == '__main__':
    main()
