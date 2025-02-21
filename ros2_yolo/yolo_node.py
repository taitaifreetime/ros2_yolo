from ros2_yolo.yolo import ROS2YOLO
import rclpy

def main(args = None):
    try:
        rclpy.init(args=args)
        node = ROS2YOLO()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()