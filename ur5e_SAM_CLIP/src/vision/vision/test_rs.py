import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt

class TestRs(Node):
    def __init__(self):
        super().__init__('test_rs')

        # set parameter use_sim_time to true
        use_sim_time = rclpy.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )
        self.set_parameters([use_sim_time])
        
        self.subscription = self.create_subscription(
            Image,
            '/depth_registered/image_rect',
            self.depth_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info("TestRs node initialized and subscribed to /depth_registered/image_rect")

    def depth_callback(self, msg):
        self.get_logger().info("Received depth image message")
        
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().info(f"Depth image shape: {depth.shape}")
            
            # Visualize the depth image using matplotlib
            plt.imshow(depth, cmap='jet')
            plt.colorbar()
            plt.pause(0.01)  # Non-blocking mode for updating plots

        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    test_rs = TestRs()

    try:
        rclpy.spin(test_rs)
    except KeyboardInterrupt:
        test_rs.get_logger().info("Shutting down TestRs node")
    finally:
        test_rs.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
