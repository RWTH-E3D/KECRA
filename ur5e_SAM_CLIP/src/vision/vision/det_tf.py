import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformStamped
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Point

class DetTF(Node):
    def __init__(self):
        super().__init__('det_tf')
        self.get_logger().info("DetTF Node Initialized")

        use_sim_time = rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time])

        self.subscription = self.create_subscription(
            Detection2DArray, 
            'obj_detection',
            self.det_callback,
            10)
        self.get_logger().info("Subscribed to 'obj_detection' topic")

        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("TransformBroadcaster initialized")

        # 存储最近一次检测到的目标变换
        self.last_objects = {}
        # 设置定时器，定期重新广播变换
        self.timer = self.create_timer(0.1, self.timer_callback)  # 每 0.1 秒广播一次

    def det_callback(self, msg: Detection2DArray):
        self.get_logger().info(f"Received detection message with {len(msg.detections)} detections")
        objs = {}

        for i, det in enumerate(msg.detections):
            det: Detection2D
            class_name = det.id  
            pos = det.results[0].pose.pose.position  # 获取物体的位置信息

            # 将类名作为键，位置信息作为值存储
            objs[class_name] = pos

            # 日志信息
            # self.get_logger().info(
            #     f"Detected {class_name} at x={pos.x}, y={pos.y}, z={pos.z} (confidence: {det.results[0].hypothesis.score})"
            # )

        # 更新最后一次检测到的目标
        self.last_objects = objs



    def timer_callback(self):
        # 定期广播最后一次检测到的目标变换
        for name, pos in self.last_objects.items():
            pos: Point

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_color_optical_frame'
            t.child_frame_id = name

            t.transform.translation.x = pos.x
            t.transform.translation.y = pos.y
            t.transform.translation.z = pos.z
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0

            # self.get_logger().info(
            #     f"Broadcasting TF for {name}: x={pos.x}, y={pos.y}, z={pos.z}, time={t.header.stamp.sec}.{t.header.stamp.nanosec}"
            # )
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)

    det_tf = DetTF()

    try:
        rclpy.spin(det_tf)
    except KeyboardInterrupt:
        det_tf.get_logger().info("Shutting down DetTF Node")
        pass

    det_tf.destroy_node()
    rclpy.shutdown()
