import os
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw
import json
from std_msgs.msg import String


# Import SAM and CLIP
from segment_anything import build_sam, SamAutomaticMaskGenerator
from mobile_sam import sam_model_registry
import clip
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class ObjDetectWithSAMCLIP(Node):
    def __init__(self):
        super().__init__('obj_detect_with_sam_clip')
        self.get_logger().info('ObjDetect node with SAM+CLIP started')
        self.original_saved = False

        # Load SAM model
        # sam_checkpoint = "/home/trb/SAM_Clip/CLIP-SAM/sam_vit_h_4b8939.pth"
        # self.mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=sam_checkpoint).to("cuda"))

        sam_checkpoint = "/home/trb/SAM_Clip/CLIP-SAM/mobile_sam.pt"  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mobile_sam_model = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)  
        self.mobile_sam_model.to(self.device)
        # self.mobile_sam_mask_generator = SamAutomaticMaskGenerator(self.mobile_sam_model)
        self.mobile_sam_mask_generator = SamAutomaticMaskGenerator(
            self.mobile_sam_model,
            pred_iou_thresh=0.9,  
            stability_score_thresh=0.95 
        )

        # Load CLIP model
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # Class names for CLIP
        self.class_names = None
        # self.class_tokens = clip.tokenize(self.class_names).to(self.device)

        self.class_names_sub = self.create_subscription(
            String, 
            'class_names',  
            self.class_names_callback,
            10
        )
        self.get_logger().info("Subscribed to 'class_names' topic.")

        self.bridge = CvBridge()

        # ROS2 Parameters
        self.declare_parameter("image_topic", "/color/image_raw/compressed", ParameterDescriptor(
            name="image_topic", description="Compressed image topic"))
        self.declare_parameter("depth_topic", "/depth_registered/image_rect", ParameterDescriptor(
            name="depth_topic", description="Depth image topic"))
        self.declare_parameter("camera_info_topic", "/color/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="Camera info topic"))
        self.declare_parameter("detection_freq", 0.1, ParameterDescriptor(
            name="detection_freq", description="Detection frequency [Hz]"))
        self.declare_parameter("view_image", True, ParameterDescriptor(
            name="view_image", description="View detection results"))
        self.declare_parameter("publish_result", True, ParameterDescriptor(
            name="publish_result", description="Publish detection results"))

        # Set use_sim_time
        use_sim_time = rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time])

        # ROS2 Subscriptions and Publishers
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.image_sub = self.create_subscription(CompressedImage, self.image_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.detection_pub = self.create_publisher(Detection2DArray, 'obj_detection', 10)
        self.detailed_publisher = self.create_publisher(String, 'detailed_objects', 10)


        # Initialize parameters
        self.image = None
        self.depth_image = None
        self.view_image = self.get_parameter("view_image").value
        self.detection_freq = self.get_parameter("detection_freq").value
        self.timer = self.create_timer(1.0 / self.detection_freq, self.timer_callback)

        # Camera intrinsic matrix
        self.camera_info = None
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.depth_instrinsic = None
        self.depth_instrinsic_inv = None

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.depth_instrinsic = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.depth_instrinsic_inv = np.linalg.inv(self.depth_instrinsic)
        self.get_logger().info("Camera info received.")

    def depth_callback(self, msg: Image):
        self.depth_image = msg

    def image_callback(self, msg: CompressedImage):
        self.image = msg

    def convert_box_xywh_to_xyxy(self, box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]

    def segment_image(self, image, segmentation_mask):
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
        segmented_image = PILImage.fromarray(segmented_image_array)
        black_image = PILImage.new("RGB", image.size, (0, 0, 0))
        transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
        transparency_mask[segmentation_mask] = 255
        transparency_mask_image = PILImage.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image
    
    @torch.no_grad()
    def retriev(self, elements: list[PILImage.Image], search_text: str) -> int:
        preprocessed_images = [self.clip_preprocess(image).to(self.device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(self.device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = self.clip_model.encode_image(stacked_images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100. * image_features @ text_features.T
        return probs[:, 0].softmax(dim=0)
    
    def get_indices_of_values_above_threshold(self, values, threshold):
        return [i for i, v in enumerate(values) if v > threshold]
    
    def class_names_callback(self, msg):
        """Callback to update `class_names` dynamically."""
        self.class_names = msg.data.strip()  # Get the value of class_names and remove extra spaces.
        self.get_logger().info(f"Received new class_names: {self.class_names}")

    def process_image(self):
        if self.class_names is None:
            self.get_logger().warn("Waiting for class_names to be received...")
            return []

        if self.image is None or self.depth_image is None or self.camera_info is None:
            self.get_logger().warn("Missing image, depth, or camera info.")
            return []

        try:
            # Decoding RGB images
            cv_image_rgb = self.bridge.compressed_imgmsg_to_cv2(self.image, desired_encoding='rgb8')
            self.get_logger().info(f"RGB image: type={type(cv_image_rgb)}, shape={cv_image_rgb.shape}, dtype={cv_image_rgb.dtype}")


            # if not self.original_saved:
            #     ori_path = os.path.join(os.getcwd(), "original_input.png")
            #     cv2.imwrite(ori_path, cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR))
            #     self.original_saved = True
            #     self.get_logger().info(f"Saved original image to: {ori_path}")

            ori_path = os.path.join(os.getcwd(), "original_input.png")
            cv2.imwrite(ori_path, cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR))

            # Check image type and format
            if not isinstance(cv_image_rgb, np.ndarray):
                raise ValueError("Input to SAM must be a numpy.ndarray")
            if cv_image_rgb.dtype != np.uint8:
                self.get_logger().warn(f"Converting RGB image dtype from {cv_image_rgb.dtype} to uint8")
                cv_image_rgb = cv_image_rgb.astype(np.uint8)
            if len(cv_image_rgb.shape) != 3 or cv_image_rgb.shape[2] != 3:
                raise ValueError(f"Input to SAM must be a 3-channel RGB image, but got shape {cv_image_rgb.shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to decode RGB image: {str(e)}")
            return []

        try:
            # Decoding depth images
            depth_image = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding="passthrough")

            # Replace invalid depth values
            depth_image = np.nan_to_num(depth_image, nan=0.0)
            self.get_logger().info(f"Depth image: type={type(depth_image)}, shape={depth_image.shape}, dtype={depth_image.dtype}")
            self.get_logger().info(f"Depth image: min={np.nanmin(depth_image)}, max={np.nanmax(depth_image)}")
        except Exception as e:
            self.get_logger().error(f"Failed to decode depth image: {str(e)}")
            return []

        try:
            # Use SAM to generate masks
            # torch.cuda.empty_cache()
            masks = self.mobile_sam_mask_generator.generate(cv_image_rgb)
        except Exception as e:
            self.get_logger().error(f"SAM mask generation failed: {str(e)}")
            return []

        try:
            # Split class_names into a list of categories separated by commas.
            class_names_list = [cls.strip() for cls in self.class_names.split(",")]

            results = []
            cropped_boxes = []

            for mask in masks:
                try:
                    # Extract masks and bounding boxes, crop target images
                    segmentation_mask = mask["segmentation"]
                    bbox = self.convert_box_xywh_to_xyxy(mask["bbox"])
                    cropped_boxes.append(self.segment_image(PILImage.fromarray(cv_image_rgb), segmentation_mask).crop(bbox))
                except Exception as e:
                    self.get_logger().warn(f"Error processing mask cropping: {str(e)}")
                    continue

            # Initialize score storage
            all_scores = []

            try:
                # Use CLIP to retrieve each category
                for class_name in class_names_list:
                    scores = self.retriev(cropped_boxes, class_name)
                    all_scores.append((class_name, scores))

                # Initialize an overlay image for highlighting all categories.
                original_image = PILImage.fromarray(cv_image_rgb)
                overlay_image = PILImage.new('RGBA', original_image.size, (0, 0, 0, 0))  # Fully transparent overlay image
                draw = ImageDraw.Draw(overlay_image)

                # Iterate through the scores and filter the matching targets
                for class_name, scores in all_scores:
                    indices = self.get_indices_of_values_above_threshold(scores, 0.2)

                    draw = ImageDraw.Draw(overlay_image)

                    for idx in indices:
                        try:
                            # Extract matching mask and category information
                            mask = masks[idx]
                            segmentation_mask = mask["segmentation"]
                            bbox = mask["bbox"]
                            confidence = scores[idx].item()

                            # Calculate the center point and 3D coordinates of the target
                            x, y, w, h = bbox
                            center_x = int(x + w / 2)
                            center_y = int(y + h / 2)
                            Z = depth_image[center_y, center_x] * 1e-3  # Depth value (converted to meters)
                            if Z <= 0:
                                self.get_logger().warn(f"Invalid depth value for class '{class_name}'. Skipping detection.")
                                continue

                            uv1 = np.array([center_x, center_y, 1.0])
                            XYZ = np.dot(self.depth_instrinsic_inv, uv1) * Z

                            # Calculate the actual width and actual height
                            real_width = w * Z / self.depth_instrinsic[0, 0]  # Use the camera's fx to calculate the actual width.
                            real_height = h * Z / self.depth_instrinsic[1, 1]  # Use the camera's fy to calculate the true height.

                            # Calculate the actual bbox area
                            real_bbox_area = real_width * real_height

                            # Calculate the actual height of the object
                            # Use the maximum and minimum values of the depth within the bounding box.
                            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                            bbox_depth_values = depth_image[y1:y2, x1:x2].flatten()
                            valid_depths = bbox_depth_values[bbox_depth_values > 0]  # Remove invalid depth values
                            if valid_depths.size > 0:
                                object_height = (valid_depths.max() - valid_depths.min()) * 1e-3  # Convert to meters
                            else:
                                object_height = 0.0

                            # Store test results
                            results.append({
                                "class_name": class_name,
                                "confidence": confidence,
                                "segmentation": segmentation_mask.tolist(),
                                "bbox": bbox,
                                "XYZ": XYZ.tolist(),
                                "real_width": real_width,
                                "real_height": real_height,
                                "real_bbox_area": real_bbox_area, 
                                # "object_height": object_height,
                                # "real_width": 0.05,
                                # "real_height": 0.05,
                                # "real_bbox_area": 0.025, 
                                "object_height": 0.05,
                            })

                            # Highlight overlay on image, using colors corresponding to categories
                            segmentation_mask_image = PILImage.fromarray(segmentation_mask.astype('uint8') * 255)
                            draw.bitmap((0, 0), segmentation_mask_image, fill=(0, 0, 0, 200))

                        except Exception as e:
                            self.get_logger().warn(f"Error processing mask visualization or 3D calculation: {str(e)}")
                            continue

            except Exception as e:
                self.get_logger().error(f"CLIP retrieval failed for class names: {str(e)}")
                return []

            # Compose the superimposed highlighted image onto the original image.
            result_image = PILImage.alpha_composite(original_image.convert('RGBA'), overlay_image)

            # Save the result image and JSON file
            image_save_path = os.path.join(os.getcwd(), "segmentation_result.png")
            result_image.save(image_save_path)
            self.get_logger().info(f"Saved visualized image to: {image_save_path}")

            results_save_path = os.path.join(os.getcwd(), "segmentation_results.json")
            with open(results_save_path, 'w') as f:
                json.dump(results, f, indent=4)
            self.get_logger().info(f"Saved detection results to: {results_save_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to overlay segmentation masks or save results: {str(e)}")

        return results


    def timer_callback(self):
        results = self.process_image()
        if not results:
            return

        # Prepare detection results and detailed objects
        detection_msg = Detection2DArray()
        detailed_objects = []

        # Sorted by actual area
        sorted_results = sorted(results, key=lambda x: x["real_bbox_area"], reverse=True)

        # Number objects with the same class name
        numbered_objects = {}
        for result in sorted_results:
            class_name = result["class_name"]
            if class_name not in numbered_objects:
                numbered_objects[class_name] = 1
            else:
                numbered_objects[class_name] += 1
            numbered_class_name = f"{class_name}{numbered_objects[class_name]}"

            # Update the class name in result
            result["numbered_class_name"] = numbered_class_name

            # Construct detection_msg
            detection = Detection2D()
            detection.id = numbered_class_name

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = numbered_class_name
            hypothesis.hypothesis.score = float(result["confidence"])

            XYZ = result["XYZ"]
            hypothesis.pose.pose.position.x = float(XYZ[0])
            hypothesis.pose.pose.position.y = float(XYZ[1])
            hypothesis.pose.pose.position.z = float(XYZ[2])
            detection.results.append(hypothesis)

            # Convert bbox to ROS2 format
            bbox = result["bbox"]
            detection.bbox.center.position.x = bbox[0] + bbox[2] / 2.0
            detection.bbox.center.position.y = bbox[1] + bbox[3] / 2.0
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])

            detection_msg.detections.append(detection)

            # Save details
            detailed_objects.append({
                "class_name": numbered_class_name,
                "real_width": result["real_width"],
                "real_height": result["real_height"],
                "real_bbox_area": result["real_bbox_area"],
                "XYZ": XYZ,
                "object_height": result["object_height"],
                "confidence": result["confidence"]
            })

        # Publish detailed objects
        detailed_msg = String()
        detailed_msg.data = json.dumps(detailed_objects, indent=4)
        self.detailed_publisher.publish(detailed_msg)

        # publish detection_msg
        if self.get_parameter("publish_result").value:
            detection_msg.header.stamp = self.get_clock().now().to_msg()
            detection_msg.header.frame_id = 'camera_color_optical_frame'
            self.get_logger().info(
                f"Publishing detection message: time={detection_msg.header.stamp.sec}.{detection_msg.header.stamp.nanosec}"
            )
            self.detection_pub.publish(detection_msg)

        # print log
        self.get_logger().info(f"Published detailed objects: {detailed_msg.data}")



def main(args=None):
    rclpy.init(args=args)
    node = ObjDetectWithSAMCLIP()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.class_names is not None:
                node.get_logger().info(f"Starting detection with class_names: {node.class_names}")
                break  # Once class_names is received, the detection logic can be entered.
        rclpy.spin(node)  # operating node
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()

