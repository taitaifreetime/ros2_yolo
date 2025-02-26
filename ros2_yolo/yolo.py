# ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from ros2_yolo.common.ros_utils import get_ros_param
from ros2_yolo.common.ros_yolo_utils import classes_to_msg, bbox_to_msg, cls_to_msg, skeleton_to_msg
from ros2_yolo.common.visualize_utils import get_bbox_color_list, get_skeleton_color_list, draw_bbox, draw_skeleton
from rclpy.qos import qos_profile_sensor_data

# msg 
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import LabelInfo, Detection2DArray, Detection2D
from hri_msgs.msg import Skeleton2DArray

# detection
from ultralytics import YOLO
from ultralytics.engine.results import Results as YoloResults

# general
from cv_bridge import CvBridge
import cv2
import os

class ROS2YOLO(Node):
    def __init__(self):
        super().__init__('yolo')

        self.det_conf_ = get_ros_param(self, 'confidence', float, default_value=0.3)
        self.class_id_ = get_ros_param(self, 'class_id', int, default_value=-1, 
                                       description="-1 means all classes a pre-trained model provides are published", 
                                       additional_constraints=">= -1")
        self.visualize_ = get_ros_param(self, 'visualize', bool, default_value=True)
        yolo_model_path = get_ros_param(self, 'yolo_model_path', str, default_value="")
        # yolo_model = yolo_model_path.split('/')[-1]
        # model_parse = yolo_model.split('.')

        if os.path.exists(yolo_model_path):
            self.model_ = YOLO(yolo_model_path)
            self.get_logger().info("Loaded yolo model {}".format(yolo_model_path))
        else: 
            self.get_logger().warning("{} not exist. Download first.".format(yolo_model_path))
            exit()

        self.img_sub_ = self.create_subscription(Image, "input_image", self.img_callback, qos_profile=qos_profile_sensor_data)
        self.compressd_img_pub_ = self.create_publisher(CompressedImage, 'result/image/compressed', 10)
        self.detection_pub_ = self.create_publisher(Detection2DArray, 'result/detections', 10)
        self.skeletons_pub_ = self.create_publisher(Skeleton2DArray, 'result/skeletons', 10)
        self.labelinfo_pub_ = self.create_publisher(LabelInfo, 'result/labelinfo', qos_profile=QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.bridge_ = CvBridge()
        
        self.names_ = self.model_.names
        self.bbox_colors_ = get_bbox_color_list(len(self.names_.keys()))
        self.skeleton_colors_ = get_skeleton_color_list()
        classes_msg = classes_to_msg(self.names_, self.det_conf_)
        self.labelinfo_pub_.publish(classes_msg)

        self.get_logger().info("det_conf: {}".format(self.det_conf_))
        self.get_logger().info("class_id: {}".format(self.class_id_))
        self.get_logger().info("visualize: {}".format(self.visualize_))
        self.get_logger().info("Published Topic")
        self.get_logger().info("  {}".format(self.detection_pub_.topic_name))
        self.get_logger().info("  {}".format(self.skeletons_pub_.topic_name))
        self.get_logger().info("  {}".format(self.compressd_img_pub_.topic_name))
        self.get_logger().info("  {}".format(self.labelinfo_pub_.topic_name))
        self.get_logger().info("Subscribed Topic")
        self.get_logger().info("  {}".format(self.img_sub_.topic_name))

    def img_callback(self, msg: Image):
        start_time = self.get_clock().now()

        cv_img = self.bridge_.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result_img = cv_img.copy()
        results:list[YoloResults] = self.model_(cv_img, conf=self.det_conf_, verbose=False)
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            detections_msg = Detection2DArray()
            detections_msg.header = msg.header
            skeletons_msg = Skeleton2DArray()
            skeletons_msg.header = msg.header
            for j in range(len(boxes)):
                box = boxes[j]   
                class_id = int(float(box.cls[0].item()))
                if ((self.class_id_ == -1 or class_id == self.class_id_)): 
                    detection_msg = Detection2D()
                    cls_msg = cls_to_msg(box)
                    color = self.bbox_colors_[class_id]
                    bbox_msg = bbox_to_msg(box.xyxy[0])
                    detection_msg.results.append(cls_msg)
                    detection_msg.bbox = bbox_msg
                    detections_msg.detections.append(detection_msg)
                    if self.visualize_: 
                        class_label = self.names_[box.cls[0].item()]
                        score = float(cls_msg.hypothesis.score)
                        draw_bbox(result_img, color, bbox_msg, class_label, score, 
                                    thickness=int(2*(msg.width/1280)), fontscale=int(1.0*(msg.width/1280)))
                
                if keypoints:
                    skeleton = keypoints[j]
                    confs = skeleton.conf[0].tolist()
                    nodes = skeleton.xy[0].tolist()
                    skeleton_msg = skeleton_to_msg(nodes, confs)
                    skeletons_msg.skeletons.append(skeleton_msg)
                    if self.visualize_: 
                        draw_skeleton(result_img, self.skeleton_colors_, skeleton_msg, 
                                    thickness=int(2*(msg.width/1280)), fontscale=int(1.0*(msg.width/1280)))

            self.detection_pub_.publish(detections_msg)
            self.skeletons_pub_.publish(skeletons_msg)
            
            end_time = self.get_clock().now()
            fps = 1.0 * 1e+9/ (end_time - start_time).nanoseconds
            if self.visualize_: 
                cv2.putText(result_img,
                    text="{:.1f} fps".format(fps),
                    org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0*(msg.width/1280), #
                    color=(0, 255, 0),
                    thickness=int(2*(msg.width/1280)), #
                    lineType=cv2.LINE_4)

            compimg_msg = self.bridge_.cv2_to_compressed_imgmsg(result_img)
            compimg_msg.header = msg.header
            self.compressd_img_pub_.publish(compimg_msg)