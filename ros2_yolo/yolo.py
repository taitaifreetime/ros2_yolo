# ros
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import LabelInfo, Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
# from ros2_trt_yolo.utils import (get_vibrant_grad_color_list, get_color_list)
# from ros2_trt_yolo.visualize import (draw, lap_layer, draw_fps)
from ros2_yolo.common.ros_utils import get_ros_param
from ros2_yolo.common.ros_yolo_utils import classes_to_msg, bbox_to_msg, cls_to_msg
from ros2_yolo.common.visualize_utils import get_color_list, draw_bbox
from rclpy.qos import qos_profile_sensor_data

# detection
from ultralytics import YOLO

# general
from cv_bridge import CvBridge
import cv2
import os

class ROS2YOLO(Node):
    def __init__(self):
        super().__init__('yolo')

        self.det_conf_ = get_ros_param(self, 'confidence', float, default_value=0.3)
        self.class_id_ = get_ros_param(self, 'class_id', int, default_value=-1)
        self.visualize_ = get_ros_param(self, 'visualize', bool, default_value=True)
        yolo_model_path = get_ros_param(self, 'yolo_model_path', str, default_value="")
        yolo_model = yolo_model_path.split('/')[-1]
        model_parse = yolo_model.split('.')
        try:    
            self.trt_enabled_ = model_parse[1] == 'engine'
        except IndexError as e: self.get_logger().error(e); exit()

        if os.path.exists(yolo_model_path):
            if self.trt_enabled_:
                self.model_ = YOLO(yolo_model_path)
                self.get_logger().info("Loaded trt yolo model {}".format(yolo_model))
        else: 
            self.get_logger().warning("{} not exist. Export first.".format(yolo_model))
            exit()

        self.img_sub_ = self.create_subscription(Image, "input_image", self.img_callback, qos_profile=qos_profile_sensor_data)
        self.compressd_img_pub_ = self.create_publisher(CompressedImage, 'result/image/compressed', 10)
        self.detection_pub_ = self.create_publisher(Detection2DArray, 'result/detections', 10)
        self.labelinfo_pub_ = self.create_publisher(LabelInfo, 'result/labelinfo', qos_profile=QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.bridge_ = CvBridge()
        
        self.names_ = self.model_.names
        self.colors_ = get_color_list(len(self.names_.keys()))
        classes_msg = classes_to_msg(self.names_, self.det_conf_)
        self.labelinfo_pub_.publish(classes_msg)

    def img_callback(self, msg: Image):
        start_time = self.get_clock().now()

        cv_img = self.bridge_.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result_img = cv_img.copy()
        if self.trt_enabled_: results = self.model_(cv_img, conf=self.det_conf_, verbose=False)
        for i in range(len(results)):
            result = results[i]
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            detections_msg = Detection2DArray()
            detections_msg.header = msg.header
            for j in range(len(boxes)):
                box = boxes[j]   
                class_id = int(float(box.cls[0].item()))
                if ((self.class_id_ == -1 or class_id == self.class_id_)): 
                    detection_msg = Detection2D()
                    cls_msg = cls_to_msg(box)
                    color = self.colors_[class_id]
                    bbox_msg = bbox_to_msg(box.xyxy[0])
                    detection_msg.results.append(cls_msg)
                    detection_msg.bbox = bbox_msg
                    detections_msg.detections.append(detection_msg)

                    if self.visualize_: 
                        class_label = self.names_[box.cls[0].item()]
                        score = float(cls_msg.hypothesis.score)
                        draw_bbox(result_img, color, bbox_msg, class_label, score, 
                                    thickness=int(2*(msg.width/1280)), fontscale=1.0*(msg.width/1280))

            self.detection_pub_.publish(detections_msg)
            
            end_time = self.get_clock().now()
            try:
                fps = 1.0 / ((end_time - start_time).nanoseconds / 1e+9)
                if self.visualize_: 
                    cv2.putText(result_img,
                        text="{:.1f} fps".format(fps),
                        org=(10, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0*(msg.width/1280), #
                        color=(0, 255, 0),
                        thickness=int(2*(msg.width/1280)), #
                        lineType=cv2.LINE_4)
            
            except ZeroDivisionError: pass

            compimg_msg = self.bridge_.cv2_to_compressed_imgmsg(result_img)
            compimg_msg.header = msg.header
            self.compressd_img_pub_.publish(compimg_msg)