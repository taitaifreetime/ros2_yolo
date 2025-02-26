import random
import cv2
import numpy as np
from vision_msgs.msg import BoundingBox2D, ObjectHypothesisWithPose

def get_bbox_color_list(num_class: int) -> list:
    """generate colors in advance

    Args:
        num_class (int): the num of classes

    Returns:
        list: rgb color list
    """
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), 
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (153, 50, 204), 
        (255, 69, 0), (218, 165, 32), (75, 0, 130), (123, 104, 238), (47, 79, 79), (199, 21, 133), (0, 206, 209), 
        (144, 238, 144), (255, 20, 147), (186, 85, 211), (34, 139, 34), (255, 160, 122), (255, 99, 71), (238, 130, 238), 
        (0, 0, 128), (139, 69, 19), (250, 128, 114), (135, 206, 250), (72, 61, 139), (230, 230, 250), (50, 205, 50), 
        (107, 142, 35), (60, 179, 113), (46, 139, 87), (255, 215, 0), (240, 128, 128), (30, 144, 255), (100, 149, 237), 
        (186, 85, 211), (147, 112, 219), (176, 196, 222), (238, 130, 238), (255, 182, 193), (221, 160, 221), (220, 20, 60), 
        (255, 105, 180), (205, 92, 92), (119, 136, 153), (112, 128, 144), (144, 238, 144), (255, 222, 173), (127, 255, 0), 
        (255, 165, 0), (255, 140, 0), (250, 128, 114), (233, 150, 122), (255, 248, 220), (238, 232, 170), (152, 251, 152), 
        (175, 238, 238), (0, 139, 139), (64, 224, 208), (72, 209, 204), (143, 188, 143), (244, 164, 96), (210, 180, 140), 
        (0, 191, 255), (105, 105, 105), (190, 190, 190), (169, 169, 169), (245, 245, 245), (255, 250, 240), (250, 240, 230)
    ]
    if num_class > len(colors):
        for class_id in range(num_class-len(colors)):
            colors.append((random.randrange(256),random.randrange(256),random.randrange(256)))
    return colors

def draw_bbox(result_img, color, bbox_msg: BoundingBox2D, class_label: str, score: float, thickness = 2, fontscale = 1.0):
    half_width = bbox_msg.size_x/2.0
    half_height = bbox_msg.size_y/2.0
    cv2.rectangle(result_img, 
                  (int(bbox_msg.center.position.x-half_width), int(bbox_msg.center.position.y-half_height)), 
                  (int(bbox_msg.center.position.x+half_width), int(bbox_msg.center.position.y+half_height)), 
                  color, thickness)
    label = f'{class_label} {score:.2f}'
    cv2.putText(result_img, 
            text=label, 
            org=(int(bbox_msg.center.position.x-half_width), int(bbox_msg.center.position.y-half_height) - 10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=fontscale, 
            color=color, 
            thickness=thickness, 
            lineType=cv2.LINE_AA)
    
def get_skeleton_color_list() -> list:
    colors = [
        (0, 0, 255),   # NOSE
        (0, 32, 255),  # NECK
        (0, 85, 255),  # RIGHT_SHOULDER
        (0, 170, 255), # RIGHT_ELBOW
        (0, 255, 255), # RIGHT_WRIST
        (0, 255, 170), # LEFT_SHOULDER
        (0, 255, 85),  # LEFT_ELBOW
        (0, 255, 0),   # LEFT_WRIST
        (85, 255, 0),  # RIGHT_HIP
        (170, 255, 0), # RIGHT_KNEE
        (255, 255, 0), # RIGHT_ANKLE 
        (255, 170, 0), # LEFT_HIP 
        (255, 85, 0),  # LEFT_KNEE 
        (255, 0, 0),   # LEFT_ANKLE 
        (255, 0, 192), # LEFT_EYE 
        (192, 0, 255), # RIGHT_EYE 
        (255, 0, 128), # LEFT_EAR 
        (128, 0, 255)  # RIGHT_EAR 
    ]
    return colors 

def draw_skeleton(result_img, colors, skeleton_msg, thickness = 2, fontscale = 1):
    for node, color in zip(skeleton_msg.skeleton, colors):
        cv2.circle(result_img, (int(node.x), int(node.y)), 5*fontscale, color, thickness=5*thickness, lineType=cv2.LINE_8, shift=0)