from vision_msgs.msg import LabelInfo, VisionClass, BoundingBox2D, ObjectHypothesisWithPose
from hri_msgs.msg import Skeleton2D, NormalizedPointOfInterest2D

def classes_to_msg(classes: dict, confidence: float) -> LabelInfo:
    classes_msg = LabelInfo()
    for id in classes.keys():
        class_msg = VisionClass()
        class_msg.class_id = int(id)
        class_msg.class_name = classes[id]
        classes_msg.class_map.append(class_msg)
    classes_msg.threshold = confidence
    return classes_msg

def bbox_to_msg(xyxy):
    bbox_msg = BoundingBox2D()
    x1, y1, x2, y2 = map(float, xyxy)  # Convert to integer for drawing
    bbox_msg.size_x = x2-x1
    bbox_msg.size_y = y2-y1
    bbox_msg.center.position.x = (x2+x1)/2.0
    bbox_msg.center.position.y = (y2+y1)/2.0
    return bbox_msg

def cls_to_msg(box):
    result_msg = ObjectHypothesisWithPose()
    result_msg.hypothesis.class_id = str(box.cls[0].item())
    result_msg.hypothesis.score = float(box.conf[0].item())
    return result_msg

def node_to_msg(node, conf):
    node_msg = NormalizedPointOfInterest2D()
    node_msg.x = node[0]
    node_msg.y = node[1]
    node_msg.c = conf
    return node_msg

def skeleton_to_msg(nodes, confs):
    skeleton_msg = Skeleton2D()
    skeleton_msg.skeleton[Skeleton2D.NOSE] = node_to_msg(nodes[0], confs[0])
    # skeleton_msg.skeleton[Skeleton2D.NECK] = node_to_msg(nodes[Skeleton2D.NECK], confs[Skeleton2D.NECK])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_SHOULDER] = node_to_msg(nodes[6], confs[6])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_ELBOW] = node_to_msg(nodes[8], confs[8])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_WRIST] = node_to_msg(nodes[10], confs[10])
    skeleton_msg.skeleton[Skeleton2D.LEFT_SHOULDER] = node_to_msg(nodes[5], confs[5])
    skeleton_msg.skeleton[Skeleton2D.LEFT_ELBOW] = node_to_msg(nodes[7], confs[7])
    skeleton_msg.skeleton[Skeleton2D.LEFT_WRIST] = node_to_msg(nodes[9], confs[9])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_HIP] = node_to_msg(nodes[12], confs[12])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_KNEE] = node_to_msg(nodes[14], confs[14])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_ANKLE] = node_to_msg(nodes[16], confs[16])
    skeleton_msg.skeleton[Skeleton2D.LEFT_HIP] = node_to_msg(nodes[11], confs[11])
    skeleton_msg.skeleton[Skeleton2D.LEFT_KNEE] = node_to_msg(nodes[13], confs[13])
    skeleton_msg.skeleton[Skeleton2D.LEFT_ANKLE] = node_to_msg(nodes[15], confs[15])
    skeleton_msg.skeleton[Skeleton2D.LEFT_EYE] = node_to_msg(nodes[1], confs[1])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_EYE] = node_to_msg(nodes[2], confs[2])
    skeleton_msg.skeleton[Skeleton2D.LEFT_EAR] = node_to_msg(nodes[3], confs[3])
    skeleton_msg.skeleton[Skeleton2D.RIGHT_EAR] = node_to_msg(nodes[4], confs[4])
    return skeleton_msg