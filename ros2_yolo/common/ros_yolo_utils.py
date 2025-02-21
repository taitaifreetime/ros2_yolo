from vision_msgs.msg import LabelInfo, VisionClass, BoundingBox2D, ObjectHypothesisWithPose

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
    x1, y1, x2, y2 = map(int, xyxy)  # Convert to integer for drawing
    bbox_msg.size_x = float(x2-x1)
    bbox_msg.size_y = float(y2-y1)
    bbox_msg.center.position.x = float(x2+x1)/2.0
    bbox_msg.center.position.y = float(y2+y1)/2.0
    return bbox_msg

def cls_to_msg(box):
    result_msg = ObjectHypothesisWithPose()
    result_msg.hypothesis.class_id = str(box.cls[0].item())
    result_msg.hypothesis.score = float(box.conf[0].item())
    return result_msg