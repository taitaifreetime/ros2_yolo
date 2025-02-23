# YOLO for ROS 2
This was tested on Jetson Orin Nano Jetpack6.2 with ROS 2 Humble.

## Requirements
- PyTorch 2.5.0
- torchvision 0.20.0
- and something

## Usage 
```
ros2 run ros2_yolo yolo_node --ros-args -p yolo_model_path:=/your_ws/src/ros2_yolo/models/yolo12n.pt -r input_image:=/camera/csi1/image_raw
```

## With Tensorrt
```
cd models
python3 to_trt.py --model_name yolo12n --bach_size 1
ros2 run ros2_yolo yolo_node --ros-args -p yolo_model_path:=/your_ws/src/ros2_yolo/models/yolo12n.engine -r input_image:=/camera/csi1/image_raw
```

## Already Tested
- yolo12
    - n
- yolo11
    - n