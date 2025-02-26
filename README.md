# YOLO for ROS 2
This was tested on Jetson Orin Nano Jetpack6.2 with ROS 2 Humble.

## Requirements
- PyTorch 2.5.0
- torchvision 0.20.0
- hri_msgs
    ```
    git clone https://github.com/taitaifreetime/hri_msgs.git
    ```
- and something

## Usage 
```
ros2 run ros2_yolo yolo_node --ros-args -p yolo_model_path:=/your_ws/src/ros2_yolo/models/yolo12n.pt -r input_image:=/camera/csi1/image_raw
# ros2 run ros2_yolo yolo_node --ros-args -p yolo_model_path:=/your_ws/src/ros2_yolo/models/yolo12n.engine -r input_image:=/camera/csi1/image_raw # using tensorrt
```

## Convert to Tensorrt
### For provided pre-trained model (see [Supported Tasks and Modes](https://docs.ultralytics.com/models/yolo12/#:~:text=to%20cloud%20infrastructure.-,Supported%20Tasks%20and%20Modes,-YOLO12%20supports%20a))
```
cd models
python3 to_trt.py --model_name yolo12n --batch_size 1
```

### For trained model using custom dataset
```
mv trained_model_using_custom_dataset.pt /your_ws/src/ros2_yolo/models/
cd /your_ws/src/ros2_yolo/models
python3 to_trt.py --model_name trained_model_using_custom_dataset --batch_size 1
```


## Already Tested
- yolo12
    - n
    - n-pose
- yolo11
    - n
    - n-pose