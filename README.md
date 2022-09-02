# Isaac ROS DNN Inference

<div align="center"><img alt="Isaac ROS DNN Inference Sample Output (DOPE)" src="https://github.com/NVlabs/Deep_Object_Pose/raw/master/dope_objects.png" width="400px"/><img alt="Isaac ROS DNN Inference Sample Output (PeopleSemSegnet)" src="resources/peoplesemsegnet_rviz2.png" width="400px"/></div>

## Overview
This repository provides two NVIDIA GPU-accelerated ROS2 nodes that perform deep learning inference using custom models. One node uses the TensorRT SDK, while the other uses the Triton SDK. This repository also contains a node to preprocess images, and convert them into tensors for use by TensorRT and Triton.

### TensorRT
TensorRT is a library that enables faster inference on NVIDIA GPUs; it provides an API for the user to load and execute inference with their own models. The TensorRT ROS2 node in this package integrates with the TensorRT API, so the user has no need to make any calls to or directly use TensorRT SDK. Instead, users simply configure the TensorRT node with their own custom models and parameters, and the node will make the necessary TensorRT API calls to load and execute the model. For further documentation on TensorRT, refer to the main page [here](https://developer.nvidia.com/tensorrt).

### Triton
Triton is a framework that brings up a generic inference server that can be configured with a model repository, which is a collection of various types of models (e.g. ONNX Runtime, TensorRT Engine Plan, TensorFlow, PyTorch). A brief tutorial on how to set up a model repository is included below, and further documentation on Triton is also available [here](https://github.com/triton-inference-server/server).

For more details about the setup of TensorRT and Triton, look [here](docs/tensorrt-and-triton-info.md).

### Isaac ROS NITROS Acceleration
This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes. 

## Performance
The following are the benchmark performance results of the prepared pipelines in this package, by supported platform:

| Pipeline               | AGX Orin | AGX Xavier | x86_64 w/ RTX 3060 Ti |
| ---------------------- | -------- | ---------- | --------------------- |
| PeopleSemSegNet (544p) | 325 fps  | 208 fps    | 452 fps               |


## Table of Contents
- [Isaac ROS DNN Inference](#isaac-ros-dnn-inference)
  - [Overview](#overview)
    - [TensorRT](#tensorrt)
    - [Triton](#triton)
    - [Isaac ROS NITROS Acceleration](#isaac-ros-nitros-acceleration)
  - [Performance](#performance)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart with Triton](#quickstart-with-triton)
  - [Quickstart with TensorRT](#quickstart-with-tensorrt)
  - [Next Steps](#next-steps)
    - [Use Different Models](#use-different-models)
    - [Customize your Dev Environment](#customize-your-dev-environment)
  - [Package Reference](#package-reference)
    - [`isaac_ros_dnn_encoders`](#isaac_ros_dnn_encoders)
      - [ROS Parameters](#ros-parameters)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
    - [`isaac_ros_triton`](#isaac_ros_triton)
      - [Usage](#usage)
      - [ROS Parameters](#ros-parameters-1)
      - [ROS Topics Subscribed](#ros-topics-subscribed-1)
      - [ROS Topics Published](#ros-topics-published-1)
    - [`isaac_ros_tensor_rt`](#isaac_ros_tensor_rt)
      - [Usage](#usage-1)
      - [ROS Parameters](#ros-parameters-2)
      - [ROS Topics Subscribed](#ros-topics-subscribed-2)
      - [ROS Topics Published](#ros-topics-published-2)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
    - [Deep Learning Troubleshooting](#deep-learning-troubleshooting)
  - [Updates](#updates)

## Latest Update
Update 2022-08-31: Update to be compatible with JetPack 5.0.2


## Supported Platforms
This package is designed and tested to be compatible with ROS2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

> **Note**: Versions of ROS2 earlier than Humble are **not** supported. This package depends on specific ROS2 implementation features that were only introduced beginning with the Humble release.


| Platform | Hardware                                                                                                                                                                                                | Software                                                                                                             | Notes                                                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)<br/>[Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack)                                                       | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                              | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.6.1+](https://developer.nvidia.com/cuda-downloads) |


### Docker
To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note:** All Isaac ROS Quickstarts, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.


## Quickstart with Triton

> **Note**: The quickstart helps with getting raw inference (tensor) results from the two nodes. To use the packages in a useful context and get meaningful results from the package, please refer [here](#use-different-models).
1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).
2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`.

    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
    ```

3. Launch the Docker container using the `run_dev.sh` script:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
4. For this example, we will use `PeopleSemSegNet ShuffleSeg`. Download the ETLT file and the `int8` inference mode cache file:
    ```bash
    mkdir -p /tmp/models/peoplesemsegnet_shuffleseg/1 && \
      cd /tmp/models/peoplesemsegnet_shuffleseg && \
      wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/files/peoplesemsegnet_shuffleseg_etlt.etlt && \
      wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/files/peoplesemsegnet_shuffleseg_cache.txt
    ```
5. Convert the ETLT file to a TensorRT plan file:
    ```bash
    /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_2:0,1x3x544x960,1x3x544x960,1x3x544x960 -t int8 -c peoplesemsegnet_shuffleseg_cache.txt -e /tmp/models/peoplesemsegnet_shuffleseg/1/model.plan -o argmax_1 peoplesemsegnet_shuffleseg_etlt.etlt
    ```
6. Create a file named `/tmp/models/peoplesemsegnet_shuffleseg/config.pbtxt` by copying the sample Triton config file:
    ```bash
    cp /workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference/resources/peoplesemsegnet_shuffleseg_config.pbtxt /tmp/models/peoplesemsegnet_shuffleseg/config.pbtxt
    ```

7. Inside the container, build and source the workspace:
    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```
8. (Optional) Run tests to verify complete and correct installation:
    ```bash
    colcon test --executor sequential
    ```
9. Run the following launch files to spin up a demo of this package:
    Launch Triton:
    ```bash
    ros2 launch isaac_ros_triton isaac_ros_triton.launch.py model_name:=peoplesemsegnet_shuffleseg model_repository_paths:=['/tmp/models'] input_binding_names:=['input_2:0'] output_binding_names:=['argmax_1']
    ```

    In **another** terminal, enter the Docker container:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

    Then, run a test node that sends tensors to Triton:
    ```bash
    source install/setup.bash && \
    ros2 run isaac_ros_dnn_inference_test run_test_publisher --ros-args -p dimensions:='[1, 3, 544, 960]'
    ```
10. Visualize and validate the output of the package:
    
    In a **third** terminal, enter the Docker container:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

    Then echo the inference result:
    ```bash
    source install/setup.bash && \
    ros2 topic echo /tensor_sub
    ```

    The expected result should look like this:
    ```
    header:
      stamp:
        sec: 0
        nanosec: 0
      frame_id: ''
    tensors:
    - name: output_tensor
      shape:
        rank: 4
        dims:
        - 1
        - 544
        - 960
        - 1
      data_type: 5
      strides:
      - 2088960
      - 3840
      - 4
      - 4
      data:
      [...]
    ```

## Quickstart with TensorRT
> **Note**: The quickstart helps with getting raw inference (tensor) results from the two nodes. To use the packages in a useful context and get meaningful results from the package, please refer  [here](#use-different-models).
1. Follow steps 1-5 of the [Quickstart with Triton](#quickstart-with-triton)
2. Inside the container, build and source the workspace:
    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```
3. (Optional) Run tests to verify complete and correct installation:
    ```bash
    colcon test --executor sequential
    ```
4. Run the following launch files to spin up a demo of this package:
    Launch TensorRT:
    ```bash
    ros2 launch isaac_ros_tensor_rt isaac_ros_tensor_rt.launch.py  engine_file_path:=/tmp/models/peoplesemsegnet_shuffleseg/1/model.plan input_binding_names:=['input_2:0'] output_binding_names:=['argmax_1']
    ```

    In **another** terminal, enter the Docker container:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

    Then, run a test node that sends tensors to TensorRT:
    ```bash
    source install/setup.bash && \
    ros2 run isaac_ros_dnn_inference_test run_test_publisher --ros-args -p dimensions:='[1, 3, 544, 960]'
    ```
5.  Visualize and validate the output of the package:
    In a **third** terminal, enter the Docker container:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

    Then echo the inference result:
    ```bash
    source install/setup.bash && \
    ros2 topic echo /tensor_sub
    ```

    The expected result should look like this:
    ```
    header:
      stamp:
        sec: 0
        nanosec: 0
      frame_id: ''
    tensors:
    - name: output_tensor
      shape:
        rank: 4
        dims:
        - 1
        - 544
        - 960
        - 1
      data_type: 5
      strides:
      - 2088960
      - 3840
      - 4
      - 4
      data:
      [...]
    ```
> **Note**: For both the Triton and the TensorRT Quickstarts, the data contents are omitted. The `data_type` field has a value of `5`, corresponding to the `uint32` data type, but the data array will be output in the terminal as `uint8` bytes. 
> 
> For further details about the `TensorList` message type, see [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg). 
> 
> To view the results as a segmentation mask, please look at [Isaac ROS Image Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation).


## Next Steps
### Use Different Models
Click [here](./docs/model-preparation.md) for more information about how to use NGC models.

We also natively support the following packages to perform DNN inference in a variety of contexts:

| Package Name                                                                                   | Use Case                                                               |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [DNN Stereo Disparity](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_disparity)     | Deep learned stereo disparity estimation                               |
| [Image Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation)         | Hardware-accelerated, deep learned semantic image segmentation         |
| [Object Detection](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection)             | Deep learning model support for object detection including DetectNet   |
| [Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation)               | Deep learned, hardware-accelerated 3D object pose estimation           |
| [Proximity Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation) | DNN-based proximity segmentation and obstacle field ranging using Bi3D |


### Customize your Dev Environment
To customize your development environment, reference [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/modify-dockerfile.md).

## Package Reference
### `isaac_ros_dnn_encoders`

#### ROS Parameters

| ROS Parameter          | Type          | Default           | Description                                                                                     |
| ---------------------- | ------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| `network_image_width`  | `uint16_t`    | `0`               | The image width that the network expects. This will be used to resize the input `image` width   |
| `network_image_height` | `uint16_t`    | `0`               | The image height that the network expects. This will be used to resize the input `image` height |
| `image_mean`           | `double list` | `[0.5, 0.5, 0.5]` | The mean of the images per channel that will be used for normalization                          |
| `image_stddev`         | `double list` | `[0.5, 0.5, 0.5]` | The standard deviation of the images per channel that will be used for normalization            |

> **Note**: the following parameters are no longer supported:
>   - network_image_encoding
>   - maintain_aspect_ratio
>   - center_crop
>   - tensor_name
>   - network_normalization_type

#### ROS Topics Subscribed
| ROS Topic | Interface                                                                                            | Description                                     |
| --------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `image`   | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | The image that should be encoded into a tensor. |

> **Limitation:** All input images are required to have height and width that are both an even number of pixels.

#### ROS Topics Published
| ROS Topic        | Interface                                                                                                                                                         | Description                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `encoded_tensor` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The resultant tensor after converting the `image`. |

### `isaac_ros_triton`
#### Usage
```bash
ros2 launch isaac_ros_triton isaac_ros_triton.launch.py model_name:=<model_name> model_repository_paths:=<model_repository_paths> max_batch_size:=<max_batch_size> input_tensor_names:=<input_tensor_names> input_binding_names:=<input_binding_names> output_tensor_names:=<output_tensor_names> output_binding_names:=<output_binding_names>
```

#### ROS Parameters

| ROS Parameter             | Type          | Default             | Description                                                                                                                                                                                                                         |
| ------------------------- | ------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_repository_paths`  | `string list` | `['']`              | The absolute paths to your model repositories in your local file system (the structure should follow Triton requirements) <br/> E.g. `['/tmp/models']`                                                                              |
| `model_name`              | `string`      | `""`                | The name of your model. Under `model_repository_paths`, there should be a directory with this name, and it should align with the model name in the model configuration under this directory <br/> E.g. `peoplesemsegnet_shuffleseg` |
| `max_batch_size`          | `uint16_t`    | `8`                 | The maximum batch size allowed for the model. It should align with the model configuration                                                                                                                                          |
| `num_concurrent_requests` | `uint16_t`    | `10`                | The number of requests the Triton server can take at a time. This should be set according to the tensor publisher frequency                                                                                                         |
| `input_tensor_names`      | `string list` | `['input_tensor']`  | A list of tensor names to be bound to specified input bindings names. Bindings occur in sequential order, so the first name here will be mapped to the first name in input_binding_names                                            |
| `input_binding_names`     | `string list` | `['']`              | A list of input tensor binding names specified by model <br/> E.g. `['input_2:0']`                                                                                                                                                  |
| `input_tensor_formats`    | `string list` | `['']`              | A list of input tensor nitros formats. This should be given in sequential order <br/> E.g. `['nitros_tensor_list_nchw_rgb_f32']`                                                                                                    |
| `output_tensor_names`     | `string list` | `['output_tensor']` | A list of tensor names to be bound to specified output binding names                                                                                                                                                                |
| `output_binding_names`    | `string list` | `['']`              | A list of tensor names to be bound to specified output binding names <br/> E.g. `['argmax_1']`                                                                                                                                      |
| `output_tensor_formats`   | `string list` | `['']`              | A list of input tensor nitros formats. This should be given in sequential order <br/> E.g. `[nitros_tensor_list_nchw_rgb_f32]`                                                                                                      |


#### ROS Topics Subscribed
| ROS Topic    | Interface                                                                                                                                                         | Description             |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `tensor_pub` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The input tensor stream |

#### ROS Topics Published
| ROS Topic    | Interface                                                                                                                                                         | Description                                                |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `tensor_sub` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The tensor list of output tensors from the model inference |

### `isaac_ros_tensor_rt`
#### Usage
```bash
ros2 launch isaac_ros_tensor_rt isaac_ros_tensor_rt.launch.py model_file_path:=<model_file_path> engine_file_path:=<engine_file_path> input_tensor_names:=<input_tensor_names> input_binding_names:=<input_binding_names> output_tensor_names:=<output_tensor_names> output_binding_names:=<output_binding_names> verbose:=<verbose> force_engine_update:=<force_engine_update>
```

#### ROS Parameters

| ROS Parameter             | Type          | Default                | Description                                                                                                                                                                                 |
| ------------------------- | ------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_file_path`         | `string`      | `model.onnx`           | The absolute path to your model file in the local file system (the model file must be .onnx) <br/> E.g. `model.onnx`                                                                        |
| `engine_file_path`        | `string`      | `/tmp/trt_engine.plan` | The absolute path to either where you want your TensorRT engine plan to be generated (from your model file) or where your pre-generated engine plan file is located <br/> E.g. `model.plan` |
| `force_engine_update`     | `bool`        | `true`                 | If set to true, the node will always try to generate a TensorRT engine plan from your model file and needs to be set to false to use the pre-generated TensorRT engine plan                 |
| `input_tensor_names`      | `string list` | `['input_tensor']`     | A list of tensor names to be bound to specified input bindings names. Bindings occur in sequential order, so the first name here will be mapped to the first name in input_binding_names    |
| `input_binding_names`     | `string list` | `['']`                 | A list of input tensor binding names specified by model <br/> E.g. `['input_2:0']`                                                                                                          |
| `input_tensor_formats`    | `string list` | `['']`                 | A list of input tensor nitros formats. This should be given in sequential order <br/> E.g. `['nitros_tensor_list_nchw_rgb_f32']`                                                            |
| `output_tensor_names`     | `string list` | `['output_tensor']`    | A list of tensor names to be bound to specified output binding names                                                                                                                        |
| `output_binding_names`    | `string list` | `['']`                 | A list of tensor names to be bound to specified output binding names <br/> E.g. `['argmax_1']`                                                                                              |
| `output_tensor_formats`   | `string list` | `['']`                 | A list of input tensor nitros formats. This should be given in sequential order <br/> E.g. `[nitros_tensor_list_nchw_rgb_f32]`                                                              |
| `verbose`                 | `bool`        | `true`                 | If set to true, the node will enable verbose logging to console from the internal TensorRT execution                                                                                        |
| `max_workspace_size`      | `int64_t`     | `67108864l`            | The size of the working space in bytes                                                                                                                                                      |
| `max_batch_size`          | `int32_t`     | `1`                    | The maximum possible batch size in case the first dimension is dynamic and used as the batch size                                                                                           |
| `dla_core`                | `int64_t`     | `-1`                   | The DLA Core to use. Fallback to GPU is always enabled. The default setting is GPU only                                                                                                     |
| `enable_fp16`             | `bool`        | `true`                 | Enables building a TensorRT engine plan file which uses FP16 precision for inference. If this setting is false, the plan file will use FP32 precision                                       |
| `relaxed_dimension_check` | `bool`        | `true`                 | Ignores dimensions of 1 for the input-tensor dimension check                                                                                                                                |
#### ROS Topics Subscribed
| ROS Topic    | Type                                                                                                                                                              | Description             |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `tensor_pub` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The input tensor stream |

#### ROS Topics Published
| ROS Topic    | Type                                                                                                                                                              | Description                                                |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `tensor_sub` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The tensor list of output tensors from the model inference |

## Troubleshooting
### Isaac ROS Troubleshooting
For solutions to problems with Isaac ROS, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

### Deep Learning Troubleshooting
For solutions to problems with using DNN models, please check [here](docs/troubleshooting.md).

## Updates
| Date       | Changes                                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 2022-08-31 | Update to be compatible with JetPack 5.0.2                                                                                   |
| 2022-06-30 | Added format string parameter in Triton/TensorRT, switched to NITROS implementation, removed parameters in DNN Image Encoder |
| 2021-11-03 | Split DOPE and U-Net into separate repositories                                                                              |
| 2021-10-20 | Initial release                                                                                                              |
