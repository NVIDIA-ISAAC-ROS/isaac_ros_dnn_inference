# Isaac ROS DNN Inference

<div align="center"><img src="https://github.com/NVlabs/Deep_Object_Pose/raw/master/dope_objects.png" width="300px"/><img src="resources/peoplesemsegnet_rviz2.png" width="300px"/></div>

## Overview
This repository provides two NVIDIA GPU-accelerated ROS2 nodes that perform deep learning inference using custom models. One node uses the TensorRT SDK, while the other uses the Triton SDK.

TensorRT is a library that enables faster inference on NVIDIA GPUs; it provides an API for the user to load and execute inference with their own models. The TensorRT ROS2 node in this package integrates this TensorRT API directly, so there is no need to make any calls to or directly use TensorRT SDK. Instead, users simply configure the TensorRT node with their own custom models and parameters, and the node will make the necessary TensorRT API calls to load and execute the model. For further documentation on TensorRT, refer to their main page [here](https://developer.nvidia.com/tensorrt).

Triton is a framework that brings up a generic inference server that a user can configure with a model repository, which is a collection of various types of models (e.g.) ONNX Runtime, TensorRT Engine Plan, TensorFlow, PyTorch). A brief tutorial on how to set up a model repository is included below, and further documentation on Triton is also available at the [Triton GitHub](https://github.com/triton-inference-server/server).

The decision between TensorRT and Triton is ultimately up to user preference. Since TensorRT has fewer configuration steps (i.e. it does not require a model repository), generally you can get started faster with TensorRT. However, the TensorRT node only supports ONNX and TensorRT Engine Plans, while the Triton node supports a wider variety of model types. In terms of performance and inference speed, they are both comparable in our benchmarks.

The user configures either node to load a specified model or (in the case of the Triton SDK) model repository. The nodes expect as input a ROS2 TensorList message and publish the inference result as a ROS2 TensorList message. The definiton of the TensorList message (and the Tensor message contained within it) is specified under `isaac_ros_common/isaac_ros_nvengine_interfaces/msg`. Users are expected to run their own models, which they have trained (and converted to a compatible model format such as ONNX), or downloaded from NGC (in ETLT format), and converted to a TensorRT Engine File using the TAO converter tool. When running the TensorRT node, it is generally a better practice to first convert your custom model into a TensorRT Engine Plan file using the TAO converter before running inference. If an ONNX model is directly provided, the TensorRT node will convert it to a TensorRT Engine Plan file first before running inference, which will extend the initial setup time of the node.

Native model support is provided as a separate ROS packages with nodes that can accept an input image message and output a tensor message result. The following packages provide additional native model support with useful walkthroughs and documentation on how to use them.

- [`isaac_ros_dope`](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_dope): [Deep Object Pose Estimation (DOPE)](https://github.com/NVlabs/Deep_Object_Pose) for 3D object pose estimation
- [`isaac_ros_unet`](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation/tree/main/isaac_ros_image_segmentation): [U-Net](https://github.com/NVlabs/Deep_Object_Pose) for semantic image segmentation


Both nodes will require a `pre-processor` (`encoder`) and `post-processor` (`decoder`) node. A `pre-processor` node should take a ROS2 message, perform the pre-processing steps dictated by the model, and then convert it to a ROS2 TensorList message. For example, a `pre-processor` node could resize an image, normalize the image, and then perform the message conversion. On the other hand, a `post-processor` node should be used to convert the output of the model inference into a usable form. For example, a `post-processor` node may perform argmax to identify the class label from a classification problem. The specific functionality of these two nodes are application-specific.
<br>
    <div align="center">![Using TensorRT or Triton](resources/pipeline.png?raw=true "Using TensorRT or Triton")</div>

This has been tested on ROS2 (Foxy) and should build and run on x86_64 and aarch64 (Jetson).

For more documentation on TensorRT, see [here](https://developer.nvidia.com/tensorrt). Note that the TensorRT node integrates the TensorRT API directly, so there is no need to make any calls or direct usage of TensorRT SDK.

For more documentation on Triton, see [here](https://github.com/triton-inference-server/server).

## System Requirements
This Isaac ROS package is designed and tested to be compatible with ROS2 Foxy on Jetson hardware, in addition to on x86 systems with an Nvidia GPU. On x86 systems, packages are only supported when run in the provided Isaac ROS Dev Docker container.

### Jetson
- AGX Xavier or Xavier NX
- JetPack 4.6

### x86_64 (in Isaac ROS Dev Docker Container)
- CUDA 11.1+ supported discrete GPU
- VPI 1.1.11
- Ubuntu 20.04+

**Note:** For best performance on Jetson, ensure that power settings are configured appropriately ([Power Management for Jetson](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0EUHA)).

### Docker
You need to use the Isaac ROS development Docker image from [Isaac ROS Common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common), based on the version 21.08 image from [Deep Learning Frameworks Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

You must first install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to make use of the Docker container development/runtime environment.

Configure `nvidia-container-runtime` as the default runtime for Docker by editing `/etc/docker/daemon.json` to include the following:
```
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
```
and then restarting Docker: `sudo systemctl daemon-reload && sudo systemctl restart docker`

Run the following script in `isaac_ros_common` to build the image and launch the container on x86_64 or Jetson:

`$ scripts/run_dev.sh <optional_path>`

### Dependencies
- [isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_common)
- [isaac_ros_nvengine](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)

## Setup
1. Create a ROS2 workspace if one is not already prepared:
   ```
   mkdir -p your_ws/src
   ```
   **Note**: The workspace can have any name; this guide assumes you name it `your_ws`.
   
2. Clone the Isaac ROS DNN Inference and Isaac ROS Common package repositories to `your_ws/src/isaac_ros_dnn_inference`. Check that you have [Git LFS](https://git-lfs.github.com/) installed before cloning to pull down all large files:
   ```
   sudo apt-get install git-lfs
   
   cd your_ws/src   
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros/isaac_ros_dnn_inference
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros/isaac_ros_common
   ```

3. Start the Docker interactive workspace:
   ```
   isaac_ros_common/scripts/run_dev.sh your_ws
   ```
   After this command, you will be inside of the container at `/workspaces/isaac_ros-dev`. Running this command in different terminals will attach to the same container.

   **Note**: The rest of this README assumes that you are inside this container.

4. Build and source the workspace:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build && . install/setup.bash
   ```
   **Note**: We recommend rebuilding the workspace each time when source files are edited. To rebuild, first clean the workspace by running `rm -r build install log`.

5. (Optional) Run tests to verify complete and correct installation:
   ```
   colcon test --executor sequential
   ```

## DNN Models
TensorRT and Triton DNN inference can work with both custom AI models and pre-trained models from the [TAO Toolkit](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html#) hosted on [NVIDIA GPU Cloud (NGC)](https://docs.nvidia.com/ngc/).
NVIDIA Train, Adapt, and Optimize (TAO) is an AI-model-adaptation platform that simplifies and accelerates the creation of enterprise AI applications and services.  

### Pre-trained Models on NGC
`TAO Toolkit` provides NVIDIA pre-trained models for Computer Vision (CV) and Conversational AI applications.
More details about pre-trained models are available [here](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html#pre-trained-models). You should be able to leverage these models for inference with the `TensorRT` and `Triton` nodes by following steps similar to the ones discussed below.

### Download Pre-trained Encrypted TLT Model (.etlt) from NGC
The following steps show how to download models, using [`PeopleSemSegnet`](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet) as an example.

1. From **File Browser** on the **PeopleSemSegnet** [page](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet), select the model `.etlt` file in the **FILE** list. Copy the `wget` command by clicking **...** in the **ACTIONS** column. 

2. Run the copied command in a terminal to download the ETLT model, as shown in the below example:  
   ```
   wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/files/peoplesemsegnet.etlt
   ```

### Convert the Encrypted TLT Model (.etlt) Format to the TensorRT Engine Plan
`tao-converter` is used to convert encrypted pre-trained models (.etlt) to the TensorRT Engine Plan.  
The pre-built `tao-converter` can be downloaded [here](https://developer.nvidia.com/tao-toolkit-get-started).    

   `tao-converter` is also included in the ISAAC-ROS docker container:  
   | Platform        | Compute library                      | Directory inside docker           |
   | --------------- | ------------------------------------ | --------------------------------- |
   | x86_64          | CUDA 11.3 / cuDNN 8.1 / TensorRT 8.0 | `/opt/nvidia/tao/cuda11.3-trt8.0` |
   | Jetson(aarch64) | Library from Jetpack 4.6             | `/opt/nvidia/tao/jp4.6`           |

   A symbolic link (`/opt/nvidia/tao/tao-converter`) is created to use `tao-converter` across different platforms.   
   **Tip**: Use `tao-converter -h` for more information on using the tool.  

Here are some examples for generating the TensorRT engine file using `tao-converter`:  

1. Generate an engine file for the fp16 data type:
   ```
   mkdir -p /workspaces/isaac_ros-dev/models
   /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_1,1x3x544x960,1x3x544x960,1x3x544x960 -t fp16 -e /workspaces/isaac_ros-dev/models/peoplesemsegnet.engine -o softmax_1 peoplesemsegnet.etlt
   ```
   **Note**: The information used above, such as the `model load key` and `input dimension`, can be retrieved from the **PeopleSemSegnet** page under the **Overview** tab. The model input node name and output node name can be found in `peoplesemsegnet_int8.txt` from `File Browser`. The output file is specified using the `-e` option. The tool needs write permission to the output directory.

2. Generate an engine file for the data type int8:  
   ```
   mkdir -p /workspaces/isaac_ros-dev/models
   cd /workspaces/isaac_ros-dev/models

   # Downloading calibration cache file for Int8.  Check model's webpage for updated wget command.
   wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/files/peoplesemsegnet_int8.txt

   # Running tao-converter
   /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_1,1x3x544x960,1x3x544x960,1x3x544x960 -t int8 -c peoplesemsegnet_int8.txt -e /workspaces/isaac_ros-dev/models/peoplesemsegnet.engine -o softmax_1 peoplesemsegnet.etlt
   ```

   **Note**: The calibration cache file (specified using the `-c` option) is required to generate the int8 engine file. For the `PeopleSemSegNet` model, it is provided in the **File Browser** tab.

### Custom AI Models
Custom user models or models re-trained through `TAO Toolkit` can be used with TensorRT and Triton DNN inference with additional configuration and encoder/decoder implementations. U-Net models are natively supported, but other model architectures can also be supported with additional work. You can implement nodes that transform and pre-process data into a `TensorList` msg (some common encoders are provided in `isaac_ros_dnn_encoders`) and translate the predicted TensorLists back into semantic messages for your graph (for example, a decoder that produces bounding boxes or image masks). To configure a custom model, you will need to specify the input and output bindings of the expected tensors to TensorRT or Triton nodes through parameters.  
   
## Triton Inference
### Setup Triton Model Repository
There are example models for using the *ONNX Runtime* backend at `/workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference/test/models/mobilenetv2-1.0_triton_onnx` and *TensorFlow* backend at `/workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference/test/models/simple_triton_tf`.

Here is an example of using the *TensorRT* backend, which uses the PeopleSemSegnet engine file generated from the **TAO-Converter** section as the model:
1. Create a `models` repository:
   ```
   mkdir -p /tmp/models/peoplesemsegnet
   ```

2. Create a `models` repository for a version(e.g. `1`):  
   ```
   mkdir -p /tmp/models/peoplesemsegnet/1
   ```
   Note that this version should match the `model_version` parameter for the Triton node.

3. Copy the generated engine file to the model repository and rename it as `model.plan`:  
   ```
   cp /workspaces/isaac_ros-dev/models/peoplesemsegnet.engine /tmp/models/peoplesemsegnet/1/model.plan
   ```

4. Create a configuration file for this model at path `/tmp/models/peoplesemsegnet/config.pbtxt`.
Note that `name` has to be the same as the model repository.
   ```
   name: "peoplesemsegnet"
   platform: "tensorrt_plan"
   max_batch_size: 0
   input [
     {
       name: "input_1"
       data_type: TYPE_FP32
       dims: [ 1, 3, 544, 960 ]
     }
   ]
   output [
     {
       name: "softmax_1"
       data_type: TYPE_FP32
       dims: [ 1, 544, 960, 2 ]
     }
   ]
   version_policy: {
     specific {
       versions: [ 1 ]
     }
   }
   ```

### Launch Triton Node

1. Build `isaac_ros_triton` package:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_triton && . install/setup.bash
   ```

2. The example launch file at `src/isaac_ros_dnn_inference/isaac_ros_triton/launch/isaac_ros_triton.py` loads and runs the `mobilenetv2-1.0` model:
   ```
   ros2 launch src/isaac_ros_dnn_inference/isaac_ros_triton/launch/isaac_ros_triton.py
   ```
   Now the Triton node is set up and running. It listens to the topic `/tensor_pub` and publishes to the topic `/tensor_sub`.

3. In a separate terminal, spin up a node that sends tensors to the Triton node:
   ```
   your_ws/src/isaac_ros_common/scripts/run_dev.sh your_ws
   . install/setup.bash
   ros2 run isaac_ros_dnn_inference_test run_test_publisher
   ```
   This test executable is configured to send random tensors with corresponding dimensions to the `/tensor_pub` topic.

4. View the output tensors from the Triton node, which should match the output dimensions of mobilenet:
   ```
   ros2 topic echo /tensor_sub
   ```
   **Note:** that the received tensor has the dimension [1, 1000], while the tensor printed out has a length of 4000 because the the data type being sent is float32, while the tensor data buffer is specified as uint8. This means that each float32 term corresponds to 4 uint8 terms.

## TensorRT Inference
### Setup ONNX file or TensorRT Engine Plan
1. TensorRT inference supports a model in either ONNX format or as a TensorRT Engine Plan. Therefore, in order to run inference using the TensorRT node, either convert your model into ONNX format, or convert it into a Engine Plan file for your hardware platform. An example for converting `.etlt` formatted models from NGC is shown above in the **DNN Models** section of the README.

2. The example model `mobilenetv2-1.0` will be used by default when using the provided launch file. To use a custom model ONNX or TensorRT Engine file, copy your ONNX or generated plan file into a known location on your filesystem: `cp mobilenetv2-1.0.onnx /tmp/model.onnx` or `cp model.plan /tmp/model.plan`.

### Run TensorRT Node

1. Build the `isaac_ros_tensor_rt` package:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_tensor_rt && . install/setup.bash
   ```

2. Start the TensorRT node (the default example ONNX model is `mobilenetv2-1.0`):
 ```
   ros2 launch src/isaac_ros_dnn_inference/isaac_ros_tensor_rt/launch/isaac_ros_tensor_rt.py
   ```

   **Note:** If using an ONNX model file, TensorRT node will first generate a TensorRT engine plan file before running inference. The `engine_file_path` is the location where the TensorRT engine plan file is generated. By default it is set to `/tmp/trt_engine.plan`.
  
   **Note:** Generating a TensorRT Engine Plan file takes time initially and it will affect your performance measures. We recommend pre-generating the engine plan file for production use.

3. Start the TensorRT node (with a custom ONNX model):

   To launch the TensorRT node using a custom model ONNX file, update the following node parameters in the launch file:
   ```
   'model_file_path': '<path-to-custom-ONNX-file>'
   ```
   This will generate a TensorRT Engine Plan at `/tmp/trt_engine.plan` and then run inference on that model. The user can also specify in the node parameters an `engine_file_path` to generate the TensorRT Engine Plan in different location.

4. Start the TensorRT Node (with a custom TensorRT Engine Plan):
   
   If using a TensorRT Engine Plan file to run inference, the `model_file_path` will not be used, so an ONNX file does not need to be provided. Instead inference will be run using the plan file provided by the parameter `engine_file_path`.
   
   To launch the TensorRT node using a custom TensorRT Engine Plan file, update the following node parameters in the launch file:
   ```
   'engine_file_path': '<path-to-custom-trt-plan-file>',
   'force_engine_update': False
   ```
   By setting `force_engine_update` to `false`, the TensorRT node will first attempt to run inference using the provided TensorRT Engine Plan file provided by the `engine_file_path` parameter. If it fails to read the engine plan file, it will attempt to generate a new plan file using the ONNX file specified in `model_file_path`. Normally, this means the node will simply fail and exit the program since the default `model_file_path` is a placeholder value of `model.onnx`, which presumably does not point to any existing ONNX file. However, if the user happens to specify a valid ONNX file (i.e. the file exists on the filesystem), then that file will be used to generate the engine plan and run inference. Therefore, it is important to not specify any `model_file_path` when running with a custom TensorRT Engine Plan file.

   Once the TensorRT node is set up, it listens to the topic `/tensor_pub` and publishes results to the topic `/tensor_sub`.

5. In a separate terminal, spin up a node that sends tensors to the TensorRT Node:
   ```
   your_ws/src/isaac_ros_common/scripts/run_dev.sh your_ws
   . install/setup.bash
   ros2 run isaac_ros_dnn_inference_test run_test_publisher
   ```
   This test executable is configured to send random tensors with corresponding dimensions to the `/tensor_pub` topic.

6. View the output tensors from the TensorRT node, which should match the output dimensions of mobilenet:
   ```
   ros2 topic echo /tensor_sub
   ```
   Note that the received tensor has the dimension [1, 1000] while the tensor printed out has a length of 4000 because the the data type being sent is float32 while the tensor data buffer is specified as uint8. This means that each float32 term corresponds to 4 uint8 terms.

## Package Reference
### `isaac_ros_tensor_rt`
#### Overview
The `isaac_ros_tensor_rt` package offers functionality to run inference on any TensorRT compatible model. It directly integrates the TensorRT API and thus does not require the user to develop any additional code to use the TensorRT SDK. You only need to provide a model in ONNX or TensorRT Engine Plan format to the TensorRT node through node options, and then launch the node to run inference. The launched node will run continously and process in real-time any incoming tensors published to it.
 
#### Available Components
| Component      | Topics Subscribed                                                  | Topics Published                                                       | Parameters                                                                                                                                                                                                               |
| -------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TensorRTNode` | `/tensor_pub`: The input tensor stream | `/tensor_sub`: The tensor list of output tensors from the model inference | `model_file_path`: The absolute path to your model file in the local file system (the model file must be `.onnx`) <br> `engine_file_path`: The absolute path to either where you want your TensorRT engine plan to be generated (from your model file) or where your pre-generated engine plan file is located <br> `force_engine_update`: If set to `true`, the node will always try to generate a TensorRT engine plan from your model file and needs to be set to false to use the pre-generated TensorRT engine plan. This parameter is set to `true` by default.<br> `input_tensor_names`: A list of tensor names to be bound to specified input bindings names. Bindings occur in sequential order, so the first name here will be mapped to the first name in `input_binding_names`. <br> `input_binding_names`: A list of input tensor binding names specified by model <br> `output_tensor_names`: A list of tensor names to be bound to specified  output binding names <br> `output_binding_names`: A list of output tensor binding names specified by model <br> `verbose`: If set to `true`, the node will enable verbose logging to console from the internal TensorRT execution. This parameter is set to `true` by default. <br> `max_workspace_size`: The size of the working space in bytes. The default value is 64MB <br> `dla_core`: The DLA Core to use. Fallback to GPU is always enabled. The default setting is GPU only. <br> `max_batch_size`: The maximum possible batch size in case the first dimension is dynamic and used as the batch size <br> `enable_fp16`: Enables building a TensorRT engine plan file which uses FP16 precision for inference. If this setting is `false`, the plan file will use FP32 precision. This setting is `true` by default <br> `relaxed_dimension_check`: Ignores dimensions of 1 for the input-tensor dimension check. |

### `isaac_ros_triton`
#### Overview
The `isaac_ros_triton` package offers functionality to run inference through a native Triton Inference Server. It allows multiple backends (*e.g.* Tensorflow, PyTorch, TensorRT) and model types. Model repositories and model configuration files need to be set up following the [Triton server instructions](https://github.com/triton-inference-server/server).

#### Available Components
| Component      | Topics Subscribed                                                  | Topics Published                                                       | Parameters                                                                                                                                                                                                               |
| -------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TritonNode` | `/tensor_pub`: The input tensor stream | `/tensor_sub`: The tensor list of output tensors from the model inference | `storage_type`: The tensor allocation storage type for `RosBridgeTensorSubscriber`. The default value is `1`. <br> `model_repository_paths`: The absolute paths to your model repositories in your local file system (repositories structure should follow Triton requirements). <br> `model_name`: The name of your model. Under `model_repository_paths`, there should be a directory with this name, and it should align with the model name in the model configuration under this directory. <br> `max_batch_size`: The maximum batch size allowed for the model. It should align with the model configuration. The default value is `8`. <br> `num_concurrent_requests`: The number of requests the Triton server can take at a time. This should be set according to the tensor publisher frequency. The default value is `65535`. <br> `input_tensor_names`: A list of tensor names to be bound to specified input bindings names. Bindings occur in sequential order, so the first name here will be mapped to the first name in `input_binding_names`. <br> `input_binding_names`: A list of input tensor binding names specified by model. <br> `output_tensor_names`: A list of tensor names to be bound to specified  output binding names. <br> `output_binding_names`: A list of output tensor binding names specified by model. |

### `isaac_ros_dnn_encoders`
#### Overview
The `isaac_ros_dnn_encoders` package offers functionality for encoding ROS2 messages into ROS2 `Tensor`
messages, including the ability to resize and normalize the tensor before outputting it. Currently, this package only supports ROS2 `Image` messages. The tensor output will be a `NCHW` tensor, where `N` is the number of batches (this will be `1` since this package targets inference), `C` is the number of color channels of the image, `H` is height of the image, and `W` is the width of the image. Therefore, a neural network that uses this package for preprocessing should support `NCHW` inputs.
 

#### Using `isaac_ros_dnn_encoders`
This package is not meant to be a standalone package, but serves as a preprocessing step before sending data to
`TensorRT` or `Triton`. Ensure that the preprocessing steps of your desired network match the preprocessing steps
performed by this node. This node is capable of image color space conversion, image resizing and image normalization. To use this node, simply add it to a launch file for your pipeline. The [`isaac_ros_unet`](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation/tree/main/isaac_ros_unet) package contains samples.

#### Available Components
| Component         | Topics Subscribed                                              | Topics Published                                                                                                                                                                                                           | Parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DnnImageEncoderNode` | `image`: The image that should be encoded into a tensor | `encoded_tensor`: The resultant tensor after converting the `image` <br> | `network_image_width`: The image width that the network expects. This will be used to resize the input `image` width. The default value is `224`. <br> `network_image_height`: The image height that the network expects. This will be used to resize the input `image` height. The default value is `224`. <br> `network_image_encoding`: The image encoding that the network expects. This will be used to convert the color space of the `image`. This should be either `rgb8` (default), `bgr8`, or `mono8`. <br> `maintain_aspect_ratio`: A flag for the encoder to crop the input image to get the aspect ratio of `network_image_width` and `network_image_height` before resizing. The default value is set to `False`. <br> `center_crop`: A flag for the encoder to crop the center of the image if `maintain_aspect_ratio` is set to `True`. The default value is set to `False`. <br>  `tensor_name`: The name of the input tensor, which is `input` by default. <br> `network_normalization_type`: The type of network normalization that should be performed on the network. This can be either `none` for no normalization, `unit_scaling` for normalization between 0 to 1, `positive_negative` for normalization between -1 to 1, and `image_normalization` for performing this normalization: <br>`(image / 255 - mean) / standard_deviation` <br> The default value is `unit_scaling`. <br> `image_mean`: If `network_normalization_type` is set to `image_normalization`, the mean of the images per channel will be used for this normalization process, which is `[0.5, 0.5, 0.5]` by default. <br> `image_stddev`: If `network_normalization_type` is set to `image_normalization`, the standard deviation of the images per channel will be used for this normalization process, which is `[0.5, 0.5, 0.5]` by default. |

**Note:** For best results, crop/resize input images to the same dimensions your DNN model is expecting. `DnnImageEncoderNode` will skew the aspect ratio of input images to the target dimensions.

## Walkthroughs
### Inference on PeopleSemSegnet using Triton
This walkthrough will run inference on the PeopleSemSegnet from NGC using `Triton`.
1. Obtain the PeopleSemSegnet ETLT file. The input dimension should be `NCHW` and the output dimension should be `NHWC` that has gone through an activation layer (e.g. softmax). The PeopleSemSegnet model follows this criteria.
   ```
   # Create a model repository for version 1
   mkdir -p /tmp/models/peoplesemsegnet/1

   # Download the model
   cd /tmp/models/peoplesemsegnet
   wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/files/peoplesemsegnet.etlt
   ```

2. Convert the `.etlt` file to a TensorRT plan file (which defaults to fp32).
   ```
   /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_1,1x3x544x960,1x3x544x960,1x3x544x960 -e /tmp/models/peoplesemsegnet/1/model.plan -o softmax_1 peoplesemsegnet.etlt
   ```
   **Note**: The TensorRT plan file should be named `model.plan`.

3. Clone [Isaac ROS Image Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation) repository into your workspace to make available the `isaac_ros_unet` package.
   ```
   cd /workspaces/isaac_ros-dev/src
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation
   ```

4. Create file `/tmp/models/peoplesemsegnet/config.pbtxt` with the following content:
   ```
   name: "peoplesemsegnet"
   platform: "tensorrt_plan"
   max_batch_size: 0
   input [
     {
       name: "input_1"
       data_type: TYPE_FP32
       dims: [ 1, 3, 544, 960 ]
     }
   ]
   output [
     {
       name: "softmax_1"
       data_type: TYPE_FP32
       dims: [ 1, 544, 960, 2 ]
     }
   ]
   version_policy: {
     specific {
       versions: [ 1 ]
     }
   }
   ```

5. Modify the `isaac_ros_unet` launch file located in `/workspaces/isaac_ros-dev/src/isaac_ros_image_segmentation/isaac_ros_unet/launch/isaac_ros_unet_triton.launch.py`. You will need to update the following lines as:
   ```
   'model_name': 'peoplesemsegnet',
   'model_repository_paths': ['/tmp/models'],
   ```
   The rest of the parameters are already set for PeopleSemSegnet. If you are using a custom model, these parameters will also need to be modified.

6. Rebuild and source `isaac_ros_unet`:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_unet && . install/setup.bash
   ```

7. Start `isaac_ros_unet` using the launch file:
   ```
   ros2 launch isaac_ros_unet isaac_ros_unet_triton.launch.py
   ```

8. Setup `image_publisher` package if not already installed.
   ```
   cd /workspaces/isaac_ros-dev/src 
   git clone --single-branch -b ros2 https://github.com/ros-perception/image_pipeline.git
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to image_publisher && . install/setup.bash
   ```

9. In a separate terminal, publish an image to `/image` using `image_publisher`. For testing purposes, we recommend using PeopleSemSegnet sample image, which is located [here](https://developer.nvidia.com/sites/default/files/akamai/NGC_Images/models/peoplenet/input_11ft45deg_000070.jpg).
   ```   
   ros2 run image_publisher image_publisher_node /workspaces/isaac_ros-dev/src/isaac_ros_image_segmentation/isaac_ros_unet/test/test_cases/unet_sample/image.jpg --ros-args -r image_raw:=image
   ```

    <div align="center"><img src="isaac_ros_unet/test/test_cases/unet_sample/image.jpg" width="600px"/></div>

10. In another terminal, launch `rqt_image_viewer` as follows:
   ```
   ros2 run rqt_image_view rqt_image_view
   ```

11. Inside the `rqt_image_view` GUI, change the topic to `/unet/colored_segmentation_mask` to view a colorized segmentation mask. You may also view the raw segmentation, which is published to `/unet/raw_segmentation_mask`, where the raw pixels correspond to the class labels making it unsuitable for human visual inspection.

    <div align="center"><img src="resources/peoplesemsegnet_segimage.png" width="600px"/></div>

These steps can easily be adapted to using TensorRT by referring to the TensorRT inference section and modifying step 4-5.

**Note:** For best results, crop/resize input images to the same dimensions your DNN model is expecting.

If you are interested in using a custom model of the U-Net architecture, please read the analogous steps for configuring [DOPE](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_dope).
  
To configure the launch file for your specific model, consult earlier documentation that describes each of these parameters. Once again, remember to verify that the preprocessing and postprocessing supported by the nodes fit your models. For example, the model should expect a `NCHW` formatted tensor, and output a `NHWC` tensor that has gone through a activation layer (e.g. softmax).

## Troubleshooting
### Nodes crashed on initial launch reporting shared libraries have a file format not recognized
Many dependent shared library binary files are stored in `git-lfs`. These files need to be fetched in order for Isaac ROS nodes to function correctly.

#### Symptoms
```
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so: file format not recognized; treating as linker script
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so:1: syntax error
collect2: error: ld returned 1 exit status
make[2]: *** [libgxe_node.so] Error 1
make[1]: *** [CMakeFiles/gxe_node.dir/all] Error 2
make: *** [all] Error 2
```
#### Solution
Run `git lfs pull` in each Isaac ROS repository you have checked out, especially `isaac_ros_common`, to ensure all of the large binary files have been downloaded.

# Updates

| Date       | Changes                                          |
| ---------- | ------------------------------------------------ |
| 2021-11-03 | Split DOPE and U-Net into separate repositories. |
| 2021-10-20 | Initial release                                  |
