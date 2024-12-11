# Isaac ROS DNN Inference

NVIDIA-accelerated DNN model inference ROS 2 packages using NVIDIA Triton/TensorRT for both Jetson and x86_64 with CUDA-capable GPU.

<div align="center"><img alt="bounding box for people detection" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_peoplenet.jpg/" width="300px"/>
<img alt="segementation mask for people detection" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_inference_peoplesemsegnet.jpg/" width="300px"/></div>

## Webinar Available

Learn how to use this package by watching our on-demand webinar:
[Accelerate YOLOv5 and Custom AI Models in ROS with NVIDIA Isaac](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1407606/3998202/isaac-ros-webinar-series)

---

## Overview

[Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference) contains ROS 2 packages for performing DNN
inference, providing AI-based perception for robotics applications. DNN
inference uses a pre-trained DNN model to ingest an input Tensor and
output a prediction to an output Tensor.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_inference_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_inference_nodegraph.png/" width="800px"/></a></div>

Above is a typical graph of nodes for DNN inference on image data. The
input image is resized to match the input resolution of the DNN; the
image resolution may be reduced to improve DNN inference performance
,which typically scales directly with the number of pixels in the image.
DNN inference requires input Tensors, so a DNN encoder node is used to
convert from an input image to Tensors, including any data
pre-processing that is required for the DNN model. Once DNN inference is
performed, the DNN decoder node is used to convert the output Tensors to
results that can be used by the application.

TensorRT and Triton are two separate ROS nodes to perform DNN inference.
The TensorRT node uses
[TensorRT](https://developer.nvidia.com/tensorrt) to provide
high-performance deep learning inference. TensorRT optimizes the DNN
model for inference on the target hardware, including Jetson and
discrete GPUs. It also supports specific operations that are commonly
used by DNN models. For newer or bespoke DNN models, TensorRT may not
support inference on the model. For these models, use the Triton node.

The Triton node uses the [Triton Inference
Server](https://developer.nvidia.com/nvidia-triton-inference-server),
which provides a compatible frontend supporting a combination of
different inference backends (e.g. ONNX Runtime, TensorRT Engine Plan,
TensorFlow, PyTorch). In-house benchmark results measure little
difference between using TensorRT directly or configuring Triton to use
TensorRT as a backend.

Some DNN models may require custom DNN encoders to convert the input
data to the Tensor format needed for the model, and custom DNN decoders
to convert from output Tensors into results that can be used in the
application. Leverage the DNN encoder and DNN decoder node(s) for image
bounding box detection and image segmentation, or your own custom
node(s).

> [!Note]
> DNN inference can be performed on different types of input
> data, including audio, video, text, and various sensor data, such as
> LIDAR, camera, and RADAR. This package provides implementations for
> DNN encode and DNN decode functions for images, which are commonly
> used for perception in robotics. The DNNs operate on Tensors for
> their input, output, and internal transformations, so the input image
> needs to be converted to a Tensor for DNN inferencing.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                       | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                                       | Orin NX<br/><br/>                                                                                                                                                       | Orin Nano 8GB<br/><br/>                                                                                                                                                 | x86_64 w/ RTX 4090<br/><br/>                                                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [TensorRT Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_tensor_rt_benchmark/scripts/isaac_ros_tensor_rt_dope_node.py)<br/><br/><br/>DOPE<br/><br/>            | VGA<br/><br/><br/><br/>  | [47.9 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_dope_node-agx_orin.json)<br/><br/><br/>24 ms @ 30Hz<br/><br/>   | [18.1 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_dope_node-orin_nx.json)<br/><br/><br/>56 ms @ 30Hz<br/><br/>   | [13.1 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_dope_node-orin_nano.json)<br/><br/><br/>81 ms @ 30Hz<br/><br/> | [298 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_dope_node-x86-4090.json)<br/><br/><br/>4.6 ms @ 30Hz<br/><br/>    |
| [Triton Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_triton_benchmark/scripts/isaac_ros_triton_dope_node.py)<br/><br/><br/>DOPE<br/><br/>                    | VGA<br/><br/><br/><br/>  | [47.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_triton_dope_node-agx_orin.json)<br/><br/><br/>24 ms @ 30Hz<br/><br/>      | [20.3 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_triton_dope_node-orin_nx.json)<br/><br/><br/>530 ms @ 30Hz<br/><br/>     | [14.5 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_triton_dope_node-orin_nano.json)<br/><br/><br/>780 ms @ 30Hz<br/><br/>   | [276 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_triton_dope_node-x86-4090.json)<br/><br/><br/>4.6 ms @ 30Hz<br/><br/>       |
| [TensorRT Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_tensor_rt_benchmark/scripts/isaac_ros_tensor_rt_ps_node.py)<br/><br/><br/>PeopleSemSegNet<br/><br/>   | 544p<br/><br/><br/><br/> | [460 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_ps_node-agx_orin.json)<br/><br/><br/>4.1 ms @ 30Hz<br/><br/>     | [348 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_ps_node-orin_nx.json)<br/><br/><br/>6.1 ms @ 30Hz<br/><br/>     | [238 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_ps_node-orin_nano.json)<br/><br/><br/>7.0 ms @ 30Hz<br/><br/>   | [685 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_tensor_rt_ps_node-nuc_4060ti.json)<br/><br/><br/>2.9 ms @ 30Hz<br/><br/>    |
| [Triton Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_triton_benchmark/scripts/isaac_ros_triton_ps_node.py)<br/><br/><br/>PeopleSemSegNet<br/><br/>           | 544p<br/><br/><br/><br/> | [304 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_triton_ps_node-agx_orin.json)<br/><br/><br/>4.8 ms @ 30Hz<br/><br/>        | [206 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_triton_ps_node-orin_nx.json)<br/><br/><br/>6.5 ms @ 30Hz<br/><br/>        | –<br/><br/><br/><br/>                                                                                                                                                   | [677 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_triton_ps_node-nuc_4060ti.json)<br/><br/><br/>2.2 ms @ 30Hz<br/><br/>       |
| [DNN Image Encoder Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_dnn_image_encoder_benchmark/scripts/isaac_ros_dnn_image_encoder_node.py)<br/><br/><br/><br/> | VGA<br/><br/><br/><br/>  | [420 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dnn_image_encoder_node-agx_orin.json)<br/><br/><br/>12 ms @ 30Hz<br/><br/> | [382 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dnn_image_encoder_node-orin_nx.json)<br/><br/><br/>11 ms @ 30Hz<br/><br/> | –<br/><br/><br/><br/>                                                                                                                                                   | [574 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dnn_image_encoder_node-x86-4090.json)<br/><br/><br/>5.2 ms @ 30Hz<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_dnn_image_encoder`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_image_encoder/index.html)
  * [Migration Guide](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_image_encoder/index.html#migration-guide)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_image_encoder/index.html#api)
* [`isaac_ros_tensor_proc`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_tensor_proc/index.html)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_tensor_proc/index.html#api)
* [`isaac_ros_tensor_rt`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_tensor_rt/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_tensor_rt/index.html#quickstart)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_tensor_rt/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_tensor_rt/index.html#api)
* [`isaac_ros_triton`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_triton/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_triton/index.html#quickstart)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_triton/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_triton/index.html#api)

## Latest

Update 2024-12-10: Update to be compatible with JetPack 6.1
