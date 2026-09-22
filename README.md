# Isaac ROS DNN Inference

NVIDIA-accelerated DNN model inference ROS 2 packages using NVIDIA Triton/TensorRT for both Jetson and x86_64 with CUDA-capable GPU.

<div align="center"><img alt="bounding box for people detection" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-5.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_peoplenet.jpg/" width="300px"/>
<img alt="segementation mask for people detection" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-5.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_inference_peoplesemsegnet.jpg/" width="300px"/></div>

## Webinar Available

Learn how to use this package by watching our on-demand webinar:
[Accelerate YOLOv5 and Custom AI Models in ROS with NVIDIA Isaac](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1407606/3998202/isaac-ros-webinar-series)

---

## Overview

Isaac ROS DNN Inference contains ROS 2 packages for performing DNN
inference, providing AI-based perception for robotics applications. DNN
inference uses a pre-trained DNN model to ingest an input Tensor and
output a prediction to an output Tensor.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-5.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_inference_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-5.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_inference_nodegraph.png/" width="800px"/></a></div>

Above is a typical graph of nodes for DNN inference on image data. The
input image is resized to match the input resolution of the DNN; the
image resolution may be reduced to improve DNN inference performance,
which typically scales directly with the number of pixels in the image.
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
Server](https://developer.nvidia.com/dynamo-triton),
which provides a compatible frontend supporting a combination of
different inference backends (e.g. ONNX Runtime, TensorRT Engine Plan,
TensorFlow, PyTorch). In-house benchmark results measure little
difference between using TensorRT directly or configuring Triton to use
TensorRT as a backend.

Some DNN models may require custom DNN encoders to convert the input
data to the Tensor format needed for the model, and custom DNN decoders
to convert from output Tensors into results that can be used in the
application. Leverage the DNN encoder and DNN decoder nodes for image
bounding box detection and image segmentation, or your own custom
nodes.

> [!Note]
> DNN inference can be performed on different types of input
> data, including audio, video, text, and various sensor data, such as
> LIDAR, camera, and RADAR. This package provides implementations for
> DNN encode and DNN decode functions for images, which are commonly
> used for perception in robotics. The DNNs operate on Tensors for
> their input, output, and internal transformations, so the input image
> needs to be converted to a Tensor for DNN inferencing.

## ROS 2 Native `rosidl::Buffer` Acceleration

This package uses `rosidl::Buffer`, a feature built into ROS 2 Lyrical, to
avoid unnecessary copies of large payloads between CPU and accelerator
memory. The CUDA buffer backend builds on this native ROS 2 feature to provide
CUDA memory storage and transport. Most applications can use standard ROS
messages and conversion packages without depending directly on a buffer
backend. See [rosidl::Buffer and Buffer Backends](https://nvidia-isaac-ros.github.io/concepts/rosidl_buffer/index.html) for details.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                            | Input Size<br/><br/>   | AGX Thor T5000<br/><br/>                                                                                                                                                         | AGX Thor T4000<br/><br/>                                                                                                                                                           | AGX Orin<br/><br/>                                                                                                                                                               | Orin Nano Super 8GB<br/><br/>                                                                                                                                                     | DGX Spark<br/><br/>                                                                                                                                                                | x86_64 w/ RTX 5090<br/><br/>                                                                                                                                                       | x86_64 w/ RTX 5070<br/><br/>                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [TensorRT Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/benchmarks/isaac_ros_tensor_rt_benchmark/scripts/isaac_ros_tensor_rt_dope_node.py)<br/><br/><br/>DOPE<br/><br/>          | VGA<br/><br/>          | [193 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_dope_node-agx_thor.json)<br/><br/><br/>1.4 ms @ 30Hz<br/><br/>    | [128 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_dope_node-thor-t4000.json)<br/><br/><br/>1.3 ms @ 30Hz<br/><br/>    | [47.4 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_dope_node-agx_orin.json)<br/><br/><br/>1.9 ms @ 30Hz<br/><br/>   | [20.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_dope_node-orin_nano.json)<br/><br/>                               | [108 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_dope_node-dgx_spark.json)<br/><br/><br/>1.2 ms @ 30Hz<br/><br/>     | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_dope_node-x86-5090.json)<br/><br/><br/>0.53 ms @ 30Hz<br/><br/>     | [161 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_dope_node-x86-rtx5070.json)<br/><br/><br/>0.48 ms @ 30Hz<br/><br/>   |
| [Triton Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/benchmarks/isaac_ros_triton_benchmark/scripts/isaac_ros_triton_dope_node.py)<br/><br/><br/>DOPE<br/><br/>                  | VGA<br/><br/>          | [174 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_dope_node-agx_thor.json)<br/><br/><br/>6.2 ms @ 30Hz<br/><br/>       | [116 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_dope_node-thor-t4000.json)<br/><br/><br/>10 ms @ 30Hz<br/><br/>        | [45.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_dope_node-agx_orin.json)<br/><br/><br/>22 ms @ 30Hz<br/><br/>       | [20.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_dope_node-orin_nano.json)<br/><br/>                                  | [98.1 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_dope_node-dgx_spark.json)<br/><br/><br/>8.6 ms @ 30Hz<br/><br/>       | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_dope_node-x86-5090.json)<br/><br/><br/>2.9 ms @ 30Hz<br/><br/>         | [156 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_dope_node-x86-rtx5070.json)<br/><br/><br/>6.4 ms @ 30Hz<br/><br/>       |
| [TensorRT Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/benchmarks/isaac_ros_tensor_rt_benchmark/scripts/isaac_ros_tensor_rt_ps_node.py)<br/><br/><br/>PeopleSemSegNet<br/><br/> | 544p<br/><br/>         | [792 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_ps_node-agx_thor.json)<br/><br/><br/>1.2 ms @ 30Hz<br/><br/>      | [524 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_ps_node-thor-t4000.json)<br/><br/><br/>1.4 ms @ 30Hz<br/><br/>      | [481 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_ps_node-agx_orin.json)<br/><br/><br/>1.5 ms @ 30Hz<br/><br/>      | [234 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_ps_node-orin_nano.json)<br/><br/><br/>1.8 ms @ 30Hz<br/><br/>      | [1030 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_ps_node-dgx_spark.json)<br/><br/><br/>0.87 ms @ 30Hz<br/><br/>     | [3000 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_ps_node-x86-5090.json)<br/><br/><br/>0.46 ms @ 30Hz<br/><br/>      | [2320 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_tensor_rt_ps_node-x86-rtx5070.json)<br/><br/><br/>0.52 ms @ 30Hz<br/><br/>    |
| [Triton Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/benchmarks/isaac_ros_triton_benchmark/scripts/isaac_ros_triton_ps_node.py)<br/><br/><br/>PeopleSemSegNet<br/><br/>         | 544p<br/><br/>         | [195 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_ps_node-agx_thor.json)<br/><br/><br/>6.3 ms @ 30Hz<br/><br/>         | [191 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_ps_node-thor-t4000.json)<br/><br/><br/>6.7 ms @ 30Hz<br/><br/>         | [189 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_ps_node-agx_orin.json)<br/><br/><br/>5.5 ms @ 30Hz<br/><br/>         | [122 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_ps_node-orin_nano.json)<br/><br/><br/>8.4 ms @ 30Hz<br/><br/>         | [173 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_ps_node-dgx_spark.json)<br/><br/><br/>5.7 ms @ 30Hz<br/><br/>          | [2440 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_ps_node-x86-5090.json)<br/><br/><br/>0.53 ms @ 30Hz<br/><br/>         | [1630 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_triton_ps_node-x86-rtx5070.json)<br/><br/><br/>0.74 ms @ 30Hz<br/><br/>       |
| [DNN Image Encoder Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/benchmarks/isaac_ros_dnn_image_encoder_benchmark/scripts/isaac_ros_dnn_image_encoder_node.py)<br/><br/>         | 1200p<br/><br/>        | [767 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_dnn_image_encoder_node-agx_thor.json)<br/><br/><br/>2.5 ms @ 30Hz<br/><br/> | [418 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_dnn_image_encoder_node-thor-t4000.json)<br/><br/><br/>4.6 ms @ 30Hz<br/><br/> | [423 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_dnn_image_encoder_node-agx_orin.json)<br/><br/><br/>4.5 ms @ 30Hz<br/><br/> | [475 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_dnn_image_encoder_node-orin_nano.json)<br/><br/><br/>4.5 ms @ 30Hz<br/><br/> | [1130 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_dnn_image_encoder_node-dgx_spark.json)<br/><br/><br/>1.9 ms @ 30Hz<br/><br/> | [2870 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_dnn_image_encoder_node-x86-5090.json)<br/><br/><br/>0.91 ms @ 30Hz<br/><br/> | [449 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-5.0/results/isaac_ros_dnn_image_encoder_node-x86-rtx5070.json)<br/><br/><br/>2.5 ms @ 30Hz<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_dnn_image_encoder`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_image_encoder/index.html)
  * [Using the Launch Graph](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_inference/isaac_ros_dnn_image_encoder/index.html#using-the-launch-graph)
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

Update 2026-09-21: Migrated the TensorRT and Triton nodes from NITROS to rosidl::Buffer with the CUDA buffer backend
