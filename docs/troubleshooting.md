# DNN Inference Troubleshooting

## Seeing operation failed followed by the process dying

One cause of this issue is when the GPU being used does not have enough memory to run the model. For example, DOPE may require up to 6GB of VRAM to operate, depending on the application.

### Symptom

```log
[component_container_mt-1] 2022-06-27 08:35:37.518 ERROR extensions/tensor_ops/Reshape.cpp@71: reshape tensor failed.
[component_container_mt-1] 2022-06-27 08:35:37.518 ERROR extensions/tensor_ops/TensorOperator.cpp@151: operation failed.
[component_container_mt-1] 2022-06-27 08:35:37.518 ERROR gxf/std/entity_executor.cpp@200: Entity with 102 not found!
[component_container_mt-1] INFO: infer_simple_runtime.cpp:69 TrtISBackend id:164 initialized model: Ketchup
[component_container_mt-1] 2022-06-27 08:35:37.518 WARN  gxf/std/greedy_scheduler.cpp@221: Error while executing entity 87 named 'VERAGYEWGZ_reshaper': GXF_FAILURE
[component_container_mt-1] [ERROR] [1656318937.518424053] [dope_encoder]: [NitrosPublisher] Vault ("vault/vault", eid=102) was stopped. The graph may have been terminated due to an error.
[component_container_mt-1] terminate called after throwing an instance of 'std::runtime_error'
[component_container_mt-1]   what():  [NitrosPublisher] Vault ("vault/vault", eid=102) was stopped. The graph may have been terminated due to an error.
[ERROR] [component_container_mt-1]: process has died [pid 13378, exit code -6, cmd '/opt/ros/humble/install/lib/rclcpp_components/component_container_mt --ros-args -r __node:=dope_container -r __ns:=/'].
```

### Solution

Try using the Isaac ROS TensorRT node or the Isaac ROS Triton node with the TensorRT backend instead. Otherwise, a discrete GPU with more VRAM may be required.

## Triton fails to create the TensorRT engine and load a model

### Symptom

```log
1: [component_container_mt-1] I0331 05:56:07.479791 11359 tensorrt.cc:5591] TRITONBACKEND_ModelInitialize: detectnet (version 1)
1: [component_container_mt-1] I0331 05:56:07.483989 11359 tensorrt.cc:5640] TRITONBACKEND_ModelInstanceInitialize: detectnet (GPU device 0)
1: [component_container_mt-1] I0331 05:56:08.169240 11359 logging.cc:49] Loaded engine size: 21 MiB
1: [component_container_mt-1] E0331 05:56:08.209208 11359 logging.cc:43] 1: [runtime.cpp::parsePlan::314] Error Code 1: Serialization (Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed.)
1: [component_container_mt-1] I0331 05:56:08.213483 11359 tensorrt.cc:5678] TRITONBACKEND_ModelInstanceFinalize: delete instance state
1: [component_container_mt-1] I0331 05:56:08.213525 11359 tensorrt.cc:5617] TRITONBACKEND_ModelFinalize: delete model state
1: [component_container_mt-1] E0331 05:56:08.214059 11359 model_lifecycle.cc:596] failed to load 'detectnet' version 1: Internal: unable to create TensorRT engine
1: [component_container_mt-1] ERROR: infer_trtis_server.cpp:1057 Triton: failed to load model detectnet, triton_err_str:Invalid argument, err_msg:load failed for model 'detectnet': version 1 is at UNAVAILABLE state: Internal: unable to create TensorRT engine;
1: [component_container_mt-1] 
1: [component_container_mt-1] ERROR: infer_trtis_backend.cpp:54 failed to load model: detectnet, nvinfer error:NVDSINFER_TRITON_ERROR
1: [component_container_mt-1] ERROR: infer_simple_runtime.cpp:33 failed to initialize backend while ensuring model:detectnet ready, nvinfer error:NVDSINFER_TRITON_ERROR
1: [component_container_mt-1] ERROR: Error in createNNBackend() <infer_simple_context.cpp:76> [UID = 16]: failed to initialize triton simple runtime for model:detectnet, nvinfer error:NVDSINFER_TRITON_ERROR
1: [component_container_mt-1] ERROR: Error in initialize() <infer_base_context.cpp:79> [UID = 16]: create nn-backend failed, check config file settings, nvinfer error:NVDSINFER_TRITON_ERROR
```

### Solution

This error can occur when TensorRT attempts to load an incompatible `model.plan` file. The incompatibility may arise due to a versioning or platform mismatch between the time of plan generation and the time of plan execution.

Delete the `model.plan` file that is being passed in as an argument to the Triton node's `model_repository_paths` parameter, and then use the source package's instructions to regenerate the `model.plan` file from the original weights file (often a `.etlt` or `.onnx` file).
