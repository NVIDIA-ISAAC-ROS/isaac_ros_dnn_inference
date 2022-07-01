# DNN Inference Troubleshooting
## Seeing operation failed followed by the process dying
One cause of this issue is when the GPU being used does not have enough memory to run the model. For example, DOPE may require up to 6GB of VRAM to operate, depending on the application.

### Symptom
```
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
