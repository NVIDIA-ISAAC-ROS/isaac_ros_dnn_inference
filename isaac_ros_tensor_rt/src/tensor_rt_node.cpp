/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_tensor_rt/tensor_rt_node.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include "rclcpp/logger.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace dnn_inference
{

constexpr char app_yaml_filename[] = "config/tensor_rt_inference.yaml";
const char * dnn_inference_extensions[] = {
  "gxf/std/libgxf_std.so",
  "gxf/cuda/libgxf_cuda.so",
  "gxf/serialization/libgxf_serialization.so",
  "gxf/tensor_rt/libgxf_tensor_rt.so",
  "gxf/libgxf_ros_bridge.so"
};
constexpr uint32_t extensions_length = 5;
constexpr char package_name[] = "isaac_ros_tensor_rt";
constexpr char group_name[] = "inference";
constexpr char codelet_type[] = "nvidia::isaac::TensorRtInference";
constexpr int64_t default_max_workspace_size = 67108864l;
constexpr int64_t default_dla_core = -1;


TensorRTNode::TensorRTNode(const rclcpp::NodeOptions & options, std::string custom_app_yaml)
: nvengine::GXENode(
    options, custom_app_yaml, dnn_inference_extensions,
    extensions_length, package_name),
  model_file_path_(declare_parameter<std::string>("model_file_path", "model.onnx")),
  engine_file_path_(declare_parameter<std::string>("engine_file_path", "/tmp/trt_engine.plan")),
  input_tensor_names_(declare_parameter<std::vector<std::string>>(
      "input_tensor_names", std::vector<std::string>())),
  input_binding_names_(declare_parameter<std::vector<std::string>>(
      "input_binding_names", std::vector<std::string>())),
  output_tensor_names_(declare_parameter<std::vector<std::string>>(
      "output_tensor_names", std::vector<std::string>())),
  output_binding_names_(declare_parameter<std::vector<std::string>>(
      "output_binding_names", std::vector<std::string>())),
  force_engine_update_(declare_parameter<bool>("force_engine_update", true)),
  verbose_(declare_parameter<bool>("verbose", true)),
  max_workspace_size_(declare_parameter<int64_t>(
      "max_workspace_size", default_max_workspace_size)),
  dla_core_(declare_parameter<int64_t>("dla_core", default_dla_core)),
  max_batch_size_(declare_parameter<int32_t>("max_batch_size", 1)),
  enable_fp16_(declare_parameter<bool>("enable_fp16", true)),
  relaxed_dimension_check_(declare_parameter<bool>("relaxed_dimension_check", true))
{
  // Forward NVEngine bridge parameters
  SetParameterInt64(
    "tx", "nvidia::isaac_ros::RosBridgeTensorSubscriber", "node_address",
    reinterpret_cast<int64_t>(this));
  SetParameterInt64(
    "rx", "nvidia::isaac_ros::RosBridgeTensorPublisher", "node_address",
    reinterpret_cast<int64_t>(this));

  // Forward TensorRT inference Parameters
  if (!force_engine_update_ && model_file_path_.empty()) {
    SetParameterStr(group_name, codelet_type, "model_file_path", "model.onnx");
  } else {
    SetParameterStr(group_name, codelet_type, "model_file_path", model_file_path_);
  }
  SetParameterStr(group_name, codelet_type, "engine_file_path", engine_file_path_);
  SetParameterBool(group_name, codelet_type, "force_engine_update", force_engine_update_);
  SetParameterBool(group_name, codelet_type, "verbose", verbose_);

  if (input_tensor_names_.empty()) {
    throw std::invalid_argument(
            "Empty input_tensor_names, this needs to be set based on the input tensor messages");
  }

  SetParameter1DStrVector(group_name, codelet_type, "input_tensor_names", input_tensor_names_);

  if (input_binding_names_.empty()) {
    throw std::invalid_argument("Empty input_binding_names, this needs to be set per the model");
  }

  SetParameter1DStrVector(group_name, codelet_type, "input_binding_names", input_binding_names_);

  if (output_tensor_names_.empty()) {
    throw std::invalid_argument(
            "Empty output_tensor_names, this needs to be set based on the desired output "
            "tensor messages");
  }

  SetParameter1DStrVector(group_name, codelet_type, "output_tensor_names", output_tensor_names_);

  if (output_binding_names_.empty()) {
    throw std::invalid_argument("Empty output_binding_names, this needs to be set per the model");
  }

  SetParameter1DStrVector(group_name, codelet_type, "output_binding_names", output_binding_names_);
  SetParameterInt64(group_name, codelet_type, "max_workspace_size", max_workspace_size_);
  // Only set DLA core if user sets in node options
  if (dla_core_ != default_dla_core) {
    SetParameterInt64(group_name, codelet_type, "dla_core", dla_core_);
  }
  SetParameterInt32(group_name, codelet_type, "max_batch_size", max_batch_size_);
  SetParameterBool(group_name, codelet_type, "enable_fp16_", enable_fp16_);
  SetParameterBool(group_name, codelet_type, "relaxed_dimension_check", relaxed_dimension_check_);
  RunGraph();
}

TensorRTNode::TensorRTNode(const rclcpp::NodeOptions & options)
: TensorRTNode(options, std::string(app_yaml_filename)) {}

TensorRTNode::~TensorRTNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::dnn_inference::TensorRTNode)
