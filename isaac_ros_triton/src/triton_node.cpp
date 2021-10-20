/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_triton/triton_node.hpp"

#include <string>
#include <vector>

#include "rclcpp/logger.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace dnn_inference
{

constexpr char APP_YAML_FILENAME[] = "config/triton_inference.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_triton";
const char * EXTENSIONS[] = {
  "gxf/std/libgxf_std.so",
  "gxf/cuda/libgxf_cuda.so",
  "gxf/serialization/libgxf_serialization.so",
  "gxf/triton/libgxf_triton_ext.so",
  "gxf/libgxf_ros_bridge.so"
};
constexpr uint32_t EXTENSION_LENGTH = 5;

TritonNode::TritonNode(const rclcpp::NodeOptions & options)
: nvengine::GXENode(options, APP_YAML_FILENAME, EXTENSIONS, EXTENSION_LENGTH, PACKAGE_NAME),
  storage_type_(declare_parameter<int32_t>("storage_type", 1)),
  model_name_(declare_parameter<std::string>("model_name", "")),
  // uint32_t is not supported as a parameter type
  max_batch_size_((uint32_t)declare_parameter<uint16_t>("max_batch_size", 8)),
  num_concurrent_requests_((uint32_t)declare_parameter<uint16_t>(
      "num_concurrent_requests", 65535)),
  model_repository_paths_(declare_parameter<std::vector<std::string>>(
      "model_repository_paths", std::vector<std::string>())),
  input_tensor_names_(declare_parameter<std::vector<std::string>>(
      "input_tensor_names", std::vector<std::string>())),
  input_binding_names_(declare_parameter<std::vector<std::string>>(
      "input_binding_names", std::vector<std::string>())),
  output_tensor_names_(declare_parameter<std::vector<std::string>>(
      "output_tensor_names", std::vector<std::string>())),
  output_binding_names_(declare_parameter<std::vector<std::string>>(
      "output_binding_names", std::vector<std::string>()))
{
  // Forward NVEngine bridge parameters
  SetParameterInt64(
    "tx", "nvidia::isaac_ros::RosBridgeTensorSubscriber", "node_address",
    reinterpret_cast<int64_t>(this));
  SetParameterInt64(
    "rx", "nvidia::isaac_ros::RosBridgeTensorPublisher", "node_address",
    reinterpret_cast<int64_t>(this));
  SetParameterInt32(
    "tx", "nvidia::isaac_ros::RosBridgeTensorSubscriber", "storage_type",
    storage_type_);

  // Forward Triton inference parameters
  if (model_name_.empty()) {
    throw std::invalid_argument(
            "Empty model_name, this needs to be set per the model in the model respository");
  }

  SetParameterStr(
    "triton_request", "nvidia::triton::TritonInferencerImpl", "model_name", model_name_);
  SetParameterUInt32(
    "triton_request", "nvidia::triton::TritonInferencerImpl", "max_batch_size", max_batch_size_);
  SetParameterUInt32(
    "triton_request", "nvidia::triton::TritonInferencerImpl", "num_concurrent_requests",
    num_concurrent_requests_);

  if (model_repository_paths_.empty()) {
    throw std::invalid_argument(
            "Empty model_repository_paths, this needs to be set per the model repository");
  }

  SetParameter1DStrVector(
    "triton_server", "nvidia::triton::TritonServer", "model_repository_paths",
    model_repository_paths_);

  if (input_tensor_names_.empty()) {
    throw std::invalid_argument(
            "Empty input_tensor_names, this needs to be set based on the input tensor messages");
  }

  SetParameter1DStrVector(
    "triton_request", "nvidia::triton::TritonInferenceRequest",
    "input_tensor_names", input_tensor_names_);

  if (input_binding_names_.empty()) {
    throw std::invalid_argument("Empty input_binding_names, this needs to be set per the model");
  }

  SetParameter1DStrVector(
    "triton_request", "nvidia::triton::TritonInferenceRequest",
    "input_binding_names", input_binding_names_);

  if (output_tensor_names_.empty()) {
    throw std::invalid_argument(
            "Empty output_tensor_names, this needs to be set based on the desired output "
            "tensor messages");
  }

  SetParameter1DStrVector(
    "triton_response", "nvidia::triton::TritonInferenceResponse",
    "output_tensor_names", output_tensor_names_);

  if (output_binding_names_.empty()) {
    throw std::invalid_argument("Empty output_binding_names, this needs to be set per the model");
  }

  SetParameter1DStrVector(
    "triton_response", "nvidia::triton::TritonInferenceResponse",
    "output_binding_names", output_binding_names_);
  RunGraph();
}

TritonNode::~TritonNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::dnn_inference::TritonNode)
