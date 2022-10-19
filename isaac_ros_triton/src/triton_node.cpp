// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_triton_node/triton_node.hpp"

#include <cstdio>
#include <chrono>
#include <memory>
#include <thread>
#include <string>
#include <utility>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "triton_request/input";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "tensor_pub";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nhwc_rgb_f32";
constexpr char OUTPUT_TOPIC_NAME[] = "tensor_sub";

constexpr char APP_YAML_FILENAME[] = "config/triton_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_triton";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_nitros", "gxf/std/libgxf_std.so"},
  {"isaac_ros_nitros", "gxf/cuda/libgxf_cuda.so"},
  {"isaac_ros_nitros", "gxf/serialization/libgxf_serialization.so"},
  {"isaac_ros_nitros", "gxf/triton/libgxf_triton_ext.so"}
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_triton",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule.yaml"
};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
      .use_compatible_format_only = true,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .use_compatible_format_only = true,
      .frame_id_source_key = INPUT_COMPONENT_KEY
    }
  }
};
#pragma GCC diagnostic pop

TritonNode::TritonNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  model_name_(declare_parameter<std::string>("model_name", "")),
  // uint32_t is not supported as a parameter type
  max_batch_size_((uint32_t)declare_parameter<uint16_t>("max_batch_size", 8)),
  num_concurrent_requests_((uint32_t)declare_parameter<uint16_t>("num_concurrent_requests", 10)),
  model_repository_paths_(declare_parameter<StringList>("model_repository_paths", StringList())),
  input_tensor_names_(declare_parameter<StringList>("input_tensor_names", StringList())),
  input_binding_names_(declare_parameter<StringList>("input_binding_names", StringList())),
  input_tensor_formats_(declare_parameter<StringList>("input_tensor_formats", StringList())),
  output_tensor_names_(declare_parameter<StringList>("output_tensor_names", StringList())),
  output_binding_names_(declare_parameter<StringList>("output_binding_names", StringList())),
  output_tensor_formats_(declare_parameter<StringList>("output_tensor_formats", StringList()))
{
  RCLCPP_DEBUG(get_logger(), "[TritonNode] In TritonNode's constructor");

  if (model_name_.empty()) {
    throw std::invalid_argument(
            "[TritonNode] Empty model_name, "
            "this needs to be set per the model in the model respository");
  }

  if (model_repository_paths_.empty()) {
    throw std::invalid_argument(
            "[TritonNode] Empty model_repository_paths, "
            "this needs to be set per the model repository");
  }

  if (input_tensor_names_.empty()) {
    throw std::invalid_argument(
            "[TritonNode] Empty input_tensor_names, "
            "this needs to be set based on the input tensor messages");
  }

  if (input_binding_names_.empty()) {
    throw std::invalid_argument(
            "[TritonNode] Empty input_binding_names, "
            "this needs to be set per the model");
  }

  if (output_tensor_names_.empty()) {
    throw std::invalid_argument(
            "[TritonNode] Empty output_tensor_names, "
            "this needs to be set based on the desired output tensor messages");
  }

  if (output_binding_names_.empty()) {
    throw std::invalid_argument("Empty output_binding_names, this needs to be set per the model");
  }

  if (!input_tensor_formats_.empty()) {
    config_map_[INPUT_COMPONENT_KEY].compatible_data_format = input_tensor_formats_[0];
    config_map_[INPUT_COMPONENT_KEY].use_compatible_format_only = true;
    RCLCPP_INFO(
      get_logger(),
      "[TritonNode] Set input data format to: \"%s\"",
      input_tensor_formats_[0].c_str());
  }

  if (!output_tensor_formats_.empty()) {
    config_map_[OUTPUT_COMPONENT_KEY].compatible_data_format = output_tensor_formats_[0];
    config_map_[OUTPUT_COMPONENT_KEY].use_compatible_format_only = true;
    RCLCPP_INFO(
      get_logger(),
      "[TritonNode] Set output data format to: \"%s\"",
      output_tensor_formats_[0].c_str());
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

void TritonNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "In TritonNode postLoadGraphCallback().");

  // Forward Triton inference implementation parameters
  getNitrosContext().setParameterStr(
    "triton_request", "nvidia::triton::TritonInferencerImpl", "model_name",
    model_name_);
  getNitrosContext().setParameterUInt32(
    "triton_request", "nvidia::triton::TritonInferencerImpl", "max_batch_size",
    max_batch_size_);
  getNitrosContext().setParameterUInt32(
    "triton_request", "nvidia::triton::TritonInferencerImpl", "num_concurrent_requests",
    num_concurrent_requests_);

  // Forward Triton server parameters
  getNitrosContext().setParameter1DStrVector(
    "triton_server", "nvidia::triton::TritonServer", "model_repository_paths",
    model_repository_paths_);

  // Forward Triton inference request parameters
  getNitrosContext().setParameter1DStrVector(
    "triton_request", "nvidia::triton::TritonInferenceRequest", "input_tensor_names",
    input_tensor_names_);
  getNitrosContext().setParameter1DStrVector(
    "triton_request", "nvidia::triton::TritonInferenceRequest", "input_binding_names",
    input_binding_names_);

  // Forward Triton inference response parameters
  getNitrosContext().setParameter1DStrVector(
    "triton_response", "nvidia::triton::TritonInferenceResponse", "output_tensor_names",
    output_tensor_names_);
  getNitrosContext().setParameter1DStrVector(
    "triton_response", "nvidia::triton::TritonInferenceResponse", "output_binding_names",
    output_binding_names_);
}

TritonNode::~TritonNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::TritonNode)
