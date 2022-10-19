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

#include "isaac_ros_tensor_rt/tensor_rt_node.hpp"

#include <filesystem>
#include <string>
#include <vector>

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

constexpr char TENSOR_RT_ENTITY_NAME[] = "inference";
constexpr char TENSOR_RT_COMPONENT_TYPE[] = "nvidia::gxf::TensorRtInference";

constexpr char INPUT_COMPONENT_KEY[] = "inference/rx";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "tensor_pub";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nhwc_rgb_f32";
constexpr char OUTPUT_TOPIC_NAME[] = "tensor_sub";

constexpr char APP_YAML_FILENAME[] = "config/tensor_rt_inference.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_tensor_rt";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_nitros", "gxf/std/libgxf_std.so"},
  {"isaac_ros_nitros", "gxf/cuda/libgxf_cuda.so"},
  {"isaac_ros_nitros", "gxf/serialization/libgxf_serialization.so"},
  {"isaac_ros_nitros", "gxf/tensor_rt/libgxf_tensor_rt.so"}
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_tensor_rt",
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

constexpr int64_t default_max_workspace_size = 67108864l;
constexpr int64_t default_dla_core = -1;

TensorRTNode::TensorRTNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  model_file_path_(declare_parameter<std::string>("model_file_path", "model.onnx")),
  engine_file_path_(declare_parameter<std::string>("engine_file_path", "/tmp/trt_engine.plan")),
  input_tensor_names_(declare_parameter<StringList>("input_tensor_names", StringList())),
  input_binding_names_(declare_parameter<StringList>("input_binding_names", StringList())),
  input_tensor_formats_(declare_parameter<StringList>("input_tensor_formats", StringList())),
  output_tensor_names_(declare_parameter<StringList>("output_tensor_names", StringList())),
  output_binding_names_(declare_parameter<StringList>("output_binding_names", StringList())),
  output_tensor_formats_(declare_parameter<StringList>("output_tensor_formats", StringList())),
  force_engine_update_(declare_parameter<bool>("force_engine_update", true)),
  verbose_(declare_parameter<bool>("verbose", true)),
  max_workspace_size_(declare_parameter<int64_t>(
      "max_workspace_size", default_max_workspace_size)),
  dla_core_(declare_parameter<int64_t>("dla_core", default_dla_core)),
  max_batch_size_(declare_parameter<int32_t>("max_batch_size", 1)),
  enable_fp16_(declare_parameter<bool>("enable_fp16", true)),
  relaxed_dimension_check_(declare_parameter<bool>("relaxed_dimension_check", true))
{
  RCLCPP_DEBUG(get_logger(), "[TensorRTNode] In TensorRTNode's constructor");

  if (engine_file_path_.empty()) {
    throw std::invalid_argument(
            "[TensorRTNode] Empty engine_file_path_, "
            "this needs to be set per the engine");
  }

  if (input_tensor_names_.empty()) {
    throw std::invalid_argument(
            "[TensorRTNode] Empty input_tensor_names, "
            "this needs to be set based on the input tensor messages");
  }

  if (input_binding_names_.empty()) {
    throw std::invalid_argument(
            "[TensorRTNode] Empty input_binding_names, "
            "this needs to be set per the model");
  }

  if (output_tensor_names_.empty()) {
    throw std::invalid_argument(
            "[TensorRTNode] Empty output_tensor_names, "
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
      "[TensorRTNode] Set input data format to: \"%s\"",
      input_tensor_formats_[0].c_str());
  }

  if (!output_tensor_formats_.empty()) {
    config_map_[OUTPUT_COMPONENT_KEY].compatible_data_format = output_tensor_formats_[0];
    config_map_[OUTPUT_COMPONENT_KEY].use_compatible_format_only = true;
    RCLCPP_INFO(
      get_logger(),
      "[TensorRTNode] Set output data format to: \"%s\"",
      output_tensor_formats_[0].c_str());
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

void TensorRTNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "In TensorRTNode postLoadGraphCallback().");

  // Forward TensorRT inference Parameters
  if (!force_engine_update_ && model_file_path_.empty()) {
    getNitrosContext().setParameterStr(
      TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "model_file_path",
      "model.onnx");
  } else {
    getNitrosContext().setParameterStr(
      TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "model_file_path",
      model_file_path_);
  }

  getNitrosContext().setParameterStr(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "engine_file_path",
    engine_file_path_);

  getNitrosContext().setParameterBool(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "force_engine_update",
    force_engine_update_);

  getNitrosContext().setParameterBool(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "verbose",
    verbose_);

  getNitrosContext().setParameter1DStrVector(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "input_tensor_names",
    input_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "input_binding_names",
    input_binding_names_);

  getNitrosContext().setParameter1DStrVector(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "output_tensor_names",
    output_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "output_binding_names",
    output_binding_names_);

  getNitrosContext().setParameterInt64(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "max_workspace_size",
    max_workspace_size_);

  // Only set DLA core if user sets in node options
  if (dla_core_ != default_dla_core) {
    getNitrosContext().setParameterInt64(
      TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE,
      "dla_core", dla_core_);
  }

  getNitrosContext().setParameterInt32(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "max_batch_size",
    max_batch_size_);

  getNitrosContext().setParameterBool(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE,
    "enable_fp16_", enable_fp16_);

  getNitrosContext().setParameterBool(
    TENSOR_RT_ENTITY_NAME, TENSOR_RT_COMPONENT_TYPE, "relaxed_dimension_check",
    relaxed_dimension_check_);
}

TensorRTNode::~TensorRTNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::TensorRTNode)
