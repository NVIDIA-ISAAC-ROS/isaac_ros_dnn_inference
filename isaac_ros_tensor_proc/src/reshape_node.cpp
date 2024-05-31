// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_tensor_proc/reshape_node.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
namespace
{

constexpr char INPUT_COMPONENT_KEY[] = "reshaper/data_receiver";
constexpr char INPUT_DEFAULT_IMAGE_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "tensor";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] =
  "nitros_tensor_list_nchw_rgb_f32";
constexpr char OUTPUT_TOPIC_NAME[] = "reshaped_tensor";

constexpr char APP_YAML_FILENAME[] = "config/reshape_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_tensor_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"gxf_isaac_tensorops", "gxf/lib/libgxf_isaac_tensorops.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_dnn_encoders",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule_reshaper.yaml"};

const std::map<gxf::optimizer::ComponentKey, std::string>
COMPATIBLE_DATA_FORMAT_MAP = {
  {INPUT_COMPONENT_KEY, INPUT_DEFAULT_IMAGE_FORMAT},
  {OUTPUT_COMPONENT_KEY, OUTPUT_DEFAULT_TENSOR_FORMAT}
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_IMAGE_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
      .use_compatible_format_only = false,
    }},
  {OUTPUT_COMPONENT_KEY,
    {.type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .use_compatible_format_only = false,
      .frame_id_source_key = INPUT_COMPONENT_KEY}}};
}  // namespace
#pragma GCC diagnostic pop

ReshapeNode::ReshapeNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options, APP_YAML_FILENAME, CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES, EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES, EXTENSIONS, PACKAGE_NAME),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "input_tensor")),
  input_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "input_tensor_shape",
      std::vector<int64_t>())),
  output_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "output_tensor_shape",
      std::vector<int64_t>())),
  num_blocks_(declare_parameter<int64_t>("num_blocks", 40))
{
  rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos");
  rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos");
  for (auto & config : config_map_) {
    if (config.second.topic_name == INPUT_TOPIC_NAME) {
      config.second.qos = input_qos;
    } else {
      config.second.qos = output_qos;
    }
  }

  if (input_tensor_shape_.empty() || output_tensor_shape_.empty()) {
    throw std::invalid_argument("[ReshapeNode] The input or output tensor shape is empty!");
  }

  const int64_t input_element_count{std::accumulate(
      input_tensor_shape_.begin(),
      input_tensor_shape_.end(),
      1,
      std::multiplies<int64_t>())};

  const int64_t output_element_count{std::accumulate(
      output_tensor_shape_.begin(),
      output_tensor_shape_.end(),
      1,
      std::multiplies<int64_t>())};

  if (input_element_count != output_element_count) {
    throw std::invalid_argument(
            "[ReshapeNode] The input element count and output element count do not match!");
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();
  startNitrosNode();
}

void ReshapeNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "In ReshapeNode postLoadGraphCallback().");

  int64_t block_size{std::accumulate(
      input_tensor_shape_.begin(), input_tensor_shape_.end(), 1, std::multiplies<int64_t>())};
  if (block_size <= 0) {
    throw std::invalid_argument(
            "Calculated block size is less than equal to zero! Double check the input tensor shape."
    );
  }

  getNitrosContext().setParameterUInt64(
    "reshaper",
    "nvidia::gxf::BlockMemoryPool",
    "block_size", block_size * sizeof(float));

  getNitrosContext().setParameterUInt64(
    "reshaper",
    "nvidia::gxf::BlockMemoryPool",
    "num_blocks", num_blocks_);
  std::vector<int32_t> final_shape;
  std::transform(
    output_tensor_shape_.begin(),
    output_tensor_shape_.end(),
    std::back_inserter(final_shape),
    [](const int64_t value) {
      return static_cast<int32_t>(value);
    }
  );
  getNitrosContext().setParameter1DInt32Vector(
    "reshaper", "nvidia::isaac::tensor_ops::Reshape", "output_shape",
    final_shape);

  getNitrosContext().setParameterStr(
    "reshaper", "nvidia::isaac::tensor_ops::Reshape", "output_name",
    output_tensor_name_);
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::ReshapeNode)
