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

#include "isaac_ros_tensor_proc/interleaved_to_planar_node.hpp"

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

constexpr char INPUT_COMPONENT_KEY[] = "interleaved_to_planar/data_receiver";
constexpr char INPUT_DEFAULT_IMAGE_FORMAT[] = "nitros_tensor_list_nhwc_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "interleaved_tensor";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] =
  "nitros_tensor_list_nchw_rgb_f32";
constexpr char OUTPUT_TOPIC_NAME[] = "planar_tensor";

constexpr char APP_YAML_FILENAME[] = "config/interleaved_to_planar_node.yaml";
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
  "config/namespace_injector_rule_interleaved_to_planar.yaml"};

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

InterleavedToPlanarNode::InterleavedToPlanarNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options, APP_YAML_FILENAME, CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES, EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES, EXTENSIONS, PACKAGE_NAME),
  input_tensor_shape_(declare_parameter<std::vector<int64_t>>(
      "input_tensor_shape",
      std::vector<int64_t>())),
  num_blocks_(declare_parameter<int64_t>("num_blocks", 40)),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "input_tensor"))
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

  if (input_tensor_shape_.empty()) {
    throw std::invalid_argument("[InterleavedToPlanarNode] The input shape is empty!");
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();
  startNitrosNode();
}

void InterleavedToPlanarNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "In InterleavedToPlanarNode postLoadGraphCallback().");

  int64_t block_size{std::accumulate(
      input_tensor_shape_.begin(), input_tensor_shape_.end(), 1, std::multiplies<int64_t>())};
  if (block_size <= 0) {
    throw std::invalid_argument(
            "Calculated block size is less than equal to zero! Double check the input tensor shape."
    );
  }

  getNitrosContext().setParameterUInt64(
    "interleaved_to_planar",
    "nvidia::gxf::BlockMemoryPool",
    "block_size", block_size * sizeof(float));

  getNitrosContext().setParameterUInt64(
    "interleaved_to_planar",
    "nvidia::gxf::BlockMemoryPool",
    "num_blocks", num_blocks_);

  getNitrosContext().setParameterStr(
    "interleaved_to_planar", "nvidia::isaac::tensor_ops::InterleavedToPlanar", "output_name",
    output_tensor_name_);
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode)
