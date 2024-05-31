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

#include "isaac_ros_tensor_proc/normalize_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{
namespace
{

// Map to store the nitros tensor format to nitros image type
const std::unordered_map<std::string, std::string> tensor_to_image_type(
  {{"nitros_tensor_list_nhwc_rgb_f32", "nitros_image_rgb8"},
    {"nitros_tensor_list_nhwc_bgr_f32", "nitros_image_bgr8"}});

constexpr double PER_PIXEL_SCALE = 255.0;

constexpr char INPUT_COMPONENT_KEY[] = "normalizer/data_receiver";
constexpr char INPUT_DEFAULT_IMAGE_FORMAT[] = "nitros_image_rgb8";
constexpr char INPUT_TOPIC_NAME[] = "image";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] =
  "nitros_tensor_list_nhwc_rgb_f32";
constexpr char OUTPUT_TOPIC_NAME[] = "normalized_tensor";

constexpr char APP_YAML_FILENAME[] = "config/normalizer_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_tensor_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_tensorops", "gxf/lib/libgxf_isaac_tensorops.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_dnn_encoders",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule_normalizer.yaml"};

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

NormalizeNode::NormalizeNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options, APP_YAML_FILENAME, CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES, EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES, EXTENSIONS, PACKAGE_NAME),
  image_mean_(declare_parameter<std::vector<double>>(
      "image_mean",
      {0.5, 0.5, 0.5})),
  image_stddev_(declare_parameter<std::vector<double>>(
      "image_stddev",
      {0.5, 0.5, 0.5})),
  input_image_width_(declare_parameter<uint16_t>("input_image_width", 0)),
  input_image_height_(declare_parameter<uint16_t>("input_image_height", 0)),
  num_blocks_(declare_parameter<int64_t>("num_blocks", 40)),
  output_tensor_name_(declare_parameter<std::string>("output_tensor_name", "image"))
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

  if (input_image_width_ == 0) {
    throw std::invalid_argument(
            "[NormalizeNode] Invalid input_image_width");
  }
  if (input_image_height_ == 0) {
    throw std::invalid_argument(
            "[NormalizeNode] Invalid input_image_height");
  }

  if (image_mean_.size() != 3 || image_stddev_.size() != 3) {
    throw std::invalid_argument(
            "[NormalizeNode] Did not receive 3 image mean channels or 3 image stddev channels");
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();
  startNitrosNode();
}

void NormalizeNode::preLoadGraphCallback()
{
  RCLCPP_INFO(
    get_logger(),
    "In NormalizeNode Node preLoadGraphCallback().");

  std::vector<double> scale = {1.0 / (PER_PIXEL_SCALE * image_stddev_[0]),
    1.0 / (PER_PIXEL_SCALE * image_stddev_[1]),
    1.0 / (PER_PIXEL_SCALE * image_stddev_[2])};
  std::vector<double> offset = {-(PER_PIXEL_SCALE * image_mean_[0]),
    -(PER_PIXEL_SCALE * image_mean_[1]),
    -(PER_PIXEL_SCALE * image_mean_[2])};

  std::string scales = "[" + std::to_string(scale[0]) + "," +
    std::to_string(scale[1]) + "," +
    std::to_string(scale[2]) + "]";
  NitrosNode::preLoadGraphSetParameter(
    "normalizer", "nvidia::isaac::tensor_ops::Normalize", "scales", scales);

  std::string offsets = "[" + std::to_string(offset[0]) + "," +
    std::to_string(offset[1]) + "," +
    std::to_string(offset[2]) + "]";
  NitrosNode::preLoadGraphSetParameter(
    "normalizer",
    "nvidia::isaac::tensor_ops::Normalize",
    "offsets", offsets);
}

void NormalizeNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "In Normalizer Node postLoadGraphCallback().");

  const gxf::optimizer::ComponentInfo output_comp_info = {
    "nvidia::isaac_ros::MessageRelay",     // component_type_name
    "sink",                                // component_name
    "sink"                                 // entity_name
  };
  const std::string negotiated_tensor_format =
    getFinalDataFormat(output_comp_info);
  const auto image_type = tensor_to_image_type.find(negotiated_tensor_format);
  if (image_type == std::end(tensor_to_image_type)) {
    RCLCPP_ERROR(
      get_logger(), "Unsupported NITROS tensor type[%s].",
      negotiated_tensor_format.c_str());
    throw std::runtime_error("Unsupported NITROS tensor type.");
  }

  uint64_t block_size = calculate_image_size(
    image_type->second, input_image_width_, input_image_height_);

  RCLCPP_DEBUG(
    get_logger(), "postLoadGraphCallback() block_size = %ld.",
    block_size);

  getNitrosContext().setParameterUInt64(
    "normalizer",
    "nvidia::gxf::BlockMemoryPool",
    "block_size", block_size * sizeof(float));

  getNitrosContext().setParameterStr(
    "normalizer", "nvidia::isaac::tensor_ops::Normalize", "output_name",
    output_tensor_name_);

  getNitrosContext().setParameterUInt64(
    "normalizer",
    "nvidia::gxf::BlockMemoryPool",
    "num_blocks", num_blocks_);
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::NormalizeNode)
