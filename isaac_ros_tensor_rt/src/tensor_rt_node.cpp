// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <vector>

#include "NvInferPluginUtils.h"

#include "isaac_ros_common/qos.hpp"
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

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nhwc_rgb_f32";
constexpr char OUTPUT_TOPIC_NAME[] = "tensor_sub";

constexpr char APP_YAML_FILENAME[] = "config/tensor_rt_inference.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_tensor_rt";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"gxf_isaac_tensor_rt", "gxf/lib/libgxf_isaac_tensor_rt.so"}
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_tensor_rt",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {
  "config/isaac_ros_tensor_rt.yaml"
};
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

namespace
{
constexpr int64_t default_max_workspace_size = 67108864l;
constexpr int64_t default_dla_core = -1;

class TensorRT_Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity == Severity::kERROR) {
      RCLCPP_ERROR(rclcpp::get_logger("TRT"), "TRT ERROR: %s", msg);
    }
  }
} tensor_rt_logger;

bool readTensorShapesFromEngine(
  const std::string & engine_file_path, const std::vector<std::string> & binding_names,
  std::vector<nvinfer1::Dims> & tensor_shapes,
  std::vector<nvinfer1::DataType> & tensor_data_types)
{
  // Try to load TensorRT engine and query model output dimension
  std::vector<char> plan;
  // Open the file in binary mode and seek to the end
  std::ifstream file(engine_file_path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }

  // Get the size of the file and seek back to the beginning
  const size_t size = file.tellg();
  file.seekg(0);
  // Reserve enough space in the output buffer and read the file contents into it
  plan.resize(size);
  const bool ret = static_cast<bool>(file.read(plan.data(), size));
  file.close();

  if (!ret) {
    return false;
  }

  // Add plugins from TRT
  if (!initLibNvInferPlugins(&tensor_rt_logger, "")) {
    return false;
  }

  std::unique_ptr<nvinfer1::IRuntime> infer_runtime(
    nvinfer1::createInferRuntime(tensor_rt_logger));
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine(
    infer_runtime->deserializeCudaEngine(plan.data(), plan.size()));

  for (uint64_t i = 0; i < binding_names.size(); i++) {
    const std::string & binding_name = binding_names[i];
    tensor_shapes.push_back(cuda_engine->getTensorShape(binding_name.c_str()));
    tensor_data_types.push_back(cuda_engine->getTensorDataType(binding_name.c_str()));
  }

  return true;
}

bool readTensorShapesFromOnnx(
  const std::string & onnx_file_path, const size_t output_count,
  std::vector<nvinfer1::Dims> & tensor_shapes,
  std::vector<nvinfer1::DataType> & tensor_data_types)
{
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(tensor_rt_logger));
  std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig(builder->createBuilderConfig());
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  std::unique_ptr<nvonnxparser::IParser> onnx_parser(
    nvonnxparser::createParser(*network, tensor_rt_logger));
  if (!onnx_parser->parseFromFile(
      onnx_file_path.c_str(),
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
  {
    return false;
  }

  for (uint64_t i = 0; i < output_count; i++) {
    auto * bind_tensor = network->getOutput(i);
    tensor_shapes.push_back(bind_tensor->getDimensions());
    tensor_data_types.push_back(bind_tensor->getType());
  }

  return true;
}

}  // namespace

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
  custom_plugin_lib_(declare_parameter<std::string>("custom_plugin_lib", "")),
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
  relaxed_dimension_check_(declare_parameter<bool>("relaxed_dimension_check", true)),
  num_blocks_(declare_parameter<int64_t>("num_blocks", 40))
{
  RCLCPP_DEBUG(get_logger(), "[TensorRTNode] In TensorRTNode's constructor");

  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  rclcpp::QoS input_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT",
    "input_qos");
  rclcpp::QoS output_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT",
    "output_qos");
  for (auto & config : config_map_) {
    if (config.second.topic_name == INPUT_TOPIC_NAME) {
      config.second.qos = input_qos_;
    }
    if (config.second.topic_name == OUTPUT_TOPIC_NAME) {
      config.second.qos = output_qos_;
    }
  }

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

  if (!custom_plugin_lib_.empty()) {
    if (!dlopen(custom_plugin_lib_.c_str(), RTLD_NOW)) {
      const char * error = dlerror();
      throw std::invalid_argument(
        "[TensorRTNode] Preload plugins failed: " + std::string(error ? error : "Unknown error"));
    }
    RCLCPP_INFO(
      get_logger(),
      "[TensorRTNode] plugins: \"%s\" loaded successfully",
      custom_plugin_lib_.c_str());
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

size_t TensorRTNode::determineMaxTensorBlockSize()
{
  std::vector<nvinfer1::Dims> shapes;
  std::vector<nvinfer1::DataType> data_types;

  const auto & tensor_names = output_binding_names_;

  // Read tensor information from engine or ONNX file.
  if (readTensorShapesFromEngine(engine_file_path_, tensor_names, shapes, data_types)) {
    RCLCPP_INFO(
      get_logger(), "Read tensor shape information from TRT Model Engine: %s",
      engine_file_path_.c_str());
  } else if (readTensorShapesFromOnnx(model_file_path_, tensor_names.size(), shapes, data_types)) {
    RCLCPP_INFO(
      get_logger(), "Read tensor shape information from ONNX file: %s",
      model_file_path_.c_str());
  } else {
    RCLCPP_ERROR(
      get_logger(),
      "Unable to read tensor shape info from TRT Model Engine or from ONNX file.");
    return 0;
  }

  // Calculate maximum number of bytes needed for any single output tensor.
  size_t max_tensor_size_bytes = 1;
  for (uint64_t i = 0; i < shapes.size(); i++) {
    const auto & shape = shapes[i];
    const auto data_type = data_types[i];

    uint64_t tensor_element_count = 1;
    for (int j = 0; j < shape.nbDims; j++) {
      tensor_element_count *= std::max(shape.d[j], 1L);
    }

    uint64_t bytes_per_element;
    switch (data_type) {
      case nvinfer1::DataType::kINT8:  bytes_per_element = sizeof(int8_t); break;
      case nvinfer1::DataType::kFLOAT: bytes_per_element = sizeof(float_t); break;
      case nvinfer1::DataType::kINT32: bytes_per_element = sizeof(int32_t); break;

      // Fallback to max size of 8 bytes (64 bits) for other types
      default: bytes_per_element = sizeof(size_t);
    }

    const size_t tensor_size_bytes = tensor_element_count * bytes_per_element;
    max_tensor_size_bytes = std::max(max_tensor_size_bytes, tensor_size_bytes);
  }

  return max_tensor_size_bytes;
}


void TensorRTNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "In TensorRTNode postLoadGraphCallback().");

  uint64_t block_size = determineMaxTensorBlockSize();
  if (!block_size) {
    block_size = default_max_workspace_size;
    RCLCPP_WARN(
      get_logger(), "Failed to get block size from model, set to the default size: %ld.",
      default_max_workspace_size);
  }
  getNitrosContext().setParameterUInt64(
    TENSOR_RT_ENTITY_NAME, "nvidia::gxf::BlockMemoryPool", "block_size", block_size);

  const uint64_t output_num_blocks = output_tensor_names_.size() * num_blocks_;
  getNitrosContext().setParameterUInt64(
    TENSOR_RT_ENTITY_NAME, "nvidia::gxf::BlockMemoryPool", "num_blocks", output_num_blocks);

  RCLCPP_INFO(
    get_logger(), "Tensors %ld bytes, num outputs %ld x tensors per output %ld = %ld blocks",
    block_size, num_blocks_, output_tensor_names_.size(), output_num_blocks);


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
