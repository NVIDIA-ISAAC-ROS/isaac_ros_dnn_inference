// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "NvInferPluginUtils.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace TrtConv = tensorrt_conversions;
using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

constexpr char INPUT_TOPIC_NAME[] = "tensor_pub";
constexpr char OUTPUT_TOPIC_NAME[] = "tensor_sub";

constexpr int64_t default_max_workspace_size = 67108864l;
constexpr int64_t default_dla_core = -1;

namespace
{

class TensorRT_Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    // Suppress logs below the desired severity level
    if (severity > log_level) {
      return;
    }
    if (severity == Severity::kINTERNAL_ERROR) {
      RCLCPP_ERROR(rclcpp::get_logger("TRT"), "TRT INTERNAL_ERROR: %s", msg);
    }
    if (severity == Severity::kERROR) {
      RCLCPP_ERROR(rclcpp::get_logger("TRT"), "TRT ERROR: %s", msg);
    }
    if (severity == Severity::kINFO) {
      RCLCPP_INFO(rclcpp::get_logger("TRT"), "TRT INFO: %s", msg);
    }
    if (severity == Severity::kWARNING) {
      RCLCPP_WARN(rclcpp::get_logger("TRT"), "TRT WARNING: %s", msg);
    }
    if (severity == Severity::kVERBOSE) {
      RCLCPP_DEBUG(rclcpp::get_logger("TRT"), "TRT VERBOSE: %s", msg);
    }
  }

  void setReportableSeverity(Severity severity)
  {
    log_level = severity;
  }

private:
  Severity log_level = Severity::kINFO;
};
TensorRT_Logger tensor_rt_logger;

size_t GetElementSizeFromDataType(nvinfer1::DataType data_type)
{
  size_t element_size = 1;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      element_size = sizeof(float);
      break;
    case nvinfer1::DataType::kHALF:
      element_size = sizeof(uint16_t);
      break;
    case nvinfer1::DataType::kINT8:
      element_size = sizeof(int8_t);
      break;
    case nvinfer1::DataType::kINT32:
      element_size = sizeof(int32_t);
      break;
    case nvinfer1::DataType::kBOOL:
      element_size = sizeof(bool);
      break;
    case nvinfer1::DataType::kUINT8:
      element_size = sizeof(uint8_t);
      break;
    case nvinfer1::DataType::kINT64:
      element_size = sizeof(int64_t);
      break;
    case nvinfer1::DataType::kBF16:
      element_size = sizeof(uint16_t);
      break;
    case nvinfer1::DataType::kFP8:
      element_size = sizeof(uint8_t);
      break;
    case nvinfer1::DataType::kINT4:
      element_size = 1;  // 4 bits, round up
      break;
    case nvinfer1::DataType::kFP4:
      element_size = 1;  // 4 bits, round up
      break;
    case nvinfer1::DataType::kE8M0:
      element_size = sizeof(uint8_t);
      break;
    default:
      element_size = 1;
      break;
  }
  return element_size;
}

nvinfer1::Dims ResolveInputDims(
  const TrtConv::Tensor & input_tensor,
  const std::string & tensor_name,
  const nvinfer1::Dims & binding_dims_template)
{
  // TensorRT requires setInputShape() to use the engine binding rank. Some
  // upstream tensors are unbatched, for example CHW image tensors feeding an
  // explicit-batch NCHW engine, so use the cached binding dims as the shape
  // template instead of copying the incoming tensor metadata rank directly.
  nvinfer1::Dims dims = binding_dims_template;
  const auto & input_shape = input_tensor.shape;
  const int32_t input_rank = static_cast<int32_t>(input_shape.size());
  const int32_t rank_delta = dims.nbDims - input_rank;
  // The input either matches the binding rank or omits only the leading batch
  // dimension. Any other rank difference is ambiguous and cannot be mapped
  // safely to TensorRT binding dimensions.
  if (rank_delta != 0 && rank_delta != 1) {
    throw std::runtime_error(
            "[TensorRTNode] Input tensor rank does not match binding rank: " + tensor_name);
  }

  // Dynamic batch is the only runtime dimension handled here. If the tensor
  // omitted batch, only a unit-batch static engine can be inferred safely.
  if (dims.d[0] < 0) {
    dims.d[0] = rank_delta == 0 ? input_shape[0] : 1;
  } else if (rank_delta == 1 && dims.d[0] != 1) {
    throw std::runtime_error(
            "[TensorRTNode] Input tensor omits non-unit batch dimension: " + tensor_name);
  }

  // Keep non-batch dimensions from the engine binding. Supporting dynamic
  // C/H/W would require layout-specific mapping instead of assuming that
  // incoming tensor metadata and TensorRT binding dimensions use the same
  // index order.
  for (int32_t j = 1; j < dims.nbDims; ++j) {
    if (dims.d[j] < 0) {
      throw std::runtime_error(
              "[TensorRTNode] Dynamic non-batch dimensions are not supported for input tensor: " +
              tensor_name);
    }
  }
  return dims;
}

}  // namespace

TensorRTNode::TensorRTNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("tensor_rt_node", options),
  model_file_path_(declare_parameter<std::string>("model_file_path", "model.onnx")),
  engine_file_path_(declare_parameter<std::string>("engine_file_path", "/tmp/trt_engine.plan")),
  custom_plugin_lib_(declare_parameter<std::string>("custom_plugin_lib", "")),
  input_tensor_names_(declare_parameter<StringList>("input_tensor_names", StringList())),
  input_binding_names_(declare_parameter<StringList>("input_binding_names", StringList())),
  output_tensor_names_(declare_parameter<StringList>("output_tensor_names", StringList())),
  output_binding_names_(declare_parameter<StringList>("output_binding_names", StringList())),
  force_engine_update_(declare_parameter<bool>("force_engine_update", true)),
  verbose_(declare_parameter<bool>("verbose", true)),
  max_workspace_size_(declare_parameter<int64_t>(
      "max_workspace_size", default_max_workspace_size)),
  dla_core_(declare_parameter<int64_t>("dla_core", default_dla_core)),
  max_batch_size_(declare_parameter<int32_t>("max_batch_size", 1)),
  enable_fp16_(declare_parameter<bool>("enable_fp16", true)),
  relaxed_dimension_check_(declare_parameter<bool>("relaxed_dimension_check", true)),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 1)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 1))
{
  RCLCPP_DEBUG(get_logger(), "[TensorRTNode] In TensorRTNode's constructor");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("TensorRTNode");

  if (engine_file_path_.empty()) {
    throw std::invalid_argument(
            "[TensorRTNode] Empty engine_file_path_, "
            "this needs to be set per the engine");
  }

  if (input_tensor_names_.empty()) {
    throw std::invalid_argument("[TensorRTNode] Empty input_tensor_names");
  }

  if (input_binding_names_.empty()) {
    throw std::invalid_argument("[TensorRTNode] Empty input_binding_names");
  }

  if (output_tensor_names_.empty()) {
    throw std::invalid_argument("[TensorRTNode] Empty output_tensor_names");
  }

  if (output_binding_names_.empty()) {
    throw std::invalid_argument("[TensorRTNode] Empty output_binding_names");
  }
  if (input_tensor_names_.size() != input_binding_names_.size()) {
    throw std::invalid_argument(
            "[TensorRTNode] input_tensor_names and input_binding_names must have equal sizes");
  }
  if (output_tensor_names_.size() != output_binding_names_.size()) {
    throw std::invalid_argument(
            "[TensorRTNode] output_tensor_names and output_binding_names must have equal sizes");
  }

  if (!custom_plugin_lib_.empty()) {
    if (!dlopen(custom_plugin_lib_.c_str(), RTLD_NOW)) {
      const char * error = dlerror();
      throw std::invalid_argument(
              "[TensorRTNode] Preload plugins failed: " +
              std::string(error ? error : "Unknown error"));
    }
    RCLCPP_INFO(
      get_logger(),
      "[TensorRTNode] TRT plugins: \"%s\" loaded successfully",
      custom_plugin_lib_.c_str());
  }

  tensor_rt_logger.setReportableSeverity(nvinfer1::ILogger::Severity::kINFO);
  // Initialize TensorRT engine (populates output_binding_infos_).
  InitializeTensorRTEngine();

  // Create subscribers for input and output tensors
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  input_sub_ = create_subscription<TensorList>(
    INPUT_TOPIC_NAME, input_qos,
    std::bind(&TensorRTNode::InputTensorCallback, this, std::placeholders::_1),
    sub_options);
  output_pub_ = create_publisher<TensorList>(
    OUTPUT_TOPIC_NAME, output_qos, pub_options);

  RCLCPP_INFO(get_logger(), "[TensorRTNode] TensorRT Node initialized successfully");
}

void TensorRTNode::InitializeTensorRTEngine()
{
  RCLCPP_INFO(get_logger(), "Initializing TensorRT engine...");

  // Initialize TensorRT plugins
  if (!initLibNvInferPlugins(&tensor_rt_logger, "")) {
    throw std::runtime_error("[TensorRTNode] Failed to initialize TensorRT plugins");
  }

  // Create TensorRT runtime
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(tensor_rt_logger));
  if (!runtime_) {
    throw std::runtime_error("[TensorRTNode] Failed to create TensorRT runtime");
  }

  // Try to load existing engine or build from model
  if (std::filesystem::exists(engine_file_path_) && !force_engine_update_) {
    LoadEngineFromFile();
  } else {
    BuildEngineFromModel();
  }
  if (!cuda_engine_) {
    throw std::runtime_error("[TensorRTNode] Failed to create TensorRT engine");
  }

  // Create execution context
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(cuda_engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("[TensorRTNode] Failed to create TensorRT execution context");
  }

  // Initialize binding information
  SetupBindings();

  // Binding Information
  RCLCPP_INFO(get_logger(), "Number of CUDA bindings: %d", cuda_engine_->getNbIOTensors());
  for (int32_t i = 0; i < cuda_engine_->getNbIOTensors(); ++i) {
    RCLCPP_INFO(
      get_logger(), "Tensor name %s: Format %s",
      cuda_engine_->getIOTensorName(i),
      cuda_engine_->getTensorFormatDesc(cuda_engine_->getIOTensorName(i)));
  }

  RCLCPP_INFO(get_logger(), "TensorRT engine initialized successfully");
}

void TensorRTNode::InputTensorCallback(
  const TensorList::SharedPtr tensor_list)
{
  RCLCPP_DEBUG(get_logger(), "Received input tensor list");
  try {
    // Perform inference
    // Publish result
    output_pub_->publish(DoInference(*tensor_list));
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Error during inference: %s", e.what());
  }
}

TensorList TensorRTNode::DoInference(
  const TensorList & input_tensor_list)
{
  // Hold read handles alive past enqueueV3 so the read-done events are recorded
  // AFTER TRT finishes reading, preventing upstream from recycling input buffers early.
  std::vector<TrtConv::TensorBinding<cuda_buffer_backend::ReadHandle>> input_bindings;
  input_bindings.reserve(input_binding_names_.size());

  for (size_t i = 0; i < input_binding_names_.size(); ++i) {
    const std::string & binding_name = input_binding_names_[i];
    const std::string & tensor_name = input_tensor_names_[i];

    const TrtConv::Tensor * input_tensor =
      isaac_ros_tensor_msgs::FindTensorByName(input_tensor_list, tensor_name);
    if (!input_tensor) {
      throw std::runtime_error(
              "[TensorRTNode] Input tensor not found: " + tensor_name);
    }

    const auto binding_dims_it = input_binding_dims_.find(binding_name);
    if (binding_dims_it == input_binding_dims_.end()) {
      throw std::runtime_error(
              "[TensorRTNode] Input binding dimensions not found: " + binding_name);
    }

    const nvinfer1::Dims dims = ResolveInputDims(
      *input_tensor, tensor_name, binding_dims_it->second);

    auto handle = cuda_buffer_backend::from_input_buffer(input_tensor->data, *cuda_stream_);
    const nvinfer1::DataType dtype = TrtConv::to_trt_data_type(
      input_tensor->dtype_code, input_tensor->dtype_bits, input_tensor->dtype_lanes);
    const size_t size_bytes = TrtConv::num_elements(dims) *
      TrtConv::bytes_per_element(input_tensor->dtype_bits, input_tensor->dtype_lanes);

    TrtConv::TensorBinding<cuda_buffer_backend::ReadHandle> binding(
      std::move(handle), binding_name, dims, dtype, size_bytes, input_tensor->byte_offset);
    binding.as_input(*context_, binding_name.c_str());
    input_bindings.push_back(std::move(binding));
  }

  // Hold write handles alive past enqueueV3 so the write-done events are recorded
  // AFTER TRT finishes writing, preventing downstream from reading stale data.
  std::vector<std::string> output_names;
  std::vector<TrtConv::Tensor> output_tensors;
  std::vector<TrtConv::TensorBinding<cuda_buffer_backend::WriteHandle>> output_bindings;
  output_names.reserve(output_binding_names_.size());
  output_tensors.reserve(output_binding_names_.size());
  output_bindings.reserve(output_binding_names_.size());

  for (size_t i = 0; i < output_binding_names_.size(); ++i) {
    const std::string & binding_name = output_binding_names_[i];
    const std::string & tensor_name = output_tensor_names_[i];

    const auto tensor_dims = context_->getTensorShape(binding_name.c_str());
    const auto trt_dtype = cuda_engine_->getTensorDataType(binding_name.c_str());
    const TrtConv::DLDataType dlpack_dtype = TrtConv::from_trt_data_type(trt_dtype);

    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(tensor_dims.nbDims));
    for (int j = 0; j < tensor_dims.nbDims; ++j) {
      shape.push_back(tensor_dims.d[j]);
    }

    TrtConv::Tensor output_tensor = TrtConv::allocate_tensor(
      shape, dlpack_dtype.code, dlpack_dtype.bits, dlpack_dtype.lanes);
    auto binding = TrtConv::from_output_tensor(binding_name, output_tensor, *cuda_stream_);
    binding.as_output(*context_, binding_name.c_str());
    output_bindings.push_back(std::move(binding));
    output_names.push_back(tensor_name);
    output_tensors.push_back(std::move(output_tensor));
  }

  if (!context_->enqueueV3(*cuda_stream_)) {
    throw std::runtime_error("[TensorRTNode] TensorRT inference failed");
  }

  // Handles go out of scope here, recording events AFTER enqueueV3 is submitted.
  // This ensures correct event ordering for the buffer synchronization protocol.
  output_bindings.clear();
  input_bindings.clear();

  TensorList output_tensor_list;
  output_tensor_list.header = input_tensor_list.header;
  output_tensor_list.names = std::move(output_names);
  output_tensor_list.tensors = std::move(output_tensors);
  return output_tensor_list;
}

void TensorRTNode::LoadEngineFromFile()
{
  std::ifstream file(engine_file_path_, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("[TensorRTNode] Cannot open engine file: " + engine_file_path_);
  }

  const size_t size = file.tellg();
  file.seekg(0);

  std::vector<char> engine_data(size);
  if (!file.read(engine_data.data(), size)) {
    throw std::runtime_error("[TensorRTNode] Failed to read engine file");
  }

  cuda_engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  RCLCPP_INFO(
    get_logger(), "Loaded TensorRT engine from file: %s successfully",
    engine_file_path_.c_str());
}

void TensorRTNode::BuildEngineFromModel()
{
  RCLCPP_INFO(get_logger(), "Building TensorRT engine from model: %s", model_file_path_.c_str());

  if (!std::filesystem::exists(model_file_path_)) {
    throw std::runtime_error("[TensorRTNode] Model file does not exist: " + model_file_path_);
  }

  // Create builder
  std::unique_ptr<nvinfer1::IBuilder> builder(
    nvinfer1::createInferBuilder(tensor_rt_logger));
  if (!builder) {
    throw std::runtime_error("[TensorRTNode] Failed to create TensorRT builder");
  }

  // Create builder config
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) {
    throw std::runtime_error("[TensorRTNode] Failed to create TensorRT builder config");
  }
  config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
  // Set max workspace size
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);

  // Enable FP16 if requested
  if (enable_fp16_) {
    // kFP16 flag has been deprecated starting with TensorRT 10.12
    // If your hardware supports TF32 (TensorFloat32), you can enable it with
    // builder.setFlag(nvinfer1::BuilderFlag::kTF32).
    // TF32 offers near-FP32 precision with FP16-like performance on compatible GPUs.
    // check the support matrix for more details:
    // https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
    config->setFlag(nvinfer1::BuilderFlag::kTF32);
    RCLCPP_INFO(get_logger(), "[TensorRTNode] FP16 mode enabled");
  }

  // Set DLA core if specified
  if (dla_core_ != default_dla_core && builder->getNbDLACores() > 0) {
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setDLACore(dla_core_);
    RCLCPP_INFO(get_logger(), "[TensorRTNode] Using DLA core: %ld", dla_core_);
  }

  // Create network
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  if (!network) {
    throw std::runtime_error("[TensorRTNode] Failed to create TensorRT network");
  }

  // Parse ONNX model
  std::unique_ptr<nvonnxparser::IParser> parser(
    nvonnxparser::createParser(*network, tensor_rt_logger));
  if (!parser) {
    throw std::runtime_error("[TensorRTNode] Failed to create ONNX parser");
  }

  if (!parser->parseFromFile(
      model_file_path_.c_str(),
      static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)))
  {
    RCLCPP_ERROR(
      get_logger(), "[TensorRTNode] Failed to parse ONNX model: %s",
      model_file_path_.c_str());
    throw std::runtime_error("[TensorRTNode] Failed to parse ONNX model");
  }

  // Provides optimization profile for dynamic size input bindings
  nvinfer1::IOptimizationProfile * optimization_profile = builder->createOptimizationProfile();
  // Checks input dimensions and adds to optimization profile if needed
  const int number_inputs = network->getNbInputs();
  for (int i = 0; i < number_inputs; ++i) {
    auto * bind_tensor = network->getInput(i);
    const char * bind_name = bind_tensor->getName();
    nvinfer1::Dims dims = bind_tensor->getDimensions();

    // Validates binding info
    if (dims.nbDims <= 0) {
      throw std::runtime_error(
              "[TensorRTNode] Invalid input tensor dimensions for binding " +
              std::string(bind_name));
    }
    for (int j = 1; j < dims.nbDims; ++j) {
      if (dims.d[j] <= 0) {
        RCLCPP_ERROR(
          get_logger(),
          "Input binding %s requires dynamic size on dimension No.%d which is not supported",
          bind_tensor->getName(), j);
        throw std::runtime_error(
                "[TensorRTNode] Input binding " + std::string(bind_name) +
                " requires dynamic size on dimension No." + std::to_string(j) +
                " which is not supported");
      }
    }
    if (dims.d[0] == -1) {
      // Only case with first dynamic dimension is supported and assumed to be batch size.
      // Always optimizes for 1-batch.
      dims.d[0] = 1;
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMIN, dims);
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kOPT, dims);
      dims.d[0] = max_batch_size_;
      if (max_batch_size_ <= 0) {
        RCLCPP_ERROR(
          get_logger(),
          "[TensorRTNode] Maximum batch size %d is invalid. Uses 1 instead.", max_batch_size_);
        dims.d[0] = 1;
      }
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMAX, dims);
    }
  }
  config->addOptimizationProfile(optimization_profile);

  // Build engine
  std::unique_ptr<nvinfer1::IHostMemory> serialized_engine(builder->buildSerializedNetwork(
      *network, *config));
  if (!serialized_engine) {
    throw std::runtime_error("[TensorRTNode] Failed to build TensorRT engine");
  }
  if (serialized_engine->size() == 0 || serialized_engine->data() == nullptr) {
    throw std::runtime_error("[TensorRTNode] Fail to serialize TensorRT Engine.");
  }
  RCLCPP_INFO(
    get_logger(), "[TensorRTNode] Serialized engine size: %zu",
    static_cast<size_t>(serialized_engine->size()));

  // Deserialize engine to a file for future use
  cuda_engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));

  // Save engine to file
  std::ofstream engine_file(engine_file_path_, std::ios::binary);
  if (engine_file.good()) {
    engine_file.write(
      static_cast<const char *>(serialized_engine->data()),
      serialized_engine->size());
    RCLCPP_INFO(
      get_logger(), "[TensorRTNode] Saved TensorRT engine to: %s",
      engine_file_path_.c_str());
  }
  RCLCPP_INFO(get_logger(), "Input bindings setup completed");
}

void TensorRTNode::SetupBindings()
{
  const int num_bindings = cuda_engine_->getNbIOTensors();

  for (int i = 0; i < num_bindings; ++i) {
    const char * tensor_name = cuda_engine_->getIOTensorName(i);
    auto binding_dims = cuda_engine_->getTensorShape(tensor_name);
    auto binding_data_type = cuda_engine_->getTensorDataType(tensor_name);
    const bool binding_is_input = cuda_engine_->getTensorIOMode(tensor_name) ==
      nvinfer1::TensorIOMode::kINPUT;

    // Calculate binding size. Replace dynamic dims (-1) with max_batch_size_
    // (bounded below by 1) so this is a true upper bound on bytes written.
    const int64_t dynamic_dim_bound = std::max<int64_t>(max_batch_size_, 1);
    size_t binding_size = 1;
    for (int j = 0; j < binding_dims.nbDims; ++j) {
      const int64_t dim = binding_dims.d[j] > 0 ? binding_dims.d[j] : dynamic_dim_bound;
      binding_size *= static_cast<size_t>(dim);
    }

    // Get element size
    const size_t element_size = GetElementSizeFromDataType(binding_data_type);
    binding_size = binding_size * element_size;
    if (binding_is_input) {
      input_binding_dims_[tensor_name] = binding_dims;
    } else {
      output_binding_infos_[tensor_name] = binding_size;
    }

    RCLCPP_DEBUG(
      get_logger(), "[TensorRTNode] Binding %d: %s (%s) - "
      "dims: %d of type - size: %zu bytes (total: %zu bytes)",
      i, tensor_name, binding_is_input ? "input" : "output", binding_dims.nbDims,
      binding_size, binding_size);
  }
}

TensorRTNode::~TensorRTNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::TensorRTNode)
