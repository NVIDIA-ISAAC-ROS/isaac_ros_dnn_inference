// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <algorithm>

#include "NvInferPluginUtils.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros/types/nitros_type_manager.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_shape.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "tensor_pub";

constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nhwc_rgb_f32";
constexpr char OUTPUT_TOPIC_NAME[] = "tensor_sub";

namespace
{
constexpr int64_t default_max_workspace_size = 67108864l;
constexpr int64_t default_dla_core = -1;

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
  Severity log_level = Severity::kINFO;  // Default to INFO level;
};
TensorRT_Logger tensor_rt_logger;


size_t GetElementSizeFromDataType(
  nvinfer1::DataType data_type)
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

nvidia::isaac_ros::nitros::NitrosDataType GetNitrosDataTypeFromInferDataType(
  nvinfer1::DataType infer_data_type)
{
  nvidia::isaac_ros::nitros::NitrosDataType nitros_data_type;
  switch (infer_data_type) {
    case nvinfer1::DataType::kFLOAT:
      nitros_data_type = nvidia::isaac_ros::nitros::NitrosDataType::kFloat32;
      break;
    case nvinfer1::DataType::kINT8:
      nitros_data_type = nvidia::isaac_ros::nitros::NitrosDataType::kInt8;
      break;
    case nvinfer1::DataType::kINT32:
      nitros_data_type = nvidia::isaac_ros::nitros::NitrosDataType::kInt32;
      break;
    case nvinfer1::DataType::kINT64:
      nitros_data_type = nvidia::isaac_ros::nitros::NitrosDataType::kInt64;
      break;
    case nvinfer1::DataType::kUINT8:
      nitros_data_type = nvidia::isaac_ros::nitros::NitrosDataType::kUnsigned8;
      break;
    case nvinfer1::DataType::kHALF:
    default:
      throw std::runtime_error("[TensorRTNode] Unsupported tensor data type");
  }

  return nitros_data_type;
}

}  // namespace

TensorRTNode::TensorRTNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("tensor_rt_node", options),
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
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 1)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 1))
{
  RCLCPP_DEBUG(get_logger(), "[TensorRTNode] In TensorRTNode's constructor");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("TensorRTNode");

  // Determine input and output formats
  std::string input_format = INPUT_DEFAULT_TENSOR_FORMAT;
  std::string output_format = OUTPUT_DEFAULT_TENSOR_FORMAT;

  if (!input_tensor_formats_.empty()) {
    input_format = input_tensor_formats_[0];
    RCLCPP_INFO(get_logger(), "[TensorRTNode] Set input data format to: \"%s\"",
            input_format.c_str());
  }

  if (!output_tensor_formats_.empty()) {
    output_format = output_tensor_formats_[0];
  }

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

  // Note: Input and output formats are now handled during Managed NITROS initialization
  if (!custom_plugin_lib_.empty()) {
    if (!dlopen(custom_plugin_lib_.c_str(), RTLD_NOW)) {
      const char * error = dlerror();
      throw std::invalid_argument(
        "[TensorRTNode] Preload plugins failed: " + std::string(error ? error : "Unknown error"));
    }
    RCLCPP_INFO(
      get_logger(),
      "[TensorRTNode] TRT plugins: \"%s\" loaded successfully",
      custom_plugin_lib_.c_str());
  }

  tensor_rt_logger.setReportableSeverity(nvinfer1::ILogger::Severity::kINFO);
  // Initialize TensorRT engine (populates output_binding_infos_).
  InitializeTensorRTEngine();

  // Size the pool block large enough for the largest output tensor the engine
  // can produce; falling back to memory_pool_block_size_ if that's already larger.
  size_t max_output_binding_size = 0;
  for (const auto & kv : output_binding_infos_) {
    max_output_binding_size = std::max(max_output_binding_size, kv.second);
  }
  const int64_t actual_block_size = std::max(
    memory_pool_block_size_, static_cast<int64_t>(max_output_binding_size));
  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(actual_block_size),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "[TensorRTNode] Failed to create memory pool");

  // Create subscribers for input and output tensors
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  input_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
    INPUT_TOPIC_NAME, input_qos,
    std::bind(&TensorRTNode::InputTensorCallback, this, std::placeholders::_1),
    sub_options);
  output_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
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
    RCLCPP_INFO(get_logger(), "Tensor name %s: Format %s",
                cuda_engine_->getIOTensorName(i),
                cuda_engine_->getTensorFormatDesc(cuda_engine_->getIOTensorName(i)));
  }

  RCLCPP_INFO(get_logger(), "TensorRT engine initialized successfully");
}

void TensorRTNode::InputTensorCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr tensor_list)
{
  RCLCPP_DEBUG(get_logger(), "Received input tensor list");
  try {
    // Perform inference
    auto output_tensor_list = DoInference(*tensor_list);

    // Publish result
    output_pub_->publish(output_tensor_list);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Error during inference: %s", e.what());
  }
}

nvidia::isaac_ros::nitros::NitrosTensorList TensorRTNode::DoInference(
  const nvidia::isaac_ros::nitros::NitrosTensorList & input_tensor_list)
{
  // Hold read handles alive past enqueueV3 so the read-done events are recorded
  // AFTER TRT finishes reading, preventing upstream from recycling input buffers early.
  std::vector<nvidia::isaac_ros::nitros::ReadHandle> input_read_handles;
  input_read_handles.reserve(input_binding_names_.size());

  for (size_t i = 0; i < input_binding_names_.size(); ++i) {
    const std::string & binding_name = input_binding_names_[i];
    const std::string & tensor_name = input_tensor_names_[i];

    auto input_tensor = input_tensor_list.get_tensor_by_name(tensor_name);
    if (!input_tensor) {
      throw std::runtime_error(
        "[TensorRTNode] Input tensor not found: " + tensor_name);
    }

    const auto input_shape = input_tensor->shape().dims();
    // TensorRT requires setInputShape() to use the engine binding rank. Some
    // upstream tensors are unbatched, for example CHW image tensors feeding an
    // explicit-batch NCHW engine, so use the cached binding dims as the shape
    // template instead of copying the incoming tensor metadata rank directly.
    const auto binding_dims_it = input_binding_dims_.find(binding_name);
    if (binding_dims_it == input_binding_dims_.end()) {
      throw std::runtime_error("[TensorRTNode] Input binding dimensions not found: " +
                               binding_name);
    }

    nvinfer1::Dims dims = binding_dims_it->second;
    const auto input_rank = static_cast<int32_t>(input_shape.size());
    const int32_t rank_delta = dims.nbDims - input_rank;
    // The input either matches the binding rank or omits only the leading batch
    // dimension. Any other rank difference is ambiguous and cannot be mapped
    // safely to TensorRT binding dimensions.
    if (rank_delta != 0 && rank_delta != 1) {
      throw std::runtime_error("[TensorRTNode] Input tensor rank does not match binding rank: " +
                               tensor_name);
    }

    // Dynamic batch is the only runtime dimension handled here. If the tensor
    // omitted batch, only a unit-batch static engine can be inferred safely.
    if (dims.d[0] < 0) {
      dims.d[0] = rank_delta == 0 ? input_shape[0] : 1;
    } else if (rank_delta == 1 && dims.d[0] != 1) {
      throw std::runtime_error("[TensorRTNode] Input tensor omits non-unit batch dimension: " +
                               tensor_name);
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
    if (!context_->setInputShape(binding_name.c_str(), dims)) {
      throw std::runtime_error("[TensorRTNode] Failed to update input binding dimensions: " +
                               binding_name);
    }

    auto read_handle = input_tensor->get_read_handle(*cuda_stream_);
    const uint8_t * buffer_ptr = read_handle.get_ptr();

    if (!context_->setTensorAddress(binding_name.c_str(),
        const_cast<void *>(static_cast<const void *>(buffer_ptr))))
    {
      throw std::runtime_error(
        "[TensorRTNode] Failed to set input tensor address for: " +
        binding_name);
    }
    input_read_handles.push_back(std::move(read_handle));
  }

  // Hold write handles alive past enqueueV3 so the write-done events are recorded
  // AFTER TRT finishes writing, preventing downstream from reading stale data.
  std::vector<nvidia::isaac_ros::nitros::NitrosTensor> output_tensors;
  std::vector<nvidia::isaac_ros::nitros::WriteHandle> output_write_handles;
  output_tensors.reserve(output_binding_names_.size());
  output_write_handles.reserve(output_binding_names_.size());

  for (size_t i = 0; i < output_binding_names_.size(); ++i) {
    const std::string & binding_name = output_binding_names_[i];

    auto tensor_dims = context_->getTensorShape(binding_name.c_str());
    std::vector<int32_t> shape_dims;
    for (int j = 0; j < tensor_dims.nbDims; ++j) {
      shape_dims.push_back(tensor_dims.d[j]);
    }
    nvidia::isaac_ros::nitros::NitrosTensorShape shape(shape_dims);
    auto tensor_data_type = cuda_engine_->getTensorDataType(binding_name.c_str());
    auto nitros_data_type = GetNitrosDataTypeFromInferDataType(tensor_data_type);
    nvidia::isaac_ros::nitros::NitrosTensor tensor;
    auto write_handle = tensor.from_pool(
      binding_name, pool_, shape, nitros_data_type, *cuda_stream_);

    tensor.set_name(binding_name);
    if (!context_->setTensorAddress(binding_name.c_str(), write_handle.get_ptr())) {
      throw std::runtime_error("[TensorRTNode] Failed to set output tensor address for: " +
                               binding_name);
    }
    output_write_handles.push_back(std::move(write_handle));
    output_tensors.push_back(std::move(tensor));
  }

  if (!context_->enqueueV3(*cuda_stream_)) {
    throw std::runtime_error("[TensorRTNode] TensorRT inference failed");
  }

  // Handles go out of scope here, recording events AFTER enqueueV3 is submitted.
  // This ensures correct event ordering for the buffer synchronization protocol.
  output_write_handles.clear();
  input_read_handles.clear();

  nvidia::isaac_ros::nitros::NitrosTensorList output_tensor_list;
  output_tensor_list.set_timestamp_sec(input_tensor_list.get_timestamp_sec());
  output_tensor_list.set_timestamp_nsec(input_tensor_list.get_timestamp_nsec());
  output_tensor_list.set_frame_id(input_tensor_list.get_frame_id());

  for (size_t i = 0; i < output_tensors.size(); ++i) {
    output_tensors[i].set_name(output_tensor_names_[i]);
    output_tensor_list.add_tensor(std::move(output_tensors[i]));
  }

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
  RCLCPP_INFO(get_logger(), "Loaded TensorRT engine from file: %s successfully",
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
    RCLCPP_ERROR(get_logger(), "[TensorRTNode] Failed to parse ONNX model: %s",
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
      throw std::runtime_error("[TensorRTNode] Invalid input tensor dimensions for binding " +
                               std::string(bind_name));
    }
    for (int j = 1; j < dims.nbDims; ++j) {
      if (dims.d[j] <= 0) {
        RCLCPP_ERROR(get_logger(),
            "Input binding %s requires dynamic size on dimension No.%d which is not supported",
            bind_tensor->getName(), j);
        throw std::runtime_error("[TensorRTNode] Input binding " + std::string(bind_name) +
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
        RCLCPP_ERROR(get_logger(),
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
  RCLCPP_INFO(get_logger(), "[TensorRTNode] Serialized engine size: %zu",
    static_cast<size_t>(serialized_engine->size()));

  // Deserialize engine to a file for future use
  cuda_engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));

  // Save engine to file
  std::ofstream engine_file(engine_file_path_, std::ios::binary);
  if (engine_file.good()) {
    engine_file.write(static_cast<const char *>(serialized_engine->data()),
            serialized_engine->size());
    RCLCPP_INFO(get_logger(), "[TensorRTNode] Saved TensorRT engine to: %s",
            engine_file_path_.c_str());
  }
  RCLCPP_INFO(get_logger(), "Input bindings setup completed");
}

void TensorRTNode::SetupBindings()
{
  const int num_bindings = cuda_engine_->getNbIOTensors();
  bool binding_is_input = false;

  for (int i = 0; i < num_bindings; ++i) {
    const char * tensor_name = cuda_engine_->getIOTensorName(i);
    auto binding_dims = cuda_engine_->getTensorShape(tensor_name);
    auto binding_data_type = cuda_engine_->getTensorDataType(tensor_name);
    binding_is_input = cuda_engine_->getTensorIOMode(tensor_name) ==
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
    size_t element_size = GetElementSizeFromDataType(binding_data_type);
    binding_size = binding_size * element_size;
    if (binding_is_input) {
      input_binding_dims_[tensor_name] = binding_dims;
    } else {
      output_binding_infos_[tensor_name] = binding_size;
    }

    RCLCPP_DEBUG(get_logger(), "[TensorRTNode] Binding %d: %s (%s) - "
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
