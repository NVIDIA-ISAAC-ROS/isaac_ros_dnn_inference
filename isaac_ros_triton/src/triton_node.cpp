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

#include "isaac_ros_triton/triton_node.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace Nitros = nvidia::isaac_ros::nitros;

static constexpr char INPUT_TOPIC_NAME[] = "tensor_pub";
static constexpr char OUTPUT_TOPIC_NAME[] = "tensor_sub";
static constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
static constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nhwc_rgb_f32";
// Triton server readiness poll attempts during server initialization/shutdown/restart
static constexpr int MAX_READINESS_POLL_ATTEMPTS = 2000;
// Triton server readiness poll interval during server initialization/shutdown/restart
static constexpr auto READINESS_POLL_INTERVAL = std::chrono::milliseconds(5);

namespace
{

void InferenceComplete(
  TRITONSERVER_InferenceResponse * response,
  const uint32_t flags,
  void * userp)
{
  if (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
    // Retrieve the std::promise object passed as userp
    TRITONSERVER_Error * err = TRITONSERVER_InferenceResponseError(response);
    if (err != nullptr) {
      const char * error_str = TRITONSERVER_ErrorMessage(err);
      std::cerr << "Inference error: " << (error_str ? error_str : "Unknown error") << std::endl;
      TRITONSERVER_ErrorDelete(err);
      return;
    }

    std::promise<TRITONSERVER_InferenceResponse *> * response_promise =
      reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse *> *>(userp);
    if (response_promise != nullptr) {
      response_promise->set_value(response);
    }
  }
}

}  // namespace

TritonNode::TritonNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("triton_node", options),
  model_name_(declare_parameter<std::string>("model_name", "")),
  max_batch_size_((uint32_t)declare_parameter<uint16_t>("max_batch_size", 8)),
  num_concurrent_requests_((uint32_t)declare_parameter<uint16_t>("num_concurrent_requests", 10)),
  model_repository_paths_(declare_parameter<StringList>("model_repository_paths", StringList())),
  enable_triton_logging_(declare_parameter<bool>("enable_triton_logging", true)),
  enable_strict_model_(declare_parameter<bool>("enable_strict_model", false)),
  input_tensor_names_(declare_parameter<StringList>("input_tensor_names", StringList())),
  input_binding_names_(declare_parameter<StringList>("input_binding_names", StringList())),
  input_tensor_formats_(declare_parameter<StringList>("input_tensor_formats", StringList())),
  output_tensor_names_(declare_parameter<StringList>("output_tensor_names", StringList())),
  output_binding_names_(declare_parameter<StringList>("output_binding_names", StringList())),
  output_tensor_formats_(declare_parameter<StringList>("output_tensor_formats", StringList())),
  log_level_(declare_parameter<int>("log_level", 0))
{
  RCLCPP_DEBUG(get_logger(), "[TritonNode] Constructing Triton inference server wrapper");

  if (model_name_.empty()) {
    throw std::invalid_argument("[TritonNode] Empty model_name");
  }
  if (model_repository_paths_.empty()) {
    throw std::invalid_argument("[TritonNode] Empty model_repository_paths");
  }
  if (input_tensor_names_.empty()) {
    throw std::invalid_argument("[TritonNode] Empty input_tensor_names");
  }
  if (input_binding_names_.empty()) {
    throw std::invalid_argument("[TritonNode] Empty input_binding_names");
  }
  if (output_tensor_names_.empty()) {
    throw std::invalid_argument("[TritonNode] Empty output_tensor_names");
  }
  if (output_binding_names_.empty()) {
    throw std::invalid_argument("[TritonNode] Empty output_binding_names");
  }
  if (log_level_ < 0 || log_level_ > 3) {
    throw std::invalid_argument("[TritonNode] Invalid triton_logging_level");
  }

  // Validate tensor configuration consistency
  if (input_tensor_names_.size() != input_binding_names_.size()) {
    RCLCPP_ERROR(get_logger(),
      "[TritonNode] Input tensor names (%zu) and binding names (%zu) count mismatch",
      input_tensor_names_.size(), input_binding_names_.size());
    throw std::invalid_argument(
      "[TritonNode] Input tensor names and binding names count mismatch");
  }

  if (output_tensor_names_.size() != output_binding_names_.size()) {
    RCLCPP_ERROR(get_logger(),
      "[TritonNode] Output tensor names (%zu) and binding names (%zu) count mismatch",
      output_tensor_names_.size(), output_binding_names_.size());
    throw std::invalid_argument("Output tensor names and binding names count mismatch");
  }

  input_qos_ = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos");
  output_qos_ = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos");

  input_format_ = INPUT_DEFAULT_TENSOR_FORMAT;
  output_format_ = OUTPUT_DEFAULT_TENSOR_FORMAT;
  if (!input_tensor_formats_.empty()) {
    input_format_ = input_tensor_formats_[0];
    RCLCPP_INFO(get_logger(), "[TritonNode] Set input data format to: '%s'", input_format_.c_str());
  }
  if (!output_tensor_formats_.empty()) {
    output_format_ = output_tensor_formats_[0];
    RCLCPP_INFO(
      get_logger(), "[TritonNode] Set output data format to: '%s'",
      output_format_.c_str());
  }

  nitros_pub_ = std::make_shared<Nitros::ManagedNitrosPublisher<Nitros::NitrosTensorList>>(
    this,
    OUTPUT_TOPIC_NAME,
    output_format_,
    Nitros::NitrosDiagnosticsConfig{},
    output_qos_);

  nitros_sub_ = std::make_shared<Nitros::ManagedNitrosSubscriber<Nitros::NitrosTensorListView>>(
    this,
    INPUT_TOPIC_NAME,
    input_format_,
    std::bind(&TritonNode::InputCallback, this, std::placeholders::_1),
    Nitros::NitrosDiagnosticsConfig{},
    input_qos_);

  RCLCPP_INFO(get_logger(), "[TritonNode] Managed NITROS pub/sub initialized");

  // Initialize CUDA stream
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      cuda_stream_, "isaac_ros_triton_node"),
    "Error initializing CUDA stream");

  // Initialize Triton server - must succeed
  if (!InitializeTritonServer()) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Triton server initialization failed");
    throw std::runtime_error("[TritonNode] Failed to initialize Triton server");
  }

  // Initialize input and output bindings map
  if (!InitializeBindingsMap()) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to initialize input and output bindings map");
    throw std::runtime_error("[TritonNode] Failed to initialize input and output bindings map");
  }
  RCLCPP_INFO(get_logger(), "[TritonNode] Triton node ready with model: %s", model_name_.c_str());
}

void TritonNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & view)
{
  try {
    // Build output NitrosTensorList
    std_msgs::msg::Header header;
    header.stamp.sec = view.GetTimestampSeconds();
    header.stamp.nanosec = view.GetTimestampNanoseconds();
    header.frame_id = view.GetFrameId();

    // Perform Triton inference - no fallback
    if (!triton_server_ready_) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Triton server not ready, dropping input");
      return;
    }

    if (!DoTritonInference(view, header)) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Triton inference failed, dropping input");
      return;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Error in InputCallback: %s", e.what());
  }
}

bool TritonNode::InitializeBindingsMap()
{
  for (size_t i = 0; i < input_tensor_names_.size(); ++i) {
    input_bindings_map_[input_tensor_names_[i]] = input_binding_names_[i];
  }

  for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
    output_bindings_map_[output_tensor_names_[i]] = output_binding_names_[i];
    // output_bindings_map_[output_binding_names_[i]] = output_tensor_names_[i];
  }

  RCLCPP_INFO(get_logger(), "[TritonNode] Input and output bindings map initialized successfully");
  return true;
}

bool TritonNode::InitializeTritonServer()
{
  std::lock_guard<std::mutex> lock(triton_mutex_);

  try {
    RCLCPP_INFO(get_logger(), "[TritonNode] Initializing Triton server for model: %s",
                model_name_.c_str());

    struct ServerOptionsDeleter
    {
      void operator()(TRITONSERVER_ServerOptions * opts) const
      {
        if (opts) {TRITONSERVER_ServerOptionsDelete(opts);}
      }
    };

    // Initialize Triton server options
    TRITONSERVER_ServerOptions * triton_server_options = nullptr;
    TRITONSERVER_Error * err = TRITONSERVER_ServerOptionsNew(
      reinterpret_cast<TRITONSERVER_ServerOptions **>(&triton_server_options));
    if (err != nullptr) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to create server options: %s",
                   TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }
    std::unique_ptr<TRITONSERVER_ServerOptions, ServerOptionsDeleter> options_guard(
      reinterpret_cast<TRITONSERVER_ServerOptions *>(triton_server_options),
      ServerOptionsDeleter());

    // Set model repository paths
    auto server_options = reinterpret_cast<TRITONSERVER_ServerOptions *>(triton_server_options);
    for (const auto & repo_path : model_repository_paths_) {
      if (!std::filesystem::exists(repo_path)) {
        RCLCPP_ERROR(get_logger(), "[TritonNode] Model repository path does not exist: %s",
                    repo_path.c_str());
        return false;
      }
      err = TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, repo_path.c_str());
      if (err != nullptr) {
        RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to set repository path: %s",
                     TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        return false;
      }
      RCLCPP_INFO(get_logger(), "[TritonNode] Added model repository: %s", repo_path.c_str());
    }

    // Set server options
    // Triton Logging Level: 0 = Error, 1 = Warn, 2 = Info, 3+ = Verbose
    TRITONSERVER_ServerOptionsSetLogVerbose(server_options, log_level_);
    TRITONSERVER_ServerOptionsSetLogInfo(server_options, enable_triton_logging_);
    TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, enable_strict_model_);
    TRITONSERVER_ServerOptionsSetServerId(server_options, "isaac_ros_triton_server");

    // Create Triton server
    TRITONSERVER_Server * server_ptr = nullptr;
    err = TRITONSERVER_ServerNew(&server_ptr, server_options);
    if (err != nullptr) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to create Triton server: %s",
                   TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    RCLCPP_INFO(get_logger(), "[TritonNode] Triton server created successfully");

    // Store server pointer with custom deleter
    triton_server_ = std::unique_ptr<void, void(*)(void *)>(
      server_ptr,
      [](void * ptr) {
        if (ptr) {
          TRITONSERVER_ServerDelete(reinterpret_cast<TRITONSERVER_Server *>(ptr));
        }
      }
    );
    options_guard.release();

    // Wait for server to be ready
    auto server = reinterpret_cast<TRITONSERVER_Server *>(triton_server_.get());
    bool server_live = false;
    bool server_ready = false;

    for (int i = 0; i < MAX_READINESS_POLL_ATTEMPTS; ++i) {
      err = TRITONSERVER_ServerIsLive(server, &server_live);
      if (err == nullptr && server_live) {
        err = TRITONSERVER_ServerIsReady(server, &server_ready);
        if (err == nullptr && server_ready) {
          break;
        }
      }
      if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
        err = nullptr;
      }
      std::this_thread::sleep_for(READINESS_POLL_INTERVAL);
    }

    if (!server_live || !server_ready) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Triton server failed to become ready");
      return false;
    }

    RCLCPP_INFO(get_logger(), "[TritonNode] Triton server is live and ready");

    // Log configuration
    RCLCPP_INFO(get_logger(), "[TritonNode] Triton server configuration:");
    RCLCPP_INFO(get_logger(), "  Model name: %s", model_name_.c_str());
    RCLCPP_INFO(get_logger(), "  Max batch size: %u", max_batch_size_);
    RCLCPP_INFO(get_logger(), "  Concurrent requests: %u", num_concurrent_requests_);

    RCLCPP_INFO(get_logger(), "  Input tensors (%zu):", input_tensor_names_.size());
    for (size_t i = 0; i < input_tensor_names_.size(); ++i) {
      RCLCPP_INFO(get_logger(), "    %s -> %s",
                 input_tensor_names_[i].c_str(), input_binding_names_[i].c_str());
    }

    RCLCPP_INFO(get_logger(), "  Output tensors (%zu):", output_tensor_names_.size());
    for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
      RCLCPP_INFO(get_logger(), "    %s -> %s",
                 output_binding_names_[i].c_str(), output_tensor_names_[i].c_str());
    }

    triton_server_ready_ = true;
    RCLCPP_INFO(get_logger(), "[TritonNode] Triton server initialized successfully");
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Exception in InitializeTritonServer: %s", e.what());
    return false;
  }
}

void TritonNode::ShutdownTritonServer()
{
  std::lock_guard<std::mutex> lock(triton_mutex_);

  if (triton_server_) {
    // Call TRITONSERVER_ServerStop to initiate server shutdown
    auto server = reinterpret_cast<TRITONSERVER_Server *>(triton_server_.get());
    TRITONSERVER_Error * error = TRITONSERVER_ServerStop(server);
    if (error != nullptr) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Error stopping Triton server: %s",
                   TRITONSERVER_ErrorMessage(error));
      TRITONSERVER_ErrorDelete(error);
    } else {
      RCLCPP_INFO(get_logger(), "[TritonNode] Triton server shutdown initiated successfully");
    }
    // Wait for server to stop
    bool server_live = true;
    for (int i = 0; i < MAX_READINESS_POLL_ATTEMPTS && server_live; ++i) {
      error = TRITONSERVER_ServerIsLive(server, &server_live);
      if (error != nullptr) {
        TRITONSERVER_ErrorDelete(error);
        break;
      }
      if (server_live) {
        std::this_thread::sleep_for(READINESS_POLL_INTERVAL);
      }
    }

    // The unique_ptr's custom deleter will handle TRITONSERVER_ServerDelete
    triton_server_.reset();
  }

  triton_server_ready_ = false;
  RCLCPP_INFO(get_logger(), "[TritonNode] Triton server shutdown complete");
}

bool TritonNode::DoTritonInference(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & view,
  const std_msgs::msg::Header & header)
{
  std::lock_guard<std::mutex> lock(triton_mutex_);

  if (!triton_server_ready_) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Triton server not ready");
    return false;
  }

  try {
    // Validate input tensors
    const auto tensors = view.GetAllTensor();
    if (tensors.size() != input_tensor_names_.size()) {
      RCLCPP_ERROR(get_logger(),
        "[TritonNode] Expected %zu input tensors, got %zu",
        input_tensor_names_.size(), tensors.size());
      return false;
    }

    // Build output tensor list
    Nitros::NitrosTensorListBuilder list_builder;
    list_builder.WithHeader(header);

    // Execute inference using Triton C API
    if (!ExecuteInference(view, list_builder)) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Triton inference execution failed");
      return false;
    }

    nitros_pub_->publish(list_builder.Build());
    RCLCPP_DEBUG(get_logger(), "[TritonNode] Triton inference completed successfully");
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Exception in DoTritonInference: %s", e.what());
    return false;
  }
}

void InferRequestRelease(
  TRITONSERVER_InferenceRequest * request, const uint32_t flags, void * userp)
{
  (void)flags;
  (void)request;
  std::promise<void> * barrier = reinterpret_cast<std::promise<void> *>(userp);
  barrier->set_value();
}

// Minimal allocator function (required by the API)
TRITONSERVER_Error * ResponseAlloc(
  TRITONSERVER_ResponseAllocator * allocator,
  const char * tensor_name,
  size_t byte_size,
  TRITONSERVER_MemoryType preferred_memory_type,
  int64_t preferred_memory_type_id,
  void * userp,
  void ** buffer,
  void ** buffer_userp,
  TRITONSERVER_MemoryType * actual_memory_type,
  int64_t * actual_memory_type_id)
{
  (void)allocator;
  (void)tensor_name;
  (void)userp;

  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;
  if (byte_size == 0) {
    *buffer = nullptr;
    return nullptr;
  }
  cudaStream_t cuda_stream = *reinterpret_cast<cudaStream_t *>(buffer_userp);
  void * allocated_buffer = nullptr;
  if (preferred_memory_type == TRITONSERVER_MEMORY_GPU) {
    auto err = cudaSetDevice(*actual_memory_type_id);
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "failed to set device");
    }
    err = cudaMallocAsync(&allocated_buffer, byte_size, cuda_stream);
    if (err != cudaSuccess) {
      std::string error_message = "failed to allocate " + std::to_string(byte_size) +
        " bytes of memory";
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error_message.c_str());
    }
    err = cudaStreamSynchronize(cuda_stream);
    if (err != cudaSuccess) {
      std::string error_message = "failed to synchronize CUDA stream";
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error_message.c_str());
    }
  } else if (preferred_memory_type == TRITONSERVER_MEMORY_CPU) {
    allocated_buffer = malloc(byte_size);
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    *actual_memory_type_id = preferred_memory_type_id;
  } else if (preferred_memory_type == TRITONSERVER_MEMORY_CPU_PINNED) {
    cudaError_t err = cudaSetDevice(*actual_memory_type_id);
    if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
      (err != cudaErrorInsufficientDriver))
    {
      std::string error_message = "unable to recover current CUDA device: " +
        std::string(cudaGetErrorString(err));
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error_message.c_str());
    }
    err = cudaHostAlloc(&allocated_buffer, byte_size, cudaHostAllocPortable);
    if (err != cudaSuccess) {
      std::string error_message = "failed to allocate " + std::to_string(byte_size) +
        " bytes of memory";
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error_message.c_str());
    }
    *actual_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
    *actual_memory_type_id = preferred_memory_type_id;
  } else {
    std::string error_message = "not supported memory type: " +
      std::to_string(preferred_memory_type);
    std::cerr << "[TritonNode] " << error_message << std::endl;
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error_message.c_str());
  }

  if (allocated_buffer != nullptr) {
    *buffer = allocated_buffer;
  }
  return nullptr;
}

// Minimal release function (required by the API)
TRITONSERVER_Error * ResponseRelease(
  TRITONSERVER_ResponseAllocator * allocator,
  void * buffer,
  void * buffer_userp,
  size_t byte_size,
  TRITONSERVER_MemoryType memory_type,
  int64_t memory_type_id)
{
  (void)allocator;
  (void)byte_size;
  cudaError_t err = cudaSuccess;
  switch (memory_type) {
    case TRITONSERVER_MEMORY_GPU:
      err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFree(buffer);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "failed to free memory");
        }
      }
      break;
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFreeHost(buffer);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "failed to free memory");
        }
      }
      break;
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory" << std::endl;
      break;
  }
  if (buffer_userp != nullptr) {
    free(buffer_userp);
  }
  return nullptr;
}

inline Nitros::NitrosDataType TritonToNitrosType(TRITONSERVER_DataType t)
{
  switch (t) {
    case TRITONSERVER_TYPE_INT8: return Nitros::NitrosDataType::kInt8;
    case TRITONSERVER_TYPE_UINT8: return Nitros::NitrosDataType::kUnsigned8;
    case TRITONSERVER_TYPE_INT16: return Nitros::NitrosDataType::kInt16;
    case TRITONSERVER_TYPE_UINT16: return Nitros::NitrosDataType::kUnsigned16;
    case TRITONSERVER_TYPE_INT32: return Nitros::NitrosDataType::kInt32;
    case TRITONSERVER_TYPE_UINT32: return Nitros::NitrosDataType::kUnsigned32;
    case TRITONSERVER_TYPE_INT64: return Nitros::NitrosDataType::kInt64;
    case TRITONSERVER_TYPE_UINT64: return Nitros::NitrosDataType::kUnsigned64;
    case TRITONSERVER_TYPE_FP32: return Nitros::NitrosDataType::kFloat32;
    case TRITONSERVER_TYPE_FP64: return Nitros::NitrosDataType::kFloat64;
    default: return Nitros::NitrosDataType::kFloat32;
  }
}

bool TritonNode::ProcessInferenceResponse(
  TRITONSERVER_InferenceResponse * response,
  Nitros::NitrosTensorListBuilder & list_builder,
  std::unordered_map<std::string, std::string> & output_bindings_map,
  std::vector<std::string> output_tensor_names
)
{
  // Process response
  uint32_t output_count = 0;
  auto * err = TRITONSERVER_InferenceResponseOutputCount(response, &output_count);
  if (err != nullptr) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to get output count: %s",
      TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    TRITONSERVER_InferenceResponseDelete(response);
    return false;
  }
  typedef struct OutputInfo
  {
    const char * name;
    TRITONSERVER_DataType dtype;
    std::vector<int32_t> dims;
    uint64_t dims_count;
    uint64_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_id;
    const void * buffer;
    void * userp;
  } OutputInfo;
  std::vector<OutputInfo> output_infos;

  for (uint32_t i = 0; i < output_count; ++i) {
    OutputInfo output_info;
    const int64_t * shape;
    uint64_t dims_count;

    err = TRITONSERVER_InferenceResponseOutput(
      response, i, &output_info.name, &output_info.dtype, &shape,
      &dims_count, &output_info.buffer, &output_info.byte_size,
      &output_info.memory_type, &output_info.memory_id, &output_info.userp);
    if (err != nullptr) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to get output info: %s",
        TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }
    for (uint64_t d = 0; d < dims_count; ++d) {
      output_info.dims.push_back(static_cast<int32_t>(shape[d]));
    }

    output_infos.push_back(output_info);
  }

  // Build output tensors
  for (uint32_t i = 0; i < output_tensor_names.size(); ++i) {
    std::string tensor_name = output_tensor_names[i];
    // Use the corresponding output tensor name from configuration
    std::string binding_name = output_bindings_map[tensor_name];

    // find matching name from output_infos with binding_name
    auto output_info = std::find_if(output_infos.begin(),
                                    output_infos.end(),
        [&binding_name](const OutputInfo & info) {
          return info.name == binding_name;
    });

    if (output_info == output_infos.end()) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to find output info for tensor: %s",
        binding_name.c_str());
      return false;
    }

    // Copy data to GPU
    void * gpu_buffer = nullptr;
    cudaError_t cuda_err = cudaMallocAsync(&gpu_buffer, output_info->byte_size, cuda_stream_);
    if (cuda_err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] CUDA malloc failed: %s",
        cudaGetErrorString(cuda_err));
      return false;
    }

    if (output_info->memory_type == TRITONSERVER_MEMORY_GPU) {
      cuda_err = cudaMemcpyAsync(gpu_buffer, output_info->buffer, output_info->byte_size,
                            cudaMemcpyDeviceToDevice);
    } else {
      cuda_err = cudaMemcpyAsync(gpu_buffer, output_info->buffer, output_info->byte_size,
                            cudaMemcpyHostToDevice);
    }
    if (cuda_err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] CUDA memcpy failed: %s",
        cudaGetErrorString(cuda_err));
      cudaFree(gpu_buffer);
      return false;
    }

    cuda_err = cudaStreamSynchronize(cuda_stream_);
    if (cuda_err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] CUDA stream synchronization failed: %s",
        cudaGetErrorString(cuda_err));
      cudaFree(gpu_buffer);
      return false;
    }

    auto nitros_tensor = Nitros::NitrosTensorBuilder()
      .WithShape(Nitros::NitrosTensorShape(output_info->dims))
      .WithDataType(TritonToNitrosType(output_info->dtype))
      .WithData(gpu_buffer)
      .WithReleaseCallback([gpu_buffer]() {
          cudaFree(gpu_buffer);
        })
      .Build();
    list_builder.AddTensor(tensor_name, nitros_tensor);
  }
  return true;
}

bool TritonNode::ExecuteInference(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & view,
  nvidia::isaac_ros::nitros::NitrosTensorListBuilder & list_builder)
{
  auto server = reinterpret_cast<TRITONSERVER_Server *>(triton_server_.get());

  // Create inference request
  TRITONSERVER_InferenceRequest * request = nullptr;
  TRITONSERVER_Error * err = TRITONSERVER_InferenceRequestNew(
    &request, server, model_name_.c_str(), -1);
  if (err != nullptr) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to create inference request: %s",
                 TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    return false;
  }

  std::string request_id_str = model_name_ + "_" + std::to_string(request_id_++);
  TRITONSERVER_InferenceRequestSetId(request, request_id_str.c_str());
  struct RequestDeleter
  {
    void operator()(TRITONSERVER_InferenceRequest * req) const
    {
      if (req != nullptr) {
        TRITONSERVER_InferenceRequestDelete(req);
      }
    }
  };
  std::unique_ptr<TRITONSERVER_InferenceRequest, RequestDeleter> request_guard(request);

  try {
    // Add input tensors
    const auto tensors = view.GetAllTensor();
    for (size_t i = 0; i < tensors.size() && i < input_binding_names_.size(); ++i) {
      const auto & tensor = tensors[i];

      // Get tensor dimensions
      std::vector<int64_t> dims;
      for (size_t d = 0; d < tensor.GetRank(); ++d) {
        dims.push_back(static_cast<int64_t>(tensor.GetDimension(d)));
      }
      std::string input_binding_name = input_bindings_map_[tensor.GetName()];
      // Use FP32 as default data type (can be enhanced to detect from tensor metadata)
      TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_FP32;
      switch (static_cast<int>(tensor.GetElementType())) {
        case static_cast<int>(isaac_ros::nitros::NitrosDataType::kInt8):
          dtype = TRITONSERVER_TYPE_INT8;
          break;
        case static_cast<int>(isaac_ros::nitros::NitrosDataType::kUnsigned8):
          dtype = TRITONSERVER_TYPE_UINT8;
          break;
        case static_cast<int>(Nitros::NitrosDataType::kInt16):
          dtype = TRITONSERVER_TYPE_INT16;
          break;
        case static_cast<int>(Nitros::NitrosDataType::kUnsigned16):
          dtype = TRITONSERVER_TYPE_UINT16;
          break;
        case static_cast<int>(Nitros::NitrosDataType::kInt32):
          dtype = TRITONSERVER_TYPE_INT32;
          break;
        case static_cast<int>(Nitros::NitrosDataType::kUnsigned32):
          dtype = TRITONSERVER_TYPE_UINT32;
          break;
        case static_cast<int>(Nitros::NitrosDataType::kInt64):
          dtype = TRITONSERVER_TYPE_INT64;
          break;
        case static_cast<int>(Nitros::NitrosDataType::kUnsigned64):
          dtype = TRITONSERVER_TYPE_UINT64;
          break;
        default:
          dtype = TRITONSERVER_TYPE_FP32;
          break;
      }

      // Add input to request
      auto err = TRITONSERVER_InferenceRequestAddInput(
        request, input_binding_name.c_str(), dtype, dims.data(), dims.size());
      if (err != nullptr) {
        RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to add input to request: %s",
                     TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        return false;
      }

      // Add tensor data
      void * buffer = const_cast<void *>(reinterpret_cast<const void *>(tensor.GetBuffer()));
      err = TRITONSERVER_InferenceRequestAppendInputData(
        request, input_binding_name.c_str(), buffer, tensor.GetTensorSize(),
        TRITONSERVER_MEMORY_GPU, 0);
      if (err != nullptr) {
        RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to add tensor data to request: %s",
                     TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        return false;
      }
    }

    // Request output tensors
    for (const auto & output_name : output_binding_names_) {
      err = TRITONSERVER_InferenceRequestAddRequestedOutput(request, output_name.c_str());
      if (err != nullptr) {
        RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to add output to request: %s",
                     TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        return false;
      }
    }
    TRITONSERVER_ResponseAllocator * allocator = nullptr;
    TRITONSERVER_Error * err = TRITONSERVER_ResponseAllocatorNew(
        &allocator,
        ResponseAlloc,
        ResponseRelease,
        nullptr
    );
    struct ResponseAllocatorDeleter
    {
      void operator()(TRITONSERVER_ResponseAllocator * allocator) const
      {
        if (allocator != nullptr) {
          TRITONSERVER_ResponseAllocatorDelete(allocator);
        }
      }
    };
    std::unique_ptr<TRITONSERVER_ResponseAllocator,
      ResponseAllocatorDeleter> allocator_guard(allocator);

    // Execute inference asynchronously
    auto promise = std::make_unique<std::promise<TRITONSERVER_InferenceResponse *>>();
    std::future<TRITONSERVER_InferenceResponse *> completed = promise->get_future();
    // Register a new promise for the request callback barrier.
    auto release_promise = std::make_unique<std::promise<void>>();
    std::future<void> release_future = release_promise->get_future();
    err = TRITONSERVER_InferenceRequestSetReleaseCallback(
      request, InferRequestRelease,
      reinterpret_cast<void *>(release_promise.get()));
    if (err != nullptr) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to set release callback: %s",
                   TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    err = TRITONSERVER_InferenceRequestSetResponseCallback(
      request, allocator, reinterpret_cast<void *>(&cuda_stream_), InferenceComplete,
      reinterpret_cast<void *>(promise.get()));
    if (err != nullptr) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to set response callback: %s",
                   TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    err = TRITONSERVER_ServerInferAsync(server, request, nullptr);
    if (err != nullptr) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to infer asynchronously: %s",
                  TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    // The InferResponseComplete function sets the std::promise so
    // that this thread will block until the response is returned.
    TRITONSERVER_InferenceResponse * completed_response = completed.get();
    if (TRITONSERVER_InferenceResponseError(completed_response) != nullptr) {
      std::string error_message =
        TRITONSERVER_ErrorMessage(TRITONSERVER_InferenceResponseError(completed_response));
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to get inference response: %s",
                   error_message.c_str());
      TRITONSERVER_InferenceResponseDelete(completed_response);
      return false;
    }
    RCLCPP_DEBUG(get_logger(), "[TritonNode] Inference request callback completed");

    bool ret = ProcessInferenceResponse(completed_response, list_builder, output_bindings_map_,
      output_tensor_names_);
    if (!ret) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to process inference response");
      TRITONSERVER_InferenceResponseDelete(completed_response);
      return false;
    }

    release_future.get();
    TRITONSERVER_InferenceResponseDelete(completed_response);
    RCLCPP_DEBUG(get_logger(), "[TritonNode] Inference response processed");

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Exception in ExecuteInference: %s", e.what());
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
    return false;
  }
}

TritonNode::~TritonNode()
{
  ShutdownTritonServer();
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::TritonNode)
