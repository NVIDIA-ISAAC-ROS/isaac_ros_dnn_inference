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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

namespace TritonConv = triton_conversions;
using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

static constexpr char INPUT_TOPIC_NAME[] = "tensor_pub";
static constexpr char OUTPUT_TOPIC_NAME[] = "tensor_sub";
static constexpr int MAX_READINESS_POLL_ATTEMPTS = 2000;
static constexpr auto READINESS_POLL_INTERVAL = std::chrono::milliseconds(5);

namespace
{

void InferenceComplete(
  TRITONSERVER_InferenceResponse * response,
  const uint32_t flags,
  void * userp)
{
  if (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
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
  (void)buffer_userp;

  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;
  if (byte_size == 0) {
    *buffer = nullptr;
    return nullptr;
  }

  cudaStream_t cuda_stream = *reinterpret_cast<cudaStream_t *>(userp);
  void * allocated_buffer = nullptr;
  if (preferred_memory_type == TRITONSERVER_MEMORY_GPU) {
    auto err = cudaSetDevice(*actual_memory_type_id);
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "failed to set device");
    }
    err = cudaMallocAsync(&allocated_buffer, byte_size, cuda_stream);
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to allocate ") + std::to_string(byte_size) + " bytes").c_str());
    }
    err = cudaStreamSynchronize(cuda_stream);
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "failed to synchronize CUDA stream");
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
      return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to recover current CUDA device: ") +
        cudaGetErrorString(err)).c_str());
    }
    err = cudaHostAlloc(&allocated_buffer, byte_size, cudaHostAllocPortable);
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to allocate ") + std::to_string(byte_size) + " bytes").c_str());
    }
    *actual_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
    *actual_memory_type_id = preferred_memory_type_id;
  } else {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      (std::string("not supported memory type: ") +
      std::to_string(preferred_memory_type)).c_str());
  }

  if (allocated_buffer != nullptr) {
    *buffer = allocated_buffer;
  }
  return nullptr;
}

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
  (void)buffer_userp;
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
      break;
  }
  return nullptr;
}

void CopyResponseToTensor(
  TritonConv::Tensor & tensor, const void * src_buffer, size_t byte_size,
  TRITONSERVER_MemoryType memory_type, cudaStream_t stream)
{
  TritonConv::copy_to_tensor(tensor, src_buffer, byte_size, memory_type, stream);
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
  output_tensor_names_(declare_parameter<StringList>("output_tensor_names", StringList())),
  output_binding_names_(declare_parameter<StringList>("output_binding_names", StringList())),
  log_level_(declare_parameter<int>("log_level", 0)),
  backend_directory_(declare_parameter<std::string>("backend_directory", "")),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10))
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
  if (input_queue_size_ <= 0 || output_queue_size_ <= 0) {
    throw std::invalid_argument("[TritonNode] Queue sizes must be positive");
  }

  if (input_tensor_names_.size() != input_binding_names_.size()) {
    RCLCPP_ERROR(
      get_logger(),
      "[TritonNode] Input tensor names (%zu) and binding names (%zu) count mismatch",
      input_tensor_names_.size(), input_binding_names_.size());
    throw std::invalid_argument(
            "[TritonNode] Input tensor names and binding names count mismatch");
  }

  if (output_tensor_names_.size() != output_binding_names_.size()) {
    RCLCPP_ERROR(
      get_logger(),
      "[TritonNode] Output tensor names (%zu) and binding names (%zu) count mismatch",
      output_tensor_names_.size(), output_binding_names_.size());
    throw std::invalid_argument("Output tensor names and binding names count mismatch");
  }

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  input_sub_ = create_subscription<TensorList>(
    INPUT_TOPIC_NAME, input_qos,
    std::bind(&TritonNode::InputCallback, this, std::placeholders::_1),
    sub_options);
  output_pub_ = create_publisher<TensorList>(
    OUTPUT_TOPIC_NAME, output_qos, pub_options);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("TritonNode");

  if (!InitializeTritonServer()) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Triton server initialization failed");
    throw std::runtime_error("[TritonNode] Failed to initialize Triton server");
  }

  if (!InitializeBindingsMap()) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Failed to initialize input and output bindings map");
    throw std::runtime_error("[TritonNode] Failed to initialize input and output bindings map");
  }
  RCLCPP_INFO(get_logger(), "[TritonNode] Triton node ready with model: %s", model_name_.c_str());
}

void TritonNode::InputCallback(const TensorList::SharedPtr tensor_list)
{
  try {
    if (!triton_server_ready_) {
      RCLCPP_ERROR(get_logger(), "[TritonNode] Triton server not ready, dropping input");
      return;
    }
    TensorList output_tensor_list = DoInference(*tensor_list);
    output_pub_->publish(output_tensor_list);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "[TritonNode] Error in InputCallback: %s", e.what());
  }
}

TensorList TritonNode::DoInference(
  const TensorList & input_tensor_list)
{
  std::lock_guard<std::mutex> lock(triton_mutex_);

  if (!triton_server_ready_) {
    throw std::runtime_error("[TritonNode] Triton server not ready");
  }
  auto output_tensors = ExecuteInference(input_tensor_list);
  TensorList output_tensor_list;
  output_tensor_list.header = input_tensor_list.header;
  // Publish the ROS-side tensor names (e.g. "output"), not the Triton model
  // binding names (e.g. "mobilenetv20_output_flatten0_reshape0").
  output_tensor_list.names.assign(output_tensor_names_.begin(), output_tensor_names_.end());
  output_tensor_list.tensors = std::move(output_tensors);
  RCLCPP_DEBUG(get_logger(), "[TritonNode] Triton inference completed successfully");
  return output_tensor_list;
}

bool TritonNode::InitializeBindingsMap()
{
  for (size_t i = 0; i < input_tensor_names_.size(); ++i) {
    input_bindings_map_[input_tensor_names_[i]] = input_binding_names_[i];
  }

  for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
    output_bindings_map_[output_tensor_names_[i]] = output_binding_names_[i];
  }

  RCLCPP_INFO(get_logger(), "[TritonNode] Input and output bindings map initialized successfully");
  return true;
}

bool TritonNode::InitializeTritonServer()
{
  std::lock_guard<std::mutex> lock(triton_mutex_);

  try {
    RCLCPP_INFO(
      get_logger(), "[TritonNode] Initializing Triton server for model: %s",
      model_name_.c_str());

    struct ServerOptionsDeleter
    {
      void operator()(TRITONSERVER_ServerOptions * opts) const
      {
        if (opts) {TRITONSERVER_ServerOptionsDelete(opts);}
      }
    };

    TRITONSERVER_ServerOptions * triton_server_options = nullptr;
    TRITONSERVER_Error * err = TRITONSERVER_ServerOptionsNew(
      reinterpret_cast<TRITONSERVER_ServerOptions **>(&triton_server_options));
    if (err != nullptr) {
      RCLCPP_ERROR(
        get_logger(), "[TritonNode] Failed to create server options: %s",
        TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }
    std::unique_ptr<TRITONSERVER_ServerOptions, ServerOptionsDeleter> options_guard(
      reinterpret_cast<TRITONSERVER_ServerOptions *>(triton_server_options),
      ServerOptionsDeleter());

    auto server_options = reinterpret_cast<TRITONSERVER_ServerOptions *>(triton_server_options);
    for (const auto & repo_path : model_repository_paths_) {
      if (!std::filesystem::exists(repo_path)) {
        RCLCPP_ERROR(
          get_logger(), "[TritonNode] Model repository path does not exist: %s",
          repo_path.c_str());
        return false;
      }
      err = TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, repo_path.c_str());
      if (err != nullptr) {
        RCLCPP_ERROR(
          get_logger(), "[TritonNode] Failed to set repository path: %s",
          TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        return false;
      }
      RCLCPP_INFO(get_logger(), "[TritonNode] Added model repository: %s", repo_path.c_str());
    }

    err = TRITONSERVER_ServerOptionsSetModelControlMode(
      server_options, TRITONSERVER_MODEL_CONTROL_EXPLICIT);
    if (err != nullptr) {
      RCLCPP_ERROR(
        get_logger(), "[TritonNode] Failed to set explicit model control mode: %s",
        TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    err = TRITONSERVER_ServerOptionsSetStartupModel(server_options, model_name_.c_str());
    if (err != nullptr) {
      RCLCPP_ERROR(
        get_logger(), "[TritonNode] Failed to set startup model '%s': %s",
        model_name_.c_str(), TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }
    RCLCPP_INFO(
      get_logger(), "[TritonNode] Configured Triton to load startup model: %s",
      model_name_.c_str());

    TRITONSERVER_ServerOptionsSetLogVerbose(server_options, log_level_);
    TRITONSERVER_ServerOptionsSetLogInfo(server_options, enable_triton_logging_);
    TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, enable_strict_model_);
    TRITONSERVER_ServerOptionsSetServerId(server_options, "isaac_ros_triton_server");

    std::string effective_backend_dir = backend_directory_;
    if (effective_backend_dir.empty()) {
      const char * ldpath_env = std::getenv("LD_LIBRARY_PATH");
      if (ldpath_env) {
        std::istringstream ss(ldpath_env);
        std::string entry;

        while (std::getline(ss, entry, ':')) {
          std::filesystem::path p(entry);
          if ((p.filename() == "lib" || p.filename() == "lib64") &&
            p.parent_path().filename() == "tritonserver")
          {
            auto candidate = p.parent_path() / "backends";
            if (std::filesystem::exists(candidate)) {
              effective_backend_dir = candidate.string();
              break;
            }
          }
        }
      }
    }
    if (!effective_backend_dir.empty()) {
      err = TRITONSERVER_ServerOptionsSetBackendDirectory(
        server_options, effective_backend_dir.c_str());
      if (err != nullptr) {
        RCLCPP_ERROR(
          get_logger(), "[TritonNode] Failed to set backend directory '%s': %s",
          effective_backend_dir.c_str(), TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        return false;
      }
      RCLCPP_INFO(
        get_logger(), "[TritonNode] Using backend directory: %s",
        effective_backend_dir.c_str());
    } else {
      RCLCPP_WARN(
        get_logger(),
        "[TritonNode] No backend directory resolved; Triton will use its compiled-in default");
    }

    TRITONSERVER_Server * server_ptr = nullptr;
    err = TRITONSERVER_ServerNew(&server_ptr, server_options);
    if (err != nullptr) {
      RCLCPP_ERROR(
        get_logger(), "[TritonNode] Failed to create Triton server: %s",
        TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    RCLCPP_INFO(get_logger(), "[TritonNode] Triton server created successfully");

    triton_server_ = std::unique_ptr<void, void (*)(void *)>(
      server_ptr,
      [](void * ptr) {
        if (ptr) {
          TRITONSERVER_ServerDelete(reinterpret_cast<TRITONSERVER_Server *>(ptr));
        }
      }
    );
    options_guard.release();

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
    triton_server_ready_ = true;
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
    auto server = reinterpret_cast<TRITONSERVER_Server *>(triton_server_.get());
    TRITONSERVER_Error * error = TRITONSERVER_ServerStop(server);
    if (error != nullptr) {
      RCLCPP_ERROR(
        get_logger(), "[TritonNode] Error stopping Triton server: %s",
        TRITONSERVER_ErrorMessage(error));
      TRITONSERVER_ErrorDelete(error);
    } else {
      RCLCPP_INFO(get_logger(), "[TritonNode] Triton server shutdown initiated successfully");
    }

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

    triton_server_.reset();
  }

  triton_server_ready_ = false;
  RCLCPP_INFO(get_logger(), "[TritonNode] Triton server shutdown complete");
}

std::vector<TritonConv::Tensor> TritonNode::ProcessInferenceResponse(
  TRITONSERVER_InferenceResponse * response)
{
  uint32_t output_count = 0;
  TRITONSERVER_Error * err = TRITONSERVER_InferenceResponseOutputCount(response, &output_count);
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("[TritonNode] Failed to get output count: " + msg);
  }

  struct OutputInfo
  {
    const char * name;
    TRITONSERVER_DataType dtype;
    std::vector<int64_t> shape;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    const void * buffer;
  };
  std::vector<OutputInfo> output_infos;

  for (uint32_t i = 0; i < output_count; ++i) {
    OutputInfo output_info;
    const int64_t * shape;
    uint64_t dims_count;
    int64_t memory_type_id = 0;
    void * userp = nullptr;

    err = TRITONSERVER_InferenceResponseOutput(
      response, i, &output_info.name, &output_info.dtype, &shape,
      &dims_count, &output_info.buffer, &output_info.byte_size,
      &output_info.memory_type, &memory_type_id, &userp);
    if (err != nullptr) {
      const std::string msg = TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      throw std::runtime_error("[TritonNode] Failed to get output info: " + msg);
    }
    output_info.shape.reserve(dims_count);
    for (uint64_t d = 0; d < dims_count; ++d) {
      output_info.shape.push_back(shape[d]);
    }
    output_infos.push_back(std::move(output_info));
  }

  std::vector<TritonConv::Tensor> output_tensors;
  output_tensors.reserve(output_tensor_names_.size());

  for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
    const std::string & tensor_name = output_tensor_names_[i];
    const std::string & binding_name = output_bindings_map_.at(tensor_name);

    const auto output_info_it = std::find_if(
      output_infos.begin(), output_infos.end(),
      [&binding_name](const OutputInfo & info) {
        return info.name == binding_name;
      });
    if (output_info_it == output_infos.end()) {
      throw std::runtime_error(
              "[TritonNode] Failed to find output info for tensor: " + binding_name);
    }
    const TritonConv::DLDataType dlpack_dtype =
      TritonConv::from_triton_data_type(output_info_it->dtype);
    TritonConv::Tensor tensor = TritonConv::allocate_tensor(
      output_info_it->shape, dlpack_dtype.code, dlpack_dtype.bits, dlpack_dtype.lanes);

    CopyResponseToTensor(
      tensor, output_info_it->buffer, output_info_it->byte_size,
      output_info_it->memory_type, *cuda_stream_);
    cudaError_t cuda_err = cudaStreamSynchronize(*cuda_stream_);
    if (cuda_err != cudaSuccess) {
      throw std::runtime_error(
              std::string("[TritonNode] CUDA stream synchronization failed: ") +
              cudaGetErrorString(cuda_err));
    }
    output_tensors.push_back(std::move(tensor));
  }

  return output_tensors;
}

std::vector<TritonConv::TritonTensor<cuda_buffer_backend::ReadHandle>>
TritonNode::PrepareInputHandlesForExternalConsumer(
  const TensorList & tensor_list,
  const StringList & input_tensor_names,
  cudaStream_t stream)
{
  std::vector<TritonConv::TritonTensor<cuda_buffer_backend::ReadHandle>> input_handles;
  input_handles.reserve(input_tensor_names.size());

  for (const auto & tensor_name : input_tensor_names) {
    const TritonConv::Tensor * tensor =
      isaac_ros_tensor_msgs::FindTensorByName(tensor_list, tensor_name);
    if (!tensor) {
      throw std::invalid_argument("[TritonNode] Input tensor not found: " + tensor_name);
    }
    auto triton_tensor = TritonConv::from_input_tensor(tensor_name, *tensor, stream);
    input_handles.push_back(std::move(triton_tensor));
  }

  // Read handles queue producer-event waits on this stream, but Triton consumes
  // the raw pointers on backend-owned streams. Complete those waits before the
  // pointers cross that stream-ownership boundary.
  const cudaError_t synchronize_error = cudaStreamSynchronize(stream);
  if (synchronize_error != cudaSuccess) {
    throw std::runtime_error(
            std::string("[TritonNode] Failed to synchronize input stream: ") +
            cudaGetErrorString(synchronize_error));
  }

  return input_handles;
}

void InferRequestRelease(
  TRITONSERVER_InferenceRequest * request, const uint32_t flags, void * userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    TRITONSERVER_InferenceRequestDelete(request);
    std::promise<void> * barrier = reinterpret_cast<std::promise<void> *>(userp);
    barrier->set_value();
  }
}

std::vector<triton_conversions::Tensor> TritonNode::ExecuteInference(
  const TensorList & tensor_list)
{
  auto server = reinterpret_cast<TRITONSERVER_Server *>(triton_server_.get());

  TRITONSERVER_InferenceRequest * request = nullptr;
  TRITONSERVER_Error * err = TRITONSERVER_InferenceRequestNew(
    &request, server, model_name_.c_str(), -1);
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("[TritonNode] Failed to create inference request: " + msg);
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

  std::vector<TritonConv::TritonTensor<cuda_buffer_backend::ReadHandle>> input_tensors;
  input_tensors = PrepareInputHandlesForExternalConsumer(
    tensor_list, input_tensor_names_,
    *cuda_stream_);

  for (size_t i = 0; i < input_tensors.size() && i < input_binding_names_.size(); ++i) {
    const std::string & tensor_name = input_tensor_names_[i];
    const std::string & binding_name = input_bindings_map_.at(tensor_name);

    const auto & shape = input_tensors[i].shape();
    err = TRITONSERVER_InferenceRequestAddInput(
      request, binding_name.c_str(), input_tensors[i].data_type(),
      shape.data(), shape.size());
    if (err != nullptr) {
      const std::string msg = TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      throw std::runtime_error("[TritonNode] Failed to add input to request: " + msg);
    }

    err = TRITONSERVER_InferenceRequestAppendInputData(
      request, binding_name.c_str(), input_tensors[i].data(), input_tensors[i].size_bytes(),
      TRITONSERVER_MEMORY_GPU, 0);
    //  triton_tensor.memory_type(), triton_tensor.memory_type_id());
    if (err != nullptr) {
      const std::string msg = TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      throw std::runtime_error("[TritonNode] Failed to add tensor data to request: " + msg);
    }
  }

  for (const auto & output_name : output_binding_names_) {
    err = TRITONSERVER_InferenceRequestAddRequestedOutput(request, output_name.c_str());
    if (err != nullptr) {
      const std::string msg = TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
      throw std::runtime_error("[TritonNode] Failed to add output to request: " + msg);
    }
  }
  TRITONSERVER_ResponseAllocator * allocator = nullptr;
  err = TRITONSERVER_ResponseAllocatorNew(
    &allocator,
    ResponseAlloc,
    ResponseRelease,
    nullptr
  );
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("[TritonNode] Failed to create response allocator: " + msg);
  }
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
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("[TritonNode] Failed to set release callback: " + msg);
  }

  err = TRITONSERVER_InferenceRequestSetResponseCallback(
    request, allocator, reinterpret_cast<void *>(cuda_stream_.get()), InferenceComplete,
    reinterpret_cast<void *>(promise.get()));
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("[TritonNode] Failed to set response callback: " + msg);
  }
  err = TRITONSERVER_ServerInferAsync(server, request, nullptr);
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("[TritonNode] Failed to infer asynchronously: " + msg);
  }
  request_guard.release();

  // The InferResponseComplete function sets the std::promise so
  // that this thread will block until the response is returned.
  TRITONSERVER_InferenceResponse * completed_response = completed.get();
  struct InferenceResponseDeleter
  {
    void operator()(TRITONSERVER_InferenceResponse * response) const
    {
      if (response != nullptr) {
        TRITONSERVER_InferenceResponseDelete(response);
      }
    }
  };
  std::unique_ptr<TRITONSERVER_InferenceResponse,
    InferenceResponseDeleter> response_guard(completed_response);

  // Keep release_promise alive until Triton invokes InferRequestRelease.
  release_future.get();
  if (TRITONSERVER_InferenceResponseError(completed_response) != nullptr) {
    std::string error_message =
      TRITONSERVER_ErrorMessage(TRITONSERVER_InferenceResponseError(completed_response));
    RCLCPP_ERROR(
      get_logger(), "[TritonNode] Failed to get inference response: %s",
      error_message.c_str());
    throw std::runtime_error("[TritonNode] Failed to get inference response: " + error_message);
  }

  std::vector<TritonConv::Tensor> output_tensors = ProcessInferenceResponse(completed_response);
  return output_tensors;
}

TritonNode::~TritonNode()
{
  ShutdownTritonServer();
}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_inference::TritonNode)
