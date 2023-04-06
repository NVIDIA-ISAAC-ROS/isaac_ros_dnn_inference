// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "triton_inferencer_impl.hpp"

#include <inttypes.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

#include "extensions/triton/triton_options.hpp"

#include "infer_icontext.h"
#include "infer_options.h"
#include "nvdsinferserver_config.pb.h"

namespace nvidia {
namespace triton {

struct Inference {
    // Entity that will preserve lifetime for input tensors for the Inference Request
    std::vector<nvidia::gxf::Entity> preserved_input;

    // Raw NvDs Ouptuts of the Inference Request
    // WAR: The ownership should ideally be handed to a GXF Tensor: GXF-204
    nvdsinferserver::SharedIBatchArray raw_output;

    // Inference Status to be modified by Inference Request completion
    NvDsInferStatus infer_status { NVDSINFER_SUCCESS };

    // Future event indicates completion via callback
    std::promise<void> response_promise;
    std::future<void> response_completion;

    // Indicates whether this inference is currently active; this can be asynchronously accessed
    // via isAcceptingRequest()
    std::atomic<bool> is_active = { false };

    // Indicates whether this inference is complete
    bool is_complete = { false };
};

/**
 * @brief Helper function for translating TRITONSERVER_DataType to nvidia::gxf::PrimitiveType
 *
 *
 * @param datatype
 * @return nvidia::gxf::PrimitiveType
 */
static nvidia::gxf::PrimitiveType NvDsToGxfDataType(nvdsinferserver::InferDataType datatype) {
  // Unsupported:
  //  nvdsinferserver::InferDataType::kFp16
  //  nvdsinferserver::InferDataType::kString
  //  nvdsinferserver::InferDataType::kBool
  static const std::unordered_map<nvdsinferserver::InferDataType, nvidia::gxf::PrimitiveType>
    sNvDsToGxf{
      {nvdsinferserver::InferDataType::kUint8, nvidia::gxf::PrimitiveType::kUnsigned8},
      {nvdsinferserver::InferDataType::kUint16, nvidia::gxf::PrimitiveType::kUnsigned16},
      {nvdsinferserver::InferDataType::kUint32, nvidia::gxf::PrimitiveType::kUnsigned32},
      {nvdsinferserver::InferDataType::kUint64, nvidia::gxf::PrimitiveType::kUnsigned64},
      {nvdsinferserver::InferDataType::kInt8, nvidia::gxf::PrimitiveType::kInt8},
      {nvdsinferserver::InferDataType::kInt16, nvidia::gxf::PrimitiveType::kInt16},
      {nvdsinferserver::InferDataType::kInt32, nvidia::gxf::PrimitiveType::kInt32},
      {nvdsinferserver::InferDataType::kInt64, nvidia::gxf::PrimitiveType::kInt64},
      {nvdsinferserver::InferDataType::kFp32, nvidia::gxf::PrimitiveType::kFloat32},
      {nvdsinferserver::InferDataType::kFp64, nvidia::gxf::PrimitiveType::kFloat64},
      {nvdsinferserver::InferDataType::kString, nvidia::gxf::PrimitiveType::kCustom},
  };
  auto const i = sNvDsToGxf.find(datatype);
  if (i == sNvDsToGxf.end()) {
    GXF_LOG_WARNING("Unsupported NvDs data type: %d", datatype);
    return nvidia::gxf::PrimitiveType::kCustom;
  }
  return i->second;
}

/**
 * @brief Helper function for translating nvidia::gxf::PrimitiveType to
 * nvdsinferserver::InferDataType
 *
 * @param datatype
 * @return nvdsinferserver::InferDataType
 */
static nvdsinferserver::InferDataType GxfToNvDsDataType(
  nvidia::gxf::PrimitiveType datatype) {
  static const std::unordered_map<nvidia::gxf::PrimitiveType, nvdsinferserver::InferDataType>
    sGxfToNvDsData{
      {nvidia::gxf::PrimitiveType::kUnsigned8, nvdsinferserver::InferDataType::kUint8},
      {nvidia::gxf::PrimitiveType::kUnsigned16, nvdsinferserver::InferDataType::kUint16},
      {nvidia::gxf::PrimitiveType::kUnsigned32, nvdsinferserver::InferDataType::kUint32},
      {nvidia::gxf::PrimitiveType::kUnsigned64, nvdsinferserver::InferDataType::kUint64},
      {nvidia::gxf::PrimitiveType::kInt8, nvdsinferserver::InferDataType::kUint8},
      {nvidia::gxf::PrimitiveType::kInt16, nvdsinferserver::InferDataType::kInt16},
      {nvidia::gxf::PrimitiveType::kInt32, nvdsinferserver::InferDataType::kInt32},
      {nvidia::gxf::PrimitiveType::kInt64, nvdsinferserver::InferDataType::kInt64},
      {nvidia::gxf::PrimitiveType::kFloat32, nvdsinferserver::InferDataType::kFp32},
      {nvidia::gxf::PrimitiveType::kFloat64, nvdsinferserver::InferDataType::kFp64},
      {nvidia::gxf::PrimitiveType::kCustom, nvdsinferserver::InferDataType::kString},
  };
  // NOTE: Unsupported nvdsinferserver::InferDataType are:
  // - kFp16
  // - kBool
  // - kNone
  auto const i = sGxfToNvDsData.find(datatype);
  if (i == sGxfToNvDsData.end()) {
    GXF_LOG_WARNING("Unsupported GXF data type: %d", datatype);
    return nvdsinferserver::InferDataType::kNone;
  }
  return i->second;
}

/**
 * @brief Helper function for translating nvidia::gxf::MemoryStorageType to
 * nvdsinferserver::InferMemType
 *
 * @param memory_type
 * @return nvdsinferserver::InferMemType
 */
static nvdsinferserver::InferMemType GxfMemTypeToNvDsMemType(
  nvidia::gxf::MemoryStorageType memory_type) {
  static const std::unordered_map<nvidia::gxf::MemoryStorageType, nvdsinferserver::InferMemType>
    sGxfToNvDsMem{
      {nvidia::gxf::MemoryStorageType::kHost, nvdsinferserver::InferMemType::kCpu},
      {nvidia::gxf::MemoryStorageType::kSystem, nvdsinferserver::InferMemType::kCpuCuda},
      {nvidia::gxf::MemoryStorageType::kDevice, nvdsinferserver::InferMemType::kGpuCuda},
  };
  auto const i = sGxfToNvDsMem.find(memory_type);
  if (i == sGxfToNvDsMem.end()) {
    GXF_LOG_WARNING("Unsupported GXF data type: %d", memory_type);
    return nvdsinferserver::InferMemType::kNone;
  }
  return i->second;
}

static nvdsinferserver::config::MemoryType GxfMemTypeToNvDsConfMemType(
  nvidia::gxf::MemoryStorageType memory_type) {
  static const std::unordered_map<nvidia::gxf::MemoryStorageType,
  nvdsinferserver::config::MemoryType>
    sGxfToNvDsMem{
      {nvidia::gxf::MemoryStorageType::kHost,
       nvdsinferserver::config::MemoryType::MEMORY_TYPE_CPU},
      {nvidia::gxf::MemoryStorageType::kSystem,
      nvdsinferserver::config::MemoryType::MEMORY_TYPE_CPU},
      {nvidia::gxf::MemoryStorageType::kDevice,
      nvdsinferserver::config::MemoryType::MEMORY_TYPE_GPU},
  };
  auto const i = sGxfToNvDsMem.find(memory_type);
  if (i == sGxfToNvDsMem.end()) {
    GXF_LOG_WARNING("Unsupported GXF data type: %d", memory_type);
    return nvdsinferserver::config::MemoryType::MEMORY_TYPE_DEFAULT;
  }
  return i->second;
}

static nvidia::gxf::MemoryStorageType NvDsMemTypeToGxfMemType(
  nvdsinferserver::InferMemType memory_type) {
  static const std::unordered_map<nvdsinferserver::InferMemType, nvidia::gxf::MemoryStorageType>
    sNvDsMemToGxf{
      {nvdsinferserver::InferMemType::kCpu, nvidia::gxf::MemoryStorageType::kHost},
      {nvdsinferserver::InferMemType::kCpuCuda, nvidia::gxf::MemoryStorageType::kSystem},
      {nvdsinferserver::InferMemType::kGpuCuda, nvidia::gxf::MemoryStorageType::kDevice},
  };
  auto const i = sNvDsMemToGxf.find(memory_type);
  GXF_ASSERT(i != sNvDsMemToGxf.end(), "Unsupported conversion from NvDs data type: %d",
    memory_type);
  return i->second;
}

static gxf_result_t setTritonOptions(
  nvdsinferserver::SharedIBatchArray& batchArray, const TritonOptions& triton_options) {
  nvdsinferserver::SharedBufOptions options = std::make_shared<nvdsinferserver::BufOptions>();
  if (!options) {
    GXF_LOG_ERROR("Unable to create Triton Options: SharedBufOptions");
    return GXF_NULL_POINTER;
  }
  options->setValue(OPTION_SEQUENCE_ID, triton_options.sequence_id);
  options->setValue(OPTION_SEQUENCE_START, triton_options.start);
  options->setValue(OPTION_SEQUENCE_END, triton_options.end);
  options->setValue(OPTION_PRIORITY, triton_options.priority);
  options->setValue(OPTION_TIMEOUT, triton_options.timeout);

  if (!batchArray) {
    GXF_LOG_ERROR("Input batch array for setting Triton Options is null");
    return GXF_NULL_POINTER;
  }
  batchArray->setIOptions(std::move(options));
  if (!batchArray->getOptions()) {
    GXF_LOG_ERROR("Batch array unable to getOptions()");
    return GXF_NULL_POINTER;
  }
  return GXF_SUCCESS;
}

gxf_result_t TritonInferencerImpl::initialize() {
  scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);

  // Initialize pool of Inference Requests. This is needed to occur before start() for proper
  // behavior with Scheduling Terms dependent upon isAcceptingRequest().
  inference_pool_.resize(num_concurrent_requests_.get());
  for (size_t num = 0; num < num_concurrent_requests_.get(); num++) {
    inference_pool_[num] = new Inference();
  }
  if (!inference_pool_.size()) {
    GXF_LOG_ERROR("Inference Pool Empty");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t TritonInferencerImpl::construct() {
  if (!inference_pool_.size()) {
    GXF_LOG_ERROR("Inference Pool Empty");
    return GXF_FAILURE;
  }

  if (nvidia::gxf::Shape::kMaxRank != NVDSINFER_MAX_DIMS) {
    GXF_LOG_WARNING("GXF and NvDs Max Rank are mistmatched, which may cause problems.");
  }

  nvdsinferserver::config::InferenceConfig inference_config;
  inference_config.set_unique_id(static_cast<uint32_t>(eid()));
  inference_config.set_max_batch_size(max_batch_size_.get());
  inference_config.mutable_backend()->mutable_triton()->set_model_name(model_name_.get());
  inference_config.mutable_backend()->mutable_triton()->set_version(model_version_.get());
  inference_config.add_gpu_ids(0);

  // ensure no pre or post processing is attached
  inference_config.clear_preprocess();
  inference_config.clear_postprocess();

  if (inference_mode_.get() == TritonInferenceMode::kDirect) {
    if (!server_handle_.try_get()) {
      GXF_LOG_ERROR("Triton Server Handle is null with Direct inference mode");
      return GXF_ARGUMENT_INVALID;
    }
    auto maybe_server = server_handle_.try_get().value()->getServer();
    if (!maybe_server) {
      GXF_LOG_ERROR("Triton Server is unexpected");
      return nvidia::gxf::ToResultCode(maybe_server);
    }
    server_ = maybe_server.value();

    auto maybe_model_repo = server_handle_.try_get().value()->getServer();
    if (!maybe_model_repo) {
    GXF_LOG_ERROR("Triton Server is unexpected");
    return nvidia::gxf::ToResultCode(maybe_model_repo);
    }
    auto model_repo = server_handle_.try_get().value()->getModelRepoConfig().value();

    inference_config.mutable_backend()->mutable_triton()->
      mutable_model_repo()->CopyFrom(*model_repo);

    // suggests memory output storage location to Triton
    if (output_storage_type_.try_get()) {
      auto output_mem_type = GxfMemTypeToNvDsConfMemType(
        nvidia::gxf::MemoryStorageType(output_storage_type_.try_get().value()));
      inference_config.mutable_backend()->set_output_mem_type(output_mem_type);
    }

    if (use_string_data_.get() || use_sequence_data_.get()) {
      infer_context_.reset(createInferTritonSimpleContext());
    } else {
      std::string configStr = inference_config.DebugString();
      infer_context_.reset(createInferTrtISContext(configStr.c_str(), configStr.size()));
    }

  } else if (inference_mode_.get() == TritonInferenceMode::kRemoteGrpc) {
    if (!server_endpoint_.try_get()) {
      GXF_LOG_ERROR("Remote endpoint is not set with RemoteGrpc inference mode");
      return GXF_ARGUMENT_INVALID;
    }
    inference_config.mutable_backend()->mutable_triton()->
    mutable_grpc()->set_url(server_endpoint_.try_get().value());

    // suggests memory output storage location to Triton
    if (output_storage_type_.try_get()) {
      auto output_mem_type = GxfMemTypeToNvDsConfMemType(
        nvidia::gxf::MemoryStorageType(output_storage_type_.try_get().value()));
      inference_config.mutable_backend()->set_output_mem_type(output_mem_type);
    }

    // consider adding -> enable_cuda_buffer_sharing, when the DS team enables it

    if (use_string_data_.get() || use_sequence_data_.get()) {
      infer_context_.reset(createInferTritonSimpleContext());
    } else {
      std::string configStr = inference_config.DebugString();
      infer_context_.reset(createInferTritonGrpcContext(configStr.c_str(), configStr.size()));
    }
  } else {
    GXF_LOG_ERROR("Invalid inference mode");
    return GXF_ARGUMENT_INVALID;
  }

  if (!infer_context_) {
    GXF_LOG_ERROR("Failure to create Inference Context for '%s'", model_name_.get().c_str());
    return GXF_FAILURE;
  }

  NvDsInferStatus status = NVDSINFER_SUCCESS;
  status = infer_context_->initialize(inference_config.DebugString(), nullptr);

  if (status != NVDSINFER_SUCCESS) {
    GXF_LOG_ERROR("Failure to initialize Inference Context for '%s'", model_name_.get().c_str());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t TritonInferencerImpl::destruct() {
  for (auto& inference_ptr : inference_pool_) {
    delete inference_ptr;
  }
  return GXF_SUCCESS;
}

gxf_result_t TritonInferencerImpl::inferAsync(
  const std::vector<nvidia::gxf::Entity> input_entities,
  const std::vector<std::string> input_names) {
  // Reset scheduling term event to wait for next response.
  scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_WAITING);

  // Use the current inference index to modify the inference object in the inference callback.
  size_t current_inference_index = next_inference_index_.load();
  auto& inference = inference_pool_[current_inference_index];
  if (inference->is_active) {
    GXF_LOG_ERROR("Next available Inference Context for '%s' is active. "
      "Increase num_concurrent_requests.", model_name_.get().c_str());
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }

  next_inference_index_ = (next_inference_index_ + 1) % num_concurrent_requests_.get();

  if (!infer_context_) {
    GXF_LOG_ERROR("Inference Context not initialized");
    return GXF_FAILURE;
  }

  // NOTE: Triton will own the input entity data until inference is complete.
  // This is released during getResponse().
  inference->preserved_input = input_entities;

  nvdsinferserver::SharedIBatchArray input = NvDsInferServerCreateBatchArray();
  if (!input) {
    GXF_LOG_ERROR("Unable to create nvds input tensors");
    return GXF_FAILURE;
  }

  if (input_entities.size() != input_names.size()) {
    GXF_LOG_ERROR("Mismatch in number of input_entities and input_names");
    return GXF_FAILURE;
  }

  for (size_t i = 0; i < input_entities.size(); i++) {
    auto input_entity = input_entities[i];
    auto input_name = input_names[i];
    auto maybe_entity_clone = input_entity.clone();
    if (!maybe_entity_clone) {
      GXF_LOG_ERROR("Unable to clone input entity");
      return nvidia::gxf::ToResultCode(maybe_entity_clone);
    }
    auto entity_clone = maybe_entity_clone.value();
    inference->preserved_input.push_back(entity_clone);
    auto input_tensors = entity_clone.findAll<nvidia::gxf::Tensor>().value();
    for (auto input_tensor : input_tensors) {
      const auto tensor = input_tensor.value();
      const auto& name = input_name.c_str();
      GXF_LOG_DEBUG("input tensor name = %s", name);

      if (tensor->rank() > NVDSINFER_MAX_DIMS) {
        GXF_LOG_ERROR("Tensor rank '%u' is larger than NVDSINFER_MAX_DIMS");
        return GXF_FAILURE;
      }

      // Input tensor needs to be fully batched
      uint32_t batch_size = 0;  // Offload "batch" to the fully specified InferDims instead
      nvdsinferserver::InferDims dims;

      auto dataType = GxfToNvDsDataType(tensor->element_type());
      if (dataType == nvdsinferserver::InferDataType::kString) {
        auto maybe_string_shape = input_entity.get<nvidia::gxf::Shape>();
        if (!maybe_string_shape) {
          GXF_LOG_ERROR("Found Tensor with String Datatype "\
          "but no accompanying shape specification: "\
            "%s", name);
          return GXF_FAILURE;
        }
        const auto shape = *maybe_string_shape.value();
        dims.numDims = shape.rank();
        dims.numElements = shape.size();
        for (size_t index = 0; index < shape.rank(); index++) {
          if (shape.dimension(index) <= 0) {
            GXF_LOG_ERROR("Tensor Dimension <= 0 not allowed");
            return GXF_FAILURE;
          }
          dims.d[index] = shape.dimension(index);
        }
      } else {
        dims.numDims = tensor->rank();
        dims.numElements = static_cast<uint32_t>(tensor->element_count());
        for (size_t index = 0; index < tensor->rank(); index++) {
          if (tensor->shape().dimension(index) <= 0) {
            GXF_LOG_ERROR("Tensor Dimension <= 0 not allowed");
            return GXF_FAILURE;
          }
          dims.d[index] = tensor->shape().dimension(index);
        }
      }

      nvdsinferserver::InferBufferDescription description {
        memType : GxfMemTypeToNvDsMemType(tensor->storage_type()),
        devId : 0, /* NOTE: GXF Allocator does not have concept of device ID for kGPU */
        dataType : dataType,
        dims : dims,
        elementSize : static_cast<uint32_t>(tensor->bytes_per_element()),
        name : name,
        isInput : true
      };

      auto buffer = NvDsInferServerWrapBuf(
        tensor->pointer(), tensor->size(), description, batch_size, [](void* data) {});
      input->appendIBatchBuf(buffer);
    }
  }

  auto maybe_triton_option = inference->
    preserved_input.front().get<nvidia::triton::TritonOptions>();
  if (maybe_triton_option) {
    auto result = setTritonOptions(input, *maybe_triton_option.value());
    if (result != GXF_SUCCESS) {
      return result;
    }
  }

  // Create a promise to be used in the inference callback
  inference->response_promise = std::move(std::promise<void>());

  // Create a future object to be set in the inference callback
  inference->response_completion = std::move(std::future<void>(
    inference->response_promise.get_future()));

  inference->infer_status = NVDSINFER_SUCCESS;
  inference->is_active = true;

  // Increase count to be decremented once inference response is received.
  incomplete_inference_count_++;

  NvDsInferStatus runStatus = infer_context_->run(
    input, [current_inference_index, this](
      NvDsInferStatus s, nvdsinferserver::SharedIBatchArray out) {
        auto& inference = inference_pool_[current_inference_index];
        inference->infer_status = s;
        inference->raw_output = std::move(out);
        inference->is_complete = true;

        if (scheduling_term_->getEventState() == nvidia::gxf::AsynchronousEventState::EVENT_DONE) {
          GXF_LOG_DEBUG("Triton Async Event is unexpectedly already marked DONE");
        }
        scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_DONE);
        GXF_LOG_DEBUG("Triton Async Event DONE for index = %zu", current_inference_index);

        // NOTE: Set response promise last so that the EVENT_DONE notification occurs before the
        // response wait() unblocks. This is to prevent a situation where EVENT_DONE is triggered
        // after the entire response has already been processed.
        inference->response_promise.set_value();
      });

  if (runStatus != NVDSINFER_SUCCESS) {
    GXF_LOG_ERROR("Unable to run Inference Context");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}


nvidia::gxf::Expected<nvidia::gxf::Entity> TritonInferencerImpl::getResponse() {
  if (!infer_context_) {
    GXF_LOG_ERROR("InferenceContext not initialized");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  auto& inference = inference_pool_[active_inference_index_];

  if (!inference->is_complete) {
    GXF_LOG_WARNING("Incomplete inference; response appeared out of order; invalid: index = %zu",
      active_inference_index_);
    GXF_LOG_WARNING("Inference appeared out of order");
  }
  GXF_LOG_DEBUG("Trying to load inference for index: %zu", active_inference_index_);

  // Ensure the inference to complete. This normally will not block since the shared state
  // should already be ready if this tick has been scheduled; however, if the inference response is
  // received out of order from the inference request, we will wait to enforce FIFO.
  inference->response_completion.wait();
  if (!inference->response_completion.valid() || !inference->is_active.load()) {
    GXF_LOG_ERROR("Unexpectedly incomplete response for index: %zu", active_inference_index_);
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }
  GXF_LOG_DEBUG("Successfully loaded inference for: %zu", active_inference_index_);

  // Increment the active index for the next response.
  active_inference_index_ = (active_inference_index_ + 1) % num_concurrent_requests_.get();

  if (inference->infer_status != NVDSINFER_SUCCESS) {
    GXF_LOG_ERROR("Error with NvDs Async Infer: %d", inference->infer_status);
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  if (!inference->raw_output) {
    GXF_LOG_ERROR("Unable to get valid outputs from Inference");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  auto maybe_output_entity = nvidia::gxf::Entity::New(context_);
  if (!maybe_output_entity) {
    GXF_LOG_ERROR("Unable to create maybe_output_entity");
    return maybe_output_entity;
  }

  auto maybe_input_timestamp = inference->
    preserved_input.front().get<nvidia::gxf::Timestamp>("timestamp");
  if (maybe_input_timestamp) {
    auto maybe_output_timestamp =
      maybe_output_entity.value().add<nvidia::gxf::Timestamp>("timestamp");
    if (!maybe_output_timestamp) {
      GXF_LOG_ERROR("Unable to create maybe_output_timestamp");
      return nvidia::gxf::Unexpected{GXF_FAILURE};
    }

    *(maybe_output_timestamp.value().get()) = *(maybe_input_timestamp.value().get());
  }

  auto maybe_input_triton_option = inference->
    preserved_input.front().get<nvidia::triton::TritonOptions>();
  if (maybe_input_triton_option) {
    auto maybe_output_triton_option =
      maybe_output_entity.value().add<nvidia::triton::TritonOptions>();
    if (!maybe_output_triton_option) {
      GXF_LOG_ERROR("Unable to create maybe_output_triton_option");
      return nvidia::gxf::Unexpected{GXF_FAILURE};
    }

    *(maybe_output_triton_option.value().get()) = *(maybe_input_triton_option.value().get());
  }

  // Release the ref-counted input entity
  inference->preserved_input.clear();
  auto& nvds_output = inference->raw_output;

  GXF_LOG_DEBUG("Raw Outputs size = %u", inference->raw_output->getSize());

  for (uint32_t index = 0; index < inference->raw_output->getSize(); index++) {
    const nvdsinferserver::IBatchBuffer* output_buf = inference->raw_output->getBuffer(index);
    nvdsinferserver::SharedIBatchBuffer output_safe_buf = inference->raw_output->getSafeBuf(index);
    if (!output_buf || output_safe_buf.get() != output_buf) {
      GXF_LOG_ERROR("Mismatch between safe buffer and regular buffer of NvDs");
      return nvidia::gxf::Unexpected{GXF_FAILURE};
    }

    const nvdsinferserver::InferBufferDescription& description = output_buf->getBufDesc();
    GXF_LOG_DEBUG("Batch size output '%s' = %u",
                  description.name.c_str(), output_buf->getBatchSize());
    GXF_LOG_DEBUG("Raw Outputs MemType type = %u", output_buf->getBufDesc().memType);

    auto tensor = maybe_output_entity.value().add<nvidia::gxf::Tensor>(description.name.c_str());
    if (!tensor) {
      GXF_LOG_ERROR("Unable to add tensor '%s' to output", description.name.c_str());
      return nvidia::gxf::Unexpected{tensor.error()};
    }
    std::array<int32_t, nvidia::gxf::Shape::kMaxRank> dims;
    dims[0] = output_buf->getBatchSize();
    size_t gxf_dims_index = 1;
    uint32_t rank = 1;

    // If the model is non-dynamic, then batch size = 0. In this case, we need to ignore
    // that dimension and override it with a meaningful dimension from DS's dimension. Reset rank
    // to 0 as well to ignore previously set dimension.
    if (dims[0] == 0) {
      gxf_dims_index = 0;
      rank = 0;
    }

    // Batch will be first index in outgoing GXF Tensor
    GXF_ASSERT_LE(description.dims.numDims + 1, nvidia::gxf::Shape::kMaxRank);
    for (size_t nvsd_dim_index = 0; nvsd_dim_index < description.dims.numDims &&
      gxf_dims_index < nvidia::gxf::Shape::kMaxRank; nvsd_dim_index++, gxf_dims_index++) {
      dims[gxf_dims_index] = static_cast<int32_t>(description.dims.d[nvsd_dim_index]);
      rank++;
    }

    nvidia::gxf::Shape ds_tensor_shape {dims, rank};
    uint64_t bytes_per_element = description.elementSize;

    if (description.dataType == nvdsinferserver::InferDataType::kString) {
      GXF_LOG_DEBUG("Found output type of data type String!");
      // The shape returned by the inference for a string will be the shape of the unserialized
      // data, which we preserve by adding to another component on the published entity.
      auto maybe_output_shape = maybe_output_entity.value().add<nvidia::gxf::Shape>(
        description.name.c_str());
      if (!maybe_output_shape) {
        GXF_LOG_ERROR("Unable to add Shape '%s' to output", description.name.c_str());
        return nvidia::gxf::Unexpected{tensor.error()};
      }
      *(maybe_output_shape.value().get()) = ds_tensor_shape;

      // Batch dimension does not matter here since serialization of strings is unique; it
      // should only be interpreted with helper functions. We override the shape that is used for
      // the outgoing tensor since we need to represent the fully serialized byte size.
      ds_tensor_shape = nvidia::gxf::Shape{
        static_cast<int32_t>(output_buf->getTotalBytes())};
      bytes_per_element = 1;  // sizeof(char)
    }

    nvidia::gxf::MemoryStorageType target_storage_type =
      NvDsMemTypeToGxfMemType(description.memType);
    void* buffer_pointer = output_buf->getBufPtr(0);

    // convert output tensor to requested storage type
    // if not specified in the config default,
    // do not copy (ie. take whatever Triton gives as output)
    bool needs_memory = false;
    auto memcpy_kind = cudaMemcpyDefault;
    if (output_storage_type_.try_get()) {
      target_storage_type = nvidia::gxf::MemoryStorageType(output_storage_type_.try_get().value());
      auto current_storage_type = NvDsMemTypeToGxfMemType(description.memType);
      if (target_storage_type != current_storage_type) {
        switch (current_storage_type) {
          case nvidia::gxf::MemoryStorageType::kHost: {
            switch (target_storage_type) {
              case nvidia::gxf::MemoryStorageType::kDevice: {
                needs_memory = true;
                memcpy_kind = cudaMemcpyHostToDevice;
              } break;
              case nvidia::gxf:: MemoryStorageType::kSystem: {
                needs_memory = true;
                memcpy_kind = cudaMemcpyHostToHost;
              } break;
              default:
                GXF_LOG_ERROR("Unknown target storage type '%s' for copy", target_storage_type);
                return nvidia::gxf::Unexpected{GXF_FAILURE};
            }
          } break;
          case nvidia::gxf::MemoryStorageType::kDevice: {
            switch (target_storage_type) {
              case nvidia::gxf::MemoryStorageType::kHost: {
                needs_memory = true;
                memcpy_kind = cudaMemcpyDeviceToHost;
              } break;
              case nvidia::gxf:: MemoryStorageType::kSystem: {
                needs_memory = true;
                memcpy_kind = cudaMemcpyDeviceToHost;
              } break;
              default:
                GXF_LOG_ERROR("Unknown target storage type '%s' for copy", target_storage_type);
                return nvidia::gxf::Unexpected{GXF_FAILURE};
            }
          } break;
          case nvidia::gxf:: MemoryStorageType::kSystem: {
            switch (target_storage_type) {
              case nvidia::gxf::MemoryStorageType::kHost: {
                memcpy_kind = cudaMemcpyHostToHost;
              } break;
              case nvidia::gxf::MemoryStorageType::kDevice: {
                memcpy_kind = cudaMemcpyHostToDevice;
              } break;
              default:
                GXF_LOG_ERROR("Unknown target storage type '%s' for copy", target_storage_type);
                return nvidia::gxf::Unexpected{GXF_FAILURE};
            }
          } break;
          default:
            GXF_LOG_ERROR("Unknown current storage type '%s' for copy", current_storage_type);
            return nvidia::gxf::Unexpected{GXF_FAILURE};
        }
      }

      // Allocate memory if needed
      if (needs_memory) {
        auto pool =  allocator_.try_get();
        if (!pool) {
          GXF_LOG_ERROR("Allocator must be set when the requested output storage type"
                        " does not match triton output");
          return nvidia::gxf::Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
        }

        auto result = tensor.value()->reshapeCustom(ds_tensor_shape,
                                            NvDsToGxfDataType(description.dataType),
                                            bytes_per_element,
                                            nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                            target_storage_type, pool.value());
        if (!result) { return nvidia::gxf::ForwardError(result); }
        buffer_pointer = static_cast<void*>(tensor.value()->pointer());
      }

      // Perform the datacopy
      const cudaError_t copy_error = cudaMemcpy(buffer_pointer,
                                                output_buf->getBufPtr(0),
                                                output_buf->getTotalBytes(),
                                                memcpy_kind);
      if (copy_error != cudaSuccess) {
        GXF_LOG_ERROR("cudaMemcpy error: %s \n", cudaGetErrorString(copy_error));
        return nvidia::gxf::Unexpected{GXF_FAILURE};
      }
    }

    // If memory was not allocated by this inferencer, then wrap incoming memory
    // from triton context
    if (!needs_memory) {
    // Pass NvDs output by value to copy underlying shared_ptr. Once each of the outputs GXF tensors
    // reach 0 ref count, the underlying NvDs ptr will also be decremented.
      auto result = tensor.value()->wrapMemory(
      ds_tensor_shape, NvDsToGxfDataType(description.dataType), bytes_per_element,
      nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
      target_storage_type,
      buffer_pointer,
      [nvds_output] (void *) {
        return nvidia::gxf::Success;
      });

      if (!result) {
        GXF_LOG_ERROR("Unable to reshape tensor '%s' to output", description.name.c_str());
        return nvidia::gxf::Unexpected{result.error()};
      }
    }

    GXF_ASSERT(output_buf->getTotalBytes() == tensor.value()->size(),
      "Mismatch in expected GXF Tensor byte size: %" PRIu64 " != %zu",
      output_buf->getTotalBytes(), tensor.value()->size());
  }

  // Shared instances of NvDs output will be managed through callbacks with wrapMemory
  inference->raw_output = nvdsinferserver::SharedIBatchArray();
  // Reset for next inference
  inference->is_complete = false;
  inference->is_active = false;

  // We have processed this inference, so decrement this count, and check if all responses have
  // been processed.
  incomplete_inference_count_--;
  GXF_LOG_DEBUG("incomplete_inference_count_ = %zu", incomplete_inference_count_.load());
  if (!incomplete_inference_count_.load()) {
      GXF_LOG_DEBUG("Last inference reached; setting Async state to WAIT");
      scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);
  }

  return maybe_output_entity;
}

nvidia::gxf::Expected<bool> TritonInferencerImpl::isAcceptingRequest() {
  if (!inference_pool_.size()) { return false; }

  // If the next inference context in the pool is not active, then another inference request can
  // be stored there
  const auto& inference = inference_pool_[next_inference_index_.load()];
  return !inference->is_active;
}

}  // namespace triton
}  // namespace nvidia
