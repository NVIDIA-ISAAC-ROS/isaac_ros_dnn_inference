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

#ifndef NVIDIA_TRITON_INFERENCERS_TRITON_INFERENCER_IMPL_HPP
#define NVIDIA_TRITON_INFERENCERS_TRITON_INFERENCER_IMPL_HPP

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"

#include "triton_inferencer_interface.hpp"

#include "extensions/triton/triton_server.hpp"


#include "cuda_runtime.h"


namespace nvidia {
namespace triton {

/**
 * @brief Struct to maintain members for Inference, such as input, outputs, status
 *
 * @details This maintains input Entity and raw output from the inference response. This class
 * is intended to ease the use of an inference request pool.
 *
 */
struct Inference;

/**
 * @brief Enumeration for inference mode
 *
 */
enum struct TritonInferenceMode {
  kDirect = 0,
  kRemoteGrpc = 1,
};

/**
 * @brief Triton Direct C API Implementation for inferencing.
 *
 */
class TritonInferencerImpl : public nvidia::triton::TritonInferencerInterface {
 public:
  /**
   * @brief Register parameters for usage with this component.
   *
   * @param registrar
   * @return gxf_result_t
   */
  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override {
      nvidia::gxf::Expected<void> result;

    result &= registrar->parameter(server_handle_, "server",
      "Triton Server",
      "Triton Server Handle",
      nvidia::gxf::Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);

    result &= registrar->parameter(model_name_, "model_name",
      "Triton Model Name",
      "Triton Model Name. Refer to Triton Model Repository.");

    result &= registrar->parameter(model_version_, "model_version",
      "Triton Model Version",
      "Triton Model Version. Refer to Triton Model Repository.");

    result &= registrar->parameter(max_batch_size_, "max_batch_size",
      "Triton Max Batch Size for Model",
      "Triton Max Batch Size for Model, which should match Triton Model Repository.");

    result &= registrar->parameter(num_concurrent_requests_,
      "num_concurrent_requests",
      "Maximum Number of concurrent Inference Requests",
      "Maximum Number of concurrent Inference Requests, which defines the pool.",
      1U);

    result &= registrar->parameter(allocator_,
      "allocator",
      "Allocator",
      "Allocator instance for output tensors.",
      nvidia::gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

    result &= registrar->parameter(output_storage_type_,
      "output_storage_type",
      "Specified output memory location: kHost, kDevice, kSystem" \
      "The memory storage type used by this allocator. ",
      "Can be kHost (0), kDevice (1) or kSystem (2)",
      nvidia::gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

    result &= registrar->parameter(use_string_data_,
      "use_string_data",
      "Specify whether string data is being sent to Triton",
      "Specify whether string data is being sent to Triton",
      false);

    result &= registrar->parameter(use_sequence_data_,
      "use_sequence_data",
      "Specify whether sequence data is being sent to Triton",
      "Specify whether sequence data is being sent to Triton",
      false);

    result &= registrar->parameter(scheduling_term_, "async_scheduling_term",
      "Asynchronous Scheduling Term", "Asynchronous Scheduling Term");

    result &= registrar->parameter(inference_mode_,
      "inference_mode",
      "Triton Inference mode: Direct, RemoteGrpc",
      "Triton Inference mode: Direct, RemoteGrpc");

    result &= registrar->parameter(server_endpoint_,
      "server_endpoint",
      "Triton Server Endpoint for GRPC or HTTP",
      "Triton Server Endpoint for GRPC or HTTP",
      nvidia::gxf::Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);

    return nvidia::gxf::ToResultCode(result);
  }

  gxf_result_t initialize() override;

  /**
   * @brief Allocate Triton ResponseAllocator. Reserve space for Inference Pool.
   * Create Inference Request Pool
   *
   * @return gxf_result_t
   */
  gxf_result_t construct() override;

  /**
   * @brief Deallocate Triton ResponseAllocator. Clear Inference Pool.
   *
   * @return gxf_result_t
   */
  gxf_result_t destruct() override;

  /**
   * @brief Dispatch Triton inference request asynchronously.
   *
   * @param[in] input_entities Vector of input entities that contain the tensor data that
   * correspond to Triton model inputs
   *
   * @param[in] input_names Vector of name strings for the tensors that
   * correspond to Triton model inputs
   *
   * @return gxf_result_t
   */
  gxf_result_t inferAsync(const std::vector<nvidia::gxf::Entity> input_entities,
                          const std::vector<std::string> input_names)
                          override;

  /**
   * @brief Get the Triton Response after an inference completes.
   *
   * @return nvidia::gxf::Expected<Entity>
   */
  nvidia::gxf::Expected<nvidia::gxf::Entity> getResponse() override;

  /**
   * @brief Checks if inferencer can accept a new inference request.
   *
   * @return nvidia::gxf::Expected<bool>
   */
  nvidia::gxf::Expected<bool> isAcceptingRequest() override;

 private:
  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::triton::TritonServer>> server_handle_;
  nvidia::gxf::Parameter<std::string> model_name_;
  nvidia::gxf::Parameter<int64_t> model_version_;
  nvidia::gxf::Parameter<uint32_t> max_batch_size_;
  nvidia::gxf::Parameter<uint32_t> num_concurrent_requests_;
  nvidia::gxf::Parameter<TritonInferenceMode> inference_mode_;
  nvidia::gxf::Parameter<std::string> server_endpoint_;

  // Special cases that aren't fully supported by the TRTIS backend yet
  // These will need to use the simple case
  nvidia::gxf::Parameter<bool> use_string_data_;
  nvidia::gxf::Parameter<bool> use_sequence_data_;

  // Specify the output storage type
  nvidia::gxf::Parameter<int32_t> output_storage_type_;
  // Specify allocator incase memory needs to be allocated when the requested output storage type
  // does not match the output memory type of the triton context
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;

  // Async Scheduling Term required to get/set event state.
  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::gxf::AsynchronousSchedulingTerm>>
    scheduling_term_;

  // Use a shared pointer to the server due to lack of guarantees on deinitialization order with
  // Server component.
  std::shared_ptr<nvdsinferserver::ITritonServerInstance> server_;

  // Instance of IInferContext can be used across multiple inferences of the same model.
  std::shared_ptr<nvdsinferserver::IInferContext> infer_context_;

  // Set up a pool of inferences that manage Tensor inputs and response promises. The size of the
  // pool must be large enough to accomodate multiple asynchronous requests, and it is controlled
  // via parameter interface.
  std::vector<Inference*> inference_pool_;

  // Mutex to protect counting in inference pool
  std::mutex mutex_;

  // This represents the currently active index for the inference pool. Responses will use this
  // index for the promises and future events.
  size_t active_inference_index_ { 0 };

  // This represents the next available index in the inference pool. This can be asynchronously
  // accessed via isAcceptingRequest().
  std::atomic<size_t> next_inference_index_ { 0 };

  // This represents the number of incompleted inferences for the async
  // continuation/termination conditions. This is incremented in a callback, and decremented when
  // the inference response is received.
  std::atomic<size_t> incomplete_inference_count_ { 0 };
};

}  // namespace triton
}  // namespace nvidia

namespace nvidia {
namespace gxf {

/**
 * @brief Custom parameter parser for TritonInferenceMode
 *
 */
template <>
struct ParameterParser<::nvidia::triton::TritonInferenceMode> {
  static Expected<::nvidia::triton::TritonInferenceMode> Parse(
      gxf_context_t context, gxf_uid_t component_uid,
      const char* key, const YAML::Node& node,
      const std::string& prefix) {
    const std::string value = node.as<std::string>();
    if (strcmp(value.c_str(), "Direct") == 0) {
      return ::nvidia::triton::TritonInferenceMode::kDirect;
    }
    if (strcmp(value.c_str(), "RemoteGrpc") == 0) {
      return ::nvidia::triton::TritonInferenceMode::kRemoteGrpc;
    }
    return ::nvidia::gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
  }
};

template<>
struct ParameterWrapper<::nvidia::triton::TritonInferenceMode> {
  static Expected<YAML::Node> Wrap(
    gxf_context_t context,
    const ::nvidia::triton::TritonInferenceMode& value) {
      std::string string_value;
      if (value == ::nvidia::triton::TritonInferenceMode::kDirect) {
        string_value = "Direct";
      } else if (value == ::nvidia::triton::TritonInferenceMode::kRemoteGrpc) {
        string_value = "RemoteGrpc";
      } else {
        return ::nvidia::gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
      }
    return ParameterWrapper<std::string>::Wrap(context, string_value);
  }
};

}  // namespace gxf
}  // namespace nvidia

#endif
