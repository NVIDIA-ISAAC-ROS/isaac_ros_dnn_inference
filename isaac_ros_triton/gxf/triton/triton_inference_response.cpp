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

#include <utility>

#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

#include "triton_inference_response.hpp"
#include "triton_options.hpp"

namespace nvidia {
namespace triton {

gxf_result_t TritonInferenceResponse::start() {
  if (!inferencer_.get()) {
      GXF_LOG_ERROR("Inferencer unavailable");
      return GXF_FAILURE;
  }
  if (output_tensor_names_.get().size() == 0) {
    GXF_LOG_ERROR("At least one output tensor is needed.");
    return GXF_FAILURE;
  }
  if (output_tensor_names_.get().size() != output_binding_names_.get().size()) {
    GXF_LOG_ERROR("Mismatching number of output tensor names and bindings: %lu vs %lu",
      output_tensor_names_.get().size(), output_binding_names_.get().size());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t TritonInferenceResponse::tick() {
  // This getResponse() call is expected to be a blocking statement.
  auto maybe_response = inferencer_.get()->getResponse();
  if (!maybe_response) {
    return nvidia::gxf::ToResultCode(maybe_response);
  }

  // Create a new entity for model output which will hold a Tensor Map
  auto maybe_output_tensor_map = nvidia::gxf::Entity::New(context());
  if (!maybe_output_tensor_map) {
    return nvidia::gxf::ToResultCode(maybe_output_tensor_map);
  }

  auto& bindings = output_binding_names_.get();
  auto& tensors = output_tensor_names_.get();

  // Implementation will return a tensor map with Triton Bindings. We need to translate that to the
  // expected GXF Tensor names.
  for (size_t output_index = 0; output_index < bindings.size(); output_index++) {
    auto& tensor_name = tensors[output_index];
    auto maybe_tensor = maybe_output_tensor_map.value().add<nvidia::gxf::Tensor>(
      tensor_name.c_str());
    if (!maybe_tensor) {
      return nvidia::gxf::ToResultCode(maybe_tensor);
    }

    auto& triton_binding = bindings[output_index];
    auto maybe_response_tensor = maybe_response.value().get<nvidia::gxf::Tensor>(
        triton_binding.c_str());
    if (!maybe_response_tensor) {
      GXF_LOG_ERROR("Unable to find tensor response for binding '%s'", triton_binding.c_str());
      return nvidia::gxf::ToResultCode(maybe_response_tensor);
    }

    // Move incoming response tensor to the tensor that will be transmitted. There is no better way
    // of redistributing tensors to varying sources with copy.
    *(maybe_tensor.value().get()) = std::move(*(maybe_response_tensor.value().get()));

    // For String data, we need to publish nvidia::gxf::Shape so the serialized data can be
    // interpreted correctly.
    auto maybe_response_tensor_shape = maybe_response.value().get<nvidia::gxf::Shape>(
        triton_binding.c_str());
    if (maybe_response_tensor_shape) {
      auto maybe_shape = maybe_output_tensor_map.value().add<nvidia::gxf::Shape>(
        tensor_name.c_str());
      if (!maybe_shape) {
        return nvidia::gxf::ToResultCode(maybe_shape);
      }

      *(maybe_shape.value().get()) = std::move(*(maybe_response_tensor_shape.value().get()));
    }
  }

  // Forward Triton Options so that consumer understands sequence id, end of sequence, etc.
  auto maybe_input_triton_option = maybe_response.value().get<nvidia::triton::TritonOptions>();
  if (maybe_input_triton_option) {
    auto maybe_output_triton_option =
      maybe_output_tensor_map.value().add<nvidia::triton::TritonOptions>();
    if (!maybe_output_triton_option) {
      return nvidia::gxf::ToResultCode(maybe_output_triton_option);
    }
    // Move incoming TritonOption from receiver to the outgoing tensor.
    // NOTE: This modifies the incoming Entity tensor component via the move.
    *(maybe_output_triton_option.value().get()) =
      std::move(*(maybe_input_triton_option.value().get()));
  }

  nvidia::gxf::Expected<void> result = nvidia::gxf::Unexpected{GXF_FAILURE};

  auto maybe_timestamp = maybe_response.value().get<nvidia::gxf::Timestamp>("timestamp");
  if (!maybe_timestamp) {
    result = tx_.get()->publish(maybe_output_tensor_map.value());
  } else {
    result = tx_.get()->publish(maybe_output_tensor_map.value(), maybe_timestamp.value()->acqtime);
  }

  if (!result) {
    GXF_LOG_ERROR("Error when transmitting message output tensor map");
    return nvidia::gxf::ToResultCode(result);
  }

  return GXF_SUCCESS;
}

gxf_result_t TritonInferenceResponse::stop() {
  return GXF_SUCCESS;
}

}  // namespace triton
}  // namespace nvidia
