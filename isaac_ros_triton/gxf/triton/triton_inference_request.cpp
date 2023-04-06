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

#include <string>
#include <utility>
#include <vector>

#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

#include "triton_inference_request.hpp"
#include "triton_options.hpp"

namespace nvidia {
namespace triton {

gxf_result_t TritonInferenceRequest::start() {
  auto result = inferencer_.get()->construct();
  if (input_tensor_names_.get().size() == 0) {
    GXF_LOG_ERROR("At least one input tensor is needed.");
    return GXF_FAILURE;
  }
  if (input_tensor_names_.get().size() != input_binding_names_.get().size()) {
    GXF_LOG_ERROR("Mismatching number of input tensor names and bindings: %lu vs %lu",
      input_tensor_names_.get().size(), input_binding_names_.get().size());
    return GXF_FAILURE;
  }
  if (input_tensor_names_.get().size() != rx_.get().size()) {
    GXF_LOG_ERROR("Mismatching number of input tensor names and receivers: %lu vs %lu",
      input_tensor_names_.get().size(), rx_.get().size());
    return GXF_FAILURE;
  }
  if (rx_.get().size() == 0) {
    GXF_LOG_ERROR("At least one receiver is needed.");
    return GXF_FAILURE;
  }
  return result;
}

gxf_result_t TritonInferenceRequest::tick() {
  // Create a new entity that will serve as a tensor map for the model inputs
  auto inputs_tensor_map = nvidia::gxf::Entity::New(context());
  if (!inputs_tensor_map) {
    return nvidia::gxf::ToResultCode(inputs_tensor_map);
  }

  auto& receivers = rx_.get();
  auto& binding_names = input_binding_names_.get();
  auto& tensor_names = input_tensor_names_.get();

  std::vector<nvidia::gxf::Entity> input_entities;
  std::vector<std::string> input_names;

  auto maybe_output_timestamp = inputs_tensor_map.value().add<nvidia::gxf::Timestamp>("timestamp");
  if (!maybe_output_timestamp) {
    return nvidia::gxf::ToResultCode(maybe_output_timestamp);
  }

  for (size_t input_index = 0; input_index < receivers.size(); input_index++) {
    auto& receiver = receivers[input_index];
    auto maybe_message = receiver->receive();
    if (!maybe_message) {
      return nvidia::gxf::ToResultCode(maybe_message);
    }

    // ensure entity includes tensor as input
    auto& tensor_name = tensor_names[input_index];
    auto maybe_tensor_incoming = maybe_message.value().get<nvidia::gxf::Tensor>(
      tensor_name.c_str());
    if (!maybe_tensor_incoming) {
      GXF_LOG_ERROR("Unable to find Tensor with name '%s'", tensor_name.c_str());
      return nvidia::gxf::ToResultCode(maybe_tensor_incoming);
    }

    input_entities.push_back(maybe_message.value());
    input_names.push_back(binding_names[input_index]);
  }

  return inferencer_.get()->inferAsync(input_entities, input_names);
}

gxf_result_t TritonInferenceRequest::stop() {
  auto result = inferencer_.get()->destruct();
  return result;
}

}  // namespace triton
}  // namespace nvidia
