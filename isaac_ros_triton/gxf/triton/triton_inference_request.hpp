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

#ifndef NVIDIA_TRITON_TRITON_INFERENCE_REQUEST_HPP
#define NVIDIA_TRITON_TRITON_INFERENCE_REQUEST_HPP

#include <string>
#include <vector>

#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"

#include "inferencers/triton_inferencer_interface.hpp"

namespace nvidia {
namespace triton {

/**
 * @brief Triton Inference Request that wraps generic TritonInferencer implementation.
 *
 * @details The Entity which holds this Codelet must also have TritonModelInput(s).
 *
 */
class TritonInferenceRequest : public nvidia::gxf::Codelet {
 public:
  /**
   * @brief Register Parameters
   *
   * @param registrar
   * @return gxf_result_t
   */
  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override {
    nvidia::gxf::Expected<void> result;

    result &= registrar->parameter(inferencer_, "inferencer",
      "Inferencer Implementation",
      "TritonInferenceInterface Inferencer Implementation Handle");
    result &= registrar->parameter(rx_, "rx", "Receivers",
      "List of receivers to take input tensors");
    result &= registrar->parameter(input_tensor_names_, "input_tensor_names", "Input Tensor Names",
      "Names of input tensors that exist in the ordered receivers in 'rx'.");
    result &= registrar->parameter(input_binding_names_, "input_binding_names",
      "Input Triton Binding Names",
      "Names of input bindings corresponding to Triton's Config Inputs in the same order of "
      "what is provided in 'input_tensor_names'.");

    return nvidia::gxf::ToResultCode(result);
  }

  /**
   * @brief Prepare TritonInferencerInterface
   *
   * @return gxf_result_t
   */
  gxf_result_t start() override;

  /**
   * @brief Receive tensors of TritonModelInput(s), create Tensor Map, submit
   * inference request asynchronously.
   *
   * @return gxf_result_t
   */
  gxf_result_t tick() override;

  /**
   * @brief Destroys inferencer of type TritonInferencerInterface
   *
   * @return gxf_result_t
   */
  gxf_result_t stop() override;

 private:
  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::triton::TritonInferencerInterface>>
    inferencer_;

  nvidia::gxf::Parameter<std::vector<std::string>> input_tensor_names_;
  nvidia::gxf::Parameter<std::vector<std::string>> input_binding_names_;
  gxf::Parameter<std::vector<gxf::Handle<gxf::Receiver>>> rx_;
};

}  // namespace triton
}  // namespace nvidia

#endif
