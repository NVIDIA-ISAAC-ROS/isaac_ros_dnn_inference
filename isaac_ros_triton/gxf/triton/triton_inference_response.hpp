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

#ifndef NVIDIA_TRITON_TRITON_INFERENCE_RESPONSE_HPP
#define NVIDIA_TRITON_TRITON_INFERENCE_RESPONSE_HPP

#include <string>
#include <vector>

#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

#include "inferencers/triton_inferencer_interface.hpp"

namespace nvidia {
namespace triton {

/**
 * @brief Triton Inference Response that wraps generic TritonInferencer implementation.
 *
 * @details The Entity which holds this Codelet must also have TritonModelOutput(s).
 *
 */
class TritonInferenceResponse : public nvidia::gxf::Codelet {
 public:
  /**
   * @brief Register Parameters.
   *
   * @param registrar
   * @return gxf_result_t
   */
  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override {
    nvidia::gxf::Expected<void> result;

    result &= registrar->parameter(inferencer_, "inferencer",
      "Inferencer Implementation",
      "TritonInferenceInterface Inferencer Implementation Handle");
    result &= registrar->parameter(output_tensor_names_, "output_tensor_names",
      "Output Tensor Names",
      "Names of output tensors in the order to be retrieved from the model.");
    result &= registrar->parameter(output_binding_names_, "output_binding_names",
      "Output Binding Names",
      "Names of output bindings in the model in the same "
      "order of of what is provided in output_tensor_names.");
    result &= registrar->parameter(tx_, "tx", "TX", "Transmitter to publish output tensors");
    return nvidia::gxf::ToResultCode(result);
  }

  /**
   * @brief Return success.
   *
   * @return gxf_result_t
   */
  gxf_result_t start() override;

  /**
   * @brief Gets Response from Inferencer and transmits output tensors respectively
   * to TritonModelOutput(s) Transmitters
   *
   * @return gxf_result_t
   */
  gxf_result_t tick() override;

  /**
   * @brief Return success.
   *
   * @return gxf_result_t
   */
  gxf_result_t stop() override;

 private:
  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::triton::TritonInferencerInterface>>
    inferencer_;

  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::gxf::Transmitter>> tx_;
  nvidia::gxf::Parameter<std::vector<std::string>> output_tensor_names_;
  nvidia::gxf::Parameter<std::vector<std::string>> output_binding_names_;
};

}  // namespace triton
}  // namespace nvidia

#endif
