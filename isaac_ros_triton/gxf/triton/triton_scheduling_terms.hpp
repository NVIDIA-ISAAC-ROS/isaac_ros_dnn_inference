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

#ifndef NVIDIA_TRITON_TRITON_SCHEDULING_TERMS_HPP
#define NVIDIA_TRITON_TRITON_SCHEDULING_TERMS_HPP

#include <string>
#include <utility>
#include <vector>

#include "gxf/core/handle.hpp"
#include "gxf/std/scheduling_term.hpp"

#include "inferencers/triton_inferencer_interface.hpp"

namespace nvidia {
namespace triton {

/**
 * @brief Scheduling term which permits execution only when the Triton Request Component is able to
 * accept new requests.
 *
 */
class TritonRequestReceptiveSchedulingTerm : public nvidia::gxf::SchedulingTerm {
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
    return nvidia::gxf::ToResultCode(result);
  }

  /**
   * @brief Returns success
   *
   * @return gxf_result_t
   */
  gxf_result_t initialize() override;

  /**
   * @brief Only when an inferencer can accept new inference request, SchedulingCondition is
   * Ready
   *
   * @param[in] timestamp Unused
   * @param[out] type Ready (if inferencer can accept new inference request) or Wait (otherwise)
   * @param[out] target_timestamp Unmodified
   * @return gxf_result_t
   */
  gxf_result_t check_abi(int64_t timestamp, nvidia::gxf::SchedulingConditionType* type,
    int64_t* target_timestamp) const override;

  /**
   * @brief Returns success
   *
   * @param dt Unused
   * @return gxf_result_t
   */
  gxf_result_t onExecute_abi(int64_t dt) override;

 private:
  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::triton::TritonInferencerInterface>>
    inferencer_;
};

}  // namespace triton
}  // namespace nvidia

#endif
