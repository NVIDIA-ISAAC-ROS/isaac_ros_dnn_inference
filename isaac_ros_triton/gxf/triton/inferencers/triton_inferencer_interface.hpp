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

#ifndef NVIDIA_TRITON_INFERENCERS_TRITON_INFERENCER_INTERFACE_HPP
#define NVIDIA_TRITON_INFERENCERS_TRITON_INFERENCER_INTERFACE_HPP

#include <string>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"

namespace nvidia {
namespace triton {

/**
 * @brief Interface to wrap implementation of Triton inferencing.
 *
 */
class TritonInferencerInterface : public nvidia::gxf::Component {
 public:
  /**
   * @brief Prepare and set up any members specific to implementation.
   *
   * @details Derived component may prepare any implementation specific
   * members/details here. We cannot leverage initialize() due to lack
   * of guarantees on other component initializations.
   *
   * @return gxf_result_t
   */
  virtual gxf_result_t construct() = 0;

  /**
   * @brief Destroy any members specific to implementation.
   *
   * @details Derived component may prepare any implementation specific
   * members/details here. We cannot leverage deinitialize() due to lack
   * of guarantees on other component initializations.
   *
   * @return gxf_result_t
   */
  virtual gxf_result_t destruct() = 0;

  /**
   * @brief Dispatch Triton inference request asynchronously.
   *
   * @param[in] tensors Entity that contains a tensor map with names
   * corresponding to Triton model inputs
   *
   * @return gxf_result_t
   */
  virtual gxf_result_t inferAsync(const std::vector<nvidia::gxf::Entity> input_entities,
                                  const std::vector<std::string> input_names) = 0;

  /**
   * @brief Get the Triton Response after an inference completes.
   *
   * @return nvidia::gxf::Expected<Entity>
   */
  virtual nvidia::gxf::Expected<nvidia::gxf::Entity> getResponse() = 0;

  /**
   * @brief Checks if inferencer can accept a new inference request.
   *
   * @details This will be leveraged by scheduling term that decides to
   * schedule the request codelet.
   * This allows for inferSync behavior depending upon inferencer's implementation.
   *
   * @return nvidia::gxf::Expected<bool>
   */
  virtual nvidia::gxf::Expected<bool> isAcceptingRequest() = 0;
};

}  // namespace triton
}  // namespace nvidia

#endif
