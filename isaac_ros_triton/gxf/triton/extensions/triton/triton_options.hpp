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

#ifndef NVIDIA_TRITON_TRITON_OPTIONS_HPP
#define NVIDIA_TRITON_TRITON_OPTIONS_HPP

namespace nvidia {
namespace triton {

/**
 * @brief Triton Inference Options for model control and sequence control
 *
 */
struct TritonOptions {
  uint64_t sequence_id;  // Should be non-zero because zero is reserved for non-sequence requests.
  bool start;
  bool end;
  uint64_t priority;
  uint64_t timeout;
};

}  // namespace triton
}  // namespace nvidia

#endif
