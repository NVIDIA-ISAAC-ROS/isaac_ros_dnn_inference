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

#include "gxf/core/component.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/extension_factory_helper.hpp"
#include "gxf/std/scheduling_term.hpp"

#include "inferencers/triton_inferencer_impl.hpp"
#include "inferencers/triton_inferencer_interface.hpp"

#include "triton_inference_request.hpp"
#include "triton_inference_response.hpp"
#include "triton_options.hpp"
#include "triton_scheduling_terms.hpp"
#include "triton_server.hpp"


GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0xa3c95d1cc06c4a4e, 0xa2f98d9078ab645c,
                          "NvTritonExt",
                          "Nvidia Triton Inferencing and Utilities Extension: 2.26.0 (x86_64), "
                          "2.30.0 (L4T - Jetpack 5.1)",
                          "NVIDIA",
                          "0.1.0", "LICENSE");
  GXF_EXT_FACTORY_ADD(0x26228984ffc44162, 0x9af56e3008aa2982,
                      nvidia::triton::TritonServer,
                      nvidia::gxf::Component,
                      "Triton Server Component for Direct Inference.");
  GXF_EXT_FACTORY_ADD(0x1661c0156b1c422d, 0xa6f0248cdc197b1a,
                      nvidia::triton::TritonInferencerInterface,
                      nvidia::gxf::Component,
                      "Triton Inferencer Interface where specific Direct, Remote "
                      "or IPC inferencers can implement.");
  GXF_EXT_FACTORY_ADD(0xb84cf267b2234df5, 0xac82752d9fae1014,
                      nvidia::triton::TritonInferencerImpl,
                      nvidia::triton::TritonInferencerInterface,
                      "Triton Inferencer that uses the Triton C API. Requires "
                      "Triton Server Component.");
  GXF_EXT_FACTORY_ADD(0x34395920232c446f, 0xb5b746f642ce84df,
                      nvidia::triton::TritonInferenceRequest,
                      nvidia::gxf::Codelet,
                      "Triton Inference Request Codelet that wraps Triton Implementation.");
  GXF_EXT_FACTORY_ADD(0x4dd957a7aa554117, 0x90d39a98e31ee176,
                      nvidia::triton::TritonInferenceResponse,
                      nvidia::gxf::Codelet,
                      "Triton Inference Response Codelet that wraps Triton Implementation.");
  GXF_EXT_FACTORY_ADD_0(0x087696ed229d4199, 0x876f05b92d3887f0,
                      nvidia::triton::TritonOptions,
                      "Triton Inference Options for model control and sequence control.");
  GXF_EXT_FACTORY_ADD(0xf860241212424e43, 0x9dbf9c559d496b84,
                      nvidia::triton::TritonRequestReceptiveSchedulingTerm,
                      nvidia::gxf::SchedulingTerm,
                      "Triton Scheduling Term that schedules Request Codelet when the inferencer "
                      "can accept a new request.");
GXF_EXT_FACTORY_END()
