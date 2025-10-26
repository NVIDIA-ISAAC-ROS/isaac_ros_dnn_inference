// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/tensor_rt/tensor_rt_inference.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xd43f23e4b9bf11eb, 0x9d182b7be630552b, "TensorRTExtension", "TensorRT",
                         "Nvidia", "2.8.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x06a7f0e0b9c011eb, 0x8cd623c9c2070107, nvidia::gxf::TensorRtInference,
                    nvidia::gxf::Codelet,
                    "Codelet taking input tensors and feed them into TensorRT for inference.");

GXF_EXT_FACTORY_END()

}  // extern "C"
