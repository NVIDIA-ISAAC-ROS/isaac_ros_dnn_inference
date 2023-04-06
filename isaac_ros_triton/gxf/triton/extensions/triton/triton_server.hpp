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

#ifndef NVIDIA_TRITON_TRITON_SERVER_HPP
#define NVIDIA_TRITON_TRITON_SERVER_HPP

#include <memory>
#include <string>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/handle.hpp"

#include "infer_icontext.h"
#include "nvdsinferserver_config.pb.h"

namespace nvidia {
namespace triton {

class TritonServer : public nvidia::gxf::Component {
 public:
  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override {
      nvidia::gxf::Expected<void> result;

      result &= registrar->parameter(log_level_, "log_level",
        "Triton Logging Level",
        "Set verbose logging level. 0 = Error, 1 = Warn, 2 = Info, 3+ = Verbose", 1U);

      result &= registrar->parameter(enable_strict_model_config_,
        "enable_strict_model_config",
        "Enable Strict Model Configuration",
        "Enable Strict Model Configuration to enforce presence of config. "
        "If disabled, TensorRT, TensorFlow saved-model, and ONNX models do "
        "not require a model configuration file because Triton can derive "
        "all the required settings automatically", true);

      result &= registrar->parameter(min_compute_capability_,
        "min_compute_capability",
        "Minimum Compute Capability",
        "Minimum Compute Capability for GPU. "
        "Refer to https://developer.nvidia.com/cuda-gpus", 6.0);

      result &= registrar->parameter(model_repository_paths_,
        "model_repository_paths",
        "List of Triton Model Repository Paths",
        "List of Triton Model Repository Paths. Refer to "
        "https://github.com/bytedance/triton-inference-server/blob/master/docs/"
        "model_repository.md");

      result &= registrar->parameter(tf_gpu_memory_fraction_,
        "tf_gpu_memory_fraction",
        "Tensorflow GPU Memory Fraction",
        "The portion of GPU memory to be reserved for TensorFlow Models.", 0.0);

      result &= registrar->parameter(tf_disable_soft_placement_,
        "tf_disable_soft_placement",
        "Tensorflow will use CPU operation when GPU implementation is not available",
        "Tensorflow will use CPU operation when GPU implementation is not available", true);

      result &= registrar->parameter(backend_directory_path_,
        "backend_directory_path",
        "Path to Triton Backend Directory",
        "Path to Triton Backend Directory", std::string(""));

      result &= registrar->parameter(model_control_mode_,
        "model_control_mode",
        "Triton Model Control Mode",
        "Triton Model Control Mode. 'none' will load all models at startup. 'explicit' "
        "will allow load of models when needed. Unloading is unsupported", std::string("explicit"));

      result &= registrar->parameter(backend_configs_,
        "backend_configs",
        "Triton Backend Configurations",
        "Triton Backend Configurations in format: 'backend,setting=value'. "
        "Refer to Backend specific documentation: "
        "https://github.com/triton-inference-server/tensorflow_backend#command-line-options, "
        "https://github.com/triton-inference-server/python_backend#managing-shared-memory",
        nvidia::gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

      return nvidia::gxf::ToResultCode(result);
  }

  /**
   * @brief Create Triton Server via nvdsinferserver::ITritonServerInstance with parameters.
   *
   * @details Create Shared instance with destructor.
   *
   * @return gxf_result_t
   */
  gxf_result_t initialize() override;


  /**
   * @brief Get the shared instance of nvdsinferserver::ITritonServerInstance.
   *
   * @details Shared ownership is necessary for proper deinitialization of the underlying Triton
   * server since GXF lacks guarantees on deinitialize() ordering across multiple entities.
   *
   * @return nvidia::gxf::Expected<std::shared_ptr<nvdsinferserver::ITritonServerInstance>>
   */
  nvidia::gxf::Expected<std::shared_ptr<nvdsinferserver::ITritonServerInstance>> getServer();

  /**
   * @brief Get the shared instance of config::TritonModelRepo.
   *
   * @details Shared ownership is necessary for proper deinitialization of the underlying Triton
   * server since GXF lacks guarantees on deinitialize() ordering across multiple entities.
   *
   * @return nvidia::gxf::Expected<std::shared_ptr<config::TritonModelRepo>>
   */
  nvidia::gxf::Expected<std::shared_ptr<nvdsinferserver::config::TritonModelRepo>>
  getModelRepoConfig();

 private:
  // Parameters supported by nvdsinferserver::config::TritonModelRepo
  nvidia::gxf::Parameter<uint32_t> log_level_;
  nvidia::gxf::Parameter<bool> enable_strict_model_config_;
  nvidia::gxf::Parameter<double> min_compute_capability_;
  nvidia::gxf::Parameter<std::vector<std::string>> model_repository_paths_;

  nvidia::gxf::Parameter<double> tf_gpu_memory_fraction_;
  nvidia::gxf::Parameter<bool> tf_disable_soft_placement_;

  nvidia::gxf::Parameter<std::string> backend_directory_path_;
  nvidia::gxf::Parameter<std::string> model_control_mode_;

  nvidia::gxf::Parameter<std::vector<std::string>> backend_configs_;


  // Shared instance is needed for proper deinitialize since this will be needed for each inference
  // request.
  std::shared_ptr<nvdsinferserver::ITritonServerInstance> tritonInstance_;

  // Shared instance is needed for constructing the inference config
  std::shared_ptr<nvdsinferserver::config::TritonModelRepo> tritonRepoConfig_;
};


}  // namespace triton
}  // namespace nvidia

#endif
