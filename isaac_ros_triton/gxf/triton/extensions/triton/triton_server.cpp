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

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "common/logger.hpp"
#include "nvdsinferserver_config.pb.h"

#include "triton_server.hpp"

namespace nvidia {
namespace triton {

gxf_result_t TritonServer::initialize() {
  GXF_LOG_DEBUG("Initializing Triton Server...");

  nvdsinferserver::config::TritonModelRepo model_repo_config;

  model_repo_config.set_log_level(log_level_.get());
  model_repo_config.set_strict_model_config(enable_strict_model_config_.get());
  model_repo_config.set_min_compute_capacity(static_cast<float>(min_compute_capability_.get()));
  for (const auto& model_repository_path : model_repository_paths_.get()) {
    model_repo_config.add_root(model_repository_path);
  }
  model_repo_config.set_tf_gpu_memory_fraction(static_cast<float>(tf_gpu_memory_fraction_.get()));
  model_repo_config.set_tf_disable_soft_placement(tf_disable_soft_placement_.get());
  model_repo_config.set_backend_dir(backend_directory_path_.get());
  model_repo_config.set_model_control_mode(model_control_mode_.get());

  size_t num_backend_config = 0;
  const char delim_setting = ',';
  const char delim_value = '=';

  auto maybe_backend_configs = backend_configs_.try_get();
  if (maybe_backend_configs) {
    for (const auto& config : maybe_backend_configs.value()) {
      model_repo_config.add_backend_configs();
      auto proto_config = model_repo_config.mutable_backend_configs(num_backend_config++);

      size_t delim_setting_pos = config.find(delim_setting);
      if (delim_setting_pos == std::string::npos) {
        GXF_LOG_ERROR("Unable to find '%c' in backend config: %s", delim_setting, config.c_str());
        return GXF_FAILURE;
      }
      size_t delim_value_pos = config.find(delim_value, delim_setting_pos);
      if (delim_value_pos == std::string::npos) {
        GXF_LOG_ERROR("Unable to find '%c' in backend config: %s", delim_value, config.c_str());
        return GXF_FAILURE;
      }
      if (delim_setting_pos >= delim_value_pos) {
        GXF_LOG_ERROR("Delimeter '%c' must come before '%c' in backend config: %s",
          delim_setting, delim_value, config.c_str());
        return GXF_FAILURE;
      }

      const std::string backend_name = config.substr(0, delim_setting_pos);
      const std::string backend_setting = config.substr(delim_setting_pos + 1,
        delim_value_pos - delim_setting_pos - 1);
      const std::string backend_value = config.substr(delim_value_pos + 1);

      proto_config->set_backend(backend_name);
      proto_config->set_setting(backend_setting);
      proto_config->set_value(backend_value);
    }
  }

  tritonRepoConfig_ = std::make_shared<nvdsinferserver::config::TritonModelRepo>(model_repo_config);

  nvdsinferserver::ITritonServerInstance* server_ptr = nullptr;
  auto result = NvDsTritonServerInit(&server_ptr, model_repo_config.DebugString().c_str(),
    model_repo_config.DebugString().size());
  if (result != NvDsInferStatus::NVDSINFER_SUCCESS) {
    GXF_LOG_ERROR("Error in NvDsTritonServerInit");
    return GXF_FAILURE;
  }

  std::shared_ptr<nvdsinferserver::ITritonServerInstance> server(
    server_ptr, NvDsTritonServerDeinit);
  tritonInstance_ = std::move(server);
  GXF_LOG_DEBUG("Successfully initialized Triton Server...");
  return GXF_SUCCESS;
}

nvidia::gxf::Expected<std::shared_ptr<nvdsinferserver::ITritonServerInstance>>
TritonServer::getServer() {
  if (!tritonInstance_) {
    return nvidia::gxf::Unexpected{GXF_NULL_POINTER};
  }
  return tritonInstance_;
}

nvidia::gxf::Expected<std::shared_ptr<nvdsinferserver::config::TritonModelRepo>>
TritonServer::getModelRepoConfig() {
  if (!tritonRepoConfig_) {
    return nvidia::gxf::Unexpected{GXF_NULL_POINTER};
  }
  return tritonRepoConfig_;
}


}  // namespace triton
}  // namespace nvidia
