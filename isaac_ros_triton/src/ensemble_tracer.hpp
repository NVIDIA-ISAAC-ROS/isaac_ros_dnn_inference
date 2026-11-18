// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ENSEMBLE_TRACER_HPP_
#define ENSEMBLE_TRACER_HPP_

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "isaac_common/tracing/nvtx_scope.hpp"
#include "triton/core/tritonserver.h"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

/// Ensemble-aware NVTX bridge splitting gr00t:wait_response per model. Keyed by trace id: an
/// ensemble shares one userp across child traces, which double-frees the bridge's state.
class EnsembleTracer
{
  /// Server-side phases, aligned with the single-model tracer for the same analysis tooling.
  enum Phase : int
  {
    kPhaseAccept = 0,   ///< REQUEST_START -> QUEUE_START
    kPhaseQueue,        ///< QUEUE_START -> COMPUTE_START
    kPhaseInput,        ///< COMPUTE_START -> COMPUTE_INPUT_END
    kPhaseExecute,      ///< COMPUTE_INPUT_END -> COMPUTE_OUTPUT_START
    kPhaseOutput,       ///< COMPUTE_OUTPUT_START -> COMPUTE_END
    kPhaseReportLag,    ///< COMPUTE_END -> REQUEST_END
    kPhaseCount
  };

  /// Per-model strings; marks must carry the name or "execute_ns" is a mixture across the ensemble.
  struct Labels
  {
    std::string span;                 ///< "triton:<model>"
    std::string mark[kPhaseCount];    ///< "triton:<model>:<phase>_ns (triton clock)"
  };

  /// One span per traced model, opened at REQUEST_START and closed at REQUEST_END.
  struct Entry
  {
#if defined(USE_NVTX)
    nvtxRangeId_t range{0};
#endif
    const Labels * labels{nullptr};   ///< borrowed from labels_, stable for process lifetime
    uint64_t ts[TRITONSERVER_TRACE_REQUEST_END + 1]{};
  };

public:
  static EnsembleTracer & instance()
  {
    static EnsembleTracer tracer;
    return tracer;
  }

  static void activity_fn(
    TRITONSERVER_InferenceTrace * trace, TRITONSERVER_InferenceTraceActivity activity,
    uint64_t timestamp_ns, void * userp)
  {
    auto * self = static_cast<EnsembleTracer *>(userp);
    if (self == nullptr || trace == nullptr) {
      return;
    }
    self->on_activity(trace, activity, timestamp_ns);
  }

  /// Erases this trace's entry only, so a sibling trace's release is harmless.
  static void release_fn(TRITONSERVER_InferenceTrace * trace, void * userp)
  {
    auto * self = static_cast<EnsembleTracer *>(userp);
    if (self != nullptr && trace != nullptr) {
      uint64_t id{0};
      if (trace_id(trace, &id)) {
        self->erase(id);
      }
    }
    if (trace != nullptr) {
      TRITONSERVER_Error * err = TRITONSERVER_InferenceTraceDelete(trace);
      if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
      }
    }
  }

private:
  static bool trace_id(TRITONSERVER_InferenceTrace * trace, uint64_t * id)
  {
    TRITONSERVER_Error * err = TRITONSERVER_InferenceTraceId(trace, id);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      return false;
    }
    return true;
  }

  /// "triton:<model>" -- allocated once per distinct model, so the hot path copies a handle.
  static const char * span_label(TRITONSERVER_InferenceTrace * trace)
  {
    const char * model{nullptr};
    TRITONSERVER_Error * err = TRITONSERVER_InferenceTraceModelName(trace, &model);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      return nullptr;
    }
    return model;
  }

  void on_activity(
    TRITONSERVER_InferenceTrace * trace, TRITONSERVER_InferenceTraceActivity activity,
    uint64_t timestamp_ns)
  {
#if defined(USE_NVTX)
    uint64_t id{0};
    if (!trace_id(trace, &id)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mu_);
    Entry & e = entries_[id];
    if (activity <= TRITONSERVER_TRACE_REQUEST_END) {
      e.ts[activity] = timestamp_ns;
    }

    if (activity == TRITONSERVER_TRACE_REQUEST_START) {
      const char * model = span_label(trace);
      const Labels & lbl = labels_for(model != nullptr ? model : "unknown");
      e.labels = &lbl;

      nvtxEventAttributes_t attr{};
      attr.version = NVTX_VERSION;
      attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      attr.colorType = NVTX_COLOR_ARGB;
      attr.color = 0xFF4682B4;
      attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
      attr.message.ascii = lbl.span.c_str();
      e.range = nvtxDomainRangeStartEx(domain(), &attr);
      nvidia::isaac_ros::tracing::name_thread("triton:sched");
    } else if (activity == TRITONSERVER_TRACE_COMPUTE_START) {
      // COMPUTE_* is not a range: Triton reports the four timestamps in one burst afterwards.
      nvidia::isaac_ros::tracing::name_thread("triton:backend");
    } else if (activity == TRITONSERVER_TRACE_REQUEST_END) {
      if (e.range != 0) {
        nvtxDomainRangeEnd(domain(), e.range);
        e.range = 0;
      }
      if (e.labels == nullptr) {
        // REQUEST_END with no REQUEST_START: nothing to name the marks with, so skip.
        return;
      }
      // The six phases tile REQUEST_START -> REQUEST_END exactly, so the report can check closure.
      mark_phase(
        e, kPhaseAccept, TRITONSERVER_TRACE_REQUEST_START, TRITONSERVER_TRACE_QUEUE_START);
      mark_phase(
        e, kPhaseQueue, TRITONSERVER_TRACE_QUEUE_START, TRITONSERVER_TRACE_COMPUTE_START);
      mark_phase(
        e, kPhaseInput, TRITONSERVER_TRACE_COMPUTE_START, TRITONSERVER_TRACE_COMPUTE_INPUT_END);
      mark_phase(
        e, kPhaseExecute, TRITONSERVER_TRACE_COMPUTE_INPUT_END,
        TRITONSERVER_TRACE_COMPUTE_OUTPUT_START);
      mark_phase(
        e, kPhaseOutput, TRITONSERVER_TRACE_COMPUTE_OUTPUT_START,
        TRITONSERVER_TRACE_COMPUTE_END);
      mark_phase(
        e, kPhaseReportLag, TRITONSERVER_TRACE_COMPUTE_END, TRITONSERVER_TRACE_REQUEST_END);
    }
#else
    (void)trace;
    (void)activity;
    (void)timestamp_ns;
#endif
  }

  void erase(uint64_t id)
  {
    std::lock_guard<std::mutex> lock(mu_);
#if defined(USE_NVTX)
    auto it = entries_.find(id);
    if (it != entries_.end() && it->second.range != 0) {
      // Close anything still open so an aborted request cannot leak an unterminated range.
      nvtxDomainRangeEnd(domain(), it->second.range);
    }
#endif
    entries_.erase(id);
  }

#if defined(USE_NVTX)
  /// Separate domain from isaac_deploy / isaac_deploy.triton so ensemble spans are filterable.
  static nvtxDomainHandle_t domain()
  {
    static const nvtxDomainHandle_t handle = nvtxDomainCreateA("isaac_deploy.ensemble");
    return handle;
  }

  /// Triton-clock duration as a payload mark, immune to callback-delivery skew.
  void mark_phase(
    const Entry & e, Phase phase, TRITONSERVER_InferenceTraceActivity from,
    TRITONSERVER_InferenceTraceActivity to)
  {
    // Zero-length phases are emitted, not skipped: dropping them would shrink n and bias the mean.
    if (e.labels == nullptr || e.ts[from] == 0 || e.ts[to] < e.ts[from]) {
      return;
    }
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = e.labels->mark[phase].c_str();
    attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attr.payload.ullValue = e.ts[to] - e.ts[from];
    nvtxDomainMarkEx(domain(), &attr);
  }
#endif

  /// Interns the label set for a model. Called under mu_, once per distinct model name.
  const Labels & labels_for(const std::string & model)
  {
    auto it = labels_.find(model);
    if (it != labels_.end()) {
      return it->second;
    }
    static const char * kPhaseNames[kPhaseCount] = {
      "accept", "queue", "input", "execute", "output", "report_lag"};
    Labels lbl;
    lbl.span = "triton:" + model;
    for (int p = 0; p < kPhaseCount; ++p) {
      lbl.mark[p] = "triton:" + model + ":" + kPhaseNames[p] + "_ns (triton clock)";
    }
    return labels_.emplace(model, std::move(lbl)).first->second;
  }

  std::mutex mu_;
  std::unordered_map<uint64_t, Entry> entries_;
  std::unordered_map<std::string, Labels> labels_;
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ENSEMBLE_TRACER_HPP_
