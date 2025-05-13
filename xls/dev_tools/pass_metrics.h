// Copyright 2025 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_DEV_TOOLS_PASS_METRICS_H_
#define XLS_DEV_TOOLS_PASS_METRICS_H_

#include <string>

#include "absl/time/time.h"
#include "xls/passes/pass_metrics.pb.h"

namespace xls {

// Returns a string summarizing the data in the pipeline metrics proto in
// formatted table form. If `show_all_changed_passes` is true then all passes
// which change the IR are shown in the hiearchical table.
std::string SummarizePassPipelineMetrics(
    const PassPipelineMetricsProto& metrics, bool show_all_changed_passes);

// Returns the total duration of the pass pipeline.
absl::Duration PassPipelineDuration(const PassPipelineMetricsProto& metrics);

}  // namespace xls

#endif  // XLS_DEV_TOOLS_PASS_METRICS_H_
