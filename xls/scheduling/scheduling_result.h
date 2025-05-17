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

#ifndef XLS_SCHEDULING_SCHEDULING_RESULT_H_
#define XLS_SCHEDULING_SCHEDULING_RESULT_H_

#include "xls/passes/pass_metrics.pb.h"
#include "xls/scheduling/pipeline_schedule.pb.h"

namespace xls {

// Data structure gathering together the artifacts created by scheduling.
struct SchedulingResult {
  PackagePipelineSchedulesProto schedules;
  PassPipelineMetricsProto pass_pipeline_metrics;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_RESULT_H_
