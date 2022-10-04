// Copyright 2022 The XLS Authors
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

// Dead Code Elimination.
//
#ifndef XLS_TOOLS_SCHEDULING_OPTIONS_FLAGS_H_
#define XLS_TOOLS_SCHEDULING_OPTIONS_FLAGS_H_

#include "absl/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/scheduling/pipeline_schedule.h"

ABSL_DECLARE_FLAG(int64_t, clock_period_ps);
ABSL_DECLARE_FLAG(int64_t, pipeline_stages);
ABSL_DECLARE_FLAG(std::string, delay_model);
ABSL_DECLARE_FLAG(int64_t, clock_margin_percent);
ABSL_DECLARE_FLAG(int64_t, period_relaxation_percent);
ABSL_DECLARE_FLAG(int64_t, additional_input_delay_ps);
ABSL_DECLARE_FLAG(std::vector<std::string>, scheduling_constraints);

namespace xls {

// If you don't have a `Package` at hand, you can pass in `nullptr` and it will
// skip some checks.
absl::StatusOr<SchedulingOptions> SetUpSchedulingOptions(Package* p);
absl::StatusOr<DelayEstimator*> SetUpDelayEstimator();

}  // namespace xls

#endif  // XLS_TOOLS_SCHEDULING_OPTIONS_FLAGS_H_
