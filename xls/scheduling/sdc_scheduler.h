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

#ifndef XLS_SCHEDULING_SDC_SCHEDULER_H_
#define XLS_SCHEDULING_SDC_SCHEDULER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/proc.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

// Schedule to minimize the total pipeline registers using SDC scheduling
// the constraint matrix is totally unimodular, this ILP problem can be solved
// by LP.
//
// References:
//   - Cong, Jason, and Zhiru Zhang. "An efficient and versatile scheduling
//   algorithm based on SDC formulation." 2006 43rd ACM/IEEE Design Automation
//   Conference. IEEE, 2006.
//   - Zhang, Zhiru, and Bin Liu. "SDC-based modulo scheduling for pipeline
//   synthesis." 2013 IEEE/ACM International Conference on Computer-Aided Design
//   (ICCAD). IEEE, 2013.
absl::StatusOr<ScheduleCycleMap> SDCScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints);

}  // namespace xls

#endif  // XLS_SCHEDULING_SDC_SCHEDULER_H_
