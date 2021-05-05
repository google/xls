// Copyright 2020 The XLS Authors
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

#ifndef XLS_SCHEDULING_SCHEDULING_PASS_H_
#define XLS_SCHEDULING_SCHEDULING_PASS_H_

#include "absl/types/optional.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {

// Defines the pass types for passes which operate on the IR and the schedule
// including passes which create the schedule.
// TODO(meheff): 2021/04/30 Remove this pass type as it is superceded by
// CodegenPass.

// Data structure operated on by scheduling passes. Contains the IR and the
// associated schedule.
struct SchedulingUnit {
  Package* package;
  absl::optional<PipelineSchedule> schedule;

  // Methods required by CompoundPassBase.
  std::string DumpIr() const;
  std::string name() const { return package->name(); }
};

// Options passed to each scheduling pass.
struct SchedulingPassOptions : public PassOptions {
  // The options to use when creating and mutating the schedule.
  SchedulingOptions scheduling_options;

  // Delay estimator to use for scheduling.
  const DelayEstimator* delay_estimator = nullptr;
};

using SchedulingPassResults = PassResults;
using SchedulingPass =
    PassBase<SchedulingUnit, SchedulingPassOptions, SchedulingPassResults>;
using SchedulingCompoundPass =
    CompoundPassBase<SchedulingUnit, SchedulingPassOptions,
                     SchedulingPassResults>;
using SchedulingInvariantChecker = SchedulingCompoundPass::InvariantChecker;

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_PASS_H_
