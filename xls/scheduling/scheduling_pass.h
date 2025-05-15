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

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

// Defines the pass types for passes which operate on the IR and the schedule
// including passes which create the schedule.

// Data structure operated on by scheduling passes. Contains the IR and the
// associated schedule.
class SchedulingContext {
 public:
  // Create a SchedulingContext that operates only on `f`.
  static SchedulingContext CreateForSingleFunction(FunctionBase* f);
  // Create a SchedulingContext that operates on all functions and procs in `p`.
  static SchedulingContext CreateForWholePackage(Package* p);

  SchedulingContext() = delete;

  const PackagePipelineSchedules& schedules() const { return schedules_; }
  PackagePipelineSchedules& schedules() { return schedules_; }

  // Gets list of FunctionBases to be operated on by scheduling passes.
  absl::StatusOr<std::vector<FunctionBase*>> GetSchedulableFunctions() const;
  // Returns true if every function in `GetSchedulableFunctions()` has a
  // schedule in `schedules()`.
  bool IsScheduled() const;

  const TransformMetrics& transform_metrics() const {
    return ir_->transform_metrics();
  }

 protected:
  explicit SchedulingContext(Package* p)
      : schedulable_unit_(p), ir_(p), schedules_({}) {}
  explicit SchedulingContext(FunctionBase* fb_to_schedule)
      : schedulable_unit_(fb_to_schedule),
        ir_(fb_to_schedule->package()),
        schedules_({}) {}

 private:
  // If FunctionBase, only schedule the FunctionBase, else schedule the whole
  // package.
  std::variant<Package*, FunctionBase*> schedulable_unit_;
  Package* ir_;
  PackagePipelineSchedules schedules_;
};

// Options passed to each scheduling pass.
struct SchedulingPassOptions : public PassOptionsBase {
  // The options to use when creating and mutating the schedule.
  SchedulingOptions scheduling_options;

  // Delay estimator to use for scheduling.
  const DelayEstimator* delay_estimator = nullptr;
  const synthesis::Synthesizer* synthesizer = nullptr;
};

using SchedulingPass = PassBase<SchedulingPassOptions, SchedulingContext>;
using SchedulingCompoundPass =
    CompoundPassBase<SchedulingPassOptions, SchedulingContext>;
using SchedulingInvariantChecker = SchedulingCompoundPass::InvariantChecker;
using SchedulingFunctionBasePass =
    FunctionBasePass<SchedulingPassOptions, SchedulingContext>;

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_PASS_H_
