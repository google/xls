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
class SchedulingUnit {
 public:
  // Create a SchedulingUnit that operates only on `f`.
  static SchedulingUnit CreateForSingleFunction(FunctionBase* f);
  // Create a SchedulingUnit that operates on all functions and procs in `p`.
  static SchedulingUnit CreateForWholePackage(Package* p);

  SchedulingUnit() = delete;

  const PackagePipelineSchedules& schedules() const { return schedules_; }
  PackagePipelineSchedules& schedules() { return schedules_; }

  // Methods required by CompoundPassBase.
  // Dumps the IR for schedulable FunctionBases along with their schedules. The
  // schedules are guarded by "//" comments.
  std::string DumpIr() const;
  std::string name() const { return ir_->name(); }
  int64_t GetNodeCount() const { return ir_->GetNodeCount(); }

  Package* GetPackage() const { return ir_; }
  // Gets list of FunctionBases to be operated on by scheduling passes.
  absl::StatusOr<std::vector<FunctionBase*>> GetSchedulableFunctions() const;
  // Returns true if every function in `GetSchedulableFunctions()` has a
  // schedule in `schedules()`.
  bool IsScheduled() const;

  const TransformMetrics& transform_metrics() const {
    return ir_->transform_metrics();
  }

 protected:
  explicit SchedulingUnit(Package* p)
      : schedulable_unit_(p), ir_(p), schedules_({}) {}
  explicit SchedulingUnit(FunctionBase* fb_to_schedule)
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

using SchedulingPassResults = PassResults;
using SchedulingPass =
    PassBase<SchedulingUnit, SchedulingPassOptions, SchedulingPassResults>;
using SchedulingCompoundPass =
    CompoundPassBase<SchedulingUnit, SchedulingPassOptions,
                     SchedulingPassResults>;
using SchedulingInvariantChecker = SchedulingCompoundPass::InvariantChecker;

// Abstract base class for scheduling passes operating at function/proc scope.
// The derived classes must define RunOnFunctionBaseInternal.
class SchedulingOptimizationFunctionBasePass : public SchedulingPass {
 public:
  SchedulingOptimizationFunctionBasePass(std::string_view short_name,
                                         std::string_view long_name)
      : SchedulingPass(short_name, long_name) {}

  // Runs the pass on a single function/proc.
  absl::StatusOr<bool> RunOnFunctionBase(FunctionBase* f, SchedulingUnit* s,
                                         const SchedulingPassOptions& options,
                                         SchedulingPassResults* results) const;

 protected:
  // Iterates over each function and proc in the package calling
  // RunOnFunctionBase.
  absl::StatusOr<bool> RunInternal(
      SchedulingUnit* s, const SchedulingPassOptions& options,
      SchedulingPassResults* results) const override;

  virtual absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, SchedulingUnit* s, const SchedulingPassOptions& options,
      SchedulingPassResults* results) const = 0;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_PASS_H_
