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

#include <optional>
#include <string>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {

// Defines the pass types for passes which operate on the IR and the schedule
// including passes which create the schedule.

// Data structure operated on by scheduling passes. Contains the IR and the
// associated schedule.
template <typename IrT = Package*>
struct SchedulingUnit {
  IrT ir;
  std::optional<PipelineSchedule> schedule;

  // Methods required by CompoundPassBase.
  std::string DumpIr() const {
    // Dump the IR followed by the schedule. The schedule is commented out
    // ('//') so the output is parsable.
    std::string out = ir->DumpIr();
    if (schedule.has_value()) {
      absl::StrAppend(&out, "\n\n// Pipeline Schedule\n");
      for (auto line : absl::StrSplit(schedule->ToString(), '\n')) {
        absl::StrAppend(&out, "// ", line, "\n");
      }
    }
    return out;
  }
  std::string name() const { return ir->name(); }
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
    PassBase<SchedulingUnit<>, SchedulingPassOptions, SchedulingPassResults>;
using SchedulingCompoundPass =
    CompoundPassBase<SchedulingUnit<>, SchedulingPassOptions,
                     SchedulingPassResults>;
using SchedulingInvariantChecker = SchedulingCompoundPass::InvariantChecker;

// Abstract base class for scheduling passes operating at function/proc scope.
// The derived classes must define RunOnFunctionBaseInternal.
class SchedulingFunctionBasePass : public SchedulingPass {
 public:
  SchedulingFunctionBasePass(std::string_view short_name,
                             std::string_view long_name)
      : SchedulingPass(short_name, long_name) {}

  // Runs the pass on a single function/proc.
  absl::StatusOr<bool> RunOnFunctionBase(SchedulingUnit<FunctionBase*>* s,
                                         const SchedulingPassOptions& options,
                                         SchedulingPassResults* results) const;

 protected:
  // Iterates over each function and proc in the package calling
  // RunOnFunctionBase.
  absl::StatusOr<bool> RunInternal(
      SchedulingUnit<>* s, const SchedulingPassOptions& options,
      SchedulingPassResults* results) const override;

  virtual absl::StatusOr<bool> RunOnFunctionBaseInternal(
      SchedulingUnit<FunctionBase*>* s, const SchedulingPassOptions& options,
      SchedulingPassResults* results) const = 0;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_PASS_H_
