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

#include "xls/scheduling/scheduling_pass.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
/* static */ SchedulingUnit SchedulingUnit::CreateForSingleFunction(
    FunctionBase* f) {
  return SchedulingUnit(f);
}
/* static */ SchedulingUnit SchedulingUnit::CreateForWholePackage(Package* p) {
  return SchedulingUnit(p);
}

std::string SchedulingUnit::DumpIr() const {
  // Dump the IR followed by the schedule. The schedule is commented out
  // ('//') so the output is parsable.
  std::string out = ir_->DumpIr();

  absl::StatusOr<std::vector<FunctionBase*>> schedulable_functions =
      GetSchedulableFunctions();
  if (!schedulable_functions.ok()) {
    absl::StrAppend(&out, "\n\n// ", schedulable_functions.status().message(),
                    "\n");
    return out;
  }
  for (FunctionBase* scheduled_function : *schedulable_functions) {
    auto itr = schedules_.find(scheduled_function);
    if (itr == schedules_.end()) {
      continue;
    }
    const PipelineSchedule& schedule = itr->second;
    absl::StrAppend(&out, "\n\n// Pipeline Schedule\n");
    for (auto line : absl::StrSplit(schedule.ToString(), '\n')) {
      absl::StrAppend(&out, "// ", line, "\n");
    }
  }
  return out;
}

absl::StatusOr<std::vector<FunctionBase*>>
SchedulingUnit::GetSchedulableFunctions() const {
  // Package* means 'all FunctionBases in the package'
  if (std::holds_alternative<Package*>(schedulable_unit_)) {
    XLS_RET_CHECK_EQ(ir_, std::get<Package*>(schedulable_unit_));
    return ir_->GetFunctionBases();
  }
  // Otherwise, return the specified FunctionBase (if it still exists in the
  // package).
  FunctionBase* f = std::get<FunctionBase*>(schedulable_unit_);
  // Check that schedulable_unit_ is still in the package.
  bool found = false;
  for (FunctionBase* current_fb : ir_->GetFunctionBases()) {
    if (current_fb == f) {
      found = true;
      break;
    }
  }
  XLS_RET_CHECK(found) << "FunctionBase to schedule not found in the "
                          "package; did a pass remove it?";
  return std::vector<FunctionBase*>{f};
}

bool SchedulingUnit::IsScheduled() const {
  std::vector<FunctionBase*> schedulable_functions;
  return absl::c_all_of(schedulable_functions, [this](FunctionBase* f) {
    return schedules_.contains(f);
  });
}

absl::StatusOr<bool> SchedulingOptimizationFunctionBasePass::RunOnFunctionBase(
    FunctionBase* f, SchedulingUnit* s, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  XLS_RET_CHECK_EQ(f->package(), s->GetPackage());
  VLOG(2) << absl::StreamFormat("Running %s on function_base %s [pass #%d]",
                                long_name(), f->name(),
                                results->invocations.size());
  VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

  XLS_ASSIGN_OR_RETURN(bool changed,
                       RunOnFunctionBaseInternal(f, s, options, results));

  VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
  XLS_VLOG_LINES(3, f->DumpIr());
  return changed;
}

absl::StatusOr<bool> SchedulingOptimizationFunctionBasePass::RunInternal(
    SchedulingUnit* s, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  bool changed = false;
  XLS_ASSIGN_OR_RETURN(std::vector<FunctionBase*> schedulable_functions,
                       s->GetSchedulableFunctions());
  for (FunctionBase* f : schedulable_functions) {
    XLS_ASSIGN_OR_RETURN(bool function_changed,
                         RunOnFunctionBaseInternal(f, s, options, results));
    changed = changed || function_changed;
  }
  return changed;
}

}  // namespace xls
