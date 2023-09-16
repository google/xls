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

namespace xls {

absl::StatusOr<bool> SchedulingOptimizationFunctionBasePass::RunOnFunctionBase(
    SchedulingUnit<FunctionBase*>* s, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  FunctionBase* f = s->ir;
  XLS_VLOG(2) << absl::StreamFormat("Running %s on function_base %s [pass #%d]",
                                    long_name(), f->name(),
                                    results->invocations.size());
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

  XLS_ASSIGN_OR_RETURN(bool changed,
                       RunOnFunctionBaseInternal(s, options, results));

  XLS_VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
  XLS_VLOG_LINES(3, f->DumpIr());
  return changed;
}

absl::StatusOr<bool> SchedulingOptimizationFunctionBasePass::RunInternal(
    SchedulingUnit<>* s, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  bool changed = false;
  for (FunctionBase* f : s->ir->GetFunctionBases()) {
    SchedulingUnit<FunctionBase*> unit{f, s->schedule};
    XLS_ASSIGN_OR_RETURN(bool function_changed,
                         RunOnFunctionBaseInternal(&unit, options, results));
    s->schedule = unit.schedule;
    changed = changed || function_changed;
  }
  return changed;
}

}  // namespace xls
