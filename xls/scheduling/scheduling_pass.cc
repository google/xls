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

#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
/* static */ SchedulingContext SchedulingContext::CreateForSingleFunction(
    FunctionBase* f) {
  return SchedulingContext(f);
}
/* static */ SchedulingContext SchedulingContext::CreateForWholePackage(
    Package* p) {
  return SchedulingContext(p);
}

absl::StatusOr<std::vector<FunctionBase*>>
SchedulingContext::GetSchedulableFunctions() const {
  // Package* means 'all FunctionBases in the package'
  if (std::holds_alternative<Package*>(schedulable_unit_)) {
    XLS_RET_CHECK_EQ(ir_, std::get<Package*>(schedulable_unit_));
    // FFI functions are not schedulable.
    std::vector<FunctionBase*> schedulable_functions = ir_->GetFunctionBases();
    std::erase_if(schedulable_functions, [](FunctionBase* f) {
      return f->ForeignFunctionData().has_value();
    });
    return schedulable_functions;
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

bool SchedulingContext::IsScheduled() const {
  std::vector<FunctionBase*> schedulable_functions;
  return absl::c_all_of(schedulable_functions, [this](FunctionBase* f) {
    return package_schedule_.HasSchedule(f);
  });
}

}  // namespace xls
