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

#include "xls/passes/passes.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/status_macros.h"

namespace xls {

absl::StatusOr<bool> FunctionBasePass::Run(Package* p,
                                           const PassOptions& options,
                                           PassResults* results) const {
  bool changed = false;
  for (FunctionBase* f : p->GetFunctionsAndProcs()) {
    XLS_ASSIGN_OR_RETURN(bool function_changed,
                         RunOnFunctionBase(f, options, results));
    changed |= function_changed;
  }
  return changed;
}

absl::StatusOr<bool> ProcPass::Run(Package* p, const PassOptions& options,
                                   PassResults* results) const {
  bool changed = false;
  for (const auto& proc : p->procs()) {
    XLS_ASSIGN_OR_RETURN(bool proc_changed,
                         RunOnProc(proc.get(), options, results));
    changed |= proc_changed;
  }
  return changed;
}

}  // namespace xls
