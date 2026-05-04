// Copyright 2026 The XLS Authors
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

#include "xls/dev_tools/dev_passes/remove_identifiers_pass.h"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/remove_identifiers.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

absl::StatusOr<bool> RemoveIdentifiersPass::RunInternal(
    Package* p, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> new_p,
                       StripPackage(p, options_));
  std::vector<FunctionBase*> old_funcs = p->GetFunctionBases();
  std::vector<Channel*> old_chans(p->channels().begin(), p->channels().end());
  XLS_RETURN_IF_ERROR(p->SetTop(std::nullopt));
  for (FunctionBase* f : old_funcs) {
    XLS_RETURN_IF_ERROR(p->RemoveFunctionBase(f));
  }
  for (Channel* c : old_chans) {
    XLS_RETURN_IF_ERROR(p->RemoveChannel(c));
  }
  XLS_RETURN_IF_ERROR(p->ClearFiles());
  XLS_ASSIGN_OR_RETURN(Package::PackageMergeResult merge,
                       p->ImportFromPackage(new_p.get()));
  if (new_p->HasTop()) {
    std::string_view new_top_name =
        merge.name_updates.contains(new_p->GetTop().value()->name())
            ? merge.name_updates.at(new_p->GetTop().value()->name())
            : new_p->GetTop().value()->name();
    XLS_RETURN_IF_ERROR(p->SetTopByName(new_top_name))
        << "Unable to update top to " << new_top_name;
  }
  for (FunctionBase* f : p->GetFunctionBases()) {
    XLS_RETURN_IF_ERROR(f->RebuildSideTables());
  }
  return true;
}

}  // namespace xls
