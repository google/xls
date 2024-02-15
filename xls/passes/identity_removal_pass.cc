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

#include "xls/passes/identity_removal_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Identity Removal performs one forward pass over the nodes and replaces
// identities with their respective operands.
absl::StatusOr<bool> IdentityRemovalPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (Node* node : f->nodes()) {
    if (node->op() == Op::kIdentity) {
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(node->operand(0)));
      changed = true;
    }
  }
  return changed;
}

REGISTER_OPT_PASS(IdentityRemovalPass);

}  // namespace xls
