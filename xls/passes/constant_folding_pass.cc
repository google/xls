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

#include "xls/passes/constant_folding_pass.h"

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {
// Check if we can do constant folding on this node.
bool NodeIsConstantFoldable(Node* node) {
  if (node->Is<Literal>()) {
    // Already a constant, nothing to do.
    return false;
  }
  if (TypeHasToken(node->GetType())) {
    // Tokens can't be folded.
    return false;
  }
  if (OpIsSideEffecting(node->op()) && !node->Is<Gate>()) {
    // Side effecting ops other than 'gate' can't be folded through.
    return false;
  }
  // Only ops with all literal operands can be folded.
  return absl::c_all_of(node->operands(),
                        [](Node* operand) { return operand->Is<Literal>(); });
}
}  // namespace

absl::StatusOr<bool> ConstantFoldingPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (Node* node : TopoSort(f)) {
    // Fold any non-side-effecting op with constant parameters. Avoid any types
    // with tokens because literal tokens are not allowed.
    // TODO(meheff): 2019/6/26 Consider not folding loops with large trip counts
    // to avoid hanging at compile time.
    if (NodeIsConstantFoldable(node)) {
      XLS_VLOG(2) << "Folding: " << *node;
      std::vector<Value> operand_values;
      for (Node* operand : node->operands()) {
        operand_values.push_back(operand->As<Literal>()->value());
      }
      XLS_ASSIGN_OR_RETURN(Value result, InterpretNode(node, operand_values));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Literal>(result).status());
      changed = true;
    }
  }

  return changed;
}

REGISTER_OPT_PASS(ConstantFoldingPass);

}  // namespace xls
