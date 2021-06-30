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

#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/node_iterator.h"

namespace xls {

absl::StatusOr<bool> ConstantFoldingPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  bool changed = false;
  for (Node* node : TopoSort(f)) {
    // TODO(meheff): 2019/6/26 Consider not folding loops with large trip counts
    // to avoid hanging at compile time.
    if (node->operand_count() > 0 &&
        std::all_of(node->operands().begin(), node->operands().end(),
                    [](Node* o) { return o->Is<Literal>(); })) {
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

}  // namespace xls
