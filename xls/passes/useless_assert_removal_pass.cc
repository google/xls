// Copyright 2021 The XLS Authors
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

#include "xls/passes/useless_assert_removal_pass.h"

#include <optional>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

absl::StatusOr<bool> UselessAssertRemovalPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  StatelessQueryEngine query_engine;

  bool changed = false;
  // Remove asserts with literal true conditions.
  for (Node* node : context.TopoSort(f)) {
    if (node->op() == Op::kAssert) {
      Assert* current_assert = node->As<Assert>();
      Node* condition = current_assert->condition();
      std::optional<Bits> constant_condition =
          query_engine.KnownValueAsBits(condition);
      if (constant_condition.has_value() && constant_condition->IsOne()) {
        // Set token operand user to the assert user (rewire token).
        XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(node->As<Assert>()->token()));
        XLS_RETURN_IF_ERROR(f->RemoveNode(node));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xls
