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

#include "xls/dev_tools/dev_passes/remove_one_hot_select_pass.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "cppitertools/enumerate.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

absl::StatusOr<bool> RemoveOneHotSelectPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* pass_results, OptimizationContext& context) const {
  bool changed = false;
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> nodes, context.TopoSort(f));
  for (Node* node : nodes) {
    if (!node->Is<OneHotSelect>()) {
      continue;
    }
    OneHotSelect* ohs = node->As<OneHotSelect>();
    if (ohs->selector()->BitCountOrDie() > kMaxBits) {
      VLOG(1) << "Skipping " << node->ToString()
              << " due to excessive selector width";
      continue;
    }
    changed = true;
    int64_t limit = (1 << ohs->selector()->BitCountOrDie());
    std::vector<Node*> new_cases;
    new_cases.reserve(limit);
    XLS_ASSIGN_OR_RETURN(
        std::back_inserter(new_cases),
        f->MakeNode<Literal>(ohs->loc(),
                             ZeroOfType(ohs->cases()[0]->GetType())));
    // NB 0-case is already handled.
    for (int64_t i = 1; i < limit; ++i) {
      Bits case_sel = UBits(i, ohs->selector()->BitCountOrDie());
      std::vector<Node*> selected;
      selected.reserve(ohs->selector()->BitCountOrDie());
      std::vector<int64_t> selected_indices;
      selected_indices.reserve(ohs->selector()->BitCountOrDie());
      for (const auto& [idx, b] : iter::enumerate(case_sel)) {
        if (b) {
          selected.push_back(ohs->cases()[idx]);
          selected_indices.push_back(idx);
        }
      }
      XLS_ASSIGN_OR_RETURN(
          std::back_inserter(new_cases),
          NaryOrIfNeeded(f, selected,
                         NodeNameFormat("%s_cases_%s", ohs,
                                        absl::StrJoin(selected_indices, "_")),
                         ohs->loc()));
    }
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<Select>(ohs->selector(), new_cases,
                                         /*default_value=*/std::nullopt)
            .status());
  }
  return changed;
}

}  // namespace xls
