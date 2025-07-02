// Copyright 2025 The XLS Authors
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

#include "xls/passes/select_merging_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

absl::StatusOr<bool> MergeNode(Node* node) {
  // We only consider PrioritySelect and OneHotSelect
  if (!node->Is<PrioritySelect>() && !node->Is<OneHotSelect>()) {
    return false;
  }

  // Try to merge consecutive selects
  Node* selector;
  absl::Span<Node* const> cases;
  std::optional<Node*> default_value = std::nullopt;
  if (node->Is<OneHotSelect>()) {
    selector = node->As<OneHotSelect>()->selector();
    cases = node->As<OneHotSelect>()->cases();
  } else {
    XLS_RET_CHECK(node->Is<PrioritySelect>());
    selector = node->As<PrioritySelect>()->selector();
    cases = node->As<PrioritySelect>()->cases();
    default_value = node->As<PrioritySelect>()->default_value();
  }
  auto is_single_user_matching_select = [select_op = node->op()](Node* n) {
    return n->op() == select_op && HasSingleUse(n);
  };
  if (std::any_of(cases.begin(), cases.end(), is_single_user_matching_select) ||
      (default_value.has_value() &&
       is_single_user_matching_select(*default_value))) {
    std::vector<Node*> temp_cases;
    if (selector->BitCountOrDie() == 1 && default_value.has_value() &&
        !is_single_user_matching_select(*default_value)) {
      // The default value is not the case with a matching select - and the
      // selector is a single bit. We can get an equivalent select by negating
      // the selector & swapping the default value with the case, and this
      // will simplify the resulting merge.
      CHECK_EQ(cases.size(), 1);
      temp_cases = {*default_value};
      default_value = cases.front();
      cases = absl::MakeConstSpan(temp_cases);
      XLS_ASSIGN_OR_RETURN(selector, node->function_base()->MakeNode<UnOp>(
                                         node->loc(), selector, Op::kNot));
    }
    // Cases for the replacement one-hot-select.
    std::vector<Node*> new_cases;
    // Pieces of the selector for the replacement one-hot-select. These are
    // concatted together.
    std::vector<Node*> new_selector_parts;
    std::optional<Node*> new_default_value = default_value;
    // When iterating through the cases to perform this optimization, cases
    // which are to remain unmodified (ie, not a single-use one-hot-select)
    // are passed over. This lambda gathers the passed over cases and
    // updates new_cases and new_selector_parts.
    int64_t unhandled_selector_bits = 0;
    auto add_unhandled_selector_bits = [&](int64_t index) -> absl::Status {
      if (unhandled_selector_bits != 0) {
        Node* selector_part;
        if (index == selector->BitCountOrDie() &&
            unhandled_selector_bits == selector->BitCountOrDie()) {
          selector_part = selector;
        } else {
          XLS_ASSIGN_OR_RETURN(selector_part,
                               node->function_base()->MakeNode<BitSlice>(
                                   node->loc(), selector,
                                   /*start=*/index - unhandled_selector_bits,
                                   /*width=*/
                                   unhandled_selector_bits));
        }
        new_selector_parts.push_back(selector_part);
        for (int64_t i = index - unhandled_selector_bits; i < index; ++i) {
          new_cases.push_back(cases[i]);
        }
      }
      unhandled_selector_bits = 0;
      return absl::OkStatus();
    };
    // Iterate through the cases merging single-use matching-select cases.
    for (int64_t i = 0; i < cases.size(); ++i) {
      if (is_single_user_matching_select(cases[i])) {
        Node* operand_selector;
        absl::Span<Node* const> operand_cases;
        std::optional<Node*> operand_default_value = std::nullopt;
        if (cases[i]->Is<OneHotSelect>()) {
          operand_selector = cases[i]->As<OneHotSelect>()->selector();
          operand_cases = cases[i]->As<OneHotSelect>()->cases();
        } else {
          XLS_RET_CHECK(cases[i]->Is<PrioritySelect>());
          operand_selector = cases[i]->As<PrioritySelect>()->selector();
          operand_cases = cases[i]->As<PrioritySelect>()->cases();
          operand_default_value =
              cases[i]->As<PrioritySelect>()->default_value();
        }
        XLS_RETURN_IF_ERROR(add_unhandled_selector_bits(i));
        // The selector bits for the predecessor bit-select need to be
        // ANDed with the original selector bit in the successor bit-select.
        // Example:
        //
        //   X = one_hot_select(selector={A, B, C},
        //                      cases=[x, y z])
        //   Y = one_hot_select(selector={..., S, ...},
        //                      cases=[..., X, ...])
        // Becomes:
        //
        //   Y = one_hot_select(
        //     selector={..., S & A, S & B, S & C, ...},
        //     cases=[..., A, B, C, ...])
        //
        Node* selector_bit;
        if (selector->BitCountOrDie() == 1) {
          CHECK_EQ(i, 0);
          selector_bit = selector;
        } else {
          XLS_ASSIGN_OR_RETURN(selector_bit,
                               node->function_base()->MakeNode<BitSlice>(
                                   node->loc(), selector,
                                   /*start=*/i, /*width=*/1));
        }
        Node* selector_bit_mask;
        if (operand_cases.size() == 1) {
          selector_bit_mask = selector_bit;
        } else {
          XLS_ASSIGN_OR_RETURN(
              selector_bit_mask,
              node->function_base()->MakeNode<ExtendOp>(
                  node->loc(), selector_bit,
                  /*new_bit_count=*/operand_cases.size(), Op::kSignExt));
        }
        XLS_ASSIGN_OR_RETURN(
            Node * masked_selector,
            node->function_base()->MakeNode<NaryOp>(
                node->loc(),
                std::vector<Node*>{selector_bit_mask, operand_selector},
                Op::kAnd));
        new_selector_parts.push_back(masked_selector);
        absl::c_copy(operand_cases, std::back_inserter(new_cases));
        if (operand_default_value.has_value()) {
          // We also need to handle the scenario where this case is
          // selected, but the case evaluates to its default value.
          Node* operand_selector_is_zero;
          if (operand_selector->BitCountOrDie() == 1) {
            XLS_ASSIGN_OR_RETURN(
                operand_selector_is_zero,
                node->function_base()->MakeNode<UnOp>(
                    cases[i]->loc(), operand_selector, Op::kNot));
          } else {
            XLS_ASSIGN_OR_RETURN(
                Node * operand_selector_zero,
                node->function_base()->MakeNode<Literal>(
                    cases[i]->loc(), ZeroOfType(operand_selector->GetType())));
            XLS_ASSIGN_OR_RETURN(operand_selector_is_zero,
                                 node->function_base()->MakeNode<CompareOp>(
                                     node->loc(), operand_selector,
                                     operand_selector_zero, Op::kEq));
          }
          XLS_ASSIGN_OR_RETURN(
              Node * masked_operand_selector_is_zero,
              node->function_base()->MakeNode<NaryOp>(
                  cases[i]->loc(),
                  std::vector<Node*>{selector_bit, operand_selector_is_zero},
                  Op::kAnd));
          new_selector_parts.push_back(masked_operand_selector_is_zero);
          new_cases.push_back(*operand_default_value);
        }
      } else {
        unhandled_selector_bits++;
      }
    }
    XLS_RETURN_IF_ERROR(add_unhandled_selector_bits(cases.size()));
    if (default_value.has_value() &&
        is_single_user_matching_select(*default_value)) {
      Node* operand_selector;
      absl::Span<Node* const> operand_cases;
      std::optional<Node*> operand_default_value = std::nullopt;
      if (default_value.value()->Is<OneHotSelect>()) {
        operand_selector =
            default_value.value()->As<OneHotSelect>()->selector();
        operand_cases = default_value.value()->As<OneHotSelect>()->cases();
      } else {
        XLS_RET_CHECK(default_value.value()->Is<PrioritySelect>());
        operand_selector =
            default_value.value()->As<PrioritySelect>()->selector();
        operand_cases = default_value.value()->As<PrioritySelect>()->cases();
        operand_default_value =
            default_value.value()->As<PrioritySelect>()->default_value();
      }
      new_selector_parts.push_back(operand_selector);
      absl::c_copy(operand_cases, std::back_inserter(new_cases));
      new_default_value = operand_default_value;
    }
    // Reverse selector parts because concat operand zero is the msb.
    std::reverse(new_selector_parts.begin(), new_selector_parts.end());
    XLS_ASSIGN_OR_RETURN(Node * new_selector,
                         node->function_base()->MakeNode<Concat>(
                             node->loc(), new_selector_parts));
    if (node->Is<OneHotSelect>()) {
      XLS_RET_CHECK(!new_default_value.has_value());
      VLOG(2) << absl::StrFormat("Merging consecutive one-hot-selects: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<OneHotSelect>(new_selector, new_cases)
              .status());
    } else {
      XLS_RET_CHECK(node->Is<PrioritySelect>());
      XLS_RET_CHECK(new_default_value.has_value());
      VLOG(2) << absl::StrFormat("Merging consecutive priority-selects: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<PrioritySelect>(
                                  new_selector, new_cases, *new_default_value)
                              .status());
    }
    return true;
  }

  return false;
}

absl::StatusOr<bool> SelectMergingPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  bool changed = false;

  for (Node* node : context.TopoSort(func)) {
    XLS_ASSIGN_OR_RETURN(bool node_changed, MergeNode(node));
    changed = changed || node_changed;
  }

  return changed;
}


}  // namespace xls
