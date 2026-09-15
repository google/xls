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

#include "xls/solvers/symex/z3_encoding_visitor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_op_translator.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

namespace {

using ::xls::solvers::z3::BitsToZ3;
using ::xls::solvers::z3::TypeToSort;

}  // namespace

Z3EncodingVisitor::Z3EncodingVisitor(Z3_context ctx, Function* fn)
    : xls::solvers::z3::IrTranslator(ctx, fn, /*imported_params=*/std::nullopt),
      op_translator_(ctx) {}

Z3_ast Z3EncodingVisitor::GetNodeAst(const Node* node) const {
  auto it = translations().find(node);
  return it != translations().end() ? it->second : nullptr;
}

absl::Status Z3EncodingVisitor::HandleSel(Select* sel) {
  // TODO: also recording the full translation could be useful for allowing
  // model checks with only some selects constrained.
  std::string name(sel->GetName());
  Z3_symbol sym = Z3_mk_string_symbol(ctx(), name.c_str());
  Z3_sort sort = TypeToSort(ctx(), *sel->GetType());
  NoteTranslation(sel, Z3_mk_const(ctx(), sym, sort));
  return absl::OkStatus();
}

absl::Status Z3EncodingVisitor::HandlePrioritySel(PrioritySelect* psel) {
  std::string name(psel->GetName());
  Z3_symbol sym = Z3_mk_string_symbol(ctx(), name.c_str());
  Z3_sort sort = TypeToSort(ctx(), *psel->GetType());
  NoteTranslation(psel, Z3_mk_const(ctx(), sym, sort));
  return absl::OkStatus();
}

absl::StatusOr<Z3_ast> Z3EncodingVisitor::EncodeMuxBranchCondition(
    const Node* mux_node, int64_t arm_index) {
  XLS_RET_CHECK(GenericSelect::IsSelect(mux_node) &&
                !mux_node->Is<OneHotSelect>())
      << "Node is not a supported multiplexer: " << mux_node->ToString();

  GenericSelect sel = *GenericSelect::TryFrom(const_cast<Node*>(mux_node));
  const Node* sel_operand = sel.selector();
  Z3_ast selector_ast = GetNodeAst(sel_operand);
  if (selector_ast == nullptr) {
    std::string sel_name(sel_operand->GetName());
    Z3_symbol sym = Z3_mk_string_symbol(ctx(), sel_name.c_str());
    Z3_sort sort = TypeToSort(ctx(), *sel_operand->GetType());
    selector_ast = Z3_mk_const(ctx(), sym, sort);
  }

  int64_t num_cases = sel.cases().size();
  bool is_default = (arm_index >= num_cases);

  if (mux_node->op() == Op::kSel) {
    int64_t sel_width = sel_operand->BitCountOrDie();
    if (is_default) {
      // Default fallback arm predicate: selector >= num_cases.
      Z3_ast limit = BitsToZ3(ctx(), UBits(num_cases, sel_width));
      return op_translator_.UGeBool(selector_ast, limit);
    }
    // Explicit arm predicate: selector == arm_index.
    Z3_ast target = BitsToZ3(ctx(), UBits(arm_index, sel_width));
    return op_translator_.EqBool(selector_ast, target);
  }

  if (mux_node->op() == Op::kPrioritySel) {
    if (is_default) {
      // Default fallback arm: selector == 0 (no priority bits set).
      Z3_ast zero = BitsToZ3(ctx(), UBits(0, sel_operand->BitCountOrDie()));
      return op_translator_.EqBool(selector_ast, zero);
    }

    // Priority arm k predicate: selector[k] == 1 && selector[0..k-1] == 0.
    std::vector<Z3_ast> conds;
    conds.reserve(arm_index + 1);
    Z3_ast one_b1 = op_translator_.Fill(true, 1);
    Z3_ast zero_b1 = op_translator_.Fill(false, 1);
    conds.push_back(op_translator_.EqBool(
        op_translator_.Extract(selector_ast, arm_index), one_b1));
    for (int64_t j = 0; j < arm_index; ++j) {
      conds.push_back(op_translator_.EqBool(
          op_translator_.Extract(selector_ast, j), zero_b1));
    }
    return Z3_mk_and(ctx(), conds.size(), conds.data());
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported multiplexer opcode: ", mux_node->ToString()));
}

absl::StatusOr<Z3_ast> Z3EncodingVisitor::EncodeMuxArmEquality(
    const Node* mux_node, int64_t arm_index) {
  Z3_ast mux_var = GetNodeAst(mux_node);
  XLS_RET_CHECK_NE(mux_var, nullptr)
      << "Mux node not found in encoder map: " << mux_node->ToString();

  XLS_RET_CHECK(GenericSelect::IsSelect(mux_node) &&
                !mux_node->Is<OneHotSelect>())
      << "Node is not a supported multiplexer: " << mux_node->ToString();

  GenericSelect sel = *GenericSelect::TryFrom(const_cast<Node*>(mux_node));
  int64_t num_cases = sel.cases().size();
  Node* chosen_arm =
      (arm_index >= num_cases) ? *sel.default_value() : sel.cases()[arm_index];

  Z3_ast arm_ast = GetNodeAst(chosen_arm);
  XLS_RET_CHECK_NE(arm_ast, nullptr)
      << "Arm node not found in encoder map: " << chosen_arm->ToString();

  return op_translator_.EqBool(mux_var, arm_ast);
}

}  // namespace xls::solvers::symex
