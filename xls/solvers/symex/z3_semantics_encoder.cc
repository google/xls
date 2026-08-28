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

#include "xls/solvers/symex/z3_semantics_encoder.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

namespace {

using ::xls::solvers::z3::BitsToZ3;
using ::xls::solvers::z3::BooleanToBitVector;
using ::xls::solvers::z3::TypeToSort;

}  // namespace

Z3SemanticsEncoder::Z3SemanticsEncoder(Z3_context ctx) : ctx_(ctx) {}

Z3_sort Z3SemanticsEncoder::GetTypeSort(const Type& type) {
  std::string key = type.ToString();
  auto it = type_sort_cache_.find(key);
  if (it != type_sort_cache_.end()) {
    return it->second;
  }

  Z3_sort sort = TypeToSort(ctx_, type);
  type_sort_cache_[key] = sort;
  return sort;
}

Z3_ast Z3SemanticsEncoder::CreateTuple(const TupleType* tuple_type,
                                       absl::Span<const Z3_ast> elements) {
  Z3_sort sort = GetTypeSort(*tuple_type);
  Z3_func_decl constr = Z3_get_datatype_sort_constructor(ctx_, sort, 0);
  return Z3_mk_app(ctx_, constr, elements.size(), elements.data());
}

Z3_ast Z3SemanticsEncoder::GetTupleElement(const TupleType* tuple_type,
                                           Z3_ast tuple, int64_t index) {
  Z3_sort sort = GetTypeSort(*tuple_type);
  Z3_func_decl proj_fn = Z3_get_tuple_sort_field_decl(ctx_, sort, index);
  return Z3_mk_app(ctx_, proj_fn, 1, &tuple);
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::TranslateValue(const Type* type,
                                                          const Value& val) {
  if (val.IsBits()) {
    return BitsToZ3(ctx_, val.bits());
  }
  if (val.IsTuple()) {
    const TupleType* tuple_type = type->AsTupleOrDie();
    std::vector<Z3_ast> elements;
    elements.reserve(val.elements().size());
    for (int64_t i = 0; i < val.elements().size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Z3_ast elem_ast,
          TranslateValue(tuple_type->element_type(i), val.elements()[i]));
      elements.push_back(elem_ast);
    }
    return CreateTuple(tuple_type, elements);
  }
  return absl::InvalidArgumentError("Unsupported value type");
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::TranslateLiteral(
    const Literal* literal) {
  return TranslateValue(literal->GetType(), literal->value());
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::TranslateParam(const Param* param) {
  std::string name(param->name());
  Z3_symbol sym = Z3_mk_string_symbol(ctx_, name.c_str());
  Z3_sort sort = GetTypeSort(*param->GetType());
  return Z3_mk_const(ctx_, sym, sort);
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::TranslateNode(
    const Node* node, absl::Span<const Z3_ast> operand_asts) {
  switch (node->op()) {
    case Op::kIdentity:
      return operand_asts[0];
    case Op::kLiteral:
      return TranslateLiteral(node->As<Literal>());
    case Op::kParam:
      return TranslateParam(node->As<Param>());
    case Op::kAdd:
      return Z3_mk_bvadd(ctx_, operand_asts[0], operand_asts[1]);
    case Op::kSub:
      return Z3_mk_bvsub(ctx_, operand_asts[0], operand_asts[1]);
    case Op::kAnd:
      return Z3_mk_bvand(ctx_, operand_asts[0], operand_asts[1]);
    case Op::kOr:
      return Z3_mk_bvor(ctx_, operand_asts[0], operand_asts[1]);
    case Op::kXor:
      return Z3_mk_bvxor(ctx_, operand_asts[0], operand_asts[1]);
    case Op::kZeroExt: {
      const auto* ext = node->As<ExtendOp>();
      int64_t diff = ext->new_bit_count() - ext->operand(0)->BitCountOrDie();
      return Z3_mk_zero_ext(ctx_, diff, operand_asts[0]);
    }
    case Op::kSignExt: {
      const auto* ext = node->As<ExtendOp>();
      int64_t diff = ext->new_bit_count() - ext->operand(0)->BitCountOrDie();
      return Z3_mk_sign_ext(ctx_, diff, operand_asts[0]);
    }
    case Op::kBitSlice: {
      const auto* slice = node->As<BitSlice>();
      return Z3_mk_extract(ctx_, slice->start() + slice->width() - 1,
                           slice->start(), operand_asts[0]);
    }
    case Op::kEq: {
      // Convert boolean formula into 1-bit BV: (eq ? 1 : 0).
      Z3_ast eq_res = Z3_mk_eq(ctx_, operand_asts[0], operand_asts[1]);
      return BooleanToBitVector(ctx_, eq_res);
    }
    case Op::kNe: {
      Z3_ast eq_res = Z3_mk_eq(ctx_, operand_asts[0], operand_asts[1]);
      Z3_ast ne_res = Z3_mk_not(ctx_, eq_res);
      return BooleanToBitVector(ctx_, ne_res);
    }
    case Op::kUGt: {
      // Unsigned greater-than comparison: (ugt ? 1 : 0).
      Z3_ast ugt_res = Z3_mk_bvugt(ctx_, operand_asts[0], operand_asts[1]);
      return BooleanToBitVector(ctx_, ugt_res);
    }
    case Op::kUGe: {
      Z3_ast uge_res = Z3_mk_bvuge(ctx_, operand_asts[0], operand_asts[1]);
      return BooleanToBitVector(ctx_, uge_res);
    }
    case Op::kULt: {
      Z3_ast ult_res = Z3_mk_bvult(ctx_, operand_asts[0], operand_asts[1]);
      return BooleanToBitVector(ctx_, ult_res);
    }
    case Op::kULe: {
      Z3_ast ule_res = Z3_mk_bvule(ctx_, operand_asts[0], operand_asts[1]);
      return BooleanToBitVector(ctx_, ule_res);
    }
    case Op::kTuple: {
      const TupleType* tuple_type = node->GetType()->AsTupleOrDie();
      return CreateTuple(tuple_type, operand_asts);
    }
    case Op::kTupleIndex: {
      const auto* tuple_index = node->As<TupleIndex>();
      const TupleType* tuple_type =
          tuple_index->operand(0)->GetType()->AsTupleOrDie();
      return GetTupleElement(tuple_type, operand_asts[0], tuple_index->index());
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported IR operation: ", node->op()));
  }
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::EncodeMuxBranchCondition(
    const Node* mux_node, int64_t arm_index, Z3_ast selector_ast) {
  if (mux_node->op() == Op::kSel) {
    const auto* sel = mux_node->As<Select>();
    const Node* sel_operand = sel->selector();
    if (selector_ast == nullptr) {
      std::string sel_name(sel_operand->GetName());
      Z3_symbol sym = Z3_mk_string_symbol(ctx_, sel_name.c_str());
      Z3_sort sort = GetTypeSort(*sel_operand->GetType());
      selector_ast = Z3_mk_const(ctx_, sym, sort);
    }

    int64_t sel_width = sel_operand->BitCountOrDie();
    int64_t num_cases = sel->cases().size();
    bool is_default = (arm_index >= num_cases);
    if (is_default) {
      // Default fallback arm predicate: selector >= num_cases.
      Z3_ast limit = BitsToZ3(ctx_, UBits(num_cases, sel_width));
      return Z3_mk_bvuge(ctx_, selector_ast, limit);
    }
    // Explicit arm predicate: selector == arm_index.
    Z3_ast target = BitsToZ3(ctx_, UBits(arm_index, sel_width));
    return Z3_mk_eq(ctx_, selector_ast, target);
  }

  if (mux_node->op() == Op::kPrioritySel) {
    const auto* psel = mux_node->As<PrioritySelect>();
    const Node* sel_operand = psel->selector();
    if (selector_ast == nullptr) {
      std::string sel_name(sel_operand->GetName());
      Z3_symbol sym = Z3_mk_string_symbol(ctx_, sel_name.c_str());
      Z3_sort sort = GetTypeSort(*sel_operand->GetType());
      selector_ast = Z3_mk_const(ctx_, sym, sort);
    }

    int64_t num_cases = psel->cases().size();
    bool is_default = (arm_index >= num_cases);
    if (is_default) {
      Z3_ast zero = BitsToZ3(ctx_, UBits(0, sel_operand->BitCountOrDie()));
      return Z3_mk_eq(ctx_, selector_ast, zero);
    }
    std::vector<Z3_ast> conds;
    Z3_sort b1 = Z3_mk_bv_sort(ctx_, 1);
    Z3_ast one_b1 = Z3_mk_unsigned_int64(ctx_, 1, b1);
    Z3_ast zero_b1 = Z3_mk_unsigned_int64(ctx_, 0, b1);
    conds.push_back(Z3_mk_eq(
        ctx_, Z3_mk_extract(ctx_, arm_index, arm_index, selector_ast), one_b1));
    for (int64_t j = 0; j < arm_index; ++j) {
      conds.push_back(
          Z3_mk_eq(ctx_, Z3_mk_extract(ctx_, j, j, selector_ast), zero_b1));
    }
    return Z3_mk_and(ctx_, conds.size(), conds.data());
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Node is not a supported multiplexer: ", mux_node->op()));
}

}  // namespace xls::solvers::symex
