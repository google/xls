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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
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
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {

namespace {

using ::xls::solvers::z3::BitsToZ3;
using ::xls::solvers::z3::BooleanToBitVector;

}  // namespace

Z3SemanticsEncoder::Z3SemanticsEncoder(Z3_context ctx) : ctx_(ctx) {}

Z3_sort Z3SemanticsEncoder::GetTypeSort(const xls::Type& type) {
  std::string key = type.ToString();
  auto it = type_sort_cache_.find(key);
  if (it != type_sort_cache_.end()) {
    return it->second;
  }

  Z3_sort sort = nullptr;
  switch (type.kind()) {
    case xls::TypeKind::kBits: {
      sort = Z3_mk_bv_sort(ctx_, type.GetFlatBitCount());
      break;
    }
    case xls::TypeKind::kTuple: {
      const xls::TupleType* tuple_type = type.AsTupleOrDie();
      int64_t num_elements = tuple_type->size();
      Z3_symbol tuple_name = Z3_mk_string_symbol(ctx_, key.c_str());

      std::vector<Z3_symbol> field_names;
      std::vector<Z3_sort> field_sorts;
      field_names.reserve(num_elements);
      field_sorts.reserve(num_elements);
      for (int64_t i = 0; i < num_elements; ++i) {
        field_names.push_back(Z3_mk_int_symbol(ctx_, static_cast<int>(i)));
        field_sorts.push_back(GetTypeSort(*tuple_type->element_type(i)));
      }

      Z3_func_decl mk_tuple_decl;
      std::vector<Z3_func_decl> proj_decls(num_elements);
      sort = Z3_mk_tuple_sort(
          ctx_, tuple_name, num_elements,
          field_names.empty() ? nullptr : field_names.data(),
          field_sorts.empty() ? nullptr : field_sorts.data(), &mk_tuple_decl,
          proj_decls.empty() ? nullptr : proj_decls.data());
      tuple_constructor_cache_[key] = mk_tuple_decl;
      for (int64_t i = 0; i < num_elements; ++i) {
        tuple_projector_cache_[{key, i}] = proj_decls[i];
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported type: " << xls::TypeKindToString(type.kind());
  }

  type_sort_cache_[key] = sort;
  return sort;
}

Z3_ast Z3SemanticsEncoder::CreateTuple(const xls::TupleType* tuple_type,
                                       absl::Span<const Z3_ast> elements) {
  std::string key = tuple_type->ToString();
  GetTypeSort(*tuple_type);
  Z3_func_decl constr = tuple_constructor_cache_.at(key);
  return Z3_mk_app(ctx_, constr, elements.size(), elements.data());
}

Z3_ast Z3SemanticsEncoder::GetTupleElement(const xls::TupleType* tuple_type,
                                           Z3_ast tuple, int64_t index) {
  std::string key = tuple_type->ToString();
  GetTypeSort(*tuple_type);
  Z3_func_decl proj_fn = tuple_projector_cache_.at({key, index});
  return Z3_mk_app(ctx_, proj_fn, 1, &tuple);
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::TranslateValue(
    const xls::Type* type, const xls::Value& val) {
  GetTypeSort(*type);
  if (val.IsBits()) {
    return BitsToZ3(ctx_, val.bits());
  }
  if (val.IsTuple()) {
    const xls::TupleType* tuple_type = type->AsTupleOrDie();
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
    const xls::Literal* literal) {
  return TranslateValue(literal->GetType(), literal->value());
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::TranslateParam(
    const xls::Param* param) {
  std::string name(param->name());
  Z3_symbol sym = Z3_mk_string_symbol(ctx_, name.c_str());
  Z3_sort sort = GetTypeSort(*param->GetType());
  return Z3_mk_const(ctx_, sym, sort);
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::TranslateNode(
    const xls::Node* node, absl::Span<const Z3_ast> operand_asts) {
  switch (node->op()) {
    case xls::Op::kIdentity:
      return operand_asts[0];
    case xls::Op::kLiteral:
      return TranslateLiteral(node->As<xls::Literal>());
    case xls::Op::kParam:
      return TranslateParam(node->As<xls::Param>());
    case xls::Op::kAdd:
      return Z3_mk_bvadd(ctx_, operand_asts[0], operand_asts[1]);
    case xls::Op::kAnd:
      return Z3_mk_bvand(ctx_, operand_asts[0], operand_asts[1]);
    case xls::Op::kZeroExt: {
      const auto* ext = node->As<xls::ExtendOp>();
      int64_t diff = ext->new_bit_count() - ext->operand(0)->BitCountOrDie();
      return Z3_mk_zero_ext(ctx_, diff, operand_asts[0]);
    }
    case xls::Op::kBitSlice: {
      const auto* slice = node->As<xls::BitSlice>();
      return Z3_mk_extract(ctx_, slice->start() + slice->width() - 1,
                           slice->start(), operand_asts[0]);
    }
    case xls::Op::kEq: {
      Z3_ast eq_res = Z3_mk_eq(ctx_, operand_asts[0], operand_asts[1]);
      return BooleanToBitVector(ctx_, eq_res);
    }
    case xls::Op::kUGt: {
      Z3_ast ugt_res = Z3_mk_bvugt(ctx_, operand_asts[0], operand_asts[1]);
      return BooleanToBitVector(ctx_, ugt_res);
    }
    case xls::Op::kTuple: {
      const xls::TupleType* tuple_type = node->GetType()->AsTupleOrDie();
      return CreateTuple(tuple_type, operand_asts);
    }
    case xls::Op::kTupleIndex: {
      const auto* tuple_index = node->As<xls::TupleIndex>();
      const xls::TupleType* tuple_type =
          tuple_index->operand(0)->GetType()->AsTupleOrDie();
      return GetTupleElement(tuple_type, operand_asts[0], tuple_index->index());
    }
    case xls::Op::kSel: {
      const auto* sel = node->As<xls::Select>();
      Z3_ast selector = operand_asts[0];
      int64_t num_cases = sel->cases().size();
      Z3_ast result = nullptr;
      if (sel->default_value().has_value()) {
        result = operand_asts.back();
      } else {
        result = operand_asts[num_cases];
        --num_cases;
      }
      int64_t sel_width = sel->selector()->BitCountOrDie();
      for (int64_t i = num_cases - 1; i >= 0; --i) {
        Z3_ast case_idx = BitsToZ3(ctx_, xls::UBits(i, sel_width));
        Z3_ast is_match = Z3_mk_eq(ctx_, selector, case_idx);
        result = Z3_mk_ite(ctx_, is_match, operand_asts[i + 1], result);
      }
      return result;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported IR operation: ", node->op()));
  }
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::EncodeMuxBranchCondition(
    const xls::Node* mux_node, int64_t arm_index) {
  if (mux_node->op() == xls::Op::kSel) {
    const auto* sel = mux_node->As<xls::Select>();
    const xls::Node* sel_operand = sel->selector();
    std::string sel_name(sel_operand->GetName());
    Z3_symbol sym = Z3_mk_string_symbol(ctx_, sel_name.c_str());
    Z3_sort sort = GetTypeSort(*sel_operand->GetType());
    Z3_ast sel_ast = Z3_mk_const(ctx_, sym, sort);

    int64_t sel_width = sel_operand->BitCountOrDie();
    int64_t num_cases = sel->cases().size();
    if (arm_index < num_cases) {
      Z3_ast target = BitsToZ3(ctx_, xls::UBits(arm_index, sel_width));
      return Z3_mk_eq(ctx_, sel_ast, target);
    }
    Z3_ast limit = BitsToZ3(ctx_, xls::UBits(num_cases, sel_width));
    return Z3_mk_bvuge(ctx_, sel_ast, limit);
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Node is not a supported multiplexer: ", mux_node->op()));
}

absl::StatusOr<Z3_ast> Z3SemanticsEncoder::EncodeParamBinding(
    const xls::Param* param, const xls::Value& concrete_value) {
  XLS_ASSIGN_OR_RETURN(Z3_ast param_ast, TranslateParam(param));
  XLS_ASSIGN_OR_RETURN(Z3_ast val_ast,
                       TranslateValue(param->GetType(), concrete_value));
  return Z3_mk_eq(ctx_, param_ast, val_ast);
}

}  // namespace xls::solvers::symex
