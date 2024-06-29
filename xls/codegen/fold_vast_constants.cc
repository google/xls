// Copyright 2024 The XLS Authors
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

#include "xls/codegen/fold_vast_constants.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls {
namespace verilog {
namespace {

// Based upon the util version.
uint64_t Log2Ceiling(uint64_t n) {
  int floor = (absl::bit_width(n) - 1);
  return floor + (((n & (n - 1)) == 0) ? 0 : 1);
}

// Helper class used internally by the exposed API to bind a type map and avoid
// passing it around.
class ConstantFoldingContext {
 public:
  explicit ConstantFoldingContext(
      const absl::flat_hash_map<Expression*, DataType*>& type_map)
      : type_map_(type_map) {}

  absl::StatusOr<Expression*> FoldConstants(Expression* expr) {
    if (expr->IsLiteral()) {
      return expr;
    }
    if (auto* ref = dynamic_cast<ParameterRef*>(expr);
        ref && ref->parameter()->rhs()) {
      return FoldConstants(ref->parameter()->rhs());
    }
    if (auto* ref = dynamic_cast<EnumMemberRef*>(expr);
        ref && ref->member()->rhs()) {
      return FoldConstants(ref->member()->rhs());
    }
    if (auto* op = dynamic_cast<BinaryInfix*>(expr); op) {
      return FoldBinaryOp(op);
    }
    if (auto* ternary = dynamic_cast<Ternary*>(expr); ternary) {
      absl::StatusOr<int64_t> test_value = FoldEntireExpr(ternary->test());
      absl::StatusOr<int64_t> consequent_value =
          FoldEntireExpr(ternary->consequent());
      absl::StatusOr<int64_t> alternate_value =
          FoldEntireExpr(ternary->alternate());
      if (test_value.ok() && consequent_value.ok() && alternate_value.ok()) {
        return *test_value ? MakeFoldedConstant(ternary, *consequent_value)
                           : MakeFoldedConstant(ternary, *alternate_value);
      }
    }
    if (auto* call = dynamic_cast<SystemFunctionCall*>(expr); call) {
      if (call->name() == "clog2" && call->args().has_value() &&
          (*call->args()).size() == 1) {
        absl::StatusOr<int64_t> arg_value = FoldEntireExpr((*call->args())[0]);
        if (arg_value.ok()) {
          return MakeFoldedConstant(call, Log2Ceiling(*arg_value));
        }
      }
    }
    return expr;
  }

  absl::StatusOr<int64_t> FoldEntireExpr(Expression* expr) {
    XLS_ASSIGN_OR_RETURN(Expression * folded, FoldConstants(expr));
    if (folded->IsLiteral()) {
      return folded->AsLiteralOrDie()->ToInt64();
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Expression does not entirely fold to a constant: ",
                     expr->Emit(nullptr)));
  }

  absl::StatusOr<Literal*> MakeFoldedConstant(Expression* original,
                                              int64_t value) {
    int64_t effective_size = 32;
    bool is_signed = true;
    const auto it = type_map_.find(original);
    if (it != type_map_.end()) {
      XLS_ASSIGN_OR_RETURN(effective_size, it->second->FlatBitCountAsInt64());
      is_signed = it->second->is_signed();
    }
    return original->file()->Make<Literal>(
        original->loc(),
        is_signed ? SBits(value, effective_size)
                  : UBits(static_cast<uint64_t>(value), effective_size),
        value < 0 ? FormatPreference::kHex : FormatPreference::kUnsignedDecimal,
        /*effective_bit_count=*/effective_size,
        /*emit_bit_count=*/effective_size != 32,
        /*declared_as_signed=*/is_signed);
  }

  absl::StatusOr<DataType*> FoldConstants(DataType* data_type) {
    if (data_type->FlatBitCountAsInt64().ok()) {
      return data_type;
    }
    if (auto* bit_vector_type = dynamic_cast<BitVectorType*>(data_type);
        bit_vector_type && !bit_vector_type->size_expr()->IsLiteral()) {
      XLS_ASSIGN_OR_RETURN(int64_t folded_size,
                           FoldEntireExpr(bit_vector_type->size_expr()));
      return data_type->file()->Make<BitVectorType>(
          data_type->loc(),
          data_type->file()->PlainLiteral(static_cast<int32_t>(folded_size),
                                          data_type->loc()),
          data_type->is_signed(),
          /*size_expr_is_max=*/bit_vector_type->max().has_value());
    }
    if (auto* array_type = dynamic_cast<UnpackedArrayType*>(data_type);
        array_type) {
      XLS_ASSIGN_OR_RETURN(std::vector<int64_t> folded_dims,
                           FoldDims(array_type->dims()));
      XLS_ASSIGN_OR_RETURN(DataType * folded_element_type,
                           FoldConstants(array_type->element_type()));
      return array_type->file()->template Make<UnpackedArrayType>(
          data_type->loc(), folded_element_type, folded_dims);
    }
    if (auto* array_type = dynamic_cast<PackedArrayType*>(data_type);
        array_type) {
      XLS_ASSIGN_OR_RETURN(std::vector<int64_t> folded_dims,
                           FoldDims(array_type->dims()));
      XLS_ASSIGN_OR_RETURN(DataType * folded_element_type,
                           FoldConstants(array_type->element_type()));
      return array_type->file()->template Make<PackedArrayType>(
          data_type->loc(), folded_element_type, folded_dims,
          /*dims_are_max=*/array_type->dims_are_max());
    }
    if (auto* enum_def = dynamic_cast<Enum*>(data_type); enum_def) {
      XLS_ASSIGN_OR_RETURN(DataType * folded_base_type,
                           FoldConstants(enum_def->BaseType()));
      return data_type->file()->Make<Enum>(enum_def->loc(), enum_def->kind(),
                                           folded_base_type,
                                           enum_def->members());
    }
    if (auto* struct_def = dynamic_cast<Struct*>(data_type); struct_def) {
      XLS_ASSIGN_OR_RETURN(std::vector<Def*> folded_member_defs,
                           FoldTypesOfDefs(struct_def->members()));
      return struct_def->file()->Make<Struct>(struct_def->loc(),
                                              folded_member_defs);
    }
    if (auto* type_def_type = dynamic_cast<TypedefType*>(data_type);
        type_def_type) {
      VerilogFile* file = type_def_type->file();
      Typedef* type_def = type_def_type->type_def();
      XLS_ASSIGN_OR_RETURN(DataType * folded_base_type,
                           FoldConstants(type_def_type->BaseType()));
      return file->Make<TypedefType>(
          type_def_type->loc(),
          file->Make<Typedef>(
              type_def->loc(),
              file->Make<Def>(type_def->loc(), type_def->GetName(),
                              type_def->data_kind(), folded_base_type)));
    }
    return absl::InternalError(absl::StrCat("Could not constant-fold type: ",
                                            data_type->Emit(nullptr)));
  }

 private:
  absl::StatusOr<std::vector<int64_t>> FoldDims(
      absl::Span<Expression* const> dims) {
    std::vector<int64_t> result(dims.size());
    result.reserve(dims.size());
    int i = 0;
    for (Expression* dim : dims) {
      XLS_ASSIGN_OR_RETURN(result[i++], FoldEntireExpr(dim));
    }
    return result;
  }

  absl::StatusOr<std::vector<Def*>> FoldTypesOfDefs(
      absl::Span<Def* const> defs) {
    std::vector<Def*> result(defs.size());
    int i = 0;
    for (Def* def : defs) {
      XLS_ASSIGN_OR_RETURN(DataType * folded_type,
                           FoldConstants(def->data_type()));
      result[i++] = def->file()->Make<Def>(
          def->loc(), def->GetName(), def->data_kind(), folded_type,
          def->init().has_value() ? *def->init() : nullptr);
    }
    return result;
  }

  absl::StatusOr<Expression*> FoldBinaryOp(Operator* op) {
    auto* binop = dynamic_cast<BinaryInfix*>(op);
    XLS_ASSIGN_OR_RETURN(Expression * folded_lhs, FoldConstants(binop->lhs()));
    XLS_ASSIGN_OR_RETURN(Expression * folded_rhs, FoldConstants(binop->rhs()));
    if (!folded_lhs->IsLiteral() || !folded_rhs->IsLiteral()) {
      return op->file()->Make<BinaryInfix>(op->loc(), folded_lhs, folded_rhs,
                                           op->kind());
    }
    Literal* lhs_literal = folded_lhs->AsLiteralOrDie();
    Literal* rhs_literal = folded_rhs->AsLiteralOrDie();
    XLS_ASSIGN_OR_RETURN(int64_t lhs_value, lhs_literal->ToInt64());
    XLS_ASSIGN_OR_RETURN(int64_t rhs_value, rhs_literal->ToInt64());
    bool signed_input = lhs_literal->is_declared_as_signed() &&
                        rhs_literal->is_declared_as_signed();
    std::optional<bool> bool_result;
    std::optional<int64_t> int_result;
    switch (op->kind()) {
      case OperatorKind::kAdd:
        int_result = lhs_value + rhs_value;
        break;
      case OperatorKind::kSub:
        int_result = lhs_value - rhs_value;
        break;
      case OperatorKind::kMul:
        int_result = lhs_value * rhs_value;
        break;
      case OperatorKind::kDiv:
        int_result = lhs_value / rhs_value;
        break;
      case OperatorKind::kMod:
        int_result = lhs_value % rhs_value;
        break;
      case OperatorKind::kEq:
        bool_result = lhs_value == rhs_value;
        break;
      case OperatorKind::kNe:
        bool_result = lhs_value != rhs_value;
        break;
      case OperatorKind::kGe:
        bool_result = signed_input ? lhs_value >= rhs_value
                                   : static_cast<uint64_t>(lhs_value) >=
                                         static_cast<uint64_t>(rhs_value);
        break;
      case OperatorKind::kGt:
        bool_result = signed_input ? lhs_value > rhs_value
                                   : static_cast<uint64_t>(lhs_value) >
                                         static_cast<uint64_t>(rhs_value);
        break;
      case OperatorKind::kLe:
        bool_result = signed_input ? lhs_value <= rhs_value
                                   : static_cast<uint64_t>(lhs_value) <=
                                         static_cast<uint64_t>(rhs_value);
        break;
      case OperatorKind::kLt:
        bool_result = signed_input ? lhs_value < rhs_value
                                   : static_cast<uint64_t>(lhs_value) <
                                         static_cast<uint64_t>(rhs_value);
        break;
      default:
        break;
    }
    if (int_result.has_value()) {
      return MakeFoldedConstant(op, *int_result);
    }
    if (bool_result.has_value()) {
      return MakeFoldedConstant(op, static_cast<int64_t>(*bool_result));
    }
    return op->file()->Make<BinaryInfix>(op->loc(), folded_lhs, folded_rhs,
                                         op->kind());
  }

  const absl::flat_hash_map<Expression*, DataType*>& type_map_;
};

}  // namespace

absl::StatusOr<DataType*> FoldVastConstants(
    DataType* data_type,
    const absl::flat_hash_map<Expression*, DataType*>& type_map) {
  auto context = std::make_unique<ConstantFoldingContext>(type_map);
  return context->FoldConstants(data_type);
}

absl::StatusOr<int64_t> FoldEntireVastExpr(
    Expression* expr,
    const absl::flat_hash_map<Expression*, DataType*>& type_map) {
  auto context = std::make_unique<ConstantFoldingContext>(type_map);
  return context->FoldEntireExpr(expr);
}

absl::StatusOr<Expression*> FoldVastConstants(
    Expression* expr,
    const absl::flat_hash_map<Expression*, DataType*>& type_map) {
  auto context = std::make_unique<ConstantFoldingContext>(type_map);
  return context->FoldConstants(expr);
}

}  // namespace verilog
}  // namespace xls
