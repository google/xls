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

#include <cstdint>
#include <optional>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"

namespace xls::dslx {
namespace {

absl::StatusOr<BitsLikeProperties> GetBitsLikeOrError(
    const Expr* node, const Type* type, const FileTable& file_table) {
  std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type);
  if (!bits_like.has_value()) {
    return TypeInferenceErrorStatus(
        node->span(), type,
        "Operation can only be applied to bits-typed operands.", file_table);
  }
  return *bits_like;
}

absl::Status ValidateBinopShift(const Binop* binop, const Type* type,
                                const TypeInfo& ti,
                                const FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(Type * rhs_type, ti.GetItemOrError(binop->rhs()));
  XLS_ASSIGN_OR_RETURN(BitsLikeProperties rhs_bits_like,
                       GetBitsLikeOrError(binop->rhs(), rhs_type, file_table));
  XLS_ASSIGN_OR_RETURN(bool rhs_is_signed, rhs_bits_like.is_signed.GetAsBool());
  if (rhs_is_signed) {
    return TypeInferenceErrorStatus(binop->rhs()->span(), rhs_type,
                                    "Shift amount must be unsigned.",
                                    file_table);
  }
  XLS_ASSIGN_OR_RETURN(Type * lhs_type, ti.GetItemOrError(binop->lhs()));
  XLS_ASSIGN_OR_RETURN(BitsLikeProperties lhs_bits_like,
                       GetBitsLikeOrError(binop->lhs(), lhs_type, file_table));

  if (ti.IsKnownConstExpr(binop->rhs())) {
    XLS_ASSIGN_OR_RETURN(InterpValue rhs_value, ti.GetConstExpr(binop->rhs()));
    XLS_ASSIGN_OR_RETURN(uint64_t number_value,
                         rhs_value.GetBitValueUnsigned());
    const TypeDim& lhs_size = lhs_bits_like.size;
    XLS_ASSIGN_OR_RETURN(int64_t lhs_bits_count, lhs_size.GetAsInt64());
    if (lhs_bits_count < number_value) {
      return TypeInferenceErrorStatus(
          binop->rhs()->span(), rhs_type,
          absl::StrFormat(
              "Shift amount is larger than shift value bit width of %d.",
              lhs_bits_count),
          file_table);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateConcreteType(const AstNode* node, const Type* type,
                                  const TypeInfo& ti,
                                  const FileTable& file_table) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  if (const auto* literal = dynamic_cast<const Number*>(node);
      literal != nullptr) {
    // A literal can have its own explicit type annotation that ultimately
    // doesn't even fit the hard coded value. For example, `u4:0xffff`, or
    // something more subtly wrong, like `uN[N]:0xffff`, where N proves to be
    // too small.
    if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type);
        bits_like.has_value()) {
      return TryEnsureFitsInType(*literal, bits_like.value(), *type);
    }
    return TypeInferenceErrorStatus(
        literal->span(), type,
        "Non-bits type used to define a numeric literal.", file_table);
  }
  if (const auto* binop = dynamic_cast<const Binop*>(node); binop != nullptr) {
    if ((GetBinopSameTypeKinds().contains(binop->binop_kind()) ||
         GetBinopShifts().contains(binop->binop_kind())) &&
        !IsBitsLike(*type)) {
      return TypeInferenceErrorStatus(
          binop->span(), type,
          "Binary operations can only be applied to bits-typed operands.",
          file_table);
    }
    if (GetBinopLogicalKinds().contains(binop->binop_kind()) &&
        !IsBitsLikeWithNBitsAndSignedness(*type, false, 1)) {
      return TypeInferenceErrorStatus(binop->span(), type,
                                      "Logical binary operations can only be "
                                      "applied to boolean operands.",
                                      file_table);
    }
    // Confirm that the shift amount is unsigned and fits in the lhs type.
    if (GetBinopShifts().contains(binop->binop_kind())) {
      XLS_RETURN_IF_ERROR(ValidateBinopShift(binop, type, ti, file_table));
    }
  }
  if (const auto* unop = dynamic_cast<const Unop*>(node);
      unop != nullptr && !IsBitsLike(*type)) {
    return TypeInferenceErrorStatus(
        unop->span(), type,
        "Unary operations can only be applied to bits-typed operands.",
        file_table);
  }
  if (const auto* index = dynamic_cast<const Index*>(node)) {
    const Type& lhs_type = **ti.GetItem(index->lhs());
    XLS_RETURN_IF_ERROR(
        ValidateArrayTypeForIndex(*index, lhs_type, file_table));
    if (std::holds_alternative<Expr*>(index->rhs())) {
      const Type& rhs_type = **ti.GetItem(std::get<Expr*>(index->rhs()));
      return ValidateArrayIndex(*index, lhs_type, rhs_type, ti, file_table);
    }
  }
  if (const auto* tuple_index = dynamic_cast<const TupleIndex*>(node)) {
    const Type& lhs_type = **ti.GetItem(tuple_index->lhs());
    const Type& rhs_type = **ti.GetItem(tuple_index->index());
    XLS_RETURN_IF_ERROR(
        ValidateTupleTypeForIndex(*tuple_index, lhs_type, file_table));
    XLS_RETURN_IF_ERROR(
        ValidateTupleIndex(*tuple_index, lhs_type, rhs_type, ti, file_table));
  }

  // For a cast node we have to validate that the types being cast to/from are
  // compatible via the `IsAcceptableCast` predicate.
  if (const auto* cast = dynamic_cast<const Cast*>(node); cast != nullptr) {
    // Retrieve the type of the operand from the TypeInfo.
    std::optional<const Type*> from_type = ti.GetItem(cast->expr());
    XLS_RET_CHECK(from_type.has_value());
    XLS_RET_CHECK(from_type.value() != nullptr);
    XLS_RET_CHECK(type != nullptr);

    const Type& to_type = *type;
    if (!IsAcceptableCast(*from_type.value(), to_type)) {
      return TypeInferenceErrorStatus(
          cast->span(), type,
          absl::Substitute("Cannot cast from type `$0` to type `$1`",
                           from_type.value()->ToString(), to_type.ToString()),
          file_table);
    }
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
