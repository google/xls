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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_collector.h"

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

// A non-recursive visitor that contains per-node-type handlers for
// `ValidateConcreteType`.
class TypeValidator : public AstNodeVisitorWithDefault {
 public:
  explicit TypeValidator(const Type* type, const TypeInfo& ti,
                         const TypeAnnotation* annotation,
                         WarningCollector& warning_collector,
                         const FileTable& file_table)
      : type_(type),
        ti_(ti),
        annotation_(annotation),
        warning_collector_(warning_collector),
        file_table_(file_table) {}

  absl::Status HandleNumber(const Number* literal) override {
    // A literal can have its own explicit type annotation that ultimately
    // doesn't even fit the hard coded value. For example, `u4:0xffff`, or
    // something more subtly wrong, like `uN[N]:0xffff`, where N proves to be
    // too small.
    if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type_);
        bits_like.has_value()) {
      XLS_RETURN_IF_ERROR(
          TryEnsureFitsInType(*literal, bits_like.value(), *type_));
      return DefaultHandler(literal);
    }
    return TypeInferenceErrorStatus(
        literal->span(), type_,
        "Non-bits type used to define a numeric literal.", file_table_);
  }

  absl::Status HandleConstantDef(const ConstantDef* def) override {
    WarnOnInappropriateConstantName(def->name_def()->identifier(), def->span(),
                                    *def->owner(), &warning_collector_);
    return DefaultHandler(def);
  }

  absl::Status HandleBinop(const Binop* binop) override {
    if ((GetBinopSameTypeKinds().contains(binop->binop_kind()) ||
         GetBinopShifts().contains(binop->binop_kind())) &&
        !IsBitsLike(*type_)) {
      return TypeInferenceErrorStatus(
          binop->span(), type_,
          "Binary operations can only be applied to bits-typed operands.",
          file_table_);
    }
    if (GetBinopLogicalKinds().contains(binop->binop_kind()) &&
        !IsBitsLikeWithNBitsAndSignedness(*type_, false, 1)) {
      return TypeInferenceErrorStatus(binop->span(), type_,
                                      "Logical binary operations can only be "
                                      "applied to boolean operands.",
                                      file_table_);
    }
    // Confirm that the shift amount is unsigned and fits in the lhs type.
    if (GetBinopShifts().contains(binop->binop_kind())) {
      XLS_RETURN_IF_ERROR(ValidateBinopShift(*binop));
    }
    if (binop->binop_kind() == BinopKind::kConcat) {
      return ValidateConcatOperandTypes(*binop);
    }
    return DefaultHandler(binop);
  }

  absl::Status HandleUnop(const Unop* unop) override {
    if (!IsBitsLike(*type_)) {
      return TypeInferenceErrorStatus(
          unop->span(), type_,
          "Unary operations can only be applied to bits-typed operands.",
          file_table_);
    }
    return DefaultHandler(unop);
  }

  absl::Status HandleIndex(const Index* index) override {
    XLS_RETURN_IF_ERROR(absl::visit(
        Visitor{[&](Slice* slice) { return ValidateSliceLhs(index); },
                [&](WidthSlice* width_slice) -> absl::Status {
                  XLS_RETURN_IF_ERROR(ValidateSliceLhs(index));
                  return ValidateWidthSlice(index, width_slice);
                },
                [&](Expr* expr) { return ValidateNonSliceIndex(index); }},
        index->rhs()));
    return DefaultHandler(index);
  }

  absl::Status HandleTupleIndex(const TupleIndex* tuple_index) override {
    const Type& lhs_type = **ti_.GetItem(tuple_index->lhs());
    const Type& rhs_type = **ti_.GetItem(tuple_index->index());
    XLS_RETURN_IF_ERROR(
        ValidateTupleTypeForIndex(*tuple_index, lhs_type, file_table_));
    XLS_RETURN_IF_ERROR(
        ValidateTupleIndex(*tuple_index, lhs_type, rhs_type, ti_, file_table_));
    return DefaultHandler(tuple_index);
  }

  absl::Status HandleCast(const Cast* cast) override {
    // For a cast node we have to validate that the types being cast to/from are
    // compatible via the `IsAcceptableCast` predicate.

    // Retrieve the type of the operand from the TypeInfo.
    std::optional<const Type*> from_type = ti_.GetItem(cast->expr());
    XLS_RET_CHECK(from_type.has_value());
    XLS_RET_CHECK(from_type.value() != nullptr);
    XLS_RET_CHECK(type_ != nullptr);

    const Type& to_type = *type_;
    if (!IsAcceptableCast(*from_type.value(), to_type)) {
      return TypeInferenceErrorStatus(
          cast->span(), type_,
          absl::Substitute("Cannot cast from type `$0` to type `$1`",
                           from_type.value()->ToString(), to_type.ToString()),
          file_table_);
    }
    return DefaultHandler(cast);
  }

  // TODO: In type_annotation_resolver.cc ResolveElementType, if the container
  // type of ElementTypeNotation is not a subscriptable type, it just returns
  // the container type without reporting an error, so it must be checked here.
  // This check can be removed after figuring out the reasoning behind that.
  absl::Status HandleFor(const For* forexpr) override {
    const Type* type = *ti_.GetItem(forexpr->iterable());
    if (!dynamic_cast<const ArrayType*>(type)) {
      return TypeInferenceErrorStatus(
          forexpr->iterable()->span(), type,
          absl::Substitute("Expect array type annotation on a for loop's "
                           "iterable, got `$0`.",
                           type->ToString()),
          file_table_);
    }
    return DefaultHandler(forexpr);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    if (node->parent() != nullptr) {
      if (const auto* binop = dynamic_cast<Binop*>(node->parent());
          binop != nullptr && binop->binop_kind() == BinopKind::kConcat &&
          (binop->lhs() == node || binop->rhs() == node)) {
        XLS_RETURN_IF_ERROR(
            PreValidateConcatOperand(dynamic_cast<const Expr*>(node)));
      }
    }
    return absl::OkStatus();
  }

  // Check if the size of the range is a non-negative number that fits a u32.
  absl::Status HandleRange(const Range* range) override {
    // If range's type annotation is inferenced, its dimension is a subtract
    // expr of S32 operands and we only need to validate for this case.
    const ArrayTypeAnnotation* array_type =
        dynamic_cast<const ArrayTypeAnnotation*>(annotation_);
    if (!array_type) {
      return absl::OkStatus();
    }
    const Binop* binop = dynamic_cast<const Binop*>(array_type->dim());
    if (!binop || binop->binop_kind() != BinopKind::kSub) {
      return absl::OkStatus();
    }
    for (const Expr* operand : {binop->lhs(), binop->rhs()}) {
      const Cast* cast = dynamic_cast<const Cast*>(operand);
      if (!cast) {
        return absl::OkStatus();
      }
      const BuiltinTypeAnnotation* s32_type =
          dynamic_cast<const BuiltinTypeAnnotation*>(cast->type_annotation());
      if (!s32_type || s32_type->builtin_type() != BuiltinType::kS32) {
        return absl::OkStatus();
      }
    }

    XLS_ASSIGN_OR_RETURN(InterpValue start,
                         ti_.GetConstExpr(binop->rhs()->GetChildren(false)[0]));
    XLS_ASSIGN_OR_RETURN(InterpValue end,
                         ti_.GetConstExpr(binop->lhs()->GetChildren(false)[0]));
    if (start.Gt(end)->IsTrue()) {
      return TypeInferenceErrorStatus(
          range->span(), type_,
          absl::Substitute(
              "Range expr `$0` start value `$1` is larger than end value `$2`",
              range->ToString(), start.ToString(true), end.ToString(true)),
          file_table_);
    }
    XLS_ASSIGN_OR_RETURN(int64_t start_bits, start.GetBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t end_bits, end.GetBitCount());
    int64_t bits = std::max(start_bits, end_bits);
    XLS_ASSIGN_OR_RETURN(
        start, start.IsSigned() ? start.SignExt(bits) : start.ZeroExt(bits));
    XLS_ASSIGN_OR_RETURN(
        end, end.IsSigned() ? end.SignExt(bits) : end.ZeroExt(bits));
    if (!end.Sub(start)->FitsInNBitsUnsigned(32)) {
      return TypeInferenceErrorStatus(
          range->span(), type_,
          absl::Substitute("Range expr `$0` has size `$1` larger than u32",
                           range->ToString(), end.Sub(start)->ToString(true)),
          file_table_);
    }
    return DefaultHandler(range);
  }

 private:
  absl::Status ValidateBinopShift(const Binop& binop) {
    XLS_ASSIGN_OR_RETURN(Type * rhs_type, ti_.GetItemOrError(binop.rhs()));
    XLS_ASSIGN_OR_RETURN(
        BitsLikeProperties rhs_bits_like,
        GetBitsLikeOrError(binop.rhs(), rhs_type, file_table_));
    XLS_ASSIGN_OR_RETURN(bool rhs_is_signed,
                         rhs_bits_like.is_signed.GetAsBool());
    if (rhs_is_signed) {
      return TypeInferenceErrorStatus(binop.rhs()->span(), rhs_type,
                                      "Shift amount must be unsigned.",
                                      file_table_);
    }
    XLS_ASSIGN_OR_RETURN(Type * lhs_type, ti_.GetItemOrError(binop.lhs()));
    XLS_ASSIGN_OR_RETURN(
        BitsLikeProperties lhs_bits_like,
        GetBitsLikeOrError(binop.lhs(), lhs_type, file_table_));

    if (ti_.IsKnownConstExpr(binop.rhs())) {
      XLS_ASSIGN_OR_RETURN(InterpValue rhs_value,
                           ti_.GetConstExpr(binop.rhs()));
      XLS_ASSIGN_OR_RETURN(uint64_t number_value,
                           rhs_value.GetBitValueUnsigned());
      const TypeDim& lhs_size = lhs_bits_like.size;
      XLS_ASSIGN_OR_RETURN(int64_t lhs_bits_count, lhs_size.GetAsInt64());
      if (lhs_bits_count < number_value) {
        return TypeInferenceErrorStatus(
            binop.rhs()->span(), rhs_type,
            absl::StrFormat(
                "Shift amount is larger than shift value bit width of %d.",
                lhs_bits_count),
            file_table_);
      }
    }
    return absl::OkStatus();
  }

  // Checks for pitfalls with a concat operand. This is meant to be used on the
  // operands at the time of their own validation, which is before the concat
  // node itself is concretized. If we were to defer it until the concat node is
  // concretized and validated itself, then in a case like
  //   `let x: s2 = s1:0 ++ s1:1`
  // it would be unification of the s2 and the automatic u2 type of the concat
  // result that would fail, yielding a more puzzling error.
  absl::Status PreValidateConcatOperand(const Expr* expr) {
    std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type_);
    if (bits_like.has_value()) {
      XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());
      if (is_signed) {
        return TypeInferenceErrorStatus(
            expr->span(), nullptr,
            absl::StrCat("Concatenation requires operand types to both be "
                         "unsigned bits; got: ",
                         type_->ToString()),
            file_table_);
      }
      return absl::OkStatus();
    }
    if (!type_->IsArray()) {
      return TypeInferenceErrorStatus(
          expr->span(), type_,
          absl::StrCat("Concatenation requires operand types to be either "
                       "both-arrays or both-bits; got: ",
                       type_->ToString()),
          file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status ValidateConcatOperandTypes(const Binop& concat) {
    const Type* lhs = *ti_.GetItem(concat.lhs());
    const Type* rhs = *ti_.GetItem(concat.rhs());
    std::optional<BitsLikeProperties> lhs_bits_like = GetBitsLike(*lhs);
    std::optional<BitsLikeProperties> rhs_bits_like = GetBitsLike(*rhs);
    if (lhs_bits_like.has_value() && rhs_bits_like.has_value()) {
      // Bits-like operands would have been validated in detail by
      // `PreValidateConcatOperand` when those operands are concretized and
      // validated, which is before the Binop we are currently processing.
      return absl::OkStatus();
    }

    const auto* lhs_array = dynamic_cast<const ArrayType*>(lhs);
    const auto* rhs_array = dynamic_cast<const ArrayType*>(rhs);
    bool lhs_is_array = lhs_array != nullptr && !lhs_bits_like.has_value();
    bool rhs_is_array = rhs_array != nullptr && !rhs_bits_like.has_value();

    if (lhs_is_array != rhs_is_array) {
      return TypeInferenceErrorStatus(
          concat.span(), nullptr,
          absl::StrFormat("Attempting to concatenate array/non-array "
                          "values together; got lhs: `%s`; rhs: `%s`.",
                          lhs->ToString(), rhs->ToString()),
          file_table_);
    }

    if (lhs_is_array) {
      if (lhs_array->element_type() != rhs_array->element_type()) {
        return TypeMismatchErrorStatus(
            lhs_array->element_type(), rhs_array->element_type(),
            concat.lhs()->span(), concat.rhs()->span(), file_table_);
      }
      return absl::OkStatus();
    }

    if (lhs->HasEnum() || rhs->HasEnum()) {
      return TypeInferenceErrorStatus(
          concat.span(), nullptr,
          absl::StrFormat("Enum values must be cast to unsigned bits before "
                          "concatenation; got lhs: `%s`; rhs: `%s`",
                          lhs->ToString(), rhs->ToString()),
          file_table_);
    }
    return TypeInferenceErrorStatus(
        concat.span(), nullptr,
        absl::StrFormat(
            "Concatenation requires operand types to be "
            "either both-arrays or both-bits; got lhs: `%s`; rhs: `%s`",
            lhs->ToString(), rhs->ToString()),
        file_table_);
  }

  absl::Status ValidateSliceLhs(const Index* index) {
    const Type& lhs_type = **ti_.GetItem(index->lhs());
    // Type inference v2 deduces array slices, while in v1 this was a planned
    // task that was never done. For now, we artificially restrict it after the
    // fact, in case there are downstream issues.
    std::optional<BitsLikeProperties> lhs_bits_like = GetBitsLike(lhs_type);
    if (!lhs_bits_like.has_value()) {
      return TypeInferenceErrorStatus(index->span(), &lhs_type,
                                      "Value to slice is not of 'bits' type.",
                                      file_table_);
    }
    XLS_ASSIGN_OR_RETURN(bool lhs_is_signed,
                         lhs_bits_like->is_signed.GetAsBool());
    if (lhs_is_signed) {
      return TypeInferenceErrorStatus(index->span(), &lhs_type,
                                      "Bit slice LHS must be unsigned.",
                                      file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status ValidateWidthSlice(const Index* index,
                                  const WidthSlice* width_slice) {
    const Type& width_type = **ti_.GetItem(width_slice->width());
    if (!IsBitsLike(width_type)) {
      return TypeInferenceErrorStatus(
          index->span(), &width_type,
          "A bits type is required for a width-based slice.", file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status ValidateNonSliceIndex(const Index* index) {
    const Type& lhs_type = **ti_.GetItem(index->lhs());
    XLS_RETURN_IF_ERROR(
        ValidateArrayTypeForIndex(*index, lhs_type, file_table_));
    if (std::holds_alternative<Expr*>(index->rhs())) {
      const Type& rhs_type = **ti_.GetItem(std::get<Expr*>(index->rhs()));
      return ValidateArrayIndex(*index, lhs_type, rhs_type, ti_, file_table_);
    }
    return absl::OkStatus();
  }

  const Type* type_;
  const TypeInfo& ti_;
  const TypeAnnotation* annotation_;
  WarningCollector& warning_collector_;
  const FileTable& file_table_;
};

}  // namespace

absl::Status ValidateConcreteType(const AstNode* node, const Type* type,
                                  const TypeInfo& ti,
                                  const TypeAnnotation* annotation,
                                  WarningCollector& warning_collector,
                                  const FileTable& file_table) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  TypeValidator validator(type, ti, annotation, warning_collector, file_table);
  return node->Accept(&validator);
}

}  // namespace xls::dslx
