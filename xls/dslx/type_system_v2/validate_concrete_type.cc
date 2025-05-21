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
#include <string_view>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/exhaustiveness/match_exhaustiveness_checker.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

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
                         const ImportData& import_data,
                         const FileTable& file_table)
      : type_(type),
        ti_(ti),
        annotation_(annotation),
        warning_collector_(warning_collector),
        import_data_(import_data),
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

  absl::Status HandleNameRef(const NameRef* ref) override {
    if (IsNameRefToParametricFunction(ref) && !type_->IsFunction()) {
      // Bare parametric function references that are not invocations don't get
      // typechecked the normal way, because we only annotate their types when
      // invocations are encountered, hence the need for a check here.
      return TypeInferenceErrorStatus(
          ref->span(), type_,
          absl::Substitute("Expected type `$0` but got `$1`, which is a "
                           "parametric function not being invoked.",
                           type_->ToString(), ref->ToString()),
          file_table_);
    }
    return DefaultHandler(ref);
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
                  return ValidateSliceLhs(index);
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

  absl::Status HandleInvocation(const Invocation* invocation) override {
    using BuiltinValidator =
        absl::Status (TypeValidator::*)(const Invocation&, const FunctionType&);
    static const auto* builtin_validators =
        new absl::flat_hash_map<std::string_view, BuiltinValidator>{
            {"decode", &TypeValidator::ValidateDecodeInvocation},
            {"update", &TypeValidator::ValidateUpdateInvocation},
            {"widening_cast", &TypeValidator::ValidateWideningCastInvocation}};

    std::optional<std::string_view> builtin =
        GetBuiltinFnName(invocation->callee());
    if (builtin.has_value()) {
      const auto it = builtin_validators->find(*builtin);
      if (it != builtin_validators->end()) {
        const auto& signature = dynamic_cast<const FunctionType&>(
            **ti_.GetItem(invocation->callee()));
        return (this->*it->second)(*invocation, signature);
      }
    }
    return DefaultHandler(invocation);
  }

  absl::Status HandleFormatMacro(const FormatMacro* macro) override {
    for (const Expr* arg : macro->args()) {
      const Type& type = **ti_.GetItem(arg);
      XLS_RETURN_IF_ERROR(
          ValidateFormatMacroArgument(type, arg->span(), file_table_));
    }
    return absl::OkStatus();
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
    if (!type->IsArray()) {
      return TypeInferenceErrorStatus(
          forexpr->iterable()->span(), type,
          absl::Substitute("Expect array type annotation on a for loop's "
                           "iterable, got `$0`.",
                           type->ToString()),
          file_table_);
    }
    return DefaultHandler(forexpr);
  }

  absl::Status HandleQuickCheck(const QuickCheck* node) override {
    std::optional<const Type*> fn_return_type = ti_.GetItem(node->fn()->body());
    CHECK(fn_return_type.has_value());
    std::optional<BitsLikeProperties> fn_return_type_bits_like =
        GetBitsLike(**fn_return_type);
    std::optional<BitsLikeProperties> node_bits_like = GetBitsLike(*type_);
    CHECK(node_bits_like.has_value());
    if (*node_bits_like != *fn_return_type_bits_like) {
      return TypeMismatchErrorStatus(*type_, **fn_return_type, node->span(),
                                     node->fn()->body()->span(), file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status HandleMatch(const Match* node) override {
    std::optional<const Type*> matched_type = ti_.GetItem(node->matched());
    CHECK(matched_type.has_value());

    Type* matched = const_cast<Type*>(*matched_type);
    MatchExhaustivenessChecker exhaustiveness_checker(
        node->matched()->span(), import_data_, ti_, *matched);

    for (MatchArm* arm : node->arms()) {
      for (NameDefTree* pattern : arm->patterns()) {
        bool exhaustive_before = exhaustiveness_checker.IsExhaustive();
        exhaustiveness_checker.AddPattern(*pattern);
        if (exhaustive_before) {
          warning_collector_.Add(
              pattern->span(), WarningKind::kAlreadyExhaustiveMatch,
              "`match` is already exhaustive before this pattern");
        }
      }
    }

    if (!exhaustiveness_checker.IsExhaustive()) {
      std::optional<InterpValue> sample =
          exhaustiveness_checker.SampleSimplestUncoveredValue();
      XLS_RET_CHECK(sample.has_value());
      return TypeInferenceErrorStatus(
          node->span(), matched,
          absl::StrFormat(
              "`match` patterns are not exhaustive; e.g. `%s` is not covered; "
              "please add additional patterns to complete the match or a "
              "default case via `_ => ...`",
              sample->ToString()),
          file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status HandleRange(const Range* range) override {
    if (IsRangeInMatchArm(range)) {
      return absl::OkStatus();
    }
    XLS_ASSIGN_OR_RETURN(
        InterpValue start,
        ConstexprEvaluator::EvaluateToValue(
            const_cast<ImportData*>(&import_data_), const_cast<TypeInfo*>(&ti_),
            &warning_collector_, ParametricEnv(), range->start(), type_));
    XLS_ASSIGN_OR_RETURN(
        InterpValue end,
        ConstexprEvaluator::EvaluateToValue(
            const_cast<ImportData*>(&import_data_), const_cast<TypeInfo*>(&ti_),
            &warning_collector_, ParametricEnv(), range->end(), type_));

    if (start.Gt(end)->IsTrue()) {
      return RangeStartGreaterThanEndErrorStatus(range->span(), range, start,
                                                 end, file_table_);
    }
    XLS_ASSIGN_OR_RETURN(InterpValue diff, end.Sub(start));
    if (range->inclusive_end()) {
      diff = InterpValue::MakeUnsigned(diff.GetBitsOrDie());
      XLS_ASSIGN_OR_RETURN(diff, diff.IncrementZeroExtendIfOverflow());
    }
    if (!diff.FitsInNBitsUnsigned(32)) {
      return RangeTooLargeErrorStatus(range->span(), range, diff, file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    if (node->parent() != nullptr &&
        node->parent()->kind() == AstNodeKind::kBinop) {
      if (const auto* binop = dynamic_cast<Binop*>(node->parent());
          binop->binop_kind() == BinopKind::kConcat &&
          (binop->lhs() == node || binop->rhs() == node)) {
        XLS_RETURN_IF_ERROR(
            PreValidateConcatOperand(dynamic_cast<const Expr*>(node)));
      }
    }
    return absl::OkStatus();
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

  absl::Status ValidateDecodeInvocation(const Invocation& invocation,
                                        const FunctionType& signature) {
    const Type& return_type = signature.return_type();
    // First, it must be a bits type!
    if (!IsBitsLike(return_type)) {
      return TypeInferenceErrorStatus(
          invocation.span(), &return_type,
          absl::Substitute("`decode` return type must be a bits type, saw `$0`",
                           return_type.ToString()),
          file_table_);
    }
    // Second, it must be unsigned.
    XLS_ASSIGN_OR_RETURN(bool is_signed, IsSigned(return_type));
    if (is_signed) {
      return TypeInferenceErrorStatus(
          invocation.span(), &return_type,
          absl::Substitute("`decode` return type must be unsigned, saw `$0`",
                           return_type.ToString()),
          file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status ValidateUpdateIndexUnsigned(const Span& span, const Type& type) {
    if (!IsBitsLike(type)) {
      return TypeInferenceErrorStatus(
          span, &type,
          absl::Substitute("`update` index type must be a bits type; got `$0`",
                           type.ToString()),
          file_table_);
    }
    XLS_ASSIGN_OR_RETURN(bool is_signed, IsSigned(type));
    if (is_signed) {
      return TypeInferenceErrorStatus(
          span, &type,
          absl::Substitute("`update` index type must be unsigned; got `$0`",
                           type.ToString()),
          file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status ValidateUpdateInvocation(const Invocation& invocation,
                                        const FunctionType& signature) {
    const Type& first = *signature.params()[0];
    const ArrayType& array_type = first.AsArray();
    const Type& index_type = *signature.params()[1];

    // 1. index must be tuple of uNs or scalar uN
    if (index_type.IsTuple()) {
      const TupleType& index_tuple = index_type.AsTuple();
      for (int i = 0; i < index_tuple.size(); ++i) {
        XLS_RETURN_IF_ERROR(ValidateUpdateIndexUnsigned(
            invocation.span(), index_tuple.GetMemberType(i)));
      }
    } else {
      XLS_RETURN_IF_ERROR(
          ValidateUpdateIndexUnsigned(invocation.span(), index_type));
    }

    // 2. `value` must be an appropriate type for each index
    int index_dimensions =
        index_type.IsTuple() ? index_type.AsTuple().size() : 1;
    const Type* element_type = &array_type;
    for (int64_t i = 0; i < index_dimensions; ++i) {
      if (element_type->IsArray()) {
        element_type = &element_type->AsArray().element_type();
      } else {
        return TypeInferenceErrorStatus(
            invocation.span(), &index_type,
            absl::Substitute("Array dimension in `update` expected to be "
                             "larger than the number of indices ($0); got $1",
                             index_dimensions,
                             array_type.GetAllDims().size() - 1),
            file_table_);
      }
    }

    // 3. Base types must match
    const Type& value_type = *signature.params()[2];
    if (*element_type != value_type) {
      return TypeMismatchErrorStatus(*element_type, value_type,
                                     invocation.span(), invocation.span(),
                                     file_table_);
    }

    return absl::OkStatus();
  }

  absl::Status ValidateWideningCastInvocation(const Invocation& invocation,
                                              const FunctionType& signature) {
    // This logic is based on `TypecheckIsAcceptableWideningCast` in v1, which
    // is forked here due to nonstandard error handling.
    const Type& from = *signature.params()[0];
    const Type& to = signature.return_type();
    std::optional<BitsLikeProperties> from_bits_like = GetBitsLike(from);
    std::optional<BitsLikeProperties> to_bits_like = GetBitsLike(to);

    if (!from_bits_like.has_value() || !to_bits_like.has_value()) {
      return TypeInferenceErrorStatus(
          invocation.span(), &to,
          absl::StrFormat(
              "widening_cast must cast bits to bits, not `%s` to `%s`.",
              from.ToErrorString(), to.ToErrorString()),
          file_table_);
    }

    XLS_ASSIGN_OR_RETURN(bool signed_input,
                         from_bits_like->is_signed.GetAsBool());
    XLS_ASSIGN_OR_RETURN(bool signed_output,
                         to_bits_like->is_signed.GetAsBool());

    XLS_ASSIGN_OR_RETURN(int64_t old_bit_count,
                         from_bits_like->size.GetAsInt64());
    XLS_ASSIGN_OR_RETURN(int64_t new_bit_count,
                         to_bits_like->size.GetAsInt64());

    bool can_cast =
        ((signed_input == signed_output) && (new_bit_count >= old_bit_count)) ||
        (!signed_input && signed_output && (new_bit_count > old_bit_count));

    if (!can_cast) {
      return TypeInferenceErrorStatus(
          invocation.span(), &to,
          absl::StrFormat("Cannot cast from type `%s` (%d bits) to `%s` (%d "
                          "bits) with widening_cast",
                          ToTypeString(from_bits_like.value()), old_bit_count,
                          ToTypeString(to_bits_like.value()), new_bit_count),
          file_table_);
    }

    return absl::OkStatus();
  }

  const Type* type_;
  const TypeInfo& ti_;
  const TypeAnnotation* annotation_;
  WarningCollector& warning_collector_;
  const ImportData& import_data_;
  const FileTable& file_table_;
};

}  // namespace

absl::Status ValidateConcreteType(const AstNode* node, const Type* type,
                                  const TypeInfo& ti,
                                  const TypeAnnotation* annotation,
                                  WarningCollector& warning_collector,
                                  const ImportData& import_data,
                                  const FileTable& file_table) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  TypeValidator validator(type, ti, annotation, warning_collector, import_data,
                          file_table);
  return node->Accept(&validator);
}

}  // namespace xls::dslx
