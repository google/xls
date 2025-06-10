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

#include "xls/dslx/type_system_v2/solve_for_parametrics.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/zip_ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

bool operator==(InterpValueOrTypeAnnotation lhs,
                InterpValueOrTypeAnnotation rhs) {
  const bool lhs_is_interp_value = std::holds_alternative<InterpValue>(lhs);
  const bool rhs_is_interp_value = std::holds_alternative<InterpValue>(rhs);
  if (lhs_is_interp_value ^ rhs_is_interp_value) {
    return false;
  }
  return lhs_is_interp_value
             ? std::get<InterpValue>(lhs).Eq(std::get<InterpValue>(rhs))
             : std::get<const TypeAnnotation*>(lhs)->ToString() ==
                   std::get<const TypeAnnotation*>(rhs)->ToString();
}

// Unwraps the given type annotation if it refers to an alias to a struct;
// otherwise returns the same annotation.
absl::StatusOr<const TypeRefTypeAnnotation*> CanonicalizeTypeRefTypeAnnotation(
    const TypeRefTypeAnnotation* annotation, ImportData& import_data) {
  XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_ref,
                       GetStructOrProcRef(annotation, import_data));
  if (!struct_ref.has_value() ||
      struct_ref->def == ToAstNode(annotation->type_ref()->type_definition())) {
    return annotation;
  }
  XLS_ASSIGN_OR_RETURN(
      TypeDefinition type_def,
      ToTypeDefinition(const_cast<StructDefBase*>(struct_ref->def)));
  return annotation->owner()->Make<TypeRefTypeAnnotation>(
      annotation->span(),
      annotation->owner()->Make<TypeRef>(annotation->span(), type_def),
      struct_ref->parametrics);
}

// Used on both the resolvable and dependent annotations, to keep track of what
// was last encountered on each.
class Visitor : public AstNodeVisitorWithDefault {
 public:
  absl::Status HandleNameRef(const NameRef* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    if (std::holds_alternative<const NameDef*>(node->name_def())) {
      last_variable_ = std::get<const NameDef*>(node->name_def());
    }
    return absl::OkStatus();
  }

  absl::Status HandleTypeVariableTypeAnnotation(
      const TypeVariableTypeAnnotation* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    last_tvta_ = node;
    return absl::OkStatus();
  }

  absl::Status HandleBuiltinTypeAnnotation(
      const BuiltinTypeAnnotation* node) override {
    return HandlePossibleIntegerAnnotation(node);
  }

  absl::Status HandleArrayTypeAnnotation(
      const ArrayTypeAnnotation* node) override {
    absl::Status initial_result = HandlePossibleIntegerAnnotation(node);
    if (initial_result.ok() && !last_signedness_and_bit_count_.has_value()) {
      // Not an integer annotation, but an array of something that it still
      // needs to retain for resolution later.
      last_direct_annotation_ = node;
    }
    return initial_result;
  }

  absl::Status HandleTupleTypeAnnotation(
      const TupleTypeAnnotation* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    last_direct_annotation_ = node;
    return absl::OkStatus();
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    last_direct_annotation_ = node;
    return absl::OkStatus();
  }

  absl::Status HandleFunctionTypeAnnotation(
      const FunctionTypeAnnotation* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    last_direct_annotation_ = node;
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    last_variable_ = std::nullopt;
    last_signedness_and_bit_count_ = std::nullopt;
    last_tvta_ = std::nullopt;
    last_direct_annotation_ = std::nullopt;
    return absl::OkStatus();
  }

  std::optional<const NameDef*> last_variable() const { return last_variable_; }

  std::optional<const TypeVariableTypeAnnotation*> last_tvta() const {
    return last_tvta_;
  }

  std::optional<const TypeAnnotation*> last_direct_annotation() const {
    return last_direct_annotation_;
  }

  std::optional<SignednessAndBitCountResult> last_signedness_and_bit_count()
      const {
    return last_signedness_and_bit_count_;
  }

 private:
  template <typename T>
  absl::Status HandlePossibleIntegerAnnotation(const T* annotation) {
    XLS_RETURN_IF_ERROR(DefaultHandler(annotation));
    absl::StatusOr<SignednessAndBitCountResult> result =
        GetSignednessAndBitCount(annotation,
                                 /*ignore_missing_dimensions=*/true);
    if (result.ok()) {
      last_direct_annotation_ = annotation;
      last_signedness_and_bit_count_ = *result;
    }
    return absl::OkStatus();
  }

  std::optional<const NameDef*> last_variable_;
  std::optional<const TypeVariableTypeAnnotation*> last_tvta_;
  std::optional<const TypeAnnotation*> last_direct_annotation_;
  std::optional<SignednessAndBitCountResult> last_signedness_and_bit_count_;
};

// Handles "mismatches" between the resolvable and dependent annotations when
// "zipping" them. These mismatches are the interesting part, e.g. when there is
// a value in one and a parametric name in the other.
class Resolver {
 public:
  Resolver(
      ImportData& import_data, Visitor* resolvable_visitor,
      Visitor* dependent_visitor,
      const absl::flat_hash_set<const ParametricBinding*>& bindings_to_resolve,
      absl::AnyInvocable<absl::StatusOr<InterpValue>(
          const TypeAnnotation* expected_type, const Expr*)>
          expr_evaluator)
      : import_data_(import_data),
        resolvable_visitor_(resolvable_visitor),
        dependent_visitor_(dependent_visitor),
        expr_evaluator_(std::move(expr_evaluator)) {
    for (const ParametricBinding* binding : bindings_to_resolve) {
      bindings_to_resolve_.emplace(binding->name_def(), binding);
    }
  }

  absl::Status AcceptMismatch(const AstNode* resolvable,
                              const AstNode* dependent) {
    XLS_RETURN_IF_ERROR(dependent->Accept(dependent_visitor_));

    // Scenario 1: They are 2 differently-formulated TRTAs for the same struct.
    if (resolvable->kind() == AstNodeKind::kTypeAnnotation &&
        dependent->kind() == AstNodeKind::kTypeAnnotation) {
      const auto* resolvable_annotation =
          down_cast<const TypeAnnotation*>(resolvable);
      const auto* dependent_annotation =
          down_cast<const TypeAnnotation* const>(dependent);
      if (resolvable_annotation->IsAnnotation<TypeRefTypeAnnotation>() &&
          dependent_annotation->IsAnnotation<TypeRefTypeAnnotation>()) {
        XLS_ASSIGN_OR_RETURN(
            resolvable_annotation,
            CanonicalizeTypeRefTypeAnnotation(
                resolvable_annotation->AsAnnotation<TypeRefTypeAnnotation>(),
                import_data_));
        XLS_ASSIGN_OR_RETURN(
            dependent_annotation,
            CanonicalizeTypeRefTypeAnnotation(
                dependent_annotation->AsAnnotation<TypeRefTypeAnnotation>(),
                import_data_));
        return ZipAst(resolvable_annotation, dependent_annotation,
                      resolvable_visitor_, dependent_visitor_, options());
      }
    }

    // Scenario 2: `dependent` is S or N or whatever, and `resolvable` is the
    // expr it equates to. Here we don't even care what kind of annotation we
    // are dealing with.
    if (dependent_visitor_->last_variable().has_value()) {
      return ResolveVariable(resolvable, *dependent_visitor_->last_variable());
    }

    XLS_RETURN_IF_ERROR(resolvable->Accept(resolvable_visitor_));

    // Scenario 3: `dependent` is a TVTA and `resolvable` is a direct type
    // annotation. This means the generic type referred to by the TVTA is the
    // resovable type.
    if (dependent_visitor_->last_tvta().has_value() &&
        resolvable_visitor_->last_direct_annotation().has_value()) {
      return ResolveVariable(
          *resolvable_visitor_->last_direct_annotation(),
          std::get<const NameDef*>(
              (*dependent_visitor_->last_tvta())->type_variable()->name_def()));
    }

    // Scenario 4: The 2 annotations just aren't related.
    if (!resolvable_visitor_->last_signedness_and_bit_count().has_value() ||
        !dependent_visitor_->last_signedness_and_bit_count().has_value()) {
      return absl::InvalidArgumentError(
          absl::Substitute("Mismatch: $0 vs. $1", resolvable->ToString(),
                           dependent->ToString()));
    }

    // Scenario 5: `dependent` is a more-expanded form of `resolvable` or vice
    // versa. The more compact one is a built-in type like `u24`, so the zipper
    // is not going to line up the 24 and the N, much less the implicit
    // signedness value.
    SignednessAndBitCountResult resolvable_signedness_and_bit_count =
        *resolvable_visitor_->last_signedness_and_bit_count();
    SignednessAndBitCountResult dependent_signedness_and_bit_count =
        *dependent_visitor_->last_signedness_and_bit_count();
    XLS_RETURN_IF_ERROR(ResolveIntegerTypeComponent(
        resolvable_signedness_and_bit_count.bit_count,
        dependent_signedness_and_bit_count.bit_count));
    return ResolveIntegerTypeComponent(
        resolvable_signedness_and_bit_count.signedness,
        dependent_signedness_and_bit_count.signedness);
  }

  ZipAstOptions options() {
    return ZipAstOptions{.check_defs_for_name_refs = true,
                         .refs_to_same_parametric_are_different = true,
                         .accept_mismatch_callback = [&](const AstNode* lhs,
                                                         const AstNode* rhs) {
                           return AcceptMismatch(lhs, rhs);
                         }};
  }

  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>&
  results() {
    return results_;
  }

 private:
  // Evaluates `resolvable`, which must be an `Expr`, and sets the value of
  // `variable` to that, if `variable` is one of the parametrics we are being
  // asked to solve for.
  absl::Status ResolveVariable(const AstNode* resolvable,
                               const NameDef* variable) {
    const auto it = bindings_to_resolve_.find(variable);
    if (it == bindings_to_resolve_.end()) {
      // If we are not being asked to solve for the variable at hand, we don't
      // care about it.
      return absl::OkStatus();
    }
    const Expr* expr = dynamic_cast<const Expr*>(resolvable);
    if (expr == nullptr) {
      return absl::InvalidArgumentError(
          absl::Substitute("Could not evaluate: $0", resolvable->ToString()));
    }
    const ParametricBinding* binding = it->second;
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         expr_evaluator_(binding->type_annotation(), expr));
    return NoteResult(binding, std::move(value));
  }

  // Resolves `resolvable` to the given type annotation.
  absl::Status ResolveVariable(const TypeAnnotation* direct_annotation,
                               const NameDef* variable) {
    const auto it = bindings_to_resolve_.find(variable);
    if (it == bindings_to_resolve_.end()) {
      // If we are not being asked to solve for the variable at hand, we don't
      // care about it.
      return absl::OkStatus();
    }
    const ParametricBinding* binding = it->second;
    if (dynamic_cast<const GenericTypeAnnotation*>(
            binding->type_annotation()) == nullptr) {
      // Don't try to resolve a parametric that isn't a type.
      return absl::OkStatus();
    }
    return NoteResult(binding, direct_annotation);
  }

  // Variant that takes an int64_t `value` instead of an `Expr`.
  absl::Status ResolveVariable(int64_t value, const NameDef* variable) {
    const auto it = bindings_to_resolve_.find(variable);
    if (it == bindings_to_resolve_.end()) {
      return absl::OkStatus();
    }
    XLS_ASSIGN_OR_RETURN(
        SignednessAndBitCountResult signedness_and_bit_count,
        GetSignednessAndBitCount(it->second->type_annotation(),
                                 /*ignore_missing_dimensions=*/true));
    XLS_ASSIGN_OR_RETURN(
        bool is_signed,
        Evaluate(CreateBoolAnnotation(*variable->owner(), variable->span()),
                 signedness_and_bit_count.signedness));
    XLS_ASSIGN_OR_RETURN(
        int64_t bit_count,
        Evaluate(CreateU32Annotation(*variable->owner(), variable->span()),
                 signedness_and_bit_count.bit_count));
    return NoteResult(it->second,
                      is_signed ? InterpValue::MakeSBits(bit_count, value)
                                : InterpValue::MakeUBits(bit_count, value));
  }

  // Variant that takes a value or `Expr`.
  template <typename T>
  absl::Status ResolveVariable(std::variant<T, const Expr*> value,
                               const NameDef* variable) {
    if (std::holds_alternative<T>(value)) {
      return ResolveVariable(std::get<T>(value), variable);
    }
    return ResolveVariable(std::get<const Expr*>(value), variable);
  }

  // Resolves a variable for the signedness or bit count of an integer type like
  // `xN[S][N]`, `uN[N]`, etc. The input is expected to be a field from a
  // `SignednessAndBitCountResult` object obtained using the type annotation.
  // If the component is not a variable, this does nothing.
  template <typename T>
  absl::Status ResolveIntegerTypeComponent(
      const std::variant<T, const Expr*>& resolvable,
      const std::variant<T, const Expr*>& dependent) {
    if (!std::holds_alternative<const Expr*>(dependent)) {
      // If the dependent value is not actually dependent on anything, there is
      // nothing to do. We would hit this for the static component in a
      // dependent annotation that only has one component parameterized, like
      // `xN[S][32]`.
      return absl::OkStatus();
    }
    XLS_RETURN_IF_ERROR(
        std::get<const Expr*>(dependent)->Accept(dependent_visitor_));
    if (dependent_visitor_->last_variable().has_value()) {
      XLS_RETURN_IF_ERROR(
          ResolveVariable(resolvable, *dependent_visitor_->last_variable()));
    }
    return absl::OkStatus();
  }

  // Records the given `result` for `variable` in the result map, and ensures
  // that it doesn't conflict with a value already recorded.
  absl::Status NoteResult(const ParametricBinding* variable,
                          InterpValueOrTypeAnnotation result) {
    const auto [it, emplaced] = results_.emplace(variable, result);
    if (!emplaced && result != it->second) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Determined conflicting values for $0: $1 vs. $2",
          variable->ToString(), ToString(it->second), ToString(result)));
    }
    return absl::OkStatus();
  }

  template <typename T>
  absl::StatusOr<T> Evaluate(const TypeAnnotation* expected_type,
                             std::variant<T, const Expr*> value_or_expr) {
    if (std::holds_alternative<T>(value_or_expr)) {
      return std::get<T>(value_or_expr);
    }
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        expr_evaluator_(expected_type, std::get<const Expr*>(value_or_expr)));
    return value.GetBitValueUnsigned();
  }

  ImportData& import_data_;
  Visitor* const resolvable_visitor_;
  Visitor* const dependent_visitor_;
  absl::flat_hash_map<const NameDef*, const ParametricBinding*>
      bindings_to_resolve_;
  absl::AnyInvocable<absl::StatusOr<InterpValue>(
      const TypeAnnotation* expected_type, const Expr*)>
      expr_evaluator_;
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      results_;
};

}  // namespace

absl::StatusOr<
    absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>>
SolveForParametrics(ImportData& import_data,
                    const TypeAnnotation* resolvable_type,
                    const TypeAnnotation* parametric_dependent_type,
                    absl::flat_hash_set<const ParametricBinding*> parametrics,
                    absl::AnyInvocable<absl::StatusOr<InterpValue>(
                        const TypeAnnotation* expected_type, const Expr*)>
                        expr_evaluator) {
  Visitor resolvable_visitor;
  Visitor dependent_visitor;
  Resolver resolver(import_data, &resolvable_visitor, &dependent_visitor,
                    parametrics, std::move(expr_evaluator));
  XLS_RETURN_IF_ERROR(ZipAst(resolvable_type, parametric_dependent_type,
                             &resolvable_visitor, &dependent_visitor,
                             resolver.options()));
  return std::move(resolver.results());
}

}  // namespace xls::dslx
