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

#include "xls/dslx/type_system_v2/inference_table_to_type_info.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// An object that facilitates the conversion of an `InferenceTable` to
// `TypeInfo`.
class InferenceTableConverter {
 public:
  InferenceTableConverter(const InferenceTable& table, Module& module,
                          ImportData& import_data,
                          WarningCollector& warning_collector,
                          TypeInfo* base_type_info, const FileTable& file_table,
                          const absl::flat_hash_set<const TypeAnnotation*>&
                              auto_literal_annotations)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        base_type_info_(base_type_info),
        file_table_(file_table),
        auto_literal_annotations_(auto_literal_annotations) {}

  // Generates the resulting type info for the given invocation, as a child of
  // the base type info.
  absl::Status AddInvocationAndGenerateTypeInfo(
      const ParametricInvocation* parametric_invocation) {
    ParametricEnv caller_env;
    if (parametric_invocation->caller_invocation().has_value()) {
      XLS_ASSIGN_OR_RETURN(caller_env,
                           ParametricInvocationToEnv(
                               *parametric_invocation->caller_invocation()));
    }
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * invocation_type_info,
        import_data_.type_info_owner().New(&module_, base_type_info_));
    invocation_type_info_.emplace(parametric_invocation, invocation_type_info);
    XLS_ASSIGN_OR_RETURN(ParametricEnv callee_env,
                         ParametricInvocationToEnv(parametric_invocation));
    XLS_RETURN_IF_ERROR(GenerateTypeInfo(parametric_invocation));
    VLOG(5) << "Adding invocation type info for "
            << parametric_invocation->callee().ToString()
            << " with caller env: " << caller_env.ToString();
    return base_type_info_->AddInvocationTypeInfo(
        parametric_invocation->node(), &parametric_invocation->caller(),
        caller_env, callee_env, invocation_type_info);
  }

  // Generates type info for either a particular parametric invocation (storing
  // the result in a child of `base_type_info_`), or the static nodes in the
  // table (storing the result in `base_type_info_` itself).
  absl::Status GenerateTypeInfo(
      std::optional<const ParametricInvocation*> parametric_invocation) {
    TypeInfo* ti;
    std::vector<const AstNode*> nodes;
    if (parametric_invocation.has_value()) {
      ti = invocation_type_info_.at(*parametric_invocation);
      nodes =
          table_.GetNodesWithInvocationSpecificTypes(*parametric_invocation);
    } else {
      ti = base_type_info_;
      nodes = table_.GetStaticNodes();
    }

    for (const AstNode* node : nodes) {
      std::optional<const TypeAnnotation*> annotation;
      const std::optional<const NameRef*> type_variable =
          table_.GetTypeVariable(node);
      if (type_variable.has_value()) {
        // A type variable implies unification may be needed, so don't just use
        // the type annotation of the node if it has a variable associated with
        // it.
        std::optional<Span> node_span = node->GetSpan();
        CHECK(node_span.has_value());
        if ((node->kind() == AstNodeKind::kConstantDef ||
             node->kind() == AstNodeKind::kNameDef) &&
            !VariableHasAnyExplicitTypeAnnotations(*type_variable)) {
          // The motivation for disallowing this, irrespective of its
          // unifiability, is that otherwise a snippet like this would present
          // a serious ambiguity:
          //   const X = 3;
          //   const Y = X + 1;
          // If we auto-annotate the `3` as `u2` and the `1` becomes `u2` via
          // normal promotion, `X + 1` surprisingly overflows. What is really
          // desired here is probably a common type for `X` and `Y` that fits
          // both. We want the programmer to write that type on the `X` line at
          // a minimum, which will then predictably propagate to `Y` if they
          // don't say otherwise.
          return absl::InvalidArgumentError(absl::Substitute(
              "TypeInferenceError: A variable or constant cannot be defined "
              "with an implicit type. `$0` at $1 must have a type annotation "
              "on at least one side of its assignment.",
              node->ToString(), node_span->ToString(file_table_)));
        }
        XLS_ASSIGN_OR_RETURN(annotation,
                             UnifyTypeAnnotations(parametric_invocation,
                                                  *type_variable, *node_span));
      } else {
        annotation = table_.GetTypeAnnotation(node);
      }
      if (!annotation.has_value()) {
        return absl::FailedPreconditionError(
            absl::Substitute("Node should have either a type annotation or "
                             "annotated type variable: $0",
                             node->ToString()));
      }
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                           Concretize(*annotation, parametric_invocation));
      XLS_RETURN_IF_ERROR(ValidateConcreteTypeForNode(node, type.get()));
      ti->SetItem(node, *type);
    }
    return absl::OkStatus();
  }

  // Returns the resulting base type info for the entire conversion.
  TypeInfo* GetBaseTypeInfo() { return base_type_info_; }

 private:
  // Converts the given type annotation to a concrete `Type`, either statically
  // or in the context of a parametric invocation.
  absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation,
      std::optional<const ParametricInvocation*> parametric_invocation) {
    XLS_ASSIGN_OR_RETURN(annotation, ResolveIfVariableTypeAnnotation(
                                         parametric_invocation, annotation));
    if (const auto* tuple =
            dynamic_cast<const TupleTypeAnnotation*>(annotation)) {
      std::vector<std::unique_ptr<Type>> member_types;
      member_types.reserve(tuple->members().size());
      for (const TypeAnnotation* member : tuple->members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> concrete_member_type,
                             Concretize(member, parametric_invocation));
        member_types.push_back(std::move(concrete_member_type));
      }
      return std::make_unique<TupleType>(std::move(member_types));
    }
    absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
        GetSignednessAndBitCount(annotation);
    if (!signedness_and_bit_count.ok()) {
      return absl::UnimplementedError(absl::Substitute(
          "Type inference version 2 is a work in progress and cannot yet "
          "handle non-bits-like type annotation `$0`.",
          annotation->ToString()));
    }
    XLS_ASSIGN_OR_RETURN(
        bool signedness,
        EvaluateBoolOrExpr(parametric_invocation,
                           signedness_and_bit_count->signedness));
    XLS_ASSIGN_OR_RETURN(
        int64_t bit_count,
        EvaluateS64OrExpr(parametric_invocation,
                          signedness_and_bit_count->bit_count));
    return std::make_unique<BitsType>(signedness, bit_count);
  }

  // Constexpr-evaluates the given expression, whose dependencies must already
  // be noted as constexpr's in the `TypeInfo` corresponding to the scope for
  // the expression.
  absl::StatusOr<InterpValue> Evaluate(
      const InvocationScopedExpr& scoped_expr) {
    TypeInfo* type_info = base_type_info_;
    // Note that `scoped_expr` will not have an `invocation()` in a case like
    //  fn foo<X: u32>(...) { ... }
    //  fn bar() {
    //    foo<SOME_CONSTANT + 1>(...);
    //  }
    // The only scoped expr there is the expression being passed for `X`, which
    // is in a non-parametric caller and therefore cannot possibly refer to any
    // parametrics.
    if (scoped_expr.invocation().has_value()) {
      type_info = invocation_type_info_.at(*scoped_expr.invocation());
    }
    // This is the type of the parametric binding we are talking about, which is
    // typically a built-in type, but the way we are concretizing it here would
    // support it being a complex type that even refers to other parametrics.
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<Type> type,
        Concretize(scoped_expr.type_annotation(), scoped_expr.invocation()));
    type_info->SetItem(scoped_expr.expr(), *type);
    type_info->SetItem(scoped_expr.type_annotation(),
                       MetaType(type->CloneToUnique()));
    // TODO: https://github.com/google/xls/issues/193 - The if-statement below
    // is here temporarily to enable easy testing of parametric variables in
    // inference_table_test. The equivalent is done by `TypecheckModuleV2`, and
    // that's where the logic belongs, but that doesn't yet deal with parametric
    // variables.
    if (auto* number = dynamic_cast<const Number*>(scoped_expr.expr());
        number != nullptr && number->type_annotation() != nullptr) {
      type_info->SetItem(number->type_annotation(),
                         MetaType(type->CloneToUnique()));
    }
    // Note: the `ParametricEnv` is irrelevant here, because we have guaranteed
    // that any parametric that may be referenced by the expr has been noted as
    // a normal constexpr in `type_info`.
    return ConstexprEvaluator::EvaluateToValue(
        &import_data_, type_info, &warning_collector_, ParametricEnv(),
        scoped_expr.expr(), /*type=*/nullptr);
  }

  // Generates a `ParametricEnv` for the given invocation, which is needed for
  // the way `TypeInfo` stores invocation-specific data. This function caches
  // the per-invocation result, because the storage of downstream invocations
  // may require it (e.g. if a parametric function `foo` invokes a parametric
  // function `bar` multiple times, or both `bar` and `baz`).
  absl::StatusOr<ParametricEnv> ParametricInvocationToEnv(
      const ParametricInvocation* invocation) {
    const auto it = converted_parametric_envs_.find(invocation);
    if (it != converted_parametric_envs_.end()) {
      return it->second;
    }
    absl::flat_hash_map<std::string, InterpValue> values;
    for (const ParametricBinding* binding :
         invocation->callee().parametric_bindings()) {
      InvocationScopedExpr expr =
          table_.GetParametricValue(*binding->name_def(), *invocation);
      XLS_ASSIGN_OR_RETURN(InterpValue value, Evaluate(expr));
      invocation_type_info_.at(invocation)
          ->NoteConstExpr(binding->name_def(), value);
      values.emplace(binding->name_def()->identifier(), value);
    }
    ParametricEnv env(values);
    converted_parametric_envs_.emplace(invocation, env);
    return env;
  }

  absl::StatusOr<bool> EvaluateBoolOrExpr(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::variant<bool, const Expr*> value_or_expr) {
    if (std::holds_alternative<bool>(value_or_expr)) {
      return std::get<bool>(value_or_expr);
    }
    const Expr* expr = std::get<const Expr*>(value_or_expr);
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Evaluate(InvocationScopedExpr(
            parametric_invocation, CreateBoolAnnotation(module_, expr->span()),
            expr)));
    return value.GetBitValueUnsigned();
  }

  absl::StatusOr<int64_t> EvaluateS64OrExpr(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::variant<int64_t, const Expr*> value_or_expr) {
    if (std::holds_alternative<int64_t>(value_or_expr)) {
      return std::get<int64_t>(value_or_expr);
    }
    const Expr* expr = std::get<const Expr*>(value_or_expr);
    std::optional<const TypeAnnotation*> type_annotation =
        table_.GetTypeAnnotation(expr);
    if (!type_annotation.has_value()) {
      type_annotation = CreateS64Annotation(module_, expr->span());
    }
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         Evaluate(InvocationScopedExpr(
                             parametric_invocation, *type_annotation, expr)));
    return value.GetBitValueSigned();
  }

  // Comes up with one type annotation reconciling the information in any
  // type annotations that have been associated with the given type variable. If
  // the information has unreconcilable conflicts, returns an error. The given
  // `parametric_invocation` argument is used as a context for the evaluation of
  // any expressions inside the type annotations.
  absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      const NameRef* type_variable, const Span& span) {
    VLOG(5) << "Unifying type annotations for variable "
            << type_variable->ToString();
    XLS_ASSIGN_OR_RETURN(
        std::vector<const TypeAnnotation*> annotations,
        table_.GetTypeAnnotationsForTypeVariable(type_variable));
    return UnifyTypeAnnotations(parametric_invocation, annotations, span);
  }

  // Overload that unifies specific type annotations.
  absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::vector<const TypeAnnotation*> annotations, const Span& span) {
    if (annotations.empty()) {
      return absl::InvalidArgumentError(
          "Failed to unify because there are no type annotations.");
    }
    for (int i = 0; i < annotations.size(); i++) {
      XLS_ASSIGN_OR_RETURN(
          annotations[i], ResolveIfVariableTypeAnnotation(parametric_invocation,
                                                          annotations[i]));
    }
    if (annotations.size() == 1) {
      // This is here mainly for preservation of shorthand annotations appearing
      // in the source code, in case they get put in subsequent error messages.
      // General unification would normalize the format.
      return annotations[0];
    }
    if (const auto* first_tuple_annotation =
            dynamic_cast<const TupleTypeAnnotation*>(annotations[0])) {
      std::vector<const TupleTypeAnnotation*> tuple_annotations;
      tuple_annotations.reserve(annotations.size());
      for (const TypeAnnotation* annotation : annotations) {
        const auto* tuple_annotation =
            dynamic_cast<const TupleTypeAnnotation*>(annotation);
        // If the DSLX programmer annotates a tuple with a non-tuple annotation,
        // it will fail before now, because we need to distribute the components
        // of it to the RHS early on.
        CHECK(tuple_annotation);
        // Since all but one must have been fabricated by us, they should have
        // the same structure.
        CHECK_EQ(tuple_annotation->members().size(),
                 first_tuple_annotation->members().size());
        tuple_annotations.push_back(tuple_annotation);
      }
      return UnifyTupleTypeAnnotations(parametric_invocation, tuple_annotations,
                                       span);
    }
    std::optional<bool> unified_signedness;
    std::optional<int64_t> unified_bit_count;
    for (int i = 0; i < annotations.size(); ++i) {
      const TypeAnnotation* current_annotation = annotations[i];
      VLOG(5) << "Annotation " << i << ": " << current_annotation->ToString();
      absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
          GetSignednessAndBitCount(current_annotation);
      if (!signedness_and_bit_count.ok()) {
        return TypeMismatchErrorStatus(current_annotation, annotations[0],
                                       file_table_);
      }
      XLS_ASSIGN_OR_RETURN(
          bool current_annotation_signedness,
          EvaluateBoolOrExpr(parametric_invocation,
                             signedness_and_bit_count->signedness));
      XLS_ASSIGN_OR_RETURN(
          int64_t current_annotation_bit_count,
          EvaluateS64OrExpr(parametric_invocation,
                            signedness_and_bit_count->bit_count));

      // Unify the signedness. Currently there must be strict agreement, except
      // for auto literals.
      if (!unified_signedness.has_value()) {
        unified_signedness = current_annotation_signedness;
      } else if (current_annotation_signedness != *unified_signedness &&
                 !auto_literal_annotations_.contains(current_annotation)) {
        return SignednessMismatchErrorStatus(current_annotation,
                                             annotations[i - 1], file_table_);
      }

      // Unify the bit count. Currently there must be strict agreement, except
      // for auto literals whose auto size is smaller than we need.
      if (!unified_bit_count.has_value()) {
        unified_bit_count = current_annotation_bit_count;
      } else if (*unified_bit_count != current_annotation_bit_count &&
                 (!auto_literal_annotations_.contains(current_annotation) ||
                  current_annotation_bit_count > unified_bit_count)) {
        return BitCountMismatchErrorStatus(current_annotation,
                                           annotations[i - 1], file_table_);
      }
      VLOG(5) << "Unified type so far has signedness: " << *unified_signedness
              << " and bit count: " << *unified_bit_count;
    }
    return CreateUnOrSnAnnotation(module_, span, *unified_signedness,
                                  *unified_bit_count);
  }

  // Unifies multiple annotations for a tuple. This function assumes the
  // passed-in array is nonempty and the member counts match. Unifying a tuple
  // type amounts to unifying the annotations for each member.
  absl::StatusOr<const TupleTypeAnnotation*> UnifyTupleTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::vector<const TupleTypeAnnotation*> annotations, const Span& span) {
    const int member_count = annotations[0]->members().size();
    std::vector<TypeAnnotation*> unified_member_annotations(member_count);
    for (int i = 0; i < member_count; i++) {
      std::vector<const TypeAnnotation*> annotations_for_member;
      annotations_for_member.reserve(annotations.size());
      for (const TupleTypeAnnotation* annotation : annotations) {
        annotations_for_member.push_back(annotation->members()[i]);
      }
      XLS_ASSIGN_OR_RETURN(const TypeAnnotation* unified_member_annotation,
                           UnifyTypeAnnotations(parametric_invocation,
                                                annotations_for_member, span));
      unified_member_annotations[i] =
          const_cast<TypeAnnotation*>(unified_member_annotation);
    }
    return module_.Make<TupleTypeAnnotation>(span, unified_member_annotations);
  }

  // If `annotation` is a `TypeVariableTypeAnnotation`, then this function gets
  // the underlying annotations for the referenced variable, unifies them, and
  // returns the unified annotation. Otherwise, it is a no-op that returns
  // `annotation`.
  absl::StatusOr<const TypeAnnotation*> ResolveIfVariableTypeAnnotation(
      std::optional<const ParametricInvocation*> parametric_invocation,
      const TypeAnnotation* annotation) {
    if (const auto* variable_type_annotation =
            dynamic_cast<const TypeVariableTypeAnnotation*>(annotation)) {
      XLS_ASSIGN_OR_RETURN(
          annotation,
          UnifyTypeAnnotations(parametric_invocation,
                               variable_type_annotation->type_variable(),
                               annotation->span()));
    }
    return annotation;
  }

  // Checks if the given concrete type ultimately makes sense for the given
  // node, based on the intrinsic properties of the node, like being an add
  // operation or containing an embedded literal.
  absl::Status ValidateConcreteTypeForNode(const AstNode* node,
                                           const Type* type) {
    if (const auto* literal = dynamic_cast<const Number*>(node)) {
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
          "Non-bits type used to define a numeric literal.", file_table_);
    }
    if (const auto* binop = dynamic_cast<const Binop*>(node);
        binop != nullptr &&
        GetBinopSameTypeKinds().contains(binop->binop_kind()) &&
        !IsBitsLike(*type)) {
      return TypeInferenceErrorStatus(
          binop->span(), type,
          "Binary operations can only be applied to bits-typed operands.",
          file_table_);
    }
    return absl::OkStatus();
  }

  // Determines if the given `type_variable` has any annotations in the table
  // that were explicitly written in the DSLX source.
  bool VariableHasAnyExplicitTypeAnnotations(const NameRef* type_variable) {
    absl::StatusOr<std::vector<const TypeAnnotation*>> annotations =
        table_.GetTypeAnnotationsForTypeVariable(type_variable);
    return annotations.ok() &&
           absl::c_any_of(
               *annotations, [this](const TypeAnnotation* annotation) {
                 return !auto_literal_annotations_.contains(annotation);
               });
  }

  const InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  TypeInfo* const base_type_info_;
  const FileTable& file_table_;
  absl::flat_hash_map<const ParametricInvocation*, TypeInfo*>
      invocation_type_info_;
  absl::flat_hash_map<const ParametricInvocation*, ParametricEnv>
      converted_parametric_envs_;
  const absl::flat_hash_set<const TypeAnnotation*>& auto_literal_annotations_;
};

}  // namespace

absl::StatusOr<TypeInfo*> InferenceTableToTypeInfo(
    const InferenceTable& table, Module& module, ImportData& import_data,
    WarningCollector& warning_collector, const FileTable& file_table,
    const absl::flat_hash_set<const TypeAnnotation*>&
        auto_literal_annotations) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * base_type_info,
                       import_data.type_info_owner().New(&module));
  InferenceTableConverter converter(table, module, import_data,
                                    warning_collector, base_type_info,
                                    file_table, auto_literal_annotations);
  XLS_RETURN_IF_ERROR(converter.GenerateTypeInfo(
      /*parametric_invocation=*/std::nullopt));
  for (const ParametricInvocation* invocation :
       table.GetParametricInvocations()) {
    XLS_RETURN_IF_ERROR(converter.AddInvocationAndGenerateTypeInfo(invocation));
  }
  return converter.GetBaseTypeInfo();
}

}  // namespace xls::dslx
