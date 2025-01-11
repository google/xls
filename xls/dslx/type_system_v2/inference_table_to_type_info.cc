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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/solve_for_parametrics.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// A size and signedness with a flag for whether it is automatic. Automatic
// values have more flexible unification rules.
struct SignednessAndSize {
  bool is_auto;
  bool is_signed;
  int64_t size;
};

// Represents a step in `TypeInfo` conversion order where the `ParametricEnv`
// for a parametric invocation is converted.
struct ParametricEnvConversion {
  const ParametricInvocation* parametric_invocation;
};

// Represents a step in `TypeInfo` conversion where the `TypeInfo` for a certain
// node is converted. The node's `TypeInfo` may or may not be scoped to a
// `ParametricInvocation`.
struct NodeConversion {
  std::optional<const ParametricInvocation*> parametric_invocation;
  const AstNode* node;
};

using TypeInfoConversionStep =
    std::variant<ParametricEnvConversion, NodeConversion>;

const TypeAnnotation* SignednessAndSizeToAnnotation(
    Module& module, const SignednessAndSize& signedness_and_size,
    const Span& span) {
  return CreateUnOrSnAnnotation(module, span, signedness_and_size.is_signed,
                                signedness_and_size.size);
}

// Traverses an AST and flattens it into a `vector` in the order the `TypeInfo`
// needs to be built such that prerequisites will be present in `TypeInfo` when
// evaluations are done.
class ConversionOrderVisitor : public AstNodeVisitorWithDefault {
 public:
  ConversionOrderVisitor(
      const InferenceTable& table,
      absl::flat_hash_map<const TypeAnnotation*, const ParametricInvocation*>
          invocation_scoped_annotations)
      : table_(table),
        invocation_scoped_annotations_(invocation_scoped_annotations) {
    parametric_invocation_stack_.push(std::nullopt);
  }

  absl::Status HandleFunction(const Function* node) override {
    if (node->IsParametric()) {
      // Parametric functions are traversed when invocations of them are
      // encountered.
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    // We generally want a post-order traversal here because that is the
    // dependency flow, e.g., if you are going to ask `ConstexprEvaluator` to
    // evaluate `N + 1`, you want to have established what `N` and `1` are
    // first.
    for (const AstNode* node : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(node->Accept(this));
    }
    XLS_ASSIGN_OR_RETURN(
        std::vector<const ParametricInvocation*> related_invocations,
        GetRelatedInvocations(node));
    // The first time we hit some dependency of a parametric invocation (the
    // `Invocation` node itself or some related node), we need to load the
    // callee's parametric binding exprs and request that its ParametricEnv be
    // converted.
    for (const ParametricInvocation* invocation : related_invocations) {
      XLS_RETURN_IF_ERROR(HandleParametricBindingExprsInternal(invocation));
      steps_.push_back(ParametricEnvConversion{invocation});
      handled_parametric_invocations_.insert(invocation);
    }
    // Convert the actual encountered node.
    steps_.push_back(NodeConversion{
        .parametric_invocation = parametric_invocation_stack_.top(),
        .node = node});
    // We can now convert the actual parametric function(s) in the invocation
    // context(s).
    for (const ParametricInvocation* invocation : related_invocations) {
      XLS_RETURN_IF_ERROR(HandleParametricFunctionInternal(invocation));
    }
    return absl::OkStatus();
  }

  const std::vector<TypeInfoConversionStep>& steps() const { return steps_; }

 private:
  absl::StatusOr<std::vector<const ParametricInvocation*>>
  GetRelatedInvocations(const AstNode* node) {
    std::vector<const ParametricInvocation*> referenced_invocations;
    if (std::optional<const TypeAnnotation*> annotation =
            table_.GetTypeAnnotation(node);
        annotation.has_value()) {
      const auto it = invocation_scoped_annotations_.find(*annotation);
      if (it != invocation_scoped_annotations_.end()) {
        referenced_invocations.push_back(it->second);
      }
    }
    if (std::optional<const NameRef*> variable = table_.GetTypeVariable(node);
        variable.has_value()) {
      XLS_ASSIGN_OR_RETURN(std::vector<const TypeAnnotation*> annotations,
                           table_.GetTypeAnnotationsForTypeVariable(*variable));
      for (const TypeAnnotation* annotation : annotations) {
        const auto it = invocation_scoped_annotations_.find(annotation);
        if (it != invocation_scoped_annotations_.end()) {
          referenced_invocations.push_back(it->second);
        }
      }
    }
    // In a case like `foo(foo(...))` where `foo` is a parametric function, the
    // implicit parametrics of the outer invocation depend on the inference of
    // the inner invocation.
    std::list<const ParametricInvocation*> nested_invocations;
    for (const ParametricInvocation* invocation : referenced_invocations) {
      XLS_ASSIGN_OR_RETURN(
          std::vector<const AstNode*> descendants,
          CollectUnder(&invocation->node(), /*want_types=*/false));
      for (const AstNode* descendant : descendants) {
        if (descendant == &invocation->node()) {
          continue;
        }
        if (const auto* descendant_invocation =
                dynamic_cast<const Invocation*>(descendant)) {
          std::optional<const ParametricInvocation*>
              descendant_parametric_invocation =
                  table_.GetParametricInvocation(descendant_invocation);
          if (descendant_parametric_invocation.has_value()) {
            nested_invocations.push_front(*descendant_parametric_invocation);
          }
        }
      }
    }
    std::vector<const ParametricInvocation*> result;
    auto needs_handling = [&](const ParametricInvocation* invocation) {
      return !handled_parametric_invocations_.contains(invocation);
    };
    absl::c_copy_if(nested_invocations, std::back_inserter(result),
                    needs_handling);
    absl::c_copy_if(referenced_invocations, std::back_inserter(result),
                    needs_handling);
    return result;
  }

  absl::Status HandleParametricBindingExprsInternal(
      const ParametricInvocation* parametric_invocation) {
    for (ExprOrType explicit_parametric :
         parametric_invocation->node().explicit_parametrics()) {
      if (std::holds_alternative<Expr*>(explicit_parametric)) {
        XLS_RETURN_IF_ERROR(std::get<Expr*>(explicit_parametric)->Accept(this));
      }
    }
    parametric_invocation_stack_.push(parametric_invocation);
    for (const ParametricBinding* binding :
         parametric_invocation->callee().parametric_bindings()) {
      if (binding->expr() != nullptr) {
        XLS_RETURN_IF_ERROR(binding->expr()->Accept(this));
      }
    }
    parametric_invocation_stack_.pop();
    return absl::OkStatus();
  }

  absl::Status HandleParametricFunctionInternal(
      const ParametricInvocation* parametric_invocation) {
    parametric_invocation_stack_.push(parametric_invocation);
    for (const AstNode* child :
         parametric_invocation->callee().GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    parametric_invocation_stack_.pop();
    return absl::OkStatus();
  }

  const InferenceTable& table_;
  const absl::flat_hash_map<const TypeAnnotation*, const ParametricInvocation*>
      invocation_scoped_annotations_;
  std::vector<TypeInfoConversionStep> steps_;
  std::stack<std::optional<const ParametricInvocation*>>
      parametric_invocation_stack_;
  absl::flat_hash_set<const ParametricInvocation*>
      handled_parametric_invocations_;
};

// An object that facilitates the conversion of an `InferenceTable` to
// `TypeInfo`.
class InferenceTableConverter {
 public:
  InferenceTableConverter(
      const InferenceTable& table, Module& module, ImportData& import_data,
      WarningCollector& warning_collector, TypeInfo* base_type_info,
      const FileTable& file_table,
      const absl::flat_hash_set<const TypeAnnotation*>&
          auto_literal_annotations,
      const absl::flat_hash_map<const TypeAnnotation*,
                                const ParametricInvocation*>&
          invocation_scoped_type_annotations)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        base_type_info_(base_type_info),
        file_table_(file_table),
        auto_literal_annotations_(auto_literal_annotations),
        invocation_scoped_type_annotations_(
            invocation_scoped_type_annotations) {}

  // Creates the child `TypeInfo` object for the given parametric invocation,
  // leaving it initially empty.
  absl::Status AddInvocation(
      const ParametricInvocation* parametric_invocation) {
    VLOG(5) << "Adding invocation type info for "
            << parametric_invocation->callee().ToString();
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * invocation_type_info,
        import_data_.type_info_owner().New(&module_, base_type_info_));
    invocation_type_info_.emplace(parametric_invocation, invocation_type_info);
    return absl::OkStatus();
  }

  // Generates the final `ParametricEnv` objects for the given invocation, and
  // adds the invocation's data to the base type info.
  absl::Status GenerateParametricEnvs(
      const ParametricInvocation* parametric_invocation) {
    VLOG(5) << "Populating invocation type info for "
            << parametric_invocation->callee().ToString();
    ParametricEnv caller_env;
    if (parametric_invocation->caller_invocation().has_value()) {
      XLS_ASSIGN_OR_RETURN(caller_env,
                           ParametricInvocationToEnv(
                               *parametric_invocation->caller_invocation()));
    }
    XLS_ASSIGN_OR_RETURN(ParametricEnv callee_env,
                         ParametricInvocationToEnv(parametric_invocation));
    VLOG(5) << "Caller env: " << caller_env.ToString();
    VLOG(5) << "Callee env: " << callee_env.ToString();
    return base_type_info_->AddInvocationTypeInfo(
        parametric_invocation->node(),
        parametric_invocation->caller().has_value()
            ? *parametric_invocation->caller()
            : nullptr,
        caller_env, callee_env,
        invocation_type_info_.at(parametric_invocation));
  }

  // Generates type info for one node.
  absl::Status GenerateTypeInfo(
      std::optional<const ParametricInvocation*> parametric_invocation,
      const AstNode* node) {
    VLOG(5) << "Generate type info for node: " << node->ToString();
    TypeInfo* ti = parametric_invocation.has_value()
                       ? invocation_type_info_.at(*parametric_invocation)
                       : base_type_info_;

    std::optional<const TypeAnnotation*> annotation;
    const std::optional<const NameRef*> type_variable =
        table_.GetTypeVariable(node);
    if (type_variable.has_value()) {
      // A type variable implies unification may be needed, so don't just use
      // the type annotation of the node if it has a variable associated with
      // it.
      std::optional<Span> node_span = node->GetSpan();
      CHECK(node_span.has_value());
      if (node->parent() != nullptr &&
          ((node->parent()->kind() == AstNodeKind::kConstantDef ||
            node->parent()->kind() == AstNodeKind::kNameDef)) &&
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
            node->parent()->ToString(), node_span->ToString(file_table_)));
      }
      XLS_ASSIGN_OR_RETURN(annotation,
                           UnifyTypeAnnotations(parametric_invocation,
                                                *type_variable, *node_span));
    } else {
      annotation = table_.GetTypeAnnotation(node);
    }
    if (!annotation.has_value()) {
      // The caller may have passed a node that is in the AST but not in the
      // table, and it may not be needed in the table.
      VLOG(5) << "No type information for: " << node->ToString();
      return absl::OkStatus();
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         Concretize(*annotation, parametric_invocation));
    XLS_RETURN_IF_ERROR(ValidateConcreteTypeForNode(node, type.get()));
    if (const auto* literal = dynamic_cast<const Number*>(node);
        literal != nullptr && literal->type_annotation() != nullptr) {
      ti->SetItem(literal->type_annotation(),
                  *std::make_unique<MetaType>(type->CloneToUnique()));
    }
    ti->SetItem(node, *type);
    if (const auto* constant_def = dynamic_cast<const ConstantDef*>(node)) {
      VLOG(5) << "Checking constant def value: " << constant_def->ToString()
              << " with type: " << type->ToString();
      absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
          &import_data_, ti, &warning_collector_, ParametricEnv(),
          constant_def->value(), type.get());
      if (value.ok()) {
        VLOG(5) << "Constnat def: " << constant_def->ToString()
                << " has value: " << value->ToString();
        ti->NoteConstExpr(constant_def, *value);
        ti->NoteConstExpr(constant_def->value(), *value);
        ti->NoteConstExpr(constant_def->name_def(), *value);
      }
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
    VLOG(5) << "Concretize: " << annotation->ToString()
            << " in context invocation: " << ToString(parametric_invocation);
    parametric_invocation =
        GetEffectiveParametricInvocation(parametric_invocation, annotation);
    VLOG(5) << "Effective invocation: " << ToString(parametric_invocation);

    XLS_ASSIGN_OR_RETURN(annotation, ResolveVariableTypeAnnotations(
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
    if (const auto* array = CastToNonBitsArrayTypeAnnotation(annotation)) {
      XLS_ASSIGN_OR_RETURN(
          int64_t size, EvaluateS64OrExpr(parametric_invocation, array->dim()));
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> element_type,
          Concretize(array->element_type(), parametric_invocation));
      return std::make_unique<ArrayType>(std::move(element_type),
                                         TypeDim(InterpValue::MakeS64(size)));
    }
    if (std::optional<StructOrProcDef> struct_or_proc =
            GetStructOrProcDef(annotation);
        struct_or_proc.has_value()) {
      const StructDefBase* struct_def_base =
          dynamic_cast<const StructDefBase*>(ToAstNode(*struct_or_proc));
      CHECK(struct_def_base != nullptr);
      std::vector<std::unique_ptr<Type>> member_types;
      member_types.reserve(struct_def_base->members().size());
      for (const StructMemberNode* member : struct_def_base->members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> concrete_member_type,
                             Concretize(member->type(), parametric_invocation));
        member_types.push_back(std::move(concrete_member_type));
      }
      if (std::holds_alternative<const StructDef*>(*struct_or_proc)) {
        return std::make_unique<StructType>(
            std::move(member_types),
            *std::get<const StructDef*>(*struct_or_proc));
      }
      return std::make_unique<ProcType>(
          std::move(member_types), *std::get<const ProcDef*>(*struct_or_proc));
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
    VLOG(5) << "Concretized: " << annotation->ToString()
            << " to signed: " << signedness << ", bit count: " << bit_count;
    return std::make_unique<BitsType>(signedness, bit_count);
  }

  // Constexpr-evaluates the given expression, whose dependencies must already
  // be noted as constexpr's in the `TypeInfo` corresponding to the scope for
  // the expression.
  absl::StatusOr<InterpValue> Evaluate(
      const InvocationScopedExpr& scoped_expr) {
    VLOG(5) << "Evaluate: " << scoped_expr.expr()->ToString()
            << " in context: " << ToString(scoped_expr.invocation());
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
    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        ConstexprEvaluator::EvaluateToValue(
            &import_data_, type_info, &warning_collector_, ParametricEnv(),
            scoped_expr.expr(), /*type=*/nullptr));
    VLOG(5) << "Evaluation result for: " << scoped_expr.expr()->ToString()
            << " in context: " << ToString(scoped_expr.invocation())
            << " value: " << result.ToString();
    return result;
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
    absl::flat_hash_set<const ParametricBinding*> implicit_parametrics;
    auto infer_pending_implicit_parametrics = [&]() -> absl::Status {
      if (implicit_parametrics.empty()) {
        return absl::OkStatus();
      }
      absl::flat_hash_map<std::string, InterpValue> new_values;
      XLS_ASSIGN_OR_RETURN(new_values, InferImplicitFunctionParametrics(
                                           invocation, implicit_parametrics));
      implicit_parametrics.clear();
      values.merge(std::move(new_values));
      return absl::OkStatus();
    };
    for (const ParametricBinding* binding :
         invocation->callee().parametric_bindings()) {
      std::optional<InvocationScopedExpr> expr =
          table_.GetParametricValue(*binding->name_def(), *invocation);
      if (expr.has_value()) {
        // The expr may be a default expr which may use the inferred values of
        // any parametrics preceding it, so let's resolve any pending implicit
        // ones now.
        XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());
        // Now evaluate the expr.
        XLS_ASSIGN_OR_RETURN(InterpValue value, Evaluate(*expr));
        invocation_type_info_.at(invocation)
            ->NoteConstExpr(binding->name_def(), value);
        values.emplace(binding->name_def()->identifier(), value);
      } else {
        implicit_parametrics.insert(binding);
      }
    }
    // Resolve any implicit ones that are at the end of the list.
    XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());
    ParametricEnv env(values);
    converted_parametric_envs_.emplace(invocation, env);
    return env;
  }

  // Attempts to infer the values of the specified implicit parametrics in an
  // invocation, using the types of the regular arguments being passed. If not
  // all of `implicit_parametrics` can be determined, this function returns an
  // error.
  absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
  InferImplicitFunctionParametrics(
      const ParametricInvocation* invocation,
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics) {
    VLOG(5) << "Inferring " << implicit_parametrics.size()
            << " implicit parametrics for invocation: " << ToString(invocation);
    absl::flat_hash_map<std::string, InterpValue> values;
    const absl::Span<Param* const> formal_args = invocation->callee().params();
    const absl::Span<Expr* const> actual_args = invocation->node().args();
    TypeInfo* ti = invocation_type_info_.at(invocation);
    for (int i = 0; i < formal_args.size(); i++) {
      std::optional<const NameRef*> actual_arg_type_var =
          table_.GetTypeVariable(actual_args[i]);
      if (!actual_arg_type_var.has_value()) {
        VLOG(5) << "The actual argument: `" << actual_args[i]->ToString()
                << "` has no type variable.";
        continue;
      }
      VLOG(5) << "Using type variable: " << (*actual_arg_type_var)->ToString();
      XLS_ASSIGN_OR_RETURN(
          std::vector<const TypeAnnotation*> actual_arg_annotations,
          table_.GetTypeAnnotationsForTypeVariable(*actual_arg_type_var));
      TypeInfo* actual_arg_ti = base_type_info_;
      if (invocation->caller_invocation().has_value()) {
        actual_arg_ti =
            invocation_type_info_.at(*invocation->caller_invocation());
      }

      // The type variable for the actual argument should have at least one
      // annotation associated with it that came from the formal argument and is
      // therefore dependent on the parametric we are solving for. Let's unify
      // just the independent annotations(s) for the purposes of solving for the
      // variable.
      auto accept_predicate = [&](const TypeAnnotation* annotation) {
        return !HasAnyReferencesWithMissingTypeInfo(actual_arg_ti, annotation);
      };
      XLS_RETURN_IF_ERROR(ResolveVariableTypeAnnotations(
          invocation, actual_arg_annotations, accept_predicate));
      if (actual_arg_annotations.empty()) {
        VLOG(5) << "The actual argument type variable: "
                << (*actual_arg_type_var)->ToString()
                << " has no independent type annotations.";
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* actual_arg_type,
          UnifyTypeAnnotations(invocation, actual_arg_annotations,
                               actual_args[i]->span()));
      std::optional<const ParametricInvocation*> effective_invocation =
          GetEffectiveParametricInvocation(invocation->caller_invocation(),
                                           actual_arg_type);
      absl::flat_hash_map<const ParametricBinding*, InterpValue> resolved;
      VLOG(5) << "Infer using actual type: " << actual_arg_type->ToString()
              << " with effective invocation: "
              << ToString(effective_invocation);
      XLS_ASSIGN_OR_RETURN(
          resolved,
          SolveForParametrics(
              actual_arg_type, formal_args[i]->type_annotation(),
              implicit_parametrics,
              [&](const TypeAnnotation* expected_type, const Expr* expr) {
                return Evaluate(InvocationScopedExpr(effective_invocation,
                                                     expected_type, expr));
              }));
      for (auto& [binding, value] : resolved) {
        VLOG(5) << "Inferred implicit parametric value: " << value.ToString()
                << " for binding: " << binding->identifier()
                << " using function argument: `" << actual_args[i]->ToString()
                << "` of actual type: " << actual_arg_type->ToString();
        ti->NoteConstExpr(binding->name_def(), value);
        implicit_parametrics.erase(binding);
        values.emplace(binding->identifier(), std::move(value));
      }
    }
    if (!implicit_parametrics.empty()) {
      std::vector<std::string> binding_names;
      binding_names.reserve(implicit_parametrics.size());
      for (const ParametricBinding* binding : implicit_parametrics) {
        binding_names.push_back(binding->identifier());
      }
      return absl::InvalidArgumentError(
          absl::StrCat("Could not infer parametric(s): ",
                       absl::StrJoin(binding_names, ", ")));
    }
    return values;
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
  // any expressions inside the type annotations. If an `accept_predicate` is
  // specified, then annotations not accepted by the predicate are ignored.
  absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      const NameRef* type_variable, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate = std::nullopt) {
    VLOG(5) << "Unifying type annotations for variable "
            << type_variable->ToString();
    XLS_ASSIGN_OR_RETURN(
        std::vector<const TypeAnnotation*> annotations,
        table_.GetTypeAnnotationsForTypeVariable(type_variable));
    if (accept_predicate.has_value()) {
      FilterAnnotations(annotations, *accept_predicate);
    }
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* result,
        UnifyTypeAnnotations(parametric_invocation, annotations, span));
    VLOG(5) << "Unified type for variable " << type_variable->ToString() << ": "
            << result->ToString();
    return result;
  }

  // Overload that unifies specific type annotations.
  absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::vector<const TypeAnnotation*> annotations, const Span& span) {
    if (annotations.empty()) {
      return absl::InvalidArgumentError(
          "Failed to unify because there are no type annotations.");
    }
    XLS_RETURN_IF_ERROR(
        ResolveVariableTypeAnnotations(parametric_invocation, annotations));
    if (annotations.size() == 1 &&
        !invocation_scoped_type_annotations_.contains(annotations[0])) {
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
    if (const auto* first_array_annotation =
            CastToNonBitsArrayTypeAnnotation(annotations[0])) {
      std::vector<const ArrayTypeAnnotation*> array_annotations;
      for (int i = 0; i < annotations.size(); i++) {
        const auto* array_annotation =
            dynamic_cast<const ArrayTypeAnnotation*>(annotations[i]);
        // If the DSLX programmer puts the wrong kind of annotation on an array,
        // it will error before now.
        CHECK(array_annotation);
        array_annotations.push_back(array_annotation);
      }
      return UnifyArrayTypeAnnotations(parametric_invocation, array_annotations,
                                       span);
    }
    if (std::optional<StructOrProcDef> first_struct_or_proc =
            GetStructOrProcDef(annotations[0]);
        first_struct_or_proc.has_value()) {
      const StructDefBase* struct_def_base =
          dynamic_cast<const StructDefBase*>(ToAstNode(*first_struct_or_proc));
      for (const TypeAnnotation* annotation : annotations) {
        std::optional<StructOrProcDef> next_struct_or_proc =
            GetStructOrProcDef(annotation);
        if (!next_struct_or_proc.has_value() ||
            ToAstNode(*next_struct_or_proc) != struct_def_base) {
          return TypeMismatchErrorWithParametricResolution(
              parametric_invocation, annotations[0], annotation);
        }
      }
      return annotations[0];
    }
    std::optional<SignednessAndSize> unified_signedness_and_bit_count;
    for (int i = 0; i < annotations.size(); ++i) {
      const TypeAnnotation* current_annotation = annotations[i];
      VLOG(5) << "Annotation " << i << ": " << current_annotation->ToString();
      std::optional<const ParametricInvocation*>
          effective_parametric_invocation = GetEffectiveParametricInvocation(
              parametric_invocation, current_annotation);
      absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
          GetSignednessAndBitCount(current_annotation);
      bool current_annotation_is_auto =
          auto_literal_annotations_.contains(current_annotation);
      if (!signedness_and_bit_count.ok()) {
        return TypeMismatchErrorWithParametricResolution(
            parametric_invocation, current_annotation, annotations[0]);
      }
      XLS_ASSIGN_OR_RETURN(
          bool current_annotation_signedness,
          EvaluateBoolOrExpr(effective_parametric_invocation,
                             signedness_and_bit_count->signedness));
      XLS_ASSIGN_OR_RETURN(
          int64_t current_annotation_raw_bit_count,
          EvaluateS64OrExpr(effective_parametric_invocation,
                            signedness_and_bit_count->bit_count));
      SignednessAndSize current_annotation_signedness_and_bit_count{
          .is_auto = current_annotation_is_auto,
          .is_signed = current_annotation_signedness,
          .size = current_annotation_raw_bit_count};

      XLS_ASSIGN_OR_RETURN(
          unified_signedness_and_bit_count,
          UnifySignednessAndSize(parametric_invocation,
                                 unified_signedness_and_bit_count,
                                 current_annotation_signedness_and_bit_count,
                                 annotations[0], current_annotation));
      VLOG(5) << "Unified type so far has signedness: "
              << unified_signedness_and_bit_count->is_signed
              << " and bit count: " << unified_signedness_and_bit_count->size;
    }
    const TypeAnnotation* result = SignednessAndSizeToAnnotation(
        module_, *unified_signedness_and_bit_count, span);
    // An annotation we fabricate as a unification of a bunch of auto
    // annotations, is also considered an auto annotation itself.
    if (unified_signedness_and_bit_count->is_auto) {
      auto_literal_annotations_.insert(result);
    }
    return result;
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

  // Unifies multiple annotations for an array. This function assumes the
  // passed-in array is nonempty. Unifying an array type amounts to unifying the
  // element types and dims.
  absl::StatusOr<const ArrayTypeAnnotation*> UnifyArrayTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::vector<const ArrayTypeAnnotation*> annotations, const Span& span) {
    std::vector<const TypeAnnotation*> element_type_annotations;
    std::optional<SignednessAndSize> unified_dim;
    for (int i = 0; i < annotations.size(); i++) {
      const ArrayTypeAnnotation* annotation = annotations[i];
      std::optional<const ParametricInvocation*> effective_invocation =
          GetEffectiveParametricInvocation(parametric_invocation, annotation);
      element_type_annotations.push_back(annotation->element_type());
      XLS_ASSIGN_OR_RETURN(
          int64_t current_dim,
          EvaluateS64OrExpr(effective_invocation, annotation->dim()));
      // This flag indicates we are unifying one min dim with one explicit dim,
      // which warrants a possible different error message than other scenarios.
      const bool is_min_vs_explicit =
          unified_dim.has_value() &&
          (unified_dim->is_auto ^ annotation->dim_is_min());
      absl::StatusOr<SignednessAndSize> new_unified_dim =
          UnifySignednessAndSize(
              parametric_invocation, unified_dim,
              SignednessAndSize{.is_auto = annotation->dim_is_min(),
                                .is_signed = false,
                                .size = current_dim},
              annotations[0], annotations[i]);
      if (!new_unified_dim.ok()) {
        // We can only get here when i >= 1, because the 0th annotation can't be
        // a contradiction of preceding info.
        CHECK_GE(i, 1);
        if (is_min_vs_explicit) {
          return TypeInferenceErrorStatus(
              span, /*type=*/nullptr,
              "Annotated array size is too small for explicit element count.",
              file_table_);
        }
        return TypeMismatchErrorWithParametricResolution(
            parametric_invocation, annotations[i], annotations[i - 1]);
      }
      unified_dim = *new_unified_dim;
    }
    if (unified_dim->is_auto) {
      // This means the only type annotation for the array was fabricated
      // based on an elliptical RHS.
      return TypeInferenceErrorStatus(
          span, /*type=*/nullptr,
          "Array has ellipsis (`...`) but does not have a type annotation.",
          file_table_);
    }
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* unified_element_type,
                         UnifyTypeAnnotations(parametric_invocation,
                                              element_type_annotations, span));
    return module_.Make<ArrayTypeAnnotation>(
        span, const_cast<TypeAnnotation*>(unified_element_type),
        module_.Make<Number>(annotations[0]->span(),
                             absl::StrCat(unified_dim->size),
                             NumberKind::kOther,
                             /*type_annotation=*/nullptr));
  }

  // Returns `annotation` with any `TypeVariableTypeAnnotation`s replaced with
  // the unifications of the corresponding variables. The original `annotation`
  // is returned if there is nothing to replace, preserving the ability to look
  // it up in `auto_literal_annotations_`. If `accept_predicate` is specified,
  // then it is used to filter the annotations associated with encountered type
  // variables (the predicate is not applied to the input `annotation` itself).
  absl::StatusOr<const TypeAnnotation*> ResolveVariableTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      const TypeAnnotation* annotation,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate = std::nullopt) {
    bool replaced_anything = false;
    XLS_ASSIGN_OR_RETURN(
        AstNode * clone,
        CloneAst(
            annotation,
            [&](const AstNode* node)
                -> absl::StatusOr<std::optional<AstNode*>> {
              if (const auto* variable_type_annotation =
                      dynamic_cast<const TypeVariableTypeAnnotation*>(node)) {
                XLS_ASSIGN_OR_RETURN(
                    const TypeAnnotation* unified,
                    UnifyTypeAnnotations(
                        parametric_invocation,
                        variable_type_annotation->type_variable(),
                        annotation->span(), accept_predicate));
                replaced_anything = true;
                return const_cast<TypeAnnotation*>(unified);
              }
              return std::nullopt;
            }));
    if (!replaced_anything) {
      return annotation;
    }
    const auto* result = dynamic_cast<const TypeAnnotation*>(clone);
    CHECK(result != nullptr);
    return result;
  }

  // Variant that deeply resolves all `TypeVariableTypeAnnotation`s within a
  // vector of annotations. If `accept_predicate` is specified, then any
  // annotations not accepted by the predicate are filtered from both
  // `annotations` and the expansions of any encountered type variables.
  absl::Status ResolveVariableTypeAnnotations(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::vector<const TypeAnnotation*>& annotations,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate = std::nullopt) {
    std::vector<const TypeAnnotation*> result;
    for (const TypeAnnotation* annotation : annotations) {
      if (!accept_predicate.has_value() || (*accept_predicate)(annotation)) {
        XLS_ASSIGN_OR_RETURN(
            const TypeAnnotation* resolved_annotation,
            ResolveVariableTypeAnnotations(parametric_invocation, annotation,
                                           accept_predicate));
        result.push_back(resolved_annotation);
      }
    }
    annotations = std::move(result);
    return absl::OkStatus();
  }

  // Checks if the given concrete type ultimately makes sense for the given
  // node, based on the intrinsic properties of the node, like being an add
  // operation or containing an embedded literal.
  absl::Status ValidateConcreteTypeForNode(const AstNode* node,
                                           const Type* type) {
    if (type->IsMeta()) {
      XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
    }
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

  // Wraps `BitCountMismatchErrorStatus` with resolution of parametrics, so that
  // a nominal type like `uN[N]` will not appear with the variable in the error
  // message.
  absl::Status BitCountMismatchErrorWithParametricResolution(
      std::optional<const ParametricInvocation*> context_invocation,
      const TypeAnnotation* annotation1, const TypeAnnotation* annotation2) {
    std::optional<const ParametricInvocation*> effective_invocation1 =
        GetEffectiveParametricInvocation(context_invocation, annotation1);
    std::optional<const ParametricInvocation*> effective_invocation2 =
        GetEffectiveParametricInvocation(context_invocation, annotation2);
    if (!effective_invocation1.has_value() &&
        !effective_invocation2.has_value()) {
      return BitCountMismatchErrorStatus(annotation1, annotation2, file_table_);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type1,
                         Concretize(annotation1, effective_invocation1));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type2,
                         Concretize(annotation2, effective_invocation2));
    return BitCountMismatchErrorStatus(*type1, *type2, annotation1->span(),
                                       annotation2->span(), file_table_);
  }

  // Wraps `SignednessMismatchErrorStatus` with resolution of parametrics, so
  // that a nominal type like `uN[N]` will not appear with the variable in the
  // error message.
  absl::Status SignednessMismatchErrorWithParametricResolution(
      std::optional<const ParametricInvocation*> context_invocation,
      const TypeAnnotation* annotation1, const TypeAnnotation* annotation2) {
    std::optional<const ParametricInvocation*> effective_invocation1 =
        GetEffectiveParametricInvocation(context_invocation, annotation1);
    std::optional<const ParametricInvocation*> effective_invocation2 =
        GetEffectiveParametricInvocation(context_invocation, annotation2);
    if (!effective_invocation1.has_value() &&
        !effective_invocation2.has_value()) {
      return SignednessMismatchErrorStatus(annotation1, annotation2,
                                           file_table_);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type1,
                         Concretize(annotation1, effective_invocation1));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type2,
                         Concretize(annotation2, effective_invocation2));
    return SignednessMismatchErrorStatus(*type1, *type2, annotation1->span(),
                                         annotation2->span(), file_table_);
  }

  // Wraps `TypeMismatchErrorStatus` with resolution of parametrics, so that a
  // nominal type like `uN[N]` will not appear with the variable in the error
  // message.
  absl::Status TypeMismatchErrorWithParametricResolution(
      std::optional<const ParametricInvocation*> context_invocation,
      const TypeAnnotation* annotation1, const TypeAnnotation* annotation2) {
    std::optional<const ParametricInvocation*> effective_invocation1 =
        GetEffectiveParametricInvocation(context_invocation, annotation1);
    std::optional<const ParametricInvocation*> effective_invocation2 =
        GetEffectiveParametricInvocation(context_invocation, annotation2);
    if (!effective_invocation1.has_value() &&
        !effective_invocation2.has_value()) {
      return TypeMismatchErrorStatus(annotation1, annotation2, file_table_);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type1,
                         Concretize(annotation1, effective_invocation1));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type2,
                         Concretize(annotation2, effective_invocation2));
    return TypeMismatchErrorStatus(*type1, *type2, annotation1->span(),
                                   annotation2->span(), file_table_);
  }

  // Returns the invocation in whose context the given type annotation should be
  // evaluated. The need for this arises from the fact that a parametric callee
  // "leaks" its parametric variables into the annotations of caller nodes near
  // the call boundary. In an example like:
  //
  //    fn foo<N: u32>() -> uN[N] { ... }
  //    fn bar<X: u32>() {
  //      let z = foo<50>();
  //      ...
  //    }
  //
  // The inference table would have a type annotation `uN[N]` for nodes on the
  // `let z` line. `GetEffectiveParametricInvocation` would yield the relevant
  // `foo` invocation when passed the `uN[N]` annotation and the `bar` context
  // invocation that is being analyzed.
  std::optional<const ParametricInvocation*> GetEffectiveParametricInvocation(
      std::optional<const ParametricInvocation*> context_invocation,
      const TypeAnnotation* annotation) {
    const auto scope_it = invocation_scoped_type_annotations_.find(annotation);
    if (scope_it != invocation_scoped_type_annotations_.end()) {
      return scope_it->second;
    }
    return context_invocation;
  }

  // Removes any annotations in the given vector for which `accept_predicate`
  // returns false.
  void FilterAnnotations(
      std::vector<const TypeAnnotation*>& annotations,
      absl::FunctionRef<bool(const TypeAnnotation*)> accept_predicate) {
    annotations.erase(std::remove_if(annotations.begin(), annotations.end(),
                                     [&](const TypeAnnotation* annotation) {
                                       return !accept_predicate(annotation);
                                     }),
                      annotations.end());
  }

  // Returns true if `annotation` contains any `NameRef` whose type info has not
  // (yet) been generated. The effective `TypeInfo` is either `default_ti`; or,
  // for invocation-scoped annotations, the `TypeInfo` for the relevant
  // parametric invocation.
  bool HasAnyReferencesWithMissingTypeInfo(TypeInfo* default_ti,
                                           const TypeAnnotation* annotation) {
    TypeInfo* ti = default_ti;
    const auto it = invocation_scoped_type_annotations_.find(annotation);
    if (it != invocation_scoped_type_annotations_.end()) {
      ti = invocation_type_info_.at(it->second);
    }
    FreeVariables vars =
        GetFreeVariablesByLambda(annotation, [&](const NameRef& ref) {
          if (!std::holds_alternative<const NameDef*>(ref.name_def())) {
            return false;
          }
          const NameDef* name_def = std::get<const NameDef*>(ref.name_def());
          return !ti->GetItem(name_def).has_value() &&
                 !ti->IsKnownConstExpr(name_def);
        });
    return vars.GetFreeVariableCount() > 0;
  }

  // Returns a `SignednessAndSize` that agrees with the two given
  // `SignednessAndSize` objects if possible. `x` is optional for convenience of
  // invoking this in a loop where the first call has no preceding value.
  //
  // Any error returned is a size or signedness mismatch error suitable for
  // display to the user. The `parametric_invocation` and the passed in type
  // annotations are used only for the purpose of generating errors. It is
  // assumed that `y_annotation` should be mentioned first in errors.
  absl::StatusOr<SignednessAndSize> UnifySignednessAndSize(
      std::optional<const ParametricInvocation*> parametric_invocation,
      std::optional<SignednessAndSize> x, SignednessAndSize y,
      const TypeAnnotation* x_annotation, const TypeAnnotation* y_annotation) {
    if (!x.has_value()) {
      return y;
    }
    if (x->is_auto && y.is_auto) {
      return SignednessAndSize{
          .is_auto = true,
          .is_signed = x->is_signed || y.is_signed,
          // If we are coercing one of 2 auto annotations to signed, the one
          // being coerced needs an extra bit to keep fitting the value it was
          // sized to.
          .size = x->size == y.size && x->is_signed != y.is_signed
                      ? x->size + 1
                      : std::max(x->size, y.size)};
    }
    // Converts `annotation` into one that reflects `signedness_and_size`, for
    // error purposes, if it is auto. If it is explicit, then we would not have
    // modified `signedness_and_size`, and we want to display `annotation`'s
    // original formulation, which may not be canonical. Note that
    // `signedness_and_size` may have been modified by a prior call to
    // `UnifySignednessAndSize`, and not necessarily the current call.
    auto update_annotation = [&](const SignednessAndSize& signedness_and_size,
                                 const TypeAnnotation* annotation) {
      return signedness_and_size.is_auto
                 ? SignednessAndSizeToAnnotation(module_, signedness_and_size,
                                                 annotation->span())
                 : annotation;
    };
    auto signedness_mismatch_error = [&] {
      return SignednessMismatchErrorWithParametricResolution(
          parametric_invocation, update_annotation(y, y_annotation),
          update_annotation(*x, x_annotation));
    };
    auto bit_count_mismatch_error = [&] {
      return BitCountMismatchErrorWithParametricResolution(
          parametric_invocation, update_annotation(y, y_annotation),
          update_annotation(*x, x_annotation));
    };
    if (x->is_auto || y.is_auto) {
      SignednessAndSize& auto_value = x->is_auto ? *x : y;
      SignednessAndSize& explicit_value = x->is_auto ? y : *x;
      if (auto_value.is_signed && !explicit_value.is_signed) {
        return signedness_mismatch_error();
      }
      if (!auto_value.is_signed && explicit_value.is_signed) {
        // An auto value being coerced to be signed needs to be extended for the
        // same reason as above.
        auto_value.is_signed = true;
        ++auto_value.size;
      }
      if (explicit_value.size >= auto_value.size) {
        return explicit_value;
      }
      return bit_count_mismatch_error();
    }
    // They are both explicit and must match.
    if (x->size != y.size) {
      return bit_count_mismatch_error();
    }
    if (x->is_signed != y.is_signed) {
      return signedness_mismatch_error();
    }
    return *x;
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
  absl::flat_hash_set<const TypeAnnotation*> auto_literal_annotations_;
  // For annotations that are present in here, any `Expr` in the annotation must
  // be treated as an `InvocationScopedExpr` scoped to the invocation specified
  // here.
  const absl::flat_hash_map<const TypeAnnotation*, const ParametricInvocation*>
      invocation_scoped_type_annotations_;
};

}  // namespace

absl::StatusOr<TypeInfo*> InferenceTableToTypeInfo(
    const InferenceTable& table, Module& module, ImportData& import_data,
    WarningCollector& warning_collector, const FileTable& file_table,
    const absl::flat_hash_set<const TypeAnnotation*>& auto_literal_annotations,
    const absl::flat_hash_map<const TypeAnnotation*,
                              const ParametricInvocation*>&
        invocation_scoped_type_annotations) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * base_type_info,
                       import_data.type_info_owner().New(&module));
  InferenceTableConverter converter(
      table, module, import_data, warning_collector, base_type_info, file_table,
      auto_literal_annotations, invocation_scoped_type_annotations);

  // Figure out the order in which we can sequentially concretize all type info
  // and add all constexprs.
  ConversionOrderVisitor order_visitor(table,
                                       invocation_scoped_type_annotations);
  XLS_RETURN_IF_ERROR(module.Accept(&order_visitor));
  VLOG(5) << "TypeInfo conversion order";
  VLOG(5) << "-------------------------";
  for (const TypeInfoConversionStep& step : order_visitor.steps()) {
    absl::visit(Visitor{[&](const NodeConversion& node_conversion) {
                          VLOG(5)
                              << "Convert node: "
                              << ToString(node_conversion.parametric_invocation)
                              << ", " << node_conversion.node->ToString();
                        },
                        [&](const ParametricEnvConversion& env_conversion) {
                          VLOG(5)
                              << "Convert parametric env: "
                              << ToString(env_conversion.parametric_invocation);
                        }},
                step);
  }
  VLOG(5) << "--------------------";

  // The TypeInfo objects for invocations all need to exist before their
  // `ParametricEnv` objects are finalizable. Some invocation may have a
  // dependency on just some of the info in another invocation's environment,
  // and the order determination assumes we can use that.
  for (const ParametricInvocation* parametric_invocation :
       table.GetParametricInvocations()) {
    XLS_RETURN_IF_ERROR(converter.AddInvocation(parametric_invocation));
  }

  // Generate the type info.
  for (const TypeInfoConversionStep& step : order_visitor.steps()) {
    XLS_RETURN_IF_ERROR(
        absl::visit(Visitor{[&](const NodeConversion& node_conversion) {
                              return converter.GenerateTypeInfo(
                                  node_conversion.parametric_invocation,
                                  node_conversion.node);
                            },
                            [&](const ParametricEnvConversion& env_conversion) {
                              return converter.GenerateParametricEnvs(
                                  env_conversion.parametric_invocation);
                            }},
                    step));
  }

  return converter.GetBaseTypeInfo();
}

}  // namespace xls::dslx
