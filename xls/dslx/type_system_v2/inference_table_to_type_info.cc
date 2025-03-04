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
#include <iterator>
#include <memory>
#include <optional>
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
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
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
#include "xls/dslx/type_system/type_zero_value.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/expand_variables.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/solve_for_parametrics.h"
#include "xls/dslx/type_system_v2/type_annotation_resolver.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/type_system_v2/unify_type_annotations.h"
#include "xls/dslx/type_system_v2/validate_concrete_type.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// The result of resolving the target of a function call. If the `target_object`
// is specified, then it is an instance method being invoked on `target_object`.
// Otherwise, it is a static function which may or may not be a member.
struct FunctionAndTargetObject {
  const Function* function = nullptr;
  const std::optional<Expr*> target_object;
  std::optional<const ParametricContext*> target_struct_context;
  // This is part of a temporary hack to allow type checking of certain
  // builtins. Built-ins in the builtin_stubs.x file will NOT have this
  // value set to true.
  bool is_special_builtin = false;
};

// Traverses an AST and flattens it into a `vector` in the order the `TypeInfo`
// needs to be built such that prerequisites will be present in `TypeInfo` when
// evaluations are done.
class ConversionOrderVisitor : public AstNodeVisitorWithDefault {
 public:
  explicit ConversionOrderVisitor(bool handle_parametric_entities)
      : handle_parametric_entities_(handle_parametric_entities) {}

  absl::Status HandleFunction(const Function* node) override {
    if (!handle_parametric_entities_ && node->IsParametric()) {
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status HandleImpl(const Impl* node) override {
    std::optional<const StructDefBase*> def =
        GetStructOrProcDef(node->struct_ref());
    CHECK(def.has_value());
    if (!handle_parametric_entities_ && (*def)->IsParametric()) {
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status HandleInvocation(const Invocation* node) override {
    // Exclude the arguments of invocations, but otherwise do the equivalent of
    // DefaultHandler. We exclude the arguments, because when an argument should
    // be converted depends on whether its type is determining or determined by
    // the formal argument type (it's determining it if it's based on an
    // implicit parametric). `ConvertInvocation` decides this.
    for (const ExprOrType& parametric : node->explicit_parametrics()) {
      XLS_RETURN_IF_ERROR(ToAstNode(parametric)->Accept(this));
    }
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleLet(const Let* node) override {
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(node->type_annotation()->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->rhs()->Accept(this));
    nodes_.push_back(node);
    for (const NameDef* name_def : node->name_def_tree()->GetNameDefs()) {
      XLS_RETURN_IF_ERROR(name_def->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleMatch(const Match* node) override {
    for (const MatchArm* arm : node->arms()) {
      XLS_RETURN_IF_ERROR(arm->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->matched()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleMatchArm(const MatchArm* node) override {
    for (const NameDefTree* name_def_tree : node->patterns()) {
      XLS_RETURN_IF_ERROR(name_def_tree->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->expr()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleRestOfTuple(const RestOfTuple* node) override {
    return absl::OkStatus();
  }

  absl::Status HandleNameDefTree(const NameDefTree* node) override {
    if (!node->IsRestOfTupleLeaf()) {
      nodes_.push_back(node);
    }
    for (auto child : node->Flatten()) {
      XLS_RETURN_IF_ERROR(ToAstNode(child)->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(node->type_annotation()->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->value()->Accept(this));
    XLS_RETURN_IF_ERROR(node->name_def()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    // Prefer conversion of invocations before nodes that may use them.
    std::vector<const AstNode*> invocations;
    std::vector<const AstNode*> non_invocations;

    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      if (const auto* current_invocation =
              dynamic_cast<const Invocation*>(node);
          current_invocation != nullptr &&
          child == current_invocation->callee()) {
        continue;
      }
      if (child->kind() == AstNodeKind::kInvocation) {
        invocations.push_back(child);
      } else {
        non_invocations.push_back(child);
      }
    }
    for (const AstNode* child : invocations) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    for (const AstNode* child : non_invocations) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  const std::vector<const AstNode*>& nodes() const { return nodes_; }

 private:
  const bool handle_parametric_entities_;
  std::vector<const AstNode*> nodes_;
};

// An object that facilitates the conversion of an `InferenceTable` to
// `TypeInfo`.
class InferenceTableConverter : public UnificationErrorGenerator,
                                public Evaluator,
                                public ParametricStructInstantiator {
 public:
  InferenceTableConverter(
      InferenceTable& table, Module& module, ImportData& import_data,
      WarningCollector& warning_collector, TypeInfo* base_type_info,
      const FileTable& file_table, TypeSystemTracer& tracer,
      std::optional<std::unique_ptr<InferenceTableConverter>>
          builtins_converter)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        base_type_info_(base_type_info),
        file_table_(file_table),
        tracer_(tracer),
        resolver_(TypeAnnotationResolver::Create(
            module, table, file_table,
            /*error_generator=*/*this, /*evaluator=*/*this,
            /*parametric_struct_instantiator=*/*this, tracer_)),
        builtins_converter_(std::move(builtins_converter)) {}

  bool IsBuiltin(const Function* node) {
    // It's a builtin if we're the builtin converter and it's in the builtin
    // module.
    if (builtins_converter_.has_value()) {
      return (*builtins_converter_)->IsBuiltin(node);
    }
    return module_.GetFunction(node->identifier()).has_value();
  }

  // Converts all type info for the subtree rooted at `node`. `function` is
  // the containing function of the subtree, if any. `parametric_context` is
  // the invocation in whose context the types should be evaluated, if any.
  //
  // When `node` is an actual function argument that is being converted in order
  // to determine a parametric in its own formal type, special behavior is
  // needed, which is enabled by the `filter_param_type_annotations` flag. In
  // such a case, the argument may have one annotation that is
  // `ParamType(function_type, n)`, and since that is the very thing we are
  // really trying to infer, we can't factor it in to the type of the argument
  // value. In all other cases, the flag should be false.
  absl::Status ConvertSubtree(
      const AstNode* node, std::optional<const Function*> function,
      std::optional<const ParametricContext*> parametric_context,
      bool filter_param_type_annotations = false) {
    // Avoid converting a subtree multiple times, as a performance optimization.
    // Evaluation functions convert the `Expr` they are being asked to evaluate
    // in case it is fabricated. For an `Expr` that is actually present in the
    // original AST, this should be a no-op.
    if (!filter_param_type_annotations &&
        !converted_subtrees_[parametric_context].insert(node).second) {
      return absl::OkStatus();
    }

    VLOG(5) << "ConvertSubtree: " << node->ToString()
            << " of module: " << module_.name()
            << " in context: " << ToString(parametric_context);
    if (function.has_value() && IsBuiltin(*function)) {
      // "Converting" a built-in function is unnecessary and fraught, so
      // we skip it.
      return absl::OkStatus();
    }
    if (node->owner()->name() != module_.name()) {
      // use the other one ?
      VLOG(5) << "Wrong module in ConvertSubtree; delegating to builtins "
                 "converter";
      return (*builtins_converter_)
          ->ConvertSubtree(node, function, parametric_context,
                           filter_param_type_annotations);
    }
    ConversionOrderVisitor visitor(
        parametric_context.has_value() &&
        (node == function || (node->parent() != nullptr &&
                              node->parent()->kind() == AstNodeKind::kImpl)));
    XLS_RETURN_IF_ERROR(node->Accept(&visitor));
    for (const AstNode* node : visitor.nodes()) {
      VLOG(5) << "Next node: " << node->ToString();
      if (const auto* invocation = dynamic_cast<const Invocation*>(node)) {
        XLS_RETURN_IF_ERROR(
            ConvertInvocation(invocation, function, parametric_context));
      } else {
        XLS_RETURN_IF_ERROR(GenerateTypeInfo(
            parametric_context, node,
            /*pre_unified_type=*/
            std::nullopt,
            /*type_annotation_accept_predicate=*/
            [&](const TypeAnnotation* annotation) {
              return !filter_param_type_annotations ||
                     dynamic_cast<const ParamTypeAnnotation*>(annotation) ==
                         nullptr;
            }));
      }
    }
    return absl::OkStatus();
  }

  // Adds the invocation type info for the given parametric invocation to the
  // output type info. `converted_parametric_envs_` must be populated for both
  // the caller and callee of the parametric invocation before doing this.
  absl::Status AddInvocationTypeInfo(
      const ParametricContext* parametric_context) {
    VLOG(5) << "Adding invocation type info for "
            << parametric_context->ToString();
    ParametricEnv parent_env;
    const auto& data =
        std::get<ParametricInvocationDetails>(parametric_context->details());
    if (parametric_context->parent_context().has_value() &&
        (*parametric_context->parent_context())->is_invocation()) {
      parent_env =
          converted_parametric_envs_.at(*parametric_context->parent_context());
    }
    ParametricEnv callee_env =
        converted_parametric_envs_.at(parametric_context);
    VLOG(5) << "Parent env: " << parent_env.ToString();
    VLOG(5) << "Callee env: " << callee_env.ToString();
    const auto* invocation =
        dynamic_cast<const Invocation*>(parametric_context->node());
    CHECK_NE(invocation, nullptr);
    XLS_RETURN_IF_ERROR(base_type_info_->AddInvocationTypeInfo(
        *invocation, data.caller.has_value() ? *data.caller : nullptr,
        parent_env, callee_env,
        parametric_context_type_info_.at(parametric_context)));
    return absl::OkStatus();
  }

  // Converts the type info for the given invocation node and its argument
  // nodes. This involves resolving the callee function and applying the formal
  // types of the arguments to the actual arguments in the inference table.
  absl::Status ConvertInvocation(
      const Invocation* invocation, std::optional<const Function*> caller,
      std::optional<const ParametricContext*> caller_context) {
    TypeSystemTrace trace =
        tracer_.TraceConvertInvocation(invocation, caller_context);
    VLOG(5) << "Converting invocation: " << invocation->callee()->ToString()
            << " with module: " << invocation->owner()->name()
            << " in module: " << module_.name()
            << " in context: " << ToString(caller_context);
    XLS_ASSIGN_OR_RETURN(
        const FunctionAndTargetObject function_and_target_object,
        ResolveFunction(invocation->callee(), caller, caller_context));

    // This is a temporary hack for type-checking annotations we generate that
    // contain invocations of *certain* builtins like `element_count`. We can
    // remove this when we have type checking for builtin functions with <T:
    // type> parametrics.
    if (function_and_target_object.is_special_builtin) {
      for (ExprOrType parametric : invocation->explicit_parametrics()) {
        VLOG(5) << "Convert parametric of builtin: "
                << ToAstNode(parametric)->ToString();
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(ToAstNode(parametric), caller, caller_context));
      }
      return GenerateTypeInfo(caller_context, invocation,
                              CreateU32Annotation(module_, invocation->span()));
    }

    const Function* function = function_and_target_object.function;

    // Come up with the actual args by merging the possible target object
    // (`some_struct` in the case of `some_struct.foo(args)`), with the vector
    // of explicit actual args.
    std::vector<const Expr*> actual_args;
    if (function_and_target_object.target_object.has_value()) {
      actual_args.push_back(*function_and_target_object.target_object);
    }
    absl::c_copy(invocation->args(), std::back_inserter(actual_args));

    if (!function->IsParametric()) {
      XLS_RETURN_IF_ERROR(
          GenerateTypeInfo(function_and_target_object.target_struct_context,
                           invocation->callee()));
      // For non-parametric functions, the formal argument types can be taken at
      // face value. Apply them to the actual arguments, convert them, and
      // convert the invocation itself. We use the unified signature rather than
      // the `Function` object for this, because the `Function` may have struct
      // parametrics in it which are outside their domain here, and the unified
      // signature will not.
      XLS_ASSIGN_OR_RETURN(std::optional<const TypeAnnotation*> signature,
                           resolver_->ResolveAndUnifyTypeAnnotationsForNode(
                               function_and_target_object.target_struct_context,
                               invocation->callee()));
      CHECK(signature.has_value());
      const auto* function_type =
          dynamic_cast<const FunctionTypeAnnotation*>(*signature);
      CHECK(function_type);
      for (int i = 0; i < function_type->param_types().size(); i++) {
        const TypeAnnotation* formal_param = function_type->param_types()[i];
        const Expr* actual_param = actual_args[i];
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(actual_param, formal_param));
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(actual_param, caller, caller_context));
      }
      return GenerateTypeInfo(caller_context, invocation,
                              function_type->return_type());
    }

    // If we get here, we are dealing with a parametric function. First let's
    // make sure a valid number of parametrics are being passed in.
    if (invocation->explicit_parametrics().size() >
        function->parametric_bindings().size()) {
      return ArgCountMismatchErrorStatus(
          invocation->span(),
          absl::Substitute(
              "Too many parametric values supplied; limit: $0 given: $1",
              function->parametric_bindings().size(),
              invocation->explicit_parametrics().size()),
          file_table_);
    }

    // The parametric invocation now gets its own data structure set up in both
    // the `InferenceTable` and the `TypeInfo` hierarchy.
    XLS_ASSIGN_OR_RETURN(
        const ParametricContext* invocation_context,
        table_.AddParametricInvocation(
            *invocation, *function, caller, caller_context,
            function_and_target_object.target_struct_context.has_value()
                ? (*function_and_target_object.target_struct_context)
                      ->self_type()
                : std::nullopt));
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * invocation_type_info,
        import_data_.type_info_owner().New(
            &module_,
            function_and_target_object.target_struct_context.has_value()
                ? parametric_context_type_info_.at(
                      *function_and_target_object.target_struct_context)
                : base_type_info_));
    parametric_context_type_info_.emplace(invocation_context,
                                          invocation_type_info);

    // Assign the formal parametric types to the actual explicit parametric
    // arguments, now that we know the formal types.
    const std::vector<ParametricBinding*>& bindings =
        function->parametric_bindings();
    const std::vector<ExprOrType>& explicit_parametrics =
        invocation->explicit_parametrics();
    for (int i = 0; i < explicit_parametrics.size(); i++) {
      ExprOrType explicit_parametric = explicit_parametrics[i];
      const ParametricBinding* formal_parametric = bindings[i];
      if (std::holds_alternative<Expr*>(explicit_parametric)) {
        const Expr* parametric_value_expr =
            std::get<Expr*>(explicit_parametric);
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            parametric_value_expr, formal_parametric->type_annotation()));
      }
    }

    // Convert the explicit parametrics that are being passed.
    for (ExprOrType explicit_parametric : invocation->explicit_parametrics()) {
      XLS_RETURN_IF_ERROR(ConvertSubtree(ToAstNode(explicit_parametric), caller,
                                         caller_context));
    }

    // Convert the default expressions in the context of this invocation.
    for (const ParametricBinding* binding : function->parametric_bindings()) {
      if (binding->expr() != nullptr) {
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(binding->expr(), function, invocation_context));
      }
    }

    // Figure out any implicit parametrics and generate the `ParametricEnv`.
    XLS_RETURN_IF_ERROR(GenerateParametricFunctionEnv(
        function_and_target_object.target_struct_context, invocation_context,
        invocation));
    XLS_RETURN_IF_ERROR(AddInvocationTypeInfo(invocation_context));

    // For an instance method call like `some_object.parametric_fn(args)`, type
    // checking will annotate the callee node, `some_object.parametric_fn` as
    // `MemberType(TVTA(some_object_var), "parametric_fn")`. This means the
    // expansion of the `MemberTypeAnnotation` may have function parametrics in
    // it, e.g.`(uN[N]) -> uN[N]`. Such an annotation is disruptive to
    // unification. An annotation like this will not be present for free
    // functions. It is extra work done by type checking because it can't tell
    // it's dealing with a parametric instance method. So we scrub these from
    // the table here, and the next step is to replace them with usable,
    // parametric-free types.
    const NameRef* callee_variable =
        *table_.GetTypeVariable(invocation->callee());
    XLS_RETURN_IF_ERROR(table_.RemoveTypeAnnotationsFromTypeVariable(
        callee_variable, [](const TypeAnnotation* annotation) {
          return dynamic_cast<const MemberTypeAnnotation*>(annotation) !=
                 nullptr;
        }));

    // Apply the parametric-free formal types to the arguments and convert them.
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* parametric_free_type,
        GetParametricFreeType(CreateFunctionTypeAnnotation(module_, *function),
                              parametric_value_exprs_.at(invocation_context),
                              invocation_context->self_type()));
    XLS_RETURN_IF_ERROR(
        ConvertSubtree(parametric_free_type, caller, caller_context));

    XLS_ASSIGN_OR_RETURN(
        parametric_free_type,
        resolver_->ResolveIndirectTypeAnnotations(
            invocation_context, parametric_free_type, std::nullopt));
    const FunctionTypeAnnotation* parametric_free_function_type =
        down_cast<const FunctionTypeAnnotation*>(parametric_free_type);

    XLS_RETURN_IF_ERROR(table_.AddTypeAnnotationToVariableForParametricContext(
        caller_context, callee_variable, parametric_free_function_type));
    XLS_RETURN_IF_ERROR(
        GenerateTypeInfo(function_and_target_object.target_struct_context,
                         invocation->callee()));
    for (int i = 0; i < parametric_free_function_type->param_types().size();
         i++) {
      const TypeAnnotation* formal_type =
          parametric_free_function_type->param_types()[i];
      const Expr* actual_param = actual_args[i];
      const bool is_self_param =
          i == 0 && function_and_target_object.target_object.has_value();
      XLS_RETURN_IF_ERROR(
          table_.AddTypeAnnotationToVariableForParametricContext(
              caller_context, *table_.GetTypeVariable(actual_param),
              formal_type));
      XLS_RETURN_IF_ERROR(ConvertSubtree(
          actual_param, caller,
          is_self_param ? function_and_target_object.target_struct_context
                        : caller_context));
    }

    // Convert the actual parametric function in the context of this invocation,
    // and finally, convert the invocation node.
    XLS_RETURN_IF_ERROR(ConvertSubtree(function, function, invocation_context));
    return GenerateTypeInfo(function_and_target_object.target_struct_context,
                            invocation);
  }

  // Gets the output `TypeInfo` corresponding to the given
  // `parametric_context`, which may be `nullopt`, in which case it returns
  // the base type info.
  TypeInfo* GetTypeInfo(
      std::optional<const ParametricContext*> parametric_context) {
    return parametric_context.has_value()
               ? parametric_context_type_info_.at(*parametric_context)
               : base_type_info_;
  }

  // Generates type info for one node.
  absl::Status GenerateTypeInfo(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node,
      std::optional<const TypeAnnotation*> pre_unified_type = std::nullopt,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          type_annotation_accept_predicate = std::nullopt) {
    // Don't generate type info for the rest of tuple wildcard.
    if (node->kind() == AstNodeKind::kRestOfTuple) {
      return absl::OkStatus();
    }
    TypeSystemTrace trace = tracer_.TraceConvertNode(node);
    VLOG(5) << "GenerateTypeInfo for node: " << node->ToString()
            << " with owner: " << node->owner()->name()
            << " for module: " << module_.name();
    if (pre_unified_type.has_value()) {
      VLOG(5) << "Using pre-unified type: " << (*pre_unified_type)->ToString();
    }
    TypeInfo* ti = GetTypeInfo(parametric_context);
    std::optional<const TypeAnnotation*> annotation = pre_unified_type;
    if (!annotation.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          annotation,
          resolver_->ResolveAndUnifyTypeAnnotationsForNode(
              parametric_context, node, type_annotation_accept_predicate));
    }

    // If the node itself is a `TypeAnnotation`, then, in the absence of any
    // other information, its `Type` is whatever that concretizes to.
    bool node_is_annotation = false;
    if (!annotation.has_value()) {
      if (const auto* node_as_annotation =
              dynamic_cast<const TypeAnnotation*>(node)) {
        annotation = node_as_annotation;
        node_is_annotation = true;
      } else {
        // The caller may have passed a node that is in the AST but not in the
        // table, and it may not be needed in the table.
        VLOG(5) << "No type information for: " << node->ToString();
        return absl::OkStatus();
      }
    }

    // If we have a let assignment or a NameDefTree to a tuple type, destructure
    // the assignment.
    if (dynamic_cast<const TupleTypeAnnotation*>(*annotation) &&
        (dynamic_cast<const Let*>(node) ||
         dynamic_cast<const NameDefTree*>(node))) {
      const NameDefTree* name_def_tree = dynamic_cast<const NameDefTree*>(node);
      if (name_def_tree == nullptr) {
        const auto* let = dynamic_cast<const Let*>(node);
        name_def_tree = let->name_def_tree();
      }
      XLS_RETURN_IF_ERROR(DestructureNameDefTree(
          parametric_context, name_def_tree,
          dynamic_cast<const TupleTypeAnnotation*>(*annotation)));
    }

    // Any annotation which actually gets used as the type of a node should be
    // converted itself, in order to generate type info for expressions that are
    // embedded in it, so those do not disrupt evaluations that occur in
    // concretization. These annotations may not be scheduled for conversion
    // otherwise, because they may be fabricated.
    if (node->kind() != AstNodeKind::kTypeAnnotation) {
      XLS_RETURN_IF_ERROR(ConvertSubtree(*annotation, /*function=*/std::nullopt,
                                         parametric_context));
    }

    absl::StatusOr<std::unique_ptr<Type>> type =
        Concretize(*annotation, parametric_context);
    if (!type.ok()) {
      // When the node itself is an annotation, and we decide to concretize
      // the node itself, we can't succeed in all contexts. For example, a
      // parametric-dependent field type declaration inside a parametric struct
      // declaration can't just be concretized like that. Rather than trying to
      // identify such cases, we just consider such nodes best-effort.
      return node_is_annotation ? absl::OkStatus() : type.status();
    }

    XLS_RETURN_IF_ERROR(ValidateConcreteType(
        node, type->get(), *ti, *annotation, warning_collector_, file_table_));
    if (const auto* literal = dynamic_cast<const Number*>(node);
        literal != nullptr && literal->type_annotation() != nullptr) {
      ti->SetItem(literal->type_annotation(),
                  *std::make_unique<MetaType>((*type)->CloneToUnique()));
    }
    ti->SetItem(node, **type);
    XLS_RETURN_IF_ERROR(NoteIfConstExpr(parametric_context, node, **type, ti));
    VLOG(5) << "Generated type: " << (*ti->GetItem(node))->ToString()
            << " for node: " << node->ToString();
    return absl::OkStatus();
  }

  // Applies the appropriate type annotations to the names on the LHS of a
  // destructuring let, using the unified annotation for the RHS.
  absl::Status DestructureNameDefTree(
      std::optional<const ParametricContext*> parametric_context,
      const NameDefTree* node,
      const TupleTypeAnnotation* unified_rhs_type_annotation) {
    const auto add_annotation =
        [&](AstNode* name_def, TypeOrAnnotation type,
            std::optional<InterpValue> _) -> absl::Status {
      if (std::holds_alternative<const TypeAnnotation*>(type)) {
        std::optional<const NameRef*> type_var =
            table_.GetTypeVariable(name_def);

        const auto* type_annotation = std::get<const TypeAnnotation*>(type);
        if (type_var.has_value() &&
            // Don't attach a TypeVariableTypeAnnotation because it is an
            // indirect TypeAnnotation that may already link back to the node
            // via the TypeVariable.
            !dynamic_cast<const TypeVariableTypeAnnotation*>(type_annotation)) {
          XLS_RETURN_IF_ERROR(
              table_.AddTypeAnnotationToVariableForParametricContext(
                  parametric_context, *type_var, type_annotation));
        }
      }
      return absl::OkStatus();
    };
    return MatchTupleNodeToType(add_annotation, node,
                                unified_rhs_type_annotation, file_table_,
                                std::nullopt);
  }

  // Gets or creates the `ParametricContext` for a parameterization of a struct.
  // This boils down the `actual_parametrics` to `InterpValue`s and only deals
  // out one instance per set of equivalent `InterpValue`s for a struct. It
  // should be done upon encountering a `ColonRef` or similar node that calls
  // for a use of a struct type with some given parametrics.
  absl::StatusOr<const ParametricContext*> GetOrCreateParametricStructContext(
      std::optional<const ParametricContext*> parent_context,
      const StructOrProcRef& ref, const AstNode* node) {
    VLOG(6) << "Get or create parametric struct context for: "
            << ref.def->identifier();
    XLS_ASSIGN_OR_RETURN(
        ParametricEnv parametric_env,
        GenerateParametricStructEnv(parent_context, *ref.def, ref.parametrics,
                                    *node->GetSpan()));
    VLOG(6) << "Struct env: " << parametric_env.ToString();
    const ParametricContext* struct_context =
        table_.GetOrCreateParametricStructContext(
            ref.def, node, parametric_env,
            CreateStructAnnotation(module_, ref));
    const auto it = parametric_context_type_info_.find(struct_context);
    if (it != parametric_context_type_info_.end()) {
      return struct_context;
    }
    converted_parametric_envs_.emplace(struct_context, parametric_env);
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti, import_data_.type_info_owner().New(
                                            &module_, base_type_info_));
    parametric_context_type_info_.emplace(struct_context, ti);
    absl::flat_hash_map<std::string, InterpValue> env_values =
        parametric_env.ToMap();
    absl::flat_hash_map<const NameDef*, ExprOrType> value_exprs;
    for (int i = 0; i < ref.def->parametric_bindings().size(); i++) {
      ParametricBinding* binding = ref.def->parametric_bindings()[i];
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> binding_type,
          Concretize(binding->type_annotation(), std::nullopt));
      std::optional<InterpValue> value;
      const auto it = env_values.find(binding->identifier());
      if (it != env_values.end()) {
        value = it->second;
        value_exprs.emplace(binding->name_def(), ref.parametrics[i]);
      } else {
        // Defaulted values are not placed in the env due to a chicken-and-egg
        // problem (plus the fact that they would be superfluous as key data).
        // However, `GenerateParametricStructEnv` does ensure that any such
        // bindings do have a default expr.
        value_exprs.emplace(binding->name_def(), binding->expr());
        XLS_ASSIGN_OR_RETURN(
            value,
            Evaluate(ParametricContextScopedExpr(
                struct_context, binding->type_annotation(), binding->expr())));
      }

      VLOG(6) << "Setting binding: " << binding->identifier()
              << " in context: " << ToString(struct_context)
              << " to value: " << value->ToString();

      ti->SetItem(binding->name_def(), *binding_type);
      ti->NoteConstExpr(binding->name_def(), *value);
    }
    parametric_value_exprs_.emplace(struct_context, std::move(value_exprs));
    if (ref.def->impl().has_value()) {
      for (const ConstantDef* constant : (*ref.def->impl())->GetConstants()) {
        VLOG(6) << "Generate parametric impl constant: " << constant->ToString()
                << " in context: " << ToString(struct_context);
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(constant, std::nullopt, struct_context));
      }
    }
    return struct_context;
  }

  // Generates an env for a parametric struct using the given parametric value
  // expressions. If there are fewer actual parametrics than bindings, then the
  // remainder of the bindings must all have default exprs. None of the
  // parametrics will be inferred by this function; if any inference is needed,
  // it must be done before producing the `actual_parametrics` vector passed
  // into this function.
  absl::StatusOr<ParametricEnv> GenerateParametricStructEnv(
      std::optional<const ParametricContext*> parent_context,
      const StructDefBase& struct_def,
      const std::vector<ExprOrType>& actual_parametrics,
      const Span& error_span) {
    absl::flat_hash_map<std::string, InterpValue> values;
    CHECK_GE(struct_def.parametric_bindings().size(),
             actual_parametrics.size());
    std::vector<std::string> missing_parametric_names;
    for (int i = 0; i < struct_def.parametric_bindings().size(); i++) {
      const ParametricBinding* binding = struct_def.parametric_bindings()[i];
      if (i >= actual_parametrics.size()) {
        // The default must exist but is not computed yet or put in the env.
        CHECK(binding->expr() != nullptr);
        continue;
      }
      ExprOrType parametric = actual_parametrics[i];
      if (std::holds_alternative<Expr*>(parametric)) {
        VLOG(6) << "Actual parametric: " << binding->identifier()
                << " expr: " << std::get<Expr*>(parametric)->ToString();
        XLS_ASSIGN_OR_RETURN(InterpValue value,
                             Evaluate(ParametricContextScopedExpr(
                                 parent_context, binding->type_annotation(),
                                 std::get<Expr*>(parametric))));
        VLOG(6) << "Actual parametric: " << binding->identifier()
                << " value: " << value.ToString();
        values.emplace(binding->identifier(), value);
      } else {
        return absl::UnimplementedError(
            "Type inference version 2 is a work in progress and cannot yet "
            "handle generic type parametrics for structs");
      }
    }
    return ParametricEnv(values);
  }

  // Returns the resulting base type info for the entire conversion.
  TypeInfo* GetBaseTypeInfo() { return base_type_info_; }

 private:
  // Converts the given type annotation to a concrete `Type`, either statically
  // or in the context of a parametric invocation.
  absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context) {
    TypeSystemTrace trace = tracer_.TraceConcretize(annotation);
    VLOG(5) << "Concretize: " << annotation->ToString()
            << " in context invocation: " << ToString(parametric_context);
    VLOG(5) << "Effective context: " << ToString(parametric_context);

    XLS_ASSIGN_OR_RETURN(annotation, resolver_->ResolveIndirectTypeAnnotations(
                                         parametric_context, annotation,
                                         /*accept_predicate=*/std::nullopt));
    if (dynamic_cast<const AnyTypeAnnotation*>(annotation) != nullptr) {
      return absl::InvalidArgumentError(
          "Attempting to concretize `Any` type, which means there was "
          "insufficient type info.");
    }
    if (const auto* tuple =
            dynamic_cast<const TupleTypeAnnotation*>(annotation)) {
      std::vector<std::unique_ptr<Type>> member_types;
      member_types.reserve(tuple->members().size());
      for (const TypeAnnotation* member : tuple->members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> concrete_member_type,
                             Concretize(member, parametric_context));
        member_types.push_back(std::move(concrete_member_type));
      }
      return std::make_unique<TupleType>(std::move(member_types));
    }
    if (const auto* array = CastToNonBitsArrayTypeAnnotation(annotation)) {
      XLS_ASSIGN_OR_RETURN(int64_t size,
                           EvaluateU32OrExpr(parametric_context, array->dim()));
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> element_type,
          Concretize(array->element_type(), parametric_context));
      return std::make_unique<ArrayType>(std::move(element_type),
                                         TypeDim(InterpValue::MakeU32(size)));
    }
    if (const auto* function =
            dynamic_cast<const FunctionTypeAnnotation*>(annotation)) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> return_type,
          Concretize(function->return_type(), parametric_context));
      std::vector<std::unique_ptr<Type>> param_types;
      param_types.reserve(function->param_types().size());
      for (const TypeAnnotation* argument : function->param_types()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> param_type,
                             Concretize(argument, parametric_context));
        param_types.push_back(std::move(param_type));
      }
      return std::make_unique<FunctionType>(std::move(param_types),
                                            std::move(return_type));
    }
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc,
                         GetStructOrProcRef(annotation, file_table_));
    if (struct_or_proc.has_value()) {
      const StructDefBase* struct_def_base = struct_or_proc->def;
      CHECK(struct_def_base != nullptr);
      std::vector<std::unique_ptr<Type>> member_types;
      member_types.reserve(struct_def_base->members().size());
      std::optional<const ParametricContext*> struct_context;
      if (struct_def_base->IsParametric()) {
        XLS_ASSIGN_OR_RETURN(struct_context, GetOrCreateParametricStructContext(
                                                 parametric_context,
                                                 *struct_or_proc, annotation));
      }
      for (const StructMemberNode* member : struct_def_base->members()) {
        XLS_ASSIGN_OR_RETURN(
            const TypeAnnotation* parametric_free_member_type,
            GetParametricFreeStructMemberType(struct_context, *struct_or_proc,
                                              member->type()));
        XLS_ASSIGN_OR_RETURN(
            std::unique_ptr<Type> concrete_member_type,
            Concretize(parametric_free_member_type, parametric_context));
        member_types.push_back(std::move(concrete_member_type));
      }
      if (struct_def_base->kind() == AstNodeKind::kStructDef) {
        return std::make_unique<StructType>(
            std::move(member_types),
            *dynamic_cast<const StructDef*>(struct_def_base));
      }
      return std::make_unique<ProcType>(
          std::move(member_types),
          *dynamic_cast<const ProcDef*>(struct_def_base));
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
        EvaluateBoolOrExpr(parametric_context,
                           signedness_and_bit_count->signedness));
    XLS_ASSIGN_OR_RETURN(
        int64_t bit_count,
        EvaluateU32OrExpr(parametric_context,
                          signedness_and_bit_count->bit_count));
    VLOG(5) << "Concretized: " << annotation->ToString()
            << " to signed: " << signedness << ", bit count: " << bit_count;
    return std::make_unique<BitsType>(signedness, bit_count);
  }

  // Helper that notes the constexpr value for `node` in `ti`, if applicable,
  // once its concrete `type` has been determined.
  absl::Status NoteIfConstExpr(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node, const Type& type, TypeInfo* ti) {
    if (const auto* constant_def = dynamic_cast<const ConstantDef*>(node)) {
      VLOG(6) << "Checking constant def value: " << constant_def->ToString()
              << " with type: " << type.ToString();
      absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
          &import_data_, ti, &warning_collector_, ParametricEnv(),
          constant_def->value(), &type);
      if (value.ok()) {
        VLOG(6) << "Constant def: " << constant_def->ToString()
                << " has value: " << value->ToString();
        ti->NoteConstExpr(constant_def, *value);
        ti->NoteConstExpr(constant_def->value(), *value);
        ti->NoteConstExpr(constant_def->name_def(), *value);
      }
    }
    if (const auto* zero_macro = dynamic_cast<const ZeroMacro*>(node)) {
      VLOG(6) << "Checking zero_macro value: " << zero_macro->ToString()
              << " with type: " << type.ToString();

      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           MakeZeroValue(type, import_data_, *node->GetSpan()));
      ti->NoteConstExpr(zero_macro, value);
    }
    if (const auto* all_ones_macro = dynamic_cast<const AllOnesMacro*>(node)) {
      VLOG(6) << "Checking all_ones_macro value: " << all_ones_macro->ToString()
              << " with type: " << type.ToString();

      XLS_ASSIGN_OR_RETURN(
          InterpValue value,
          MakeAllOnesValue(type, import_data_, *node->GetSpan()));
      ti->NoteConstExpr(all_ones_macro, value);
    }
    if (const auto* name_ref = dynamic_cast<const NameRef*>(node)) {
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        const NameDef* name_def =
            std::get<const NameDef*>(name_ref->name_def());
        if (ti->IsKnownConstExpr(name_def)) {
          ti->NoteConstExpr(name_ref, *ti->GetConstExprOption(name_def));
        }
      }
    }
    if (const auto* colon_ref = dynamic_cast<const ColonRef*>(node)) {
      return NoteColonRefIfConstExpr(parametric_context, colon_ref, type, ti);
    }
    if (const auto* number = dynamic_cast<const Number*>(node)) {
      XLS_ASSIGN_OR_RETURN(InterpValue value, EvaluateNumber(*number, type));
      ti->NoteConstExpr(number, value);
    }
    if (const auto* let = dynamic_cast<const Let*>(node)) {
      return NoteLetIfConstExpr(let, type, ti);
    }
    if (const auto* index = dynamic_cast<const Index*>(node)) {
      // A `Slice` actually has its bounds stored in `TypeInfo` out-of-band from
      // the real type info, mirroring the `StartAndWidthExprs` that we store in
      // the `InferenceTable`.
      if (std::holds_alternative<Slice*>(index->rhs()) ||
          std::holds_alternative<WidthSlice*>(index->rhs())) {
        XLS_RETURN_IF_ERROR(ConcretizeSlice(parametric_context, index, ti));
      }
    }
    return absl::OkStatus();
  }

  // Adds the concrete start and width value of the slice requested by the given
  // `index` node to the given `TypeInfo`.
  absl::Status ConcretizeSlice(
      std::optional<const ParametricContext*> parametric_context,
      const Index* index, TypeInfo* ti) {
    std::optional<StartAndWidthExprs> start_and_width_exprs =
        table_.GetSliceStartAndWidthExprs(index);
    CHECK(start_and_width_exprs.has_value());
    StartAndWidth concrete_start_and_width;
    XLS_ASSIGN_OR_RETURN(
        concrete_start_and_width.start,
        EvaluateU32OrExpr(parametric_context, start_and_width_exprs->start));
    XLS_ASSIGN_OR_RETURN(
        concrete_start_and_width.width,
        EvaluateU32OrExpr(parametric_context, start_and_width_exprs->width));
    const Type& array_type = **ti->GetItem(index->lhs());
    int64_t array_size;
    if (array_type.IsArray()) {
      XLS_ASSIGN_OR_RETURN(array_size,
                           array_type.AsArray().size().GetAsInt64());
    } else {
      std::optional<BitsLikeProperties> bits_like = GetBitsLike(array_type);
      CHECK(bits_like.has_value());
      XLS_ASSIGN_OR_RETURN(array_size, bits_like->size.GetAsInt64());
    }

    if (concrete_start_and_width.start < 0 ||
        concrete_start_and_width.start + concrete_start_and_width.width >
            array_size) {
      return TypeInferenceErrorStatus(
          index->span(), nullptr,
          absl::StrCat("Slice range out of bounds for array of size ",
                       array_size),
          file_table_);
    }

    if (std::holds_alternative<Slice*>(index->rhs())) {
      ti->AddSliceStartAndWidth(
          std::get<Slice*>(index->rhs()),
          parametric_context.has_value()
              ? converted_parametric_envs_.at(*parametric_context)
              : ParametricEnv{},
          concrete_start_and_width);
    }
    return absl::OkStatus();
  }

  absl::Status NoteColonRefIfConstExpr(
      std::optional<const ParametricContext*> parametric_context,
      const ColonRef* colon_ref, const Type& type, TypeInfo* ti) {
    const std::optional<const AstNode*> target =
        table_.GetColonRefTarget(colon_ref);
    VLOG(6) << "Checking ColonRef constexpr value for: "
            << colon_ref->ToString() << " with target: "
            << (target.has_value() ? (*target)->ToString() : "none");
    // In a case like `S<parametrics>::CONSTANT`, what we do here is
    // constexpr-evaluate `CONSTANT` against the parametric context `TypeInfo`
    // for `S<parametrics>`. This will be `evaluation_ti`. Then we map the
    // result of that to the `ColonRef` node, in the `TypeInfo` where the
    // `ColonRef` resides (which is `ti`). In a non-parametric case like
    // `S::CONSTANT`, there is only one `TypeInfo` involved, so this logic
    // that figures out `evaluation_ti` is a no-op.
    TypeInfo* evaluation_ti = ti;
    if (target.has_value() && (*target)->kind() == AstNodeKind::kConstantDef) {
      if (std::holds_alternative<TypeRefTypeAnnotation*>(
              colon_ref->subject())) {
        XLS_ASSIGN_OR_RETURN(
            std::optional<StructOrProcRef> struct_or_proc,
            GetStructOrProcRef(
                std::get<TypeRefTypeAnnotation*>(colon_ref->subject()),
                file_table_));
        if (struct_or_proc.has_value() && struct_or_proc->def->IsParametric()) {
          XLS_ASSIGN_OR_RETURN(
              const ParametricContext* struct_context,
              GetOrCreateParametricStructContext(parametric_context,
                                                 *struct_or_proc, colon_ref));
          evaluation_ti = parametric_context_type_info_.at(struct_context);
        }
      }

      // Evaluate the value, and note it if successful.
      absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
          &import_data_, evaluation_ti, &warning_collector_, ParametricEnv(),
          dynamic_cast<const ConstantDef*>(*target)->value(), &type);
      if (value.ok()) {
        VLOG(6) << "Noting constexpr for ColonRef: " << colon_ref->ToString()
                << ", value: " << value->ToString();
        ti->NoteConstExpr(colon_ref, *value);
      }
    }
    return absl::OkStatus();
  }

  absl::Status NoteLetIfConstExpr(const Let* let, const Type& type,
                                  TypeInfo* ti) {
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti, &warning_collector_, ParametricEnv(), let->rhs(),
        &type);
    if (let->is_const()) {
      if (!value.ok()) {
        return value.status();
      }
      // Reminder: we don't allow name destructuring in constant defs, so this
      // is expected to never fail.
      XLS_RET_CHECK_EQ(let->name_def_tree()->GetNameDefs().size(), 1);
      NameDef* name_def = let->name_def_tree()->GetNameDefs()[0];
      WarnOnInappropriateConstantName(name_def->identifier(), let->span(),
                                      *let->owner(), &warning_collector_);
    } else if (!value.ok()) {
      return absl::OkStatus();
    }
    ti->NoteConstExpr(let, *value);
    ti->NoteConstExpr(let->rhs(), *value);
    const auto note_members =
        [&](AstNode* name_def, TypeOrAnnotation _,
            std::optional<InterpValue> const_expr) -> absl::Status {
      if (const_expr.has_value()) {
        ti->NoteConstExpr(name_def, *const_expr);
      }
      return absl::OkStatus();
    };
    XLS_RETURN_IF_ERROR(MatchTupleNodeToType(note_members, let->name_def_tree(),
                                             &type, file_table_, *value));
    return absl::OkStatus();
  }

  // Constexpr-evaluates the given expression, whose dependencies must already
  // be noted as constexpr's in the `TypeInfo` corresponding to the scope for
  // the expression.
  absl::StatusOr<InterpValue> Evaluate(
      const ParametricContextScopedExpr& scoped_expr) override {
    VLOG(7) << "Evaluate: " << scoped_expr.expr()->ToString()
            << " with owner: " << scoped_expr.expr()->owner()
            << " in module: " << module_.name()
            << " in context: " << ToString(scoped_expr.context());

    TypeInfo* type_info = base_type_info_;
    // Note that `scoped_expr` will not have a `context()` in a case like
    //  fn foo<X: u32>(...) { ... }
    //  fn bar() {
    //    foo<SOME_CONSTANT + 1>(...);
    //  }
    // The only scoped expr there is the expression being passed for `X`, which
    // is in a non-parametric caller and therefore cannot possibly refer to any
    // parametrics.
    if (scoped_expr.context().has_value()) {
      type_info = parametric_context_type_info_.at(*scoped_expr.context());
    }
    return Evaluate(scoped_expr.context(), type_info,
                    scoped_expr.type_annotation(), scoped_expr.expr());
  }

  // Variant that uses a specific `TypeInfo`. Use this directly when there is a
  // need to target a temporary `TypeInfo` object, e.g. for `StructInstance`
  // parametric values. When populating a real output `TypeInfo` object, prefer
  // the variant that takes an `ParametricContextScopedExpr`.
  absl::StatusOr<InterpValue> Evaluate(
      std::optional<const ParametricContext*> parametric_context,
      TypeInfo* type_info, const TypeAnnotation* type_annotation,
      const Expr* expr) {
    TypeSystemTrace trace = tracer_.TraceEvaluate(parametric_context, expr);

    // This is the type of the parametric binding we are talking about, which is
    // typically a built-in type, but the way we are concretizing it here would
    // support it being a complex type that even refers to other parametrics.
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         Concretize(type_annotation, parametric_context));
    type_info->SetItem(expr, *type);
    if (type_annotation->owner() == &module_) {
      // Prevent bleed-over from a different module.
      type_info->SetItem(type_annotation, MetaType(type->CloneToUnique()));
    }
    // TODO: https://github.com/google/xls/issues/193 - The if-statement below
    // is here temporarily to enable easy testing of parametric variables in
    // inference_table_test. The equivalent is done by `TypecheckModuleV2`, and
    // that's where the logic belongs, but that doesn't yet deal with parametric
    // variables.
    if (auto* number = dynamic_cast<const Number*>(expr);
        number != nullptr && number->type_annotation() != nullptr) {
      type_info->SetItem(number->type_annotation(),
                         MetaType(type->CloneToUnique()));
    }
    // Note: the `ParametricEnv` is irrelevant here, because we have guaranteed
    // that any parametric that may be referenced by the expr has been noted as
    // a normal constexpr in `type_info`.
    XLS_ASSIGN_OR_RETURN(InterpValue result,
                         ConstexprEvaluator::EvaluateToValue(
                             &import_data_, type_info, &warning_collector_,
                             ParametricEnv(), expr, /*type=*/nullptr));
    VLOG(7) << "Evaluation result for: " << expr->ToString()
            << " in context: " << ToString(parametric_context)
            << " value: " << result.ToString();
    return result;
  }

  // Determines what function is being invoked by a `callee` expression.
  absl::StatusOr<const FunctionAndTargetObject> ResolveFunction(
      const Expr* callee, std::optional<const Function*> caller_function,
      std::optional<const ParametricContext*> caller_context) {
    const AstNode* function_node = nullptr;
    std::optional<Expr*> target_object;
    std::optional<const ParametricContext*> target_struct_context =
        caller_context;
    if (const auto* colon_ref = dynamic_cast<const ColonRef*>(callee)) {
      std::optional<const AstNode*> target =
          table_.GetColonRefTarget(colon_ref);
      if (target.has_value()) {
        function_node = *target;
      }
    } else if (const auto* name_ref = dynamic_cast<const NameRef*>(callee)) {
      // Either a local function or a built-in function call.
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
        function_node = def->definer();
      } else if (std::holds_alternative<BuiltinNameDef*>(
                     name_ref->name_def())) {
        if (builtins_converter_.has_value()) {
          // Delegate to builtins converter.
          VLOG(5) << "ResolveFunction of builtin; delegating";
          return (*builtins_converter_)
              ->ResolveFunction(callee, caller_function, caller_context);
        }
        // Look it up in our module
        BuiltinNameDef* def = std::get<BuiltinNameDef*>(name_ref->name_def());
        auto fn_name = def->identifier();
        std::optional<Function*> builtin_fn = module_.GetFunction(fn_name);
        if (builtin_fn.has_value()) {
          function_node = *builtin_fn;
        } else if (fn_name == "array_size" || fn_name == "bit_count" ||
                   fn_name == "element_count") {
          VLOG(5) << "Could not find built-in function " << fn_name
                  << "; special-casing for now";
          return FunctionAndTargetObject{.is_special_builtin = true};
        } else {
          return TypeInferenceErrorStatus(
              name_ref->span(), nullptr,
              absl::Substitute("Cannot find built-in method `$0`", fn_name),
              file_table_);
        }
      }
    } else if (const auto* attr = dynamic_cast<const Attr*>(callee)) {
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(attr->lhs(), caller_function, caller_context));
      target_object = attr->lhs();
      XLS_ASSIGN_OR_RETURN(
          std::optional<const TypeAnnotation*> target_object_type,
          resolver_->ResolveAndUnifyTypeAnnotationsForNode(caller_context,
                                                           *target_object));
      XLS_ASSIGN_OR_RETURN(
          std::optional<StructOrProcRef> struct_or_proc_ref,
          GetStructOrProcRef(*target_object_type, file_table_));
      if (!struct_or_proc_ref.has_value()) {
        return TypeInferenceErrorStatus(
            attr->span(), nullptr,
            absl::Substitute(
                "Cannot invoke method `$0` on non-struct type `$1`",
                attr->attr(), (*target_object_type)->ToString()),
            file_table_);
      }
      if (struct_or_proc_ref->def->IsParametric()) {
        XLS_ASSIGN_OR_RETURN(target_struct_context,
                             GetOrCreateParametricStructContext(
                                 caller_context, *struct_or_proc_ref, callee));
      }
      std::optional<Impl*> impl = struct_or_proc_ref->def->impl();
      CHECK(impl.has_value());
      std::optional<Function*> instance_method =
          (*impl)->GetFunction(attr->attr());
      if (instance_method.has_value()) {
        function_node = *instance_method;
      } else {
        return TypeInferenceErrorStatusForAnnotation(
            callee->span(), *target_object_type,
            absl::Substitute(
                "Name '$0' is not defined by the impl for struct '$1'.",
                attr->attr(), struct_or_proc_ref->def->identifier()),
            file_table_);
      }
    }

    if (function_node != nullptr) {
      const auto* fn = dynamic_cast<const Function*>(function_node);
      if (fn == nullptr) {
        return TypeInferenceErrorStatus(
            callee->span(), nullptr,
            absl::Substitute("Invocation callee `$0` is not a function",
                             callee->ToString()),
            file_table_);
      }
      return FunctionAndTargetObject{fn, target_object, target_struct_context};
    }
    return TypeInferenceErrorStatus(
        callee->span(), nullptr,
        "An invocation callee must be a function, with a possible scope "
        "indicated using `::` or `.`",
        file_table_);
  }

  // Determines any implicit parametric values in the given `invocation`, and
  // generates its `ParametricEnv` in `converted_parametric_envs_`. Also
  // populates `parametric_value_exprs_` for the invocation.
  absl::Status GenerateParametricFunctionEnv(
      std::optional<const ParametricContext*> callee_struct_context,
      const ParametricContext* invocation_context,
      const Invocation* invocation) {
    absl::flat_hash_map<std::string, InterpValue> values;
    absl::flat_hash_set<const ParametricBinding*> implicit_parametrics;
    ParametricBindings bindings(invocation_context->parametric_bindings());
    auto infer_pending_implicit_parametrics = [&]() -> absl::Status {
      if (implicit_parametrics.empty()) {
        return absl::OkStatus();
      }
      absl::flat_hash_map<std::string, InterpValue> new_values;
      XLS_ASSIGN_OR_RETURN(
          new_values, InferImplicitFunctionParametrics(invocation_context,
                                                       implicit_parametrics));
      implicit_parametrics.clear();
      values.merge(std::move(new_values));
      return absl::OkStatus();
    };

    for (int i = 0; i < invocation_context->parametric_bindings().size(); i++) {
      const ParametricBinding* binding =
          invocation_context->parametric_bindings()[i];

      if (i < invocation->explicit_parametrics().size() &&
          dynamic_cast<const GenericTypeAnnotation*>(
              binding->type_annotation())) {
        // This is a <T: type> reference
        ExprOrType actual_parametric_type =
            invocation->explicit_parametrics()[i];
        const NameRef* name_ref = module_.Make<NameRef>(
            module_.span(), binding->identifier(), binding->name_def());
        XLS_RETURN_IF_ERROR(
            table_.AddTypeAnnotationToVariableForParametricContext(
                invocation_context, name_ref,
                std::get<TypeAnnotation*>(actual_parametric_type)));
        continue;
      }

      std::optional<ParametricContextScopedExpr> expr =
          table_.GetParametricValue(*binding->name_def(), *invocation_context);
      if (expr.has_value()) {
        // The expr may be a default expr which may use the inferred values of
        // any parametrics preceding it, so let's resolve any pending implicit
        // ones now.
        XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());
        // Now evaluate the expr.
        XLS_ASSIGN_OR_RETURN(InterpValue value, Evaluate(*expr));
        parametric_context_type_info_.at(invocation_context)
            ->NoteConstExpr(binding->name_def(), value);
        values.emplace(binding->name_def()->identifier(), value);
      } else {
        implicit_parametrics.insert(binding);
      }
    }

    // Resolve any implicit ones that are at the end of the list.
    XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());

    // Create the value exprs. This is basically an alternate format for the
    // `ParametricEnv` that is more readily useful for scrubbing the parametrics
    // from type annotations.
    absl::flat_hash_map<const NameDef*, ExprOrType> actual_parametrics;
    for (const auto& [name, value] : values) {
      const ParametricBinding* binding = bindings.at(name);
      XLS_ASSIGN_OR_RETURN(
          Number * value_expr,
          MakeTypeCheckedNumber(module_, table_, binding->span(), value,
                                binding->type_annotation()));
      actual_parametrics.emplace(binding->name_def(), value_expr);
    }
    if (callee_struct_context.has_value()) {
      for (const auto& [name_def, expr] :
           parametric_value_exprs_.at(*callee_struct_context)) {
        actual_parametrics.emplace(name_def, expr);
      }
    }
    parametric_value_exprs_.emplace(invocation_context,
                                    std::move(actual_parametrics));
    ParametricEnv env(std::move(values));
    converted_parametric_envs_.emplace(invocation_context, env);
    return absl::OkStatus();
  }

  // Attempts to infer the values of the specified implicit parametrics in an
  // invocation, using the types of the regular arguments being passed. If not
  // all of `implicit_parametrics` can be determined, this function returns an
  // error.
  absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
  InferImplicitFunctionParametrics(
      const ParametricContext* invocation_context,
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics) {
    VLOG(5) << "Inferring " << implicit_parametrics.size()
            << " implicit parametrics for invocation: "
            << ToString(invocation_context);
    const auto& context_data =
        std::get<ParametricInvocationDetails>(invocation_context->details());
    const auto* invocation =
        dynamic_cast<const Invocation*>(invocation_context->node());
    CHECK_NE(invocation, nullptr);
    const absl::Span<Param* const> formal_args = context_data.callee->params();
    const absl::Span<Expr* const> actual_args = invocation->args();
    TypeInfo* ti = parametric_context_type_info_.at(invocation_context);
    std::vector<const TypeAnnotation*> formal_types;
    formal_types.reserve(formal_args.size());
    for (int i = 0; i < formal_args.size(); i++) {
      const Param* param = formal_args[i];

      // The implicit `self` argument of an instance method has no corresponding
      // actual arg, so we must leave it out in order for the formal args to
      // line up. It's also not part of the task here to try to infer the
      // parametrics of the struct referred to by `Self`; only the parametric
      // bindings belonging to the function.
      if (i == 0 &&
          dynamic_cast<const SelfTypeAnnotation*>(param->type_annotation())) {
        continue;
      }
      formal_types.push_back(param->type_annotation());
    }

    TypeInfo* actual_arg_ti = base_type_info_;
    if (invocation_context->parent_context().has_value()) {
      actual_arg_ti = parametric_context_type_info_.at(
          *invocation_context->parent_context());
    }
    return InferImplicitParametrics(
        invocation_context->parent_context(), invocation_context,
        implicit_parametrics, formal_types, actual_args, ti, actual_arg_ti,
        /*caller_accept_predicate=*/[](const TypeAnnotation*) { return true; },
        /*pre_use_actual_arg=*/
        [&](const Expr* actual_arg) {
          // If an argument is essentially being used to figure out its own
          // formal type (due to the formal type depending on an implicit
          // parametric), then we need to convert the actual argument here,
          // before it is used to figure out the type. If it's having a known
          // formal type imposed on it, then `ConvertInvocation` will convert it
          // after deciding the formal type.
          return ConvertSubtree(actual_arg, context_data.caller,
                                invocation_context->parent_context(),
                                /*filter_param_type_annotations=*/true);
        });
  }

  // Attempts to infer the values of the specified implicit parametrics in an
  // invocation or struct instance, using the types of the regular arguments or
  // members being passed. If not all of `implicit_parametrics` can be
  // determined, this function returns an error. The `caller_accept_predicate`
  // allows the caller to filter some type annotations of the actual arguments
  // from consideration. The `pre_use_actual_arg` callback allows the caller to
  // be notified and do any desired prework before some actual argument gets
  // used to infer parametrics.
  absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
  InferImplicitParametrics(
      std::optional<const ParametricContext*> actual_arg_context,
      std::optional<const ParametricContext*> target_context,
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics,
      absl::Span<const TypeAnnotation* const> formal_types,
      absl::Span<Expr* const> actual_args, TypeInfo* output_ti,
      TypeInfo* actual_arg_ti,
      absl::FunctionRef<bool(const TypeAnnotation*)> caller_accept_predicate =
          [](const TypeAnnotation*) { return true; },
      absl::FunctionRef<absl::Status(const Expr*)> pre_use_actual_arg =
          [](const Expr*) { return absl::OkStatus(); }) {
    TypeSystemTrace trace =
        tracer_.TraceInferImplicitParametrics(implicit_parametrics);
    absl::flat_hash_map<std::string, InterpValue> values;
    ParametricBindings bindings(implicit_parametrics);
    for (int i = 0; i < formal_types.size() && !implicit_parametrics.empty();
         i++) {
      std::optional<const NameRef*> actual_arg_type_var =
          table_.GetTypeVariable(actual_args[i]);
      if (!actual_arg_type_var.has_value()) {
        VLOG(6) << "The actual argument: `" << actual_args[i]->ToString()
                << "` has no type variable.";
        continue;
      }
      VLOG(6) << "Using type variable: " << (*actual_arg_type_var)->ToString();
      XLS_RETURN_IF_ERROR(pre_use_actual_arg(actual_args[i]));
      XLS_ASSIGN_OR_RETURN(
          std::vector<const TypeAnnotation*> actual_arg_annotations,
          table_.GetTypeAnnotationsForTypeVariable(actual_arg_context,
                                                   *actual_arg_type_var));

      // The type variable for the actual argument should have at least one
      // annotation associated with it that came from the formal argument and is
      // therefore dependent on the parametric we are solving for. Let's unify
      // just the independent annotations(s) for the purposes of solving for the
      // variable.
      auto accept_predicate = [&](const TypeAnnotation* annotation) {
        return caller_accept_predicate(annotation) &&
               dynamic_cast<const ParamTypeAnnotation*>(annotation) ==
                   nullptr &&
               !HasAnyReferencesWithMissingTypeInfo(actual_arg_ti, annotation);
      };
      XLS_RETURN_IF_ERROR(resolver_->ResolveIndirectTypeAnnotations(
          actual_arg_context, actual_arg_annotations, accept_predicate));
      if (actual_arg_annotations.empty()) {
        VLOG(6) << "The actual argument type variable: "
                << (*actual_arg_type_var)->ToString()
                << " has no independent type annotations.";
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* actual_arg_type,
          resolver_->ResolveAndUnifyTypeAnnotations(
              actual_arg_context, actual_arg_annotations,
              actual_args[i]->span(), caller_accept_predicate));
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(actual_arg_type, std::nullopt, actual_arg_context));
      VLOG(5) << "Infer using actual type: " << actual_arg_type->ToString()
              << " and formal type: " << formal_types[i]->ToString()
              << " with effective context: " << ToString(actual_arg_context);
      if (auto tvta = dynamic_cast<const TypeVariableTypeAnnotation*>(
              formal_types[i])) {
        // Do not call "SolveForParametrics" if the implicit parametric is
        // a TypeVariableTypeAnnotation, because it will never become an
        // InterpValue. Instead, assign the formal type the actual arg type in
        // the target context.
        CHECK(target_context.has_value());
        XLS_RETURN_IF_ERROR(
            table_.AddTypeAnnotationToVariableForParametricContext(
                target_context, tvta->type_variable(), actual_arg_type));
        implicit_parametrics.erase(
            bindings.at(tvta->type_variable()->identifier()));
      } else {
        absl::flat_hash_map<const ParametricBinding*, InterpValue> resolved;
        XLS_ASSIGN_OR_RETURN(
            resolved,
            SolveForParametrics(
                actual_arg_type, formal_types[i], implicit_parametrics,
                [&](const TypeAnnotation* expected_type, const Expr* expr) {
                  return Evaluate(ParametricContextScopedExpr(
                      actual_arg_context, expected_type, expr));
                }));
        for (auto& [binding, value] : resolved) {
          VLOG(5) << "Inferred implicit parametric value: " << value.ToString()
                  << " for binding: " << binding->identifier()
                  << " using function argument: `" << actual_args[i]->ToString()
                  << "` of actual type: " << actual_arg_type->ToString();
          output_ti->NoteConstExpr(binding->name_def(), value);
          implicit_parametrics.erase(binding);
          values.emplace(binding->identifier(), std::move(value));
        }
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
      std::optional<const ParametricContext*> parametric_context,
      std::variant<bool, const Expr*> value_or_expr) override {
    if (std::holds_alternative<bool>(value_or_expr)) {
      return std::get<bool>(value_or_expr);
    }
    const Expr* expr = std::get<const Expr*>(value_or_expr);

    XLS_RETURN_IF_ERROR(ConvertSubtree(expr, std::nullopt, parametric_context));

    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Evaluate(ParametricContextScopedExpr(
            parametric_context, CreateBoolAnnotation(module_, expr->span()),
            expr)));
    return value.GetBitValueUnsigned();
  }

  absl::StatusOr<int64_t> EvaluateU32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) override {
    if (std::holds_alternative<int64_t>(value_or_expr)) {
      return std::get<int64_t>(value_or_expr);
    }
    const Expr* expr = std::get<const Expr*>(value_or_expr);

    XLS_RETURN_IF_ERROR(ConvertSubtree(expr, std::nullopt, parametric_context));

    std::optional<const TypeAnnotation*> type_annotation =
        table_.GetTypeAnnotation(expr);
    if (!type_annotation.has_value()) {
      type_annotation = CreateU32Annotation(module_, expr->span());
    }
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         Evaluate(ParametricContextScopedExpr(
                             parametric_context, *type_annotation, expr)));
    int64_t result;
    if (value.IsSigned()) {
      XLS_ASSIGN_OR_RETURN(result, value.GetBitValueSigned());
    } else {
      XLS_ASSIGN_OR_RETURN(result, value.GetBitValueUnsigned());
    }
    return result;
  }

  absl::StatusOr<const TypeAnnotation*> InstantiateParametricStruct(
      std::optional<const ParametricContext*> parent_context,
      const StructDef& struct_def,
      const std::vector<InterpValue>& explicit_parametrics,
      std::optional<const StructInstanceBase*> instantiator_node) override {
    // The goal here is to come up with a complete parametric value `Expr`
    // vector, which has a value for every formal binding, by inferring or
    // defaulting whichever ones are not explicit. The algorithm is the same as
    // for parametric function invocations, and the differences are in the
    // logistics. We build this as a map, `resolved_parametrics`, and convert it
    // to a vector at the end.
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * instance_type_info,
        import_data_.type_info_owner().New(
            &module_, parent_context.has_value()
                          ? parametric_context_type_info_.at(*parent_context)
                          : base_type_info_));
    ParametricBindings bindings(struct_def.parametric_bindings());
    absl::flat_hash_map<std::string, ExprOrType> resolved_parametrics;
    auto set_value = [&](const ParametricBinding* binding,
                         InterpValue value) -> absl::Status {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> binding_type,
          Concretize(binding->type_annotation(), parent_context));
      instance_type_info->SetItem(binding->name_def(), *binding_type);
      instance_type_info->NoteConstExpr(binding->name_def(), value);
      XLS_ASSIGN_OR_RETURN(
          Number * value_expr,
          MakeTypeCheckedNumber(module_, table_, binding->span(), value,
                                binding->type_annotation()));
      resolved_parametrics.emplace(binding->identifier(), value_expr);
      return absl::OkStatus();
    };
    absl::flat_hash_set<const ParametricBinding*> implicit_parametrics;
    absl::flat_hash_map<std::string, int> indices;
    std::vector<const TypeAnnotation*> formal_member_types;
    std::vector<Expr*> actual_member_exprs;
    for (const StructMemberNode* member : struct_def.members()) {
      formal_member_types.push_back(member->type());
    }
    if (instantiator_node.has_value()) {
      for (const auto& [name, expr] :
           (*instantiator_node)->GetOrderedMembers(&struct_def)) {
        actual_member_exprs.push_back(expr);
      }
      CHECK_EQ(actual_member_exprs.size(), formal_member_types.size());
    }
    auto infer_pending_implicit_parametrics = [&]() -> absl::Status {
      VLOG(6) << "Infer implicit parametrics: " << implicit_parametrics.size();
      if (implicit_parametrics.empty()) {
        return absl::OkStatus();
      }
      CHECK(instantiator_node.has_value());
      absl::flat_hash_map<std::string, InterpValue> new_values;
      // Note: setting target_context to null is temporary until we support
      // generic type parametrics for structs.
      XLS_ASSIGN_OR_RETURN(
          new_values,
          InferImplicitParametrics(
              parent_context, /*target_context=*/std::nullopt,
              implicit_parametrics, formal_member_types, actual_member_exprs,
              instance_type_info, instance_type_info,
              /*caller_accept_predicate=*/
              [&](const TypeAnnotation* annotation) {
                // When inferring a parametric using a member of the actual
                // struct, we may have e.g. a member with 2 annotations like
                // `decltype(Foo<N>.x)` and `uN[32]`. The decltype one in this
                // example is not useful for the inference of `N`, and more
                // generally, any decltype-ish annotation that refers back to
                // the struct we are processing is going to be unhelpful, so we
                // weed those out here.
                return !RefersToStruct(parent_context, annotation, struct_def);
              }));
      implicit_parametrics.clear();
      for (const auto& [name, value] : new_values) {
        XLS_RETURN_IF_ERROR(set_value(bindings.at(name), value));
      }
      return absl::OkStatus();
    };
    for (int i = 0; i < struct_def.parametric_bindings().size(); i++) {
      const ParametricBinding* binding = struct_def.parametric_bindings()[i];
      if (i < explicit_parametrics.size()) {
        XLS_RETURN_IF_ERROR(set_value(binding, explicit_parametrics[i]));
      } else if (binding->expr() != nullptr) {
        XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());
        XLS_ASSIGN_OR_RETURN(
            InterpValue value,
            Evaluate(parent_context, instance_type_info,
                     binding->type_annotation(), binding->expr()));
        XLS_RETURN_IF_ERROR(set_value(binding, value));
      } else {
        implicit_parametrics.insert(binding);
      }
    }
    XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());
    std::vector<ExprOrType> resolved_parametrics_vector;
    resolved_parametrics_vector.reserve(
        struct_def.parametric_bindings().size());
    for (const ParametricBinding* binding : struct_def.parametric_bindings()) {
      resolved_parametrics_vector.push_back(
          resolved_parametrics.at(binding->identifier()));
    }
    return CreateStructAnnotation(module_, const_cast<StructDef*>(&struct_def),
                                  resolved_parametrics_vector,
                                  instantiator_node);
  }

  // Determines whether the given type annotation has any reference to the given
  // `struct_def`, taking into consideration the expansions of any variable or
  // indirect type annotations in the annotation tree.
  bool RefersToStruct(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation, const StructDef& struct_def) {
    if (auto* element_annotation =
            dynamic_cast<const ElementTypeAnnotation*>(annotation)) {
      annotation = element_annotation->container_type();
    }
    if (auto* member_annotation =
            dynamic_cast<const MemberTypeAnnotation*>(annotation)) {
      const std::vector<const TypeAnnotation*> annotations =
          ExpandVariables(member_annotation, table_, parametric_context);
      for (const TypeAnnotation* annotation : annotations) {
        std::optional<const StructDefBase*> def =
            GetStructOrProcDef(annotation);
        if (def.has_value() && *def == &struct_def) {
          return true;
        }
      }
      return false;
    }
    std::optional<const StructDefBase*> def = GetStructOrProcDef(annotation);
    return def.has_value() && *def == &struct_def;
  }

  // Converts the type of the given struct `member` into one that has any
  // `NameRef`s to struct parametrics or parametric impl constants replaced with
  // their values, derived from `struct_or_proc_ref.parametrics`. For example,
  // if the member type is `uN[N][C]`, where `N` is a parametric with the value
  // 32, and `C` is a constant that is not a struct or impl member, this would
  // return `uN[32][C]`.
  absl::StatusOr<const TypeAnnotation*> GetParametricFreeStructMemberType(
      std::optional<const ParametricContext*> struct_context,
      const StructOrProcRef& struct_or_proc_ref,
      const TypeAnnotation* member_type) {
    if (!struct_or_proc_ref.def->IsParametric()) {
      return member_type;
    }
    absl::flat_hash_map<const NameDef*, ExprOrType> parametrics_and_constants;
    std::vector<ExprOrType> parametric_vector;
    std::vector<ParametricBinding*> bindings =
        struct_or_proc_ref.def->parametric_bindings();
    CHECK_GE(bindings.size(), struct_or_proc_ref.parametrics.size());
    for (int i = 0; i < bindings.size(); i++) {
      const ParametricBinding* binding = bindings[i];
      ExprOrType value_expr;
      if (i >= struct_or_proc_ref.parametrics.size()) {
        XLS_ASSIGN_OR_RETURN(
            AstNode * clone,
            table_.Clone(binding->expr(),
                         NameRefMapper(parametrics_and_constants)));
        value_expr = dynamic_cast<Expr*>(clone);
      } else {
        value_expr = struct_or_proc_ref.parametrics[i];
      }
      parametrics_and_constants.emplace(binding->name_def(), value_expr);
      parametric_vector.push_back(value_expr);
    }
    // If there is an impl, load the impl constants into the map for erasure as
    // well.
    if (struct_or_proc_ref.def->impl().has_value()) {
      for (const ConstantDef* constant :
           (*struct_or_proc_ref.def->impl())->GetConstants()) {
        XLS_ASSIGN_OR_RETURN(InterpValue value,
                             parametric_context_type_info_.at(*struct_context)
                                 ->GetConstExpr(constant->name_def()));
        Expr* value_expr = module_.Make<Number>(
            constant->span(), value.ToString(/*humanize=*/true),
            NumberKind::kOther, nullptr);
        parametrics_and_constants.emplace(constant->name_def(), value_expr);
      }
    }
    return GetParametricFreeType(member_type, parametrics_and_constants,
                                 struct_context.has_value()
                                     ? (*struct_context)->self_type()
                                     : std::nullopt);
  }

  // Returns `type` with parametrics and parametric constants replaced with
  // `actual_values`. If `real_self_type` is specified, then any references to
  // `Self` in `type` are replaced with `real_self_type` in the returned type.
  absl::StatusOr<const TypeAnnotation*> GetParametricFreeType(
      const TypeAnnotation* type,
      const absl::flat_hash_map<const NameDef*, ExprOrType> actual_values,
      std::optional<const TypeAnnotation*> real_self_type = std::nullopt) {
    CloneReplacer replacer = NameRefMapper(actual_values);
    if (real_self_type.has_value()) {
      replacer = ChainCloneReplacers(
          std::move(replacer),
          [&](const AstNode* node) -> absl::StatusOr<std::optional<AstNode*>> {
            if (const auto* self =
                    dynamic_cast<const SelfTypeAnnotation*>(node)) {
              return const_cast<TypeAnnotation*>(*real_self_type);
            }
            return std::nullopt;
          });
    }
    XLS_ASSIGN_OR_RETURN(AstNode * clone,
                         table_.Clone(type, std::move(replacer)));
    const auto* result = dynamic_cast<const TypeAnnotation*>(clone);
    CHECK(result != nullptr);
    return result;
  }

  // Wraps `BitCountMismatchErrorStatus` with resolution of parametrics, so that
  // a nominal type like `uN[N]` will not appear with the variable in the error
  // message.
  absl::Status BitCountMismatchError(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation1,
      const TypeAnnotation* annotation2) override {
    if (!parametric_context.has_value()) {
      return BitCountMismatchErrorStatus(annotation1, annotation2, file_table_);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type1,
                         Concretize(annotation1, parametric_context));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type2,
                         Concretize(annotation2, parametric_context));
    return BitCountMismatchErrorStatus(*type1, *type2, annotation1->span(),
                                       annotation2->span(), file_table_);
  }

  // Wraps `SignednessMismatchErrorStatus` with resolution of parametrics, so
  // that a nominal type like `uN[N]` will not appear with the variable in the
  // error message.
  absl::Status SignednessMismatchError(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation1,
      const TypeAnnotation* annotation2) override {
    if (!parametric_context.has_value()) {
      return SignednessMismatchErrorStatus(annotation1, annotation2,
                                           file_table_);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type1,
                         Concretize(annotation1, parametric_context));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type2,
                         Concretize(annotation2, parametric_context));
    return SignednessMismatchErrorStatus(*type1, *type2, annotation1->span(),
                                         annotation2->span(), file_table_);
  }

  // Wraps `TypeMismatchErrorStatus` with resolution of parametrics, so that a
  // nominal type like `uN[N]` will not appear with the variable in the error
  // message.
  absl::Status TypeMismatchError(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation1,
      const TypeAnnotation* annotation2) override {
    if (!parametric_context.has_value()) {
      return TypeMismatchErrorStatus(annotation1, annotation2, file_table_);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type1,
                         Concretize(annotation1, parametric_context));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type2,
                         Concretize(annotation2, parametric_context));
    return TypeMismatchErrorStatus(*type1, *type2, annotation1->span(),
                                   annotation2->span(), file_table_);
  }

  // Returns true if `annotation` contains any `NameRef` whose type info has not
  // (yet) been generated. The effective `TypeInfo` is either `default_ti`; or,
  // for invocation-scoped annotations, the `TypeInfo` for the relevant
  // parametric invocation.
  bool HasAnyReferencesWithMissingTypeInfo(TypeInfo* ti,
                                           const TypeAnnotation* annotation) {
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

  InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  TypeInfo* const base_type_info_;
  const FileTable& file_table_;
  TypeSystemTracer& tracer_;
  std::unique_ptr<TypeAnnotationResolver> resolver_;
  absl::flat_hash_map<const ParametricContext*, TypeInfo*>
      parametric_context_type_info_;
  absl::flat_hash_map<const ParametricContext*, ParametricEnv>
      converted_parametric_envs_;
  absl::flat_hash_map<const ParametricContext*,
                      absl::flat_hash_map<const NameDef*, ExprOrType>>
      parametric_value_exprs_;
  std::optional<std::unique_ptr<InferenceTableConverter>> builtins_converter_;
  absl::flat_hash_map<std::optional<const ParametricContext*>,
                      absl::flat_hash_set<const AstNode*>>
      converted_subtrees_;
};

}  // namespace

absl::StatusOr<TypeInfo*> InferenceTableToTypeInfo(
    InferenceTable& table, Module& module, ImportData& import_data,
    WarningCollector& warning_collector, const FileTable& file_table,
    std::unique_ptr<Module> builtins_module) {
  VLOG(1) << "InferenceTableToTypeInfo: module " << &module;
  VLOG(5) << "Inference table before conversion:";
  VLOG(5) << table.ToString();

  XLS_ASSIGN_OR_RETURN(TypeInfo * base_type_info,
                       import_data.type_info_owner().New(&module));
  XLS_ASSIGN_OR_RETURN(
      TypeInfo * builtins_type_info,
      import_data.type_info_owner().New(std::move(builtins_module.get())));
  std::unique_ptr<TypeSystemTracer> builtins_tracer =
      TypeSystemTracer::Create();
  std::optional<std::unique_ptr<InferenceTableConverter>> builtins_converter =
      std::make_unique<InferenceTableConverter>(
          table, *builtins_module, import_data, warning_collector,
          builtins_type_info, file_table, *builtins_tracer,
          /*builtins_converter=*/std::nullopt);

  std::unique_ptr<TypeSystemTracer> module_tracer = TypeSystemTracer::Create();
  InferenceTableConverter converter(
      table, module, import_data, warning_collector, base_type_info, file_table,
      *module_tracer, std::move(builtins_converter));
  absl::Status status =
      converter.ConvertSubtree(&module, /*function=*/std::nullopt,
                               /*parametric_context=*/std::nullopt);

  VLOG(5) << "Inference table after conversion:";
  VLOG(5) << table.ToString();

  VLOG(5) << "Builtins module traces after conversion:";
  VLOG(5) << builtins_tracer->ConvertTracesToString();

  VLOG(5) << "User module traces after conversion:";
  VLOG(5) << module_tracer->ConvertTracesToString();

  if (!status.ok()) {
    return status;
  }
  return converter.GetBaseTypeInfo();
}

}  // namespace xls::dslx
