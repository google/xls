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
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_zero_value.h"
#include "xls/dslx/type_system_v2/expand_variables.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/solve_for_parametrics.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/validate_concrete_type.h"
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

const TypeAnnotation* SignednessAndSizeToAnnotation(
    Module& module, const SignednessAndSize& signedness_and_size,
    const Span& span) {
  return CreateUnOrSnAnnotation(module, span, signedness_and_size.is_signed,
                                signedness_and_size.size);
}

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
class InferenceTableConverter {
 public:
  InferenceTableConverter(
      InferenceTable& table, Module& module, ImportData& import_data,
      WarningCollector& warning_collector, TypeInfo* base_type_info,
      const FileTable& file_table,
      std::optional<std::unique_ptr<InferenceTableConverter>>
          builtins_converter)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        base_type_info_(base_type_info),
        file_table_(file_table),
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
                           UnifyTypeAnnotationsForNode(
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
        const ParametricContext* parametric_context,
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
    parametric_context_type_info_.emplace(parametric_context,
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
      if (std::holds_alternative<Expr*>(explicit_parametric)) {
        XLS_RETURN_IF_ERROR(ConvertSubtree(std::get<Expr*>(explicit_parametric),
                                           caller, caller_context));
      }
    }

    // Convert the default expressions in the context of this invocation.
    for (const ParametricBinding* binding : function->parametric_bindings()) {
      if (binding->expr() != nullptr) {
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(binding->expr(), function, parametric_context));
      }
    }

    // Figure out any implicit parametrics and generate the `ParametricEnv`.
    XLS_RETURN_IF_ERROR(GenerateParametricFunctionEnv(
        function_and_target_object.target_struct_context, parametric_context));
    XLS_RETURN_IF_ERROR(AddInvocationTypeInfo(parametric_context));

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
                              parametric_value_exprs_.at(parametric_context),
                              parametric_context->self_type()));
    XLS_RETURN_IF_ERROR(
        ConvertSubtree(parametric_free_type, caller, caller_context));

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
    XLS_RETURN_IF_ERROR(ConvertSubtree(function, function, parametric_context));
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
    VLOG(5) << "GenerateTypeInfo for node: " << node->ToString()
            << " with owner: " << node->owner()->name()
            << " for module: " << module_.name();
    if (pre_unified_type.has_value()) {
      VLOG(5) << "Using pre-unified type: " << (*pre_unified_type)->ToString();
    }
    TypeInfo* ti = GetTypeInfo(parametric_context);
    std::optional<const TypeAnnotation*> annotation = pre_unified_type;
    if (!annotation.has_value()) {
      XLS_ASSIGN_OR_RETURN(annotation, UnifyTypeAnnotationsForNode(
                                           parametric_context, node,
                                           type_annotation_accept_predicate));
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

    XLS_RETURN_IF_ERROR(
        ValidateConcreteType(node, type->get(), *ti, file_table_));
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

  absl::StatusOr<std::optional<const TypeAnnotation*>>
  UnifyTypeAnnotationsForNode(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate = std::nullopt) {
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
          !VariableHasAnyExplicitTypeAnnotations(parametric_context,
                                                 *type_variable)) {
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
      return UnifyTypeAnnotations(parametric_context, *type_variable,
                                  *node_span, accept_predicate);
    } else {
      return table_.GetTypeAnnotation(node);
    }
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
      CHECK(std::holds_alternative<Expr*>(parametric));
      VLOG(6) << "Actual parametric: " << binding->identifier()
              << " expr: " << std::get<Expr*>(parametric)->ToString();
      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           Evaluate(ParametricContextScopedExpr(
                               parent_context, binding->type_annotation(),
                               std::get<Expr*>(parametric))));
      VLOG(6) << "Actual parametric: " << binding->identifier()
              << " value: " << value.ToString();
      values.emplace(binding->identifier(), value);
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
    VLOG(5) << "Concretize: " << annotation->ToString()
            << " in context invocation: " << ToString(parametric_context);
    VLOG(5) << "Effective context: " << ToString(parametric_context);

    XLS_ASSIGN_OR_RETURN(annotation, ResolveVariableTypeAnnotations(
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
      const ParametricContextScopedExpr& scoped_expr) {
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
          UnifyTypeAnnotationsForNode(caller_context, *target_object));
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
      const ParametricContext* context) {
    absl::flat_hash_map<std::string, InterpValue> values;
    absl::flat_hash_set<const ParametricBinding*> implicit_parametrics;
    absl::flat_hash_map<std::string, const ParametricBinding*> bindings;
    auto infer_pending_implicit_parametrics = [&]() -> absl::Status {
      if (implicit_parametrics.empty()) {
        return absl::OkStatus();
      }
      absl::flat_hash_map<std::string, InterpValue> new_values;
      XLS_ASSIGN_OR_RETURN(new_values, InferImplicitFunctionParametrics(
                                           context, implicit_parametrics));
      implicit_parametrics.clear();
      values.merge(std::move(new_values));
      return absl::OkStatus();
    };
    for (const ParametricBinding* binding : context->parametric_bindings()) {
      bindings.emplace(binding->identifier(), binding);
      std::optional<ParametricContextScopedExpr> expr =
          table_.GetParametricValue(*binding->name_def(), *context);
      if (expr.has_value()) {
        // The expr may be a default expr which may use the inferred values of
        // any parametrics preceding it, so let's resolve any pending implicit
        // ones now.
        XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());
        // Now evaluate the expr.
        XLS_ASSIGN_OR_RETURN(InterpValue value, Evaluate(*expr));
        parametric_context_type_info_.at(context)->NoteConstExpr(
            binding->name_def(), value);
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
      XLS_ASSIGN_OR_RETURN(Number * value_expr,
                           MakeTypeCheckedNumber(binding->span(), value,
                                                 binding->type_annotation()));
      actual_parametrics.emplace(binding->name_def(), value_expr);
    }
    if (callee_struct_context.has_value()) {
      for (const auto& [name_def, expr] :
           parametric_value_exprs_.at(*callee_struct_context)) {
        actual_parametrics.emplace(name_def, expr);
      }
    }
    parametric_value_exprs_.emplace(context, std::move(actual_parametrics));
    ParametricEnv env(std::move(values));
    converted_parametric_envs_.emplace(context, env);
    return absl::OkStatus();
  }

  // Attempts to infer the values of the specified implicit parametrics in an
  // invocation, using the types of the regular arguments being passed. If not
  // all of `implicit_parametrics` can be determined, this function returns an
  // error.
  absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
  InferImplicitFunctionParametrics(
      const ParametricContext* context,
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics) {
    VLOG(5) << "Inferring " << implicit_parametrics.size()
            << " implicit parametrics for invocation: " << ToString(context);
    const auto& context_data =
        std::get<ParametricInvocationDetails>(context->details());
    const auto* invocation = dynamic_cast<const Invocation*>(context->node());
    CHECK_NE(invocation, nullptr);
    const absl::Span<Param* const> formal_args = context_data.callee->params();
    const absl::Span<Expr* const> actual_args = invocation->args();
    TypeInfo* ti = parametric_context_type_info_.at(context);
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
    if (context->parent_context().has_value()) {
      actual_arg_ti =
          parametric_context_type_info_.at(*context->parent_context());
    }
    return InferImplicitParametrics(
        context->parent_context(), implicit_parametrics, formal_types,
        actual_args, ti, actual_arg_ti,
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
                                context->parent_context(),
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
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics,
      absl::Span<const TypeAnnotation* const> formal_types,
      absl::Span<Expr* const> actual_args, TypeInfo* output_ti,
      TypeInfo* actual_arg_ti,
      absl::FunctionRef<bool(const TypeAnnotation*)> caller_accept_predicate =
          [](const TypeAnnotation*) { return true; },
      absl::FunctionRef<absl::Status(const Expr*)> pre_use_actual_arg =
          [](const Expr*) { return absl::OkStatus(); }) {
    absl::flat_hash_map<std::string, InterpValue> values;
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
               !HasAnyReferencesWithMissingTypeInfo(actual_arg_ti, annotation);
      };
      XLS_RETURN_IF_ERROR(ResolveVariableTypeAnnotations(
          actual_arg_context, actual_arg_annotations, accept_predicate));
      if (actual_arg_annotations.empty()) {
        VLOG(6) << "The actual argument type variable: "
                << (*actual_arg_type_var)->ToString()
                << " has no independent type annotations.";
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* actual_arg_type,
          UnifyTypeAnnotations(actual_arg_context, actual_arg_annotations,
                               actual_args[i]->span(),
                               caller_accept_predicate));
      absl::flat_hash_map<const ParametricBinding*, InterpValue> resolved;
      VLOG(5) << "Infer using actual type: " << actual_arg_type->ToString()
              << " with effective context: " << ToString(actual_arg_context);
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
      std::variant<bool, const Expr*> value_or_expr) {
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
      std::variant<int64_t, const Expr*> value_or_expr) {
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

  // Comes up with one type annotation reconciling the information in any
  // type annotations that have been associated with the given type variable. If
  // the information has unreconcilable conflicts, returns an error. The given
  // `parametric_context` argument is used as a context for the evaluation of
  // any expressions inside the type annotations. If an `accept_predicate` is
  // specified, then annotations not accepted by the predicate are ignored.
  absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* type_variable, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "Unifying type annotations for variable "
            << type_variable->ToString();
    XLS_ASSIGN_OR_RETURN(std::vector<const TypeAnnotation*> annotations,
                         table_.GetTypeAnnotationsForTypeVariable(
                             parametric_context, type_variable));
    if (accept_predicate.has_value()) {
      FilterAnnotations(annotations, *accept_predicate);
    }
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* result,
                         UnifyTypeAnnotations(parametric_context, annotations,
                                              span, accept_predicate));
    VLOG(6) << "Unified type for variable " << type_variable->ToString() << ": "
            << result->ToString();
    return result;
  }

  // Overload that unifies specific type annotations.
  absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TypeAnnotation*> annotations, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    XLS_RETURN_IF_ERROR(ResolveVariableTypeAnnotations(
        parametric_context, annotations, accept_predicate));
    // Remove all singular `Any` annotations, and if that's all we had, the
    // result is one singular `Any`.
    FilterAnnotations(annotations, [&](const TypeAnnotation* annotation) {
      const auto* any_annotation =
          dynamic_cast<const AnyTypeAnnotation*>(annotation);
      return any_annotation == nullptr || any_annotation->multiple();
    });
    if (annotations.empty()) {
      return module_.Make<AnyTypeAnnotation>();
    }
    // Remove all multiple `Any` annotations, and if that's all we had, the
    // result is one multiple `Any`.
    FilterAnnotations(annotations, [&](const TypeAnnotation* annotation) {
      return dynamic_cast<const AnyTypeAnnotation*>(annotation) == nullptr;
    });
    if (annotations.empty()) {
      return module_.Make<AnyTypeAnnotation>(/*multiple=*/true);
    }
    if (annotations.size() == 1) {
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                           GetStructOrProcRef(annotations[0], file_table_));
      if (!struct_or_proc_ref.has_value()) {
        // This is here mainly for preservation of shorthand annotations
        // appearing in the source code, in case they get put in subsequent
        // error messages. General unification would normalize the format.
        return annotations[0];
      }
    }
    if (const auto* first_tuple_annotation =
            dynamic_cast<const TupleTypeAnnotation*>(annotations[0])) {
      std::vector<const TupleTypeAnnotation*> tuple_annotations;
      tuple_annotations.reserve(annotations.size());
      for (const TypeAnnotation* annotation : annotations) {
        const auto* tuple_annotation =
            dynamic_cast<const TupleTypeAnnotation*>(annotation);
        if (tuple_annotation == nullptr) {
          return TypeMismatchErrorWithParametricResolution(
              parametric_context, annotations[0], annotation);
        }
        tuple_annotations.push_back(tuple_annotation);
      }
      return UnifyTupleTypeAnnotations(parametric_context, tuple_annotations,
                                       span, accept_predicate);
    }
    if (const auto* first_array_annotation =
            CastToNonBitsArrayTypeAnnotation(annotations[0])) {
      std::vector<const ArrayTypeAnnotation*> array_annotations;
      for (int i = 0; i < annotations.size(); i++) {
        const auto* array_annotation =
            CastToNonBitsArrayTypeAnnotation(annotations[i]);
        if (array_annotation == nullptr) {
          return TypeMismatchErrorWithParametricResolution(
              parametric_context, annotations[0], annotations[i]);
        }
        array_annotations.push_back(array_annotation);
      }
      return UnifyArrayTypeAnnotations(parametric_context, array_annotations,
                                       span, accept_predicate);
    }
    if (const auto* first_function_annotation =
            dynamic_cast<const FunctionTypeAnnotation*>(annotations[0])) {
      std::vector<const FunctionTypeAnnotation*> function_annotations;
      function_annotations.reserve(annotations.size());
      for (int i = 0; i < annotations.size(); i++) {
        const auto* function_annotation =
            dynamic_cast<const FunctionTypeAnnotation*>(annotations[i]);
        if (function_annotation == nullptr) {
          return TypeMismatchErrorWithParametricResolution(
              parametric_context, annotations[0], annotations[i]);
        }
        VLOG(5) << "Annotation " << i << ": "
                << function_annotation->ToString();
        function_annotations.push_back(function_annotation);
      }
      return UnifyFunctionTypeAnnotations(
          parametric_context, function_annotations, span, accept_predicate);
    }
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> first_struct_or_proc,
                         GetStructOrProcRef(annotations[0], file_table_));
    if (first_struct_or_proc.has_value()) {
      const auto* struct_def =
          dynamic_cast<const StructDef*>(first_struct_or_proc->def);
      CHECK(struct_def != nullptr);
      std::vector<const TypeAnnotation*> annotations_to_unify;
      for (const TypeAnnotation* annotation : annotations) {
        XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> next_struct_or_proc,
                             GetStructOrProcRef(annotation, file_table_));
        if (!next_struct_or_proc.has_value() ||
            next_struct_or_proc->def != struct_def) {
          return TypeMismatchErrorWithParametricResolution(
              parametric_context, annotations[0], annotation);
        }
        if (struct_def->IsParametric()) {
          annotations_to_unify.push_back(annotation);
        }
      }
      // A non-parametric struct is trivially unifiable, because nothing can
      // validly vary between the annotations. A parametric struct needs to have
      // its parameters unified.
      return annotations_to_unify.empty()
                 ? annotations[0]
                 : UnifyParametricStructAnnotations(
                       parametric_context, *struct_def, annotations_to_unify);
    }
    std::optional<SignednessAndSize> unified_signedness_and_bit_count;
    for (int i = 0; i < annotations.size(); ++i) {
      const TypeAnnotation* current_annotation = annotations[i];
      VLOG(6) << "Annotation " << i << ": " << current_annotation->ToString();
      absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
          GetSignednessAndBitCount(current_annotation);
      bool current_annotation_is_auto =
          table_.IsAutoLiteral(current_annotation);
      if (!signedness_and_bit_count.ok()) {
        return TypeMismatchErrorWithParametricResolution(
            parametric_context, current_annotation, annotations[0]);
      }
      XLS_ASSIGN_OR_RETURN(
          bool current_annotation_signedness,
          EvaluateBoolOrExpr(parametric_context,
                             signedness_and_bit_count->signedness));
      XLS_ASSIGN_OR_RETURN(
          int64_t current_annotation_raw_bit_count,
          EvaluateU32OrExpr(parametric_context,
                            signedness_and_bit_count->bit_count));
      SignednessAndSize current_annotation_signedness_and_bit_count{
          .is_auto = current_annotation_is_auto,
          .is_signed = current_annotation_signedness,
          .size = current_annotation_raw_bit_count};

      XLS_ASSIGN_OR_RETURN(
          unified_signedness_and_bit_count,
          UnifySignednessAndSize(parametric_context,
                                 unified_signedness_and_bit_count,
                                 current_annotation_signedness_and_bit_count,
                                 annotations[0], current_annotation));
      VLOG(6) << "Unified type so far has signedness: "
              << unified_signedness_and_bit_count->is_signed
              << " and bit count: " << unified_signedness_and_bit_count->size;
    }
    const TypeAnnotation* result = SignednessAndSizeToAnnotation(
        module_, *unified_signedness_and_bit_count, span);
    // An annotation we fabricate as a unification of a bunch of auto
    // annotations, is also considered an auto annotation itself.
    if (unified_signedness_and_bit_count->is_auto) {
      table_.MarkAsAutoLiteral(result);
    }
    return result;
  }

  // Unifies multiple annotations for a tuple. This function assumes the
  // passed-in array is nonempty. It attempts to expand any `AnyTypeAnnotations`
  // with `multiple=true` to match the size of the first annotation.. Unifying a
  // tuple type amounts to unifying the annotations for each member.
  absl::StatusOr<const TupleTypeAnnotation*> UnifyTupleTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TupleTypeAnnotation*> annotations, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    const int member_count = annotations[0]->members().size();
    std::vector<const TupleTypeAnnotation*> expanded_annotations;

    for (const TupleTypeAnnotation* tuple_annotation : annotations) {
      if (tuple_annotation->members().size() != member_count) {
        // If the sizes don't match, the annotation must contain an
        // `AnyAnnotation` with `multiple = true`. Attempt to expand the
        // multiple any to match the size.
        std::vector<TypeAnnotation*> expanded_members;
        expanded_members.reserve(member_count);
        int delta = member_count - tuple_annotation->members().size();
        for (auto* member : tuple_annotation->members()) {
          const auto* any_annotation = dynamic_cast<AnyTypeAnnotation*>(member);
          if (any_annotation != nullptr && any_annotation->multiple()) {
            for (int i = 0; i < delta + 1; i++) {
              expanded_members.push_back(module_.Make<AnyTypeAnnotation>());
            }
          } else {
            expanded_members.push_back(member);
          }
        }
        tuple_annotation = module_.Make<TupleTypeAnnotation>(
            tuple_annotation->span(), expanded_members);
      }
      CHECK_EQ(tuple_annotation->members().size(), member_count);
      expanded_annotations.push_back(tuple_annotation);
    }

    std::vector<TypeAnnotation*> unified_member_annotations(member_count);
    for (int i = 0; i < member_count; i++) {
      std::vector<const TypeAnnotation*> annotations_for_member;
      annotations_for_member.reserve(annotations.size());
      for (const TupleTypeAnnotation* annotation : expanded_annotations) {
        annotations_for_member.push_back(annotation->members()[i]);
      }
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* unified_member_annotation,
          UnifyTypeAnnotations(parametric_context, annotations_for_member, span,
                               accept_predicate));
      unified_member_annotations[i] =
          const_cast<TypeAnnotation*>(unified_member_annotation);
    }
    return module_.Make<TupleTypeAnnotation>(span, unified_member_annotations);
  }

  // Unifies multiple annotations for an array. This function assumes the
  // passed-in array is nonempty. Unifying an array type amounts to unifying the
  // element types and dims.
  absl::StatusOr<const ArrayTypeAnnotation*> UnifyArrayTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const ArrayTypeAnnotation*> annotations, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    std::vector<const TypeAnnotation*> element_type_annotations;
    std::optional<SignednessAndSize> unified_dim;
    for (int i = 0; i < annotations.size(); i++) {
      const ArrayTypeAnnotation* annotation = annotations[i];
      element_type_annotations.push_back(annotation->element_type());

      XLS_ASSIGN_OR_RETURN(
          int64_t current_dim,
          EvaluateU32OrExpr(parametric_context, annotation->dim()));
      // This flag indicates we are unifying one min dim with one explicit dim,
      // which warrants a possible different error message than other scenarios.
      const bool is_min_vs_explicit =
          unified_dim.has_value() &&
          (unified_dim->is_auto ^ annotation->dim_is_min());
      absl::StatusOr<SignednessAndSize> new_unified_dim =
          UnifySignednessAndSize(
              parametric_context, unified_dim,
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
            parametric_context, annotations[i], annotations[i - 1]);
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
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* unified_element_type,
        UnifyTypeAnnotations(parametric_context, element_type_annotations, span,
                             accept_predicate));
    XLS_ASSIGN_OR_RETURN(
        Number * size_expr,
        MakeTypeCheckedNumber(
            annotations[0]->span(), unified_dim->size,
            CreateU32Annotation(module_, annotations[0]->span())));
    return module_.Make<ArrayTypeAnnotation>(
        span, const_cast<TypeAnnotation*>(unified_element_type), size_expr);
  }

  // Unifies multiple annotations for a function type. This assumes the
  // passed-in array is nonempty. Unifying a function type amounts to unifying
  // the return type and the argument types.
  absl::StatusOr<const FunctionTypeAnnotation*> UnifyFunctionTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const FunctionTypeAnnotation*> annotations, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "UnifyFunctionTypeAnnotations: " << annotations.size();
    // Plausible return types for the function.
    std::vector<const TypeAnnotation*> return_types;
    return_types.reserve(annotations.size());
    // Plausible types for each function argument (rows are argument indices).
    std::vector<std::vector<const TypeAnnotation*>> param_types;
    param_types.resize(annotations[0]->param_types().size());
    for (const FunctionTypeAnnotation* annotation : annotations) {
      VLOG(6) << "Return type to unify: "
              << annotation->return_type()->ToString();
      return_types.push_back(annotation->return_type());
      if (annotation->param_types().size() !=
          annotations[0]->param_types().size()) {
        return ArgCountMismatchErrorStatus(
            span,
            absl::Substitute("Expected $0 argument(s) but got $1.",
                             annotation->param_types().size(),
                             annotations[0]->param_types().size()),
            file_table_);
      }
      for (int i = 0; i < annotation->param_types().size(); i++) {
        const TypeAnnotation* param_type = annotation->param_types()[i];
        VLOG(6) << "Param type " << i
                << " to unify: " << param_type->ToString();
        param_types[i].push_back(param_type);
      }
    }

    // Unify the return type and argument types.
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* unified_return_type,
                         UnifyTypeAnnotations(parametric_context, return_types,
                                              span, accept_predicate));
    std::vector<TypeAnnotation*> unified_param_types;
    unified_param_types.reserve(param_types.size());
    for (const std::vector<const TypeAnnotation*>& argument : param_types) {
      XLS_ASSIGN_OR_RETURN(const TypeAnnotation* unified_param_type,
                           UnifyTypeAnnotations(parametric_context, argument,
                                                span, accept_predicate));
      unified_param_types.push_back(
          const_cast<TypeAnnotation*>(unified_param_type));
    }
    return module_.Make<FunctionTypeAnnotation>(
        unified_param_types, const_cast<TypeAnnotation*>(unified_return_type));
  }

  // Unifies multiple annotations for a parametric struct, and produces an
  // annotation with all of the parametric bindings of the struct having
  // explicit values. Values that can't be sourced from the annotations are
  // inferred or defaulted. If this can't be done, or there is a disagreement on
  // explicit values among the annotations, then this returns an error. Note
  // that the explicit parametric values returned are not necessarily
  // concretized to literals, but they are exprs without any references to
  // other parametrics of the struct. Therefore, `Concretize()` should be able
  // to operate on the returned annotation.
  absl::StatusOr<const TypeAnnotation*> UnifyParametricStructAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const StructDef& struct_def,
      std::vector<const TypeAnnotation*> annotations) {
    VLOG(6) << "Unifying parametric struct annotations; struct def: "
            << struct_def.identifier();
    std::vector<InterpValue> explicit_parametrics;
    std::optional<const StructInstance*> instantiator;

    // Go through the annotations, and check that they have no disagreement in
    // their explicit parametric values. For example, one annotation may be
    // `SomeStruct<32>` and one may be `SomeStruct<N>` where `N` is a parametric
    // of the enclosing function. We are in a position now to decide if `N` is
    // 32 or not.
    for (const TypeAnnotation* annotation : annotations) {
      VLOG(6) << "Annotation: " << annotation->ToString();
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                           GetStructOrProcRef(annotation, file_table_));
      CHECK(struct_or_proc_ref.has_value());
      if (struct_or_proc_ref->instantiator.has_value()) {
        instantiator = struct_or_proc_ref->instantiator;
      }
      for (int i = 0; i < struct_or_proc_ref->parametrics.size(); i++) {
        ExprOrType parametric = struct_or_proc_ref->parametrics[i];
        const ParametricBinding* binding = struct_def.parametric_bindings()[i];
        CHECK(std::holds_alternative<Expr*>(parametric));
        XLS_ASSIGN_OR_RETURN(InterpValue value,
                             Evaluate(ParametricContextScopedExpr(
                                 parametric_context, binding->type_annotation(),
                                 std::get<Expr*>(parametric))));
        if (i == explicit_parametrics.size()) {
          explicit_parametrics.push_back(value);
        } else if (value != explicit_parametrics[i]) {
          return absl::InvalidArgumentError(absl::Substitute(
              "Value mismatch for parametric `$0` of struct `$1`: $2 vs. $3",
              binding->identifier(), struct_def.identifier(), value.ToString(),
              explicit_parametrics[i].ToString()));
        }
      }
    }

    // Now we can forget about the multiple annotations. We have the agreeing,
    // computed explicit parametrics in `explicit_parametrics`. The goal now is
    // to come up with a complete parametric value `Expr` vector, which has a
    // value for every formal binding, by inferring or defaulting whichever ones
    // are not explicit. The algorithm is the same as for parametric function
    // invocations, and the differences are in the logistics. We build this as a
    // map, `resolved_parametrics`, and convert it to a vector at the end.
    XLS_ASSIGN_OR_RETURN(TypeInfo * instance_type_info,
                         import_data_.type_info_owner().New(
                             &module_, parametric_context.has_value()
                                           ? parametric_context_type_info_.at(
                                                 *parametric_context)
                                           : base_type_info_));
    absl::flat_hash_map<std::string, const ParametricBinding*> bindings;
    absl::flat_hash_map<std::string, ExprOrType> resolved_parametrics;
    auto set_value = [&](const ParametricBinding* binding,
                         InterpValue value) -> absl::Status {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> binding_type,
          Concretize(binding->type_annotation(), parametric_context));
      instance_type_info->SetItem(binding->name_def(), *binding_type);
      instance_type_info->NoteConstExpr(binding->name_def(), value);
      XLS_ASSIGN_OR_RETURN(Number * value_expr,
                           MakeTypeCheckedNumber(binding->span(), value,
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
    if (instantiator.has_value()) {
      for (const auto& [name, expr] :
           (*instantiator)->GetOrderedMembers(&struct_def)) {
        actual_member_exprs.push_back(expr);
      }
      CHECK_EQ(actual_member_exprs.size(), formal_member_types.size());
    }
    auto infer_pending_implicit_parametrics = [&]() -> absl::Status {
      VLOG(6) << "Infer implicit parametrics: " << implicit_parametrics.size();
      if (implicit_parametrics.empty()) {
        return absl::OkStatus();
      }
      CHECK(instantiator.has_value());
      absl::flat_hash_map<std::string, InterpValue> new_values;
      XLS_ASSIGN_OR_RETURN(
          new_values,
          InferImplicitParametrics(
              parametric_context, implicit_parametrics, formal_member_types,
              actual_member_exprs, instance_type_info, instance_type_info,
              /*caller_accept_predicate=*/
              [&](const TypeAnnotation* annotation) {
                // When inferring a parametric using a member of the actual
                // struct, we may have e.g. a member with 2 annotations like
                // `decltype(Foo<N>.x)` and `uN[32]`. The decltype one in this
                // example is not useful for the inference of `N`, and more
                // generally, any decltype-ish annotation that refers back to
                // the struct we are processing is going to be unhelpful, so we
                // weed those out here.
                return !RefersToStruct(parametric_context, annotation,
                                       struct_def);
              }));
      implicit_parametrics.clear();
      for (const auto& [name, value] : new_values) {
        XLS_RETURN_IF_ERROR(set_value(bindings.at(name), value));
      }
      return absl::OkStatus();
    };
    for (int i = 0; i < struct_def.parametric_bindings().size(); i++) {
      const ParametricBinding* binding = struct_def.parametric_bindings()[i];
      bindings.emplace(binding->identifier(), binding);
      if (i < explicit_parametrics.size()) {
        XLS_RETURN_IF_ERROR(set_value(binding, explicit_parametrics[i]));
      } else if (binding->expr() != nullptr) {
        XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());
        XLS_ASSIGN_OR_RETURN(
            InterpValue value,
            Evaluate(parametric_context, instance_type_info,
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
                                  resolved_parametrics_vector, instantiator);
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

  // Returns `annotation` with any indirect annotations resolved into direct
  // annotations. An indirect annotation is an internally-generated one that
  // depends on the resolved type of another entity. This may be a
  // `TypeVariableTypeAnnotation`, a `MemberTypeAnnotation`, or an
  // `ElementTypeAnnotation`. The original `annotation` is returned if there is
  // nothing to resolve, preserving the ability to identify it as an auto
  // literal annotation.
  //
  // If `accept_predicate` is specified, then it is used to filter annotations
  // for entities referred to by `annotation`. For example, the caller may be
  // trying to solve for the value of an implicit parametric `N` by expanding a
  // `TypeVariableTypeAnnotation` that has 2 associated annotations in the
  // inference table: `u32` and `uN[N]`. In that case, the caller does not want
  // attempted resolution of the `uN[N]` annotation by this function. The
  // predicate is not applied to the input `annotation` itself.
  absl::StatusOr<const TypeAnnotation*> ResolveVariableTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "Resolving variables in: " << annotation->ToString()
            << " in context: " << ToString(parametric_context);
    bool replaced_anything = false;
    XLS_ASSIGN_OR_RETURN(
        AstNode * clone,
        table_.Clone(
            annotation,
            [&](const AstNode* node)
                -> absl::StatusOr<std::optional<AstNode*>> {
              if (const auto* variable_type_annotation =
                      dynamic_cast<const TypeVariableTypeAnnotation*>(node)) {
                XLS_ASSIGN_OR_RETURN(
                    const TypeAnnotation* unified,
                    UnifyTypeAnnotations(
                        parametric_context,
                        variable_type_annotation->type_variable(),
                        annotation->span(), accept_predicate));
                replaced_anything = true;
                return const_cast<TypeAnnotation*>(unified);
              }
              if (const auto* member_type =
                      dynamic_cast<const MemberTypeAnnotation*>(node)) {
                replaced_anything = true;
                XLS_ASSIGN_OR_RETURN(
                    const TypeAnnotation* result,
                    ExpandMemberType(parametric_context, member_type,
                                     accept_predicate));
                VLOG(5) << "Member type expansion for: "
                        << member_type->member_name()
                        << " yielded: " << result->ToString();
                return const_cast<TypeAnnotation*>(result);
              }
              if (const auto* element_type =
                      dynamic_cast<const ElementTypeAnnotation*>(node)) {
                replaced_anything = true;
                XLS_ASSIGN_OR_RETURN(
                    const TypeAnnotation* result,
                    ExpandElementType(parametric_context, element_type,
                                      accept_predicate));
                return const_cast<TypeAnnotation*>(result);
              }
              if (const auto* return_type =
                      dynamic_cast<const ReturnTypeAnnotation*>(node)) {
                replaced_anything = true;
                XLS_ASSIGN_OR_RETURN(
                    const TypeAnnotation* result,
                    ExpandReturnType(parametric_context, return_type,
                                     accept_predicate));
                return const_cast<TypeAnnotation*>(result);
              }
              if (const auto* param_type =
                      dynamic_cast<const ParamTypeAnnotation*>(node)) {
                replaced_anything = true;
                if (accept_predicate.has_value() &&
                    !(*accept_predicate)(param_type)) {
                  return module_.Make<AnyTypeAnnotation>();
                }
                XLS_ASSIGN_OR_RETURN(
                    const TypeAnnotation* result,
                    ExpandParamType(parametric_context, param_type,
                                    accept_predicate));
                return const_cast<TypeAnnotation*>(result);
              }
              if (const auto* self_type =
                      dynamic_cast<const SelfTypeAnnotation*>(node)) {
                std::optional<const TypeAnnotation*> expanded =
                    ExpandSelfType(parametric_context, self_type);
                replaced_anything = true;
                CHECK(expanded.has_value());
                return const_cast<TypeAnnotation*>(*expanded);
              }
              return std::nullopt;
            }));

    // If the result is a `TypeRefTypeAnnotation`, check if it resolves to a
    // different type in the inference table.
    XLS_ASSIGN_OR_RETURN(
        clone,
        table_.Clone(
            clone,
            [&](const AstNode* node)
                -> absl::StatusOr<std::optional<AstNode*>> {
              std::optional<const TypeAnnotation*> latest =
                  dynamic_cast<const TypeAnnotation*>(node);
              while (latest.has_value() &&
                     dynamic_cast<const TypeRefTypeAnnotation*>(*latest)) {
                const auto* type_ref =
                    dynamic_cast<const TypeRefTypeAnnotation*>(*latest);
                latest =
                    table_.GetTypeAnnotation(ToAstNode(TypeDefinitionGetNameDef(
                        type_ref->type_ref()->type_definition())));
                if (latest.has_value()) {
                  node = const_cast<TypeAnnotation*>(*latest);
                  replaced_anything = true;
                }
              }
              return const_cast<AstNode*>(node);
            }));
    if (!replaced_anything) {
      VLOG(6) << "No variables needed resolution in: "
              << annotation->ToString();
      return annotation;
    }
    const auto* result = dynamic_cast<const TypeAnnotation*>(clone);
    CHECK(result != nullptr);
    return result;
  }

  // Converts `member_type` into a regular `TypeAnnotation` that expresses the
  // type of the given struct member independently of the struct type. For
  // example, if `member_type` refers to `SomeStruct.foo`, and the type
  // annotation of the referenced `foo` field is `u32[5]`, then the result will
  // be the `u32[5]` annotation. The `accept_predicate` may be used to exclude
  // type annotations dependent on an implicit parametric that this utility is
  // being used to help infer.
  absl::StatusOr<const TypeAnnotation*> ExpandMemberType(
      std::optional<const ParametricContext*> parametric_context,
      const MemberTypeAnnotation* member_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* struct_type,
        ResolveVariableTypeAnnotations(
            parametric_context, member_type->struct_type(), accept_predicate));
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(struct_type, file_table_));
    if (!struct_or_proc_ref.has_value()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Invalid access of member `$0` of non-struct type: `$1`",
          member_type->member_name(), struct_type->ToString()));
    }
    const StructDefBase* struct_def = struct_or_proc_ref->def;
    if (struct_def->IsParametric()) {
      XLS_ASSIGN_OR_RETURN(
          parametric_context,
          GetOrCreateParametricStructContext(parametric_context,
                                             *struct_or_proc_ref, member_type));
    }
    std::optional<StructMemberNode*> member =
        struct_def->GetMemberByName(member_type->member_name());
    if (!member.has_value() && struct_def->impl().has_value()) {
      // If the member is not in the struct itself, it may be in the impl.
      std::optional<ImplMember> impl_member =
          (*struct_def->impl())->GetMember(member_type->member_name());
      if (impl_member.has_value()) {
        if (std::holds_alternative<ConstantDef*>(*impl_member)) {
          XLS_ASSIGN_OR_RETURN(
              std::optional<const TypeAnnotation*> member_type,
              UnifyTypeAnnotationsForNode(
                  parametric_context, std::get<ConstantDef*>(*impl_member)));
          CHECK(member_type.has_value());
          return GetParametricFreeStructMemberType(
              parametric_context, *struct_or_proc_ref, *member_type);
        }
        if (std::holds_alternative<Function*>(*impl_member)) {
          return GetParametricFreeStructMemberType(
              parametric_context, *struct_or_proc_ref,
              CreateFunctionTypeAnnotation(module_,
                                           *std::get<Function*>(*impl_member)));
        }
        return absl::UnimplementedError(
            absl::StrCat("Impl member type is not supported: ",
                         ToAstNode(*impl_member)->ToString()));
      }
    }
    if (!member.has_value()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "No member `$0` in struct `$1`.", member_type->member_name(),
          struct_def->identifier()));
    }
    return GetParametricFreeStructMemberType(
        parametric_context, *struct_or_proc_ref, (*member)->type());
  }

  // Converts `element_type` into a regular `TypeAnnotation` that expresses the
  // element type of the given array or tuple, independently of the array or
  // tuple type. For example, if `element_type` refers to an array whose type is
  // actually `u32[5]`, then the result will be a `u32` annotation. The
  // `accept_predicate` may be used to exclude type annotations dependent on an
  // implicit parametric that this utility is being used to help infer.
  absl::StatusOr<const TypeAnnotation*> ExpandElementType(
      std::optional<const ParametricContext*> parametric_context,
      const ElementTypeAnnotation* element_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* container_type,
                         ResolveVariableTypeAnnotations(
                             parametric_context, element_type->container_type(),
                             accept_predicate));
    if (const auto* array_type =
            dynamic_cast<const ArrayTypeAnnotation*>(container_type)) {
      return array_type->element_type();
    }
    if (const auto* tuple_type =
            dynamic_cast<const TupleTypeAnnotation*>(container_type)) {
      if (!element_type->tuple_index().has_value()) {
        return TypeInferenceErrorStatusForAnnotation(
            tuple_type->span(), tuple_type,
            "Tuples should not be indexed with array-style syntax. Use "
            "`tuple.<number>` syntax instead.",
            file_table_);
      }
      XLS_ASSIGN_OR_RETURN(
          uint64_t index,
          (*element_type->tuple_index())->GetAsUint64(file_table_));
      if (index >= tuple_type->members().size()) {
        return TypeInferenceErrorStatusForAnnotation(
            tuple_type->span(), tuple_type,
            absl::StrCat("Out-of-bounds tuple index specified: ", index),
            file_table_);
      }
      return tuple_type->members()[index];
    }
    if (element_type->allow_bit_vector_destructuring()) {
      absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
          GetSignednessAndBitCount(container_type);
      if (signedness_and_bit_count.ok()) {
        VLOG(6) << "Destructuring bit vector type: "
                << container_type->ToString();
        XLS_ASSIGN_OR_RETURN(
            bool signedness,
            EvaluateBoolOrExpr(parametric_context,
                               signedness_and_bit_count->signedness));
        return CreateUnOrSnElementAnnotation(module_, container_type->span(),
                                             signedness);
      }
    }
    return container_type;
  }

  absl::StatusOr<const TypeAnnotation*> ExpandReturnType(
      std::optional<const ParametricContext*> parametric_context,
      const ReturnTypeAnnotation* return_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "Expand return type: " << return_type->ToString();
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* function_type,
                         ResolveVariableTypeAnnotations(
                             parametric_context, return_type->function_type(),
                             [&](const TypeAnnotation* annotation) {
                               return annotation != return_type &&
                                      (!accept_predicate.has_value() ||
                                       (*accept_predicate)(annotation));
                             }));
    TypeAnnotation* result_type =
        dynamic_cast<const FunctionTypeAnnotation*>(function_type)
            ->return_type();
    VLOG(6) << "Resulting return type: " << result_type->ToString();
    return result_type;
  }

  absl::StatusOr<const TypeAnnotation*> ExpandParamType(
      std::optional<const ParametricContext*> parametric_context,
      const ParamTypeAnnotation* param_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "Expand param type: " << param_type->ToString();
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* function_type,
                         ResolveVariableTypeAnnotations(
                             parametric_context, param_type->function_type(),
                             [&](const TypeAnnotation* annotation) {
                               return annotation != param_type &&
                                      (!accept_predicate.has_value() ||
                                       (*accept_predicate)(annotation));
                             }));
    const std::vector<TypeAnnotation*>& resolved_types =
        dynamic_cast<const FunctionTypeAnnotation*>(function_type)
            ->param_types();
    CHECK(param_type->param_index() < resolved_types.size());
    VLOG(6) << "Resulting argument type: "
            << resolved_types[param_type->param_index()]->ToString();
    return resolved_types[param_type->param_index()];
  }

  std::optional<const TypeAnnotation*> ExpandSelfType(
      std::optional<const ParametricContext*> parametric_context,
      const SelfTypeAnnotation* self_type) {
    std::optional<const TypeAnnotation*> expanded =
        table_.GetTypeAnnotation(self_type);
    if (expanded.has_value()) {
      return expanded;
    }
    if (parametric_context.has_value()) {
      return (*parametric_context)->self_type();
    }
    return std::nullopt;
  }

  // Variant that deeply resolves all `TypeVariableTypeAnnotation`s within a
  // vector of annotations. If `accept_predicate` is specified, then any
  // annotations not accepted by the predicate are filtered from both
  // `annotations` and the expansions of any encountered type variables.
  absl::Status ResolveVariableTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TypeAnnotation*>& annotations,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    std::vector<const TypeAnnotation*> result;
    for (const TypeAnnotation* annotation : annotations) {
      if (!accept_predicate.has_value() || (*accept_predicate)(annotation)) {
        XLS_ASSIGN_OR_RETURN(
            const TypeAnnotation* resolved_annotation,
            ResolveVariableTypeAnnotations(parametric_context, annotation,
                                           accept_predicate));
        result.push_back(resolved_annotation);
      }
    }
    annotations = std::move(result);
    return absl::OkStatus();
  }

  // Determines if the given `type_variable` has any annotations in the table
  // that were explicitly written in the DSLX source.
  bool VariableHasAnyExplicitTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* type_variable) {
    absl::StatusOr<std::vector<const TypeAnnotation*>> annotations =
        table_.GetTypeAnnotationsForTypeVariable(parametric_context,
                                                 type_variable);
    return annotations.ok() &&
           absl::c_any_of(*annotations,
                          [this](const TypeAnnotation* annotation) {
                            return !table_.IsAutoLiteral(annotation);
                          });
  }

  // Wraps `BitCountMismatchErrorStatus` with resolution of parametrics, so that
  // a nominal type like `uN[N]` will not appear with the variable in the error
  // message.
  absl::Status BitCountMismatchErrorWithParametricResolution(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation1, const TypeAnnotation* annotation2) {
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
  absl::Status SignednessMismatchErrorWithParametricResolution(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation1, const TypeAnnotation* annotation2) {
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
  absl::Status TypeMismatchErrorWithParametricResolution(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation1, const TypeAnnotation* annotation2) {
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

  // Returns a `SignednessAndSize` that agrees with the two given
  // `SignednessAndSize` objects if possible. `x` is optional for convenience of
  // invoking this in a loop where the first call has no preceding value.
  //
  // Any error returned is a size or signedness mismatch error suitable for
  // display to the user. The `parametric_context` and the passed in type
  // annotations are used only for the purpose of generating errors. It is
  // assumed that `y_annotation` should be mentioned first in errors.
  absl::StatusOr<SignednessAndSize> UnifySignednessAndSize(
      std::optional<const ParametricContext*> parametric_context,
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
          parametric_context, update_annotation(y, y_annotation),
          update_annotation(*x, x_annotation));
    };
    auto bit_count_mismatch_error = [&] {
      return BitCountMismatchErrorWithParametricResolution(
          parametric_context, update_annotation(y, y_annotation),
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

  // Fabricates a `Number` node and sets the given type annotation for it in the
  // inference table.
  absl::StatusOr<Number*> MakeTypeCheckedNumber(
      const Span& span, const InterpValue& value,
      const TypeAnnotation* type_annotation) {
    VLOG(5) << "Creating type-checked number: " << value.ToString()
            << " of type: " << type_annotation->ToString();
    Number* number = module_.Make<Number>(
        span, value.ToString(/*humanize=*/true), NumberKind::kOther, nullptr);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(number, type_annotation));
    return number;
  }

  // Variant that takes a raw `int64_t` value for the number.
  absl::StatusOr<Number*> MakeTypeCheckedNumber(
      const Span& span, int64_t value, const TypeAnnotation* type_annotation) {
    return MakeTypeCheckedNumber(span, InterpValue::MakeS64(value),
                                 type_annotation);
  }

  InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  TypeInfo* const base_type_info_;
  const FileTable& file_table_;
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
  std::optional<std::unique_ptr<InferenceTableConverter>> builtins_converter =
      std::make_unique<InferenceTableConverter>(
          table, *builtins_module, import_data, warning_collector,
          builtins_type_info, file_table,
          /*builtins_converter=*/std::nullopt);
  InferenceTableConverter converter(table, module, import_data,
                                    warning_collector, base_type_info,
                                    file_table, std::move(builtins_converter));
  XLS_RETURN_IF_ERROR(
      converter.ConvertSubtree(&module, /*function=*/std::nullopt,
                               /*parametric_context=*/std::nullopt));

  VLOG(5) << "Inference table after conversion:";
  VLOG(5) << table.ToString();

  return converter.GetBaseTypeInfo();
}

}  // namespace xls::dslx
