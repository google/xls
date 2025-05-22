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

#include "xls/dslx/type_system_v2/inference_table_converter_impl.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
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
#include "xls/dslx/frontend/builtin_stubs_utils.h"
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
#include "xls/dslx/type_system_v2/fast_concretizer.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/populate_table_visitor.h"
#include "xls/dslx/type_system_v2/solve_for_parametrics.h"
#include "xls/dslx/type_system_v2/type_annotation_filter.h"
#include "xls/dslx/type_system_v2/type_annotation_resolver.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/type_system_v2/unify_type_annotations.h"
#include "xls/dslx/type_system_v2/validate_concrete_type.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// Traverses an AST and flattens it into a `vector` in the order the `TypeInfo`
// needs to be built such that prerequisites will be present in `TypeInfo` when
// evaluations are done.
class ConversionOrderVisitor : public AstNodeVisitorWithDefault {
 public:
  explicit ConversionOrderVisitor(const AstNode* root,
                                  bool handle_parametric_entities,
                                  const ImportData& import_data)
      : root_(root),
        handle_parametric_entities_(handle_parametric_entities),
        import_data_(import_data) {}

  absl::Status HandleFunction(const Function* node) override {
    if (!handle_parametric_entities_ && node->IsParametric()) {
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status HandleProc(const Proc* node) override {
    if (!handle_parametric_entities_ && node->IsParametric()) {
      return absl::OkStatus();
    }
    // Proc boundaries in the enclosing module scope are a "break point." The
    // caller needs to set up a new `TypeInfo` for the proc and then dive in. We
    // only dive in if the visitor's root is the proc subtree.
    if (node == root_) {
      return DefaultHandler(node);
    }
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleImpl(const Impl* node) override {
    XLS_ASSIGN_OR_RETURN(std::optional<const StructDefBase*> def,
                         GetStructOrProcDef(node->struct_ref(), import_data_));
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

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(node->type_annotation()->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->value()->Accept(this));
    XLS_RETURN_IF_ERROR(node->name_def()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleUnrollFor(const UnrollFor* node) override {
    // node->body() will not be handled because unroll_for generates new
    // unrolled body statements.
    if (node->type_annotation()) {
      XLS_RETURN_IF_ERROR(node->type_annotation()->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->iterable()->Accept(this));
    XLS_RETURN_IF_ERROR(node->names()->Accept(this));
    XLS_RETURN_IF_ERROR(node->init()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    // Prefer conversion of invocations before nodes that may use them.
    std::vector<const AstNode*> invocations;
    std::vector<const AstNode*> non_invocations;

    const Invocation* current_invocation =
        node->kind() == AstNodeKind::kInvocation
            ? down_cast<const Invocation*>(node)
            : nullptr;
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      if (current_invocation != nullptr &&
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
  const AstNode* const root_;
  const bool handle_parametric_entities_;
  const ImportData& import_data_;
  std::vector<const AstNode*> nodes_;
};

// RAII guard for a frame on the proc type info stack.
class ProcTypeInfoFrame {
 public:
  // Pushes `ti` onto `stack` and returns a guard that pops it when destroyed.
  static std::unique_ptr<ProcTypeInfoFrame> Push(std::stack<TypeInfo*>* stack,
                                                 TypeInfo* ti) {
    return absl::WrapUnique(new ProcTypeInfoFrame(stack, ti));
  }

  ProcTypeInfoFrame(const ProcTypeInfoFrame&) = delete;
  ProcTypeInfoFrame(ProcTypeInfoFrame&& frame) = default;

  ProcTypeInfoFrame& operator=(const ProcTypeInfoFrame&) = delete;
  ProcTypeInfoFrame& operator=(ProcTypeInfoFrame&&) = default;

  ~ProcTypeInfoFrame() {
    CHECK(!stack_->empty() && stack_->top() == ti_);
    stack_->pop();
  }

 private:
  ProcTypeInfoFrame(std::stack<TypeInfo*>* stack, TypeInfo* ti)
      : stack_(stack), ti_(ti) {
    stack_->push(ti);
  }

  std::stack<TypeInfo*>* stack_;
  TypeInfo* ti_;
};

class InferenceTableConverterImpl : public InferenceTableConverter,
                                    public UnificationErrorGenerator,
                                    public Evaluator,
                                    public ParametricStructInstantiator {
 public:
  InferenceTableConverterImpl(InferenceTable& table, Module& module,
                              ImportData& import_data,
                              WarningCollector& warning_collector,
                              TypeInfo* base_type_info,
                              const FileTable& file_table,
                              std::unique_ptr<TypeSystemTracer> tracer)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        base_type_info_(base_type_info),
        file_table_(file_table),
        tracer_(std::move(tracer)),
        resolver_(TypeAnnotationResolver::Create(
            module, table, file_table,
            /*error_generator=*/*this, /*evaluator=*/*this,
            /*parametric_struct_instantiator=*/*this, *tracer_, import_data_)),
        fast_concretizer_(FastConcretizer::Create(file_table)) {}

  // Returns true if the given function is a builtin.
  bool IsBuiltin(const Function* node) {
    return node->owner()->name() == kBuiltinStubsModuleName;
  }

  absl::Status ConvertSubtree(
      const AstNode* node, std::optional<const Function*> function,
      std::optional<const ParametricContext*> parametric_context,
      bool filter_param_type_annotations = false) override {
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
    if ((!parametric_context.has_value() ||
         (*parametric_context)->is_invocation()) &&
        node->owner() != &module_) {
      VLOG(5) << "Wrong module in ConvertSubtree; delegating to converter for  "
              << node->owner()->name();
      XLS_ASSIGN_OR_RETURN(
          InferenceTableConverter * converter,
          import_data_.GetInferenceTableConverter(node->owner()));
      return converter->ConvertSubtree(node, function, parametric_context,
                                       filter_param_type_annotations);
    }
    ConversionOrderVisitor visitor(
        node,
        parametric_context.has_value() &&
            (node == function ||
             (node->parent() != nullptr &&
              node->parent()->kind() == AstNodeKind::kImpl)),
        import_data_);
    XLS_RETURN_IF_ERROR(node->Accept(&visitor));
    for (const AstNode* node : visitor.nodes()) {
      VLOG(5) << "Next node: " << node->ToString();
      if (node->kind() == AstNodeKind::kInvocation) {
        XLS_RETURN_IF_ERROR(ConvertInvocation(
            down_cast<const Invocation*>(node), parametric_context));
      } else {
        XLS_RETURN_IF_ERROR(
            GenerateTypeInfo(parametric_context, node,
                             /*pre_unified_type=*/
                             std::nullopt,
                             filter_param_type_annotations
                                 ? TypeAnnotationFilter::FilterParamTypes()
                                 : TypeAnnotationFilter::None()));
      }
    }
    return absl::OkStatus();
  }

  // Adds the invocation type info for the given parametric invocation to the
  // output type info. `converted_parametric_envs_` must be populated for both
  // the caller and callee of the parametric invocation before doing this.
  absl::Status AddInvocationTypeInfo(
      const Invocation* invocation,
      const ParametricContext* parametric_context) {
    VLOG(5) << "Adding invocation type info for "
            << parametric_context->ToString();
    ParametricEnv parent_env;
    const auto& data =
        std::get<ParametricInvocationDetails>(parametric_context->details());
    if (parametric_context->parent_context().has_value() &&
        (*parametric_context->parent_context())->is_invocation()) {
      const auto it = converted_parametric_envs_.find(
          *parametric_context->parent_context());
      // Note that if a parametric function `g` is invoked by a default
      // parametric expr for a function `f`, there is a chicken-and-egg problem
      // where we cannot have produced the `ParametricEnv` for `f` at the time
      // we deal with the `g` invocation. However, because such invocations boil
      // down to constants without being converted to IR, we don't care if we
      // properly key the `TypeInfo` for that `g` invocation for access by IR
      // conversion (and v1 does not achieve that either).
      if (it != converted_parametric_envs_.end()) {
        parent_env = it->second;
      }
    }
    ParametricEnv callee_env =
        converted_parametric_envs_.at(parametric_context);
    VLOG(5) << "Parent env: " << parent_env.ToString();
    VLOG(5) << "Callee env: " << callee_env.ToString();
    CHECK_NE(invocation, nullptr);

    XLS_ASSIGN_OR_RETURN(TypeInfo * parent_ti,
                         GetTypeInfo(invocation->owner(), parametric_context));
    XLS_RETURN_IF_ERROR(parent_ti->AddInvocationTypeInfo(
        *invocation, data.caller.has_value() ? *data.caller : nullptr,
        parent_env, callee_env,
        IsBuiltin(data.callee) ? nullptr : parametric_context->type_info()));
    return absl::OkStatus();
  }

  // Make sure binding and parametric match - both values or both types
  absl::Status ValidateParametricsAgainstBindings(
      const std::vector<ParametricBinding*>& formal_bindings,
      const std::vector<ExprOrType>& explicit_parametrics) {
    int i = 0;
    for (ExprOrType parametric : explicit_parametrics) {
      if (i < formal_bindings.size()) {
        const ParametricBinding* binding = formal_bindings.at(i);
        bool formal_is_type_parametric =
            binding->type_annotation()->IsAnnotation<GenericTypeAnnotation>();
        if (std::holds_alternative<Expr*>(parametric) &&
            formal_is_type_parametric) {
          const Expr* expr = std::get<Expr*>(parametric);
          return TypeInferenceErrorStatus(
              expr->span(), nullptr,
              absl::Substitute("Expected parametric type, saw `$0`",
                               expr->ToString()),
              file_table_);
        }
        if (std::holds_alternative<TypeAnnotation*>(parametric) &&
            !formal_is_type_parametric) {
          const TypeAnnotation* type = std::get<TypeAnnotation*>(parametric);
          return TypeInferenceErrorStatus(
              type->span(), nullptr,
              absl::Substitute("Expected parametric value, saw `$0`",
                               type->ToString()),
              file_table_);
        }
      } else {
        break;
      }
      i++;
    }
    return absl::OkStatus();
  }

  // Converts the type info for the given invocation node and its argument
  // nodes. This involves resolving the callee function and applying the formal
  // types of the arguments to the actual arguments in the inference table.
  absl::Status ConvertInvocation(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context) {
    TypeSystemTrace trace =
        tracer_->TraceConvertInvocation(invocation, caller_context);
    std::optional<const Function*> caller = GetContainingFunction(invocation);
    VLOG(5) << "Converting invocation: " << invocation->callee()->ToString()
            << " with module: " << invocation->callee()->owner()->name()
            << " in module: " << module_.name()
            << " in context: " << ToString(caller_context);
    XLS_ASSIGN_OR_RETURN(
        const FunctionAndTargetObject function_and_target_object,
        ResolveFunction(invocation->callee(), caller, caller_context));

    // This is one condition for enabling an "implicit token" in v1.
    // TODO: https://github.com/google/xls/issues/193 - Handle the condition
    // based on root type info of the callee.
    if (caller.has_value() && IsBuiltin(function_and_target_object.function) &&
        GetBuiltinFnRequiresImplicitToken(invocation->callee())) {
      XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                           GetTypeInfo((*caller)->owner(), caller_context));
      ti->NoteRequiresImplicitToken(**caller, true);
    }

    const Function* function = function_and_target_object.function;

    if (caller.has_value() && function == *caller) {
      return TypeInferenceErrorStatus(
          invocation->span(), nullptr,
          absl::Substitute("Recursion of function `$0` detected -- recursion "
                           "is currently unsupported.",
                           function->identifier()),
          file_table_);
    }

    // Come up with the actual args by merging the possible target object
    // (`some_struct` in the case of `some_struct.foo(args)`), with the vector
    // of explicit actual args.
    std::vector<const Expr*> actual_args;
    if (function_and_target_object.target_object.has_value()) {
      actual_args.push_back(*function_and_target_object.target_object);
    }
    absl::c_copy(invocation->args(), std::back_inserter(actual_args));

    // Note that functions in procs are treated as if they are parametric (as in
    // v1), because there needs to be a `TypeInfo` with separate const-exprs per
    // instantiation of a proc.
    if (!function->IsParametric() && !function->IsInProc()) {
      std::optional<std::string_view> builtin =
          GetBuiltinFnName(invocation->callee());
      if (builtin.has_value() || function->owner() != &module_) {
        const FunctionTypeAnnotation* ft_annotation =
            CreateFunctionTypeAnnotation(module_, *function);
        if (builtin.has_value() && builtin == "join") {
          ft_annotation =
              ExpandVarargs(module_, ft_annotation, actual_args.size());
        }
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(invocation->callee(), ft_annotation));
      }

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
          (*signature)->AsAnnotation<FunctionTypeAnnotation>();
      XLS_RETURN_IF_ERROR(
          table_.AddTypeAnnotationToVariableForParametricContext(
              caller_context, *table_.GetTypeVariable(invocation),
              function_type->return_type()));
      for (int i = 0; i < function_type->param_types().size(); i++) {
        const TypeAnnotation* formal_param = function_type->param_types()[i];
        const Expr* actual_param = actual_args[i];
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(actual_param, formal_param));
        TypeSystemTrace arg_trace =
            tracer_->TraceConvertActualArgument(actual_param);
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(actual_param, caller, caller_context));
      }
      return GenerateTypeInfo(caller_context, invocation,
                              function_type->return_type());
    }

    XLS_RETURN_IF_ERROR(ValidateParametricsAgainstBindings(
        function->parametric_bindings(), invocation->explicit_parametrics()));

    // If we get here, we are dealing with a parametric function. First let's
    // make sure a valid number of parametrics and regular arguments are being
    // passed in.
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
    const int formal_param_count_without_self =
        (function->params().size() - (function->IsMethod() ? 1 : 0));
    if (invocation->args().size() != formal_param_count_without_self) {
      // Note that the eventual unification of the signature would also catch
      // this, but this redundant check ensures that an arg count mismatch error
      // takes precedence over "could not infer parametric: N" errors that are
      // caused by too few regular args.
      return ArgCountMismatchErrorStatus(
          invocation->span(),
          absl::Substitute("Expected $0 argument(s) but got $1.",
                           function->params().size(),
                           invocation->args().size()),
          file_table_);
    }

    // The parametric invocation now gets its own data structure set up in both
    // the `InferenceTable` and the `TypeInfo` hierarchy.

    std::optional<const ParametricContext*> caller_or_target_struct_context =
        function_and_target_object.target_struct_context.has_value()
            ? function_and_target_object.target_struct_context
            : caller_context;
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * base_type_info,
        GetTypeInfo(function->owner(), caller_or_target_struct_context));
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * invocation_type_info,
        import_data_.type_info_owner().New(function->owner(), base_type_info));

    XLS_ASSIGN_OR_RETURN(
        ParametricContext * invocation_context,
        table_.AddParametricInvocation(
            *invocation, *function, caller, caller_context,
            function_and_target_object.target_struct_context.has_value()
                ? (*function_and_target_object.target_struct_context)
                      ->self_type()
                : std::nullopt,
            invocation_type_info));
    VLOG(5) << "ConvertInvocation for: " << invocation->ToString()
            << " for module: " << module_.name()
            << " with invocation_type_info of module "
            << invocation_type_info->module()->name()
            << " in invocation (parametric) context: "
            << ToString(invocation_context);

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

    // Figure out any implicit parametrics and generate the `ParametricEnv`.
    XLS_ASSIGN_OR_RETURN(ParametricEnv env,
                         GenerateParametricFunctionEnv(
                             function_and_target_object.target_struct_context,
                             invocation_context, invocation));
    const bool canonicalized = table_.MapToCanonicalInvocationTypeInfo(
        invocation_context, std::move(env));
    XLS_RETURN_IF_ERROR(AddInvocationTypeInfo(invocation, invocation_context));

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
          return annotation->IsAnnotation<MemberTypeAnnotation>();
        }));

    const FunctionTypeAnnotation* parametric_free_function_type = nullptr;
    if (canonicalized) {
      parametric_free_function_type =
          std::get<ParametricInvocationDetails>(invocation_context->details())
              .parametric_free_function_type;
    } else {
      // Apply the parametric-free formal types to the arguments and convert
      // them.
      XLS_ASSIGN_OR_RETURN(const TypeAnnotation* parametric_free_type,
                           GetParametricFreeType(
                               CreateFunctionTypeAnnotation(module_, *function),
                               parametric_value_exprs_.at(invocation_context),
                               invocation_context->self_type()));
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(parametric_free_type, caller, caller_context));

      XLS_ASSIGN_OR_RETURN(parametric_free_type,
                           resolver_->ResolveIndirectTypeAnnotations(
                               invocation_context, parametric_free_type,
                               TypeAnnotationFilter::None()));
      parametric_free_function_type =
          down_cast<const FunctionTypeAnnotation*>(parametric_free_type);

      invocation_context->SetParametricFreeFunctionType(
          parametric_free_function_type);
    }

    XLS_RETURN_IF_ERROR(table_.AddTypeAnnotationToVariableForParametricContext(
        caller_context, callee_variable, parametric_free_function_type));
    XLS_RETURN_IF_ERROR(table_.AddTypeAnnotationToVariableForParametricContext(
        caller_context, *table_.GetTypeVariable(invocation),
        parametric_free_function_type->return_type()));
    XLS_RETURN_IF_ERROR(GenerateTypeInfo(caller_or_target_struct_context,
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
      TypeSystemTrace arg_trace =
          tracer_->TraceConvertActualArgument(actual_param);
      XLS_RETURN_IF_ERROR(ConvertSubtree(
          actual_param, caller,
          is_self_param ? function_and_target_object.target_struct_context
                        : caller_context));
    }

    // Convert the actual parametric function in the context of this invocation,
    // and finally, convert the invocation node. If the function is in a proc,
    // we need that proc's type info to be on the proc type info stack while
    // converting the function.
    std::unique_ptr<ProcTypeInfoFrame> proc_type_info_frame;
    if (function->IsInProc()) {
      XLS_ASSIGN_OR_RETURN(proc_type_info_frame,
                           PushProcTypeInfo(*function->proc()));
    }
    if (!canonicalized) {
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(function, function, invocation_context));
    }
    return GenerateTypeInfo(caller_context, invocation);
  }

  // Gets the output `TypeInfo` corresponding to the given
  // `parametric_context`, which may be `nullopt`, in which case it returns
  // the base type info. If the proposed type info doesn't belong to the
  // provided module, its root type info is returned.
  absl::StatusOr<TypeInfo*> GetTypeInfo(
      const Module* module,
      std::optional<const ParametricContext*> parametric_context) {
    TypeInfo* return_ti = base_type_info_;
    if (!proc_type_info_stack_.empty()) {
      return_ti = proc_type_info_stack_.top();
    }
    if (parametric_context.has_value()) {
      return_ti = (*parametric_context)->type_info();
    }
    if (return_ti->module() != module) {
      return import_data_.GetRootTypeInfo(module);
    }
    return return_ti;
  }

  // Retrieves the `ParametricEnv` corresponding to `parametric_context` if the
  // context is not `nullopt`, it is for an invocation, and we have converted
  // the invocation.
  //
  // TODO: https://github.com/google/xls/issues/193 - we should probably stop
  // limiting this to invocations once the bytecode interpreter is able to
  // look up type info for parametric impls.
  ParametricEnv GetParametricEnv(
      std::optional<const ParametricContext*> parametric_context) {
    if (parametric_context.has_value() &&
        (*parametric_context)->is_invocation()) {
      const auto it = converted_parametric_envs_.find(*parametric_context);
      if (it != converted_parametric_envs_.end()) {
        return it->second;
      }
    }
    return ParametricEnv{};
  }

  bool IsProcAtTopOfTypeInfoStack(const Proc* proc) {
    absl::StatusOr<TypeInfo*> ti =
        base_type_info_->GetTopLevelProcTypeInfo(proc);
    return ti.ok() && !proc_type_info_stack_.empty() &&
           proc_type_info_stack_.top() == *ti;
  }

  absl::StatusOr<std::unique_ptr<ProcTypeInfoFrame>> PushProcTypeInfo(
      const Proc* proc) {
    absl::StatusOr<TypeInfo*> ti =
        base_type_info_->GetTopLevelProcTypeInfo(proc);
    if (!ti.ok()) {
      XLS_ASSIGN_OR_RETURN(
          ti, import_data_.type_info_owner().New(&module_, base_type_info_));
      XLS_RETURN_IF_ERROR(base_type_info_->SetTopLevelProcTypeInfo(proc, *ti));
    }
    return ProcTypeInfoFrame::Push(&proc_type_info_stack_, *ti);
  }

  // Generates type info for one node.
  absl::Status GenerateTypeInfo(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node,
      std::optional<const TypeAnnotation*> pre_unified_type = std::nullopt,
      TypeAnnotationFilter type_annotation_filter =
          TypeAnnotationFilter::None()) {
    TypeSystemTrace trace = tracer_->TraceConvertNode(node);
    VLOG(5) << "GenerateTypeInfo for node: " << node->ToString()
            << " of kind: `" << AstNodeKindToString(node->kind()) << "`"
            << " with owner: " << node->owner()->name()
            << " for module: " << module_.name()
            << " in parametric context: " << ToString(parametric_context);
    if (pre_unified_type.has_value()) {
      VLOG(5) << "Using pre-unified type: " << (*pre_unified_type)->ToString();
    }

    // A proc requires its own top-level `TypeInfo`, so dealing with a proc root
    // node has the following sequence:
    // 1. The conversion order visitor at module level stops at the proc root
    //    without diving in.
    // 2. We get here with the proc root.
    // 3. Below, we create the top-level proc type info and push it onto the
    //    stack. This makes it so that this `TypeInfo` is treated as the
    //    base-level `TypeInfo` until that is popped (see `GetTypeInfo()`).
    // 4. Also below, we kick off conversion of the proc's subtree.
    // 5. The last step in (4) is to re-enter here with the `Proc` node and the
    //    type info for the proc still at the top of the stack. That call
    //    will skip over the following block.
    // 6. The original call pops the stack at the end.
    std::unique_ptr<ProcTypeInfoFrame> proc_type_info_frame;
    if (node->kind() == AstNodeKind::kProc) {
      const Proc* proc = down_cast<const Proc*>(node);
      if (!IsProcAtTopOfTypeInfoStack(proc)) {
        XLS_ASSIGN_OR_RETURN(proc_type_info_frame, PushProcTypeInfo(proc));
        return ConvertSubtree(proc, std::nullopt, parametric_context);
      }
    }

    if (node->kind() == AstNodeKind::kImport) {
      const Import* import = down_cast<const Import*>(node);
      XLS_ASSIGN_OR_RETURN(ModuleInfo * imported_module_info,
                           import_data_.Get(ImportTokens(import->subject())));
      base_type_info_->AddImport(const_cast<Import*>(import),
                                 &imported_module_info->module(),
                                 imported_module_info->type_info());
      return absl::OkStatus();
    }

    XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                         GetTypeInfo(node->owner(), parametric_context));
    std::optional<const TypeAnnotation*> annotation = pre_unified_type;

    // Resolve and unify the type information for the node, if it was not
    // pre-unified.
    bool node_is_annotation = false;
    if (!annotation.has_value()) {
      if (node->kind() == AstNodeKind::kTypeAnnotation) {
        // If the node itself is a `TypeAnnotation`, it doesn't need type
        // unification.
        node_is_annotation = true;
        annotation = down_cast<const TypeAnnotation*>(node);

        // Builtin fragments of bits-like types, like `uN` and `xN[true]` have
        // no `Type` conversion. We only convert the complete thing.
        if (IsBitsLikeFragment(*annotation)) {
          return absl::OkStatus();
        }
      } else {
        // In most cases, come up with a unified type annotation using the
        // table (or the table of the owning module).
        std::unique_ptr<TypeAnnotationResolver> foreign_resolver;
        TypeAnnotationResolver* resolver = resolver_.get();
        if (parametric_context.has_value() &&
            (*parametric_context)->is_invocation()) {
          auto details = std::get<ParametricInvocationDetails>(
              (*parametric_context)->details());
          if (details.callee->owner() != &module_) {
            foreign_resolver = TypeAnnotationResolver::Create(
                *details.callee->owner(), table_, file_table_,
                /*error_generator=*/*this, /*evaluator=*/*this,
                /*parametric_struct_instantiator=*/*this, *tracer_,
                import_data_);
            resolver = foreign_resolver.get();
          }
        }
        XLS_ASSIGN_OR_RETURN(
            annotation, resolver->ResolveAndUnifyTypeAnnotationsForNode(
                            parametric_context, node, type_annotation_filter));
      }
    }

    if (node->kind() == AstNodeKind::kFunction) {
      // Don't require an implicit token by default. `ConvertInvocation` will
      // set this to true, for the caller of an invocation, in the narrow cases
      // where we want it.
      const Function& function = *down_cast<const Function*>(node);
      if (!ti->GetRequiresImplicitToken(function).has_value()) {
        ti->NoteRequiresImplicitToken(function, false);
      }
    }
    if (node->kind() == AstNodeKind::kFormatMacro) {
      // `FormatMacro` is essentially an invocation of a "parametric builtin"
      // represented with a custom node, so it meets the condition in
      // `ConvertInvocation` for the caller to require an implicit token.
      std::optional<const Function*> caller = GetContainingFunction(node);
      if (caller.has_value()) {
        ti->NoteRequiresImplicitToken(**caller, true);
      }
    }

    if (!annotation.has_value()) {
      // The caller may have passed a node that is in the AST but not in the
      // table, and it may not be needed in the table.
      VLOG(5) << "No type information for: " << node->ToString();
      return absl::OkStatus();
    }

    absl::StatusOr<std::unique_ptr<Type>> type =
        Concretize(*annotation, parametric_context,
                   /*needs_conversion_before_eval=*/node->kind() !=
                       AstNodeKind::kTypeAnnotation);

    if (!type.ok()) {
      // When the node itself is an annotation, and we decide to concretize
      // the node itself, we can't succeed in all contexts. For example, a
      // parametric-dependent field type declaration inside a parametric struct
      // declaration can't just be concretized like that. Rather than trying to
      // identify such cases, we just consider such nodes best-effort.
      return node_is_annotation ? absl::OkStatus() : type.status();
    }

    XLS_RETURN_IF_ERROR(ValidateConcreteType(node, type->get(), *ti,
                                             *annotation, warning_collector_,
                                             import_data_, file_table_));
    if (node->kind() == AstNodeKind::kNumber) {
      if (const auto* literal = down_cast<const Number*>(node);
          literal->type_annotation() != nullptr) {
        ti->SetItem(literal->type_annotation(),
                    *std::make_unique<MetaType>((*type)->CloneToUnique()));
      }
    }
    ti->SetItem(node, **type);
    XLS_RETURN_IF_ERROR(NoteIfConstExpr(parametric_context, node, **type, ti));
    VLOG(5) << "Generated type: " << (*ti->GetItem(node))->ToString()
            << " for node: " << node->ToString() << " in ti module "
            << ti->module()->name();
    return absl::OkStatus();
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
    XLS_ASSIGN_OR_RETURN(TypeInfo * struct_base_ti,
                         GetTypeInfo(ref.def->owner(), parent_context));
    auto type_info_factory = [&] {
      return import_data_.type_info_owner().New(ref.def->owner(),
                                                struct_base_ti);
    };
    XLS_ASSIGN_OR_RETURN(
        const ParametricContext* struct_context,
        table_.GetOrCreateParametricStructContext(
            ref.def, node, parametric_env, CreateStructAnnotation(module_, ref),
            type_info_factory));
    const auto it = converted_parametric_envs_.find(struct_context);
    if (it != converted_parametric_envs_.end()) {
      return struct_context;
    }
    converted_parametric_envs_.emplace(struct_context, parametric_env);
    TypeInfo* ti = struct_context->type_info();
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
  TypeInfo* GetBaseTypeInfo() override { return base_type_info_; }

 private:
  // Converts the given type annotation to a concrete `Type`, either statically
  // or in the context of a parametric invocation. The
  // `needs_conversion_before_eval` flag indicates if the annotation needs its
  // subtree converted before evaluating parts of it (this is not applicable if
  // the fast concretizer handles it).
  absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context,
      bool needs_conversion_before_eval = false) {
    TypeSystemTrace trace = tracer_->TraceConcretize(annotation);
    VLOG(5) << "Concretize: " << annotation->ToString()
            << " in context invocation: " << ToString(parametric_context);
    VLOG(5) << "Effective context: " << ToString(parametric_context);

    absl::StatusOr<std::unique_ptr<Type>> type =
        fast_concretizer_->Concretize(annotation);
    if (type.ok()) {
      return std::move(*type);
    }

    // Any annotation which actually gets used as the type of a node should be
    // converted itself, in order to generate type info for expressions that
    // are embedded in it, so those do not disrupt evaluations that occur in
    // concretization. These annotations may not be scheduled for conversion
    // otherwise, because they may be fabricated.
    if (needs_conversion_before_eval) {
      XLS_RETURN_IF_ERROR(ConvertSubtree(annotation, /*function=*/std::nullopt,
                                         parametric_context));
    }

    XLS_ASSIGN_OR_RETURN(annotation, resolver_->ResolveIndirectTypeAnnotations(
                                         parametric_context, annotation,
                                         TypeAnnotationFilter::None()));
    if (annotation->IsAnnotation<AnyTypeAnnotation>()) {
      return absl::InvalidArgumentError(
          "Attempting to concretize `Any` type, which means there was "
          "insufficient type info.");
    }
    if (IsToken(annotation)) {
      return std::make_unique<TokenType>();
    }
    if (annotation->IsAnnotation<TupleTypeAnnotation>()) {
      const auto* tuple = annotation->AsAnnotation<TupleTypeAnnotation>();
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
    if (annotation->IsAnnotation<FunctionTypeAnnotation>()) {
      const auto* function = annotation->AsAnnotation<FunctionTypeAnnotation>();
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
    if (annotation->IsAnnotation<ChannelTypeAnnotation>()) {
      const auto* channel = annotation->AsAnnotation<ChannelTypeAnnotation>();
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> payload_type,
                           Concretize(channel->payload(), parametric_context));
      std::unique_ptr<Type> type = std::make_unique<ChannelType>(
          std::move(payload_type), channel->direction());
      if (channel->dims().has_value()) {
        for (Expr* dim : *channel->dims()) {
          XLS_ASSIGN_OR_RETURN(int64_t size,
                               EvaluateU32OrExpr(parametric_context, dim));
          type = std::make_unique<ArrayType>(
              std::move(type), TypeDim(InterpValue::MakeU32(size)));
        }
      }
      return type;
    }
    XLS_ASSIGN_OR_RETURN(std::optional<const EnumDef*> enum_def,
                         GetEnumDef(annotation, import_data_));
    if (enum_def.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          TypeInfo * ti, GetTypeInfo((*enum_def)->owner(), parametric_context));
      std::vector<InterpValue> members;
      BitsType* underlying_type = nullptr;

      // Handle special case: if the enum has no values, its explicitly
      // annotated type should be used.
      if ((*enum_def)->values().empty()) {
        CHECK((*enum_def)->type_annotation());
        XLS_ASSIGN_OR_RETURN(
            std::unique_ptr<Type> type,
            Concretize((*enum_def)->type_annotation(), parametric_context));
        ti->SetItem((*enum_def)->type_annotation(), std::move(type));
        XLS_ASSIGN_OR_RETURN(
            underlying_type,
            ti->GetItemAs<BitsType>((*enum_def)->type_annotation()));
      } else {
        XLS_ASSIGN_OR_RETURN(
            underlying_type,
            ti->GetItemAs<BitsType>((*enum_def)->values()[0].value));

        for (const EnumMember& value : (*enum_def)->values()) {
          absl::StatusOr<InterpValue> evaluated_value =
              ti->GetConstExpr(value.value);
          // Evaluate any enum value that has not been evaluated, such as const
          // expressions.
          if (!evaluated_value.ok()) {
            evaluated_value = ConstexprEvaluator::EvaluateToValue(
                &import_data_, ti, &warning_collector_,
                GetParametricEnv(parametric_context), value.value);
            if (!evaluated_value.ok()) {
              return NotConstantErrorStatus(value.value->span(), value.value,
                                            file_table_);
            }
            ti->NoteConstExpr(value.value, *evaluated_value);
            ti->NoteConstExpr(value.name_def, *evaluated_value);
          }
          // Evaluated enum value has numeric type, which needs to be converted
          // to enum type.
          XLS_ASSIGN_OR_RETURN(auto bits, evaluated_value->GetBits());
          InterpValue enum_value = InterpValue::MakeEnum(
              bits, evaluated_value->IsSigned(), *enum_def);
          ti->NoteConstExpr(value.value, enum_value);
          ti->NoteConstExpr(value.name_def, enum_value);
          members.push_back(enum_value);
        }
      }
      return std::make_unique<EnumType>(**enum_def, underlying_type->size(),
                                        underlying_type->is_signed(), members);
    }
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc,
                         GetStructOrProcRef(annotation, import_data_));
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
            *down_cast<const StructDef*>(struct_def_base));
      }
      return std::make_unique<ProcType>(
          std::move(member_types), *down_cast<const ProcDef*>(struct_def_base));
    }
    XLS_ASSIGN_OR_RETURN(SignednessAndBitCountResult signedness_and_bit_count,
                         GetSignednessAndBitCountWithUserFacingError(
                             annotation, file_table_, [&] {
                               return absl::UnimplementedError(absl::Substitute(
                                   "Type inference version 2 is a work in "
                                   "progress and cannot yet handle "
                                   "non-bits-like type annotation `$0`.",
                                   annotation->ToString()));
                             }));
    XLS_ASSIGN_OR_RETURN(
        bool signedness,
        EvaluateBoolOrExpr(parametric_context,
                           signedness_and_bit_count.signedness));
    XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                         EvaluateU32OrExpr(parametric_context,
                                           signedness_and_bit_count.bit_count));
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
          &import_data_, ti, &warning_collector_,
          GetParametricEnv(parametric_context), constant_def->value(), &type);
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
      // Imports should have already been marked constexpr in their own module.
      if (IsImport(colon_ref)) {
        return absl::OkStatus();
      }
      return NoteColonRefIfConstExpr(parametric_context, colon_ref, type, ti);
    }
    if (const auto* number = dynamic_cast<const Number*>(node)) {
      XLS_ASSIGN_OR_RETURN(InterpValue value, EvaluateNumber(*number, type));
      ti->NoteConstExpr(number, value);
    }
    if (const auto* let = dynamic_cast<const Let*>(node)) {
      return NoteLetIfConstExpr(let, parametric_context, type, ti);
    }
    if (const auto* index = dynamic_cast<const Index*>(node)) {
      // A `Slice` actually has its bounds stored in `TypeInfo` out-of-band from
      // the real type info, mirroring the `StartAndWidthExprs` that we store in
      // the `InferenceTable`.
      if (std::holds_alternative<Slice*>(index->rhs())) {
        std::optional<StartAndWidthExprs> start_and_width_exprs =
            table_.GetSliceStartAndWidthExprs(ToAstNode(index->rhs()));
        CHECK(start_and_width_exprs.has_value());
        StartAndWidth start_and_width;
        XLS_ASSIGN_OR_RETURN(start_and_width.start,
                             EvaluateU32OrExpr(parametric_context,
                                               start_and_width_exprs->start));
        XLS_ASSIGN_OR_RETURN(start_and_width.width,
                             EvaluateU32OrExpr(parametric_context,
                                               start_and_width_exprs->width));
        ti->AddSliceStartAndWidth(std::get<Slice*>(index->rhs()),
                                  GetParametricEnv(parametric_context),
                                  start_and_width);
      }
    }
    if (const auto* const_assert = dynamic_cast<const ConstAssert*>(node)) {
      return CheckConstAssert(const_assert, parametric_context, type, ti);
    }
    if (const auto* unroll_for = dynamic_cast<const UnrollFor*>(node)) {
      return ExpandUnrollFor(unroll_for, parametric_context, type, ti);
    }
    if (const auto* format_macro = dynamic_cast<const FormatMacro*>(node)) {
      return NoteFormatMacroVerbosityConstExpr(format_macro, type, ti);
    }
    if (const auto* invocation = dynamic_cast<const Invocation*>(node);
        invocation != nullptr && IsBuiltinFn(invocation->callee())) {
      NameRef* callee_nameref = dynamic_cast<NameRef*>(invocation->callee());
      CHECK_NE(callee_nameref, nullptr);
      std::optional<const Type*> callee_type =
          ti->GetItem(invocation->callee());
      if (callee_type.has_value()) {
        const auto& function_type = (*callee_type)->AsFunction();
        return NoteBuiltinInvocationConstExpr(callee_nameref->identifier(),
                                              invocation, function_type, ti);
      }
    }
    return absl::OkStatus();
  }

  // Used when a `ConstAssert` is concretized, to actually check if the asserted
  // expression holds.
  absl::Status CheckConstAssert(
      const ConstAssert* node,
      std::optional<const ParametricContext*> parametric_context,
      const Type& type, TypeInfo* ti) {
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti, &warning_collector_,
        GetParametricEnv(parametric_context), node->arg(), &type);
    if (!value.ok()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute("const_assert! expression is not constexpr: `$0`",
                           node->arg()->ToString()),
          file_table_);
    }
    VLOG(6) << "Evaluated const assert: " << node->arg()->ToString()
            << " to: " << value->ToString();
    if (value->IsFalse()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute("const_assert! failure: `$0`",
                           node->arg()->ToString()),
          file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status NoteColonRefIfConstExpr(
      std::optional<const ParametricContext*> parametric_context,
      const ColonRef* colon_ref, const Type& type, TypeInfo* ti) {
    // Handle enum here, at this point its type is concretized and all its
    // values are evaluated as constexpr.
    if (type.IsEnum()) {
      const auto& enum_type = type.AsEnum();
      const EnumDef& enum_def = enum_type.nominal_type();
      absl::StatusOr<Expr*> enum_value = enum_def.GetValue(colon_ref->attr());
      if (!enum_value.ok()) {
        return UndefinedNameErrorStatus(colon_ref->span(), colon_ref,
                                        colon_ref->attr(), file_table_);
      }
      XLS_ASSIGN_OR_RETURN(
          TypeInfo * eval_ti,
          GetTypeInfo((*enum_value)->owner(), parametric_context));
      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           eval_ti->GetConstExpr(*enum_value));
      ti->NoteConstExpr(colon_ref, value);
      return absl::OkStatus();
    }

    const std::optional<const AstNode*> target =
        table_.GetColonRefTarget(colon_ref);

    if (!target.has_value()) {
      std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
      if (bits_like.has_value()) {
        VLOG(6) << "ColonRef is a universal constant referenced with an "
                   "indirect annotation: "
                << colon_ref->ToString();
        XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());
        XLS_ASSIGN_OR_RETURN(uint32_t bit_count, bits_like->size.GetAsInt64());
        XLS_ASSIGN_OR_RETURN(
            InterpValueWithTypeAnnotation member,
            GetBuiltinMember(module_, is_signed, bit_count, colon_ref->attr(),
                             colon_ref->span(), type.ToString(), file_table_));
        ti->NoteConstExpr(colon_ref, std::move(member.value));
      } else {
        VLOG(6) << "ColonRef has no constexpr value: " << colon_ref->ToString();
      }
      return absl::OkStatus();
    }

    // In a case like `S<parametrics>::CONSTANT`, what we do here is
    // constexpr-evaluate `CONSTANT` against the parametric context `TypeInfo`
    // for `S<parametrics>`. This will be `evaluation_ti`. Then we map the
    // result of that to the `ColonRef` node, in the `TypeInfo` where the
    // `ColonRef` resides (which is `ti`). In a non-parametric case like
    // `S::CONSTANT`, there is only one `TypeInfo` involved, so this logic
    // that figures out `evaluation_ti` is a no-op.
    XLS_ASSIGN_OR_RETURN(TypeInfo * evaluation_ti,
                         import_data_.GetRootTypeInfoForNode(*target));
    VLOG(6) << "Checking ColonRef constexpr value for: "
            << colon_ref->ToString()
            << " with target: " << (*target)->ToString();
    if ((*target)->kind() == AstNodeKind::kConstantDef) {
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc,
                           GetStructOrProcRef(colon_ref, import_data_));
      if (struct_or_proc.has_value() && struct_or_proc->def->IsParametric()) {
        XLS_ASSIGN_OR_RETURN(
            const ParametricContext* struct_context,
            GetOrCreateParametricStructContext(parametric_context,
                                               *struct_or_proc, colon_ref));
        evaluation_ti = struct_context->type_info();
      }

      // Evaluate the value, and note it if successful.
      absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
          &import_data_, evaluation_ti, &warning_collector_,
          GetParametricEnv(parametric_context),
          down_cast<const ConstantDef*>(*target)->value(), &type);
      if (value.ok()) {
        VLOG(6) << "Noting constexpr for ColonRef: " << colon_ref->ToString()
                << ", value: " << value->ToString();
        ti->NoteConstExpr(colon_ref, *value);
      }
    }
    return absl::OkStatus();
  }

  absl::Status NoteLetIfConstExpr(
      const Let* let,
      std::optional<const ParametricContext*> parametric_context,
      const Type& type, TypeInfo* ti) {
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti, &warning_collector_,
        GetParametricEnv(parametric_context), let->rhs(), &type);
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

  // Unroll for is expanded to a sequence of statements as following.
  // ```
  // let X = unroll_for! (i, a) in iterable {
  //   body_statements
  //   last_body_expr
  // } (init);
  // ```
  // becomes
  // ```
  // let a = init;
  // let i = iterable[0];
  // body_statements
  // let a = last_body_expr;
  // let i = iterable[1];
  // body_statements
  // let a = last_body_expr;
  // ... // repeat for each element in iterable
  // let i = iterable[iterable.size() - 1];
  // body_statements
  // let X = last_body_expr;
  // ```
  absl::Status ExpandUnrollFor(
      const UnrollFor* unroll_for,
      std::optional<const ParametricContext*> parametric_context,
      const Type& type, TypeInfo* ti) {
    TypeSystemTrace trace = tracer_->TraceUnroll(unroll_for);
    std::vector<Statement*> unrolled_statements;

    CHECK_EQ(unroll_for->names()->nodes().size(), 2);
    NameDefTree* iterator_name = unroll_for->names()->nodes()[0];
    NameDefTree* accumulator_name = unroll_for->names()->nodes()[1];
    TypeAnnotation* iterator_type = nullptr;
    TypeAnnotation* accumulator_type = nullptr;
    std::optional<const TypeAnnotation*> name_type_annotation =
        table_.GetTypeAnnotation(unroll_for->names());
    if (name_type_annotation.has_value()) {
      if ((*name_type_annotation)->IsAnnotation<TupleTypeAnnotation>()) {
        const auto* types =
            (*name_type_annotation)->AsAnnotation<TupleTypeAnnotation>();
        CHECK_EQ(types->size(), 2);
        iterator_type = types->members()[0];
        accumulator_type = types->members()[1];
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid type annotation in unroll_for!: ",
                         unroll_for->type_annotation()->ToString(),
                         ", expect a tuple type with two members"));
      }
    }

    // The only thing that needs to be evaluated to a constant here is the size
    // of the iterable, while the elements need not to be constants. However if
    // the entire iterable is a constant array, we can use their evaluated
    // values directly.
    absl::StatusOr<InterpValue> const_iterable =
        ConstexprEvaluator::EvaluateToValue(
            &import_data_, ti, &warning_collector_,
            GetParametricEnv(parametric_context), unroll_for->iterable());
    const std::vector<InterpValue>* iterable_values = nullptr;
    uint64_t size = 0;
    if (const_iterable.ok()) {
      XLS_ASSIGN_OR_RETURN(iterable_values, const_iterable->GetValues());
      size = iterable_values->size();
    } else {
      absl::StatusOr<uint64_t> size_from_type(TypeMissingErrorStatus(
          *unroll_for->iterable(), nullptr, file_table_));
      std::optional<Type*> iterable_type = ti->GetItem(unroll_for->iterable());
      if (iterable_type.has_value() && (*iterable_type)->IsArray() &&
          std::holds_alternative<InterpValue>(
              (*iterable_type)->AsArray().size().value())) {
        InterpValue dim =
            std::get<InterpValue>((*iterable_type)->AsArray().size().value());
        size_from_type = dim.GetBitValueUnsigned();
      }
      XLS_ASSIGN_OR_RETURN(size, size_from_type);
    }

    bool has_result_value = !accumulator_name->IsWildcardLeaf();
    Expr* accumulator_value = unroll_for->init();
    for (uint64_t i = 0; i < size; i++) {
      if (has_result_value) {
        Let* accumulator =
            module_.Make<Let>(accumulator_name->span(), accumulator_name,
                              accumulator_type, accumulator_value, false);
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(accumulator, accumulator_type));
        unrolled_statements.push_back(module_.Make<Statement>(accumulator));
      }
      if (!iterator_name->IsWildcardLeaf()) {
        Let* iterator = nullptr;
        if (iterable_values && (*iterable_values)[i].FitsInUint64()) {
          Number* value =
              module_.Make<Number>(unroll_for->iterable()->span(),
                                   (*iterable_values)[i].ToString(true),
                                   NumberKind::kOther, iterator_type);
          XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(value, iterator_type));

          XLS_RETURN_IF_ERROR(
              ConvertSubtree(value, std::nullopt, parametric_context));
          XLS_ASSIGN_OR_RETURN(Type * value_type, ti->GetItemOrError(value));
          ti->SetItem(value, MetaType(value_type->CloneToUnique()));

          iterator = module_.Make<Let>(iterator_name->span(), iterator_name,
                                       iterator_type, value, false);
        } else {
          Expr* index = module_.Make<Number>(
              unroll_for->iterable()->span(), std::to_string(i),
              NumberKind::kOther,
              CreateU32Annotation(module_, unroll_for->span()), false);
          XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
              index, CreateU32Annotation(module_, unroll_for->span())));
          XLS_RETURN_IF_ERROR(
              ConvertSubtree(index, std::nullopt, parametric_context));
          XLS_ASSIGN_OR_RETURN(Type * index_type, ti->GetItemOrError(index));
          ti->SetItem(index, MetaType(index_type->CloneToUnique()));
          Expr* element =
              module_.Make<Index>(unroll_for->iterable()->span(),
                                  unroll_for->iterable(), index, false);
          iterator = module_.Make<Let>(iterator_name->span(), iterator_name,
                                       iterator_type, element, false);
        }
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(iterator, iterator_type));
        unrolled_statements.push_back(module_.Make<Statement>(iterator));
      }

      XLS_ASSIGN_OR_RETURN(
          AstNode * copy_body,
          table_.Clone(unroll_for->body(), &NoopCloneReplacer));
      StatementBlock* copy_body_statementblock =
          down_cast<StatementBlock*>(copy_body);
      absl::Span<Statement* const> statements =
          copy_body_statementblock->statements();
      if (has_result_value) {
        // If accumulator is not wildcard, we expect the last body statement to
        // be an expr updating the accumulator, which will be handled in the
        // next iteration.
        CHECK_GT(statements.size(), 0);
        unrolled_statements.insert(unrolled_statements.end(),
                                   statements.begin(), statements.end() - 1);
        accumulator_value = std::get<Expr*>(statements.back()->wrapped());
      } else {
        unrolled_statements.insert(unrolled_statements.end(),
                                   statements.begin(), statements.end());
      }
    }
    // Handle the final result of unroll_for! expr.
    if (has_result_value) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(accumulator_value, accumulator_type));
      Statement* last = module_.Make<Statement>(accumulator_value);
      unrolled_statements.push_back(last);
    }
    StatementBlock* unrolled_statement_block = module_.Make<StatementBlock>(
        unroll_for->span(), unrolled_statements, !has_result_value);

    XLS_RETURN_IF_ERROR(ConvertSubtree(unrolled_statement_block, std::nullopt,
                                       parametric_context));
    if (has_result_value) {
      absl::StatusOr<InterpValue> const_result =
          ConstexprEvaluator::EvaluateToValue(
              &import_data_, ti, &warning_collector_,
              GetParametricEnv(parametric_context),
              std::get<Expr*>(unrolled_statements.back()->wrapped()));
      if (const_result.ok()) {
        ti->NoteConstExpr(unroll_for, *const_result);
      }
    }

    ti->NoteUnrolledLoop(unroll_for, GetParametricEnv(parametric_context),
                         unrolled_statement_block);
    return absl::OkStatus();
  }

  absl::Status NoteFormatMacroVerbosityConstExpr(const FormatMacro* node,
                                                 const Type& type,
                                                 TypeInfo* ti) {
    if (!node->verbosity().has_value()) {
      return absl::OkStatus();
    }
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti, &warning_collector_, ParametricEnv(),
        *node->verbosity(), &type);
    if (!value.ok()) {
      return TypeInferenceErrorStatus(
          (*node->verbosity())->span(), &type,
          absl::Substitute("$0 verbosity values must be compile-time "
                           "constants; got `$1`.",
                           node->macro(), (*node->verbosity())->ToString()),
          file_table_);
    }
    absl::StatusOr<int64_t> value_as_int64 = value->GetBitValueViaSign();
    if (!value_as_int64.ok() || *value_as_int64 < 0) {
      return TypeInferenceErrorStatus(
          (*node->verbosity())->span(), &type,
          absl::Substitute(
              "$0 verbosity values must be positive integers; got `$1`.",
              node->macro(), (*node->verbosity())->ToString()),
          file_table_);
    }
    ti->NoteConstExpr(*node->verbosity(), *value);
    return absl::OkStatus();
  }

  // Constexpr-evaluates the given expression, whose dependencies must already
  // be noted as constexpr's in the `TypeInfo` corresponding to the scope for
  // the expression.
  absl::StatusOr<InterpValue> Evaluate(
      const ParametricContextScopedExpr& scoped_expr) override {
    VLOG(7) << "Evaluate: " << scoped_expr.expr()->ToString()
            << " with owner: " << scoped_expr.expr()->owner()->name()
            << " in module: " << module_.name()
            << " in context: " << ToString(scoped_expr.context());

    // Note that `scoped_expr` will not have a `context()` in a case like
    //  fn foo<X: u32>(...) { ... }
    //  fn bar() {
    //    foo<SOME_CONSTANT + 1>(...);
    //  }
    // The only scoped expr there is the expression being passed for `X`, which
    // is in a non-parametric caller and therefore cannot possibly refer to any
    // parametrics.
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * type_info,
        GetTypeInfo(scoped_expr.expr()->owner(), scoped_expr.context()));
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
    TypeSystemTrace trace = tracer_->TraceEvaluate(parametric_context, expr);

    // This is the type of the parametric binding we are talking about, which is
    // typically a built-in type, but the way we are concretizing it here would
    // support it being a complex type that even refers to other parametrics.
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         Concretize(type_annotation, parametric_context));
    if (!type_info->Contains(const_cast<Expr*>(expr))) {
      type_info->SetItem(expr, *type);
    }
    if (type_annotation->owner() == type_info->module()) {
      // Prevent bleed-over from a different module.
      type_info->SetItem(type_annotation, MetaType(type->CloneToUnique()));
    }
    // TODO: https://github.com/google/xls/issues/193 - The if-statement below
    // is here temporarily to enable easy testing of parametric variables in
    // inference_table_test. The equivalent is done by `TypecheckModuleV2`, and
    // that's where the logic belongs, but that doesn't yet deal with parametric
    // variables.
    if (expr->kind() == AstNodeKind::kNumber) {
      if (auto* number = down_cast<const Number*>(expr);
          number->type_annotation() != nullptr) {
        type_info->SetItem(number->type_annotation(),
                           MetaType(type->CloneToUnique()));
      }
    }

    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        ConstexprEvaluator::EvaluateToValue(
            &import_data_, type_info, &warning_collector_,
            GetParametricEnv(parametric_context), expr, /*type=*/nullptr));
    VLOG(7) << "Evaluation result for: " << expr->ToString()
            << " in context: " << ToString(parametric_context)
            << " value: " << result.ToString();
    return result;
  }

  absl::StatusOr<const FunctionAndTargetObject> ResolveFunction(
      const Expr* callee, std::optional<const Function*> caller_function,
      std::optional<const ParametricContext*> caller_context) override {
    const AstNode* function_node = nullptr;
    std::optional<Expr*> target_object;
    std::optional<const ParametricContext*> target_struct_context =
        caller_context.has_value() && (*caller_context)->is_struct()
            ? caller_context
            : std::nullopt;
    if (callee->kind() == AstNodeKind::kColonRef) {
      const auto* colon_ref = down_cast<const ColonRef*>(callee);
      std::optional<const AstNode*> target =
          table_.GetColonRefTarget(colon_ref);
      if (target.has_value()) {
        function_node = *target;
      }
    } else if (callee->kind() == AstNodeKind::kNameRef) {
      // Either a local function or a built-in function call.
      const auto* name_ref = down_cast<const NameRef*>(callee);
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
        function_node = def->definer();
      } else if (std::holds_alternative<BuiltinNameDef*>(
                     name_ref->name_def())) {
        if (module_.name() != kBuiltinStubsModuleName) {
          VLOG(5) << "ResolveFunction of builtin; delegating";
          XLS_ASSIGN_OR_RETURN(InferenceTableConverter * builtins_converter,
                               import_data_.GetInferenceTableConverter(
                                   std::string(kBuiltinStubsModuleName)));
          // Delegate to builtins converter.
          return builtins_converter->ResolveFunction(callee, caller_function,
                                                     caller_context);
        }
        // Look it up in our module
        BuiltinNameDef* def = std::get<BuiltinNameDef*>(name_ref->name_def());
        auto fn_name = def->identifier();
        std::optional<Function*> builtin_fn = module_.GetFunction(fn_name);
        if (builtin_fn.has_value()) {
          function_node = *builtin_fn;
        } else {
          return TypeInferenceErrorStatus(
              name_ref->span(), nullptr,
              absl::Substitute("Cannot find built-in method `$0`", fn_name),
              file_table_);
        }
      }
    } else if (callee->kind() == AstNodeKind::kAttr) {
      const auto* attr = down_cast<const Attr*>(callee);
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(attr->lhs(), caller_function, caller_context));
      target_object = attr->lhs();
      XLS_ASSIGN_OR_RETURN(
          std::optional<const TypeAnnotation*> target_object_type,
          resolver_->ResolveAndUnifyTypeAnnotationsForNode(caller_context,
                                                           *target_object));
      XLS_ASSIGN_OR_RETURN(
          std::optional<StructOrProcRef> struct_or_proc_ref,
          GetStructOrProcRef(*target_object_type, import_data_));
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

  // Given an invocation of the `map` builtin, creates a FunctionTypeAnnotation
  // for the `mapper` argument
  absl::StatusOr<const FunctionTypeAnnotation*> CreateMapperFunctionType(
      const Invocation* invocation,
      const ParametricContext* invocation_context) {
    std::vector<ExprOrType> explicit_parametrics =
        invocation->explicit_parametrics();
    if (!explicit_parametrics.empty()) {
      return ArgCountMismatchErrorStatus(
          invocation->span(),
          absl::Substitute(
              "Expected 0 parametric arguments to `map` but got $0.",
              explicit_parametrics.size()),
          file_table_);
    }

    Expr* array_arg = invocation->args()[0];
    std::optional<const ParametricContext*> caller_context =
        invocation_context->parent_context();
    XLS_RETURN_IF_ERROR(
        ConvertSubtree(array_arg, std::nullopt, caller_context));
    XLS_ASSIGN_OR_RETURN(std::optional<const TypeAnnotation*> array_type,
                         resolver_->ResolveAndUnifyTypeAnnotationsForNode(
                             caller_context, array_arg));
    CHECK(array_type.has_value());

    Expr* mapper = invocation->args()[1];
    FunctionRef* mapper_fn = dynamic_cast<FunctionRef*>(mapper);
    std::vector<ExprOrType> mapper_explicit_parametrics;
    if (mapper_fn != nullptr && !mapper_fn->explicit_parametrics().empty()) {
      mapper_explicit_parametrics = mapper_fn->explicit_parametrics();
      mapper = mapper_fn->callee();
    }

    // Create an invocation of the mapper with array[0], to make the
    // return type of mapper line up with the return type of `map`.
    XLS_ASSIGN_OR_RETURN(
        Number * index,
        MakeTypeCheckedNumber(module_, table_, array_arg->span(), 0,
                              CreateU32Annotation(module_, array_arg->span())));
    Expr* array_of_0 = module_.Make<Index>(array_arg->span(), array_arg, index);
    Invocation* mapper_invocation = module_.Make<Invocation>(
        mapper->span(), mapper, std::vector<Expr*>{array_of_0},
        std::move(mapper_explicit_parametrics));
    XLS_ASSIGN_OR_RETURN(
        const NameRef* mapper_invocation_var,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Invocation*>(invocation),
            absl::Substitute("internal_mapper_invocation_at_$0_in_$1",
                             invocation->span().ToString(file_table_),
                             invocation->owner()->name())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(mapper_invocation, mapper_invocation_var));

    std::unique_ptr<PopulateTableVisitor> visitor =
        CreatePopulateTableVisitor(&module_, &table_, &import_data_,
                                   /*typecheck_imported_module=*/nullptr);
    XLS_RETURN_IF_ERROR(visitor->PopulateFromInvocation(mapper_invocation));

    XLS_RETURN_IF_ERROR(
        ConvertSubtree(mapper_invocation, std::nullopt, caller_context));
    XLS_ASSIGN_OR_RETURN(std::optional<const TypeAnnotation*> mapper_type,
                         resolver_->ResolveAndUnifyTypeAnnotationsForNode(
                             caller_context, mapper_invocation->callee()));
    CHECK(mapper_type.has_value());
    return (*mapper_type)->AsAnnotation<FunctionTypeAnnotation>();
  }

  // Determines any implicit parametric values in the given `invocation`, and
  // generates its `ParametricEnv` in `converted_parametric_envs_`. Also
  // populates `parametric_value_exprs_` for the invocation.
  absl::StatusOr<ParametricEnv> GenerateParametricFunctionEnv(
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
      XLS_ASSIGN_OR_RETURN(new_values, InferImplicitFunctionParametrics(
                                           invocation_context, invocation,
                                           implicit_parametrics));
      implicit_parametrics.clear();
      values.merge(std::move(new_values));
      return absl::OkStatus();
    };

    std::vector<ExprOrType> explicit_parametrics =
        invocation->explicit_parametrics();
    if (IsBuiltinFn(invocation->callee()) &&
        invocation->callee()->ToString() == "map") {
      XLS_ASSIGN_OR_RETURN(
          const FunctionTypeAnnotation* mapper_fn_type,
          CreateMapperFunctionType(invocation, invocation_context));

      // These must be the first actual two parametrics because they're listed
      // as the first two formals.
      explicit_parametrics.push_back(
          const_cast<FunctionTypeAnnotation*>(mapper_fn_type));
      explicit_parametrics.push_back(
          const_cast<TypeAnnotation*>(mapper_fn_type->return_type()));
    }

    for (int i = 0; i < invocation_context->parametric_bindings().size(); i++) {
      const ParametricBinding* binding =
          invocation_context->parametric_bindings()[i];

      if (i < explicit_parametrics.size() &&
          binding->type_annotation()->IsAnnotation<GenericTypeAnnotation>()) {
        // This is a <T: type> reference
        ExprOrType actual_parametric_type = explicit_parametrics[i];
        XLS_RETURN_IF_ERROR(
            table_.AddTypeAnnotationToVariableForParametricContext(
                invocation_context, binding,
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
        if (binding->expr() != nullptr) {
          // Convert the default expr even when unused, or we will not notice if
          // it has type errors. While we don't fundamentally need to care if
          // that happens, v1 does check.
          XLS_RETURN_IF_ERROR(ConvertSubtree(binding->expr(), std::nullopt,
                                             invocation_context));
        }
        XLS_ASSIGN_OR_RETURN(InterpValue value, Evaluate(*expr));
        invocation_context->type_info()->NoteConstExpr(binding->name_def(),
                                                       value);
        values.emplace(binding->name_def()->identifier(), value);
      } else {
        implicit_parametrics.insert(binding);
      }
    }

    // Resolve any implicit ones that are at the end of the list.
    XLS_RETURN_IF_ERROR(infer_pending_implicit_parametrics());

    // Set concrete types for the parametric bindings in `TypeInfo` (downstream
    // code uses these for proc parametrics), and create the value exprs. This
    // is basically an alternate format for the `ParametricEnv` that is more
    // readily useful for scrubbing the parametrics from type annotations.
    absl::flat_hash_map<const NameDef*, ExprOrType> actual_parametrics;
    for (const auto& [name, value] : values) {
      const ParametricBinding* binding = bindings.at(name);
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> binding_type,
          Concretize(binding->type_annotation(), invocation_context));
      invocation_context->type_info()->SetItem(binding->name_def(),
                                               *binding_type);

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
    const Function& callee =
        *std::get<ParametricInvocationDetails>(invocation_context->details())
             .callee;
    if (callee.IsInProc()) {
      for (const ProcMember* member : (*callee.proc())->members()) {
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(member, std::nullopt, invocation_context));
      }
    }
    parametric_value_exprs_.emplace(invocation_context,
                                    std::move(actual_parametrics));
    ParametricEnv env(std::move(values));
    converted_parametric_envs_.emplace(invocation_context, env);
    return env;
  }

  // Attempts to infer the values of the specified implicit parametrics in an
  // invocation, using the types of the regular arguments being passed. If not
  // all of `implicit_parametrics` can be determined, this function returns an
  // error.
  absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
  InferImplicitFunctionParametrics(
      const ParametricContext* invocation_context, const Invocation* invocation,
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics) {
    VLOG(5) << "Inferring " << implicit_parametrics.size()
            << " implicit parametrics for invocation: "
            << ToString(invocation_context);
    const auto& context_data =
        std::get<ParametricInvocationDetails>(invocation_context->details());
    CHECK_NE(invocation, nullptr);
    const absl::Span<Param* const> formal_args = context_data.callee->params();
    const absl::Span<Expr* const> actual_args = invocation->args();
    TypeInfo* ti = invocation_context->type_info();
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
          param->type_annotation()->IsAnnotation<SelfTypeAnnotation>()) {
        continue;
      }
      formal_types.push_back(param->type_annotation());
    }

    XLS_ASSIGN_OR_RETURN(
        TypeInfo * actual_arg_ti,
        GetTypeInfo(&module_, invocation_context->parent_context()));
    return InferImplicitParametrics(
        invocation_context->parent_context(), invocation_context,
        implicit_parametrics, formal_types, actual_args, ti, actual_arg_ti,
        /*caller_type_annotation_filter=*/TypeAnnotationFilter::None(),
        /*pre_use_actual_arg=*/
        [&](const Expr* actual_arg) {
          // If an argument is essentially being used to figure out its own
          // formal type (due to the formal type depending on an implicit
          // parametric), then we need to convert the actual argument here,
          // before it is used to figure out the type. If it's having a known
          // formal type imposed on it, then `ConvertInvocation` will convert it
          // after deciding the formal type.
          TypeSystemTrace arg_trace =
              tracer_->TraceConvertActualArgument(actual_arg);
          return ConvertSubtree(actual_arg, context_data.caller,
                                invocation_context->parent_context(),
                                /*filter_param_type_annotations=*/true);
        });
  }

  // Attempts to infer the values of the specified implicit parametrics in an
  // invocation or struct instance, using the types of the regular arguments or
  // members being passed. If not all of `implicit_parametrics` can be
  // determined, this function returns an error. The
  // `caller_type_annotation_filter` allows the caller to filter some type
  // annotations of the actual arguments from consideration. The
  // `pre_use_actual_arg` callback allows the caller to be notified and do any
  // desired prework before some actual argument gets used to infer parametrics.
  absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
  InferImplicitParametrics(
      std::optional<const ParametricContext*> actual_arg_context,
      std::optional<const ParametricContext*> target_context,
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics,
      absl::Span<const TypeAnnotation* const> formal_types,
      absl::Span<Expr* const> actual_args, TypeInfo* output_ti,
      TypeInfo* actual_arg_ti,
      TypeAnnotationFilter caller_type_annotation_filter =
          TypeAnnotationFilter::None(),
      absl::FunctionRef<absl::Status(const Expr*)> pre_use_actual_arg =
          [](const Expr*) { return absl::OkStatus(); }) {
    TypeSystemTrace trace =
        tracer_->TraceInferImplicitParametrics(implicit_parametrics);
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
      TypeAnnotationFilter filter =
          caller_type_annotation_filter
              .Chain(TypeAnnotationFilter::FilterParamTypes())
              .Chain(TypeAnnotationFilter::FilterRefsWithMissingTypeInfo(
                  actual_arg_ti));
      XLS_RETURN_IF_ERROR(resolver_->ResolveIndirectTypeAnnotations(
          actual_arg_context, actual_arg_annotations, filter));
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
              actual_args[i]->span(), caller_type_annotation_filter));
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(actual_arg_type, std::nullopt, actual_arg_context));
      VLOG(5) << "Infer using actual type: " << actual_arg_type->ToString()
              << " and formal type: " << formal_types[i]->ToString()
              << " with effective context: " << ToString(actual_arg_context);

      absl::StatusOr<absl::flat_hash_map<const ParametricBinding*,
                                         InterpValueOrTypeAnnotation>>
          resolved = SolveForParametrics(
              actual_arg_type, formal_types[i], implicit_parametrics,
              [&](const TypeAnnotation* expected_type, const Expr* expr) {
                return Evaluate(ParametricContextScopedExpr(
                    actual_arg_context, expected_type, expr));
              });
      if (!resolved.ok()) {
        // `SolveForParametrics` does not yield user-presentable errors. When it
        // errors, it means the formal and actual argument types are not
        // reconcilable.
        XLS_ASSIGN_OR_RETURN(const TypeAnnotation* direct_actual_arg_type,
                             resolver_->ResolveIndirectTypeAnnotations(
                                 actual_arg_context, actual_arg_type,
                                 TypeAnnotationFilter::None()));
        XLS_ASSIGN_OR_RETURN(
            const TypeAnnotation* direct_formal_arg_type,
            resolver_->ResolveIndirectTypeAnnotations(
                target_context, formal_types[i], TypeAnnotationFilter::None()));
        return TypeMismatchError(actual_arg_context, direct_actual_arg_type,
                                 direct_formal_arg_type);
      }
      for (auto& [binding, value_or_type] : *resolved) {
        VLOG(5) << "Inferred implicit parametric value: "
                << ToString(value_or_type)
                << " for binding: " << binding->identifier()
                << " using function argument: `" << actual_args[i]->ToString()
                << "` of actual type: " << actual_arg_type->ToString();
        implicit_parametrics.erase(binding);
        if (std::holds_alternative<InterpValue>(value_or_type)) {
          InterpValue& value = std::get<InterpValue>(value_or_type);
          output_ti->NoteConstExpr(binding->name_def(), value);
          values.emplace(binding->identifier(), std::move(value));
        } else {
          CHECK(target_context.has_value());
          XLS_RETURN_IF_ERROR(
              table_.AddTypeAnnotationToVariableForParametricContext(
                  target_context, binding,
                  std::get<const TypeAnnotation*>(value_or_type)));
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
            parametric_context,
            CreateBoolAnnotation(*expr->owner(), expr->span()), expr)));
    return value.GetBitValueUnsigned();
  }

  absl::StatusOr<int64_t> EvaluateU32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) override {
    return Evaluate32BitIntOrExpr(parametric_context, value_or_expr,
                                  /*is_signed=*/false);
  }

  absl::StatusOr<int64_t> EvaluateS32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) override {
    return Evaluate32BitIntOrExpr(parametric_context, value_or_expr,
                                  /*is_signed=*/true);
  }

  absl::StatusOr<int64_t> Evaluate32BitIntOrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr, bool is_signed) {
    if (std::holds_alternative<int64_t>(value_or_expr)) {
      return std::get<int64_t>(value_or_expr);
    }
    const Expr* expr = std::get<const Expr*>(value_or_expr);

    XLS_RETURN_IF_ERROR(
        ConvertSubtree(expr, /*function=*/std::nullopt, parametric_context));

    std::optional<const TypeAnnotation*> type_annotation =
        table_.GetTypeAnnotation(expr);
    if (!type_annotation.has_value()) {
      type_annotation = is_signed
                            ? CreateS32Annotation(*expr->owner(), expr->span())
                            : CreateU32Annotation(*expr->owner(), expr->span());
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
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                         GetTypeInfo(struct_def.owner(), parent_context));
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * instance_type_info,
        import_data_.type_info_owner().New(struct_def.owner(), ti));
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
      absl::flat_hash_map<std::string, Expr*> actual_member_exprs_by_name;
      for (const auto& [name, expr] :
           (*instantiator_node)->GetOrderedMembers(&struct_def)) {
        actual_member_exprs_by_name.emplace(name, expr);
      }

      // If there are "splatted" members, i.e., implied copies from members of
      // an existing struct, we need synthetic longhand member initializers for
      // parametric inference.
      if (actual_member_exprs_by_name.size() != formal_member_types.size() &&
          (*instantiator_node)->kind() == AstNodeKind::kSplatStructInstance) {
        const auto* splat =
            down_cast<const SplatStructInstance*>(*instantiator_node);
        for (const StructMemberNode* member : struct_def.members()) {
          if (!actual_member_exprs_by_name.contains(member->name())) {
            XLS_ASSIGN_OR_RETURN(
                Expr * initializer,
                CreateInitializerForSplattedStructMember(*splat, *member));
            actual_member_exprs_by_name.emplace(member->name(), initializer);
          }
        }
      }

      // At this point we should have an `Expr` per formal member of the struct
      // definition.
      CHECK_EQ(actual_member_exprs_by_name.size(), formal_member_types.size());
      for (const StructMemberNode* member : struct_def.members()) {
        actual_member_exprs.push_back(
            actual_member_exprs_by_name.at(member->name()));
      }
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
              // When inferring a parametric using a member of the actual
              // struct, we may have e.g. a member with 2 annotations like
              // `decltype(Foo<N>.x)` and `uN[32]`. The decltype one in this
              // example is not useful for the inference of `N`, and more
              // generally, any decltype-ish annotation that refers back to
              // the struct we are processing is going to be unhelpful, so we
              // weed those out here.
              TypeAnnotationFilter::FilterReferencesToStruct(
                  &table_, parent_context, &struct_def, &import_data_)));

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

  // Creates a member initializer `Expr` for a splatted member of a struct
  // instance.
  //
  // For example, given a struct like:
  //   `Foo { a: u32, b: u8, c: u16 }`
  // and an instance expression like:
  //   `Foo { a: 5, ..x }`
  // the intent is `Foo { a: 5, b: x.b, c: x.c }`.
  //
  // It may be useful, e.g. for parametric inference, to actually have the `x.b`
  // and `x.c` expressions internally, so that the splatted struct instance can
  // be treated the same as a longhand one. This function creates that `Expr`
  // for the given `member`, populating the inference table as if it had been in
  // the source code.
  absl::StatusOr<Expr*> CreateInitializerForSplattedStructMember(
      const SplatStructInstance& instance, const StructMemberNode& member) {
    Expr* expr =
        module_.Make<Attr>(instance.span(), instance.splatted(), member.name());
    XLS_ASSIGN_OR_RETURN(
        const NameRef* type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, expr,
            absl::Substitute("internal_type_splatted_member_$0_at_$1_in_$2",
                             member.name(),
                             instance.span().ToString(file_table_),
                             instance.owner()->name()),
            member.type()));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(expr, type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(expr, member.type()));
    return expr;
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
        value_expr = down_cast<Expr*>(clone);
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
        XLS_ASSIGN_OR_RETURN(
            InterpValue value,
            (*struct_context)->type_info()->GetConstExpr(constant->name_def()));
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
            if (node->kind() == AstNodeKind::kTypeAnnotation &&
                down_cast<const TypeAnnotation*>(node)
                    ->IsAnnotation<SelfTypeAnnotation>()) {
              return const_cast<TypeAnnotation*>(*real_self_type);
            }
            return std::nullopt;
          });
    }
    XLS_ASSIGN_OR_RETURN(AstNode * clone,
                         table_.Clone(type, std::move(replacer)));
    return down_cast<const TypeAnnotation*>(clone);
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

  InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  TypeInfo* const base_type_info_;
  const FileTable& file_table_;
  std::unique_ptr<TypeSystemTracer> tracer_;
  std::unique_ptr<TypeAnnotationResolver> resolver_;
  std::unique_ptr<FastConcretizer> fast_concretizer_;
  absl::flat_hash_map<const ParametricContext*, ParametricEnv>
      converted_parametric_envs_;
  absl::flat_hash_map<const ParametricContext*,
                      absl::flat_hash_map<const NameDef*, ExprOrType>>
      parametric_value_exprs_;
  absl::flat_hash_map<std::optional<const ParametricContext*>,
                      absl::flat_hash_set<const AstNode*>>
      converted_subtrees_;

  // The top element in this stack is the proc-level type info for the proc (or
  // child of a proc) currently being converted, if any. There are only multiple
  // elements when dealing with a chain of spawns involving parametric procs,
  // since that leads us to analyze one proc while working on another.
  // When analyzing a function in a parametric proc, there will be a
  // `ParametricContext` for the invocation of the function, which contains its
  // own `TypeInfo` like any parametric invocation, and there will also be a
  // proc-level `TypeInfo` in this stack.
  std::stack<TypeInfo*> proc_type_info_stack_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<InferenceTableConverter>>
CreateInferenceTableConverter(InferenceTable& table, Module& module,
                              ImportData& import_data,
                              WarningCollector& warning_collector,
                              const FileTable& file_table,
                              std::unique_ptr<TypeSystemTracer> tracer) {
  VLOG(1) << "CreateInferenceTableConverter: module " << &module;
  VLOG(5) << "Inference table before conversion:";
  VLOG(5) << table.ToString();

  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data.type_info_owner().New(&module));
  return std::make_unique<InferenceTableConverterImpl>(
      table, module, import_data, warning_collector, type_info, file_table,
      std::move(tracer));
}

}  // namespace xls::dslx
