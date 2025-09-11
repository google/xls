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
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
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
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/semantics_analysis.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/constant_collector.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/fast_concretizer.h"
#include "xls/dslx/type_system_v2/flatten_in_type_order.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/populate_table_visitor.h"
#include "xls/dslx/type_system_v2/simplified_type_annotation_cache.h"
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

// Returns whether the type for the given node should be a `MetaType`, i.e. the
// node represents a type itself rather than an object of the type.
bool NeedsMetaType(const InferenceTable& table, const AstNode* node) {
  static const absl::NoDestructor<absl::flat_hash_set<AstNodeKind>>
      kMetaTypeKinds({AstNodeKind::kTypeAnnotation, AstNodeKind::kTypeAlias,
                      AstNodeKind::kEnumDef, AstNodeKind::kStructDef,
                      AstNodeKind::kProcDef});
  return kMetaTypeKinds->contains(node->kind()) ||
         (node->kind() == AstNodeKind::kColonRef &&
          IsColonRefWithTypeTarget(table, down_cast<const ColonRef*>(node))) ||
         (node->kind() == AstNodeKind::kNameDef && node->parent() &&
          node->parent()->kind() == AstNodeKind::kTypeAlias);
}

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
                                    public ParametricStructInstantiator {
 public:
  InferenceTableConverterImpl(
      InferenceTable& table, Module& module, ImportData& import_data,
      WarningCollector& warning_collector, TypeInfo* base_type_info,
      const FileTable& file_table, std::unique_ptr<TypeSystemTracer> tracer,
      std::unique_ptr<SemanticsAnalysis> semantics_analysis)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        base_type_info_(base_type_info),
        file_table_(file_table),
        tracer_(std::move(tracer)),
        semantics_analysis_(std::move(semantics_analysis)),
        evaluator_(CreateEvaluator(table_, module_, import_data_,
                                   warning_collector_, *this, *tracer_)),
        resolver_(TypeAnnotationResolver::Create(
            module, table, file_table,
            /*error_generator=*/*this, *evaluator_,
            /*parametric_struct_instantiator=*/*this, *tracer_,
            warning_collector_, import_data_, simplified_type_annotation_cache_,
            [&](std::optional<const ParametricContext*> parametric_context,
                const Invocation* invocation) {
              return TryConvertInvocationForUnification(parametric_context,
                                                        invocation);
            })),
        constant_collector_(CreateConstantCollector(
            table_, module_, import_data_, warning_collector_, file_table_,
            /*converter=*/*this, *evaluator_,
            /*parametric_struct_instantiator=*/*this, *tracer_)),
        fast_concretizer_(FastConcretizer::Create(file_table)) {}

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

    // Push the appropriate proc type info onto the stack, if any. If we do
    // this, then this `proc_type_info_frame` variable becomes the owner of the
    // stack frame.
    std::unique_ptr<ProcTypeInfoFrame> proc_type_info_frame;
    if (node->kind() == AstNodeKind::kProc ||
        (node->kind() == AstNodeKind::kFunction &&
         down_cast<const Function*>(node)->IsInProc())) {
      const Proc* proc = node->kind() == AstNodeKind::kProc
                             ? down_cast<const Proc*>(node)
                             : *down_cast<const Function*>(node)->proc();
      if (!proc->IsParametric() && !converted_procs_.insert(proc).second) {
        return absl::OkStatus();
      }
      if (!IsProcAtTopOfTypeInfoStack(proc)) {
        XLS_ASSIGN_OR_RETURN(proc_type_info_frame, PushProcTypeInfo(proc));
      }
    }

    XLS_ASSIGN_OR_RETURN(
        std::vector<const AstNode*> nodes,
        FlattenInTypeOrder(
            import_data_, node,
            /*include_parametric_entities=*/parametric_context.has_value() &&
                (node == function ||
                 (node->parent() != nullptr &&
                  node->parent()->kind() == AstNodeKind::kImpl))));
    for (const AstNode* node : nodes) {
      VLOG(5) << "Next node: " << node->ToString();
      if (node->kind() == AstNodeKind::kInvocation) {
        XLS_RETURN_IF_ERROR(ConvertInvocation(
            down_cast<const Invocation*>(node), parametric_context));
      } else if (node->kind() == AstNodeKind::kProc &&
                 !IsProcAtTopOfTypeInfoStack(down_cast<const Proc*>(node))) {
        // When we encounter a proc root, do a dedicated `ConvertSubtree` call
        // for the proc, targeted to the converter for its owning module. This
        // gets the proc's type info onto the appropriate stack. Note that
        // `ConversionOrderVisitor` treats proc roots as a "break point" and
        // only descends into a proc if it is the root of the subtree.
        InferenceTableConverter* converter = this;
        if (node->owner() != &module_) {
          XLS_ASSIGN_OR_RETURN(
              converter,
              import_data_.GetInferenceTableConverter(node->owner()));
        }
        XLS_RETURN_IF_ERROR(
            converter->ConvertSubtree(node, std::nullopt, parametric_context));
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
  // output type info.
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
      // Note that if a parametric function `g` is invoked by a default
      // parametric expr for a function `f`, there is a chicken-and-egg problem
      // where we cannot have produced the `ParametricEnv` for `f` at the time
      // we deal with the `g` invocation. However, because such invocations boil
      // down to constants without being converted to IR, we don't care if we
      // properly key the `TypeInfo` for that `g` invocation for access by IR
      // conversion (and v1 does not achieve that either).

      parent_env =
          table_.GetParametricEnv(*parametric_context->parent_context());
    }
    ParametricEnv callee_env = table_.GetParametricEnv(parametric_context);
    VLOG(5) << "Parent env: " << parent_env.ToString();
    VLOG(5) << "Callee env: " << callee_env.ToString();
    CHECK_NE(invocation, nullptr);

    XLS_ASSIGN_OR_RETURN(TypeInfo * parent_ti,
                         GetTypeInfo(invocation->owner(), parametric_context));
    XLS_RETURN_IF_ERROR(parent_ti->AddInvocationTypeInfo(
        *invocation, data.callee,
        data.caller.has_value() ? *data.caller : nullptr, parent_env,
        callee_env,
        IsBuiltin(data.callee) ? nullptr : parametric_context->type_info()));
    return absl::OkStatus();
  }

  // Make sure binding and parametric match - both values or both types
  absl::Status ValidateParametricsAgainstBindings(
      const std::vector<ParametricBinding*>& formal_bindings,
      const std::vector<ExprOrType>& explicit_parametrics) {
    int i = 0;
    for (ExprOrType parametric : explicit_parametrics) {
      if (i >= formal_bindings.size()) {
        break;
      }
      const ParametricBinding* binding = formal_bindings.at(i);
      bool formal_is_type_parametric =
          binding->type_annotation()->IsAnnotation<GenericTypeAnnotation>();
      if (formal_is_type_parametric) {
        if (std::holds_alternative<Expr*>(parametric) &&
            !IsColonRefWithTypeTarget(table_, std::get<Expr*>(parametric))) {
          const AstNode* expr = ToAstNode(parametric);
          return TypeInferenceErrorStatus(
              *expr->GetSpan(), nullptr,
              absl::Substitute("Expected parametric type, saw `$0`",
                               expr->ToString()),
              file_table_);
        }
      } else if (std::holds_alternative<TypeAnnotation*>(parametric) ||
                 IsColonRefWithTypeTarget(table_,
                                          std::get<Expr*>(parametric))) {
        const AstNode* type = ToAstNode(parametric);
        return TypeInferenceErrorStatus(
            *type->GetSpan(), nullptr,
            absl::Substitute("Expected parametric value, saw `$0`",
                             type->ToString()),
            file_table_);
      }
      i++;
    }
    return absl::OkStatus();
  }

  // Checks whether the callee is either a builtin requiring implicit token or
  // has been noted in type info that it requires implicit token. Notes whether
  // implicit token is required in the caller's root type info.
  absl::Status NoteIfRequiresImplicitToken(
      std::optional<const Function*> caller, const Function* callee_fn,
      Expr* callee) {
    if (!caller.has_value()) {
      return absl::OkStatus();
    }

    // ImplicitToken notation always happens at the `root` level, so the context
    // is not relevant when retrieving type info.
    XLS_ASSIGN_OR_RETURN(TypeInfo * callee_ti,
                         GetTypeInfo(callee_fn->owner(), std::nullopt));
    std::optional<bool> callee_noted_requires_token =
        callee_ti->GetRequiresImplicitToken(*callee_fn);
    bool callee_requires_implicit_token =
        (IsBuiltin(callee_fn) && GetBuiltinFnRequiresImplicitToken(callee)) ||
        (callee_noted_requires_token.has_value() &&
         *callee_noted_requires_token);
    if (callee_requires_implicit_token) {
      XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                           GetTypeInfo((*caller)->owner(), std::nullopt));
      std::optional<bool> already_required =
          ti->GetRequiresImplicitToken(**caller);
      if (!already_required.has_value() || !*already_required) {
        ti->NoteRequiresImplicitToken(**caller, callee_requires_implicit_token);
      }
    }
    return absl::OkStatus();
  }

  // This is invoked when the `TypeAnnotationResolver` is asked to resolve and
  // unify a type variable that has an invocation feeding it. The need for this
  // is due to the fact that `ConvertInvocation` actually adds type annotations
  // to the table after function resolution. If a variable is unified before the
  // invocations feeding it have been converted, the table will not yet contain
  // the type annotations provided by those invocations, so the unification will
  // consider partial information that may not be enough.
  absl::Status TryConvertInvocationForUnification(
      std::optional<const ParametricContext*> caller_context,
      const Invocation* invocation) {
    if (converted_invocations_[caller_context].contains(invocation)) {
      return absl::OkStatus();
    }

    // If the resolver tries to pre-emptively convert an invocation in a struct
    // context, that actually can't be done and should not be necessary. Any
    // invocation, even of an impl member, must have an invocation context or
    // nullopt as the caller.
    // TODO: https://github.com/google/xls/issues/2379 - See if we can prevent
    // these calls further upstream, e.g. via tweaks to populate-table logic.
    if (caller_context.has_value() && !(*caller_context)->is_invocation()) {
      return absl::OkStatus();
    }

    TypeSystemTrace trace = tracer_->TraceConvertInvocation(
        invocation, caller_context,
        /*convert_for_type_variable_unification=*/true);
    XLS_RETURN_IF_ERROR(
        ConvertConstantsReferencedUnder(invocation, caller_context));
    return ConvertInvocation(invocation, caller_context);
  }

  // Converts the constants that are referenced under `node`. This is done only
  // as part of `TryConvertInvocationForUnification`.
  //
  // Certain expression types, that may be function arguments, require constexpr
  // values noted ahead of time. In this example:
  //   const bar = u32:5;
  //   foo(baz[bar:])
  //
  // the constexpr value for `bar` must be noted by the time we convert the
  // slice expression, and therefore if we are converting the whole `foo(...)`
  // subtree out of AST order to unify a type variable, we will have a problem.
  // Note that the problem would not arise with simply `foo(bar)`, because the
  // constexpr value of `bar` would then be irrelevant to deciding any
  // expression type involved.
  //
  // To handle such cases, the logic here pre-converts constant definitions that
  // an invocation depends on, which do not already have constexpr values noted.
  absl::Status ConvertConstantsReferencedUnder(
      const AstNode* node,
      std::optional<const ParametricContext*> parametric_context) {
    std::vector<std::pair<const NameRef*, const NameDef*>> references;
    XLS_ASSIGN_OR_RETURN(references,
                         CollectReferencedUnder(node, /*want_types=*/true));
    for (const auto& [name_ref, name_def] : references) {
      // Avoid the callee node, because its conversion is done explicitly upon
      // function resolution during the conversion of the invocation, and we
      // cannot correctly do it separately. We also leave the name ref alone,
      // and only convert the definer of the referenced name. As in general with
      // function arguments, the name ref may depend on parametric inference for
      // the invocation, and therefore its conversion can't be done up front
      // (nor does it need to be).
      if (node->kind() == AstNodeKind::kInvocation &&
          down_cast<const Invocation*>(node)->callee() == name_ref) {
        continue;
      }

      XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                           GetTypeInfo(name_def->owner(), parametric_context));

      // If the name already has a constexpr value, it will not pose a problem.
      if (ti->GetConstExprOption(name_def).has_value()) {
        continue;
      }

      // Dig up the actual declaration, e.g. the `let x = ...;` node for a node
      // like the `x` under that.
      const AstNode* decl = name_def->parent();
      while (decl != nullptr && decl->kind() == AstNodeKind::kNameDefTree &&
             decl->parent() != nullptr) {
        decl = decl->parent();
      }

      // Recursively convert that declaration's deps and itself.
      if (decl != nullptr && (decl->kind() == AstNodeKind::kConstantDef ||
                              decl->kind() == AstNodeKind::kLet)) {
        XLS_RETURN_IF_ERROR(
            ConvertConstantsReferencedUnder(decl, parametric_context));
        XLS_RETURN_IF_ERROR(
            ConvertSubtree(decl, std::nullopt, parametric_context));
      }
    }

    return absl::OkStatus();
  }

  // Converts the type info for the given invocation node and its argument
  // nodes. This involves resolving the callee function and applying the formal
  // types of the arguments to the actual arguments in the inference table.
  absl::Status ConvertInvocation(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context) {
    TypeSystemTrace trace = tracer_->TraceConvertInvocation(
        invocation, caller_context,
        /*convert_for_type_variable_unification=*/std::nullopt);
    converted_invocations_[caller_context].insert(invocation);
    std::optional<const Function*> caller = GetContainingFunction(invocation);
    VLOG(5) << "Converting invocation: " << invocation->callee()->ToString()
            << " with module: " << invocation->callee()->owner()->name()
            << " in module: " << module_.name()
            << " in context: " << ToString(caller_context);
    XLS_ASSIGN_OR_RETURN(
        const FunctionAndTargetObject function_and_target_object,
        ResolveFunction(invocation->callee(), caller, caller_context));
    table_.SetCalleeInCallerContext(invocation, caller_context,
                                    function_and_target_object.function);

    const Function* function = function_and_target_object.function;
    std::optional<const ParametricContext*> caller_or_target_struct_context =
        function_and_target_object.target_struct_context.has_value()
            ? function_and_target_object.target_struct_context
            : caller_context;

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

    // Colon refs to types can be masquerading as value `Expr`s in the
    // arguments. We need to error if there are any of these.
    for (const Expr* arg : actual_args) {
      if (arg->kind() == AstNodeKind::kColonRef &&
          IsColonRefWithTypeTarget(table_, arg)) {
        return TypeInferenceErrorStatus(
            arg->span(), nullptr, "Cannot pass a type as a function argument.",
            file_table_);
      }
    }

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

      XLS_RETURN_IF_ERROR(GenerateTypeInfo(caller_or_target_struct_context,
                                           invocation->callee()));
      // For non-parametric functions, the formal argument types can be taken at
      // face value. Apply them to the actual arguments, convert them, and
      // convert the invocation itself. We use the unified signature rather than
      // the `Function` object for this, because the `Function` may have struct
      // parametrics in it which are outside their domain here, and the unified
      // signature will not.
      XLS_ASSIGN_OR_RETURN(
          std::optional<const TypeAnnotation*> signature,
          resolver_->ResolveAndUnifyTypeAnnotationsForNode(
              caller_or_target_struct_context, invocation->callee()));
      XLS_RET_CHECK(signature.has_value());
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
      XLS_ASSIGN_OR_RETURN(TypeInfo * parent_ti,
                           GetTypeInfo(invocation->owner(), caller_context));
      XLS_RETURN_IF_ERROR(parent_ti->AddInvocation(
          *invocation, function, caller.has_value() ? *caller : nullptr));
      if (invocation->originating_invocation().has_value()) {
        XLS_ASSIGN_OR_RETURN(
            parent_ti,
            GetTypeInfo((*invocation->originating_invocation())->owner(),
                        caller_context));
        XLS_RETURN_IF_ERROR(parent_ti->AddInvocation(
            **invocation->originating_invocation(), function,
            caller.has_value() ? *caller : nullptr));
      }
      XLS_RETURN_IF_ERROR(NoteIfRequiresImplicitToken(
          caller, function_and_target_object.function, invocation->callee()));
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
      if (std::holds_alternative<Expr*>(explicit_parametric) &&
          !IsColonRefWithTypeTarget(table_,
                                    std::get<Expr*>(explicit_parametric))) {
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

    // Skip adding type info for `map` invocations because the invocation type
    // info expected by IR conversion is that for the passed in mapper function,
    // which is generated by `CreateMapperFunctionType`.
    if (!IsMapInvocation(invocation)) {
      XLS_RETURN_IF_ERROR(
          AddInvocationTypeInfo(invocation, invocation_context));
      if (invocation->originating_invocation().has_value()) {
        XLS_RETURN_IF_ERROR(AddInvocationTypeInfo(
            *invocation->originating_invocation(), invocation_context));
      }
    }

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
      absl::flat_hash_map<const NameDef*, ExprOrType> value_exprs;
      XLS_ASSIGN_OR_RETURN(value_exprs,
                           table_.GetParametricValueExprs(invocation_context));
      XLS_ASSIGN_OR_RETURN(const TypeAnnotation* parametric_free_type,
                           GetParametricFreeType(
                               CreateFunctionTypeAnnotation(module_, *function),
                               value_exprs, invocation_context->self_type(),
                               /*clone_if_no_parametrics=*/true));
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(parametric_free_type, caller, caller_context));

      XLS_ASSIGN_OR_RETURN(parametric_free_type,
                           resolver_->ResolveIndirectTypeAnnotations(
                               invocation_context, parametric_free_type,
                               TypeAnnotationFilter::None()));

      // In a context such as a parametric proc, where parametric-dependent type
      // aliases may be used in a function signature, the resolution of indirect
      // annotations in the signature may introduce parametrics, which we also
      // want to get rid of. For example, `(value_t) -> value_t` is overtly
      // parametric-free but could resolve to `(uN[N]) -> uN[N]` if `value_t` is
      // a parametric proc-level alias. We then need to replace the resulting
      // `N`s.
      XLS_ASSIGN_OR_RETURN(
          parametric_free_type,
          GetParametricFreeType(parametric_free_type, value_exprs,
                                invocation_context->self_type(),
                                /*clone_if_no_parametrics=*/false));

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

    // Note that the callee node does not need type info if it is a non-impl
    // proc function, and generating it would be complicated due to the possible
    // use of proc-level parametrics. Unlike with impl-style member functions,
    // we don't have a target struct context for the proc.
    if (!function->IsInProc()) {
      XLS_RETURN_IF_ERROR(GenerateTypeInfo(caller_or_target_struct_context,
                                           invocation->callee()));
    }

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
    if (!canonicalized) {
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(function, function, invocation_context));
    }
    XLS_RETURN_IF_ERROR(NoteIfRequiresImplicitToken(
        caller, function_and_target_object.function, invocation->callee()));
    return GenerateTypeInfo(caller_context, invocation);
  }

  // Gets the output `TypeInfo` corresponding to the given
  // `parametric_context`, which may be `nullopt`, in which case it returns
  // the base type info. If the proposed type info doesn't belong to the
  // provided module, its root type info is returned.
  absl::StatusOr<TypeInfo*> GetTypeInfo(
      const Module* module,
      std::optional<const ParametricContext*> parametric_context) override {
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
    if (node->kind() == AstNodeKind::kModule ||
        node->kind() == AstNodeKind::kTypeRef) {
      return absl::OkStatus();
    }

    // We don't need to concretize a type reference that is "abstract" in the
    // sense of leaving some parametric bindings of the underlying type
    // unspecified. There is also no way to do so with v2's avoidance of
    // `TypeDim`. Only uses of the alias, which provide the parametrics, can be
    // concretized.
    XLS_ASSIGN_OR_RETURN(const bool is_abstract_ref,
                         IsReferenceToAbstractType(node, import_data_, table_));
    if (is_abstract_ref) {
      return absl::OkStatus();
    }

    TypeSystemTrace trace = tracer_->TraceConvertNode(node);
    VLOG(5) << "GenerateTypeInfo for node: " << node->ToString()
            << " of kind: `" << AstNodeKindToString(node->kind()) << "`"
            << " with owner: " << node->owner()->name()
            << " for module: " << module_.name()
            << " in parametric context: " << ToString(parametric_context);
    if (pre_unified_type.has_value()) {
      VLOG(5) << "Using pre-unified type: " << (*pre_unified_type)->ToString();
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
    std::unique_ptr<TypeAnnotationResolver> foreign_resolver;
    TypeAnnotationResolver* resolver = resolver_.get();
    if (parametric_context.has_value() &&
        (*parametric_context)->is_invocation()) {
      auto details = std::get<ParametricInvocationDetails>(
          (*parametric_context)->details());
      if (details.callee->owner() != &module_) {
        foreign_resolver = TypeAnnotationResolver::Create(
            *details.callee->owner(), table_, file_table_,
            /*error_generator=*/*this, /*evaluator=*/*evaluator_,
            /*parametric_struct_instantiator=*/*this, *tracer_,
            warning_collector_, import_data_, simplified_type_annotation_cache_,
            [&](std::optional<const ParametricContext*> parametric_context,
                const Invocation* invocation) {
              return TryConvertInvocationForUnification(parametric_context,
                                                        invocation);
            });
        resolver = foreign_resolver.get();
      }
    }
    if (!annotation.has_value()) {
      if (node->kind() == AstNodeKind::kTypeAnnotation) {
        // If the node itself is a `TypeAnnotation`, we can usually use it as
        // is.
        node_is_annotation = true;
        annotation = down_cast<const TypeAnnotation*>(node);

        // An annotation that is a generic argument, e.g.
        // `element_count<annotation>()`, may be an internally-fabricated
        // indirect one that needs resolution. These cannot be written in DSLX
        // code.
        if (node->parent() != nullptr &&
            node->parent()->kind() == AstNodeKind::kInvocation) {
          XLS_ASSIGN_OR_RETURN(
              annotation, resolver->ResolveAndUnifyTypeAnnotations(
                              parametric_context, {*annotation},
                              (*annotation)->span(), type_annotation_filter,
                              table_.GetAnnotationFlag(*annotation)
                                  .HasFlag(TypeInferenceFlag::kBitsLikeType)));
        }

        // Builtin fragments of bits-like types, like `uN` and `xN[true]` have
        // no `Type` conversion. We only convert the complete thing.
        if (IsBitsLikeFragment(*annotation)) {
          return absl::OkStatus();
        }
      } else {
        // In most cases, come up with a unified type annotation using the
        // table (or the table of the owning module).
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
                       AstNodeKind::kTypeAnnotation,
                   node);

    if (!type.ok()) {
      // When the node itself is an annotation, and we decide to concretize
      // the node itself, we can't succeed in all contexts. For example, a
      // parametric-dependent field type declaration inside a parametric struct
      // declaration can't just be concretized like that. Rather than trying to
      // identify such cases, we just consider such nodes best-effort.
      if (node_is_annotation) {
        return absl::OkStatus();
      }
      return type.status();
    }

    // Mark NameDef's concretized type to tell if it is actually unused.
    if (semantics_analysis_) {
      if (node->kind() == AstNodeKind::kNameDef) {
        semantics_analysis_->SetNameDefType(down_cast<const NameDef*>(node),
                                            type->get());
      }
    }

    XLS_RETURN_IF_ERROR(
        ValidateConcreteType(table_, parametric_context, node, type->get(), *ti,
                             warning_collector_, import_data_, file_table_));

    if (NeedsMetaType(table_, node)) {
      MetaType meta_type((*type)->CloneToUnique());
      ti->SetItem(node, meta_type);
    } else {
      ti->SetItem(node, **type);
    }

    trace.SetResult(**type);
    XLS_RETURN_IF_ERROR(constant_collector_->CollectConstants(
        parametric_context, node, **type, ti));
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
      const StructOrProcRef& ref, const AstNode* node) final {
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
        InferenceTable::StructContextResult lookup_result,
        table_.GetOrCreateParametricStructContext(
            ref.def, node, parametric_env, CreateStructAnnotation(module_, ref),
            type_info_factory));
    const ParametricContext* struct_context = lookup_result.context;
    if (!lookup_result.created_new) {
      return struct_context;
    }
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
            evaluator_->Evaluate(ParametricContextScopedExpr(
                struct_context, binding->type_annotation(), binding->expr())));
      }

      VLOG(6) << "Setting binding: " << binding->identifier()
              << " in context: " << ToString(struct_context)
              << " to value: " << value->ToString();

      ti->SetItem(binding->name_def(), *binding_type);
      ti->NoteConstExpr(binding->name_def(), *value);
    }
    table_.SetParametricValueExprs(struct_context, std::move(value_exprs));
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
        if (binding->expr() == nullptr) {
          missing_parametric_names.push_back(binding->identifier());
        }
        continue;
      }
      ExprOrType parametric = actual_parametrics[i];
      if (std::holds_alternative<Expr*>(parametric)) {
        VLOG(6) << "Actual parametric: " << binding->identifier()
                << " expr: " << std::get<Expr*>(parametric)->ToString();
        XLS_ASSIGN_OR_RETURN(InterpValue value,
                             evaluator_->Evaluate(ParametricContextScopedExpr(
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
    if (!missing_parametric_names.empty()) {
      return TypeInferenceErrorStatus(
          error_span, /*type=*/nullptr,
          absl::Substitute(
              "Could not infer parametric(s) for instance of struct $0: $1",
              struct_def.identifier(),
              absl::StrJoin(missing_parametric_names, ", ")),
          file_table_);
    }
    return ParametricEnv(values);
  }

  // Returns the resulting base type info for the entire conversion.
  TypeInfo* GetBaseTypeInfo() override { return base_type_info_; }

  absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context,
      bool needs_conversion_before_eval,
      std::optional<const AstNode*> node) override {
    TypeSystemTrace trace = tracer_->TraceConcretize(annotation);
    VLOG(5) << "Concretize: " << annotation->ToString()
            << " in context invocation: " << ToString(parametric_context);
    VLOG(5) << "Effective context: " << ToString(parametric_context);

    absl::StatusOr<std::unique_ptr<Type>> type =
        fast_concretizer_->Concretize(annotation);
    if (type.ok()) {
      // If resulting type is bits-like, we fabricate a simplified type
      // annotation for the node.
      if (node) {
        if (BitsType* bits_type = dynamic_cast<BitsType*>(&**type)) {
          simplified_type_annotation_cache_.MaybeAddBitsLikeTypeAnnotation(
              module_, parametric_context, *node, bits_type->is_signed(),
              *(bits_type->size().GetAsInt64()));
        }
      }
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
      return absl::InvalidArgumentError(absl::Substitute(
          "Attempting to concretize `Any` type in module $0, which means there "
          "was insufficient type info.",
          annotation->owner()->name()));
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
      XLS_ASSIGN_OR_RETURN(int64_t size, evaluator_->EvaluateU32OrExpr(
                                             parametric_context, array->dim()));
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
          XLS_ASSIGN_OR_RETURN(int64_t size, evaluator_->EvaluateU32OrExpr(
                                                 parametric_context, dim));
          type = std::make_unique<ArrayType>(
              std::move(type), TypeDim(InterpValue::MakeU32(size)));
        }
      }
      return type;
    }
    XLS_ASSIGN_OR_RETURN(std::optional<const EnumDef*> enum_def,
                         GetEnumDef(annotation, import_data_));
    if (enum_def.has_value()) {
      std::unique_ptr<Type> cached_type =
          GetCachedType(*enum_def, std::nullopt);
      if (cached_type) {
        return cached_type;
      }
      XLS_ASSIGN_OR_RETURN(
          TypeInfo * ti, GetTypeInfo((*enum_def)->owner(), parametric_context));
      std::vector<InterpValue> members;
      BitsType* underlying_type = nullptr;

      // Handle special case: if the enum has no values, its explicitly
      // annotated type should be used.
      if ((*enum_def)->values().empty()) {
        XLS_RET_CHECK((*enum_def)->type_annotation());
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
                table_.GetParametricEnv(parametric_context), value.value);
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
      std::unique_ptr<Type> type =
          std::make_unique<EnumType>(**enum_def, underlying_type->size(),
                                     underlying_type->is_signed(), members);
      XLS_RETURN_IF_ERROR(AddCachedType(*enum_def, std::nullopt, *type));
      return type;
    }
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc,
                         GetStructOrProcRef(annotation, import_data_));
    if (struct_or_proc.has_value()) {
      const StructDefBase* struct_def_base = struct_or_proc->def;
      XLS_RET_CHECK(struct_def_base != nullptr);
      std::vector<std::unique_ptr<Type>> member_types;
      member_types.reserve(struct_def_base->members().size());
      std::optional<const ParametricContext*> struct_context;
      if (struct_def_base->IsParametric()) {
        XLS_ASSIGN_OR_RETURN(struct_context, GetOrCreateParametricStructContext(
                                                 parametric_context,
                                                 *struct_or_proc, annotation));
      }
      std::unique_ptr<Type> cached_type =
          GetCachedType(struct_def_base, struct_context);
      if (cached_type) {
        return cached_type;
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
        std::unique_ptr<Type> type = std::make_unique<StructType>(
            std::move(member_types),
            *down_cast<const StructDef*>(struct_def_base));
        XLS_RETURN_IF_ERROR(
            AddCachedType(struct_def_base, struct_context, *type));
        return type;
      }
      std::unique_ptr<Type> type = std::make_unique<ProcType>(
          std::move(member_types), *down_cast<const ProcDef*>(struct_def_base));
      XLS_RETURN_IF_ERROR(
          AddCachedType(struct_def_base, struct_context, *type));
      return type;
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
        evaluator_->EvaluateBoolOrExpr(parametric_context,
                                       signedness_and_bit_count.signedness));
    XLS_ASSIGN_OR_RETURN(
        int64_t bit_count,
        evaluator_->EvaluateU32OrExpr(parametric_context,
                                      signedness_and_bit_count.bit_count));
    if (node) {
      // If resulting type is bits-like, we fabricate a simplified type
      // annotation for the node.
      simplified_type_annotation_cache_.MaybeAddBitsLikeTypeAnnotation(
          module_, parametric_context, *node, signedness, bit_count);
    }
    VLOG(5) << "Concretized: " << annotation->ToString()
            << " to signed: " << signedness << ", bit count: " << bit_count;
    return std::make_unique<BitsType>(signedness, bit_count);
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

      // Disallow the form `module.fn()`. If they really want to do that, it
      // should be `module::fn()`.
      if (attr->lhs()->kind() == AstNodeKind::kNameRef &&
          IsImportedModuleReference(down_cast<NameRef*>(attr->lhs()))) {
        return TypeInferenceErrorStatus(
            attr->span(), nullptr,
            "An invocation callee must be a function, with a possible scope.",
            file_table_);
      }

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
      XLS_RET_CHECK(impl.has_value());
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

 private:
  bool IsMapInvocation(const Invocation* node) {
    return IsBuiltinFn(node->callee()) && node->callee()->ToString() == "map";
  }

  // Convenience function that assumes the passed in annotation does not need
  // subtree conversion.
  absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context) {
    return Concretize(annotation, parametric_context,
                      /*needs_conversion_before_eval=*/false, std::nullopt);
  }

  // Given an invocation of the `map` builtin, creates a FunctionTypeAnnotation
  // for the `mapper` argument
  absl::StatusOr<const FunctionTypeAnnotation*> CreateMapperFunctionType(
      const std::optional<const Function*> caller, const Invocation* invocation,
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
    XLS_RETURN_IF_ERROR(ConvertSubtree(array_arg, caller, caller_context));
    XLS_ASSIGN_OR_RETURN(std::optional<const TypeAnnotation*> array_type,
                         resolver_->ResolveAndUnifyTypeAnnotationsForNode(
                             caller_context, array_arg));
    XLS_RET_CHECK(array_type.has_value());

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
        std::move(mapper_explicit_parametrics), /*in_parens=*/false,
        invocation);
    if (caller.has_value()) {
      // ConvertInvocation figures out the caller by querying the AST. We need
      // it to believe the fake invocation is in the caller of `map()`, in order
      // for it to correctly set up the implicit token requirements.
      mapper_invocation->SetParentNonLexical(const_cast<Function*>(*caller));
    }
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
        ConvertSubtree(mapper_invocation, caller, caller_context));
    XLS_ASSIGN_OR_RETURN(std::optional<const TypeAnnotation*> mapper_type,
                         resolver_->ResolveAndUnifyTypeAnnotationsForNode(
                             caller_context, mapper_invocation->callee()));
    XLS_RET_CHECK(mapper_type.has_value());
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
    if (IsMapInvocation(invocation)) {
      XLS_ASSIGN_OR_RETURN(
          const FunctionTypeAnnotation* mapper_fn_type,
          CreateMapperFunctionType(std::get<ParametricInvocationDetails>(
                                       invocation_context->details())
                                       .caller,
                                   invocation, invocation_context));

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
        const TypeAnnotation* type;
        if (std::holds_alternative<TypeAnnotation*>(actual_parametric_type)) {
          type = std::get<TypeAnnotation*>(actual_parametric_type);
        } else {
          Expr* expr = std::get<Expr*>(actual_parametric_type);
          XLS_RET_CHECK(expr->kind() == AstNodeKind::kColonRef);
          std::optional<const TypeAnnotation*> colon_ref_annotation =
              table_.GetTypeAnnotation(expr);
          XLS_RET_CHECK(colon_ref_annotation.has_value());
          type = *colon_ref_annotation;
        }

        XLS_ASSIGN_OR_RETURN(
            TypeInfo * actual_arg_ti,
            GetTypeInfo(&module_, invocation_context->parent_context()));

        XLS_ASSIGN_OR_RETURN(type, CleanseGenericTypeArgument(
                                       invocation_context->parent_context(),
                                       *actual_arg_ti, type));
        XLS_RETURN_IF_ERROR(
            table_.AddTypeAnnotationToVariableForParametricContext(
                invocation_context, binding, type));
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
        XLS_ASSIGN_OR_RETURN(InterpValue value, evaluator_->Evaluate(*expr));
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
      absl::flat_hash_map<const NameDef*, ExprOrType> callee_struct_value_exprs;
      XLS_ASSIGN_OR_RETURN(
          callee_struct_value_exprs,
          table_.GetParametricValueExprs(*callee_struct_context));
      for (const auto& [name_def, expr] : callee_struct_value_exprs) {
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
    ParametricEnv env(std::move(values));
    table_.SetParametricEnv(invocation_context, env);
    table_.SetParametricValueExprs(invocation_context,
                                   std::move(actual_parametrics));
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
        invocation->span(), invocation_context->parent_context(),
        invocation_context, implicit_parametrics, formal_types, actual_args, ti,
        actual_arg_ti,
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
      const Span& span,
      std::optional<const ParametricContext*> actual_arg_context,
      std::optional<const ParametricContext*> target_context,
      absl::flat_hash_set<const ParametricBinding*> implicit_parametrics,
      absl::Span<const TypeAnnotation* const> formal_types,
      absl::Span<Expr* const> actual_args, TypeInfo* output_ti,
      TypeInfo* actual_arg_ti,
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
          TypeAnnotationFilter::FilterParamTypes().Chain(
              TypeAnnotationFilter::FilterRefsToUnknownParametrics(
                  actual_arg_ti)
                  .Chain(
                      TypeAnnotationFilter::FilterFormalMemberTypes(&table_)));
      for (const TypeAnnotation* annotation : actual_arg_annotations) {
        VLOG(6) << "Pre-filtered actual arg annotation: "
                << annotation->ToString();
      }
      XLS_RETURN_IF_ERROR(resolver_->ResolveIndirectTypeAnnotations(
          actual_arg_context, actual_arg_annotations, filter));
      if (actual_arg_annotations.empty()) {
        VLOG(6) << "The actual argument type variable: "
                << (*actual_arg_type_var)->ToString()
                << " has no independent type annotations.";
        continue;
      }

      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* formal_type,
          resolver_->ResolveTypeRefs(target_context, formal_types[i]));

      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* actual_arg_type,
          resolver_->ResolveAndUnifyTypeAnnotations(
              actual_arg_context, actual_arg_annotations,
              actual_args[i]->span(), TypeAnnotationFilter::None(),
              /*require_bits_like=*/false));
      XLS_RETURN_IF_ERROR(
          ConvertSubtree(actual_arg_type, std::nullopt, actual_arg_context));
      VLOG(5) << "Infer using actual type: " << actual_arg_type->ToString()
              << " and formal type: " << formal_type->ToString()
              << " with effective context: " << ToString(actual_arg_context);

      absl::StatusOr<absl::flat_hash_map<const ParametricBinding*,
                                         InterpValueOrTypeAnnotation>>
          resolved = SolveForParametrics(
              import_data_, actual_arg_type, formal_type, implicit_parametrics,
              [&](const TypeAnnotation* expected_type, const Expr* expr) {
                return evaluator_->Evaluate(ParametricContextScopedExpr(
                    actual_arg_context, expected_type, expr));
              });
      if (!resolved.ok()) {
        VLOG(5) << "Solution failed with " << resolved.status();

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
          XLS_RET_CHECK(target_context.has_value());
          XLS_ASSIGN_OR_RETURN(
              const TypeAnnotation* type,
              CleanseGenericTypeArgument(
                  actual_arg_context, *actual_arg_ti,
                  std::get<const TypeAnnotation*>(value_or_type)));
          XLS_RETURN_IF_ERROR(
              table_.AddTypeAnnotationToVariableForParametricContext(
                  target_context, binding, type));
        }
      }
    }
    if (!implicit_parametrics.empty()) {
      std::vector<std::string> binding_names;
      binding_names.reserve(implicit_parametrics.size());
      for (const ParametricBinding* binding : implicit_parametrics) {
        binding_names.push_back(binding->identifier());
      }
      return TypeInferenceErrorStatus(
          span, /*type=*/nullptr,
          absl::Substitute("Could not infer parametric(s): $0 of $1; target "
                           "context: $2; module: $3",
                           absl::StrJoin(binding_names, ", "),
                           GetParametricBindingOwnerDescription(
                               *implicit_parametrics.begin()),
                           ToString(target_context), module_.name()),
          file_table_);
    }
    return values;
  }

  // Replaces parametrics and constants in a `type` which is being passed
  // (explicitly or implicitly) as a type parametric. This is necessary because
  // it amounts to passing the type over a parametric context boundary. For
  // example,
  //
  // fn g<T: type>(value: T) { ... }
  // fn f<A: u32>(value: uN[A]) { g<uN[A]>(value); }
  // fn main() { f(u5:1); }
  //
  // Here there is an invocation context for `f` in which `A=5`, and there is an
  // invocation context for `g` in which `A` does not exist. If we just add the
  // type annotation `uN[A]` to the type variable `T` for the `g` invocation
  // context (which is how types are passed as parametrics), then when `T` is
  // unified in the `g` context, `A` will fail to evaluate. So, using this
  // helper, we instead turn `uN[A]` into `uN[5]` before adding it to `T` in
  // this scenario.
  absl::StatusOr<const TypeAnnotation*> CleanseGenericTypeArgument(
      std::optional<const ParametricContext*> parametric_context,
      const TypeInfo& ti, const TypeAnnotation* type) {
    if (!parametric_context.has_value()) {
      return type;
    }

    std::vector<std::pair<const NameRef*, const NameDef*>> refs;
    XLS_ASSIGN_OR_RETURN(refs, CollectReferencedUnder(type));
    absl::flat_hash_map<const NameDef*, ExprOrType> values;
    for (const auto& [ref, def] : refs) {
      std::optional<InterpValue> value = ti.GetConstExprOption(def);
      if (value.has_value()) {
        std::optional<Number*> literal =
            ConvertToNumberIfBitsLike(*ref->owner(), ref->span(), *value);
        if (literal.has_value()) {
          values.emplace(def, *literal);
        }
      }
    }

    return GetParametricFreeType(type, std::move(values),
                                 (*parametric_context)->self_type());
  }

  absl::StatusOr<const TypeAnnotation*> InstantiateParametricStruct(
      Module& module, const Span& span,
      std::optional<const ParametricContext*> parent_context,
      const StructDef& struct_def,
      const std::vector<InterpValue>& explicit_parametrics,
      std::optional<const StructInstanceBase*> instantiator_node) override {
    VLOG(6) << "Instantiate parametric struct `" << struct_def.identifier()
            << "` with parent context: " << ToString(parent_context) << " and "
            << explicit_parametrics.size()
            << " explicit parametrics from module: " << module.name();

    // The goal here is to come up with a complete parametric value `Expr`
    // vector, which has a value for every formal binding, by inferring or
    // defaulting whichever ones are not explicit. The algorithm is the same as
    // for parametric function invocations, and the differences are in the
    // logistics. We build this as a map, `resolved_parametrics`, and convert it
    // to a vector at the end.
    XLS_ASSIGN_OR_RETURN(TypeInfo * instance_parent_ti,
                         GetTypeInfo(struct_def.owner(), parent_context));
    XLS_ASSIGN_OR_RETURN(TypeInfo * actual_arg_ti,
                         GetTypeInfo(&module, parent_context));
    XLS_ASSIGN_OR_RETURN(TypeInfo * instance_type_info,
                         import_data_.type_info_owner().New(
                             struct_def.owner(), instance_parent_ti));
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
      if (!instantiator_node.has_value()) {
        return TypeInferenceErrorStatus(
            span, /*type=*/nullptr,
            absl::Substitute(
                "Reference to parametric struct type `$0` must have all "
                "parametrics specified in this context. Implicit struct "
                "parametrics are only allowed in struct instance expressions.",
                struct_def.identifier()),
            file_table_);
      }

      absl::flat_hash_map<std::string, InterpValue> new_values;
      // Note: setting target_context to null is temporary until we support
      // generic type parametrics for structs.
      XLS_ASSIGN_OR_RETURN(
          new_values,
          InferImplicitParametrics(
              span, parent_context, /*target_context=*/std::nullopt,
              implicit_parametrics, formal_member_types, actual_member_exprs,
              instance_type_info, actual_arg_ti, [&](const Expr* actual_arg) {
                // Invocation arguments within a struct need to be converted
                // prior to use.
                if (actual_arg->kind() == AstNodeKind::kInvocation) {
                  return ConvertSubtree(actual_arg, std::nullopt,
                                        parent_context);
                }
                return absl::OkStatus();
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
            evaluator_->Evaluate(parent_context, instance_type_info,
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
    Expr* splatted = instance.splatted();
    if (splatted->kind() == AstNodeKind::kInvocation) {
      // If the splatted node is an invocation, clone it to preserve the
      // parentage of the original invocation node.
      XLS_ASSIGN_OR_RETURN(AstNode * clone, table_.Clone(instance.splatted(),
                                                         &NoopCloneReplacer));
      splatted = down_cast<Expr*>(clone);
    }
    Expr* expr = module_.Make<Attr>(instance.span(), splatted, member.name());
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
      const TypeAnnotation* member_type) final {
    if (!struct_or_proc_ref.def->IsParametric()) {
      return member_type;
    }
    absl::flat_hash_map<const NameDef*, ExprOrType> parametrics_and_constants;
    std::vector<ParametricBinding*> bindings =
        struct_or_proc_ref.def->parametric_bindings();
    CHECK_GE(bindings.size(), struct_or_proc_ref.parametrics.size());

    for (int i = 0; i < bindings.size(); i++) {
      const ParametricBinding* binding = bindings[i];
      XLS_ASSIGN_OR_RETURN(
          InterpValue value,
          (*struct_context)->type_info()->GetConstExpr(binding->name_def()));
      std::optional<Number*> literal =
          ConvertToNumberIfBitsLike(module_, binding->span(), value);
      XLS_RET_CHECK(literal.has_value());
      parametrics_and_constants.emplace(binding->name_def(), *literal);
    }

    // If there is an impl, load the impl constants into the map for erasure as
    // well.
    if (struct_or_proc_ref.def->impl().has_value()) {
      for (const ConstantDef* constant :
           (*struct_or_proc_ref.def->impl())->GetConstants()) {
        XLS_ASSIGN_OR_RETURN(
            InterpValue value,
            (*struct_context)->type_info()->GetConstExpr(constant->name_def()));
        std::optional<Number*> literal =
            ConvertToNumberIfBitsLike(module_, constant->span(), value);
        if (literal.has_value()) {
          parametrics_and_constants.emplace(constant->name_def(), *literal);
        }
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
      std::optional<const TypeAnnotation*> real_self_type = std::nullopt,
      bool clone_if_no_parametrics = true) {
    if (!clone_if_no_parametrics) {
      std::vector<std::pair<const NameRef*, const NameDef*>> refs;
      XLS_ASSIGN_OR_RETURN(refs, CollectReferencedUnder(type));
      const bool any_parametrics = absl::c_any_of(refs, [&](const auto& pair) {
        return pair.second->parent()->kind() == AstNodeKind::kParametricBinding;
      });
      if (!any_parametrics) {
        return type;
      }
    }

    CloneReplacer replacer = ChainCloneReplacers(
        NameRefMapper(table_, actual_values, type->owner()),
        [&](const AstNode* node, Module*,
            const absl::flat_hash_map<const AstNode*, AstNode*>&)
            -> absl::StatusOr<std::optional<AstNode*>> {
          // Explicitly leave attrs alone in an example like
          // `uN[STRUCT_CONST.n]`. With the current grammar, there is no way
          // these nodes need parametric replacement. Trying to clone them
          // across modules can make them fail to evaluate.
          if (node->kind() == AstNodeKind::kAttr) {
            return const_cast<AstNode*>(node);
          }
          return std::nullopt;
        });
    if (real_self_type.has_value()) {
      replacer = ChainCloneReplacers(
          std::move(replacer),
          [&](const AstNode* node, Module*,
              const absl::flat_hash_map<const AstNode*, AstNode*>&)
              -> absl::StatusOr<std::optional<AstNode*>> {
            if (node->kind() == AstNodeKind::kTypeAnnotation &&
                down_cast<const TypeAnnotation*>(node)
                    ->IsAnnotation<SelfTypeAnnotation>()) {
              return const_cast<TypeAnnotation*>(*real_self_type);
            }
            return std::nullopt;
          });
    }
    XLS_ASSIGN_OR_RETURN(
        AstNode * clone,
        table_.Clone(type, std::move(replacer), type->owner()));
    return down_cast<const TypeAnnotation*>(clone);
  }

  // Cache a concretized type for a (node, parametric_context) pair to be
  // reused, if the same type is expected to be concretized many times.
  absl::Status AddCachedType(
      const AstNode* node,
      std::optional<const ParametricContext*> parametric_context,
      const Type& type) {
    auto result = type_cache_.try_emplace(
        std::make_pair(node, parametric_context), type.CloneToUnique());
    if (!result.second) {
      return absl::InternalError(absl::StrCat(
          "Failed to add type ", type.ToString(), " to type cache, Type ",
          result.first->second->ToString(), " already exists for key (",
          node->ToInlineString(),
          parametric_context ? ", " + parametric_context.value()->ToString()
                             : "",
          ")"));
    }
    return absl::OkStatus();
  }

  // Get the cached concretized type for a (node, parametric_context) pair.
  std::unique_ptr<Type> GetCachedType(
      const AstNode* node,
      std::optional<const ParametricContext*> parametric_context) {
    auto cached = type_cache_.find(std::make_pair(node, parametric_context));
    if (cached != type_cache_.end()) {
      return cached->second->CloneToUnique();
    }
    return nullptr;
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

  SemanticsAnalysis* GetSemanticsAnalysis() override {
    return semantics_analysis_.get();
  }

  InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  TypeInfo* const base_type_info_;
  const FileTable& file_table_;
  std::unique_ptr<TypeSystemTracer> tracer_;
  std::unique_ptr<SemanticsAnalysis> semantics_analysis_;
  std::unique_ptr<Evaluator> evaluator_;
  std::unique_ptr<TypeAnnotationResolver> resolver_;
  std::unique_ptr<ConstantCollector> constant_collector_;
  std::unique_ptr<FastConcretizer> fast_concretizer_;
  absl::flat_hash_map<std::optional<const ParametricContext*>,
                      absl::flat_hash_set<const AstNode*>>
      converted_subtrees_;
  absl::flat_hash_map<std::optional<const ParametricContext*>,
                      absl::flat_hash_set<const Invocation*>>
      converted_invocations_;
  absl::flat_hash_set<const Proc*> converted_procs_;

  SimplifiedTypeAnnotationCache simplified_type_annotation_cache_;
  absl::flat_hash_map<
      std::pair<const AstNode*, std::optional<const ParametricContext*>>,
      std::unique_ptr<Type>>
      type_cache_;

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
CreateInferenceTableConverter(
    InferenceTable& table, Module& module, ImportData& import_data,
    WarningCollector& warning_collector, const FileTable& file_table,
    std::unique_ptr<TypeSystemTracer> tracer,
    std::unique_ptr<SemanticsAnalysis> semantics_analysis) {
  VLOG(1) << "CreateInferenceTableConverter: module " << &module;

  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data.type_info_owner().New(&module));
  return std::make_unique<InferenceTableConverterImpl>(
      table, module, import_data, warning_collector, type_info, file_table,
      std::move(tracer), std::move(semantics_analysis));
}

}  // namespace xls::dslx
