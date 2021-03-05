// Copyright 2020 The XLS Authors
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

#include "xls/dslx/typecheck.h"

#include "xls/dslx/deduce.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interpreter.h"

namespace xls::dslx {

// Checks the function's parametrics' and arguments' types.
static absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>>
CheckFunctionParams(Function* f, DeduceCtx* ctx) {
  for (ParametricBinding* parametric : f->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_binding_type,
                         ctx->Deduce(parametric->type()));
    if (parametric->expr() != nullptr) {
      // TODO(leary): 2020-07-06 Fully document the behavior of parametric
      // function calls in parametric expressions.
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr_type,
                           ctx->Deduce(parametric->expr()));
      if (*expr_type != *parametric_binding_type) {
        return XlsTypeErrorStatus(parametric->span(), *parametric_binding_type,
                                  *expr_type,
                                  "Annotated type of derived parametric value "
                                  "did not match inferred type.");
      }
    }
    ctx->type_info()->SetItem(parametric->name_def(), *parametric_binding_type);
  }

  std::vector<std::unique_ptr<ConcreteType>> param_types;
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> param_type,
                         ctx->Deduce(param));
    ctx->type_info()->SetItem(param->name_def(), *param_type);
    param_types.push_back(std::move(param_type));
  }

  return param_types;
}

// Validates type annotations on parameters / return type of `f` are consistent.
//
// Returns a XlsTypeErrorStatus when the return type deduced is inconsistent
// with the return type annotation on `f`.
static absl::Status CheckFunction(Function* f, DeduceCtx* ctx) {
  const FnStackEntry& entry = ctx->fn_stack().back();

  std::vector<std::unique_ptr<ConcreteType>> param_types;
  std::unique_ptr<ConcreteType> annotated_return_type;

  // First, get the types of the function's parametrics, args, and return type.
  if (f->IsParametric() && entry.Matches(f)) {
    // Parametric functions are evaluated per invocation. If we're currently
    // inside of this function, it just means that we already have the type
    // signature, and now we just need to evaluate the body.
    absl::optional<ConcreteType*> f_type = ctx->type_info()->GetItem(f);
    XLS_RET_CHECK(f_type.has_value());
    auto* function_type = dynamic_cast<FunctionType*>(f_type.value());
    XLS_RET_CHECK(function_type != nullptr);
    annotated_return_type = function_type->return_type().CloneToUnique();
    param_types = CloneToUnique(absl::MakeSpan(function_type->params()));
  } else {
    XLS_VLOG(1) << "Type-checking signature for function: " << f->identifier();
    XLS_ASSIGN_OR_RETURN(param_types, CheckFunctionParams(f, ctx));

    if (f->IsParametric()) {
      // We just needed the type signature so that we can instantiate this
      // invocation. Let's return this for now, and typecheck the body once we
      // have symbolic bindings.
      if (f->return_type() == nullptr) {
        annotated_return_type = TupleType::MakeNil();
      } else {
        XLS_ASSIGN_OR_RETURN(annotated_return_type,
                             ctx->Deduce(f->return_type()));
      }
      FunctionType function_type(std::move(param_types),
                                 std::move(annotated_return_type));
      ctx->type_info()->SetItem(f->name_def(), function_type);
      ctx->type_info()->SetItem(f, function_type);
      return absl::OkStatus();
    }
  }

  XLS_VLOG(1) << "Type-checking body for function: " << f->identifier();

  // Second, typecheck the return type of the function.
  // Note: if there is no annotated return type, we assume nil.
  std::unique_ptr<ConcreteType> return_type;
  if (f->return_type() == nullptr) {
    return_type = TupleType::MakeNil();
  } else {
    XLS_ASSIGN_OR_RETURN(return_type, DeduceAndResolve(f->return_type(), ctx));
  }

  // Third, typecheck the body of the function.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_return_type,
                       DeduceAndResolve(f->body(), ctx));

  // Finally, assert type consistency between body and annotated return type.
  XLS_VLOG(3) << absl::StrFormat(
      "Resolved return type: %s => %s resolved body type: %s => %s",
      (annotated_return_type == nullptr ? "none"
                                        : annotated_return_type->ToString()),
      return_type->ToString(), body_return_type->ToString(),
      body_return_type->ToString());
  if (*return_type != *body_return_type) {
    return XlsTypeErrorStatus(
        f->body()->span(), *body_return_type, *return_type,
        absl::StrFormat("Return type of function body for '%s' did not match "
                        "the annotated return type.",
                        f->identifier()));
  }

  FunctionType function_type(std::move(param_types),
                             std::move(body_return_type));
  ctx->type_info()->SetItem(f->name_def(), function_type);
  ctx->type_info()->SetItem(f, function_type);
  return absl::OkStatus();
}

// Validates a test (body) within a module.
static absl::Status CheckTest(TestFunction* t, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_return_type,
                       ctx->Deduce(t->body()));
  if (body_return_type->IsNil()) {
    return absl::OkStatus();
  }
  return XlsTypeErrorStatus(
      t->body()->span(), *body_return_type, *ConcreteType::MakeNil(),
      absl::StrFormat("Return type of test body for '%s' did not match the "
                      "expected test return type `()`.",
                      t->identifier()));
}

absl::StatusOr<NameDef*> InstantiateBuiltinParametric(
    BuiltinNameDef* builtin_name, Invocation* invocation, DeduceCtx* ctx) {
  std::vector<std::unique_ptr<ConcreteType>> arg_types;
  for (Expr* arg : invocation->args()) {
    absl::optional<ConcreteType*> arg_type = ctx->type_info()->GetItem(arg);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                         Resolve(**arg_type, ctx));
    arg_types.push_back(std::move(resolved));
  }

  absl::optional<std::string> map_fn_name;
  absl::optional<Function*> map_fn;
  absl::optional<std::vector<ParametricBinding*>>
      higher_order_parametric_bindings;
  if (builtin_name->identifier() == "map") {
    Expr* map_fn_ref = invocation->args()[1];
    if (auto* colon_ref = dynamic_cast<ColonRef*>(map_fn_ref)) {
      map_fn_name = colon_ref->attr();
      Import* import_node = colon_ref->ResolveImportSubject().value();
      absl::optional<const ImportedInfo*> imported =
          ctx->type_info()->GetImported(import_node);
      XLS_RET_CHECK(imported.has_value());
      XLS_ASSIGN_OR_RETURN(
          map_fn, (*imported)->module->GetFunctionOrError(*map_fn_name));
      higher_order_parametric_bindings = (*map_fn)->parametric_bindings();
    } else {
      auto* name_ref = dynamic_cast<NameRef*>(map_fn_ref);
      map_fn_name = name_ref->identifier();
      if (!GetParametricBuiltins().contains(*map_fn_name)) {
        XLS_ASSIGN_OR_RETURN(map_fn,
                             ctx->module()->GetFunctionOrError(*map_fn_name));
        higher_order_parametric_bindings = (*map_fn)->parametric_bindings();
      }
    }
  }

  std::vector<const ConcreteType*> arg_type_ptrs;
  arg_type_ptrs.reserve(arg_types.size());
  for (auto& arg_type : arg_types) {
    arg_type_ptrs.push_back(arg_type.get());
  }

  XLS_ASSIGN_OR_RETURN(SignatureFn fsignature, GetParametricBuiltinSignature(
                                                   builtin_name->identifier()));

  // Callback that signatures can use to request constexpr evaluation of their
  // arguments -- this is a special builtin superpower used by e.g. range().
  auto constexpr_eval = [&](int64 argno) -> absl::Status {
    Expr* arg = invocation->args()[argno];

    absl::flat_hash_map<std::string, int64> env =
        ctx->fn_stack().back().symbolic_bindings().ToMap();
    absl::flat_hash_map<std::string, int64> bit_widths;
    for (const auto& [identifier, _] : env) {
      // TODO(leary): 2020-03-05 We should carry around the bitwidths of the
      // symbolic bindings, right now we only know the value, so we're forced to
      // guess this is ok. Potentially we should carry around InterpValues in
      // symbolic bindings in general.
      bit_widths[identifier] = 32;
    }

    XLS_ASSIGN_OR_RETURN(
        int64 value,
        Interpreter::InterpretExprToInt(
            arg->owner(), ctx->type_info(), ctx->typecheck_module(),
            ctx->additional_search_paths(), ctx->import_cache(), /*env=*/env,
            /*bit_widths=*/bit_widths,
            /*expr=*/arg, FnCtx{}));
    ctx->type_info()->NoteConstExpr(arg, value);
    return absl::OkStatus();
  };

  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      fsignature(
          SignatureData{arg_type_ptrs, builtin_name->identifier(),
                        invocation->span(), higher_order_parametric_bindings,
                        constexpr_eval},
          ctx));
  FunctionType* fn_type = dynamic_cast<FunctionType*>(tab.type.get());
  XLS_RET_CHECK(fn_type != nullptr) << tab.type->ToString();

  const SymbolicBindings& fn_symbolic_bindings =
      ctx->fn_stack().back().symbolic_bindings();
  ctx->type_info()->AddInvocationSymbolicBindings(
      invocation, fn_symbolic_bindings, tab.symbolic_bindings);
  ctx->type_info()->SetItem(invocation->callee(), *fn_type);
  ctx->type_info()->SetItem(invocation, fn_type->return_type());

  if (builtin_name->identifier() == "map") {
    Expr* map_fn_ref = invocation->args()[1];
    if (GetParametricBuiltins().contains(*map_fn_name) ||
        !(*map_fn)->IsParametric()) {
      // A builtin higher-order parametric fn would've been typechecked when we
      // were going through the arguments of this invocation. If the function
      // wasn't parametric, then we're good to go.
      return nullptr;
    }

    if (ctx->type_info()->HasInstantiation(invocation, tab.symbolic_bindings)) {
      // We've already typechecked this parametric function using these
      // bindings.
      return nullptr;
    }

    // If the higher order function is parametric, we need to typecheck its body
    // with the symbolic bindings we just computed.
    if (auto* colon_ref = dynamic_cast<ColonRef*>(map_fn_ref)) {
      absl::optional<Import*> import = colon_ref->ResolveImportSubject();
      XLS_RET_CHECK(import.has_value());
      absl::optional<const ImportedInfo*> import_info =
          ctx->type_info()->GetImported(*import);
      XLS_RET_CHECK(import_info.has_value());

      XLS_ASSIGN_OR_RETURN(
          TypeInfo * invocation_imported_type_info,
          ctx->type_info_owner().New((*import_info)->module,
                                     /*parent=*/(*import_info)->type_info));
      std::shared_ptr<DeduceCtx> imported_ctx =
          ctx->MakeCtx(invocation_imported_type_info, (*import_info)->module);
      imported_ctx->fn_stack().push_back(
          FnStackEntry::Make(map_fn.value(), tab.symbolic_bindings));
      // We need to typecheck this imported function with respect to its module.
      XLS_RETURN_IF_ERROR(
          ctx->typecheck_function()(map_fn.value(), imported_ctx.get()));
      ctx->type_info()->AddInstantiation(invocation, tab.symbolic_bindings,
                                         invocation_imported_type_info);
    } else {
      // If the higher-order parametric fn is in this module, let's try to push
      // it onto the typechecking stack.
      ctx->fn_stack().push_back(
          FnStackEntry::Make(map_fn.value(), tab.symbolic_bindings));

      // Create a "derived" type info (a child type info with the current type
      // info as a parent), and note that it exists for an instantiation (in the
      // parent type info).
      TypeInfo* parent_type_info = ctx->type_info();
      ctx->AddDerivedTypeInfo();
      parent_type_info->AddInstantiation(invocation, tab.symbolic_bindings,
                                         ctx->type_info());
      auto* name_ref = dynamic_cast<NameRef*>(map_fn_ref);
      XLS_RET_CHECK(absl::holds_alternative<NameDef*>(name_ref->name_def()));
      return absl::get<NameDef*>(name_ref->name_def());
    }
  }

  return nullptr;
}

// -- Type checking "driver loop"
//
// Implementation note: CheckTopNodeInModule() currrently drives the
// typechecking process for some top level node (aliased as TopNode below for
// convenience).
//
// It keeps a stack of records called "seen" to note which functions are in the
// process of being typechecked (wip = Work In Progress), so we can determine if
// we're recursing (before typechecking has completed for a function).
//
// Normal deduction is attempted for the top level node, and if a type is
// missing, it is because parametric instantiation is left until invocations are
// observed. The TypeMissingError notes the node that had its type missing and
// its user (which would often be an invocation), which allows us to drive the
// appropriate instantiation of that parametric.

using TopNode = absl::variant<Function*, TestFunction*, StructDef*, TypeDef*>;
struct WipRecord {
  TopNode f;
  bool wip;
};

// Wrapper for information used to typecheck a top-level AST node.
struct TypecheckStackRecord {
  // Name of this top level node.
  std::string name;

  // AstNode::GetNodeTypeName() value for the node.
  std::string kind;

  // The node that needs 'name' to be typechecked. Used to detect the
  // typechecking of the higher order function in map invocations.
  AstNode* user = nullptr;
};

// Does the normal deduction processing for a given top level node.
//
// Note: This may end up in a TypeMissingError() that gets handled in the outer
// loop.
static absl::Status ProcessTopNode(TopNode f, DeduceCtx* ctx) {
  if (absl::holds_alternative<Function*>(f)) {
    XLS_RETURN_IF_ERROR(CheckFunction(absl::get<Function*>(f), ctx));
  } else if (absl::holds_alternative<TestFunction*>(f)) {
    XLS_RETURN_IF_ERROR(CheckTest(absl::get<TestFunction*>(f), ctx));
  } else {
    // Nothing special to do for these other variants, we just want to be able
    // to catch any TypeMissingErrors and try to resolve them.
    XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(f)).status());
  }
  return absl::OkStatus();
}

// Returns the identifier for the top level node.
static const std::string& Identifier(const TopNode& f) {
  if (absl::holds_alternative<Function*>(f)) {
    return absl::get<Function*>(f)->identifier();
  } else if (absl::holds_alternative<TestFunction*>(f)) {
    return absl::get<TestFunction*>(f)->identifier();
  } else if (absl::holds_alternative<StructDef*>(f)) {
    return absl::get<StructDef*>(f)->identifier();
  } else {
    XLS_CHECK(absl::holds_alternative<TypeDef*>(f))
        << ToAstNode(f)->GetNodeTypeName() << ": " << ToAstNode(f)->ToString();
    return absl::get<TypeDef*>(f)->identifier();
  }
}

// Helper that returns whether "n" is an invocation node that is calling the
// BuiltinNameDef "map" as the callee.
static bool IsCalleeMap(AstNode* n) {
  if (n == nullptr) {
    return false;
  }
  auto* invocation = dynamic_cast<Invocation*>(n);
  if (invocation == nullptr) {
    return false;
  }
  auto* name_ref = dynamic_cast<NameRef*>(invocation->callee());
  if (name_ref == nullptr) {
    return false;
  }
  AnyNameDef name_def = name_ref->name_def();
  return absl::holds_alternative<BuiltinNameDef*>(name_def) &&
         absl::get<BuiltinNameDef*>(name_def)->identifier() == "map";
}

static void VLogStack(absl::Span<const TypecheckStackRecord> stack) {
  XLS_VLOG(5) << "typecheck stack:";
  for (const TypecheckStackRecord& record : stack) {
    XLS_VLOG(5) << absl::StreamFormat(
        "  name: '%s' kind: %s user: %s", record.name, record.kind,
        record.user == nullptr ? "none" : record.user->ToString());
  }
}

static void VLogSeen(
    const absl::flat_hash_map<std::pair<std::string, std::string>, WipRecord>&
        map) {
  XLS_VLOG(5) << "seen map:";
  for (const auto& item : map) {
    XLS_VLOG(5) << absl::StreamFormat(
        "  %s :: %s => %s wip: %s", item.first.first, item.first.second,
        Identifier(item.second.f), item.second.wip ? "true" : "false");
  }
}

// Handles a TypeMissingError() that results from an attempt at deductive
// typechecking of a TopNode.
//
// * For a name that refers to a function in the "function_map", we push that
//   onto the top of the processing stack.
static absl::Status HandleMissingType(
    const absl::Status& status,
    const absl::flat_hash_map<std::string, Function*>& function_map,
    const TopNode& f,
    absl::flat_hash_map<std::pair<std::string, std::string>, WipRecord>& seen,
    std::vector<TypecheckStackRecord>& stack, DeduceCtx* ctx) {
  NodeAndUser e = ParseTypeMissingErrorMessage(status.message());
  while (true) {
    XLS_VLOG(5) << absl::StreamFormat(
        "Handling TypeMissingError; node: %s (%s); user: %s (%s)",
        (e.node == nullptr ? "none" : e.node->ToString()),
        (e.node == nullptr ? "none" : e.node->GetNodeTypeName()),
        (e.user == nullptr ? "none" : e.user->ToString()),
        (e.user == nullptr ? "none" : e.user->GetNodeTypeName()));

    // Referring to a function name in the same module.
    if (auto* name_def = dynamic_cast<NameDef*>(e.node);
        name_def != nullptr && function_map.contains(name_def->identifier())) {
      // If the callee is seen-and-not-done, we're recursing.
      const std::pair<std::string, std::string> seen_key{name_def->identifier(),
                                                         "Function"};
      auto it = seen.find(seen_key);
      if (it != seen.end() && it->second.wip) {
        return TypeInferenceErrorStatus(
            name_def->span(), nullptr,
            absl::StrFormat("Recursion detected while typechecking; name: '%s'",
                            name_def->identifier()));
      }
      // Note: the callees will often be parametric (where we've deferred
      // determining their concrete type signature until invocation time), but
      // they can *also* be concrete functions that are only invoked via these
      // "lazily checked" parametrics.
      Function* callee = function_map.at(name_def->identifier());
      seen[seen_key] = WipRecord{callee, /*wip=*/true};
      stack.push_back(TypecheckStackRecord{
          callee->identifier(),
          std::string(ToAstNode(callee)->GetNodeTypeName()), e.user});
      break;
    }

    // Referring to a parametric builtin.
    if (auto* builtin_name_def = dynamic_cast<BuiltinNameDef*>(e.node);
        builtin_name_def != nullptr &&
        GetParametricBuiltins().contains(builtin_name_def->identifier())) {
      XLS_VLOG(5) << absl::StreamFormat(
          "node: %s; identifier %s; exception user: %s",
          e.node == nullptr ? "none" : e.node->ToString(),
          builtin_name_def->identifier(),
          e.user == nullptr ? "none" : e.user->ToString());
      if (auto* invocation = dynamic_cast<Invocation*>(e.user)) {
        XLS_ASSIGN_OR_RETURN(
            NameDef * func,
            InstantiateBuiltinParametric(builtin_name_def, invocation, ctx));
        if (func != nullptr) {
          // We need to figure out what to do with this higher order
          // parametric function.
          //
          // TODO(leary): 2020-12-18 I think there's no way this returning a
          // NameDef here is fully correct now that functions can come from
          // ColonRefs.
          e.node = func;
          continue;
        }
        break;
      }
    }

    // Raise if this wasn't a) a function in this module or b) a builtin.
    return status;
  }

  return absl::OkStatus();
}

absl::Status CheckTopNodeInModule(TopNode f, DeduceCtx* ctx) {
  XLS_RET_CHECK(ToAstNode(f) != nullptr);
  absl::flat_hash_map<std::pair<std::string, std::string>, WipRecord> seen;

  auto to_seen_key =
      [](const TopNode& top_node) -> std::pair<std::string, std::string> {
    return std::pair<std::string, std::string>{
        Identifier(top_node), ToAstNode(top_node)->GetNodeTypeName()};
  };

  seen.emplace(to_seen_key(f), WipRecord{f, /*wip=*/true});

  std::vector<TypecheckStackRecord> stack = {TypecheckStackRecord{
      Identifier(f), std::string(ToAstNode(f)->GetNodeTypeName())}};

  const absl::flat_hash_map<std::string, Function*>& function_map =
      ctx->module()->GetFunctionByName();
  while (!stack.empty()) {
    XLS_VLOG(5) << "Stack still not empty...";
    VLogStack(stack);
    VLogSeen(seen);
    const TypecheckStackRecord& record = stack.back();
    const TopNode f = seen.at({record.name, record.kind}).f;

    absl::Status status = ProcessTopNode(f, ctx);
    XLS_VLOG(5) << "Process top node status: " << status
                << "; f: " << ToAstNode(f)->ToString();

    if (IsTypeMissingErrorStatus(status)) {
      XLS_RETURN_IF_ERROR(
          HandleMissingType(status, function_map, f, seen, stack, ctx));
      continue;
    }

    XLS_RETURN_IF_ERROR(status);

    XLS_VLOG(5) << "Marking as done: " << Identifier(f);
    seen[to_seen_key(f)] = WipRecord{f, /*wip=*/false};  // Mark as done.
    const TypecheckStackRecord stack_record = stack.back();
    stack.pop_back();
    const std::string& fn_name = ctx->fn_stack().back().name();

    if (IsCalleeMap(stack_record.user)) {
      // We just typechecked a higher-order parametric function (from map()).
      // Let's go back to our parent type_info mapping.
      XLS_RETURN_IF_ERROR(ctx->PopDerivedTypeInfo());
    }

    if (stack_record.name == fn_name) {
      // i.e. we just finished typechecking the body of the function we're
      // currently inside of.
      //
      // Note: if this is a local parametric function, we don't refer to our
      // parent type_info until deduce._check_parametric_invocation() to avoid
      // entering an infinite loop. See the try-catch in that function for more
      // details.
      ctx->fn_stack().pop_back();
    }
  }
  return absl::OkStatus();
}

// Helper type to place on the stack when we intend to pop off a FnStackEntry
// when done, or expect a caller to pop it off for us. That is, this helps us
// check fn_stack() invariants are as expected.
class ScopedFnStackEntry {
 public:
  // Args:
  //  expect_popped: Indicates that we expect, in the destructor for this scope,
  //    that the entry will have already been popped.
  ScopedFnStackEntry(DeduceCtx* ctx, Module* module, bool expect_popped = false)
      : ctx_(ctx),
        depth_before_(ctx->fn_stack().size()),
        expect_popped_(expect_popped) {
    ctx->fn_stack().push_back(FnStackEntry::MakeTop(module));
  }

  ScopedFnStackEntry(Function* f, DeduceCtx* ctx, bool expect_popped = false)
      : ctx_(ctx),
        depth_before_(ctx->fn_stack().size()),
        expect_popped_(expect_popped) {
    ctx->fn_stack().push_back(FnStackEntry::Make(f, SymbolicBindings()));
  }

  // Called when we close out a scope. We can't use this object as a scope
  // guard easily because we want to be able to detect if we return an
  // absl::Status early, so we have to manually put end-of-scope calls at usage
  // points.
  void Finish() {
    if (expect_popped_) {
      XLS_CHECK_EQ(ctx_->fn_stack().size(), depth_before_);
    } else {
      int64 depth_after_push = depth_before_ + 1;
      XLS_CHECK_EQ(ctx_->fn_stack().size(), depth_after_push);
      ctx_->fn_stack().pop_back();
    }
  }

 private:
  DeduceCtx* ctx_;
  int64 depth_before_;
  bool expect_popped_;
};

absl::StatusOr<TypeInfo*> CheckModule(
    Module* module, ImportCache* import_cache,
    absl::Span<const std::string> additional_search_paths) {
  std::vector<std::string> additional_search_paths_copy(
      additional_search_paths.begin(), additional_search_paths.end());
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_cache->type_info_owner().New(module));
  auto ftypecheck = [import_cache, additional_search_paths_copy](
                        Module* module) -> absl::StatusOr<TypeInfo*> {
    return CheckModule(module, import_cache, additional_search_paths_copy);
  };
  auto ctx_owned = absl::make_unique<DeduceCtx>(
      type_info, module,
      /*deduce_function=*/&Deduce,
      /*typecheck_function=*/&CheckTopNodeInModule,
      /*typecheck_module=*/ftypecheck, additional_search_paths, import_cache);
  DeduceCtx* ctx = ctx_owned.get();

  // First, populate type info with constants, enums, resolved imports, and
  // non-parametric functions.
  for (ModuleMember& member : *module->mutable_top()) {
    import_cache->SetTypecheckWorkInProgress(module, ToAstNode(member));
    XLS_VLOG(3) << "WIP for " << module->name() << " is "
                << ToAstNode(member)->ToString();
    if (absl::holds_alternative<Import*>(member)) {
      Import* import = absl::get<Import*>(member);
      XLS_ASSIGN_OR_RETURN(
          const ModuleInfo* imported,
          DoImport(ftypecheck, ImportTokens(import->subject()),
                   additional_search_paths, import_cache, import->span()));
      ctx->type_info()->AddImport(import, imported->module.get(),
                                  imported->type_info);
    } else if (absl::holds_alternative<ConstantDef*>(member) ||
               absl::holds_alternative<EnumDef*>(member)) {
      ScopedFnStackEntry scoped(ctx, module);
      XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(member)).status());
      scoped.Finish();
    } else if (absl::holds_alternative<Function*>(member)) {
      Function* f = absl::get<Function*>(member);
      if (f->IsParametric()) {
        continue;
      }

      XLS_VLOG(2) << "Typechecking function: " << f->ToString();
      ScopedFnStackEntry scoped(f, ctx,
                                /*expect_popped=*/true);
      XLS_RETURN_IF_ERROR(CheckTopNodeInModule(f, ctx));
      scoped.Finish();
      XLS_VLOG(2) << "Finished typechecking function: " << f->ToString();
    } else if (absl::holds_alternative<StructDef*>(member)) {
      StructDef* struct_def = absl::get<StructDef*>(member);
      XLS_VLOG(2) << "Typechecking struct: " << struct_def->ToString();
      ScopedFnStackEntry scoped(ctx, module);
      // Typecheck struct definitions using CheckTopNodeInModule() so that we
      // can typecheck function calls in parametric bindings, if there are any.
      XLS_RETURN_IF_ERROR(CheckTopNodeInModule(struct_def, ctx));
      scoped.Finish();
      XLS_VLOG(2) << "Finished typechecking struct: " << struct_def->ToString();
    } else if (absl::holds_alternative<TypeDef*>(member)) {
      TypeDef* type_def = absl::get<TypeDef*>(member);
      XLS_VLOG(2) << "Typechecking typedef: " << type_def->ToString();
      ScopedFnStackEntry scoped(ctx, module);
      XLS_RETURN_IF_ERROR(CheckTopNodeInModule(type_def, ctx));
      scoped.Finish();
      XLS_VLOG(2) << "Finished typechecking typedef: " << type_def->ToString();
    }
  }

  import_cache->SetTypecheckWorkInProgress(module, nullptr);

  // Then quickchecks...
  for (QuickCheck* qc : ctx->module()->GetQuickChecks()) {
    Function* f = qc->f();
    if (f->IsParametric()) {
      // TODO(leary): 2020-08-09 Add support for quickchecking parametric
      // functions.
      return TypeInferenceErrorStatus(
          f->span(), nullptr,
          "Quickchecking parametric functions is unsupported; see "
          "https://github.com/google/xls/issues/81");
    }

    XLS_VLOG(2) << "Typechecking function: " << f->ToString();
    ctx->fn_stack().push_back(FnStackEntry::Make(f, SymbolicBindings()));
    XLS_RETURN_IF_ERROR(CheckTopNodeInModule(f, ctx));
    absl::optional<const ConcreteType*> quickcheck_f_body_type =
        ctx->type_info()->GetItem(f->body());
    XLS_RET_CHECK(quickcheck_f_body_type.has_value());
    auto u1 = BitsType::MakeU1();
    if (*quickcheck_f_body_type.value() != *u1) {
      return XlsTypeErrorStatus(f->span(), *quickcheck_f_body_type.value(), *u1,
                                "Quickcheck functions must return a bool.");
    }

    XLS_VLOG(2) << "Finished typechecking function: " << f->ToString();
  }

  // Then tests...
  for (TestFunction* test : ctx->module()->GetTests()) {
    // New-style test constructs are specified using a function.
    // This function shouldn't be parametric and shouldn't take any arguments.
    if (!test->fn()->params().empty()) {
      return TypeInferenceErrorStatus(
          test->fn()->span(), nullptr,
          "Test functions shouldn't take arguments.");
    }
    if (test->fn()->IsParametric()) {
      return TypeInferenceErrorStatus(
          test->fn()->span(), nullptr,
          "Test functions shouldn't be parametric.");
    }

    ctx->fn_stack().push_back(
        FnStackEntry::Make(test->fn(), SymbolicBindings()));
    XLS_VLOG(2) << "Typechecking test: " << test->ToString();
    XLS_RETURN_IF_ERROR(CheckTopNodeInModule(test->fn(), ctx));
    XLS_VLOG(2) << "Finished typechecking test: " << test->ToString();
  }

  return type_info;
}

}  // namespace xls::dslx
