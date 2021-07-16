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
#include "xls/dslx/parametric_instantiator.h"
#include "re2/re2.h"

namespace xls::dslx {

// TODO(leary): 2021-03-29 The name here should be more like "Checkable" --
// these are the AST node variants we keep in a stacked order for typechecking.
using TopNode = absl::variant<Function*, TestFunction*, StructDef*, TypeDef*>;

// Checks the function's parametrics' and arguments' types.
static absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>>
CheckFunctionParams(Function* f, DeduceCtx* ctx) {
  for (ParametricBinding* parametric : f->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_binding_type,
                         ctx->Deduce(parametric->type_annotation()));
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
        annotated_return_type = TupleType::MakeUnit();
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
    return_type = TupleType::MakeUnit();
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

  // Implementation note: though we could have all functions have
  // NoteRequiresImplicitToken() be false unless otherwise noted, this helps
  // guarantee we did consider and make a note for every function -- the code is
  // generally complex enough it's nice to have this soundness check.
  if (absl::optional<bool> requires =
          ctx->type_info()->GetRequiresImplicitToken(f);
      !requires.has_value()) {
    ctx->type_info()->NoteRequiresImplicitToken(f, false);
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
  if (body_return_type->IsUnit()) {
    return absl::OkStatus();
  }
  return XlsTypeErrorStatus(
      t->body()->span(), *body_return_type, *ConcreteType::MakeUnit(),
      absl::StrFormat("Return type of test body for '%s' did not match the "
                      "expected test return type `()`.",
                      t->identifier()));
}

// Helper for InstantiateBuiltinParametric() that handles map invocations.
static absl::StatusOr<NameDef*> InstantiateBuiltinParametricMap(
    const TypeAndBindings& tab, absl::string_view map_fn_name, Function* map_fn,
    Invocation* invocation, DeduceCtx* ctx) {
  Expr* map_fn_ref = invocation->args()[1];
  if (GetParametricBuiltins().contains(map_fn_name) ||
      !map_fn->IsParametric()) {
    // A builtin higher-order parametric fn would've been typechecked when we
    // were going through the arguments of this invocation. If the function
    // wasn't parametric, then we're good to go.
    ctx->type_info()->SetInstantiationTypeInfo(invocation,
                                               tab.symbolic_bindings,
                                               /*type_info=*/nullptr);
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
        FnStackEntry::Make(map_fn, tab.symbolic_bindings));
    // We need to typecheck this imported function with respect to its module.
    XLS_RETURN_IF_ERROR(ctx->typecheck_function()(map_fn, imported_ctx.get()));
    XLS_VLOG(5) << "TypeInfo::AddInstantiation; invocation: "
                << invocation->ToString()
                << " symbolic_bindings: " << tab.symbolic_bindings;
    ctx->type_info()->SetInstantiationTypeInfo(
        invocation, /*caller=*/tab.symbolic_bindings,
        /*type_info=*/invocation_imported_type_info);
    return nullptr;
  }

  // If the higher-order parametric fn is in this module, let's push it onto the
  // typechecking stack.
  ctx->fn_stack().push_back(FnStackEntry::Make(map_fn, tab.symbolic_bindings));

  // Create a "derived" type info (a child type info with the current type
  // info as a parent), and note that it exists for an instantiation (in the
  // parent type info).
  TypeInfo* parent_type_info = ctx->type_info();
  ctx->AddDerivedTypeInfo();
  XLS_VLOG(5) << "TypeInfo::AddInstantiation; invocation: "
              << invocation->ToString()
              << " symbolic_bindings: " << tab.symbolic_bindings;
  parent_type_info->SetInstantiationTypeInfo(invocation, tab.symbolic_bindings,
                                             ctx->type_info());
  auto* name_ref = dynamic_cast<NameRef*>(map_fn_ref);
  XLS_RET_CHECK(absl::holds_alternative<NameDef*>(name_ref->name_def()));
  return absl::get<NameDef*>(name_ref->name_def());
}

// Instantiates a builtin parametric invocation; e.g. `update()`.
static absl::StatusOr<NameDef*> InstantiateBuiltinParametric(
    const TopNode* f, BuiltinNameDef* builtin_name, Invocation* invocation,
    DeduceCtx* ctx) {
  std::vector<std::unique_ptr<ConcreteType>> arg_types;
  std::vector<dslx::Span> arg_spans;
  arg_spans.reserve(invocation->args().size());
  arg_types.reserve(invocation->args().size());
  for (Expr* arg : invocation->args()) {
    absl::optional<ConcreteType*> arg_type = ctx->type_info()->GetItem(arg);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                         Resolve(**arg_type, ctx));
    arg_types.push_back(std::move(resolved));
    arg_spans.push_back(arg->span());
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
  } else if (builtin_name->identifier() == "fail!" ||
             builtin_name->identifier() == "cover!") {
    if (f != nullptr && absl::holds_alternative<Function*>(*f)) {
      ctx->type_info()->NoteRequiresImplicitToken(absl::get<Function*>(*f),
                                                  true);
    } else {
      return TypeInferenceErrorStatus(
          invocation->span(), nullptr,
          "Observed a fail!() or cover!() outside of a function.");
    }

    if (builtin_name->identifier() == "cover!") {
      // Make sure that the coverpoint's identifier is valid in both Verilog
      // and DSLX - notably, we don't support Verilog escaped strings.
      // TODO(rspringer): 2021-05-26: Ensure only one instance of an identifier
      // in a design.
      String* identifier_node = dynamic_cast<String*>(invocation->args()[0]);
      XLS_RET_CHECK(identifier_node != nullptr);
      if (identifier_node->text().empty()) {
        return InvalidIdentifierErrorStatus(
            invocation->span(),
            "An identifier must be specified with a cover! op.");
      }

      std::string identifier = identifier_node->text();
      if (identifier[0] == '\\') {
        return InvalidIdentifierErrorStatus(
            invocation->span(), "Verilog escaped strings are not supported.");
      }

      // We don't support Verilog "escaped strings", so we only have to worry
      // about regular identifier matching.
      if (!RE2::FullMatch(identifier, "[a-zA-Z_][a-zA-Z0-9$_]*")) {
        return InvalidIdentifierErrorStatus(
            invocation->span(),
            "A coverpoint identifer must start with a letter or underscore, "
            "and otherwise consist of letters, digits, underscores, and/or "
            "dollar signs.");
      }
    }
  }

  std::vector<const ConcreteType*> arg_type_ptrs;
  arg_type_ptrs.reserve(arg_types.size());
  for (auto& arg_type : arg_types) {
    arg_type_ptrs.push_back(arg_type.get());
  }

  XLS_VLOG(5) << "Instantiating builtin parametric: "
              << builtin_name->identifier();
  XLS_ASSIGN_OR_RETURN(SignatureFn fsignature, GetParametricBuiltinSignature(
                                                   builtin_name->identifier()));

  // Callback that signatures can use to request constexpr evaluation of their
  // arguments -- this is a special builtin superpower used by e.g. range().
  auto constexpr_eval = [&](int64_t argno) -> absl::StatusOr<InterpValue> {
    Expr* arg = invocation->args()[argno];

    auto env = MakeConstexprEnv(arg, ctx->fn_stack().back().symbolic_bindings(),
                                ctx->type_info());

    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Interpreter::InterpretExpr(
            arg->owner(), ctx->type_info(), ctx->typecheck_module(),
            ctx->additional_search_paths(), ctx->import_data(), env, arg));
    ctx->type_info()->NoteConstExpr(arg, value);
    return value;
  };

  absl::optional<std::vector<ParametricConstraint>> parametric_constraints;
  if (higher_order_parametric_bindings.has_value()) {
    XLS_ASSIGN_OR_RETURN(parametric_constraints,
                         ParametricBindingsToConstraints(
                             higher_order_parametric_bindings.value(), ctx));
  }
  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      fsignature(
          SignatureData{arg_type_ptrs, arg_spans, builtin_name->identifier(),
                        invocation->span(), std::move(parametric_constraints),
                        constexpr_eval},
          ctx));
  FunctionType* fn_type = dynamic_cast<FunctionType*>(tab.type.get());
  XLS_RET_CHECK(fn_type != nullptr) << tab.type->ToString();

  const SymbolicBindings& fn_symbolic_bindings =
      ctx->fn_stack().back().symbolic_bindings();
  XLS_VLOG(5) << "TypeInfo::AddInstantiationCallBindings; type_info: "
              << ctx->type_info() << "; node: `" << invocation->ToString()
              << "`; caller: " << fn_symbolic_bindings
              << "; callee: " << tab.symbolic_bindings;
  ctx->type_info()->AddInstantiationCallBindings(
      invocation, /*caller=*/fn_symbolic_bindings,
      /*callee=*/tab.symbolic_bindings);
  ctx->type_info()->SetItem(invocation->callee(), *fn_type);
  ctx->type_info()->SetItem(invocation, fn_type->return_type());

  if (builtin_name->identifier() == "map") {
    return InstantiateBuiltinParametricMap(tab, *map_fn_name, *map_fn,
                                           invocation, ctx);
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
  XLS_VLOG(6) << absl::StreamFormat("typecheck stack (%d):", stack.size());
  for (const TypecheckStackRecord& record : stack) {
    XLS_VLOG(6) << absl::StreamFormat(
        "  name: '%s' kind: %s user: %s", record.name, record.kind,
        record.user == nullptr ? "none" : record.user->ToString());
  }
}

static void VLogSeen(
    const absl::flat_hash_map<std::pair<std::string, std::string>, WipRecord>&
        map) {
  XLS_VLOG(6) << "seen map:";
  for (const auto& item : map) {
    XLS_VLOG(6) << absl::StreamFormat(
        "  %s :: %s => %s wip: %s", item.first.first, item.first.second,
        Identifier(item.second.f), item.second.wip ? "true" : "false");
  }
}

// Handles a TypeMissingError() that results from an attempt at deductive
// typechecking of a TopNode.
//
// * For a name that refers to a function in the "function_map", we push that
//   onto the top of the processing stack.
//
// Args:
//  f: The top node being processed when we observed the missing type status --
//    this is optional since we can encounter e.g. a parametric function in a
//    constant definition at the top level, so there's no TopNode to speak of in
//    that case.
//  status: The missing type status, this encodes the node that had the type
//    error and its user node that discovered the type was missing.
static absl::Status HandleMissingType(
    const TopNode* f, const absl::Status& status,
    const absl::flat_hash_map<std::string, Function*>& function_map,
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
      XLS_VLOG(5) << "Added a stack record for callee: "
                  << callee->identifier();
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
            InstantiateBuiltinParametric(f, builtin_name_def, invocation, ctx));
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

// Returns a key suitable for use in a "seen" set (maintained during
// typechecking traversal).
static std::pair<std::string, std::string> ToSeenKey(const TopNode& top_node) {
  return std::pair<std::string, std::string>{
      Identifier(top_node), ToAstNode(top_node)->GetNodeTypeName()};
}

static absl::Status CheckTopNodeInModuleInternal(
    const absl::flat_hash_map<std::string, Function*>& function_map,
    absl::flat_hash_map<std::pair<std::string, std::string>, WipRecord>& seen,
    std::vector<TypecheckStackRecord>& stack, DeduceCtx* ctx) {
  while (!stack.empty()) {
    XLS_VLOG(6) << "Stack still not empty...";
    VLogStack(stack);
    VLogSeen(seen);
    const TypecheckStackRecord& record = stack.back();
    const TopNode f = seen.at({record.name, record.kind}).f;

    absl::Status status = ProcessTopNode(f, ctx);
    XLS_VLOG(6) << "Process top node status: " << status
                << "; f: " << ToAstNode(f)->ToString();

    if (IsTypeMissingErrorStatus(status)) {
      XLS_RETURN_IF_ERROR(
          HandleMissingType(&f, status, function_map, seen, stack, ctx));
      continue;
    }

    XLS_RETURN_IF_ERROR(status);

    XLS_VLOG(5) << "Marking as done: " << Identifier(f);
    seen[ToSeenKey(f)] = WipRecord{f, /*wip=*/false};  // Mark as done.
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

absl::Status CheckTopNodeInModule(TopNode f, DeduceCtx* ctx) {
  XLS_RET_CHECK(ToAstNode(f) != nullptr);

  absl::flat_hash_map<std::pair<std::string, std::string>, WipRecord> seen;
  seen.emplace(ToSeenKey(f), WipRecord{f, /*wip=*/true});

  std::vector<TypecheckStackRecord> stack = {TypecheckStackRecord{
      Identifier(f), std::string(ToAstNode(f)->GetNodeTypeName())}};

  const absl::flat_hash_map<std::string, Function*>& function_map =
      ctx->module()->GetFunctionByName();
  return CheckTopNodeInModuleInternal(function_map, seen, stack, ctx);
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
      int64_t depth_after_push = depth_before_ + 1;
      XLS_CHECK_EQ(ctx_->fn_stack().size(), depth_after_push);
      ctx_->fn_stack().pop_back();
    }
  }

 private:
  DeduceCtx* ctx_;
  int64_t depth_before_;
  bool expect_popped_;
};

// Typechecks top level "member" within "module".
static absl::Status CheckModuleMember(ModuleMember member, Module* module,
                                      ImportData* import_data, DeduceCtx* ctx) {
  XLS_VLOG(6) << "WIP for " << module->name() << " is "
              << ToAstNode(member)->ToString();
  if (absl::holds_alternative<Import*>(member)) {
    Import* import = absl::get<Import*>(member);
    XLS_ASSIGN_OR_RETURN(
        const ModuleInfo* imported,
        DoImport(ctx->typecheck_module(), ImportTokens(import->subject()),
                 ctx->additional_search_paths(), import_data, import->span()));
    ctx->type_info()->AddImport(import, imported->module.get(),
                                imported->type_info);
  } else if (absl::holds_alternative<ConstantDef*>(member) ||
             absl::holds_alternative<EnumDef*>(member)) {
    ScopedFnStackEntry scoped(ctx, module);
    absl::Status status = ctx->Deduce(ToAstNode(member)).status();
    if (IsTypeMissingErrorStatus(status)) {
      // If we got a type missing error from a constant definition or enum
      // definition, there's likely a parametric function used in the definition
      // we need to instantiate -- call to HandleMissingType to form the
      // required stack, check that stack, and try again.
      //
      // TODO(leary): 2021-03-29 We can probably cut down on the boilerplate
      // here with some refactoring TLC.
      const absl::flat_hash_map<std::string, Function*>& function_map =
          ctx->module()->GetFunctionByName();
      absl::flat_hash_map<std::pair<std::string, std::string>, WipRecord> seen;
      std::vector<TypecheckStackRecord> stack;
      XLS_RETURN_IF_ERROR(HandleMissingType(/*f=*/nullptr, status, function_map,
                                            seen, stack, ctx));
      XLS_RETURN_IF_ERROR(
          CheckTopNodeInModuleInternal(function_map, seen, stack, ctx));
      status = CheckModuleMember(member, module, import_data, ctx);
    }
    XLS_RETURN_IF_ERROR(status);
    scoped.Finish();
  } else if (absl::holds_alternative<Function*>(member)) {
    Function* f = absl::get<Function*>(member);
    if (f->IsParametric()) {
      // Typechecking of parametric functions is driven by invocation sites.
      return absl::OkStatus();
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
  } else if (absl::holds_alternative<QuickCheck*>(member)) {
    QuickCheck* qc = absl::get<QuickCheck*>(member);
    Function* f = qc->f();
    if (f->IsParametric()) {
      // TODO(leary): 2020-08-09 Add support for quickchecking parametric
      // functions.
      return TypeInferenceErrorStatus(
          f->span(), nullptr,
          "Quickchecking parametric functions is unsupported; see "
          "https://github.com/google/xls/issues/81");
    }

    XLS_VLOG(2) << "Typechecking quickcheck function: " << f->ToString();
    ScopedFnStackEntry scoped(f, ctx, module);
    XLS_RETURN_IF_ERROR(CheckTopNodeInModule(f, ctx));
    scoped.Finish();

    absl::optional<const ConcreteType*> quickcheck_f_body_type =
        ctx->type_info()->GetItem(f->body());
    XLS_RET_CHECK(quickcheck_f_body_type.has_value());
    auto u1 = BitsType::MakeU1();
    if (*quickcheck_f_body_type.value() != *u1) {
      return XlsTypeErrorStatus(f->span(), *quickcheck_f_body_type.value(), *u1,
                                "Quickcheck functions must return a bool.");
    }

    XLS_VLOG(2) << "Finished typechecking quickcheck function: "
                << f->ToString();
  } else {
    XLS_RET_CHECK(absl::holds_alternative<TestFunction*>(member));
    TestFunction* test = absl::get<TestFunction*>(member);
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

    XLS_VLOG(2) << "Typechecking test: " << test->ToString();
    ScopedFnStackEntry scoped(test->fn(), ctx, module);
    XLS_RETURN_IF_ERROR(CheckTopNodeInModule(test->fn(), ctx));
    scoped.Finish();
    XLS_VLOG(2) << "Finished typechecking test: " << test->ToString();
  }
  return absl::OkStatus();
}

absl::StatusOr<TypeInfo*> CheckModule(
    Module* module, ImportData* import_data,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  // Create a deduction context to use for checking this module.
  std::vector<std::filesystem::path> additional_search_paths_copy(
      additional_search_paths.begin(), additional_search_paths.end());
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->type_info_owner().New(module));
  auto ftypecheck = [import_data, additional_search_paths_copy](
                        Module* module) -> absl::StatusOr<TypeInfo*> {
    return CheckModule(module, import_data, additional_search_paths_copy);
  };
  DeduceCtx deduce_ctx(type_info, module,
                       /*deduce_function=*/&Deduce,
                       /*typecheck_function=*/&CheckTopNodeInModule,
                       /*typecheck_module=*/ftypecheck, additional_search_paths,
                       import_data);
  DeduceCtx* ctx = &deduce_ctx;

  // First, populate type info with constants, enums, resolved imports, and
  // non-parametric functions.
  for (ModuleMember& member : *module->mutable_top()) {
    import_data->SetTypecheckWorkInProgress(module, ToAstNode(member));
    XLS_RETURN_IF_ERROR(CheckModuleMember(member, module, import_data, ctx));
  }

  // Make a note that we completed typechecking this module in the import data.
  import_data->SetTypecheckWorkInProgress(module, nullptr);
  return type_info;
}

}  // namespace xls::dslx
