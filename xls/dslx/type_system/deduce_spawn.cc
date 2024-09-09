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

#include "xls/dslx/type_system/deduce_spawn.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/ast_env.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_invocation.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

// Resolves "ref" to an AST proc.
absl::StatusOr<Proc*> ResolveColonRefToProc(const ColonRef* ref,
                                            DeduceCtx* ctx) {
  std::optional<Import*> import = ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << ref->ToString();
  std::optional<const ImportedInfo*> imported_info =
      ctx->type_info()->GetImported(*import);
  return GetMemberOrTypeInferenceError<Proc>(imported_info.value()->module,
                                             ref->attr(), ref->span());
}

// We need to evaluate/check `const_assert!`s at typechecking time; things like
// parametrics are only instantiated when a `spawn` is encountered, at which
// point we can check `const_assert!`s pass.
absl::Status TypecheckProcConstAsserts(const Proc& p, DeduceCtx* ctx) {
  for (const ConstAssert* n : p.GetConstAssertStmts()) {
    XLS_RETURN_IF_ERROR(ctx->Deduce(n).status());
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<Type>> DeduceSpawn(const Spawn* node,
                                                  DeduceCtx* ctx) {
  const ParametricEnv caller_parametric_env = ctx->GetCurrentParametricEnv();
  VLOG(5) << "Deducing type for spawn: `" << node->ToString()
          << "` caller symbolic bindings: " << caller_parametric_env;

  auto resolve_proc = [](const Instantiation* node,
                         DeduceCtx* ctx) -> absl::StatusOr<Proc*> {
    Expr* callee = node->callee();
    Proc* proc;
    if (auto* colon_ref = dynamic_cast<ColonRef*>(callee)) {
      XLS_ASSIGN_OR_RETURN(proc, ResolveColonRefToProc(colon_ref, ctx));
    } else {
      auto* name_ref = dynamic_cast<NameRef*>(callee);
      XLS_RET_CHECK(name_ref != nullptr);
      const std::string& callee_name = name_ref->identifier();
      XLS_ASSIGN_OR_RETURN(
          proc, GetMemberOrTypeInferenceError<Proc>(ctx->module(), callee_name,
                                                    name_ref->span()));
    }
    return proc;
  };

  // Resolve the proc AST node that's being instantiated.
  //
  // Note that this can be from a different module than the spawn / spawner.
  XLS_ASSIGN_OR_RETURN(Proc * proc, resolve_proc(node, ctx));

  auto resolve_config = [proc](const Instantiation* node,
                               DeduceCtx* ctx) -> absl::StatusOr<Function*> {
    return &proc->config();
  };
  auto resolve_next = [proc](const Instantiation* node,
                             DeduceCtx* ctx) -> absl::StatusOr<Function*> {
    return &proc->next();
  };

  auto resolve_init = [proc](const Instantiation* node,
                             DeduceCtx* ctx) -> absl::StatusOr<Function*> {
    return &proc->init();
  };

  XLS_RETURN_IF_ERROR(
      DeduceInstantiation(ctx, down_cast<Invocation*>(node->next()->args()[0]),
                          /*resolve_fn=*/resolve_init,
                          /*constexpr_env=*/{})
          .status());

  // Gather up the type of all the (actual) arguments.
  std::vector<InstantiateArg> config_args;
  XLS_RETURN_IF_ERROR(AppendArgsForInstantiation(
      node, node->callee(), node->config()->args(), ctx, &config_args));

  std::vector<InstantiateArg> next_args;
  XLS_RETURN_IF_ERROR(AppendArgsForInstantiation(
      node, node->callee(), node->next()->args(), ctx, &next_args));

  // For each [constexpr] arg, mark the associated Param as constexpr.
  AstEnv constexpr_env;
  size_t argc = node->config()->args().size();
  size_t paramc = proc->config().params().size();
  if (argc != paramc) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("spawn had wrong argument count; want: %d got: %d",
                        paramc, argc),
        ctx->file_table());
  }
  for (int i = 0; i < node->config()->args().size(); i++) {
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         ConstexprEvaluator::EvaluateToValue(
                             ctx->import_data(), ctx->type_info(),
                             ctx->warnings(), ctx->GetCurrentParametricEnv(),
                             node->config()->args()[i], nullptr));
    constexpr_env.Add(proc->config().params()[i], value);
  }

  // TODO(rspringer): 2022-05-26: We can't currently lazily evaluate `next` args
  // in the BytecodeEmitter, since that'd lead to a circular dependency between
  // it and the ConstexprEvaluator, so we have to do it eagerly here.
  // Un-wind that, if possible.
  XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
      ctx->import_data(), ctx->type_info(), ctx->warnings(),
      ctx->GetCurrentParametricEnv(),
      down_cast<Invocation*>(node->next()->args()[0]),
      /*type=*/nullptr));

  XLS_RETURN_IF_ERROR(
      DeduceInstantiation(ctx, node->config(), resolve_config, constexpr_env)
          .status());

  XLS_ASSIGN_OR_RETURN(TypeInfo * config_ti,
                       ctx->type_info()->GetInvocationTypeInfoOrError(
                           node->config(), caller_parametric_env));

  {
    std::unique_ptr<DeduceCtx> proc_level_ctx =
        ctx->MakeCtx(config_ti, proc->owner());
    proc_level_ctx->fn_stack().push_back(FnStackEntry::MakeTop(proc->owner()));
    XLS_RETURN_IF_ERROR(TypecheckProcConstAsserts(*proc, proc_level_ctx.get()));
  }

  // Now we need to get the [constexpr] Proc member values so we can set them
  // when typechecking the `next` function. Those values are the elements in the
  // `config` function's terminating XlsTuple.
  // 1. Get the last statement in the `config` function.
  absl::Nullable<const XlsTuple*> config_tuple = proc->GetConfigTuple();

  // 2. Extract the value of each element and associate with the corresponding
  // Proc member (in decl. order).
  constexpr_env.Clear();

  if (config_tuple == nullptr) {
    if (!proc->members().empty()) {
      return TypeInferenceErrorStatus(
          proc->config().span(), nullptr,
          absl::StrFormat("Proc '%s' has %d members to configure, but no proc "
                          "configuration tuple was provided",
                          proc->identifier(), proc->members().size()),
          ctx->file_table());
    }
  } else {
    // When a config tuple is present (and returned from the `config()`) we
    // constexpr evaluate all of its elements and place those in the constexpr
    // env.
    XLS_RET_CHECK_EQ(config_tuple->members().size(), proc->members().size());
    for (int i = 0; i < config_tuple->members().size(); i++) {
      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           ConstexprEvaluator::EvaluateToValue(
                               ctx->import_data(), config_ti, ctx->warnings(),
                               ctx->GetCurrentParametricEnv(),
                               config_tuple->members()[i], nullptr));
      constexpr_env.Add(proc->members()[i], value);
    }
  }

  // With all the proc members placed in the constexpr env, now we can deduce
  // the instantiation of `next()`, which accesses these members.
  XLS_RETURN_IF_ERROR(
      DeduceInstantiation(ctx, node->next(), resolve_next, constexpr_env)
          .status());

  return Type::MakeUnit();
}

}  // namespace xls::dslx
