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

#include "xls/dslx/type_system/deduce.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/exhaustiveness/match_exhaustiveness_checker.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/type_system/deduce_colon_ref.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_enum_def.h"
#include "xls/dslx/type_system/deduce_expr.h"
#include "xls/dslx/type_system/deduce_invocation.h"
#include "xls/dslx/type_system/deduce_spawn.h"
#include "xls/dslx/type_system/deduce_struct_def_base_utils.h"
#include "xls/dslx/type_system/deduce_struct_instance.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"

namespace xls::dslx {
namespace {

absl::StatusOr<InterpValue> EvaluateConstexprValue(DeduceCtx* ctx,
                                                   const Expr* node,
                                                   const Type* type) {
  return ConstexprEvaluator::EvaluateToValue(
      ctx->import_data(), ctx->type_info(), ctx->warnings(),
      ctx->GetCurrentParametricEnv(), node, type);
}

// Attempts to convert an expression from the full DSL AST into the
// ParametricExpression sub-AST (a limited form that we can embed into a
// TypeDim for later instantiation).
absl::StatusOr<std::unique_ptr<ParametricExpression>> ExprToParametric(
    const Expr* e, DeduceCtx* ctx) {
  if (auto* n = dynamic_cast<const NameRef*>(e)) {
    // If the NameRef refers to a constant definition, we resolve it as a
    // constant.
    if (std::holds_alternative<const NameDef*>(n->name_def())) {
      auto* name_def = std::get<const NameDef*>(n->name_def());
      std::optional<InterpValue> constant =
          ctx->type_info()->GetConstExprOption(name_def);
      if (constant.has_value()) {
        return std::make_unique<ParametricConstant>(
            std::move(constant.value()));
      }
    }

    return std::make_unique<ParametricSymbol>(n->identifier(), n->span());
  }
  if (auto* n = dynamic_cast<const Binop*>(e)) {
    XLS_ASSIGN_OR_RETURN(auto lhs, ExprToParametric(n->lhs(), ctx));
    XLS_ASSIGN_OR_RETURN(auto rhs, ExprToParametric(n->rhs(), ctx));
    switch (n->binop_kind()) {
      case BinopKind::kAdd:
        return std::make_unique<ParametricAdd>(std::move(lhs), std::move(rhs));
      case BinopKind::kDiv:
        return std::make_unique<ParametricDiv>(std::move(lhs), std::move(rhs));
      case BinopKind::kMul:
        return std::make_unique<ParametricMul>(std::move(lhs), std::move(rhs));
      default:
        return absl::InvalidArgumentError(
            "Cannot convert expression to parametric: " + e->ToString());
    }
  }
  // A `std::clog2` invocation maps to a dedicated type of
  // `ParametricExpression` called `ParametricWidth`.
  if (auto* n = dynamic_cast<const Invocation*>(e);
      n != nullptr && n->callee()->kind() == AstNodeKind::kColonRef &&
      n->callee()->ToString() == "std::clog2") {
    if (n->args().size() != 1) {
      // Note that type checking is expected to catch this before now, and this
      // is an extra precaution.
      return absl::InvalidArgumentError(
          absl::StrCat("std::clog2 expects 1 argument; got ", n->args().size(),
                       " at ", n->span().ToString(ctx->file_table())));
    }
    XLS_ASSIGN_OR_RETURN(auto arg, ExprToParametric(n->args()[0], ctx));
    return std::make_unique<ParametricWidth>(std::move(arg));
  }
  if (auto* n = dynamic_cast<const Number*>(e)) {
    auto default_type = BitsType::MakeU32();
    XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                         EvaluateConstexprValue(ctx, n, default_type.get()));
    return std::make_unique<ParametricConstant>(std::move(constexpr_value));
  }
  return absl::InvalidArgumentError(
      "Cannot convert expression to parametric: " + e->ToString());
}

absl::StatusOr<InterpValue> InterpretExpr(
    DeduceCtx* ctx, const Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BytecodeFunction> bf,
                       BytecodeEmitter::EmitExpression(
                           ctx->import_data(), ctx->type_info(), expr, env,
                           ctx->GetCurrentParametricEnv()));

  return BytecodeInterpreter::Interpret(ctx->import_data(), bf.get(),
                                        /*args=*/{});
}

absl::StatusOr<std::unique_ptr<Type>> DeduceProcMember(const ProcMember* node,
                                                       DeduceCtx* ctx) {
  VLOG(5) << "DeduceProcMember: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(auto type, ctx->Deduce(node->type_annotation()));
  auto* meta_type = dynamic_cast<MetaType*>(type.get());
  std::unique_ptr<Type>& param_type = meta_type->wrapped();

  VLOG(5) << "DeduceProcMember result: " << param_type;

  return std::move(param_type);
}

std::optional<const ChannelType*> GetChannelType(const Type* param_type) {
  // If an array: resolve to nested element type.
  const Type* element_type = param_type;
  const ArrayType* array_type = dynamic_cast<const ArrayType*>(param_type);
  if (array_type != nullptr) {
    const auto& [innermost_element_type, innermost_array_type, all_dims_known] =
        array_type->GetInnermostElementType();
    if (!all_dims_known) {
      return std::nullopt;
    }
    element_type = &innermost_element_type;
  }
  const ChannelType* channel_type =
      dynamic_cast<const ChannelType*>(element_type);
  if (channel_type != nullptr) {
    return channel_type;
  }
  return std::nullopt;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceParam(const Param* node,
                                                  DeduceCtx* ctx) {
  VLOG(5) << "DeduceParam: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(auto type, ctx->Deduce(node->type_annotation()));
  auto* meta_type = dynamic_cast<MetaType*>(type.get());
  std::unique_ptr<Type>& param_type = meta_type->wrapped();

  Function* f = dynamic_cast<Function*>(node->parent());
  if (f == nullptr) {
    VLOG(5) << "DeduceParam non function result: " << param_type;
    return std::move(param_type);
  }

  // Special case handling for parameters to config functions. These must be
  // made constexpr.
  //
  // When deducing a proc at top level, we won't have constexpr values for its
  // config params, which will cause Spawn deduction to fail, so we need to
  // create dummy InterpValues for its parameter channels.
  // Other types of params aren't allowed, example: a proc member could be
  // assigned a constexpr value based on the sum of dummy values.
  // Stack depth 2: Module "<top>" + the config function being looked at.
  bool is_root_proc =
      f->tag() == FunctionTag::kProcConfig && ctx->fn_stack().size() == 2;
  std::optional<const ChannelType*> channel_type =
      GetChannelType(param_type.get());
  bool is_channel_like_param = channel_type.has_value();
  bool is_param_constexpr = ctx->type_info()->IsKnownConstExpr(node);
  if (is_root_proc && is_channel_like_param && !is_param_constexpr) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        CreateChannelReference((*channel_type)->direction(), param_type.get()));
    ctx->type_info()->NoteConstExpr(node, value);
    ctx->type_info()->NoteConstExpr(node->name_def(), value);
  }

  VLOG(5) << "DeduceParam result: " << param_type;
  return std::move(param_type);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceConstantDef(const ConstantDef* node,
                                                        DeduceCtx* ctx) {
  VLOG(5) << "DeduceConstantDef: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> result,
                       ctx->Deduce(node->value()));
  const FnStackEntry& peek_entry = ctx->fn_stack().back();
  std::optional<FnCtx> fn_ctx;
  if (peek_entry.f() != nullptr) {
    fn_ctx.emplace(FnCtx{.module_name = peek_entry.module()->name(),
                         .fn_name = peek_entry.name(),
                         .parametric_env = peek_entry.parametric_env()});
  }

  ctx->type_info()->SetItem(node, *result);
  ctx->type_info()->SetItem(node->name_def(), *result);

  if (node->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> annotated,
                         ctx->Deduce(node->type_annotation()));
    XLS_ASSIGN_OR_RETURN(
        annotated,
        UnwrapMetaType(std::move(annotated), node->type_annotation()->span(),
                       "numeric literal type-prefix", ctx->file_table()));
    if (*annotated != *result) {
      return ctx->TypeMismatchError(node->span(), node->type_annotation(),
                                    *annotated, node->value(), *result,
                                    "Constant definition's annotated type did "
                                    "not match its expression's type");
    }
  }

  WarnOnInappropriateConstantName(node->identifier(), node->span(),
                                  *node->owner(), ctx->warnings());

  XLS_ASSIGN_OR_RETURN(
      InterpValue constexpr_value,
      EvaluateConstexprValue(ctx, node->value(), result.get()));
  ctx->type_info()->NoteConstExpr(node, constexpr_value);
  ctx->type_info()->NoteConstExpr(node->value(), constexpr_value);
  ctx->type_info()->NoteConstExpr(node->name_def(), constexpr_value);

  VLOG(5) << "DeduceConstantDef result: " << result->ToString();
  return result;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceTypeRef(const TypeRef* node,
                                                    DeduceCtx* ctx) {
  VLOG(5) << "DeduceTypeRef: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                       ctx->Deduce(ToAstNode(node->type_definition())));
  if (!type->IsMeta()) {
    return TypeInferenceErrorStatus(
        node->span(), type.get(),
        absl::StrFormat(
            "Expected type-reference to refer to a type definition, but this "
            "did not resolve to a type; instead got: `%s`.",
            type->ToString()),
        ctx->file_table());
  }

  VLOG(5) << "DeduceTypeRef result: " << type->ToString();
  return type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceTypeAlias(const TypeAlias* node,
                                                      DeduceCtx* ctx) {
  VLOG(5) << "DeduceTypeAlias: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                       ctx->Deduce(&node->type_annotation()));
  XLS_RET_CHECK(type->IsMeta());
  ctx->type_info()->SetItem(&node->name_def(), *type);

  VLOG(5) << "DeduceTypeAlias result: " << type->ToString();
  return type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceUseTreeEntry(
    const UseTreeEntry* node, DeduceCtx* ctx) {
  // Similar to a ColonRef, we have to reach in to the imported module to find
  // the type and add it.
  XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported,
                       ctx->type_info()->GetImportedOrError(node));
  const TypeInfo& imported_type_info = *imported->type_info;
  const Module& imported_module = *imported->module;
  std::optional<const ModuleMember*> member =
      imported_module.FindMemberWithName(
          node->GetLeafNameDef().value()->identifier());
  XLS_RET_CHECK(member.has_value());
  std::optional<Type*> type =
      imported_type_info.GetItem(ToAstNode(*member.value()));
  XLS_RET_CHECK(type.has_value());
  return type.value()->CloneToUnique();
}

static absl::Status BindNames(const NameDefTree* name_def_tree,
                              const Type& type, DeduceCtx* ctx,
                              std::optional<InterpValue> constexpr_value) {
  const auto set_item =
      [&](AstNode* name_def, TypeOrAnnotation type,
          std::optional<InterpValue> constexpr_value) -> absl::Status {
    if (std::holds_alternative<const Type*>(type)) {
      ctx->type_info()->SetItem(name_def, *(std::get<const Type*>(type)));
      if (constexpr_value.has_value()) {
        ctx->type_info()->NoteConstExpr(name_def, constexpr_value.value());
      }
    }
    return absl::OkStatus();
  };
  return MatchTupleNodeToType(set_item, name_def_tree, &type, ctx->file_table(),
                              constexpr_value);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceLet(const Let* node,
                                                DeduceCtx* ctx) {
  VLOG(5) << "DeduceLet: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> rhs,
                       ctx->DeduceAndResolve(node->rhs()));

  if (node->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> annotated,
                         ctx->DeduceAndResolve(node->type_annotation()));
    XLS_ASSIGN_OR_RETURN(
        annotated,
        UnwrapMetaType(std::move(annotated), node->type_annotation()->span(),
                       "let type-annotation", ctx->file_table()));
    if (*rhs != *annotated) {
      return ctx->TypeMismatchError(
          node->type_annotation()->span(), nullptr, *annotated, nullptr, *rhs,
          "Annotated type did not match inferred type "
          "of right hand side expression.");
    }
  }

  std::optional<InterpValue> maybe_constexpr_value;
  XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
      ctx->import_data(), ctx->type_info(), ctx->warnings(),
      ctx->GetCurrentParametricEnv(), node->rhs(), rhs.get()))
      << "while evaluating: " << node->rhs()->ToString();
  if (ctx->type_info()->IsKnownConstExpr(node->rhs())) {
    XLS_ASSIGN_OR_RETURN(maybe_constexpr_value,
                         ctx->type_info()->GetConstExpr(node->rhs()));
  }

  XLS_RETURN_IF_ERROR(
      BindNames(node->name_def_tree(), *rhs, ctx, maybe_constexpr_value));

  if (node->name_def_tree()->IsWildcardLeaf()) {
    ctx->warnings()->Add(
        node->name_def_tree()->span(), WarningKind::kUselessLetBinding,
        "`let _ = expr;` statement can be simplified to `expr;` -- there is no "
        "need for a `let` binding here");
  }

  if (node->is_const()) {
    TypeInfo* ti = ctx->type_info();
    XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                         ti->GetConstExpr(node->rhs()));
    ti->NoteConstExpr(node, constexpr_value);
    // Reminder: we don't allow name destructuring in constant defs, so this
    // is expected to never fail.
    XLS_RET_CHECK_EQ(node->name_def_tree()->GetNameDefs().size(), 1);

    NameDef* name_def = node->name_def_tree()->GetNameDefs()[0];
    ti->NoteConstExpr(name_def, ti->GetConstExpr(node->rhs()).value());

    WarnOnInappropriateConstantName(name_def->identifier(), node->span(),
                                    *node->owner(), ctx->warnings());
  }

  VLOG(5) << "DeduceLet rhs: " << rhs->ToString();
  ctx->type_info()->SetItem(node->name_def_tree(), *rhs);

  return Type::MakeUnit();
}

absl::StatusOr<std::unique_ptr<Type>> DeduceLambda(const Lambda* node,
                                                   DeduceCtx* ctx) {
  return absl::UnimplementedError("lambdas not yet supported in type system");
}

// The types that need to be deduced for `for`-like loops (including
// `unroll_for!`).
struct ForLoopTypes {
  // The type of the container the loop iterates through.
  std::unique_ptr<Type> iterable_type;

  // The element type of the container indicated by `iterable_type`.
  std::unique_ptr<Type> iterable_element_type;

  // The type of the loop accumulator (which is the same type as the
  // init parameter "passed in" to the loop after its body).
  std::unique_ptr<Type> accumulator_type;
};

// Deduces and type-checks the init and iterable expressions of a loop,
// returning the init type.
absl::StatusOr<ForLoopTypes> DeduceForLoopTypes(const ForLoopBase* node,
                                                DeduceCtx* ctx) {
  // Type of the init value to the for loop (also the accumulator type).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> init_type,
                       ctx->DeduceAndResolve(node->init()));

  // Type of the iterable (whose elements are being used as the induction
  // variable in the for loop).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> iterable_type,
                       ctx->DeduceAndResolve(node->iterable()));
  auto* iterable_array_type = dynamic_cast<ArrayType*>(iterable_type.get());
  if (iterable_array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->iterable()->span(), iterable_type.get(),
        "For loop iterable value is not an array.", ctx->file_table());
  }
  std::unique_ptr<Type> iterable_element_type =
      iterable_array_type->element_type().CloneToUnique();

  std::vector<std::unique_ptr<Type>> target_annotated_type_elems;
  target_annotated_type_elems.push_back(iterable_element_type->CloneToUnique());
  target_annotated_type_elems.push_back(init_type->CloneToUnique());
  auto target_annotated_type =
      std::make_unique<TupleType>(std::move(target_annotated_type_elems));

  // If there was an explicitly annotated type, ensure it matches our inferred
  // one.
  if (node->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> annotated_type,
                         ctx->DeduceAndResolve(node->type_annotation()));
    XLS_ASSIGN_OR_RETURN(
        annotated_type,
        UnwrapMetaType(std::move(annotated_type),
                       node->type_annotation()->span(),
                       "for-loop annotated type", ctx->file_table()));

    const TupleType* annotated_tuple_type =
        dynamic_cast<const TupleType*>(annotated_type.get());
    if (annotated_tuple_type == nullptr) {
      return ctx->TypeMismatchError(
          node->span(), node->type_annotation(), *annotated_type, nullptr,
          *target_annotated_type,
          "For-loop annotated type should be a tuple containing a type for the "
          "iterable and a type for the accumulator.");
    }
    const std::vector<std::unique_ptr<Type>>& annotated_tuple_members =
        annotated_tuple_type->members();
    if (annotated_tuple_members.size() != 2) {
      return ctx->TypeMismatchError(
          node->span(), node->type_annotation(), *annotated_type, nullptr,
          *target_annotated_type,
          absl::StrFormat(
              "For-loop annotated type should specify a type for the iterable "
              "and a type for the accumulator; got %d types.",
              annotated_tuple_members.size()));
    }
    if (*iterable_element_type != *annotated_tuple_members[0]) {
      return ctx->TypeMismatchError(
          node->span(), node->type_annotation(), *annotated_type, nullptr,
          *target_annotated_type,
          "For-loop annotated index type did not match inferred type.");
    }
    if (*init_type != *annotated_tuple_members[1]) {
      return ctx->TypeMismatchError(
          node->span(), node->type_annotation(), *annotated_type, nullptr,
          *target_annotated_type,
          "For-loop annotated accumulator type did not match inferred type.");
    }
  }

  // Bind the names to their associated types for use in the body.
  NameDefTree* bindings = node->names();

  if (!bindings->IsIrrefutable()) {
    return TypeInferenceErrorStatus(
        bindings->span(), nullptr,
        absl::StrFormat("for-loop bindings must be irrefutable (i.e. the "
                        "pattern must match all possible values)"),
        ctx->file_table());
  }

  XLS_RETURN_IF_ERROR(
      BindNames(bindings, *target_annotated_type, ctx, std::nullopt));

  return ForLoopTypes{.iterable_type = std::move(iterable_type),
                      .iterable_element_type = std::move(iterable_element_type),
                      .accumulator_type = std::move(init_type)};
}

// Type-checks the body of a loop, whose type should match that of the init
// expression (previously determined by the caller). The `actual_body` is either
// `node->body()` or the modified version of it, if modified by the caller (e.g.
// the unrolled copy of an `unroll_for!` body).
absl::Status TypecheckLoopBody(const ForLoopBase* node, const Expr* actual_body,
                               const Type& init_type, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> body_type,
                       ctx->DeduceAndResolve(actual_body));
  if (init_type != *body_type) {
    return ctx->TypeMismatchError(node->span(), node->init(), init_type,
                                  actual_body, *body_type,
                                  "For-loop init value type did not match "
                                  "for-loop body's result type.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Type>> DeduceFor(const For* node,
                                                DeduceCtx* ctx) {
  VLOG(5) << "DeduceFor: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(ForLoopTypes loop_types, DeduceForLoopTypes(node, ctx));
  XLS_RETURN_IF_ERROR(
      TypecheckLoopBody(node, node->body(), *loop_types.accumulator_type, ctx));
  return std::move(loop_types.accumulator_type);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceUnrollFor(const UnrollFor* node,
                                                      DeduceCtx* ctx) {
  VLOG(5) << "DeduceUnrollFor: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(ForLoopTypes loop_types, DeduceForLoopTypes(node, ctx));
  absl::StatusOr<InterpValue> iterable = EvaluateConstexprValue(
      ctx, node->iterable(), loop_types.iterable_type.get());
  if (!iterable.ok() || !iterable->HasValues()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "unroll_for! must use a constexpr iterable expression at: ",
        node->iterable()->span().ToString(ctx->file_table())));
  }
  const auto* types =
      dynamic_cast<const TupleTypeAnnotation*>(node->type_annotation());
  TypeAnnotation* index_type_annot = nullptr;
  TypeAnnotation* acc_type_annot = nullptr;
  if (types) {
    // Deducing the `ForLoopTypes` should have errored gracefully if this was
    // not the case.
    CHECK_EQ(types->members().size(), 2);
    index_type_annot = types->members()[0];
    acc_type_annot = types->members()[1];
  }
  CHECK_EQ(node->names()->nodes().size(), 2);
  const NameDefTree& index_name = *node->names()->nodes()[0];
  std::optional<NameDef*> index_def;
  if (index_name.is_leaf() &&
      std::holds_alternative<NameDef*>(index_name.leaf())) {
    index_def = std::get<NameDef*>(index_name.leaf());
  }
  const NameDefTree& acc_name = *node->names()->nodes()[1];
  std::optional<NameDefTree*> acc_def;
  std::vector<Statement*> unrolled_statements;
  // Generate an initializer like `let acc = init;` unless the accumulator is
  // an unnamed leaf. Note that it may be a destructured tuple within the
  // index-acc tuple.
  if (!acc_name.is_leaf() ||
      std::holds_alternative<NameDef*>(acc_name.leaf())) {
    acc_def = const_cast<NameDefTree*>(&acc_name);
    XLS_ASSIGN_OR_RETURN(
        Statement::Wrapped wrapped_init,
        Statement::NodeToWrapped(node->owner()->Make<Let>(
            node->span(), *acc_def, acc_type_annot, node->init(),
            /*is_const=*/false)));
    unrolled_statements.push_back(node->owner()->Make<Statement>(wrapped_init));
  }
  // Unroll the loop to a series of clones of `node->body()`. If there is a
  // declared accumulator, then all but the last iteration are unrolled as
  // `let acc = clone_of_body;`. Each clone has any references to the index
  // replaced with a literal index value.
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       iterable->GetValues());
  for (size_t i = 0; i < values->size(); i++) {
    const InterpValue& element = values->at(i);
    if (!element.IsBits()) {
      // Currently, the iterable elements have to be of bits type, to simplify
      // their conversion into literals.
      // TODO: https://github.com/google/xls/issues/704 - this restriction could
      // be relaxed.
      return absl::InvalidArgumentError(
          absl::StrCat("unroll_for! must iterate through a range or aggregate "
                       "type whose elements are all bits at: ",
                       node->iterable()->span().ToString(ctx->file_table())));
    }
    CloneReplacer index_replacer = &NoopCloneReplacer;
    if (index_def.has_value()) {
      Number* index = node->owner()->Make<Number>(
          node->iterable()->span(), element.ToString(/*humanize=*/true),
          NumberKind::kOther, index_type_annot);
      ctx->type_info()->SetItem(index, *loop_types.iterable_element_type);
      ctx->type_info()->NoteConstExpr(index, element);
      index_replacer = NameRefReplacer(*index_def, index);
    }
    XLS_ASSIGN_OR_RETURN(
        AstNode * clone,
        CloneAst(node->body(),
                 ChainCloneReplacers(std::move(index_replacer),
                                     &PreserveTypeDefinitionsReplacer)));
    AstNode* iteration = clone;
    if (acc_def.has_value() && i < values->size() - 1) {
      iteration =
          node->owner()->Make<Let>(node->span(), *acc_def, acc_type_annot,
                                   down_cast<StatementBlock*>(clone),
                                   /*is_const=*/false);
    }
    XLS_ASSIGN_OR_RETURN(Statement::Wrapped wrapped,
                         Statement::NodeToWrapped(iteration));
    unrolled_statements.push_back(node->owner()->Make<Statement>(wrapped));
  }
  // Store the unrolled loop, deduce its type, and use that as the type of the
  // `unroll_for!`.
  StatementBlock* unrolled = node->owner()->Make<StatementBlock>(
      node->span(), std::move(unrolled_statements), /*trailing_semi=*/false);
  unrolled->SetParentNonLexical(node->parent());
  ctx->type_info()->NoteUnrolledLoop(node, ctx->GetCurrentParametricEnv(),
                                     unrolled);
  XLS_RETURN_IF_ERROR(
      TypecheckLoopBody(node, unrolled, *loop_types.accumulator_type, ctx));
  return std::move(loop_types.accumulator_type);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceCast(const Cast* node,
                                                 DeduceCtx* ctx) {
  VLOG(5) << "DeduceCast: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                       ctx->DeduceAndResolve(node->type_annotation()));
  XLS_ASSIGN_OR_RETURN(
      type, UnwrapMetaType(std::move(type), node->type_annotation()->span(),
                           "cast type", ctx->file_table()));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> expr,
                       ctx->DeduceAndResolve(node->expr()));

  if (!IsAcceptableCast(/*from=*/*expr, /*to=*/*type)) {
    return ctx->TypeMismatchError(
        node->span(), node->expr(), *expr, node->type_annotation(), *type,
        absl::StrFormat("Cannot cast from expression type %s to %s.",
                        expr->ToErrorString(), type->ToErrorString()));
  }
  VLOG(5) << "DeduceCast result: " << type->ToString();

  return type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceConstAssert(const ConstAssert* node,
                                                        DeduceCtx* ctx) {
  VLOG(5) << "DeduceConstAssert: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                       ctx->DeduceAndResolve(node->arg()));
  auto want = BitsType::MakeU1();
  if (*type != *want) {
    return ctx->TypeMismatchError(
        node->span(), /*lhs_node=*/node->arg(), *type, nullptr, *want,
        "const_assert! takes a (constexpr) boolean argument");
  }

  const ParametricEnv parametric_env = ctx->GetCurrentParametricEnv();
  XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
      ctx->import_data(), ctx->type_info(), ctx->warnings(), parametric_env,
      node->arg(), type.get()));
  if (!ctx->type_info()->IsKnownConstExpr(node->arg())) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("const_assert! expression is not constexpr"),
        ctx->file_table());
  }

  XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                       ctx->type_info()->GetConstExpr(node->arg()));
  if (constexpr_value.IsFalse()) {
    XLS_ASSIGN_OR_RETURN(
        ConstexprEnvData constexpr_env_data,
        MakeConstexprEnv(ctx->import_data(), ctx->type_info(), ctx->warnings(),
                         node->arg(), parametric_env));
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("const_assert! failure: `%s` constexpr environment: %s",
                        node->arg()->ToString(),
                        EnvMapToString(constexpr_env_data.env)),
        ctx->file_table());
  }

  VLOG(5) << "DeduceConstAssert result: " << type->ToString();
  return type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceAttr(const Attr* node,
                                                 DeduceCtx* ctx) {
  VLOG(5) << "DeduceAttr: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type, ctx->Deduce(node->lhs()));
  auto* struct_type = dynamic_cast<StructType*>(type.get());
  if (struct_type == nullptr) {
    return TypeInferenceErrorStatus(node->span(), type.get(),
                                    absl::StrFormat("Expected a struct for "
                                                    "attribute access; got %s",
                                                    type->ToString()),
                                    ctx->file_table());
  }

  std::string_view attr_name = node->attr();
  if (!struct_type->HasNamedMember(attr_name)) {
    const StructDef& struct_def = struct_type->nominal_type();
    std::optional<Function*> fn = struct_def.GetImplFunction(attr_name);
    if (fn.has_value()) {
      XLS_ASSIGN_OR_RETURN(TypeInfo * fn_ti,
                           ctx->import_data()->GetRootTypeInfo((*fn)->owner()));
      std::optional<Type*> fn_type = fn_ti->GetItem(*fn);
      if (fn_type.has_value()) {
        return (*fn_type)->CloneToUnique();
      }
    }
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Struct '%s' does not have a "
                        "member with name "
                        "'%s'",
                        struct_type->nominal_type().identifier(), attr_name),
        ctx->file_table());
  }

  std::optional<const Type*> result =
      struct_type->GetMemberTypeByName(attr_name);
  XLS_RET_CHECK(result.has_value());  // We checked above we had named member.

  auto result_type = result.value()->CloneToUnique();
  VLOG(5) << "DeduceAttr result: " << result_type->ToString();
  return result_type;
}

// Returns whether "e" is definitely a meaningless expression-statement; i.e. if
// in statement context definitely has no side-effects and thus should be
// flagged.
//
// Note that some invocations of functions will have no side-effects and will be
// meaningless, but because we don't look inside of callees to see if they are
// side-effecting, we conservatively mark those as potentially useful.
static bool DefinitelyMeaninglessExpression(Expr* e) {
  absl::StatusOr<std::vector<AstNode*>> nodes_under_e =
      CollectUnder(e, /*want_types=*/true);
  if (!nodes_under_e.ok()) {
    LOG(WARNING) << "Could not collect nodes under `" << e->ToString()
                 << "`; status: " << nodes_under_e.status();
    return false;
  }
  for (AstNode* n : nodes_under_e.value()) {
    // In the DSL side effects can only be caused by invocations or
    // invocation-like AST nodes.
    switch (n->kind()) {
      case AstNodeKind::kInvocation:
      case AstNodeKind::kFormatMacro:
      case AstNodeKind::kSpawn:
        return false;
      default:
        continue;
    }
  }
  return true;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceStatement(const Statement* node,
                                                      DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> result,
                       ctx->Deduce(ToAstNode(node->wrapped())));
  return result;
}

// Warns if the next-to-last statement in a block has a trailing semi and the
// last statement is a nil tuple expression, as this is redundant; i.e.
//
//    {
//      foo;
//      ()  <-- useless, semi on previous statement implies it
//    }
static void DetectUselessTrailingTuplePattern(const StatementBlock* block,
                                              DeduceCtx* ctx) {
  // TODO(https://github.com/google/xls/issues/1124) 2023-08-31 Proc config
  // parsing functions synthesize a tuple at the end, and we don't want to flag
  // that since the user didn't even create it.
  if (block->parent()->kind() == AstNodeKind::kFunction &&
      dynamic_cast<const Function*>(block->parent())->tag() ==
          FunctionTag::kProcConfig) {
    return;
  }

  // Need at least a statement (i.e. with semicolon after it) and an
  // expression-statement at the end to match this pattern.
  if (block->statements().size() < 2) {
    return;
  }

  // Make sure we ignore this if we're only following an implicit prologue (as
  // is used to convert implicit-token-parameter semantics for now).
  // TODO(https://github.com/google/xls/issues/1401): Remove once we no longer
  // support implicit token parameter semantics.
  const Statement* next_to_last_stmt =
      block->statements()[block->statements().size() - 2];
  if (next_to_last_stmt->GetSpan().has_value() &&
      next_to_last_stmt->GetSpan()->limit() <=
          block->span().start().BumpCol()) {
    return;
  }

  // Trailing statement has to be an expression-statement.
  const Statement* last_stmt = block->statements().back();
  if (!std::holds_alternative<Expr*>(last_stmt->wrapped())) {
    return;
  }

  // It has to be a tuple.
  const auto* last_expr = std::get<Expr*>(last_stmt->wrapped());
  auto* trailing_tuple = dynamic_cast<const XlsTuple*>(last_expr);
  if (trailing_tuple == nullptr) {
    return;
  }

  // Tuple has to be nil.
  if (!trailing_tuple->empty()) {
    return;
  }

  ctx->warnings()->Add(
      trailing_tuple->span(), WarningKind::kTrailingTupleAfterSemi,
      absl::StrFormat("Block has a trailing nil (empty) tuple after a "
                      "semicolon -- this is implied, please remove it"));
}

absl::StatusOr<std::unique_ptr<Type>> DeduceStatementBlock(
    const StatementBlock* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceStatementBlock: " << node->ToString();

  std::unique_ptr<Type> last;
  for (const Statement* s : node->statements()) {
    XLS_ASSIGN_OR_RETURN(last, ctx->Deduce(s));
  }
  // If there's a trailing semicolon this block always yields unit `()`.
  if (node->trailing_semi()) {
    last = Type::MakeUnit();
  }

  // We only want to check the last statement for "useless expression-statement"
  // property if it is not yielding a value from a block; e.g.
  //
  //    {
  //      my_invocation!();
  //      u32:42  // <- ok, no trailing semi
  //    }
  //
  // vs
  //
  //    {
  //      my_invocation!();
  //      u32:42;  // <- useless, trailing semi means block yields nil
  //    }
  const bool should_check_last_statement = node->trailing_semi();
  for (int64_t i = 0; i < static_cast<int64_t>(node->statements().size()) -
                              (should_check_last_statement ? 0 : 1);
       ++i) {
    const Statement* s = node->statements()[i];
    if (std::holds_alternative<Expr*>(s->wrapped()) &&
        DefinitelyMeaninglessExpression(std::get<Expr*>(s->wrapped()))) {
      Expr* e = std::get<Expr*>(s->wrapped());
      ctx->warnings()->Add(e->span(), WarningKind::kUselessExpressionStatement,
                           absl::StrFormat("Expression statement `%s` appears "
                                           "useless (i.e. has no side-effects)",
                                           e->ToString()));
    }
  }

  DetectUselessTrailingTuplePattern(node, ctx);
  return last;
}

static absl::StatusOr<std::unique_ptr<Type>> DeduceWidthSliceType(
    const Index* node, const Type& subject_type,
    const BitsLikeProperties& subject_bits_like, const WidthSlice& width_slice,
    DeduceCtx* ctx) {
  // Start expression; e.g. in `x[a+:u4]` this is `a`.
  Expr* start = width_slice.start();

  // Determined type of the start expression (must be bits kind).
  std::optional<BitsLikeProperties> start_bits_like;

  if (Number* start_number = dynamic_cast<Number*>(start);
      start_number != nullptr && start_number->type_annotation() == nullptr) {
    // A literal number with no annotated type as the slice start.
    //
    // By default, we use the "subject" type (converted to unsigned) as the type
    // for the slice start.
    start_bits_like.emplace(
        BitsLikeProperties{.is_signed = TypeDim::CreateBool(false),
                           .size = subject_bits_like.size});

    // Get the start number as an integral value, after we make sure it fits.
    XLS_ASSIGN_OR_RETURN(Bits start_bits,
                         start_number->GetBits(64, ctx->file_table()));
    XLS_ASSIGN_OR_RETURN(int64_t start_int, start_bits.ToInt64());

    if (start_int < 0) {
      return TypeInferenceErrorStatus(
          start_number->span(), nullptr,
          absl::StrFormat("Width-slice start value cannot be negative, only "
                          "unsigned values are permitted; got start value: %d.",
                          start_int),
          ctx->file_table());
    }

    XLS_ASSIGN_OR_RETURN(int64_t bit_count, start_bits_like->size.GetAsInt64());

    // Make sure the start_int literal fits in the type we determined.
    absl::Status fits_status = SBitsWithStatus(start_int, bit_count).status();
    if (!fits_status.ok()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::StrFormat("Cannot fit slice start %d in %d bits (width "
                          "inferred from slice subject).",
                          start_int, bit_count),
          ctx->file_table());
    }

    BitsType start_type(/*is_signed=*/false, bit_count);
    ctx->type_info()->SetItem(start, start_type);
    XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
        ctx->import_data(), ctx->type_info(), ctx->warnings(),
        ctx->GetCurrentParametricEnv(), start_number, &start_type));
  } else {
    // Aside from a bare literal (with no type) we should be able to deduce the
    // start expression's type.
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> start_type, ctx->Deduce(start));
    start_bits_like = GetBitsLike(*start_type);
    if (!start_bits_like.has_value()) {
      return TypeInferenceErrorStatus(
          start->span(), start_type.get(),
          "Start expression for width slice must be bits typed.",
          ctx->file_table());
    }
  }

  XLS_ASSIGN_OR_RETURN(bool start_is_signed,
                       start_bits_like->is_signed.GetAsBool());

  // Validate that the start is unsigned.
  if (start_is_signed) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        "Start index for width-based slice must be unsigned.",
        ctx->file_table());
  }

  // If the width of the width_type is bigger than the subject, we flag an
  // error (prevent requesting over-slicing at compile time).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> width_type,
                       ctx->Deduce(width_slice.width()));
  XLS_ASSIGN_OR_RETURN(
      width_type,
      UnwrapMetaType(std::move(width_type), width_slice.width()->span(),
                     "width slice type", ctx->file_table()));

  XLS_ASSIGN_OR_RETURN(TypeDim width_ctd, width_type->GetTotalBitCount());
  const TypeDim& subject_ctd = subject_bits_like.size;
  if (std::holds_alternative<InterpValue>(width_ctd.value()) &&
      std::holds_alternative<InterpValue>(subject_ctd.value())) {
    XLS_ASSIGN_OR_RETURN(int64_t width_bits, width_ctd.GetAsInt64());
    XLS_ASSIGN_OR_RETURN(int64_t subject_bits, subject_ctd.GetAsInt64());
    if (width_bits > subject_bits) {
      return ctx->TypeMismatchError(
          start->span(), nullptr, subject_type, nullptr, *width_type,
          absl::StrFormat("Slice type must have <= original number of bits; "
                          "attempted slice from %d to %d bits.",
                          subject_bits, width_bits));
    }
  }

  // Validate that the width type is bits-based (e.g. no enums, since sliced
  // value could be out of range of the valid enum values).
  if (!IsBitsLike(*width_type)) {
    return TypeInferenceErrorStatus(
        node->span(), width_type.get(),
        "A bits type is required for a width-based slice.", ctx->file_table());
  }

  // The width type is the thing returned from the width-slice.
  return width_type;
}

// Attempts to resolve one of the bounds (start or limit) of slice into a
// DSLX-compile-time constant.
static absl::StatusOr<std::optional<int64_t>> TryResolveBound(
    Slice* slice, Expr* bound, std::string_view bound_name, Type* s32,
    const absl::flat_hash_map<std::string, InterpValue>& env, DeduceCtx* ctx) {
  if (bound == nullptr) {
    return std::nullopt;
  }

  absl::StatusOr<InterpValue> bound_value = InterpretExpr(ctx, bound, env);
  if (!bound_value.ok()) {
    const absl::Status& status = bound_value.status();
    if (absl::StrContains(status.message(), "could not find slot or binding")) {
      return TypeInferenceErrorStatus(
          bound->span(), nullptr,
          absl::StrFormat(
              "Unable to resolve slice %s to a compile-time constant.",
              bound_name),
          ctx->file_table());
    }
  }
  // Return error if the slice bound is not signed.
  if (bound_value->tag() != InterpValueTag::kSBits) {
    std::string error_suffix = ".";
    if (bound_value->tag() == InterpValueTag::kUBits) {
      error_suffix = " -- consider casting to a signed value?";
    }
    return TypeInferenceErrorStatus(
        bound->span(), nullptr,
        absl::StrFormat(
            "Slice %s must be a signed compile-time-constant value%s",
            bound_name, error_suffix),
        ctx->file_table());
  }

  XLS_ASSIGN_OR_RETURN(int64_t as_64b, bound_value->GetBitValueViaSign());
  VLOG(3) << absl::StreamFormat("Slice %s bound @ %s has value: %d", bound_name,
                                bound->span().ToString(ctx->file_table()),
                                as_64b);
  return as_64b;
}

// Deduces the concrete type for an Index AST node with a slice spec.
//
// Precondition: node->rhs() is either a Slice or a WidthSlice.
static absl::StatusOr<std::unique_ptr<Type>> DeduceSliceType(
    const Index* node, DeduceCtx* ctx, std::unique_ptr<Type> lhs_type) {
  VLOG(5) << "DeduceSliceType: " << node->ToString();

  std::optional<BitsLikeProperties> lhs_bits_like = GetBitsLike(*lhs_type);
  if (!lhs_bits_like.has_value()) {
    // TODO(leary): 2019-10-28 Only slicing bits types for now, and only with
    // Number AST nodes, generalize to arrays and constant expressions.
    return TypeInferenceErrorStatus(node->span(), lhs_type.get(),
                                    "Value to slice is not of 'bits' type.",
                                    ctx->file_table());
  }

  XLS_ASSIGN_OR_RETURN(bool lhs_is_signed,
                       lhs_bits_like->is_signed.GetAsBool());
  if (lhs_is_signed) {
    return TypeInferenceErrorStatus(node->span(), lhs_type.get(),
                                    "Bit slice LHS must be unsigned.",
                                    ctx->file_table());
  }

  if (std::holds_alternative<WidthSlice*>(node->rhs())) {
    auto* width_slice = std::get<WidthSlice*>(node->rhs());
    return DeduceWidthSliceType(node, *lhs_type, lhs_bits_like.value(),
                                *width_slice, ctx);
  }

  XLS_ASSIGN_OR_RETURN(
      ConstexprEnvData constexpr_env_data,
      MakeConstexprEnv(ctx->import_data(), ctx->type_info(), ctx->warnings(),
                       node, ctx->GetCurrentParametricEnv()));

  std::unique_ptr<BitsType> s32 = BitsType::MakeS32();
  auto* slice = std::get<Slice*>(node->rhs());

  // Constexpr evaluate start & limit, skipping deducing in the case of
  // undecorated literals.
  auto should_deduce = [](Expr* expr) {
    if (Number* number = dynamic_cast<Number*>(expr);
        number != nullptr && number->type_annotation() == nullptr) {
      return false;
    }
    return true;
  };

  if (slice->start() != nullptr) {
    if (should_deduce(slice->start())) {
      XLS_RETURN_IF_ERROR(Deduce(slice->start(), ctx).status());
    } else {
      // If the slice start is untyped, assume S32, and check it fits in that
      // size.
      XLS_RETURN_IF_ERROR(
          TryEnsureFitsInBitsType(*down_cast<Number*>(slice->start()), *s32));
      ctx->type_info()->SetItem(slice->start(), *s32);
    }
  }
  XLS_ASSIGN_OR_RETURN(std::optional<int64_t> start,
                       TryResolveBound(slice, slice->start(), "start",
                                       s32.get(), constexpr_env_data.env, ctx));

  if (slice->limit() != nullptr) {
    if (should_deduce(slice->limit())) {
      XLS_RETURN_IF_ERROR(Deduce(slice->limit(), ctx).status());
    } else {
      // If the slice limit is untyped, assume S32, and check it fits in that
      // size.
      XLS_RETURN_IF_ERROR(
          TryEnsureFitsInBitsType(*down_cast<Number*>(slice->limit()), *s32));
      ctx->type_info()->SetItem(slice->limit(), *s32);
    }
  }
  XLS_ASSIGN_OR_RETURN(std::optional<int64_t> limit,
                       TryResolveBound(slice, slice->limit(), "limit",
                                       s32.get(), constexpr_env_data.env, ctx));

  const ParametricEnv& fn_parametric_env = ctx->GetCurrentParametricEnv();
  XLS_ASSIGN_OR_RETURN(TypeDim lhs_bit_count_ctd, lhs_type->GetTotalBitCount());
  int64_t bit_count;
  if (std::holds_alternative<TypeDim::OwnedParametric>(
          lhs_bit_count_ctd.value())) {
    auto& owned_parametric =
        std::get<TypeDim::OwnedParametric>(lhs_bit_count_ctd.value());
    ParametricExpression::Evaluated evaluated =
        owned_parametric->Evaluate(ToParametricEnv(fn_parametric_env));
    InterpValue v = std::get<InterpValue>(evaluated);
    bit_count = v.GetBitValueViaSign().value();
  } else {
    XLS_ASSIGN_OR_RETURN(bit_count, lhs_bit_count_ctd.GetAsInt64());
  }
  XLS_ASSIGN_OR_RETURN(StartAndWidth saw,
                       ResolveBitSliceIndices(bit_count, start, limit));
  ctx->type_info()->AddSliceStartAndWidth(slice, fn_parametric_env, saw);

  // Make sure the start and end types match and that the limit fits.
  std::optional<BitsLikeProperties> start_bits_like;
  std::optional<BitsLikeProperties> limit_bits_like;
  if (slice->start() == nullptr && slice->limit() == nullptr) {
    start_bits_like.emplace(
        BitsLikeProperties{.is_signed = TypeDim::CreateBool(true),
                           .size = TypeDim::CreateU32(32)});
    limit_bits_like.emplace(
        BitsLikeProperties{.is_signed = TypeDim::CreateBool(true),
                           .size = TypeDim::CreateU32(32)});
  } else if (slice->start() != nullptr && slice->limit() == nullptr) {
    std::optional<Type*> start_type = ctx->type_info()->GetItem(slice->start());
    XLS_RET_CHECK(start_type.has_value());
    start_bits_like = GetBitsLike(*start_type.value());
    limit_bits_like.emplace(Clone(start_bits_like.value()));
  } else if (slice->start() == nullptr && slice->limit() != nullptr) {
    std::optional<Type*> limit_type = ctx->type_info()->GetItem(slice->limit());
    XLS_RET_CHECK(limit_type.has_value());
    limit_bits_like = GetBitsLike(*limit_type.value());
    start_bits_like.emplace(Clone(limit_bits_like.value()));
  } else {
    std::optional<Type*> start_type = ctx->type_info()->GetItem(slice->start());
    XLS_RET_CHECK(start_type.has_value());
    start_bits_like = GetBitsLike(*start_type.value());

    std::optional<Type*> limit_type = ctx->type_info()->GetItem(slice->limit());
    XLS_RET_CHECK(limit_type.has_value());
    limit_bits_like = GetBitsLike(*limit_type.value());
  }

  if (*start_bits_like != *limit_bits_like) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat(
            "Slice limit type (%s) did not match slice start type (%s).",
            ToTypeString(*limit_bits_like), ToTypeString(*start_bits_like)),
        ctx->file_table());
  }
  const TypeDim& type_width_dim = start_bits_like->size;
  XLS_ASSIGN_OR_RETURN(int64_t type_width, type_width_dim.GetAsInt64());
  if (Bits::MinBitCountSigned(saw.start + saw.width) > type_width) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Slice limit does not fit in index type: %d.",
                        saw.start + saw.width),
        ctx->file_table());
  }
  return std::make_unique<BitsType>(/*signed=*/false, saw.width);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceIndex(const Index* node,
                                                  DeduceCtx* ctx) {
  VLOG(5) << "DeduceIndex: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> lhs_type,
                       ctx->Deduce(node->lhs()));

  if (std::holds_alternative<Slice*>(node->rhs()) ||
      std::holds_alternative<WidthSlice*>(node->rhs())) {
    return DeduceSliceType(node, ctx, std::move(lhs_type));
  }
  Expr* rhs = std::get<Expr*>(node->rhs());

  XLS_RETURN_IF_ERROR(
      ValidateArrayTypeForIndex(*node, *lhs_type, ctx->file_table()));

  ctx->set_in_typeless_number_ctx(true);
  absl::Cleanup cleanup = [ctx]() { ctx->set_in_typeless_number_ctx(false); };

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> index_type,
                       ctx->Deduce(ToAstNode(rhs)));
  XLS_RET_CHECK(index_type != nullptr);

  XLS_RETURN_IF_ERROR(ValidateArrayIndex(*node, *lhs_type, *index_type,
                                         *ctx->type_info(), ctx->file_table()));

  return dynamic_cast<ArrayType*>(lhs_type.get())
      ->element_type()
      .CloneToUnique();
}

// Ensures that the name_def_tree bindings are aligned with the type "other"
// (which is the type for the matched value at this name_def_tree level).
static absl::Status Unify(NameDefTree* name_def_tree, const Type& other,
                          DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved_rhs_type,
                       ctx->Resolve(other));
  if (name_def_tree->is_leaf()) {
    auto mismatch = [&](const Type& leaf_type) -> absl::Status {
      return ctx->TypeMismatchError(
          name_def_tree->span(), nullptr, *resolved_rhs_type, nullptr,
          leaf_type,
          absl::StrFormat(
              "Conflicting types; pattern expects %s but got %s from value",
              resolved_rhs_type->ToString(), leaf_type.ToString()));
    };

    NameDefTree::Leaf leaf = name_def_tree->leaf();
    absl::Status status = absl::visit(
        Visitor{[&](NameDef* n) {
                  // Defining a name in the pattern match, we accept all types.
                  ctx->type_info()->SetItem(ToAstNode(leaf),
                                            *resolved_rhs_type);
                  return absl::OkStatus();
                },
                [](WildcardPattern*) { return absl::OkStatus(); },
                [](RestOfTuple*) { return absl::OkStatus(); },
                [&](Range* n) -> absl::Status {
                  XLS_ASSIGN_OR_RETURN(
                      Type * start_type,
                      ctx->type_info()->GetItemOrError(n->start()));
                  if (*start_type != *resolved_rhs_type) {
                    return mismatch(*start_type);
                  }
                  return absl::OkStatus();
                },
                [&](auto* n) -> absl::Status {
                  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved_leaf_type,
                                       ctx->DeduceAndResolve(n));
                  if (*resolved_leaf_type != *resolved_rhs_type) {
                    return mismatch(*resolved_leaf_type);
                  }
                  return absl::OkStatus();
                }},
        leaf);
    if (!status.ok()) {
      return status;
    }
  } else {
    const NameDefTree::Nodes& nodes = name_def_tree->nodes();
    auto* type = dynamic_cast<const TupleType*>(&other);
    if (type == nullptr) {
      return TypeInferenceErrorStatus(
          name_def_tree->span(), &other,
          "Pattern expected matched-on type to be a tuple.", ctx->file_table());
    }

    XLS_ASSIGN_OR_RETURN((auto [number_of_tuple_elements, number_of_names]),
                         GetTupleSizes(name_def_tree, type));

    int64_t tuple_index = 0;
    // Must iterate through the actual nodes size, not number_of_names, because
    // there may be a "rest of tuple" leaf which decreases the number of names.
    for (int64_t name_index = 0; name_index < nodes.size(); ++name_index) {
      NameDefTree* subtree = nodes[name_index];
      if (subtree->IsRestOfTupleLeaf()) {
        // Skip ahead.
        tuple_index += number_of_tuple_elements - number_of_names;
        continue;
      }
      const Type& subtype = type->GetMemberType(tuple_index);
      XLS_RETURN_IF_ERROR(Unify(subtree, subtype, ctx));
      tuple_index++;
    }
  }

  return absl::OkStatus();
}

static absl::Status ValidateMatchable(const Type& type, const Span& span,
                                      const FileTable& file_table) {
  class MatchableTypeVisitor : public TypeVisitor {
   public:
    MatchableTypeVisitor(const Span& span, const FileTable& file_table)
        : span_(span), file_table_(file_table) {}
    ~MatchableTypeVisitor() override = default;
    absl::Status HandleBits(const BitsType& type) override {
      return absl::OkStatus();
    }
    absl::Status HandleEnum(const EnumType& type) override {
      return absl::OkStatus();
    }
    absl::Status HandleTuple(const TupleType& type) override {
      for (const auto& member : type.members()) {
        XLS_RETURN_IF_ERROR(member->Accept(*this));
      }
      return absl::OkStatus();
    }
    absl::Status HandleArray(const ArrayType& type) override {
      std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
      if (bits_like.has_value()) {
        return absl::OkStatus();
      }
      return Error(type);
    }
    // Note: this should not be directly observable outside of the array type
    // element.
    absl::Status HandleBitsConstructor(
        const BitsConstructorType& type) override {
      return Error(type);
    }
    // -- these types are not matchable
    absl::Status HandleMeta(const MetaType& type) override {
      return Error(type);
    }
    absl::Status HandleFunction(const FunctionType& type) override {
      return Error(type);
    }
    absl::Status HandleChannel(const ChannelType& type) override {
      return Error(type);
    }
    absl::Status HandleToken(const TokenType& type) override {
      return Error(type);
    }
    absl::Status HandleStruct(const StructType& type) override {
      return Error(type);
    }
    absl::Status HandleProc(const ProcType& type) override {
      return Error(type);
    }
    absl::Status HandleModule(const ModuleType& type) override {
      return Error(type);
    }

   private:
    absl::Status Error(const Type& type) {
      return TypeInferenceErrorStatus(
          span_, &type, "Match construct cannot match on this type.",
          file_table_);
    };

    const Span& span_;
    const FileTable& file_table_;
  };
  MatchableTypeVisitor visitor(span, file_table);
  return type.Accept(visitor);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceMatch(const Match* node,
                                                  DeduceCtx* ctx) {
  VLOG(5) << "DeduceMatch: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> matched,
                       ctx->Deduce(node->matched()));

  // Validate that we can match on the type presented.
  //
  // The fact the type is matchable is assumed as a precondition in
  // exhaustiveness checking.
  XLS_RETURN_IF_ERROR(
      ValidateMatchable(*matched, node->span(), ctx->file_table()));

  if (node->arms().empty()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        "Match construct has no arms, cannot determine its type.",
        ctx->file_table());
  }

  MatchExhaustivenessChecker exhaustiveness_checker(
      node->matched()->span(), *ctx->import_data(), *ctx->type_info(),
      *matched);

  absl::flat_hash_set<std::string> seen_patterns;
  for (MatchArm* arm : node->arms()) {
    // We opportunistically identify syntactically identical match arms -- this
    // is a user error since the first should always match, the latter is
    // totally redundant.
    //
    // TODO(cdleary): 2025-01-31 We can get precise info on overlaps beyond
    // identical syntax when the exhaustiveness checker is available.
    std::string patterns_string = PatternsToString(arm);
    if (auto [it, inserted] = seen_patterns.insert(patterns_string);
        !inserted) {
      return TypeInferenceErrorStatus(
          arm->GetPatternSpan(), nullptr,
          absl::StrFormat("Exact-duplicate pattern match detected `%s` -- only "
                          "the first could possibly match",
                          patterns_string),
          ctx->file_table());
    }

    for (NameDefTree* pattern : arm->patterns()) {
      // Deduce types for all patterns with types that can be checked.
      //
      // Note that NameDef and RestOfTuple is handled in the Unify() call
      // below, and WildcardPattern has no type because it's a black hole.
      for (NameDefTree::Leaf leaf : pattern->Flatten()) {
        if (!std::holds_alternative<NameDef*>(leaf) &&
            !std::holds_alternative<WildcardPattern*>(leaf) &&
            !std::holds_alternative<RestOfTuple*>(leaf)) {
          XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(leaf)).status());
        }
      }

      XLS_RETURN_IF_ERROR(Unify(pattern, *matched, ctx));

      bool exhaustive_before = exhaustiveness_checker.IsExhaustive();
      exhaustiveness_checker.AddPattern(*pattern);
      if (exhaustive_before) {
        ctx->warnings()->Add(pattern->span(),
                             WarningKind::kAlreadyExhaustiveMatch,
                             "Match is already exhaustive before this pattern");
      }
    }
  }

  if (!exhaustiveness_checker.IsExhaustive()) {
    std::optional<InterpValue> sample =
        exhaustiveness_checker.SampleSimplestUncoveredValue();
    XLS_RET_CHECK(sample.has_value());
    return TypeInferenceErrorStatus(
        node->span(), matched.get(),
        absl::StrFormat(
            "Match %s not exhaustive; e.g. `%s` not covered; please add "
            "remaining "
            "patterns to complete the match or a default case "
            "via `_ => ...`",
            seen_patterns.size() == 1 ? "pattern is" : "patterns are",
            sample->ToString()),
        ctx->file_table());
  }

  std::vector<std::unique_ptr<Type>> arm_types;
  for (MatchArm* arm : node->arms()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> arm_type,
                         ctx->DeduceAndResolve(arm));
    arm_types.push_back(std::move(arm_type));
  }

  for (int64_t i = 1; i < arm_types.size(); ++i) {
    if (*arm_types[i] != *arm_types[0]) {
      return ctx->TypeMismatchError(
          node->arms()[i]->span(), nullptr, *arm_types[i], nullptr,
          *arm_types[0],
          "This match arm did not have the same type as the "
          "preceding match arms.");
    }
  }

  return std::move(arm_types[0]);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceBuiltinTypeAnnotation(
    const BuiltinTypeAnnotation* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceBuiltinTypeAnnotation: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> t, ConcretizeBuiltinTypeAnnotation(
                                                    *node, ctx->file_table()));
  VLOG(5) << "DeduceBuiltinTypeAnnotation result: " << t->ToString();
  return std::make_unique<MetaType>(std::move(t));
}

// As above, but converts to a TypeDim value that is boolean.
static absl::StatusOr<TypeDim> DimToConcreteBool(const Expr* dim_expr,
                                                 DeduceCtx* ctx) {
  std::unique_ptr<BitsType> u1 = BitsType::MakeU1();

  // We allow numbers in dimension position to go without type annotations -- we
  // implicitly make the type of the dimension u32, as we generally do with
  // dimension values.
  if (auto* number = dynamic_cast<const Number*>(dim_expr)) {
    if (number->type_annotation() == nullptr) {
      XLS_RETURN_IF_ERROR(TryEnsureFitsInBitsType(*number, *u1));
      ctx->type_info()->SetItem(number, *u1);
    } else {
      XLS_ASSIGN_OR_RETURN(auto dim_type, ctx->Deduce(number));
      if (*dim_type != *u1) {
        return ctx->TypeMismatchError(
            dim_expr->span(), nullptr, *dim_type, nullptr, *u1,
            absl::StrFormat("Dimension %s must be a `bool`/`u1`.",
                            dim_expr->ToString()));
      }
    }

    XLS_ASSIGN_OR_RETURN(int64_t value, number->GetAsUint64(ctx->file_table()));
    const bool value_bool = static_cast<bool>(value);
    XLS_RET_CHECK_EQ(value, value_bool);

    // No need to use the ConstexprEvaluator here. We've already got the goods.
    // It'd have trouble anyway, since this number isn't type-decorated.
    ctx->type_info()->NoteConstExpr(dim_expr,
                                    InterpValue::MakeBool(value_bool));
    return TypeDim::CreateBool(value_bool);
  }

  // First we check that it's a u1.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> dim_type, ctx->Deduce(dim_expr));
  if (*dim_type != *u1) {
    return ctx->TypeMismatchError(
        dim_expr->span(), nullptr, *dim_type, nullptr, *u1,
        absl::StrFormat("Dimension %s must be a `u1`.", dim_expr->ToString()));
  }

  // Now we try to constexpr evaluate it.
  const ParametricEnv parametric_env = ctx->GetCurrentParametricEnv();
  VLOG(5) << "Attempting to evaluate dimension expression: `"
          << dim_expr->ToString() << "` via parametric env: " << parametric_env;
  XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
      ctx->import_data(), ctx->type_info(), ctx->warnings(), parametric_env,
      dim_expr, dim_type.get()));
  if (ctx->type_info()->IsKnownConstExpr(dim_expr)) {
    XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                         ctx->type_info()->GetConstExpr(dim_expr));
    XLS_ASSIGN_OR_RETURN(uint64_t int_value,
                         constexpr_value.GetBitValueViaSign());
    bool bool_value = static_cast<bool>(int_value);
    XLS_RET_CHECK_EQ(bool_value, int_value);
    return TypeDim::CreateBool(bool_value);
  }

  // If there wasn't a known constexpr we could evaluate it to at this point, we
  // attempt to turn it into a parametric expression.
  absl::StatusOr<std::unique_ptr<ParametricExpression>> parametric_expr =
      ExprToParametric(dim_expr, ctx);
  if (parametric_expr.ok()) {
    return TypeDim(*std::move(parametric_expr));
  }

  VLOG(3) << "Could not convert dim expr to parametric expr; status: "
          << parametric_expr.status();

  // If we can't evaluate it to a parametric expression we give an error.
  return TypeInferenceErrorStatus(
      dim_expr->span(), nullptr,
      absl::StrFormat(
          "Could not evaluate dimension expression `%s` to a constant value.",
          dim_expr->ToString()),
      ctx->file_table());
}

absl::StatusOr<std::unique_ptr<Type>> DeduceChannelTypeAnnotation(
    const ChannelTypeAnnotation* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceChannelTypeAnnotation: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> payload_type,
                       Deduce(node->payload(), ctx));
  XLS_RET_CHECK(payload_type->IsMeta())
      << node->payload()->ToString() << " @ "
      << node->payload()->span().ToString(ctx->file_table());
  XLS_ASSIGN_OR_RETURN(
      payload_type,
      UnwrapMetaType(std::move(payload_type), node->payload()->span(),
                     "channel type annotation", ctx->file_table()));
  std::unique_ptr<Type> node_type =
      std::make_unique<ChannelType>(std::move(payload_type), node->direction());
  if (node->dims().has_value()) {
    std::vector<Expr*> dims = node->dims().value();

    for (const auto& dim : dims) {
      XLS_ASSIGN_OR_RETURN(TypeDim concrete_dim, DimToConcreteUsize(dim, ctx));
      node_type =
          std::make_unique<ArrayType>(std::move(node_type), concrete_dim);
    }
  }

  return std::make_unique<MetaType>(std::move(node_type));
}

absl::StatusOr<std::unique_ptr<Type>> DeduceTupleTypeAnnotation(
    const TupleTypeAnnotation* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceTupleTypeAnnotation: " << node->ToString();

  std::vector<std::unique_ptr<Type>> members;
  for (TypeAnnotation* member : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type, ctx->Deduce(member));
    XLS_ASSIGN_OR_RETURN(
        type, UnwrapMetaType(std::move(type), member->span(),
                             "tuple type member", ctx->file_table()));
    members.push_back(std::move(type));
  }
  auto t = std::make_unique<TupleType>(std::move(members));
  VLOG(5) << "DeduceTupleTypeAnnotation result: " << t->ToString();
  return std::make_unique<MetaType>(std::move(t));
}

absl::StatusOr<std::unique_ptr<Type>> DeduceArrayTypeAnnotation(
    const ArrayTypeAnnotation* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceArrayTypeAnnotation: " << node->ToString();

  std::unique_ptr<Type> t;
  if (auto* element_type =
          dynamic_cast<BuiltinTypeAnnotation*>(node->element_type());
      element_type != nullptr && element_type->GetBitCount() == 0) {
    VLOG(5) << "DeduceArrayTypeAnnotation; bits type constructor: "
            << node->ToString();

    if (element_type->builtin_type() == BuiltinType::kXN) {
      // This type constructor takes a boolean as its first array argument to
      // indicate signedness.
      XLS_ASSIGN_OR_RETURN(TypeDim dim, DimToConcreteBool(node->dim(), ctx));
      t = std::make_unique<BitsConstructorType>(std::move(dim));
    } else if (element_type->builtin_type() == BuiltinType::kToken) {
      // Token types have no signedness.
      XLS_ASSIGN_OR_RETURN(TypeDim dim, DimToConcreteUsize(node->dim(), ctx));
      auto element_type = std::make_unique<TokenType>();
      t = std::make_unique<ArrayType>(std::move(element_type), std::move(dim));
    } else {
      XLS_ASSIGN_OR_RETURN(TypeDim dim, DimToConcreteUsize(node->dim(), ctx));
      // We know we can determine signedness as the `xN` case is handled above.
      bool is_signed = element_type->GetSignedness().value();
      t = std::make_unique<BitsType>(is_signed, std::move(dim));
    }
  } else {
    VLOG(5) << "DeduceArrayTypeAnnotation; element_type: "
            << node->element_type()->ToString();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> e,
                         ctx->Deduce(node->element_type()));
    XLS_ASSIGN_OR_RETURN(
        e, UnwrapMetaType(std::move(e), node->element_type()->span(),
                          "array element type position", ctx->file_table()));
    XLS_ASSIGN_OR_RETURN(TypeDim dim, DimToConcreteUsize(node->dim(), ctx));
    t = std::make_unique<ArrayType>(std::move(e), std::move(dim));
    VLOG(4) << absl::StreamFormat("Array type annotation: %s => %s",
                                  node->ToString(), t->ToString());
  }

  auto result = std::make_unique<MetaType>(std::move(t));
  VLOG(5) << "DeduceArrayTypeAnnotation result" << result->ToString();
  return result;
}

// Helper that returns the type of `struct_def`'s `i`-th parametric binding.
static const Type& GetTypeOfParametric(const StructDef* struct_def, int64_t i,
                                       DeduceCtx* ctx) {
  const ParametricBinding* parametric_binding =
      struct_def->parametric_bindings()[i];
  TypeInfo* type_info =
      ctx->import_data()
          ->GetRootTypeInfoForNode(parametric_binding->type_annotation())
          .value();
  std::optional<const Type*> type =
      type_info->GetItem(parametric_binding->type_annotation());
  CHECK(type.has_value()) << absl::StreamFormat(
      "Expected parametric binding `%s` @ %s to have a type",
      parametric_binding->ToString(),
      parametric_binding->span().ToString(ctx->file_table()));
  const MetaType& meta_type = type.value()->AsMeta();
  return *meta_type.wrapped();
}

// Returns concretized struct type using the provided bindings.
//
// For example, if we have a struct defined as `struct Foo<N: u32, M: u32>`,
// the default TupleType will be (N, M). If a type annotation provides bindings,
// (e.g. `Foo<A, 16>`), we will replace N, M with those values. In the case
// above, we will return `(A, 16)` instead.
//
// Args:
//   type_annotation: The provided type annotation for this parametric struct.
//   struct_def: The struct definition AST node.
//   base_type: The TupleType of the struct, based only on the struct
//    definition (before parametrics are applied).
static absl::StatusOr<std::unique_ptr<Type>> ConcretizeStructAnnotation(
    const TypeRefTypeAnnotation* type_annotation, const StructDef* struct_def,
    const Type& base_type, DeduceCtx* ctx) {
  VLOG(5) << "ConcreteStructAnnotation; type_annotation: "
          << type_annotation->ToString()
          << " struct_def: " << struct_def->ToString();

  // Note: if there are too *few* annotated parametrics, some of them may be
  // derived.
  if (type_annotation->parametrics().size() >
      struct_def->parametric_bindings().size()) {
    return TypeInferenceErrorStatus(
        type_annotation->span(), &base_type,
        absl::StrFormat("Expected %d parametric arguments for '%s'; got %d in "
                        "type annotation",
                        struct_def->parametric_bindings().size(),
                        struct_def->identifier(),
                        type_annotation->parametrics().size()),
        ctx->file_table());
  }

  TypeDimMap type_dim_map;
  for (int64_t i = 0; i < type_annotation->parametrics().size(); ++i) {
    ParametricBinding* defined_parametric =
        struct_def->parametric_bindings()[i];
    ExprOrType eot = type_annotation->parametrics()[i];
    XLS_RET_CHECK(std::holds_alternative<Expr*>(eot));

    const Type& parametric_expr_type = GetTypeOfParametric(struct_def, i, ctx);

    Expr* annotated_parametric = std::get<Expr*>(eot);
    VLOG(5) << absl::StreamFormat(
        "ConcreteStructAnnotation; annotated_parametric expr: `%s`",
        annotated_parametric->ToString());

    XLS_ASSIGN_OR_RETURN(TypeDim ctd, DimToConcrete(annotated_parametric,
                                                    parametric_expr_type, ctx));
    type_dim_map.Insert(defined_parametric->identifier(), std::move(ctd));
  }

  // For the remainder of the formal parameterics (i.e. after the explicitly
  // supplied ones given as arguments) we have to see if they're derived
  // parametrics. If they're *not* derived via an expression, we should have
  // been supplied some value in the annotation, so we have to flag an error!
  for (int64_t i = type_annotation->parametrics().size();
       i < struct_def->parametric_bindings().size(); ++i) {
    ParametricBinding* defined_parametric =
        struct_def->parametric_bindings()[i];
    if (defined_parametric->expr() == nullptr) {
      return TypeInferenceErrorStatus(
          type_annotation->span(), &base_type,
          absl::StrFormat("No parametric value provided for '%s' in '%s'",
                          defined_parametric->identifier(),
                          struct_def->identifier()),
          ctx->file_table());
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ParametricExpression> parametric_expr,
                         ExprToParametric(defined_parametric->expr(), ctx));
    type_dim_map.Insert(defined_parametric->identifier(),
                        TypeDim(parametric_expr->Evaluate(type_dim_map.env())));
  }

  // Now evaluate all the dimensions according to the values we've got.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Type> mapped_type,
      base_type.MapSize([&](const TypeDim& dim) -> absl::StatusOr<TypeDim> {
        if (std::holds_alternative<TypeDim::OwnedParametric>(dim.value())) {
          auto& parametric = std::get<TypeDim::OwnedParametric>(dim.value());
          return TypeDim(parametric->Evaluate(type_dim_map.env()));
        }
        return dim;
      }));

  VLOG(5) << "ConcreteStructAnnotation mapped type: "
          << mapped_type->ToString();

  // Attach the nominal parametrics to the type, so that we will remember the
  // fact that we have instantiated e.g. Foo<M:u32, N:u32> as Foo<5, 6>.
  return mapped_type->AddNominalTypeDims(type_dim_map.dims());
}

absl::StatusOr<std::unique_ptr<Type>> DeduceTypeRefTypeAnnotation(
    const TypeRefTypeAnnotation* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceTypeRefTypeAnnotation: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> base_type,
                       ctx->Deduce(node->type_ref()));
  TypeRef* type_ref = node->type_ref();
  TypeDefinition type_definition = type_ref->type_definition();

  // If it's a (potentially parametric) struct, we concretize it.
  absl::StatusOr<StructDef*> struct_def = DerefToStruct(
      node->span(), type_ref->ToString(), type_definition, ctx->type_info());
  if (struct_def.ok()) {
    VLOG(5) << "DeduceTypeRefTypeAnnotation struct_def "
            << (*struct_def)->ToString()
            << " IsParametric: " << (*struct_def)->IsParametric();
    VLOG(5) << "DeduceTypeRefTypeAnnotation node " << node->ToString()
            << " parametrics.empty: " << node->parametrics().empty();
    VLOG(5) << "DeduceTypeRefTypeAnnotation base type "
            << base_type->ToString();

    if ((*struct_def)->IsParametric() && !node->parametrics().empty()) {
      XLS_ASSIGN_OR_RETURN(base_type, ConcretizeStructAnnotation(
                                          node, *struct_def, *base_type, ctx));
      VLOG(5)
          << "DeduceTypeRefTypeAnnotation after concretize, base type is now "
          << base_type->ToString();
    }
  }
  XLS_RET_CHECK(base_type->IsMeta());
  VLOG(5) << "DeduceTypeRefTypeAnnotation result base_type: "
          << base_type->ToString();

  return std::move(base_type);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceSelfTypeAnnotation(
    const SelfTypeAnnotation* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceSelfTypeAnnotation: " << node->ToString();
  return ctx->Deduce(node->struct_ref());
}

absl::StatusOr<std::unique_ptr<Type>> DeduceMatchArm(const MatchArm* node,
                                                     DeduceCtx* ctx) {
  return ctx->Deduce(node->expr());
}

absl::StatusOr<std::unique_ptr<Type>> DeduceChannelDecl(const ChannelDecl* node,
                                                        DeduceCtx* ctx) {
  VLOG(5) << "DeduceChannelDecl: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> element_type,
                       Deduce(node->type(), ctx));
  XLS_ASSIGN_OR_RETURN(
      element_type,
      UnwrapMetaType(std::move(element_type), node->type()->span(),
                     "channel declaration type", ctx->file_table()));
  std::unique_ptr<Type> producer = std::make_unique<ChannelType>(
      element_type->CloneToUnique(), ChannelDirection::kOut);
  std::unique_ptr<Type> consumer = std::make_unique<ChannelType>(
      std::move(element_type), ChannelDirection::kIn);

  if (node->dims().has_value()) {
    std::vector<Expr*> dims = node->dims().value();

    for (const auto& dim : dims) {
      XLS_ASSIGN_OR_RETURN(TypeDim concrete_dim, DimToConcreteUsize(dim, ctx));
      producer = std::make_unique<ArrayType>(std::move(producer), concrete_dim);
      consumer = std::make_unique<ArrayType>(std::move(consumer), concrete_dim);
    }
  }

  if (node->fifo_depth().has_value()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> fifo_depth,
                         ctx->Deduce(node->fifo_depth().value()));
    auto want = BitsType::MakeU32();
    if (*fifo_depth != *want) {
      return ctx->TypeMismatchError(
          node->span(), node->fifo_depth().value(), *fifo_depth, nullptr, *want,
          "Channel declaration FIFO depth must be a u32.");
    }
  }

  // The channel name must be a constexpr u8 array.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> channel_name_type,
                       Deduce(&node->channel_name_expr(), ctx));
  if (!IsU8Array(*channel_name_type)) {
    return TypeInferenceErrorStatus(
        node->channel_name_expr().span(), channel_name_type.get(),
        "Channel name must be an array of u8s; i.e. u8[N]", ctx->file_table());
  }
  XLS_ASSIGN_OR_RETURN(InterpValue channel_name_value,
                       EvaluateConstexprValue(ctx, &node->channel_name_expr(),
                                              channel_name_type.get()));
  XLS_ASSIGN_OR_RETURN(int64_t name_length, channel_name_value.GetLength());
  if (name_length == 0) {
    return TypeInferenceErrorStatus(
        node->channel_name_expr().span(), channel_name_type.get(),
        "Channel name must not be empty", ctx->file_table());
  }

  std::vector<std::unique_ptr<Type>> elements;
  VLOG(5) << "DeduceChannelDecl producer: " << producer->ToString();
  VLOG(5) << "DeduceChannelDecl consumer: " << consumer->ToString();
  elements.push_back(std::move(producer));
  elements.push_back(std::move(consumer));
  return std::make_unique<TupleType>(std::move(elements));
}

absl::StatusOr<std::unique_ptr<Type>> DeduceRange(const Range* node,
                                                  DeduceCtx* ctx) {
  VLOG(5) << "DeduceRange: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> start_type,
                       ctx->Deduce(node->start()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> end_type,
                       ctx->Deduce(node->end()));
  if (*start_type != *end_type) {
    return ctx->TypeMismatchError(node->span(), nullptr, *start_type, nullptr,
                                  *end_type,
                                  "Range start and end types didn't match.");
  }

  if (!IsBitsLike(*start_type)) {
    return TypeInferenceErrorStatus(
        node->span(), start_type.get(),
        "Range start and end types must resolve to bits types.",
        ctx->file_table());
  }

  // Range implicitly defines a sized type, so it has to be constexpr
  // evaluatable.
  XLS_ASSIGN_OR_RETURN(
      InterpValue start_value,
      EvaluateConstexprValue(ctx, node->start(), start_type.get()));
  XLS_ASSIGN_OR_RETURN(
      InterpValue end_value,
      EvaluateConstexprValue(ctx, node->end(), end_type.get()));

  XLS_ASSIGN_OR_RETURN(InterpValue le, end_value.Le(start_value));
  XLS_ASSIGN_OR_RETURN(InterpValue lt, end_value.Lt(start_value));
  if (lt.IsTrue() || (!node->inclusive_end() && le.IsTrue())) {
    ctx->warnings()->Add(
        node->span(), WarningKind::kEmptyRangeLiteral,
        absl::StrFormat("`%s` from `%s` to `%s` is an empty range",
                        node->ToString(), start_value.ToString(),
                        end_value.ToString()));
  }

  InterpValue array_size = InterpValue::MakeUnit();
  XLS_ASSIGN_OR_RETURN(InterpValue start_gt_end, start_value.Gt(end_value));
  if (start_gt_end.IsTrue()) {
    array_size = InterpValue::MakeU32(0);
  } else {
    XLS_ASSIGN_OR_RETURN(array_size, end_value.Sub(start_value));
    if (node->inclusive_end()) {
      array_size = InterpValue::MakeUnsigned(array_size.GetBitsOrDie());
      XLS_ASSIGN_OR_RETURN(array_size,
                           array_size.IncrementZeroExtendIfOverflow());
    }
    // Zero extend to u32.
    XLS_ASSIGN_OR_RETURN(array_size, array_size.ZeroExt(32));
  }
  VLOG(5) << "DeduceRange result: " << start_type->ToString();
  return std::make_unique<ArrayType>(std::move(start_type),
                                     TypeDim(array_size));
}

absl::StatusOr<std::unique_ptr<Type>> DeduceNameRef(const NameRef* node,
                                                    DeduceCtx* ctx) {
  VLOG(5) << "DeduceNameRef: " << node->ToString();

  AstNode* name_def = ToAstNode(node->name_def());
  XLS_RET_CHECK(name_def != nullptr);

  if (std::optional<InterpValue> const_expr =
          ctx->type_info()->GetConstExprOption(name_def)) {
    ctx->type_info()->NoteConstExpr(node, const_expr.value());
  }

  std::optional<Type*> item = ctx->type_info()->GetItem(name_def);
  if (item.has_value()) {
    auto type = (*item)->CloneToUnique();
    VLOG(5) << "DeduceNameRef result: " << type->ToString();
    return type;
  }

  // If this has no corresponding type because it is a parametric function that
  // is not being invoked, we give an error instead of propagating
  // "TypeMissing".
  if ((IsParametricFunction(node->GetDefiner()) ||
       IsBuiltinParametricNameRef(node)) &&
      !ParentIsInvocationWithCallee(node)) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat(
            "Name '%s' is a parametric function, but it is not being invoked",
            node->identifier()),
        ctx->file_table());
  }

  return TypeMissingErrorStatus(/*node=*/*name_def, /*user=*/node,
                                ctx->file_table());
}

absl::StatusOr<std::unique_ptr<Type>> DeduceStructDef(const StructDef* node,
                                                      DeduceCtx* ctx) {
  XLS_RETURN_IF_ERROR(TypecheckStructDefBase(node, ctx));
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Type>> members,
      DeduceStructDefBaseMembers(node, ctx, &ValidateStructMember));
  auto wrapped = std::make_unique<StructType>(std::move(members), *node);
  auto result = std::make_unique<MetaType>(std::move(wrapped));
  ctx->type_info()->SetItem(node->name_def(), *result);
  VLOG(5) << absl::Substitute("Deduced type for struct $0 => $1",
                              node->ToString(), result->ToString());
  return result;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceProcDef(const ProcDef* node,
                                                    DeduceCtx* ctx) {
  XLS_RETURN_IF_ERROR(TypecheckStructDefBase(node, ctx));
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Type>> members,
      DeduceStructDefBaseMembers(node, ctx, &ValidateProcMember));
  auto wrapped = std::make_unique<ProcType>(std::move(members), *node);
  auto result = std::make_unique<MetaType>(std::move(wrapped));
  ctx->type_info()->SetItem(node->name_def(), *result);
  VLOG(5) << absl::Substitute("Deduced type for proc $0 => $1",
                              node->ToString(), result->ToString());
  return result;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceImpl(const Impl* node,
                                                 DeduceCtx* ctx) {
  VLOG(5) << "DeduceImpl: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                       ctx->Deduce(ToAstNode(node->struct_ref())));

  XLS_ASSIGN_OR_RETURN(
      type, UnwrapMetaType(std::move(type), node->span(), "impl struct type",
                           ctx->file_table()));

  auto* struct_type = dynamic_cast<const StructTypeBase*>(type.get());
  if (struct_type == nullptr) {
    return TypeInferenceErrorStatus(node->span(), struct_type,
                                    "Impl must be for a struct or proc type",
                                    ctx->file_table());
  }
  TypeRefTypeAnnotation* type_ref =
      dynamic_cast<TypeRefTypeAnnotation*>(node->struct_ref());
  XLS_RET_CHECK(type_ref != nullptr);
  if (type_ref->parametrics().empty()) {
    for (const auto& member : node->members()) {
      if (std::holds_alternative<ConstantDef*>(member)) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> _,
                             ctx->Deduce(ToAstNode(member)));
      } else {
        XLS_RETURN_IF_ERROR(
            ctx->typecheck_function()(*(std::get<Function*>(member)), ctx));
      }
    }
  }
  return type;
}

class DeduceVisitor : public AstNodeVisitor {
 public:
  explicit DeduceVisitor(DeduceCtx* ctx) : ctx_(ctx) {}

#define DEDUCE_DISPATCH(__type, __rule)                   \
  absl::Status Handle##__type(const __type* n) override { \
    result_ = __rule(n, ctx_);                            \
    return result_.status();                              \
  }

  DEDUCE_DISPATCH(Unop, DeduceUnop)
  DEDUCE_DISPATCH(Param, DeduceParam)
  DEDUCE_DISPATCH(ProcMember, DeduceProcMember)
  DEDUCE_DISPATCH(ConstantDef, DeduceConstantDef)
  DEDUCE_DISPATCH(Number, DeduceNumber)
  DEDUCE_DISPATCH(String, DeduceString)
  DEDUCE_DISPATCH(TypeRef, DeduceTypeRef)
  DEDUCE_DISPATCH(TypeAlias, DeduceTypeAlias)
  DEDUCE_DISPATCH(XlsTuple, DeduceXlsTuple)
  DEDUCE_DISPATCH(Conditional, DeduceConditional)
  DEDUCE_DISPATCH(Binop, DeduceBinop)
  DEDUCE_DISPATCH(EnumDef, DeduceEnumDef)
  DEDUCE_DISPATCH(Let, DeduceLet)
  DEDUCE_DISPATCH(Lambda, DeduceLambda)
  DEDUCE_DISPATCH(For, DeduceFor)
  DEDUCE_DISPATCH(Cast, DeduceCast)
  DEDUCE_DISPATCH(ConstAssert, DeduceConstAssert)
  DEDUCE_DISPATCH(StructDef, DeduceStructDef)
  DEDUCE_DISPATCH(ProcDef, DeduceProcDef)
  DEDUCE_DISPATCH(Impl, DeduceImpl)
  DEDUCE_DISPATCH(Array, DeduceArray)
  DEDUCE_DISPATCH(Attr, DeduceAttr)
  DEDUCE_DISPATCH(StatementBlock, DeduceStatementBlock)
  DEDUCE_DISPATCH(ChannelDecl, DeduceChannelDecl)
  DEDUCE_DISPATCH(ColonRef, DeduceColonRef)
  DEDUCE_DISPATCH(Index, DeduceIndex)
  DEDUCE_DISPATCH(Match, DeduceMatch)
  DEDUCE_DISPATCH(Range, DeduceRange)
  DEDUCE_DISPATCH(Spawn, DeduceSpawn)
  DEDUCE_DISPATCH(SplatStructInstance, DeduceSplatStructInstance)
  DEDUCE_DISPATCH(Statement, DeduceStatement)
  DEDUCE_DISPATCH(StructInstance, DeduceStructInstance)
  DEDUCE_DISPATCH(TupleIndex, DeduceTupleIndex)
  DEDUCE_DISPATCH(UnrollFor, DeduceUnrollFor)
  DEDUCE_DISPATCH(BuiltinTypeAnnotation, DeduceBuiltinTypeAnnotation)
  DEDUCE_DISPATCH(ChannelTypeAnnotation, DeduceChannelTypeAnnotation)
  DEDUCE_DISPATCH(ArrayTypeAnnotation, DeduceArrayTypeAnnotation)
  DEDUCE_DISPATCH(TupleTypeAnnotation, DeduceTupleTypeAnnotation)
  DEDUCE_DISPATCH(TypeRefTypeAnnotation, DeduceTypeRefTypeAnnotation)
  DEDUCE_DISPATCH(SelfTypeAnnotation, DeduceSelfTypeAnnotation)
  DEDUCE_DISPATCH(MatchArm, DeduceMatchArm)
  DEDUCE_DISPATCH(Invocation, DeduceInvocation)
  DEDUCE_DISPATCH(FormatMacro, DeduceFormatMacro)
  DEDUCE_DISPATCH(ZeroMacro, DeduceZeroMacro)
  DEDUCE_DISPATCH(AllOnesMacro, DeduceAllOnesMacro)
  DEDUCE_DISPATCH(NameRef, DeduceNameRef)
  DEDUCE_DISPATCH(UseTreeEntry, DeduceUseTreeEntry)

  // Unhandled nodes for deduction, either they are custom visited or not
  // visited "automatically" in the traversal process (e.g. top level module
  // members).
  absl::Status HandleProc(const Proc* n) override { return Fatal(n); }
  absl::Status HandleSlice(const Slice* n) override { return Fatal(n); }
  absl::Status HandleImport(const Import* n) override { return Fatal(n); }
  absl::Status HandleUse(const Use* n) override { return Fatal(n); }
  absl::Status HandleFunction(const Function* n) override { return Fatal(n); }
  absl::Status HandleQuickCheck(const QuickCheck* n) override {
    return Fatal(n);
  }
  absl::Status HandleTestFunction(const TestFunction* n) override {
    return Fatal(n);
  }
  absl::Status HandleTestProc(const TestProc* n) override { return Fatal(n); }
  absl::Status HandleModule(const Module* n) override { return Fatal(n); }
  absl::Status HandleWidthSlice(const WidthSlice* n) override {
    return Fatal(n);
  }
  absl::Status HandleNameDefTree(const NameDefTree* n) override {
    return Fatal(n);
  }
  absl::Status HandleNameDef(const NameDef* n) override { return Fatal(n); }
  absl::Status HandleStructMemberNode(const StructMemberNode* n) override {
    return Fatal(n);
  }
  absl::Status HandleBuiltinNameDef(const BuiltinNameDef* n) override {
    return Fatal(n);
  }
  absl::Status HandleParametricBinding(const ParametricBinding* n) override {
    return Fatal(n);
  }
  absl::Status HandleWildcardPattern(const WildcardPattern* n) override {
    return Fatal(n);
  }
  absl::Status HandleRestOfTuple(const RestOfTuple* n) override {
    return Fatal(n);
  }
  absl::Status HandleVerbatimNode(const VerbatimNode* n) override {
    return Fatal(n);
  }

  // All of these annotation types are created by `type_system_v2`, so there
  // should be none of them when using `type_system` for inference.
  absl::Status HandleTypeVariableTypeAnnotation(
      const TypeVariableTypeAnnotation* n) override {
    return Fatal(n);
  }
  absl::Status HandleMemberTypeAnnotation(
      const MemberTypeAnnotation* n) override {
    return Fatal(n);
  }
  absl::Status HandleElementTypeAnnotation(
      const ElementTypeAnnotation* n) override {
    return Fatal(n);
  }
  absl::Status HandleSliceTypeAnnotation(
      const SliceTypeAnnotation* n) override {
    return Fatal(n);
  }
  absl::Status HandleFunctionTypeAnnotation(
      const FunctionTypeAnnotation* n) override {
    return Fatal(n);
  }
  absl::Status HandleReturnTypeAnnotation(
      const ReturnTypeAnnotation* n) override {
    return Fatal(n);
  }
  absl::Status HandleParamTypeAnnotation(
      const ParamTypeAnnotation* n) override {
    return Fatal(n);
  }
  absl::Status HandleAnyTypeAnnotation(const AnyTypeAnnotation* n) override {
    return Fatal(n);
  }

  absl::Status HandleFunctionRef(const FunctionRef* n) override {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> callee_type,
                         ctx_->Deduce(n->callee()));
    if (!callee_type->IsFunction()) {
      return TypeInferenceErrorStatus(
          n->span(), callee_type.get(),
          "Callee for function reference must be function-typed",
          ctx_->file_table());
    }
    result_ = std::move(callee_type);
    return absl::OkStatus();
  }

  absl::Status HandleGenericTypeAnnotation(
      const GenericTypeAnnotation* n) override {
    return Fatal(n);
  }

  absl::StatusOr<std::unique_ptr<Type>>& result() { return result_; }

 private:
  absl::Status Fatal(const AstNode* n) {
    LOG(FATAL) << "DeduceVisitor got unhandled AST node for deduction: "
               << n->ToString() << " node type name: " << n->GetNodeTypeName();
  }

  DeduceCtx* ctx_;
  absl::StatusOr<std::unique_ptr<Type>> result_;
};

absl::StatusOr<std::unique_ptr<Type>> DeduceInternal(const AstNode* node,
                                                     DeduceCtx* ctx) {
  DeduceVisitor visitor(ctx);
  XLS_RETURN_IF_ERROR(node->Accept(&visitor));
  return std::move(visitor.result());
}

}  // namespace

absl::StatusOr<std::unique_ptr<Type>> Deduce(const AstNode* node,
                                             DeduceCtx* ctx) {
  VLOG(5) << "Deduce: " << node->ToString();

  XLS_RET_CHECK(node != nullptr);
  if (std::optional<Type*> type = ctx->type_info()->GetItem(node)) {
    return (*type)->CloneToUnique();
  }
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type, DeduceInternal(node, ctx));
  XLS_RET_CHECK(type != nullptr);
  ctx->type_info()->SetItem(node, *type);

  VLOG(5) << absl::StreamFormat(
      "Deduced type of `%s` @ %p (kind: %s) => %s in %p", node->ToString(),
      node, node->GetNodeTypeName(), type->ToString(), ctx->type_info());

  return type;
}

absl::StatusOr<TypeDim> DimToConcreteUsize(const Expr* dim_expr,
                                           DeduceCtx* ctx) {
  VLOG(10) << "DimToConcreteUsize; dim_expr: `" << dim_expr->ToString() << "`";

  std::unique_ptr<BitsType> u32 = BitsType::MakeU32();
  auto validate_high_bit = [ctx, &u32](const Span& span, uint32_t value) {
    if ((value >> 31) == 0) {
      return absl::OkStatus();
    }
    return TypeInferenceErrorStatus(
        span, u32.get(),
        absl::StrFormat(
            "Dimension value is too large, high bit is set: %#x; "
            "XLS only allows sizes up to 31 bits to guard against the more "
            "common mistake of specifying a negative (constexpr) value as a "
            "size.",
            value),
        ctx->file_table());
  };

  // We allow numbers in dimension position to go without type annotations -- we
  // implicitly make the type of the dimension u32, as we generally do with
  // dimension values.
  if (auto* number = dynamic_cast<const Number*>(dim_expr)) {
    if (number->type_annotation() == nullptr) {
      XLS_RETURN_IF_ERROR(TryEnsureFitsInBitsType(*number, *u32));
      ctx->type_info()->SetItem(number, *u32);
    } else {
      XLS_ASSIGN_OR_RETURN(auto dim_type, ctx->Deduce(number));
      if (*dim_type != *u32) {
        return ctx->TypeMismatchError(
            dim_expr->span(), nullptr, *dim_type, nullptr, *u32,
            absl::StrFormat(
                "Dimension %s must be a `u32` (soon to be `usize`, see "
                "https://github.com/google/xls/issues/450 for details).",
                dim_expr->ToString()));
      }
    }

    XLS_ASSIGN_OR_RETURN(int64_t value, number->GetAsUint64(ctx->file_table()));
    const uint32_t value_u32 = static_cast<uint32_t>(value);
    XLS_RET_CHECK_EQ(value, value_u32);

    XLS_RETURN_IF_ERROR(validate_high_bit(number->span(), value_u32));

    // No need to use the ConstexprEvaluator here. We've already got the goods.
    // It'd have trouble anyway, since this number isn't type-decorated.
    ctx->type_info()->NoteConstExpr(dim_expr, InterpValue::MakeU32(value_u32));
    return TypeDim::CreateU32(value_u32);
  }

  // First we check that it's a u32 (in the future we'll want it to be a usize).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> dim_type, ctx->Deduce(dim_expr));
  if (*dim_type != *u32) {
    return ctx->TypeMismatchError(
        dim_expr->span(), nullptr, *dim_type, nullptr, *u32,
        absl::StrFormat(
            "Dimension %s must be a `u32` (soon to be `usize`, see "
            "https://github.com/google/xls/issues/450 for details).",
            dim_expr->ToString()));
  }

  // Now we try to constexpr evaluate it.
  const ParametricEnv parametric_env = ctx->GetCurrentParametricEnv();
  VLOG(5) << "Attempting to evaluate dimension expression: `"
          << dim_expr->ToString() << "` via parametric env: " << parametric_env;
  XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
      ctx->import_data(), ctx->type_info(), ctx->warnings(), parametric_env,
      dim_expr, dim_type.get()));
  if (ctx->type_info()->IsKnownConstExpr(dim_expr)) {
    XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                         ctx->type_info()->GetConstExpr(dim_expr));
    XLS_ASSIGN_OR_RETURN(uint64_t int_value,
                         constexpr_value.GetBitValueViaSign());
    uint32_t u32_value = static_cast<uint32_t>(int_value);
    XLS_RETURN_IF_ERROR(validate_high_bit(dim_expr->span(), u32_value));
    XLS_RET_CHECK_EQ(u32_value, int_value);
    return TypeDim::CreateU32(u32_value);
  }

  // If there wasn't a known constexpr we could evaluate it to at this point, we
  // attempt to turn it into a parametric expression.
  absl::StatusOr<std::unique_ptr<ParametricExpression>> parametric_expr =
      ExprToParametric(dim_expr, ctx);
  if (parametric_expr.ok()) {
    return TypeDim(*std::move(parametric_expr));
  }

  VLOG(3) << "Could not convert dim expr to parametric expr; status: "
          << parametric_expr.status();

  // If we can't evaluate it to a parametric expression we give an error.
  return TypeInferenceErrorStatus(
      dim_expr->span(), nullptr,
      absl::StrFormat(
          "Could not evaluate dimension expression `%s` to a constant value.",
          dim_expr->ToString()),
      ctx->file_table());
}

absl::StatusOr<TypeDim> DimToConcrete(const Expr* dim_expr, const Type& type,
                                      DeduceCtx* ctx) {
  std::optional<BitsLikeProperties> bits_like_type = GetBitsLike(type);
  if (!bits_like_type.has_value()) {
    return TypeInferenceErrorStatus(
        dim_expr->span(), &type,
        absl::StrFormat("Expected dimension expression `%s` to be a bits-like "
                        "type; got: %s",
                        dim_expr->ToString(), type.ToString()),
        ctx->file_table());
  }

  if (IsKnownU1(bits_like_type.value())) {
    return DimToConcreteBool(dim_expr, ctx);
  }
  if (IsKnownU32(bits_like_type.value())) {
    return DimToConcreteUsize(dim_expr, ctx);
  }

  return TypeInferenceErrorStatus(
      dim_expr->span(), &type,
      absl::StrFormat("Expected dimension expression `%s` to be a `bool` or "
                      "`u32` type; got: %s i.e. %s",
                      dim_expr->ToString(), type.ToString(),
                      ToTypeString(bits_like_type.value())),
      ctx->file_table());
}

}  // namespace xls::dslx
