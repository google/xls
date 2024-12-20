// Copyright 2021 The XLS Authors
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

#include "xls/dslx/constexpr_evaluator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_zero_value.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"

namespace xls::dslx {

/* static */ absl::Status ConstexprEvaluator::Evaluate(
    ImportData* import_data, TypeInfo* type_info,
    WarningCollector* warning_collector, const ParametricEnv& bindings,
    const Expr* expr, const Type* type) {
  VLOG(5) << "ConstexprEvaluator::Evaluate; expr: " << expr->ToString() << " @ "
          << expr->span().ToString(import_data->file_table());
  if (type != nullptr) {
    XLS_RET_CHECK(!type->IsMeta());
  }

  if (type_info->IsKnownConstExpr(expr) ||
      type_info->IsKnownNonConstExpr(expr)) {
    return absl::OkStatus();
  }
  ConstexprEvaluator evaluator(import_data, type_info, warning_collector,
                               bindings, type);
  return expr->AcceptExpr(&evaluator);
}

/* static */ absl::StatusOr<InterpValue> ConstexprEvaluator::EvaluateToValue(
    ImportData* import_data, TypeInfo* type_info,
    WarningCollector* warning_collector, const ParametricEnv& bindings,
    const Expr* expr, const Type* type) {
  XLS_RETURN_IF_ERROR(
      Evaluate(import_data, type_info, warning_collector, bindings, expr));
  if (type_info->IsKnownConstExpr(expr)) {
    return type_info->GetConstExpr(expr);
  }
  const FileTable& file_table = import_data->file_table();
  return absl::InvalidArgumentError(
      absl::StrFormat("Expression @ %s was not constexpr: `%s`",
                      expr->span().ToString(file_table), expr->ToString()));
}

// Evaluates the given expression and terminates current function execution
// if it is not constexpr.
#define EVAL_AS_CONSTEXPR_OR_RETURN(EXPR)                                     \
  if (!type_info_->IsKnownConstExpr(EXPR) &&                                  \
      !type_info_->IsKnownNonConstExpr(EXPR)) {                               \
    Type* sub_type = nullptr;                                                 \
    if (type_info_->GetItem(EXPR).has_value()) {                              \
      sub_type = type_info_->GetItem(EXPR).value();                           \
    }                                                                         \
    ConstexprEvaluator sub_eval(import_data_, type_info_, warning_collector_, \
                                bindings_, sub_type);                         \
    XLS_RETURN_IF_ERROR(EXPR->AcceptExpr(&sub_eval));                         \
  }                                                                           \
  if (!type_info_->IsKnownConstExpr(EXPR)) {                                  \
    return absl::OkStatus();                                                  \
  }

// Assigns the constexpr value of the given expression to the LHS or terminates
// execution if it's not constexpr.
#define GET_CONSTEXPR_OR_RETURN(LHS, EXPR) \
  EVAL_AS_CONSTEXPR_OR_RETURN(EXPR);       \
  LHS = type_info_->GetConstExpr(EXPR).value();

absl::Status ConstexprEvaluator::HandleAttr(const Attr* expr) {
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->lhs());
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleZeroMacro(const ZeroMacro* expr) {
  const FileTable& file_table = *expr->owner()->file_table();
  ExprOrType type_reference = expr->type();
  std::optional<Type*> maybe_type =
      type_info_->GetItem(ToAstNode(type_reference));
  if (!maybe_type.has_value()) {
    std::optional<Span> span = ToAstNode(type_reference)->GetSpan();
    std::string span_str =
        span.has_value() ? span->ToString(file_table) : "<none>";
    return absl::InternalError(
        absl::StrFormat("Could not find type for \"%s\" @ %s",
                        ToAstNode(type_reference)->ToString(), span_str));
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Type> type,
      UnwrapMetaType(maybe_type.value()->CloneToUnique(), expr->span(),
                     "zero macro input type", file_table));

  // At this point type inference should have checked that this type was
  // zero-able.
  XLS_ASSIGN_OR_RETURN(InterpValue value,
                       MakeZeroValue(*type, *import_data_, expr->span()));
  type_info_->NoteConstExpr(expr, value);
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleAllOnesMacro(const AllOnesMacro* expr) {
  const FileTable& file_table = *expr->owner()->file_table();
  ExprOrType type_reference = expr->type();
  std::optional<Type*> maybe_type =
      type_info_->GetItem(ToAstNode(type_reference));
  if (!maybe_type.has_value()) {
    std::optional<Span> span = ToAstNode(type_reference)->GetSpan();
    std::string span_str =
        span.has_value() ? span->ToString(file_table) : "<none>";
    return absl::InternalError(
        absl::StrFormat("Could not find type for \"%s\" @ %s",
                        ToAstNode(type_reference)->ToString(), span_str));
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Type> type,
      UnwrapMetaType(maybe_type.value()->CloneToUnique(), expr->span(),
                     "all-ones macro input type", file_table));

  // At this point type inference should have checked that this type has an
  // all-ones value.
  XLS_ASSIGN_OR_RETURN(InterpValue value,
                       MakeAllOnesValue(*type, *import_data_, expr->span()));
  type_info_->NoteConstExpr(expr, value);
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleArray(const Array* expr) {
  VLOG(3) << "ConstexprEvaluator::HandleArray : " << expr->ToString();
  std::vector<InterpValue> values;
  for (const Expr* member : expr->members()) {
    EVAL_AS_CONSTEXPR_OR_RETURN(member);
  }

  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleBinop(const Binop* expr) {
  VLOG(3) << "ConstexprEvaluator::HandleBinop : " << expr->ToString();
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->lhs());
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->rhs());
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleStatementBlock(
    const StatementBlock* expr) {
  bool all_statements_constexpr = true;
  Expr* last_expr = nullptr;
  for (Statement* stmt : expr->statements()) {
    if (!std::holds_alternative<Expr*>(stmt->wrapped())) {
      continue;
    }
    Expr* body_expr = std::get<Expr*>(stmt->wrapped());
    XLS_RETURN_IF_ERROR(body_expr->AcceptExpr(this));
    if (type_info_->IsKnownConstExpr(body_expr)) {
      last_expr = body_expr;
      type_info_->NoteConstExpr(body_expr,
                                type_info_->GetConstExpr(body_expr).value());
    } else {
      VLOG(10) << "ConstexprEvaluator::HandleBlock; expr was not constexpr: "
               << body_expr->ToString();
      all_statements_constexpr = false;
    }
  }
  if (all_statements_constexpr) {
    if (expr->trailing_semi()) {
      type_info_->NoteConstExpr(expr, InterpValue::MakeUnit());
    } else {
      type_info_->NoteConstExpr(expr,
                                type_info_->GetConstExpr(last_expr).value());
    }
  }
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleCast(const Cast* expr) {
  VLOG(3) << "ConstexprEvaluator::HandleCast : " << expr->ToString();
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->expr());
  return InterpretExpr(expr);
}

// Creates an InterpValue for the described channel or array of channels.
absl::StatusOr<InterpValue> ConstexprEvaluator::CreateChannelValue(
    const Type* type) {
  if (auto* array_type = dynamic_cast<const ArrayType*>(type)) {
    XLS_ASSIGN_OR_RETURN(int dim_int, array_type->size().GetAsInt64());
    std::vector<InterpValue> elements;
    elements.reserve(dim_int);
    for (int i = 0; i < dim_int; i++) {
      XLS_ASSIGN_OR_RETURN(InterpValue element,
                           CreateChannelValue(&array_type->element_type()));
      elements.push_back(element);
    }
    return InterpValue::MakeArray(elements);
  }

  // There can't be tuples or structs of channels, only arrays.
  const ChannelType* ct = dynamic_cast<const ChannelType*>(type);
  XLS_RET_CHECK_NE(ct, nullptr);
  return InterpValue::MakeChannel();
}

// While a channel's *contents* aren't constexpr, the existence of the channel
// itself is.
absl::Status ConstexprEvaluator::HandleChannelDecl(const ChannelDecl* expr) {
  VLOG(3) << "ConstexprEvaluator::HandleChannelDecl : " << expr->ToString();
  const FileTable& file_table = *expr->owner()->file_table();
  // Keep in mind that channels come in tuples, so peel out the first element.
  std::optional<Type*> maybe_decl_type = type_info_->GetItem(expr);
  if (!maybe_decl_type.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Could not find type for expr \"%s\" @ %s",
                        expr->ToString(), expr->span().ToString(file_table)));
  }

  auto* tuple_type = dynamic_cast<TupleType*>(maybe_decl_type.value());
  if (tuple_type == nullptr) {
    return TypeInferenceErrorStatus(
        expr->span(), maybe_decl_type.value(),
        "Channel decl did not have tuple type:", file_table);
  }

  // Verify that the channel tuple has exactly two elements; just yank one out
  // for channel [array] creation (they both point to the same object).
  if (tuple_type->size() != 2) {
    return TypeInferenceErrorStatus(expr->span(), tuple_type,
                                    "ChannelDecl type was a two-element tuple.",
                                    file_table);
  }

  XLS_ASSIGN_OR_RETURN(InterpValue channel,
                       CreateChannelValue(&tuple_type->GetMemberType(0)));
  type_info_->NoteConstExpr(expr, InterpValue::MakeTuple({channel, channel}));
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleColonRef(const ColonRef* expr) {
  XLS_ASSIGN_OR_RETURN(auto subject, ResolveColonRefSubjectAfterTypeChecking(
                                         import_data_, type_info_, expr));
  return absl::visit(
      Visitor{
          [&](EnumDef* enum_def) -> absl::Status {
            // LHS is an EnumDef! Extract the value of the attr being
            // referenced.
            XLS_ASSIGN_OR_RETURN(Expr * member_value_expr,
                                 enum_def->GetValue(expr->attr()));

            // Since enum defs can't [currently] be parameterized, this is safe.
            XLS_ASSIGN_OR_RETURN(
                TypeInfo * type_info,
                import_data_->GetRootTypeInfoForNode(enum_def));

            XLS_RETURN_IF_ERROR(Evaluate(import_data_, type_info,
                                         warning_collector_, bindings_,
                                         member_value_expr));
            XLS_RET_CHECK(type_info->IsKnownConstExpr(member_value_expr));
            type_info_->NoteConstExpr(
                expr, type_info->GetConstExpr(member_value_expr).value());
            return absl::OkStatus();
          },
          [&](BuiltinNameDef* builtin_name_def) -> absl::Status {
            XLS_ASSIGN_OR_RETURN(
                InterpValue value,
                GetBuiltinNameDefColonAttr(builtin_name_def, expr->attr()));
            type_info_->NoteConstExpr(expr, value);
            return absl::OkStatus();
          },
          [&](ArrayTypeAnnotation* array_type_annotation) -> absl::Status {
            XLS_ASSIGN_OR_RETURN(
                TypeInfo * type_info,
                import_data_->GetRootTypeInfoForNode(array_type_annotation));
            XLS_RET_CHECK(
                type_info->IsKnownConstExpr(array_type_annotation->dim()));
            XLS_ASSIGN_OR_RETURN(
                InterpValue dim,
                type_info->GetConstExpr(array_type_annotation->dim()));
            XLS_ASSIGN_OR_RETURN(uint64_t dim_u64, dim.GetBitValueViaSign());
            XLS_ASSIGN_OR_RETURN(InterpValue value,
                                 GetArrayTypeColonAttr(array_type_annotation,
                                                       dim_u64, expr->attr()));
            type_info_->NoteConstExpr(expr, value);
            return absl::OkStatus();
          },
          [&](Module* module) -> absl::Status {
            // Ok! The subject is a module. The only case we care about here is
            // if the attr is a constant.
            std::optional<ModuleMember*> maybe_member =
                module->FindMemberWithName(expr->attr());
            if (!maybe_member.has_value()) {
              return absl::InternalError(
                  absl::StrFormat("\"%s\" is not a member of module \"%s\".",
                                  expr->attr(), module->name()));
            }

            if (!std::holds_alternative<ConstantDef*>(*maybe_member.value())) {
              VLOG(3) << expr->ToString() << " is not constexpr evaluatable.";
              return absl::OkStatus();
            }

            XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                                 import_data_->GetRootTypeInfo(module));

            ConstantDef* constant_def =
                std::get<ConstantDef*>(*maybe_member.value());
            XLS_RETURN_IF_ERROR(Evaluate(import_data_, type_info,
                                         warning_collector_, bindings_,
                                         constant_def->value()));
            XLS_RET_CHECK(type_info->IsKnownConstExpr(constant_def->value()));
            type_info_->NoteConstExpr(
                expr, type_info->GetConstExpr(constant_def->value()).value());
            return absl::OkStatus();
          },
          [&](Impl* impl) -> absl::Status {
            XLS_RET_CHECK(type_info_->IsKnownConstExpr(expr));
            return absl::OkStatus();
          },
      },
      subject);
}

absl::Status ConstexprEvaluator::HandleConstAssert(
    const ConstAssert* const_assert) {
  const FileTable& file_table = *const_assert->owner()->file_table();
  GET_CONSTEXPR_OR_RETURN(InterpValue predicate, const_assert->arg());
  if (predicate.IsTrue()) {
    return absl::OkStatus();
  }
  return TypeInferenceErrorStatus(const_assert->span(), nullptr,
                                  "const_assert! expression was false",
                                  file_table);
}

absl::Status ConstexprEvaluator::HandleFor(const For* expr) {
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleFunctionRef(const FunctionRef* expr) {
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleIndex(const Index* expr) {
  VLOG(3) << "ConstexprEvaluator::HandleIndex : " << expr->ToString();
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->lhs());

  if (std::holds_alternative<Expr*>(expr->rhs())) {
    EVAL_AS_CONSTEXPR_OR_RETURN(std::get<Expr*>(expr->rhs()));
  } else if (std::holds_alternative<Slice*>(expr->rhs())) {
    Slice* slice = std::get<Slice*>(expr->rhs());
    if (slice->start() != nullptr) {
      EVAL_AS_CONSTEXPR_OR_RETURN(slice->start());
    }
    if (slice->limit() != nullptr) {
      EVAL_AS_CONSTEXPR_OR_RETURN(slice->limit());
    }
  } else {
    WidthSlice* width_slice = std::get<WidthSlice*>(expr->rhs());
    EVAL_AS_CONSTEXPR_OR_RETURN(width_slice->start());
  }

  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleInvocation(const Invocation* expr) {
  std::optional<std::string_view> called_name;
  auto* callee_name_ref = dynamic_cast<NameRef*>(expr->callee());
  if (callee_name_ref != nullptr) {
    called_name = callee_name_ref->identifier();
    if (called_name == "send" || called_name == "send_if" ||
        called_name == "recv" || called_name == "recv_if" ||
        called_name == "recv_non_blocking" ||
        called_name == "recv_if_non_blocking") {
      // I/O operations are never constexpr.
      return absl::OkStatus();
    }
  }

  if (called_name == "map") {
    // Map "invocations" are special - only the first (of two) args must be
    // constexpr (the second must be a fn to apply).
    EVAL_AS_CONSTEXPR_OR_RETURN(expr->args()[0])
  } else {
    // A regular invocation is constexpr iff its args are constexpr.
    for (const auto* arg : expr->args()) {
      EVAL_AS_CONSTEXPR_OR_RETURN(arg)
    }
  }

  // We don't [yet] have a static assert fn, meaning that we don't want to catch
  // runtime errors here. If we detect that a program has failed (due to
  // execution of a `fail!` or unmatched `match`, then just assume we're ok.
  absl::Status status = InterpretExpr(expr);
  if (!status.ok() && !absl::StartsWith(status.message(), "FailureError")) {
    return status;
  }

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleMatch(const Match* expr) {
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->matched());

  for (const auto* arm : expr->arms()) {
    EVAL_AS_CONSTEXPR_OR_RETURN(arm->expr());
  }

  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleNameRef(const NameRef* expr) {
  AstNode* name_def = ToAstNode(expr->name_def());

  if (type_info_->IsKnownNonConstExpr(name_def) ||
      !type_info_->IsKnownConstExpr(name_def)) {
    return absl::OkStatus();
  }
  type_info_->NoteConstExpr(expr, type_info_->GetConstExpr(name_def).value());
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> EvaluateNumber(const Number& expr,
                                           const Type& type) {
  const FileTable& file_table = *expr.owner()->file_table();
  XLS_RET_CHECK(!type.IsMeta())
      << "Got invalid type when evaluating number: " << type.ToString() << " @ "
      << expr.span().ToString(file_table);
  VLOG(4) << "Evaluating number: " << expr.ToString() << " @ "
          << expr.span().ToString(file_table);

  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);

  XLS_RET_CHECK(bits_like.has_value())
      << "Type for number should be bits-like; got: " << type;

  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());

  InterpValueTag tag =
      is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;

  const std::variant<InterpValue, TypeDim::OwnedParametric>& value =
      bits_like->size.value();
  if (!std::holds_alternative<InterpValue>(value)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot evaluate number %s as type %s is parametric",
                        expr.ToString(), type.ToString()));
  }

  XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                       std::get<InterpValue>(value).GetBitValueViaSign());

  XLS_ASSIGN_OR_RETURN(Bits bits, expr.GetBits(bit_count, file_table));
  return InterpValue::MakeBits(tag, std::move(bits));
}

absl::Status ConstexprEvaluator::HandleNumber(const Number* expr) {
  // Numbers should always be [constexpr] evaluatable.
  XLS_ASSIGN_OR_RETURN(ConstexprEnvData constexpr_env_data,
                       MakeConstexprEnv(import_data_, type_info_,
                                        warning_collector_, expr, bindings_));
  XLS_RET_CHECK(constexpr_env_data.non_constexpr.empty());

  std::unique_ptr<BitsType> temp_type;
  const Type* type_ptr;
  if (expr->type_annotation() != nullptr) {
    // If the number is annotated with a type, then extract it to pass to
    // EvaluateNumber (for consistency checking). It might be that the type is
    // parametric, in which case we'll need to fully instantiate it.
    auto maybe_type_ptr = type_info_->GetItem(expr->type_annotation());
    XLS_RET_CHECK(maybe_type_ptr.has_value());
    const MetaType* tt = dynamic_cast<const MetaType*>(maybe_type_ptr.value());
    XLS_RET_CHECK(tt != nullptr);
    type_ptr = tt->wrapped().get();

    std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type_ptr);
    XLS_RET_CHECK(bits_like.has_value())
        << "Type for number should be bits-like; got: " << type_ptr->ToString();

    // Materialize the bits type.
    XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_like->size.GetAsInt64());
    temp_type = std::make_unique<BitsType>(is_signed, bit_count);
    type_ptr = temp_type.get();
  } else if (type_ != nullptr) {
    type_ptr = type_;
  } else if (expr->number_kind() == NumberKind::kBool) {
    temp_type = std::make_unique<BitsType>(false, 1);
    type_ptr = temp_type.get();
  } else if (expr->number_kind() == NumberKind::kCharacter) {
    temp_type = std::make_unique<BitsType>(false, 8);
    type_ptr = temp_type.get();
  } else {
    // "Undecorated" numbers that make it through typechecking are `usize`,
    // which currently is u32.
    temp_type = std::make_unique<BitsType>(false, 32);
    type_ptr = temp_type.get();
  }

  XLS_RET_CHECK(type_ptr != nullptr);
  XLS_ASSIGN_OR_RETURN(InterpValue value, EvaluateNumber(*expr, *type_ptr));
  type_info_->NoteConstExpr(expr, value);

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleRange(const Range* expr) {
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->start());
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->end());
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleSplatStructInstance(
    const SplatStructInstance* expr) {
  // A struct instance is constexpr iff all its members and the basis struct are
  // constexpr.
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->splatted());

  for (const auto& [k, v] : expr->members()) {
    EVAL_AS_CONSTEXPR_OR_RETURN(v);
  }
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleString(const String* expr) {
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleStructInstance(
    const StructInstance* expr) {
  // A struct instance is constexpr iff all its members are constexpr.
  for (const auto& [k, v] : expr->GetUnorderedMembers()) {
    EVAL_AS_CONSTEXPR_OR_RETURN(v);
  }
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleConditional(const Conditional* expr) {
  // Simple enough that we don't need to invoke the interpreter.
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->test());
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->consequent());
  EVAL_AS_CONSTEXPR_OR_RETURN(ToExprNode(expr->alternate()));

  InterpValue test = type_info_->GetConstExpr(expr->test()).value();
  if (test.IsTrue()) {
    type_info_->NoteConstExpr(
        expr, type_info_->GetConstExpr(expr->consequent()).value());
  } else {
    type_info_->NoteConstExpr(
        expr, type_info_->GetConstExpr(ToExprNode(expr->alternate())).value());
  }

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleUnop(const Unop* expr) {
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->operand());

  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleTupleIndex(const TupleIndex* expr) {
  // No need to fire up the interpreter. This one is easy.
  GET_CONSTEXPR_OR_RETURN(InterpValue tuple, expr->lhs());
  GET_CONSTEXPR_OR_RETURN(InterpValue index, expr->index());

  XLS_ASSIGN_OR_RETURN(uint64_t index_value, index.GetBitValueUnsigned());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       tuple.GetValues());
  if (index_value < 0 || index_value > values->size()) {
    const FileTable& file_table = *expr->owner()->file_table();
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: Out-of-range tuple index: %d vs %d.",
        expr->span().ToString(file_table), index_value, values->size()));
  }
  type_info_->NoteConstExpr(expr, values->at(index_value));
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleUnrollFor(const UnrollFor* expr) {
  std::optional<const Expr*> unrolled =
      type_info_->GetUnrolledLoop(expr, bindings_);
  if (unrolled.has_value()) {
    return (*unrolled)->AcceptExpr(this);
  }
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleVerbatimNode(const VerbatimNode* node) {
  return absl::UnimplementedError("Should not evaluate VerbatimNode");
}

absl::Status ConstexprEvaluator::HandleXlsTuple(const XlsTuple* expr) {
  std::vector<InterpValue> values;
  for (const Expr* member : expr->members()) {
    GET_CONSTEXPR_OR_RETURN(InterpValue value, member);
    values.push_back(value);
  }

  // No need to fire up the interpreter. We can handle this one.
  type_info_->NoteConstExpr(expr, InterpValue::MakeTuple(values));
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::InterpretExpr(const Expr* expr) {
  XLS_ASSIGN_OR_RETURN(ConstexprEnvData constexpr_env_data,
                       MakeConstexprEnv(import_data_, type_info_,
                                        warning_collector_, expr, bindings_));

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(import_data_, type_info_, expr,
                                      constexpr_env_data.env, bindings_));

  std::vector<Span> rollovers;
  BytecodeInterpreterOptions options;
  options.rollover_hook([&](const Span& s) { rollovers.push_back(s); });
  XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                       BytecodeInterpreter::Interpret(import_data_, bf.get(),
                                                      /*args=*/{}));
  if (warning_collector_ != nullptr) {
    for (const Span& s : rollovers) {
      warning_collector_->Add(
          s, WarningKind::kConstexprEvalRollover,
          "constexpr evaluation detected rollover in operation");
    }
  }
  type_info_->NoteConstExpr(expr, constexpr_value);

  return absl::OkStatus();
}

absl::StatusOr<ConstexprEnvData> MakeConstexprEnv(
    ImportData* import_data, TypeInfo* type_info,
    WarningCollector* warning_collector, const Expr* node,
    const ParametricEnv& parametric_env) {
  CHECK_EQ(node->owner(), type_info->module())
      << "expr `" << node->ToString()
      << "` from module: " << node->owner()->name()
      << " vs type info module: " << type_info->module()->name();
  VLOG(5) << "Creating constexpr environment for node: `" << node->ToString()
          << "`";

  // The constexpr environment we'll build up as we walk the free variables.
  absl::flat_hash_map<std::string, InterpValue> env;

  // Seed the constexpr environment with parametric bindings.
  //
  // Implementation note: we could instead have this function expect to resolve
  // all of these via `type_info`, but being able to take the `parametric_env`
  // and use that for initial population allows us to pass in partially built
  // parametric results, e.g. in parametric instantiation, without needing to
  // put everything we discover immediately into the type info.
  for (const auto& [id, value] : parametric_env.ToMap()) {
    env.insert({id, value});
  }

  // Collect all the freevars that are constexpr.
  FreeVariables freevars = GetFreeVariablesByPos(node);
  VLOG(5) << absl::StreamFormat("free variables for `%s`: %s", node->ToString(),
                                freevars.ToString());
  freevars = freevars.DropBuiltinDefs();

  // Keep track of which name references we could not resolve to be constexpr.
  absl::flat_hash_set<const NameRef*> non_constexpr;

  for (const auto& [name, name_refs] : freevars.values()) {
#ifndef NDEBUG
    // Verify that all the name refs are referring to the same name def -- this
    // should be true of free variables for any lexical environment.
    for (int64_t i = 1; i < name_refs.size(); ++i) {
      DCHECK(name_refs[i]->name_def() == name_refs[0]->name_def());
    }
#endif

    CHECK_GE(name_refs.size(), 1);
    // Note: we dropped all builtin defs above so this variant access should be
    // safe/correct.
    const NameRef* sample_ref = name_refs.front();
    const NameDef* target_def =
        std::get<const NameDef*>(sample_ref->name_def());

    XLS_RETURN_IF_ERROR(
        ConstexprEvaluator::Evaluate(import_data, type_info, warning_collector,
                                     parametric_env, sample_ref, nullptr));
    absl::StatusOr<InterpValue> const_expr =
        type_info->GetConstExpr(target_def);
    if (const_expr.ok()) {
      env.insert({name, const_expr.value()});
    } else {
      non_constexpr.insert(sample_ref);
    }
  }

  return ConstexprEnvData{
      .freevars = freevars,
      .env = env,
      .non_constexpr = non_constexpr,
  };
}

std::string EnvMapToString(
    const absl::flat_hash_map<std::string, InterpValue>& map) {
  std::vector<const std::pair<const std::string, InterpValue>*> items;
  items.reserve(map.size());
  for (const auto& item : map) {
    items.push_back(&item);
  }
  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    return a->first < b->first;  // Order by key.
  });
  std::string guts =
      absl::StrJoin(items, ", ", [](std::string* out, const auto& item) {
        absl::StrAppend(out, item->first, ": ", item->second.ToString());
      });
  return absl::StrCat("{", guts, "}");
}

}  // namespace xls::dslx
