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

#include <variant>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/ast_utils.h"
#include "xls/dslx/builtins_metadata.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/bytecode_interpreter.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

// Visitor to collect all NameRefs defined externally to a given expression,
// notably a "for" expression. Only those nodes capable of containing an outside
// NameRef are populated, e.g., `Number` isn't populated.
// This collection is a bit specialized, so we don't use TraitVisitor here.
class NameRefCollector : public ExprVisitor {
 public:
  absl::Status HandleArray(const Array* expr) override {
    for (const auto* member : expr->members()) {
      XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleAttr(const Attr* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleBlock(const Block* expr) override {
    return expr->body()->AcceptExpr(this);
  }
  absl::Status HandleBinop(const Binop* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->rhs()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleCast(const Cast* expr) override {
    XLS_RETURN_IF_ERROR(expr->expr()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleChannelDecl(const ChannelDecl* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleColonRef(const ColonRef* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleConstantArray(const ConstantArray* expr) override {
    return HandleArray(expr);
  }
  absl::Status HandleConstRef(const ConstRef* expr) override {
    XLS_RETURN_IF_ERROR(expr->GetValue()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleFor(const For* expr) override {
    XLS_RETURN_IF_ERROR(expr->init()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleFormatMacro(const FormatMacro* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleIndex(const Index* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    if (std::holds_alternative<Expr*>(expr->rhs())) {
      XLS_RETURN_IF_ERROR(std::get<Expr*>(expr->rhs())->AcceptExpr(this));
    }
    // No NameRefs in slice RHSes.
    return absl::OkStatus();
  }
  absl::Status HandleInvocation(const Invocation* expr) override {
    for (const auto* arg : expr->args()) {
      XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleJoin(const Join* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleLet(const Let* expr) override {
    XLS_RETURN_IF_ERROR(expr->rhs()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(this));

    std::vector<NameDefTree::Leaf> leaves = expr->name_def_tree()->Flatten();
    for (const auto& leaf : leaves) {
      if (std::holds_alternative<NameDef*>(leaf)) {
        name_defs_.insert(std::get<NameDef*>(leaf));
      }
    }
    return absl::OkStatus();
  }
  absl::Status HandleMatch(const Match* expr) override {
    XLS_RETURN_IF_ERROR(expr->matched()->AcceptExpr(this));
    for (const MatchArm* arm : expr->arms()) {
      XLS_RETURN_IF_ERROR(arm->expr()->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleNameRef(const NameRef* expr) override {
    name_refs_.push_back(expr);
    return absl::OkStatus();
  }
  absl::Status HandleNumber(const Number* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRange(const Range* expr) override {
    XLS_RETURN_IF_ERROR(expr->start()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->end()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleRecv(const Recv* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRecvIf(const RecvIf* expr) override {
    XLS_RETURN_IF_ERROR(expr->condition()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleRecvIfNonBlocking(const RecvIfNonBlocking* expr) override {
    XLS_RETURN_IF_ERROR(expr->condition()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleRecvNonBlocking(const RecvNonBlocking* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleSend(const Send* expr) override {
    XLS_RETURN_IF_ERROR(expr->payload()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleSendIf(const SendIf* expr) override {
    XLS_RETURN_IF_ERROR(expr->condition()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->payload()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleSpawn(const Spawn* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleString(const String* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* expr) override {
    for (const auto& [name, member_expr] : expr->members()) {
      XLS_RETURN_IF_ERROR(member_expr->AcceptExpr(this));
    }
    XLS_RETURN_IF_ERROR(expr->splatted()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleStructInstance(const StructInstance* expr) override {
    for (const auto& [name, member_expr] : expr->GetUnorderedMembers()) {
      XLS_RETURN_IF_ERROR(member_expr->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleTernary(const Ternary* expr) override {
    XLS_RETURN_IF_ERROR(expr->test()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->consequent()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->alternate()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleUnop(const Unop* expr) override {
    XLS_RETURN_IF_ERROR(expr->operand()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleTupleIndex(const TupleIndex* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleUnrollFor(const UnrollFor* expr) override {
    XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleXlsTuple(const XlsTuple* expr) override {
    for (const Expr* member : expr->members()) {
      XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  std::vector<const NameRef*> outside_name_refs() {
    std::vector<const NameRef*> result;
    for (const NameRef* name_ref : name_refs_) {
      if (std::holds_alternative<BuiltinNameDef*>(name_ref->name_def())) {
        continue;
      }

      if (!name_defs_.contains(
              std::get<const NameDef*>(name_ref->name_def())) &&
          !IsNameParametricBuiltin(name_ref->identifier())) {
        result.push_back(name_ref);
      }
    }

    return result;
  }

  absl::flat_hash_set<const NameDef*>& inside_name_defs() { return name_defs_; }

 private:
  std::vector<const NameRef*> name_refs_;
  absl::flat_hash_set<const NameDef*> name_defs_;
};

// Fully instantiate the given parametric BitsType using the symbol mappings in
// `env`.
absl::StatusOr<std::unique_ptr<BitsType>> InstantiateParametricNumberType(
    const absl::flat_hash_map<std::string, InterpValue>& env,
    const BitsType* bits_type) {
  ParametricExpression::Env parametric_env;
  for (const auto& [k, v] : env) {
    parametric_env[k] = v;
  }
  ParametricExpression::Evaluated e =
      bits_type->size().parametric().Evaluate(parametric_env);
  if (!std::holds_alternative<InterpValue>(e)) {
    return absl::InternalError(
        absl::StrCat("Parametric number size did not evaluate to a constant: ",
                     bits_type->size().ToString()));
  }
  return std::make_unique<BitsType>(
      bits_type->is_signed(),
      std::get<InterpValue>(e).GetBitValueInt64().value());
}

}  // namespace

/* static */ absl::Status ConstexprEvaluator::Evaluate(
    ImportData* import_data, TypeInfo* type_info,
    const SymbolicBindings& bindings, const Expr* expr,
    const ConcreteType* concrete_type) {
  if (type_info->IsKnownConstExpr(expr) ||
      type_info->IsKnownNonConstExpr(expr)) {
    return absl::OkStatus();
  }
  ConstexprEvaluator evaluator(import_data, type_info, bindings, concrete_type);
  return expr->AcceptExpr(&evaluator);
}

/* static */ absl::StatusOr<InterpValue> ConstexprEvaluator::EvaluateToValue(
    ImportData* import_data, TypeInfo* type_info,
    const SymbolicBindings& bindings, const Expr* expr,
    const ConcreteType* concrete_type) {
  XLS_RETURN_IF_ERROR(Evaluate(import_data, type_info, bindings, expr));
  if (type_info->IsKnownConstExpr(expr)) {
    return type_info->GetConstExpr(expr);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Expr was not constexpr: ", expr->ToString()));
}

// Evaluates the given expression and terminates current function execution
// if it is not constexpr.
#define EVAL_AS_CONSTEXPR_OR_RETURN(EXPR)                            \
  if (!type_info_->IsKnownConstExpr(EXPR) &&                         \
      !type_info_->IsKnownNonConstExpr(EXPR)) {                      \
    ConcreteType* sub_type = nullptr;                                \
    if (type_info_->GetItem(EXPR).has_value()) {                     \
      sub_type = type_info_->GetItem(EXPR).value();                  \
    }                                                                \
    ConstexprEvaluator sub_eval(import_data_, type_info_, bindings_, \
                                sub_type);                           \
    XLS_RETURN_IF_ERROR(EXPR->AcceptExpr(&sub_eval));                \
  }                                                                  \
  if (!type_info_->IsKnownConstExpr(EXPR)) {                         \
    return absl::OkStatus();                                         \
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

absl::Status ConstexprEvaluator::HandleArray(const Array* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleArray : " << expr->ToString();
  std::vector<InterpValue> values;
  for (const Expr* member : expr->members()) {
    GET_CONSTEXPR_OR_RETURN(InterpValue value, member);
    values.push_back(value);
  }

  if (concrete_type_ != nullptr) {
    auto* array_type = dynamic_cast<const ArrayType*>(concrete_type_);
    if (array_type == nullptr) {
      return absl::InternalError(
          absl::StrCat(expr->span().ToString(), " : ",
                       "Array ConcreteType was not an ArrayType!"));
    }

    ConcreteTypeDim size = array_type->size();
    absl::StatusOr<int64_t> int_size_or = size.GetAsInt64();
    if (!int_size_or.ok()) {
      return absl::InternalError(absl::StrCat(expr->span().ToString(), " : ",
                                              int_size_or.status().message()));
    }

    int64_t int_size = int_size_or.value();
    int64_t remaining = int_size - values.size();
    while (remaining-- > 0) {
      values.push_back(values.back());
    }
  }

  // No need to fire up the interpreter. We can handle this one.
  XLS_ASSIGN_OR_RETURN(InterpValue array, InterpValue::MakeArray(values));
  type_info_->NoteConstExpr(expr, array);
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleBinop(const Binop* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleBinop : " << expr->ToString();
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->lhs());
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->rhs());
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleBlock(const Block* expr) {
  XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(this));
  if (type_info_->IsKnownConstExpr(expr->body())) {
    type_info_->NoteConstExpr(expr,
                              type_info_->GetConstExpr(expr->body()).value());
  }
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleCast(const Cast* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleCast : " << expr->ToString();
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->expr());
  return InterpretExpr(expr);
}

// Creates an InterpValue for the described channel or array of channels.
absl::StatusOr<InterpValue> ConstexprEvaluator::CreateChannelValue(
    const ConcreteType* concrete_type) {
  if (auto* array_type = dynamic_cast<const ArrayType*>(concrete_type)) {
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
  const ChannelType* ct = dynamic_cast<const ChannelType*>(concrete_type);
  XLS_RET_CHECK_NE(ct, nullptr);
  return InterpValue::MakeChannel();
}

// While a channel's *contents* aren't constexpr, the existence of the channel
// itself is.
absl::Status ConstexprEvaluator::HandleChannelDecl(const ChannelDecl* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleChannelDecl : " << expr->ToString();
  // Keep in mind that channels come in tuples, so peel out the first element.
  std::optional<ConcreteType*> maybe_decl_type = type_info_->GetItem(expr);
  if (!maybe_decl_type.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Could not find type for expr \"%s\" @ %s",
                        expr->ToString(), expr->span().ToString()));
  }

  auto* tuple_type = dynamic_cast<TupleType*>(maybe_decl_type.value());
  if (tuple_type == nullptr) {
    return TypeInferenceErrorStatus(expr->span(), maybe_decl_type.value(),
                                    "Channel decl did not have tuple type:");
  }

  // Verify that the channel tuple has exactly two elements; just yank one out
  // for channel [array] creation (they both point to the same object).
  if (tuple_type->size() != 2) {
    return TypeInferenceErrorStatus(
        expr->span(), tuple_type, "ChannelDecl type was a two-element tuple.");
  }

  XLS_ASSIGN_OR_RETURN(InterpValue channel,
                       CreateChannelValue(&tuple_type->GetMemberType(0)));
  type_info_->NoteConstExpr(expr, InterpValue::MakeTuple({channel, channel}));
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleColonRef(const ColonRef* expr) {
  XLS_ASSIGN_OR_RETURN(auto subject,
                       ResolveColonRefSubject(import_data_, type_info_, expr));
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

            XLS_RETURN_IF_ERROR(Evaluate(import_data_, type_info, bindings_,
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
            XLS_ASSIGN_OR_RETURN(uint64_t dim_u64, dim.GetBitValueUint64());
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
              XLS_VLOG(3) << "ConstRef \"" << expr->ToString()
                          << "\" is not constexpr evaluatable.";
              return absl::OkStatus();
            }

            XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                                 import_data_->GetRootTypeInfo(module));

            ConstantDef* constant_def =
                std::get<ConstantDef*>(*maybe_member.value());
            XLS_RETURN_IF_ERROR(Evaluate(import_data_, type_info, bindings_,
                                         constant_def->value()));
            XLS_RET_CHECK(type_info->IsKnownConstExpr(constant_def->value()));
            type_info_->NoteConstExpr(
                expr, type_info->GetConstExpr(constant_def->value()).value());
            return absl::OkStatus();
          },
      },
      subject);
}

absl::Status ConstexprEvaluator::HandleConstantArray(
    const ConstantArray* expr) {
  return HandleArray(expr);
}

absl::Status ConstexprEvaluator::HandleConstRef(const ConstRef* expr) {
  return HandleNameRef(expr);
}

absl::Status ConstexprEvaluator::HandleFor(const For* expr) {
  // A `for` loop evaluates constexpr if its init and enumeration values as
  // well as any external NameRefs are constexpr.
  XLS_VLOG(3) << "ConstexprEvaluator::HandleFor: " << expr->ToString();
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->init());
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->iterable());

  // Since a `for` loop can refer to vars outside the loop body itself, we need
  // to make sure that every NameRef is also constexpr.
  std::vector<NameDef*> bound_defs = expr->names()->GetNameDefs();
  absl::flat_hash_set<const NameDef*> bound_def_set(bound_defs.begin(),
                                                    bound_defs.end());

  NameRefCollector collector;
  XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(&collector));
  for (const NameRef* name_ref : collector.outside_name_refs()) {
    // We can't bind to a BuiltinNameDef, so this std::get is safe.
    XLS_RETURN_IF_ERROR(name_ref->AcceptExpr(this));
    if (!type_info_->IsKnownConstExpr(name_ref) &&
        !bound_def_set.contains(
            std::get<const NameDef*>(name_ref->name_def()))) {
      return absl::OkStatus();
    }
  }

  // When constexpr eval'ing, we also don't want names declared inside the loop
  // to shadow the [potentially] constexpr values declared outside.
  bound_def_set.insert(collector.inside_name_defs().begin(),
                       collector.inside_name_defs().end());

  // We don't [yet] have a static assert fn, meaning that we don't want to catch
  // runtime errors here. If we detect that a program has failed (due to
  // execution of a `fail!` or unmatched `match`, then just assume we're ok.
  absl::Status status = InterpretExpr(expr, bound_def_set);
  if (!status.ok() && !absl::StartsWith(status.message(), "FailureError")) {
    return status;
  }
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleIndex(const Index* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleIndex : " << expr->ToString();
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
  // Map "invocations" are special - only the first (of two) args must be
  // constexpr (the second must be a fn to apply).
  auto* callee_name_ref = dynamic_cast<NameRef*>(expr->callee());
  bool callee_is_map =
      callee_name_ref != nullptr && callee_name_ref->identifier() == "map";
  if (callee_is_map) {
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
    return absl::Status();
  }
  type_info_->NoteConstExpr(expr, type_info_->GetConstExpr(name_def).value());
  return absl::OkStatus();
}

// Evaluates a Number AST node to an InterpValue.
absl::StatusOr<InterpValue> EvaluateNumber(const Number* expr,
                                           const ConcreteType* type) {
  XLS_VLOG(4) << "Evaluating number: " << expr->ToString() << " @ "
              << expr->span();
  const BitsType* bits_type = dynamic_cast<const BitsType*>(type);
  XLS_RET_CHECK(bits_type != nullptr)
      << "Type for number should be 'bits' kind.";
  InterpValueTag tag =
      bits_type->is_signed() ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  XLS_ASSIGN_OR_RETURN(
      int64_t bit_count,
      std::get<InterpValue>(bits_type->size().value()).GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, expr->GetBits(bit_count));
  return InterpValue::MakeBits(tag, std::move(bits));
}

absl::Status ConstexprEvaluator::HandleNumber(const Number* expr) {
  // Numbers should always be [constexpr] evaluatable.
  absl::flat_hash_map<std::string, InterpValue> env;
  XLS_ASSIGN_OR_RETURN(
      env, MakeConstexprEnv(import_data_, type_info_, expr, bindings_));

  std::unique_ptr<BitsType> temp_type;
  const ConcreteType* type_ptr;
  if (expr->type_annotation() != nullptr) {
    // If the number is annotated with a type, then extract it to pass to
    // EvaluateNumber (for consistency checking). It might be that the type is
    // parametric, in which case we'll need to fully instantiate it.
    auto maybe_type_ptr = type_info_->GetItem(expr->type_annotation());
    XLS_RET_CHECK(maybe_type_ptr.has_value());
    type_ptr = maybe_type_ptr.value();
    const BitsType* bt = down_cast<const BitsType*>(type_ptr);
    if (bt->size().IsParametric()) {
      XLS_ASSIGN_OR_RETURN(temp_type, InstantiateParametricNumberType(env, bt));
      type_ptr = temp_type.get();
    }
  } else if (concrete_type_ != nullptr) {
    type_ptr = concrete_type_;
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

  XLS_ASSIGN_OR_RETURN(InterpValue value, EvaluateNumber(expr, type_ptr));
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

absl::Status ConstexprEvaluator::HandleStructInstance(
    const StructInstance* expr) {
  // A struct instance is constexpr iff all its members are constexpr.
  for (const auto& [k, v] : expr->GetUnorderedMembers()) {
    EVAL_AS_CONSTEXPR_OR_RETURN(v);
  }
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleTernary(const Ternary* expr) {
  // Simple enough that we don't need to invoke the interpreter.
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->test());
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->consequent());
  EVAL_AS_CONSTEXPR_OR_RETURN(expr->alternate());

  InterpValue test = type_info_->GetConstExpr(expr->test()).value();
  if (test.IsTrue()) {
    type_info_->NoteConstExpr(
        expr, type_info_->GetConstExpr(expr->consequent()).value());
  } else {
    type_info_->NoteConstExpr(
        expr, type_info_->GetConstExpr(expr->alternate()).value());
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

  XLS_ASSIGN_OR_RETURN(uint64_t index_value, index.GetBitValueUint64());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       tuple.GetValues());
  if (index_value < 0 || index_value > values->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: Out-of-range tuple index: %d vs %d.",
                        expr->span().ToString(), index_value, values->size()));
  }
  type_info_->NoteConstExpr(expr, values->at(index_value));
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleUnrollFor(const UnrollFor* expr) {
  return absl::OkStatus();
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

absl::Status ConstexprEvaluator::InterpretExpr(
    const Expr* expr, absl::flat_hash_set<const NameDef*> bypass_env) {
  absl::flat_hash_map<std::string, InterpValue> env;
  XLS_ASSIGN_OR_RETURN(env, MakeConstexprEnv(import_data_, type_info_, expr,
                                             bindings_, bypass_env));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BytecodeFunction> bf,
                       BytecodeEmitter::EmitExpression(import_data_, type_info_,
                                                       expr, env, bindings_));

  XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                       BytecodeInterpreter::Interpret(import_data_, bf.get(),
                                                      /*args=*/{}));
  type_info_->NoteConstExpr(expr, constexpr_value);

  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>> MakeConstexprEnv(
    ImportData* import_data, TypeInfo* type_info, const Expr* node,
    const SymbolicBindings& symbolic_bindings,
    absl::flat_hash_set<const NameDef*> bypass_env) {
  XLS_CHECK_EQ(node->owner(), type_info->module())
      << "expr `" << node->ToString()
      << "` from module: " << node->owner()->name()
      << " vs type info module: " << type_info->module()->name();
  XLS_VLOG(5) << "Creating constexpr environment for node: "
              << node->ToString();
  absl::flat_hash_map<std::string, InterpValue> env;
  absl::flat_hash_map<std::string, InterpValue> values;

  for (auto [id, value] : symbolic_bindings.ToMap()) {
    env.insert({id, value});
  }

  // Collect all the freevars that are constexpr.
  FreeVariables freevars = node->GetFreeVariables();
  XLS_VLOG(5) << "freevars for " << node->ToString() << ": "
              << freevars.GetFreeVariableCount();
  freevars = freevars.DropBuiltinDefs();
  for (const auto& [name, name_refs] : freevars.values()) {
    const NameRef* target_ref = nullptr;
    for (const NameRef* name_ref : name_refs) {
      if (!bypass_env.contains(
              std::get<const NameDef*>(name_ref->name_def()))) {
        target_ref = name_ref;
        break;
      }
    }

    if (target_ref == nullptr) {
      continue;
    }

    XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
        import_data, type_info, symbolic_bindings, target_ref, nullptr));
    absl::StatusOr<InterpValue> const_expr =
        type_info->GetConstExpr(target_ref);
    if (const_expr.ok()) {
      env.insert({name, const_expr.value()});
    }
  }

  for (const ConstRef* const_ref : freevars.GetConstRefs()) {
    XLS_VLOG(5) << "analyzing constant reference: " << const_ref->ToString()
                << " def: " << const_ref->ToString();
    Expr* const_expr = const_ref->GetValue();
    absl::StatusOr<InterpValue> value = type_info->GetConstExpr(const_expr);
    if (!value.ok()) {
      continue;
    }

    XLS_VLOG(5) << "freevar env record: " << const_ref->identifier() << " => "
                << value->ToString();
    env.insert({const_ref->identifier(), *value});
  }

  return env;
}

}  // namespace xls::dslx
