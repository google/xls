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

#include "xls/solvers/z3_dslx_translator.h"

#include <cstdint>

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/symbolic_type.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3_api.h"

namespace xls::solvers::z3 {

absl::Status DslxTranslator::VisitSymbolicTree(SymbolicType* sym) {
  auto Walk = [&](SymbolicType* x) -> absl::Status {
    if (x == nullptr) {
      return absl::OkStatus();
    }
    if (x->IsLeaf()) {
      XLS_RETURN_IF_ERROR(ProcessSymbolicLeaf(x));
    } else {
      XLS_RETURN_IF_ERROR(ProcessSymbolicNode(x));
    }
    return absl::OkStatus();
  };
  XLS_RETURN_IF_ERROR(sym->DoPostorder(Walk));
  return absl::OkStatus();
}

absl::Status DslxTranslator::ProcessSymbolicNode(SymbolicType* sym) {
  if (sym->IsTernary()) {
    XLS_RETURN_IF_ERROR(HandleTernaryIf(sym));
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(SymbolicType::OpKind op, sym->op());
  if (absl::holds_alternative<dslx::BinopKind>(op)) {
    dslx::BinopKind binop = absl::get<dslx::BinopKind>(op);
    switch (binop) {
      case dslx::BinopKind::kAdd:
        XLS_RETURN_IF_ERROR(HandleAdd(sym));
        break;
      case dslx::BinopKind::kLogicalAnd:
      case dslx::BinopKind::kAnd:
        XLS_RETURN_IF_ERROR(HandleAnd(sym));
        break;
      case dslx::BinopKind::kEq:
        XLS_RETURN_IF_ERROR(HandleEq(sym));
        break;
      case dslx::BinopKind::kNe:
        XLS_RETURN_IF_ERROR(HandleNe(sym));
        break;
      case dslx::BinopKind::kLogicalOr:
      case dslx::BinopKind::kOr:
        XLS_RETURN_IF_ERROR(HandleOr(sym));
        break;
      case dslx::BinopKind::kGt:
        XLS_RETURN_IF_ERROR(HandleGt(sym));
        break;
      case dslx::BinopKind::kGe:
        XLS_RETURN_IF_ERROR(HandleGe(sym));
        break;
      case dslx::BinopKind::kShl:
        XLS_RETURN_IF_ERROR(HandleShll(sym));
        break;
      case dslx::BinopKind::kShr:
        if (sym->IsSigned()) {
          XLS_RETURN_IF_ERROR(HandleShra(sym));
        } else {
          XLS_RETURN_IF_ERROR(HandleShrl(sym));
        }
        break;
      case dslx::BinopKind::kLe:
        XLS_RETURN_IF_ERROR(HandleLe(sym));
        break;
      case dslx::BinopKind::kLt:
        XLS_RETURN_IF_ERROR(HandleLt(sym));
        break;
      case dslx::BinopKind::kMul:
        XLS_RETURN_IF_ERROR(HandleMul(sym));
        break;
      case dslx::BinopKind::kSub:
        XLS_RETURN_IF_ERROR(HandleSub(sym));
        break;
      case dslx::BinopKind::kXor:
        XLS_RETURN_IF_ERROR(HandleXor(sym));
        break;
      case dslx::BinopKind::kConcat:
        XLS_RETURN_IF_ERROR(HandleConcat(sym));
        break;
      case dslx::BinopKind::kDiv:
        XLS_RETURN_IF_ERROR(HandleDiv(sym));
        break;
      default:
        return absl::InternalError("Invalid binary operation kind " +
                                   dslx::BinopKindToString(binop));
    }
  } else {
    dslx::UnopKind unop = absl::get<dslx::UnopKind>(op);
    switch (unop) {
      case dslx::UnopKind::kInvert:
        XLS_RETURN_IF_ERROR(HandleNot(sym));
        break;
      case dslx::UnopKind::kNegate:
        XLS_RETURN_IF_ERROR(HandleNeg(sym));
        break;
      default:
        return absl::InternalError("Invalid unary operation kind " +
                                   UnopKindToString(unop));
    }
  }
  return absl::OkStatus();
}

absl::Status DslxTranslator::ProcessSymbolicLeaf(SymbolicType* sym) {
  if (sym->IsArray()) {
    for (SymbolicType* child : sym->GetChildren()) {
      XLS_RETURN_IF_ERROR(ProcessSymbolicLeaf(child));
    }
  } else {
    if (sym->IsBits()) {
      XLS_RETURN_IF_ERROR(HandleLiteral(sym));
    } else {  // node is a function parameter.
      XLS_RETURN_IF_ERROR(HandleParam(sym));
    }
  }
  return absl::OkStatus();
}

std::unique_ptr<DslxTranslator> DslxTranslator::CreateTranslator() {
  Z3_config config = Z3_mk_config();
  Z3_set_param_value(config, "proof", "true");
  auto translator = absl::WrapUnique(new DslxTranslator(config));
  return translator;
}

absl::Status DslxTranslator::TranslatePredicate(SymbolicType* predicate) {
  XLS_RET_CHECK(predicate != nullptr);
  if (translations_.contains(predicate)) {
    XLS_VLOG(2) << "TranslatePredicate: translations already contains: "
                << predicate->ToString().value() << " translation = "
                << Z3_ast_to_string(ctx_, translations_[predicate])
                << std::endl;
    return absl::OkStatus();
  }
  return VisitSymbolicTree(predicate);
}

absl::optional<Z3_ast> DslxTranslator::GetTranslation(
    const SymbolicType* source) {
  if (!translations_.contains(source)) return absl::nullopt;
  return translations_.at(source);
}

void DslxTranslator::SetTimeout(absl::Duration timeout) {
  std::string timeout_str = absl::StrCat(absl::ToInt64Milliseconds(timeout));
  Z3_update_param_value(ctx_, "timeout", timeout_str.c_str());
}

absl::Status DslxTranslator::HandleLiteral(SymbolicType* sym) {
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(Z3_ast result, TranslateLiteralValue(sym));
  XLS_RETURN_IF_ERROR(NoteTranslation(sym, result));
  return seh.status();
}

DslxTranslator::DslxTranslator(Z3_config config)
    : config_(config), ctx_(Z3_mk_context(config_)) {}

DslxTranslator::~DslxTranslator() {
  Z3_del_context(ctx_);
  Z3_del_config(config_);
}

absl::Status DslxTranslator::NoteTranslation(SymbolicType* node,
                                             Z3_ast translated) {
  if (translations_.contains(node)) {
    XLS_VLOG(2) << "Skipping translation of " << node->ToString().value()
                << ", as it's already been recorded ";
    return absl::OkStatus();
  }
  translations_[node] = translated;

  return absl::OkStatus();
}

absl::StatusOr<Z3_ast> DslxTranslator::TranslateLiteralValue(
    SymbolicType* sym) {
  if (sym->IsBits()) {
    int64_t bit_count = sym->bit_count();
    int64_t bit_value = sym->bit_value();
    std::unique_ptr<bool[]> booleans(new bool[bit_count]);
    for (int64_t i = 0; i < bit_count; ++i) {
      booleans[i] = ((bit_value >> i) & 1) == 1;
    }
    return Z3_mk_bv_numeral(ctx_, bit_count, &booleans[0]);
  }

  return absl::InternalError("DSLX node " + sym->ToString().value() +
                             " cannot be translated to Z3 node.");
}

Z3_ast DslxTranslator::GetValue(SymbolicType* sym) {
  auto it = translations_.find(sym);
  XLS_CHECK(it != translations_.end())
      << "Node not translated: " << sym->ToString().value();
  return it->second;
}

// Wrapper around the above that verifies we're accessing a Bits value.
Z3_ast DslxTranslator::GetBitVec(SymbolicType* sym) {
  Z3_ast z3_value = GetValue(sym);

  // Since Z3 produces boolean type for comparison, eq, and neq operations, we
  // need to cast the result to BV sort so that it can be (possibly) used in
  // logical and/or operations as they only accept BV sort.
  if (IsAstBoolPredicate(ctx_, z3_value)) z3_value = BoolToBv(ctx_, z3_value);

  Z3_sort value_sort = Z3_get_sort(ctx_, z3_value);
  XLS_CHECK_EQ(Z3_get_sort_kind(ctx_, value_sort), Z3_BV_SORT);

  XLS_CHECK_EQ(sym->bit_count(), Z3_get_bv_sort_size(ctx_, value_sort));
  return z3_value;
}

int64_t DslxTranslator::GetBitVecCount(SymbolicType* sym) {
  Z3_ast value = GetValue(sym);
  Z3_sort value_sort = Z3_get_sort(ctx_, value);
  return Z3_get_bv_sort_size(ctx_, value_sort);
}

template <typename FnT>
absl::Status DslxTranslator::HandleUnary(SymbolicType* sym, FnT f) {
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->tree());
  Z3_ast result = f(ctx_, GetBitVec(nodes.left));
  XLS_RETURN_IF_ERROR(NoteTranslation(sym, result));
  return seh.status();
}

absl::Status DslxTranslator::HandleNeg(SymbolicType* sym) {
  return HandleUnary(sym, Z3_mk_bvneg);
}

absl::Status DslxTranslator::HandleNot(SymbolicType* sym) {
  return HandleUnary(sym, Z3_mk_bvnot);
}

template <typename FnT>
absl::Status DslxTranslator::HandleBinary(SymbolicType* sym, FnT f) {
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->tree());
  Z3_ast result = f(ctx_, GetBitVec(nodes.left), GetBitVec(nodes.right));
  XLS_RETURN_IF_ERROR(NoteTranslation(sym, result));
  return seh.status();
}

absl::Status DslxTranslator::HandleAdd(SymbolicType* sym) {
  return HandleBinary(sym, Z3_mk_bvadd);
}

absl::Status DslxTranslator::HandleSub(SymbolicType* sym) {
  return HandleBinary(sym, Z3_mk_bvsub);
}

absl::Status DslxTranslator::HandleGt(SymbolicType* sym) {
  if (sym->IsSigned()) return HandleBinary(sym, Z3_mk_bvsgt);
  return HandleBinary(sym, Z3_mk_bvugt);
}

absl::Status DslxTranslator::HandleLe(SymbolicType* sym) {
  if (sym->IsSigned()) return HandleBinary(sym, Z3_mk_bvsle);
  return HandleBinary(sym, Z3_mk_bvule);
}

absl::Status DslxTranslator::HandleLt(SymbolicType* sym) {
  if (sym->IsSigned()) return HandleBinary(sym, Z3_mk_bvslt);
  return HandleBinary(sym, Z3_mk_bvult);
}

absl::Status DslxTranslator::HandleGe(SymbolicType* sym) {
  if (sym->IsSigned()) return HandleBinary(sym, Z3_mk_bvsge);
  return HandleBinary(sym, Z3_mk_bvuge);
}

absl::Status DslxTranslator::HandleEqArray(SymbolicType* sym,
                                           bool invert_result) {
  const bool bool_true = true;
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->tree());
  std::vector<SymbolicType*> rhs_children = nodes.right->GetChildren();
  std::vector<SymbolicType*> lhs_children = nodes.left->GetChildren();
  XLS_RET_CHECK(rhs_children.size() == lhs_children.size());
  Z3_ast result = BoolToBv(ctx_, Z3_mk_eq(ctx_, GetBitVec(lhs_children.at(0)),
                                          GetBitVec(rhs_children.at(0))));
  for (int64_t i = 1; i < lhs_children.size(); ++i) {
    result =
        Z3_mk_bvand(ctx_, result,
                    BoolToBv(ctx_, Z3_mk_eq(ctx_, GetBitVec(lhs_children.at(i)),
                                            GetBitVec(rhs_children.at(i)))));
  }
  result = Z3_mk_eq(ctx_, result, Z3_mk_bv_numeral(ctx_, 1, &bool_true));
  if (invert_result) {
    result = Z3_mk_not(ctx_, result);
  }
  XLS_RETURN_IF_ERROR(NoteTranslation(sym, result));
  return seh.status();
}

absl::Status DslxTranslator::HandleEq(SymbolicType* sym) {
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->tree());
  if (nodes.right->IsArray()) {
    return HandleEqArray(sym);
  }
  return HandleBinary(sym, Z3_mk_eq);
}

absl::Status DslxTranslator::HandleNe(SymbolicType* sym) {
  auto f = [](Z3_context ctx, Z3_ast a, Z3_ast b) {
    return Z3_mk_not(ctx, (Z3_mk_eq(ctx, a, b)));
  };
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->tree());
  if (nodes.right->IsArray()) {
    return HandleEqArray(sym, /*invert_result=*/true);
  }
  return HandleBinary(sym, f);
}

template <typename FnT>
absl::Status DslxTranslator::HandleShift(SymbolicType* sym, FnT fshift) {
  auto f = [sym, fshift, this](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    SymbolicType::Nodes nodes = sym->tree().value();
    int64_t lhs_bit_count = GetBitVecCount(nodes.left);
    int64_t rhs_bit_count = GetBitVecCount(nodes.right);
    if (rhs_bit_count < lhs_bit_count) {
      rhs = Z3_mk_zero_ext(ctx, lhs_bit_count - rhs_bit_count, rhs);
    } else if (rhs_bit_count > lhs_bit_count) {
      if (nodes.left->IsSigned()) {
        lhs = Z3_mk_sign_ext(ctx, rhs_bit_count - lhs_bit_count, lhs);
      } else {
        lhs = Z3_mk_zero_ext(ctx, rhs_bit_count - lhs_bit_count, lhs);
      }
    }
    return fshift(ctx, lhs, rhs);
  };
  return HandleBinary(sym, f);
}

absl::Status DslxTranslator::HandleShra(SymbolicType* sym) {
  return HandleShift(sym, Z3_mk_bvashr);
}

absl::Status DslxTranslator::HandleShrl(SymbolicType* sym) {
  return HandleShift(sym, Z3_mk_bvlshr);
}

absl::Status DslxTranslator::HandleShll(SymbolicType* sym) {
  return HandleShift(sym, Z3_mk_bvshl);
}

absl::Status DslxTranslator::HandleAnd(SymbolicType* sym) {
  return HandleBinary(sym, Z3_mk_bvand);
}

absl::Status DslxTranslator::HandleOr(SymbolicType* sym) {
  return HandleBinary(sym, Z3_mk_bvor);
}

absl::Status DslxTranslator::HandleXor(SymbolicType* sym) {
  return HandleBinary(sym, Z3_mk_bvxor);
}

absl::Status DslxTranslator::HandleConcat(SymbolicType* sym) {
  return HandleBinary(sym, Z3_mk_concat);
}

absl::StatusOr<Z3_ast> DslxTranslator::CreateZ3Param(SymbolicType* sym) {
  Z3_sort sort = Z3_mk_bv_sort(ctx_, sym->bit_count());
  return Z3_mk_const(
      ctx_,
      Z3_mk_string_symbol(ctx_, std::string(sym->ToString().value()).c_str()),
      sort);
}

absl::Status DslxTranslator::HandleParam(SymbolicType* sym) {
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(Z3_ast z3_value, CreateZ3Param(sym));
  XLS_RETURN_IF_ERROR(NoteTranslation(sym, z3_value));
  return seh.status();
}

absl::Status DslxTranslator::HandleMulHelper(SymbolicType* sym,
                                             bool is_signed) {
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->tree());
  Z3_ast lhs = GetValue(nodes.left);
  Z3_ast rhs = GetValue(nodes.right);

  // In DSLX, multiply operands have to be of the same widths (no need to check
  // it here as the interpreter will catch this earlier), and the result width
  // will be the same as the operands.
  int result_size = Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, lhs));
  Z3_ast result;
  if (is_signed) {
    result = Z3_mk_bvmul(ctx_, lhs, rhs);
  } else {
    result = solvers::z3::DoUnsignedMul(ctx_, lhs, rhs, result_size);
  }
  XLS_RETURN_IF_ERROR(NoteTranslation(sym, result));
  return absl::OkStatus();
}

absl::Status DslxTranslator::HandleMul(SymbolicType* sym) {
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_RETURN_IF_ERROR(HandleMulHelper(sym, sym->IsSigned()));
  return seh.status();
}

absl::Status DslxTranslator::HandleDiv(SymbolicType* sym) {
  if (sym->IsSigned()) return HandleBinary(sym, Z3_mk_bvsdiv);
  return HandleBinary(sym, Z3_mk_bvudiv);
}

absl::Status DslxTranslator::HandleTernaryIf(SymbolicType* sym) {
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(SymbolicType * ternary, sym->TernaryRoot());
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->tree());

  XLS_RETURN_IF_ERROR(TranslatePredicate(ternary));
  XLS_RETURN_IF_ERROR(TranslatePredicate(nodes.left));
  XLS_RETURN_IF_ERROR(TranslatePredicate(nodes.right));

  absl::optional<Z3_ast> ast_test = GetTranslation(ternary);
  absl::optional<Z3_ast> ast_consequent = GetTranslation(nodes.left);
  absl::optional<Z3_ast> ast_alternate = GetTranslation(nodes.right);

  Z3_ast result = Z3_mk_ite(ctx_, ast_test.value(), ast_consequent.value(),
                            ast_alternate.value());
  XLS_RETURN_IF_ERROR(NoteTranslation(sym, result));
  return seh.status();
}


bool IsAstBoolPredicate(Z3_context ctx, Z3_ast objective) {
  static const Z3_decl_kind predicates[] = {
      Z3_OP_EQ,  Z3_OP_NOT,  Z3_OP_LE,   Z3_OP_LT,   Z3_OP_GE,
      Z3_OP_GT,  Z3_OP_ULEQ, Z3_OP_SLEQ, Z3_OP_UGEQ, Z3_OP_SGEQ,
      Z3_OP_ULT, Z3_OP_SLT,  Z3_OP_UGT,  Z3_OP_SGT};
  Z3_decl_kind objective_kind =
      Z3_get_decl_kind(ctx, Z3_get_app_decl(ctx, Z3_to_app(ctx, objective)));
  for (const auto& p : predicates) {
    if (objective_kind == p) {
      return true;
    }
  }
  return false;
}

Z3_ast BoolToBv(Z3_context ctx, Z3_ast bool_ast) {
  const bool bool_true = true;
  const bool bool_false = false;
  return Z3_mk_ite(ctx, bool_ast, Z3_mk_bv_numeral(ctx, 1, &bool_true),
                   Z3_mk_bv_numeral(ctx, 1, &bool_false));
}
}  // namespace xls::solvers::z3
