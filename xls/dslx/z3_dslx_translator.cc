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

#include "xls/dslx/z3_dslx_translator.h"

#include <cstdint>

#include "absl/status/status.h"
#include "xls/dslx/symbolic_type.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3_api.h"

namespace xls::dslx {

absl::Status VisitSymbolicTree(DslxTranslator* translator, SymbolicType* sym) {
  auto Walk = [&](SymbolicType* x) -> absl::Status {
    if (x->IsLeaf())
      XLS_RETURN_IF_ERROR(ProcessSymbolicLeaf(translator, x));
    else
      XLS_RETURN_IF_ERROR(ProcessSymbolicNode(translator, x));
    return absl::OkStatus();
  };
  XLS_RETURN_IF_ERROR(sym->DoPostorder(Walk));
  return absl::OkStatus();
}

absl::Status ProcessSymbolicNode(DslxTranslator* translator,
                                 SymbolicType* sym) {
  switch (sym->op()) {
    case BinopKind::kAdd:
      XLS_RETURN_IF_ERROR(translator->HandleAdd(sym));
      break;
    case BinopKind::kLogicalAnd:
    case BinopKind::kAnd:
      XLS_RETURN_IF_ERROR(translator->HandleAnd(sym));
      break;
    case BinopKind::kEq:
      XLS_RETURN_IF_ERROR(translator->HandleEq(sym));
      break;
    case BinopKind::kNe:
      XLS_RETURN_IF_ERROR(translator->HandleNe(sym));
      break;
    case BinopKind::kLogicalOr:
    case BinopKind::kOr:
      XLS_RETURN_IF_ERROR(translator->HandleOr(sym));
      break;
    case BinopKind::kGt:
      XLS_RETURN_IF_ERROR(translator->HandleGt(sym));
      break;
    case BinopKind::kGe:
      XLS_RETURN_IF_ERROR(translator->HandleGe(sym));
      break;
    case BinopKind::kShl:
      XLS_RETURN_IF_ERROR(translator->HandleShll(sym));
      break;
    case BinopKind::kShr:
      if (sym->IsSigned())
        XLS_RETURN_IF_ERROR(translator->HandleShra(sym));
      else
        XLS_RETURN_IF_ERROR(translator->HandleShrl(sym));
      break;
    case BinopKind::kLe:
      XLS_RETURN_IF_ERROR(translator->HandleLe(sym));
      break;
    case BinopKind::kLt:
      XLS_RETURN_IF_ERROR(translator->HandleLt(sym));
      break;
    case BinopKind::kMul:
      XLS_RETURN_IF_ERROR(translator->HandleMul(sym));
      break;
    case BinopKind::kSub:
      XLS_RETURN_IF_ERROR(translator->HandleSub(sym));
      break;
    case BinopKind::kXor:
      XLS_RETURN_IF_ERROR(translator->HandleXor(sym));
      break;
    case BinopKind::kConcat:
      XLS_RETURN_IF_ERROR(translator->HandleConcat(sym));
      break;
    case BinopKind::kDiv:
      XLS_RETURN_IF_ERROR(translator->HandleDiv(sym));
      break;
    default:
      return absl::InternalError("Invalid binary operation kind " +
                                 BinopKindToString(sym->op()));
  }
  return absl::OkStatus();
}

absl::Status ProcessSymbolicLeaf(DslxTranslator* translator,
                                 SymbolicType* sym) {
  if (sym->IsBits()) {
    XLS_RETURN_IF_ERROR(translator->HandleLiteral(sym));
  } else {  // node is a function parameter.
    XLS_RETURN_IF_ERROR(translator->HandleParam(sym));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DslxTranslator>>
DslxTranslator::CreateAndTranslate(SymbolicType* predicate) {
  Z3_config config = Z3_mk_config();
  Z3_set_param_value(config, "proof", "true");
  auto translator = absl::WrapUnique(new DslxTranslator(config));
  XLS_RETURN_IF_ERROR(VisitSymbolicTree(translator.get(), predicate));
  return translator;
}

Z3_ast DslxTranslator::GetTranslation(const SymbolicType* source) {
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
    XLS_ASSIGN_OR_RETURN(Bits bits, sym->GetBits());
    std::unique_ptr<bool[]> booleans(new bool[bits.bit_count()]);
    for (int64_t i = 0; i < bits.bit_count(); ++i) {
      booleans[i] = bits.Get(i);
    }
    return Z3_mk_bv_numeral(ctx_, bits.bit_count(), &booleans[0]);
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
  Z3_sort value_sort = Z3_get_sort(ctx_, z3_value);
  XLS_CHECK_EQ(Z3_get_sort_kind(ctx_, value_sort), Z3_BV_SORT);

  XLS_CHECK_EQ(sym->GetBitCount(), Z3_get_bv_sort_size(ctx_, value_sort));
  return z3_value;
}

int64_t DslxTranslator::GetBitVecCount(SymbolicType* sym) {
  Z3_ast value = GetValue(sym);
  Z3_sort value_sort = Z3_get_sort(ctx_, value);
  return Z3_get_bv_sort_size(ctx_, value_sort);
}

template <typename FnT>
absl::Status DslxTranslator::HandleBinary(SymbolicType* sym, FnT f) {
  solvers::z3::ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->nodes());
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

absl::Status DslxTranslator::HandleEq(SymbolicType* sym) {
  return HandleBinary(sym, Z3_mk_eq);
}

absl::Status DslxTranslator::HandleNe(SymbolicType* sym) {
  auto f = [](Z3_context ctx, Z3_ast a, Z3_ast b) {
    return Z3_mk_not(ctx, (Z3_mk_eq(ctx, a, b)));
  };
  return HandleBinary(sym, f);
}

template <typename FnT>
absl::Status DslxTranslator::HandleShift(SymbolicType* sym, FnT fshift) {
  auto f = [sym, fshift, this](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    SymbolicType::Nodes nodes = sym->nodes().value();
    int64_t lhs_bit_count = GetBitVecCount(nodes.left);
    int64_t rhs_bit_count = GetBitVecCount(nodes.right);
    if (rhs_bit_count < lhs_bit_count) {
      rhs = Z3_mk_zero_ext(ctx, lhs_bit_count - rhs_bit_count, rhs);
    } else if (rhs_bit_count > lhs_bit_count) {
      if (nodes.left->IsSigned())
        lhs = Z3_mk_sign_ext(ctx, rhs_bit_count - lhs_bit_count, lhs);
      else
        lhs = Z3_mk_zero_ext(ctx, rhs_bit_count - lhs_bit_count, lhs);
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
  Z3_sort sort = Z3_mk_bv_sort(ctx_, sym->GetBitCount());
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
  XLS_ASSIGN_OR_RETURN(SymbolicType::Nodes nodes, sym->nodes());
  Z3_ast lhs = GetValue(nodes.left);
  Z3_ast rhs = GetValue(nodes.right);

  // In DSLX, multiply operands have to be of the same widths (no need to check
  // it here as the interpreter will catch this earlier), and the result width
  // will be the same as the operands.
  int result_size = Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, lhs));
  Z3_ast result;
  if (is_signed)
    result = Z3_mk_bvmul(ctx_, lhs, rhs);
  else
    result = solvers::z3::DoUnsignedMul(ctx_, lhs, rhs, result_size);
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

absl::Status TryProve(SymbolicType* predicate, bool negate_predicate,
                      absl::Duration timeout) {
  XLS_ASSIGN_OR_RETURN(auto translator,
                       DslxTranslator::CreateAndTranslate(predicate));
  translator->SetTimeout(timeout);
  Z3_ast objective = translator->GetTranslation(predicate);
  Z3_context ctx = translator->ctx();

  if (negate_predicate) objective = Z3_mk_not(ctx, objective);

  XLS_VLOG(2) << "objective:\n" << Z3_ast_to_string(ctx, objective);

  Z3_solver solver = solvers::z3::CreateSolver(ctx, 1);
  Z3_solver_assert(ctx, solver, objective);
  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);

  XLS_VLOG(2) << solvers::z3::SolverResultToString(ctx, solver, satisfiable)
              << std::endl;

  Z3_solver_dec_ref(ctx, solver);

  return absl::OkStatus();
}

}  // namespace xls::dslx
