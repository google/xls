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

#include "xls/dslx/evaluate_sym.h"

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/evaluate.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/symbolic_type.h"
#include "xls/solvers/z3_dslx_translator.h"

namespace xls::dslx {

#define DISPATCH_DEF(__expr_type)                                              \
  absl::StatusOr<InterpValue> EvaluateSym##__expr_type(                        \
      __expr_type* expr, InterpBindings* bindings, ConcreteType* type_context, \
      AbstractInterpreter* interp) {                                           \
    return Evaluate##__expr_type(expr, bindings, type_context, interp);        \
  }

DISPATCH_DEF(Array)
DISPATCH_DEF(Attr)
DISPATCH_DEF(Carry)
DISPATCH_DEF(Cast)
DISPATCH_DEF(ColonRef)
DISPATCH_DEF(ConstRef)
DISPATCH_DEF(For)
DISPATCH_DEF(Index)
DISPATCH_DEF(Let)
DISPATCH_DEF(Match)
DISPATCH_DEF(NameRef)
DISPATCH_DEF(SplatStructInstance)
DISPATCH_DEF(String)
DISPATCH_DEF(StructInstance)
DISPATCH_DEF(Unop)
DISPATCH_DEF(While)
DISPATCH_DEF(XlsTuple)

#undef DISPATCH_DEF

template <typename... Args>
InterpValue AddSymToValue(InterpValue concrete_value, InterpBindings* bindings,
                          Args&&... args) {
  auto sym_ptr =
      std::make_unique<SymbolicType>(SymbolicType(std::forward<Args>(args)...));
  InterpValue sym_value = concrete_value.UpdateWithSym(sym_ptr.get());
  bindings->AddSymValues(std::move(sym_ptr));
  return sym_value;
}

absl::StatusOr<InterpValue> EvaluateSymTernary(Ternary* expr,
                                               InterpBindings* bindings,
                                               ConcreteType* type_context,
                                               AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue test, interp->Eval(expr->test(), bindings));

  XLS_RET_CHECK(test.IsBits() && test.GetBitCount().value() == 1);
  if (test.IsTrue()) {
    XLS_RETURN_IF_ERROR(solvers::z3::TryProve(
        test.sym(), /*negate_predicate=*/true, absl::InfiniteDuration()));
    return interp->Eval(expr->consequent(), bindings);
  }
  XLS_RETURN_IF_ERROR(solvers::z3::TryProve(
      test.sym(), /*negate_predicate=*/false, absl::InfiniteDuration()));
  return interp->Eval(expr->alternate(), bindings);
}

absl::StatusOr<InterpValue> EvaluateSymNumber(Number* expr,
                                              InterpBindings* bindings,
                                              ConcreteType* type_context,
                                              AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       EvaluateNumber(expr, bindings, type_context, interp));
  XLS_ASSIGN_OR_RETURN(Bits result_bits, result.GetBits());
  return AddSymToValue(result, bindings, result_bits,
                       result.GetBitCount().value(), result.IsSigned());
}

absl::StatusOr<InterpValue> EvaluateSymFunction(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    const SymbolicBindings& symbolic_bindings, AbstractInterpreter* interp) {
  XLS_RET_CHECK_EQ(f->owner(), interp->GetCurrentTypeInfo()->module());
  XLS_VLOG(5) << "Evaluating function: " << f->identifier()
              << " symbolic_bindings: " << symbolic_bindings;
  if (args.size() != f->params().size()) {
    return absl::InternalError(
        absl::StrFormat("EvaluateError: %s Argument arity mismatch for "
                        "invocation; want %d got %d",
                        span.ToString(), f->params().size(), args.size()));
  }

  Module* m = f->owner();
  XLS_ASSIGN_OR_RETURN(const InterpBindings* top_level_bindings,
                       GetOrCreateTopLevelBindings(m, interp));
  XLS_VLOG(5) << "Evaluated top level bindings for module: " << m->name()
              << "; keys: {"
              << absl::StrJoin(top_level_bindings->GetKeys(), ", ") << "}";
  InterpBindings fn_bindings(/*parent=*/top_level_bindings);
  XLS_RETURN_IF_ERROR(EvaluateDerivedParametrics(f, &fn_bindings, interp,
                                                 symbolic_bindings.ToMap()));

  fn_bindings.set_fn_ctx(FnCtx{m->name(), f->identifier(), symbolic_bindings});
  for (int64_t i = 0; i < f->params().size(); ++i) {
    if (args[i].IsBits())
      fn_bindings.AddValue(
          f->params()[i]->identifier(),
          AddSymToValue(args[i], &fn_bindings, f->params()[i],
                        args[i].GetBitCount().value(), args[i].IsSigned()));
    else
      return absl::InternalError("function parameter of type " +
                                 TagToString(args[i].tag()) + " not supported");
  }

  return interp->Eval(f->body(), &fn_bindings);
}

absl::StatusOr<InterpValue> EvaluateSymShift(Binop* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             AbstractInterpreter* interp) {
  XLS_VLOG(6) << "EvaluateShift: " << expr->ToString() << " @ " << expr->span();
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));

  // Optionally Retrieve a type context for the right hand side as an
  // un-type-annotated literal number is permitted.
  std::unique_ptr<ConcreteType> rhs_type = nullptr;
  absl::optional<ConcreteType*> rhs_item =
      interp->GetCurrentTypeInfo()->GetItem(expr->rhs());
  if (rhs_item.has_value()) {
    rhs_type = rhs_item.value()->CloneToUnique();
  }
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, interp->Eval(expr->rhs(), bindings,
                                                     std::move(rhs_type)));

  BinopKind binop = expr->binop_kind();
  switch (binop) {
    case BinopKind::kShl: {
      XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.Shl(rhs));
      return AddSymToValue(result, bindings,
                           SymbolicType::Nodes{lhs.sym(), rhs.sym()}, binop,
                           result.GetBitCount().value(), result.IsSigned());
    }
    case BinopKind::kShr: {
      if (lhs.IsSigned()) {
        XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.Shra(rhs));
        return AddSymToValue(result, bindings,
                             SymbolicType::Nodes{lhs.sym(), rhs.sym()}, binop,
                             result.GetBitCount().value(), result.IsSigned());
      }
      XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.Shrl(rhs));
      return AddSymToValue(result, bindings,
                           SymbolicType::Nodes{lhs.sym(), rhs.sym()}, binop,
                           result.GetBitCount().value(), result.IsSigned());
    }
    default:
      // Not an exhaustive list: this function only handles the shift operators.
      break;
  }
  return absl::InternalError(absl::StrCat("Invalid shift operation kind: ",
                                          static_cast<int64_t>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateSymBinop(Binop* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             AbstractInterpreter* interp) {
  if (GetBinopShifts().contains(expr->binop_kind())) {
    return EvaluateSymShift(expr, bindings, type_context, interp);
  }

  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       EvaluateBinop(expr, bindings, type_context, interp));
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, interp->Eval(expr->rhs(), bindings));

  if (lhs.sym() == nullptr || rhs.sym() == nullptr)
    return absl::InternalError(
        absl::StrFormat("Node %s cannot be evaluated symbolically",
                        lhs.sym() == nullptr ? TagToString(lhs.tag())
                                             : TagToString(rhs.tag())));

  return AddSymToValue(
      result, bindings, SymbolicType::Nodes{lhs.sym(), rhs.sym()},
      expr->binop_kind(), result.GetBitCount().value(), result.IsSigned());
}

}  // namespace xls::dslx
