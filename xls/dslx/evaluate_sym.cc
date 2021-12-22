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
#include "xls/dslx/symbolic_type.h"
#include "xls/solvers/z3_dslx_translator.h"

namespace xls::dslx {

#define DISPATCH_DEF(__expr_type)                                              \
  absl::StatusOr<InterpValue> EvaluateSym##__expr_type(                        \
      __expr_type* expr, InterpBindings* bindings, ConcreteType* type_context, \
      AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {    \
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
                          std::unique_ptr<SymbolicType> sym_ptr) {
  InterpValue sym_value = concrete_value.UpdateWithSym(sym_ptr.get());
  bindings->AddSymValues(std::move(sym_ptr));
  return sym_value;
}

absl::StatusOr<InterpValue> EvaluateSymTernary(
    Ternary* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue test, interp->Eval(expr->test(), bindings));
  XLS_RET_CHECK(test.IsBits() && test.GetBitCount().value() == 1);
  if (test.IsTrue()) {
    XLS_RETURN_IF_ERROR(
        test_generator->SolvePredicate(test.sym(), /*negate_predicate=*/true));
    return interp->Eval(expr->consequent(), bindings);
  }
  XLS_RETURN_IF_ERROR(
      test_generator->SolvePredicate(test.sym(), /*negate_predicate=*/false));
  return interp->Eval(expr->alternate(), bindings);
}

absl::StatusOr<InterpValue> EvaluateSymNumber(
    Number* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       EvaluateNumber(expr, bindings, type_context, interp));
  XLS_ASSIGN_OR_RETURN(Bits result_bits, result.GetBits());
  XLS_ASSIGN_OR_RETURN(int64_t bit_value, result_bits.ToInt64());

  auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeLiteral(
      ConcreteInfo{result.IsSigned(), result_bits.bit_count(), bit_value}));
  return AddSymToValue(result, bindings, std::move(sym_ptr));
}

absl::StatusOr<InterpValue> EvaluateSymFunction(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    const SymbolicBindings& symbolic_bindings, AbstractInterpreter* interp,
    ConcolicTestGenerator* test_generator) {
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
    if (args[i].IsBits()) {
      auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeParam(
          ConcreteInfo{args[i].IsSigned(), args[i].GetBitCount().value(),
                       /*bit_value=*/0, f->params()[i]->identifier()}));
      InterpValue symbolic_arg =
          AddSymToValue(args[i], &fn_bindings, std::move(sym_ptr));

      fn_bindings.AddValue(f->params()[i]->identifier(), symbolic_arg);
      test_generator->AddFnParam(symbolic_arg);
    } else {
      return absl::InternalError("function parameter of type " +
                                 TagToString(args[i].tag()) + " not supported");
    }
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
      auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
          SymbolicType::Nodes{binop, lhs.sym(), rhs.sym()},
          ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
      return AddSymToValue(result, bindings, std::move(sym_ptr));
    }
    case BinopKind::kShr: {
      if (lhs.IsSigned()) {
        XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.Shra(rhs));
        auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
            SymbolicType::Nodes{binop, lhs.sym(), rhs.sym()},
            ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
        return AddSymToValue(result, bindings, std::move(sym_ptr));
      }
      XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.Shrl(rhs));
      auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
          SymbolicType::Nodes{binop, lhs.sym(), rhs.sym()},
          ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
      return AddSymToValue(result, bindings, std::move(sym_ptr));
    }
    default:
      // Not an exhaustive list: this function only handles the shift operators.
      break;
  }
  return absl::InternalError(absl::StrCat("Invalid shift operation kind: ",
                                          static_cast<int64_t>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateSymBinop(
    Binop* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  if (GetBinopShifts().contains(expr->binop_kind())) {
    return EvaluateSymShift(expr, bindings, type_context, interp);
  }

  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       EvaluateBinop(expr, bindings, type_context, interp));
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, interp->Eval(expr->rhs(), bindings));

  if (lhs.sym() == nullptr || rhs.sym() == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Node %s cannot be evaluated symbolically",
        lhs.sym() == nullptr ? lhs.ToString() : rhs.ToString()));
  }
  // Comparators always return a bool (unsigned type) but we need to keep track
  // of signed/unsigned for Z3 translation hence we use the lhs' sign as the
  // operation's sign instead of result's.
  auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
      SymbolicType::Nodes{expr->binop_kind(), lhs.sym(), rhs.sym()},
      ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
  return AddSymToValue(result, bindings, std::move(sym_ptr));
}

}  // namespace xls::dslx
