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
#include "xls/dslx/concolic_test_generator.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/symbolic_type.h"
#include "xls/solvers/z3_dslx_translator.h"

namespace xls::dslx {

#define DISPATCH_DEF(__expr_type)                                              \
  absl::StatusOr<InterpValue> EvaluateSym##__expr_type(                        \
      __expr_type* expr, InterpBindings* bindings, ConcreteType* type_context, \
      AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {    \
    return Evaluate##__expr_type(expr, bindings, type_context, interp);        \
  }

DISPATCH_DEF(Carry)
DISPATCH_DEF(Cast)
DISPATCH_DEF(ConstRef)
DISPATCH_DEF(For)
DISPATCH_DEF(Let)
DISPATCH_DEF(Match)
DISPATCH_DEF(NameRef)
DISPATCH_DEF(SplatStructInstance)
DISPATCH_DEF(String)
DISPATCH_DEF(While)

#undef DISPATCH_DEF

template <typename... Args>
InterpValue AddSymToValue(InterpValue concrete_value,
                          ConcolicTestGenerator* test_generator,
                          std::unique_ptr<SymbolicType> sym_ptr) {
  InterpValue sym_value = concrete_value.UpdateWithSym(sym_ptr.get());
  test_generator->StoreSymPointers(std::move(sym_ptr));
  return sym_value;
}

absl::StatusOr<InterpValue> EvaluateSymAttr(
    Attr* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  TypeInfo* type_info = interp->GetCurrentTypeInfo();
  XLS_RET_CHECK_EQ(expr->owner(), type_info->module());
  // Resolve the tuple type to figure out what index of the tuple we're
  // grabbing.
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  absl::optional<const ConcreteType*> maybe_type =
      type_info->GetItem(expr->lhs());
  XLS_RET_CHECK(maybe_type.has_value())
      << "LHS of attr: " << expr << " should have type info in: " << type_info
      << " @ " << expr->lhs()->span();
  auto* struct_type = dynamic_cast<const StructType*>(maybe_type.value());

  absl::optional<int64_t> index;
  for (int64_t i = 0; i < struct_type->size(); ++i) {
    absl::string_view name = struct_type->GetMemberName(i);
    if (name == expr->attr()->identifier()) {
      index = i;
      break;
    }
  }
  XLS_RET_CHECK(index.has_value())
      << "Unable to find attribute " << expr->attr()
      << ": should be caught by type inference";
  InterpValue result = lhs.GetValuesOrDie().at(*index);
  if (result.sym() == nullptr) {
    return result.UpdateWithSym(lhs.sym()->GetChildren().at(*index));
  }
  return result;
}

absl::StatusOr<InterpValue> EvaluateSymIndex(
    Index* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  if (lhs.IsBits()) {
    return EvaluateIndex(expr, bindings, type_context, interp);
  }
  Expr* index = absl::get<Expr*>(expr->rhs());
  // Note: since we permit a type-unannotated literal number we provide a type
  // context here.
  XLS_ASSIGN_OR_RETURN(InterpValue index_value,
                       interp->Eval(index, bindings, BitsType::MakeU64()));
  XLS_ASSIGN_OR_RETURN(uint64_t index_int, index_value.GetBitValueUint64());
  int64_t length = lhs.GetLength().value();
  if (index_int >= length) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FailureError: %s Indexing out of bounds: %d vs size %d",
        expr->span().ToString(), index_int, length));
  }
  InterpValue result = lhs.GetValuesOrDie().at(index_int);
  if (result.sym() == nullptr) {
    return result.UpdateWithSym(lhs.sym()->GetChildren().at(index_int));
  }
  return result;
}

absl::StatusOr<InterpValue> EvaluateSymStructInstance(
    StructInstance* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(
      InterpValue struct_value,
      EvaluateStructInstance(expr, bindings, type_context, interp));
  XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                       ToTypeDefinition(ToAstNode(expr->struct_def())));
  XLS_ASSIGN_OR_RETURN(DerefVariant deref,
                       EvaluateToStructOrEnumOrAnnotation(
                           type_definition, bindings, interp, nullptr));
  if (!absl::holds_alternative<StructDef*>(deref)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Type definition did not dereference to a struct, found: ",
                     ToAstNode(deref)->GetNodeTypeName()));
  }

  StructDef* struct_def = absl::get<StructDef*>(deref);
  std::vector<std::string> struct_members = struct_def->GetMemberNames();
  struct_members.push_back(StructRefToText(expr->struct_def()));

  std::vector<SymbolicType*> elements;
  elements.reserve(struct_value.GetValuesOrDie().size());
  for (const InterpValue& value : struct_value.GetValuesOrDie()) {
    elements.push_back(value.sym());
  }
  auto sym_ptr =
      std::make_unique<SymbolicType>(SymbolicType::MakeArray(elements));
  InterpValue symbolic_arg =
      AddSymToValue(struct_value, test_generator, std::move(sym_ptr));
  return symbolic_arg.UpdateWithStructInfo(struct_members);
}

absl::StatusOr<InterpValue> EvaluateSymArray(
    Array* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       EvaluateArray(expr, bindings, type_context, interp));
  std::vector<SymbolicType*> elements;
  elements.reserve(result.GetValuesOrDie().size());
  for (const InterpValue& value : result.GetValuesOrDie()) {
    XLS_RET_CHECK(value.sym() != nullptr);
    elements.push_back(value.sym());
  }
  auto sym_ptr =
      std::make_unique<SymbolicType>(SymbolicType::MakeArray(elements));
  return AddSymToValue(result, test_generator, std::move(sym_ptr));
}

absl::StatusOr<InterpValue> EvaluateSymXlsTuple(
    XlsTuple* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       EvaluateXlsTuple(expr, bindings, type_context, interp));
  std::vector<SymbolicType*> elements;
  elements.reserve(result.GetValuesOrDie().size());
  for (const InterpValue& value : result.GetValuesOrDie()) {
    XLS_RET_CHECK(value.sym() != nullptr);
    elements.push_back(value.sym());
  }
  auto sym_ptr =
      std::make_unique<SymbolicType>(SymbolicType::MakeArray(elements));
  return AddSymToValue(result, test_generator, std::move(sym_ptr));
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
  return AddSymToValue(result, test_generator, std::move(sym_ptr));
}

absl::StatusOr<InterpValue> EvaluateSymColonRef(
    ColonRef* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       EvaluateColonRef(expr, bindings, type_context, interp));
  if (result.tag() == InterpValueTag::kEnum) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_value, result.GetBitsOrDie().ToInt64());
    auto sym_ptr =
        std::make_unique<SymbolicType>(SymbolicType::MakeLiteral(ConcreteInfo{
            result.IsSigned(), result.GetBitCount().value(), bit_value}));
    return AddSymToValue(result, test_generator, std::move(sym_ptr));
  }

  return result;
}

// Recursively marks all the SymbolicType members of arrays/structs/tuples as
// function parameter by modifying the tag.
std::vector<SymbolicType*> MarkMembersAsParam(
    InterpValue arg, std::string id, ConcolicTestGenerator* test_generator,
    int64_t& id_counter) {
  std::vector<SymbolicType*> elements;
  for (const InterpValue& value : arg.GetValuesOrDie()) {
    if (value.IsBits()) {
      if (value.sym() != nullptr) {
        value.sym()->MarkAsFnParam(
            absl::StrCat(id, std::to_string(id_counter++)));
        elements.push_back(value.sym());
      } else {
        auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeParam(
            ConcreteInfo{value.IsSigned(), value.GetBitsOrDie().bit_count(),
                         /*bit_value=*/0,
                         absl::StrCat(id, std::to_string(id_counter++))}));
        elements.push_back(sym_ptr.get());
        test_generator->StoreSymPointers(std::move(sym_ptr));
      }
    } else {
      for (SymbolicType* sym :
           MarkMembersAsParam(value, id, test_generator, id_counter)) {
        if (sym != nullptr) {
          elements.push_back(sym);
        }
      }
    }
  }
  return elements;
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
    if (f->identifier() == m->name()) {
      switch (args[i].tag()) {
        case InterpValueTag::kSBits:
        case InterpValueTag::kUBits:
        case InterpValueTag::kEnum: {
          auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeParam(
              ConcreteInfo{args[i].IsSigned(), args[i].GetBitCount().value(),
                           /*bit_value=*/0, f->params()[i]->identifier()}));

          InterpValue symbolic_arg =
              AddSymToValue(args[i], test_generator, std::move(sym_ptr));
          fn_bindings.AddValue(f->params()[i]->identifier(), symbolic_arg);
          test_generator->AddFnParam(symbolic_arg);
          break;
        }
        case InterpValueTag::kArray:
        case InterpValueTag::kTuple: {
          int64_t element_counter = 0;
          std::vector<SymbolicType*> elements =
              MarkMembersAsParam(args[i], f->params()[i]->identifier(),
                                 test_generator, element_counter);
          auto sym_ptr =
              std::make_unique<SymbolicType>(SymbolicType::MakeArray(elements));

          InterpValue symbolic_arg =
              AddSymToValue(args[i], test_generator, std::move(sym_ptr));

          fn_bindings.AddValue(f->params()[i]->identifier(), symbolic_arg);
          test_generator->AddFnParam(symbolic_arg);
          break;
        }
        default:
          return absl::InternalError("function parameter of type " +
                                     TagToString(args[i].tag()) +
                                     " not supported");
      }

    } else {
      fn_bindings.AddValue(f->params()[i]->identifier(), args[i]);
    }
  }

  return interp->Eval(f->body(), &fn_bindings);
}

absl::StatusOr<InterpValue> EvaluateSymUnop(
    Unop* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue arg,
                       interp->Eval(expr->operand(), bindings));
  XLS_RET_CHECK(arg.sym() != nullptr);

  switch (expr->unop_kind()) {
    case UnopKind::kInvert: {
      XLS_ASSIGN_OR_RETURN(InterpValue result, arg.BitwiseNegate());
      auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeUnary(
          SymbolicType::Nodes{expr->unop_kind(), arg.sym()},
          ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
      return AddSymToValue(result, test_generator, std::move(sym_ptr));
    }
    case UnopKind::kNegate: {
      XLS_ASSIGN_OR_RETURN(InterpValue result, arg.ArithmeticNegate());
      auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeUnary(
          SymbolicType::Nodes{expr->unop_kind(), arg.sym()},
          ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
      return AddSymToValue(result, test_generator, std::move(sym_ptr));
    }
  }
  return absl::InternalError(absl::StrCat("Invalid unary operation kind: ",
                                          static_cast<int64_t>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateSymShift(
    Binop* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
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
      return AddSymToValue(result, test_generator, std::move(sym_ptr));
    }
    case BinopKind::kShr: {
      if (lhs.IsSigned()) {
        XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.Shra(rhs));
        auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
            SymbolicType::Nodes{binop, lhs.sym(), rhs.sym()},
            ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
        return AddSymToValue(result, test_generator, std::move(sym_ptr));
      }
      XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.Shrl(rhs));
      auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
          SymbolicType::Nodes{binop, lhs.sym(), rhs.sym()},
          ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
      return AddSymToValue(result, test_generator, std::move(sym_ptr));
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
    return EvaluateSymShift(expr, bindings, type_context, interp,
                            test_generator);
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
  if (lhs.sym()->IsArray()) {
    auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
        SymbolicType::Nodes{expr->binop_kind(), lhs.sym(), rhs.sym()},
        ConcreteInfo{}));

    return AddSymToValue(result, test_generator, std::move(sym_ptr));
  }
  // Comparators always return a bool (unsigned type) but we need to keep track
  // of signed/unsigned for Z3 translation hence we use the lhs' sign as the
  // operation's sign instead of result's.
  auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeBinary(
      SymbolicType::Nodes{expr->binop_kind(), lhs.sym(), rhs.sym()},
      ConcreteInfo{result.IsSigned(), result.GetBitCount().value()}));
  return AddSymToValue(result, test_generator, std::move(sym_ptr));
}

}  // namespace xls::dslx
