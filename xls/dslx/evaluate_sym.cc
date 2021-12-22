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
DISPATCH_DEF(Let)
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

  // Every constraint in the "consequent" should be conjuncted with the "test"
  // constraint; similarly, every constraint in the "alternate" should be
  // conjuncted with the negation of "test".
  test_generator->AddConstraintToPath(test.sym(), /*negate=*/false);
  XLS_ASSIGN_OR_RETURN(InterpValue value_consequent,
                       interp->Eval(expr->consequent(), bindings));
  test_generator->PopConstraintFromPath();
  test_generator->AddConstraintToPath(test.sym(), /*negate=*/true);
  XLS_ASSIGN_OR_RETURN(InterpValue value_alternate,
                       interp->Eval(expr->alternate(), bindings));
  test_generator->PopConstraintFromPath();

  auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeTernary(
      SymbolicType::Nodes{test.sym(), value_consequent.sym(),
                          value_alternate.sym()},
      ConcreteInfo{value_consequent.IsSigned(),
                   value_consequent.GetBitsOrDie().bit_count()}));

  if (test.IsTrue()) {
    XLS_RETURN_IF_ERROR(
        test_generator->SolvePredicate(test.sym(), /*negate_predicate=*/true));
    return AddSymToValue(value_consequent, test_generator, std::move(sym_ptr));
  }
  XLS_RETURN_IF_ERROR(
      test_generator->SolvePredicate(test.sym(), /*negate_predicate=*/false));
  return AddSymToValue(value_alternate, test_generator, std::move(sym_ptr));
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

// Creates a symbolic predicate in the form of
// SymbolicTree(target pattern) = SymbolicTree(to_matched value).
static SymbolicType* CreateMatchPredicate(
    InterpValue target, InterpValue to_match,
    ConcolicTestGenerator* test_generator) {
  auto symbolic_eq =
      std::make_unique<SymbolicType>(SymbolicType::CreateLogicalOp(
          to_match.sym(), target.sym(), BinopKind::kEq));
  SymbolicType* sym_ptr = symbolic_eq.get();
  test_generator->StoreSymPointers(std::move(symbolic_eq));
  return sym_ptr;
}

// Given the predicates for the patterns P_i, chains the predicates as
// either ~(P_1 || P_2 || ...) or ~(P_1 && P_2 && ...).
static SymbolicType* ChainPredicates(std::vector<SymbolicType*> predicates,
                                     ConcolicTestGenerator* test_generator,
                                     BinopKind op, bool negate = false) {
  std::vector<SymbolicType*> conjunctions = {predicates.at(0)};
  for (int64_t i = 1; i < predicates.size(); ++i) {
    auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::CreateLogicalOp(
        conjunctions.back(), predicates.at(i), op));
    conjunctions.push_back(sym_ptr.get());
    test_generator->StoreSymPointers(std::move(sym_ptr));
  }
  if (negate) {
    auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeUnary(
        SymbolicType::Nodes{UnopKind::kInvert, conjunctions.back()},
        ConcreteInfo{/*is_signed=*/false, /*bit_count=*/1}));
    conjunctions.push_back(sym_ptr.get());
    test_generator->StoreSymPointers(std::move(sym_ptr));
  }
  return conjunctions.back();
}

// Match evaluator returns a pair denoting whether the pattern was matched along
// with the constraint for that pattern.
struct MatcherResults {
  SymbolicType* constraint;
  bool matched;
};

// Returns whether this matcher pattern is accepted and creates the constraint
// for the pattern.
//
// If the pattern is a wild card or name definition, the constraint is
//
//                 ~(P1 || P2 || ...)
//
// where P1, P2, ... are constraints before the wild card.
//
// Otherwise the constraint is P: to_match = target.
// Notice that if pattern is a tuple, the constraint is a chain in the format:
//
//     ((to_match_1 = target_1) && (to_match_2 = target_2) && ...).
//
// Args:
//  pattern: Decribes the pattern attempting to match against the value.
//  to_match: The value being matched against.
//  bindings: The bindings to populate if the pattern has bindings associated
//    with it.
//  pattern_constraint: List of constraints seen so far.
static absl::StatusOr<MatcherResults> EvaluateSymMatcher(
    NameDefTree* pattern, const InterpValue& to_match, InterpBindings* bindings,
    std::vector<SymbolicType*>& pattern_constraint, AbstractInterpreter* interp,
    ConcolicTestGenerator* test_generator) {
  if (pattern->is_leaf()) {
    NameDefTree::Leaf leaf = pattern->leaf();
    if (absl::holds_alternative<WildcardPattern*>(leaf)) {
      SymbolicType* predicate =
          ChainPredicates(pattern_constraint, test_generator,
                          BinopKind::kLogicalOr, /*negate=*/true);
      return MatcherResults{predicate, true};
    }
    if (absl::holds_alternative<NameDef*>(leaf)) {
      bindings->AddValue(absl::get<NameDef*>(leaf)->identifier(), to_match);
      SymbolicType* predicate =
          ChainPredicates(pattern_constraint, test_generator,
                          BinopKind::kLogicalOr, /*negate=*/true);
      return MatcherResults{predicate, true};
    }
    if (absl::holds_alternative<Number*>(leaf) ||
        absl::holds_alternative<ColonRef*>(leaf)) {
      XLS_ASSIGN_OR_RETURN(InterpValue target,
                           interp->Eval(ToExprNode(leaf), bindings));
      SymbolicType* predicate =
          CreateMatchPredicate(target, to_match, test_generator);
      pattern_constraint.push_back(predicate);
      return MatcherResults{predicate, target.Eq(to_match)};
    }
    XLS_RET_CHECK(absl::holds_alternative<NameRef*>(leaf));
    XLS_ASSIGN_OR_RETURN(InterpValue target,
                         interp->Eval(absl::get<NameRef*>(leaf), bindings));
    SymbolicType* predicate =
        CreateMatchPredicate(target, to_match, test_generator);
    pattern_constraint.push_back(predicate);
    return MatcherResults{predicate, target.Eq(to_match)};
  }
  // Pattern is a tuple.
  XLS_RET_CHECK_EQ(to_match.GetLength().value(), pattern->nodes().size());
  std::vector<SymbolicType*> tuple_constraints;
  bool matched = true;
  for (int64_t i = 0; i < pattern->nodes().size(); ++i) {
    NameDefTree* subtree = pattern->nodes()[i];
    const InterpValue& member = to_match.GetValuesOrDie().at(i);
    XLS_ASSIGN_OR_RETURN(
        MatcherResults result,
        EvaluateSymMatcher(subtree, member, bindings, pattern_constraint,
                           interp, test_generator));
    // If there is a wild card inside the tuple, there is no need to add it to
    // the set of constraints for that tuple.
    if (!absl::holds_alternative<WildcardPattern*>(subtree->leaf()) &&
        !absl::holds_alternative<NameDef*>(subtree->leaf())) {
      tuple_constraints.push_back(result.constraint);
    }
    if (!result.matched) {
      matched = false;
    }
  }
  // Creates a chain of constraint based on the values inside the tuple.
  SymbolicType* constraint =
      ChainPredicates(tuple_constraints, test_generator, BinopKind::kLogicalAnd,
                      /*negate=*/false);
  if (pattern_constraint.empty()) {
    pattern_constraint.push_back(constraint);
  } else {
    std::vector<SymbolicType*> chain_constraints = {pattern_constraint.back(),
                                                    constraint};
    pattern_constraint = {ChainPredicates(chain_constraints, test_generator,
                                          BinopKind::kLogicalOr,
                                          /*negate=*/false)};
  }
  return MatcherResults{constraint, matched};
}

// For each pattern in the match expression, solves the constraint corresponding
// to that pattern.
absl::StatusOr<InterpValue> EvaluateSymMatch(
    Match* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  bool found_match = false;
  Expr* matched_expr = nullptr;
  SymbolicType* matched_constraint;
  std::vector<SymbolicType*> pattern_constraint;

  XLS_ASSIGN_OR_RETURN(InterpValue to_match,
                       interp->Eval(expr->matched(), bindings));
  for (MatchArm* arm : expr->arms()) {
    for (NameDefTree* pattern : arm->patterns()) {
      XLS_ASSIGN_OR_RETURN(
          MatcherResults matcher_result,
          EvaluateSymMatcher(pattern, to_match, bindings, pattern_constraint,
                             interp, test_generator));
      XLS_RETURN_IF_ERROR(
          test_generator->SolvePredicate(matcher_result.constraint,
                                         /*negate_predicate=*/false));
      // Since we want to solve all the constraints in the match expression,
      // we can't return early here.
      if (matcher_result.matched && !found_match) {
        found_match = true;
        matched_constraint = matcher_result.constraint;
        matched_expr = arm->expr();
      }
    }
  }

  if (found_match) {
    // Every constraint after this match expression in the program needs to be
    // ANDed with the constraint that says the input matches the pattern.
    test_generator->AddConstraintToPath(matched_constraint,
                                        /*negate=*/false);
    return interp->Eval(matched_expr, bindings);
  }
  return absl::InternalError(
      absl::StrFormat("FailureError: %s The program being interpreted failed "
                      "with an incomplete match; value: %s",
                      expr->span().ToString(), to_match.ToString()));
}

absl::StatusOr<InterpValue> EvaluateSymFor(
    For* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp, ConcolicTestGenerator* test_generator) {
  XLS_ASSIGN_OR_RETURN(InterpValue iterable,
                       interp->Eval(expr->iterable(), bindings));
  std::unique_ptr<ConcreteType> concrete_iteration_type;
  if (expr->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        concrete_iteration_type,
        ConcretizeTypeAnnotation(expr->type_annotation(), bindings, interp));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue carry, interp->Eval(expr->init(), bindings));
  XLS_ASSIGN_OR_RETURN(int64_t length, iterable.GetLength());

  for (int64_t i = 0; i < length; ++i) {
    const InterpValue& x = iterable.GetValuesOrDie().at(i);
    XLS_ASSIGN_OR_RETURN(int64_t bit_value, x.GetBitsOrDie().ToInt64());
    auto sym_ptr = std::make_unique<SymbolicType>(SymbolicType::MakeLiteral(
        ConcreteInfo{x.IsSigned(), x.GetBitCount().value(), bit_value}));
    const InterpValue& x_sym =
        AddSymToValue(x, test_generator, std::move(sym_ptr));
    InterpValue iteration = InterpValue::MakeTuple({x_sym, carry});

    // If there's a type annotation, validate that the value we evaluated
    // conforms to it as a spot check.
    if (concrete_iteration_type != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          bool type_checks,
          ConcreteTypeAcceptsValue(*concrete_iteration_type, iteration));
      if (!type_checks) {
        XLS_ASSIGN_OR_RETURN(auto concrete_type,
                             ConcreteTypeFromValue(iteration));
        return absl::InternalError(absl::StrFormat(
            "EvaluateError: %s Type error found! Iteration value does not "
            "conform to type annotation at top of iteration %d:\n  got "
            "value: "
            "%s\n  type: %s\n  want: %s",
            expr->span().ToString(), i, iteration.ToString(),
            concrete_type->ToString(), concrete_iteration_type->ToString()));
      }
    }
    InterpBindings new_bindings =
        InterpBindings::CloneWith(bindings, expr->names(), iteration);
    XLS_ASSIGN_OR_RETURN(carry, interp->Eval(expr->body(), &new_bindings));
  }
  return carry;
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
  if (f->identifier() == m->name()) {
    test_generator->ResetRun();
  }
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
