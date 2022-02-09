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

#include "xls/dslx/concolic_test_generator.h"

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/status_macros.h"
#include "xls/solvers/z3_utils.h"

namespace xls::dslx {

absl::string_view GetTestTemplate() {
  constexpr absl::string_view kTestTemplate = R"(
#![test]
fn $0_test_$1() {
$2
  let _ = assert_eq($0($3), $4);
  ()
})";
  return kTestTemplate;
}

// Prints the complete type for structs and enums which interpreter can run. For
// the rest of the types invokes the InterpValue inherent method ToString.
absl::StatusOr<std::string> InterpValueToString(InterpValue value) {
  auto make_guts = [&](bool is_struct = false) {
    int64_t struct_member_count = 0;
    std::vector<std::string> struct_members = value.GetStructMembers();
    return absl::StrJoin(value.GetValuesOrDie(), ", ",
                         [&struct_member_count, struct_members](
                             std::string* out, const InterpValue& v) {
                           absl::StrAppend(
                               out, struct_members[struct_member_count++], ": ",
                               v.ToString());
                         });
  };

  switch (value.tag()) {
    case InterpValueTag::kUBits:
    case InterpValueTag::kSBits:
    case InterpValueTag::kArray:
    case InterpValueTag::kFunction:
    case InterpValueTag::kToken:
    case InterpValueTag::kChannel:
      return value.ToString();
    case InterpValueTag::kTuple: {
      std::vector<std::string> struct_members = value.GetStructMembers();
      if (struct_members.empty()) {
        return value.ToString();
      }
      // Prints a struct in the format: "StructName {foo:bar, ...}".
      return absl::StrCat(
          struct_members.back(),  // Last element contains the struct's name.
          absl::StrFormat(" {%s}", make_guts()));
    }
    case InterpValueTag::kEnum: {
      Bits bits = value.GetBitsOrDie();
      std::string bits_string =
          absl::StrCat(value.IsSigned() ? "s" : "u",
                       std::to_string(value.GetBitCount().value()), ":",
                       value.GetBitsOrDie().ToString());
      for (const EnumMember& m : value.type()->values()) {
        if (m.value->ToString() == bits_string) {
          return absl::StrFormat("%s::%s", value.type()->identifier(),
                                 m.name_def->ToString());
        }
      }
      return absl::StrFormat("%s:%s", value.type()->identifier(), bits_string);
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown InterpValueTag: ", value.tag()));
  }
}

absl::StatusOr<InterpValue> Z3AstToInterpValue(
    Z3_context ctx, InterpValue value, SymbolicType* sym, Z3_model model,
    std::vector<SymbolicType*> cast_slice_params,
    solvers::z3::DslxTranslator* translator) {
  int64_t value_int = 0;

  // If the symbolic tree for this node appears in the list of casted/sliced
  // parameters, update the node so that we create the parameter in the DSLX
  // test based on the original bitwidth/sign.
  for (SymbolicType* param : cast_slice_params) {
    if (sym->id() == param->id()) {
      sym = param;
      break;
    }
  }

  // If the input doesn't appear in the Z3 AST nodes, it's a "don't care" value
  // and can be safely initialized to zero, otherwise we evaluate the model to
  // get the input value.
  absl::optional<Z3_ast> translation = translator->GetTranslation(sym);
  if (translation.has_value()) {
    Z3_ast z3_value;
    Z3_model_eval(ctx, model, translation.value(),
                  /*model_completion=*/true, &z3_value);
    Z3_get_numeral_int64(ctx, z3_value, &value_int);
  }

  // If the parameter has been sliced, shift the value so that it corresponds to
  // the original parameter.
  //
  // e.g. if the value for a[n:m] is N, the value  for a is (N << n).
  if (sym->IsSliced()) {
    value_int <<= sym->slice_index();
  }

  int64_t bit_count = sym->bit_count();
  if (sym->IsSigned()) {
    // Computes the 2's complement in case of negative number.
    Bits bits = SBits((value_int >> (bit_count - 1))
                          ? (value_int | ~((1 << bit_count) - 1))
                          : value_int,
                      bit_count);
    if (value.IsEnum()) {
      return InterpValue::MakeEnum(bits, value.type());
    }
    return InterpValue::MakeBits(/*is_signed=*/true, bits);
  }
  Bits bits = UBits(value_int, bit_count);
  if (value.IsEnum()) {
    return InterpValue::MakeEnum(bits, value.type());
  }
  return InterpValue::MakeBits(/*is_signed=*/false, bits);
}

// The symbolic nodes for the tuples are stored in a flat array, this function
// creates the original InterpValue tuple e.g. (u32, (u32, (u32,), u32)) from
// the flat list.
InterpValue MakeTupleFromElements(InterpValue value,
                                  std::vector<InterpValue> elements,
                                  int64_t& element_count) {
  std::vector<InterpValue> elements_unflatten;
  if (value.IsBits()) {
    return elements[element_count];
  }

  for (const InterpValue& value : value.GetValuesOrDie()) {
    InterpValue tuple = MakeTupleFromElements(value, elements, element_count);
    if (value.IsBits()) {
      element_count++;
    }
    elements_unflatten.push_back(tuple);
  }
  return InterpValue::MakeTuple(elements_unflatten);
}

// Returns a list of function parameters that were casted or bit sliced in this
// predicate.
absl::StatusOr<std::vector<SymbolicType*>> CastAndSlicedParams(
    SymbolicType* predicate) {
  std::vector<SymbolicType*> fn_params;
  auto Walk = [&](SymbolicType* x) -> absl::Status {
    if (x != nullptr && (x->IsCasted() || x->IsSliced())) {
      fn_params.push_back(x);
    }
    return absl::OkStatus();
  };
  XLS_RETURN_IF_ERROR(predicate->DoPostorder(Walk));
  return fn_params;
}

// For each input in the Z3 model, creates the corresponding InterpValue.
absl::StatusOr<std::vector<InterpValue>> ExtractZ3Inputs(
    Z3_context ctx, Z3_solver solver, SymbolicType* predicate,
    std::vector<InterpValue> function_params,
    solvers::z3::DslxTranslator* translator) {
  Z3_model model = Z3_solver_get_model(ctx, solver);
  XLS_ASSIGN_OR_RETURN(std::vector<SymbolicType*> cast_slice_params,
                       CastAndSlicedParams(predicate));
  std::vector<InterpValue> concrete_inputs;
  for (const InterpValue& param : function_params) {
    if (param.sym()->IsArray()) {
      std::vector<InterpValue> elements;
      elements.reserve(param.sym()->GetChildren().size());
      for (SymbolicType* sym_child : param.sym()->GetChildren()) {
        XLS_ASSIGN_OR_RETURN(InterpValue value,
                             Z3AstToInterpValue(ctx, param, sym_child, model,
                                                cast_slice_params, translator));
        elements.push_back(value);
      }
      if (param.tag() == InterpValueTag::kArray) {
        XLS_ASSIGN_OR_RETURN(InterpValue array_value,
                             InterpValue::MakeArray(std::move(elements)));
        concrete_inputs.push_back(array_value);
      } else {
        int64_t element_count = 0;
        InterpValue tuple =
            MakeTupleFromElements(param, elements, element_count);
        concrete_inputs.push_back(
            tuple.UpdateWithStructInfo(param.GetStructMembers()));
      }
    } else {
      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           Z3AstToInterpValue(ctx, param, param.sym(), model,
                                              cast_slice_params, translator));
      concrete_inputs.push_back(value);
    }
  }
  return concrete_inputs;
}

absl::Status ConcolicTestGenerator::GenerateTest(InterpValue expected_value,
                                                 int64_t test_no) {
  std::string test_string = absl::StrReplaceAll(
      GetTestTemplate(), {
                             {"$0", entry_fn_name_},
                             {"$1", std::to_string(test_no)},
                         });

  int64_t input_ctr = 0;
  std::string inputs_string;
  for (const auto& param : function_params_values_[test_no]) {
    XLS_ASSIGN_OR_RETURN(std::string value_string, InterpValueToString(param));
    absl::StrAppend(&inputs_string, "  let ", "in_",
                    std::to_string(input_ctr++), " = ", value_string, ";\n");
  }
  test_string = absl::StrReplaceAll(test_string, {{"$2", inputs_string}});

  input_ctr = 0;
  std::string params_string;
  for (int64_t i = 0; i < function_params_.size() - 1; ++i) {
    absl::StrAppend(&params_string, "in_", std::to_string(input_ctr++), ", ");
  }
  absl::StrAppend(&params_string, "in_", std::to_string(input_ctr));
  XLS_ASSIGN_OR_RETURN(std::string expected_string,
                       InterpValueToString(expected_value));
  test_string = absl::StrReplaceAll(test_string, {
                                                     {"$3", params_string},
                                                     {"$4", expected_string},
                                                 });
  // TODO(akalan): dump the test case to a file.
  std::cerr << test_string << std::endl;
  generated_test_cases_.push_back(test_string);
  return absl::OkStatus();
}

// Translates the node in the path constraint to its Z3 representation and
// negates it if necessary. Converts the constraint to a bit vector type if it's
// a boolean.
absl::StatusOr<Z3_ast> ProcessConstraintNode(
    Z3_context ctx, ConstraintNode node,
    solvers::z3::DslxTranslator* translator) {
  bool bool_false = false;
  XLS_RETURN_IF_ERROR(translator->TranslatePredicate(node.constraint));
  Z3_ast objective = translator->GetTranslation(node.constraint).value();
  if (node.negate) {
    if (solvers::z3::IsAstBoolPredicate(ctx, objective)) {
      objective = Z3_mk_not(ctx, objective);
    } else {
      objective =
          Z3_mk_eq(ctx, objective, Z3_mk_bv_numeral(ctx, 1, &bool_false));
    }
  }
  if (solvers::z3::IsAstBoolPredicate(ctx, objective)) {
    objective = solvers::z3::BoolToBv(ctx, objective);
  }
  return objective;
}

// Creates a conjunction between the constraints in the path i.e. c1 ^ c2 ^... .
absl::StatusOr<Z3_ast> PathConstraintsConjunction(
    std::vector<ConstraintNode> path_constraints, Z3_context ctx,
    solvers::z3::DslxTranslator* translator) {
  const bool bool_true = true;
  XLS_ASSIGN_OR_RETURN(
      Z3_ast objective,
      ProcessConstraintNode(ctx, path_constraints.at(0), translator));
  for (int64_t i = 1; i < path_constraints.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Z3_ast node,
        ProcessConstraintNode(ctx, path_constraints.at(i), translator));
    objective = Z3_mk_bvand(ctx, objective, node);
  }
  return Z3_mk_eq(ctx, objective, Z3_mk_bv_numeral(ctx, 1, &bool_true));
}

absl::Status ConcolicTestGenerator::SolvePredicate(SymbolicType* predicate,
                                                   bool negate_predicate) {
  XLS_RETURN_IF_ERROR(translator_->TranslatePredicate(predicate));
  translator_->SetTimeout(absl::InfiniteDuration());

  Z3_ast objective = translator_->GetTranslation(predicate).value();
  Z3_context ctx = translator_->ctx();

  // Used for generating literal 1 in Z3 as it only accepts a pointer to
  // the value.
  bool bool_true = true;

  // Converts predicates of BV type e.g. "if(a)" to bool type so that Z3 can
  // solve the predicate.
  if (!solvers::z3::IsAstBoolPredicate(ctx, objective)) {
    objective = Z3_mk_eq(ctx, objective, Z3_mk_bv_numeral(ctx, 1, &bool_true));
  }
  if (negate_predicate) objective = Z3_mk_not(ctx, objective);

  // Compute two constraints: 1) path_constraints ^ objective
  //                          2) path_constraints ^ ~objective
  // The second one is merely used for storing in the map for faster look ups.
  Z3_ast objective_with_path = objective;
  Z3_ast objective_with_path_negate = Z3_mk_not(ctx, objective);
  if (!path_constraints_.empty()) {
    XLS_ASSIGN_OR_RETURN(
        Z3_ast path_conjunction,
        PathConstraintsConjunction(path_constraints_, ctx, translator_.get()));
    // And with the path constraint.
    objective_with_path_negate =
        Z3_mk_bvand(ctx, solvers::z3::BoolToBv(ctx, Z3_mk_not(ctx, objective)),
                    solvers::z3::BoolToBv(ctx, path_conjunction));
    objective_with_path =
        Z3_mk_bvand(ctx, solvers::z3::BoolToBv(ctx, objective),
                    solvers::z3::BoolToBv(ctx, path_conjunction));
    // Convert to boolean.
    objective_with_path = Z3_mk_eq(ctx, objective_with_path,
                                   Z3_mk_bv_numeral(ctx, 1, &bool_true));
    objective_with_path_negate = Z3_mk_eq(ctx, objective_with_path_negate,
                                          Z3_mk_bv_numeral(ctx, 1, &bool_true));
    objective = objective_with_path;
  }
  // If we have already solved this constraint skip; otherwise, add the
  // constraint and its negation to the map.
  if (solved_constraints_.contains(objective_with_path)) {
    XLS_VLOG(2) << "already contains objective "
                << Z3_ast_to_string(ctx, objective_with_path) << std::endl;
    return absl::OkStatus();
  }
  // TODO(akalan): if the predicate is already negated, negating it again is not
  // the same as the original predicate in Z3!
  // In the other words, after translating to Z3: not(not(a)) != a.
  solved_constraints_.insert(objective_with_path);
  solved_constraints_.insert(objective_with_path_negate);

  XLS_VLOG(2) << "objective:\n"
              << Z3_ast_to_string(ctx, objective) << std::endl;

  Z3_solver solver = solvers::z3::CreateSolver(ctx, 1);
  absl::Cleanup solver_cleanup = [ctx, solver] {
    Z3_solver_dec_ref(ctx, solver);
  };
  Z3_solver_assert(ctx, solver, objective);
  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  XLS_VLOG(2) << solvers::z3::SolverResultToString(ctx, solver, satisfiable)
              << std::endl;
  if (satisfiable == Z3_L_TRUE) {
    XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> concrete_values,
                         ExtractZ3Inputs(ctx, solver, predicate,
                                         function_params_, translator_.get()));
    function_params_values_.push_back(concrete_values);
  } else {
    XLS_VLOG(2) << "Predicate" << predicate->ToString().value() << " --> "
                << !negate_predicate << " is not satisfiable." << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
