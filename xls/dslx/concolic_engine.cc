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

#include "xls/dslx/concolic_engine.h"

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/status_macros.h"
#include "xls/solvers/z3_utils.h"

namespace xls::dslx {

absl::string_view GetTestTemplate() {
  constexpr absl::string_view kTestTemplate = R"(#![test]
fn $0_test_$1() {
$2
  let _ = assert_eq($0($3), $4);
  ()
})";
  return kTestTemplate;
}

absl::StatusOr<InterpValue> Z3AstToInterpValue(
    Z3_context ctx, SymbolicType* sym, Z3_model model,
    solvers::z3::DslxTranslator* translator) {
  int64_t value_int = 0;
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
  int64_t bit_count = sym->GetBitCount();
  if (sym->IsSigned())
    // Computes the 2's complement in case of negative number.
    return InterpValue::MakeSBits(bit_count,
                                  (value_int >> (bit_count - 1))
                                      ? (value_int | ~((1 << bit_count) - 1))
                                      : value_int);
  else
    return InterpValue::MakeUBits(bit_count, value_int);
}

// For each input in the Z3 model, creates the corresponding InterpValue.
absl::StatusOr<std::vector<InterpValue>> ExtractZ3Inputs(
    Z3_context ctx, Z3_solver solver, std::vector<InterpValue> function_params,
    solvers::z3::DslxTranslator* translator) {
  Z3_model model = Z3_solver_get_model(ctx, solver);
  std::vector<InterpValue> concrete_inputs;
  for (const auto& param : function_params) {
    if (!param.IsBits())
      return absl::InternalError(
          absl::StrFormat("Input parameter of type %s is not suppported.",
                          TagToString(param.tag())));
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Z3AstToInterpValue(ctx, param.sym(), model, translator));
    concrete_inputs.push_back(value);
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
    absl::StrAppend(&inputs_string, "  let ", "in_",
                    std::to_string(input_ctr++), " = ", param.ToString(),
                    ";\n");
  }
  test_string = absl::StrReplaceAll(test_string, {{"$2", inputs_string}});

  input_ctr = 0;
  std::string params_string;
  for (int64_t i = 0; i < function_params_.size() - 1; ++i) {
    absl::StrAppend(&params_string, "in_", std::to_string(input_ctr++), ", ");
  }
  absl::StrAppend(&params_string, "in_", std::to_string(input_ctr));
  test_string =
      absl::StrReplaceAll(test_string, {
                                           {"$3", params_string},
                                           {"$4", expected_value.ToString()},
                                       });
  // TODO(akalan): dump the test case to a file.
  std::cerr << test_string << std::endl;
  return absl::OkStatus();
}

absl::Status ConcolicTestGenerator::SolvePredicate(SymbolicType* predicate,
                                            bool negate_predicate) {
  XLS_RETURN_IF_ERROR(translator_->TranslatePredicate(predicate));
  translator_->SetTimeout(absl::InfiniteDuration());

  Z3_ast objective = translator_->GetTranslation(predicate).value();
  Z3_context ctx = translator_->ctx();

  // Converts predicates of BV type e.g. "if(a)" to bool type so that Z3 can
  // solve the predicate.
  if (!solvers::z3::IsAstBoolPredicate(ctx, objective)) {
    const bool bool_true = true;
    objective = Z3_mk_eq(ctx, objective, Z3_mk_bv_numeral(ctx, 1, &bool_true));
  }
  if (negate_predicate) objective = Z3_mk_not(ctx, objective);
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
    XLS_ASSIGN_OR_RETURN(
        std::vector<InterpValue> concrete_values,
        ExtractZ3Inputs(ctx, solver, function_params_, translator_.get()));
    function_params_values_.push_back(concrete_values);
  } else {
    XLS_VLOG(2) << "Predicate" << predicate->ToString().value() << " --> "
                << !negate_predicate << " is not satisfiable." << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
