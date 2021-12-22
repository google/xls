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

#ifndef XLS_DSLX_CONCOLIC_TEST_GENERATOR_H_
#define XLS_DSLX_CONCOLIC_TEST_GENERATOR_H_

#include "xls/dslx/interp_value.h"
#include "xls/dslx/symbolic_type.h"
#include "xls/solvers/z3_dslx_translator.h"
#include "../z3/src/api/z3_api.h"

namespace xls::dslx {

// Represents a constraint node in the path constraint and whether or not it
// should be negated.
struct ConstraintNode {
  SymbolicType* constraint;
  bool negate;
};

// Wrapper class for Z3 DSLX solver that extracts the inputs from the z3
// solution and generates a DSLX test case.
class ConcolicTestGenerator {
 public:
  ConcolicTestGenerator(std::string entry_fn_name)
      : entry_fn_name_(entry_fn_name) {
    translator_ = solvers::z3::DslxTranslator::CreateTranslator();
  }

  // Attempts to prove the logic formula in the "predicate" by invoking z3.
  absl::Status SolvePredicate(SymbolicType* predicate, bool negate_predicate);

  // Generates a DSLX test based on the set of inputs and the expected
  // return value.
  absl::Status GenerateTest(InterpValue expected_value, int64_t test_no);

  void AddFnParam(const InterpValue& param) {
    function_params_.push_back(param);
  }
  std::vector<std::vector<InterpValue>> GetInputValues() {
    return function_params_values_;
  }
  std::vector<std::string> GetTestCases() { return generated_test_cases_; }

  // Stores the life-time owned symbolic representation for nodes so that they
  // are not destroyed between function calls etc.
  void StoreSymPointers(std::unique_ptr<SymbolicType> sym_tree) {
    symbolic_trees_.push_back(std::move(sym_tree));
  }

  // If we reach a part of program via a constraint e.g. function call in the
  // ternary if expressions, we add that constraint to the path so that every
  // other constraint in that part of program is conjuncted with it.
  void AddConstraintToPath(SymbolicType* constraint, bool negate) {
    path_constraints_.push_back({constraint, negate});
  }
  void PopConstraintFromPath() { path_constraints_.pop_back(); }

  // clears the path constraints and the function inputs for a new run.
  void ResetRun() {
    function_params_.clear();
    path_constraints_.clear();
  }

 private:
  std::unique_ptr<solvers::z3::DslxTranslator> translator_;

  // Entry function input parameters.
  std::vector<InterpValue> function_params_;

  // For each ternary if predicate, stores the input parameters values
  // corresponding to that predicate.
  std::vector<std::vector<InterpValue>> function_params_values_;

  // Set of constraints that should be conjuncted with every constraint in the
  // program.
  std::vector<ConstraintNode> path_constraints_;

  // Stores a set of constraints that have already been solved to avoid
  // re-presocessing the same ones.
  absl::flat_hash_set<Z3_ast> solved_constraints_;

  std::vector<std::unique_ptr<SymbolicType>> symbolic_trees_;

  std::vector<std::string> generated_test_cases_;
  std::string entry_fn_name_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONCOLIC_TEST_GENERATOR_H_
