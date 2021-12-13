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

#ifndef XLS_DSLX_CONCOLIC_ENGINE_H_
#define XLS_DSLX_CONCOLIC_ENGINE_H_

#include "xls/dslx/interp_value.h"
#include "xls/dslx/symbolic_type.h"
#include "xls/solvers/z3_dslx_translator.h"
#include "../z3/src/api/z3_api.h"

namespace xls::dslx {

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

 private:
  std::unique_ptr<solvers::z3::DslxTranslator> translator_;

  // Entry function input parameters.
  std::vector<InterpValue> function_params_;

  // For each ternary if predicate, stores the input parameters values
  // corresponding to that predicate.
  std::vector<std::vector<InterpValue>> function_params_values_;

  std::string entry_fn_name_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONCOLIC_ENGINE_H_
