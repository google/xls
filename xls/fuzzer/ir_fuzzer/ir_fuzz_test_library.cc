// Copyright 2025 The XLS Authors
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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// Evaluates the IR function with a set of parameter values.
absl::StatusOr<std::vector<InterpreterResult<Value>>> EvaluateParamSets(
    Function* f, absl::Span<const std::vector<Value>> param_sets) {
  std::vector<InterpreterResult<Value>> results;
  for (const std::vector<Value>& param_set : param_sets) {
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> result,
                         InterpretFunction(f, param_set));
    results.push_back(result);
  }
  return results;
}

// Returns a human-readable string representation of the results.
std::string StringifyResults(
    absl::Span<const InterpreterResult<Value>> results) {
  std::stringstream ss;
  ss << "[";
  for (int64_t i = 0; i < results.size(); i += 1) {
    ss << results[i].value;
    ss << (i != results.size() - 1 ? ", " : "]");
  }
  return ss.str();
}

}  // namespace

// Takes an IR function from a Package object and puts it through an
// optimization pass. It then evaluates the IR function with a set of parameter
// values before and after the pass. It returns the boolean value of whether the
// results changed.
void OptimizationPassChangesOutputs(
    const PackageAndTestParams& paramaterized_package,
    const OptimizationPass& pass) {
  // Verify that valid IR was generated.
  ScopedMaybeRecord<std::string> pre("original",
                                     paramaterized_package.p->DumpIr());
  XLS_ASSERT_OK(VerifyPackage(paramaterized_package.p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           paramaterized_package.p->GetFunction(kFuzzTestName));
  VLOG(3) << "IR Fuzzer-3: Before Pass IR:" << "\n" << f->DumpIr() << "\n";
  // Interpret the IR function with the parameters before reassociation.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<InterpreterResult<Value>> before_pass_results,
      EvaluateParamSets(f, paramaterized_package.param_sets));
  // Run the pass over the IR.
  PassResults results;
  OptimizationContext context;
  XLS_ASSERT_OK_AND_ASSIGN(
      bool ir_changed, pass.Run(paramaterized_package.p.get(),
                                OptimizationPassOptions(), &results, context));
  VLOG(3) << "4. After Pass IR:" << "\n" << f->DumpIr() << "\n";
  ScopedMaybeRecord<std::string> post("after_pass",
                                      paramaterized_package.p->DumpIr());
  // Interpret the IR function with the parameters after reassociation.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<InterpreterResult<Value>> after_pass_results,
      EvaluateParamSets(f, paramaterized_package.param_sets));
  VLOG(3) << "IR Fuzzer-5: IR Changed: " << (ir_changed ? "TRUE" : "FALSE")
          << "\n";
  VLOG(3) << "IR Fuzzer-6: Before Pass Results: "
          << StringifyResults(before_pass_results) << "\n";
  VLOG(3) << "IR Fuzzer-7: After Pass Results: "
          << StringifyResults(after_pass_results) << "\n";
  // Check if the results are the same before and after reassociation.
  bool results_changed = false;
  for (int64_t i = 0; i < before_pass_results.size(); i += 1) {
    if (before_pass_results[i].value != after_pass_results[i].value) {
      results_changed = true;
      break;
    }
  }
  VLOG(3) << "8. Results Changed: " << (results_changed ? "TRUE" : "FALSE")
          << "\n";
  ASSERT_FALSE(results_changed)
      << "\n"
      << "Expected: " << StringifyResults(before_pass_results) << "\n"
      << "Actual:   " << StringifyResults(after_pass_results);
}

}  // namespace xls
