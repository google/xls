// Copyright 2026 The XLS Authors
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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_evaluator_options.h"

namespace xls {
namespace {

constexpr int64_t kArgSetCount = 10;
constexpr int64_t kOptLevel = 3;

std::string FormatArgSet(absl::Span<const Value> arg_set) {
  return absl::StrCat("[", absl::StrJoin(arg_set, ", ", ValueFormatter), "]");
}

void JitMatchesInterpreterAtOpt3(FuzzPackageWithArgs fuzz_package_with_args) {
  RecordFuzzInfo(fuzz_package_with_args);
  Package* package = fuzz_package_with_args.fuzz_package.p.get();
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           package->GetFunction(kFuzzTestName));

  absl::StatusOr<std::unique_ptr<FunctionJit>> jit =
      FunctionJit::Create(function, EvaluatorOptions(),
                          JitEvaluatorOptions().set_opt_level(kOptLevel));
  XLS_ASSERT_OK(jit.status())
      << "FunctionJit::Create failed at opt_level=" << kOptLevel << "\nIR:\n"
      << package->DumpIr();

  const std::vector<std::vector<Value>>& arg_sets =
      fuzz_package_with_args.arg_sets;
  for (size_t i = 0; i < arg_sets.size(); ++i) {
    absl::StatusOr<InterpreterResult<Value>> interp_result =
        InterpretFunction(function, arg_sets[i]);
    XLS_ASSERT_OK(interp_result.status())
        << "InterpretFunction failed for arg set #" << i << ": "
        << FormatArgSet(arg_sets[i]) << "\nIR:\n"
        << package->DumpIr();

    absl::StatusOr<InterpreterResult<Value>> jit_result =
        (*jit)->Run(arg_sets[i]);
    XLS_ASSERT_OK(jit_result.status())
        << "FunctionJit::Run failed at opt_level=" << kOptLevel
        << " for arg set #" << i << ": " << FormatArgSet(arg_sets[i])
        << "\nIR:\n"
        << package->DumpIr();

    EXPECT_EQ(jit_result->value, interp_result->value)
        << "JIT vs interpreter value mismatch at opt_level=" << kOptLevel
        << " for arg set #" << i << ": " << FormatArgSet(arg_sets[i])
        << "\nIR:\n"
        << package->DumpIr();
    EXPECT_EQ(jit_result->events, interp_result->events)
        << "JIT vs interpreter events mismatch at opt_level=" << kOptLevel
        << " for arg set #" << i << ": " << FormatArgSet(arg_sets[i])
        << "\nIR:\n"
        << package->DumpIr();
  }
}

// Exclude raw smulp/umulp because GenIrNodesPass exposes the unsummed partial
// product tuple directly, and the JIT and IR interpreter intentionally choose
// different MulpOffsetForSimulation offsets (only tuple[0] + tuple[1] is
// semantically specified).
FUZZ_TEST(FunctionJitFuzzTest, JitMatchesInterpreterAtOpt3)
    .WithDomains(PackageWithArgsDomainBuilder(kArgSetCount)
                     .WithoutOperations({Op::kSMulp, Op::kUMulp})
                     .Build());

}  // namespace
}  // namespace xls
