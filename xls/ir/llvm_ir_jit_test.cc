// Copyright 2020 Google LLC
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

#include "xls/ir/llvm_ir_jit.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_evaluator_test.h"
#include "re2/re2.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

INSTANTIATE_TEST_SUITE_P(
    LlvmIrJitTest, IrEvaluatorTest,
    testing::Values(IrEvaluatorTestParam(
        [](Function* function,
           const std::vector<Value>& args) -> xabsl::StatusOr<Value> {
          XLS_ASSIGN_OR_RETURN(auto jit, LlvmIrJit::Create(function));
          return jit->Run(args);
        },
        [](Function* function,
           const absl::flat_hash_map<std::string, Value>& kwargs)
            -> xabsl::StatusOr<Value> {
          XLS_ASSIGN_OR_RETURN(auto jit, LlvmIrJit::Create(function));
          return jit->Run(kwargs);
        })));

// This test verifies that a compiled JIT function can be re-used.
TEST(LlvmIrJitTest, ReuseTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn get_identity(x: bits[8]) -> bits[8] {
    ret identity.1: bits[8] = identity(x)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, LlvmIrJit::Create(function));
  EXPECT_THAT(jit->Run({Value(UBits(2, 8))}), IsOkAndHolds(Value(UBits(2, 8))));
  EXPECT_THAT(jit->Run({Value(UBits(4, 8))}), IsOkAndHolds(Value(UBits(4, 8))));
  EXPECT_THAT(jit->Run({Value(UBits(7, 8))}), IsOkAndHolds(Value(UBits(7, 8))));
}

}  // namespace
}  // namespace xls
