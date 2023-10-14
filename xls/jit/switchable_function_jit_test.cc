// Copyright 2023 The XLS Authors
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

#include "xls/jit/switchable_function_jit.h"

#include <vector>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

class SwitchableFunctionJitTest : public IrTestBase {
 public:
  absl::StatusOr<Function*> TestFunction(Package* p) {
    FunctionBuilder fb(TestName(), p);
    auto p1 = fb.Param("p1", p->GetBitsType(8));
    auto p2 = fb.Param("p2", p->GetBitsType(8));
    fb.Tuple({fb.Add(p1, p2), fb.UMul(p1, p2)});

    return fb.Build();
  }
};
TEST_F(SwitchableFunctionJitTest, CanExecuteInterpreter) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(auto f, TestFunction(p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(auto interp, SwitchableFunctionJit::Create(
                                            f, ExecutionType::kInterpreter));
  EXPECT_FALSE(interp->function_jit().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      interp->Run(std::vector<Value>{Value(UBits(8, 8)), Value(UBits(4, 8))}));
  EXPECT_EQ(result.value,
            Value::Tuple({Value(UBits(12, 8)), Value(UBits(32, 8))}));
}

TEST_F(SwitchableFunctionJitTest, CanExecuteJit) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(auto f, TestFunction(p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto runner, SwitchableFunctionJit::Create(f, ExecutionType::kJit));
  EXPECT_TRUE(runner->function_jit().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      runner->Run(std::vector<Value>{Value(UBits(8, 8)), Value(UBits(4, 8))}));
  EXPECT_EQ(result.value,
            Value::Tuple({Value(UBits(12, 8)), Value(UBits(32, 8))}));
}

TEST_F(SwitchableFunctionJitTest, CanExecuteDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(auto f, TestFunction(p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(auto runner, SwitchableFunctionJit::Create(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      runner->Run(std::vector<Value>{Value(UBits(8, 8)), Value(UBits(4, 8))}));
  EXPECT_EQ(result.value,
            Value::Tuple({Value(UBits(12, 8)), Value(UBits(32, 8))}));
}

}  // namespace
}  // namespace xls
