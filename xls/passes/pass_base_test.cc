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

#include "xls/passes/pass_base.h"
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"

namespace xls {
namespace {
class PassBaseTest : public IrTestBase {};

// A sneaky pass that tries to avoid returning unlucky numbers just like
// architects. Any number >=13 is increased by 1.
class ArchitectNumber : public OptimizationFunctionBasePass {
 public:
  ArchitectNumber()
      : OptimizationFunctionBasePass("architect_number", "Architect Number") {}
  ~ArchitectNumber() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override {
    Node* original = f->AsFunctionOrDie()->return_value();
    if (!original->GetType()->IsBits()) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * lit_one,
        f->MakeNode<Literal>(original->loc(),
                             Value(UBits(1, original->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(
        Node * lit_thirteen,
        f->MakeNode<Literal>(original->loc(),
                             Value(UBits(13, original->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(
        Node * add_one,
        f->MakeNode<BinOp>(original->loc(), lit_one, original, Op::kAdd));
    XLS_ASSIGN_OR_RETURN(Node * is_unlucky,
                         f->MakeNode<CompareOp>(original->loc(), original,
                                                lit_thirteen, Op::kUGe));
    XLS_ASSIGN_OR_RETURN(
        Node * maybe_add,
        f->MakeNode<Select>(original->loc(), is_unlucky,
                            absl::Span<Node* const>{original, add_one},
                            std::nullopt));
    XLS_RETURN_IF_ERROR(f->AsFunctionOrDie()->set_return_value(maybe_add));
    // Oops, we changed things and should return true here!
    return false;
  }
};

TEST_F(PassBaseTest, DetectEasyIncorrectReturn) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(13, 64));
  ASSERT_THAT(fb.Build(), status_testing::IsOk());
  ArchitectNumber pass;
  EXPECT_THAT(pass.Run(p.get(), OptimizationPassOptions(), nullptr),
              status_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::ContainsRegex(
                      "Pass architect_number indicated IR unchanged, but IR is "
                      "changed: \\[Before\\] 1 nodes != \\[after\\] 6 nodes")));
}

TEST_F(PassBaseTest, DetectEasyIncorrectReturnInCompound) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(13, 64));
  ASSERT_THAT(fb.Build(), status_testing::IsOk());
  ArchitectNumber pass;
  EXPECT_THAT(pass.Run(p.get(), OptimizationPassOptions(), nullptr),
              status_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::ContainsRegex(
                      "Pass architect_number indicated IR unchanged, but IR is "
                      "changed: \\[Before\\] 1 nodes != \\[after\\] 6 nodes")));
}

}  // namespace
}  // namespace xls
