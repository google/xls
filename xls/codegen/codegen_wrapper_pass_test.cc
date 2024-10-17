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

#include "xls/codegen/codegen_wrapper_pass.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/dce_pass.h"

namespace m = ::xls::op_matchers;

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;

class CodegenWrapperPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Block* block) {
    CodegenPassResults results;
    CodegenPassUnit unit(block->package(), block);
    return CodegenWrapperPass(std::make_unique<DeadCodeEliminationPass>())
        .Run(&unit, CodegenPassOptions(), &results);
  }
};

TEST_F(CodegenWrapperPassTest, WrappedDce) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  bb.Add(a, b);
  bb.OutputPort("c", bb.Subtract(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // The add should be eliminated.
  EXPECT_EQ(block->node_count(), 5);
  EXPECT_THAT(Run(block), IsOkAndHolds(true));
  EXPECT_EQ(block->node_count(), 4);

  EXPECT_THAT(FindNode("c", block),
              m::OutputPort(m::Sub(m::InputPort("a"), m::InputPort("b"))));

  // Pass should be idempotent.
  EXPECT_THAT(Run(block), IsOkAndHolds(false));
  EXPECT_EQ(block->node_count(), 4);
}

}  // namespace
}  // namespace xls::verilog
