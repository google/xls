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

#include "xls/codegen_v_1_5/block_conversion_wrapper_pass.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;

class BlockConversionWrapperPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Block* block) {
    PassResults results;
    OptimizationContext optimization_context;
    return BlockConversionWrapperPass(
               std::make_unique<DeadCodeEliminationPass>(),
               optimization_context)
        .Run(block->package(), BlockConversionPassOptions(), &results);
  }
};

TEST_F(BlockConversionWrapperPassTest, WrappedDce) {
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
}  // namespace xls::codegen
