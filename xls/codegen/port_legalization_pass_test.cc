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

#include "xls/codegen/port_legalization_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls::verilog {
namespace {

using status_testing::IsOkAndHolds;

class PortLegalizationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Block* block) {
    PassResults results;
    CodegenPassUnit unit(block->package(), block);
    return PortLegalizationPass().Run(&unit, CodegenPassOptions(), &results);
  }
};

TEST_F(PortLegalizationPassTest, APlusB) {
  auto p = CreatePackage();

  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // There are no zero-width ports so nothing should be removed.
  EXPECT_EQ(block->GetPorts().size(), 3);
  EXPECT_THAT(Run(block), IsOkAndHolds(false));
  EXPECT_EQ(block->GetPorts().size(), 3);
}

TEST_F(PortLegalizationPassTest, ZeroWidthInputsAndOutput) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetTupleType({}));
  bb.OutputPort("out", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // All ports should be removed.
  EXPECT_EQ(block->GetPorts().size(), 2);
  ASSERT_THAT(Run(block), IsOkAndHolds(true));
  EXPECT_EQ(block->GetPorts().size(), 0);
}

TEST_F(PortLegalizationPassTest, ZeroWidthInput) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(0));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  bb.OutputPort("out", bb.Concat({a, b}));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // One port should be removed.
  EXPECT_EQ(block->GetPorts().size(), 3);
  ASSERT_THAT(Run(block), IsOkAndHolds(true));
  ASSERT_EQ(block->GetPorts().size(), 2);
  EXPECT_EQ(absl::get<InputPort*>(block->GetPorts()[0])->GetName(), "b");
  EXPECT_EQ(absl::get<OutputPort*>(block->GetPorts()[1])->GetName(), "out");
}


}  // namespace
}  // namespace xls::verilog
