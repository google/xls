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

#include "xls/codegen_v_1_5/register_cleanup_pass.h"

#include <memory>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/register.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace m = xls::op_matchers;

namespace xls::codegen {
namespace {

class RegisterCleanupPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p,
                           const BlockConversionPassOptions& options =
                               BlockConversionPassOptions()) {
    PassResults results;
    return RegisterCleanupPass().Run(p, options, &results);
  }
};

TEST_F(RegisterCleanupPassTest, AssertConditionFromPriorStage) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg, sbb.block()->AddRegister("x_reg", p->GetBitsType(1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue condition = sbb.UGe(x, sbb.Literal(UBits(3, 32)));
  sbb.RegisterWrite(reg, condition);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Assert(sbb.AfterAll({}), sbb.RegisterRead(reg), "failure", std::nullopt,
             SourceInfo{}, "assert0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  XLS_ASSERT_OK(Run(p.get()).status());
  XLS_ASSERT_OK_AND_ASSIGN(Node * assert0, sb->GetNode("assert0"));
  EXPECT_THAT(assert0->operands()[1], Not(m::Literal()));
}

}  // namespace
}  // namespace xls::codegen
