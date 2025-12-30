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

#include "xls/codegen_v_1_5/idle_insertion_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;

namespace m = ::xls::op_matchers;

class IdleInsertionPassTest : public IrTestBase {
 protected:
  IdleInsertionPassTest() = default;

  absl::StatusOr<bool> Run(Package* p,
                           const BlockConversionPassOptions& options) {
    PassResults results;
    return IdleInsertionPass().Run(p, options, &results);
  }
};

TEST_F(IdleInsertionPassTest, PassDoesNothingIfAddIdleOutputIsFalse) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue out = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK(sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.add_idle_output(false);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(false));
}

TEST_F(IdleInsertionPassTest, SingleStagePipeline) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  sbb.StartStage(
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"inputs_valid"),
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"outputs_ready"));
  BValue out = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.OutputPort("out", out);
  // outputs_valid=1, outputs_ready=1
  sbb.EndStage(
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"aiv"),
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"outputs_valid"));
  XLS_ASSERT_OK(sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.add_idle_output(true);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));
  XLS_ASSERT_OK_AND_ASSIGN(Node * idle_output, block->GetOutputPort("idle"));

  // Check structure:
  // stage_idle =
  //   NAND(inputs_valid, active_inputs_valid, OR(NOT(outputs_valid),
  //                                              outputs_ready))
  // idle = stage_idle (since 1 stage)

  EXPECT_THAT(idle_output,
              m::OutputPort(m::Nand(m::Name("inputs_valid"), m::Name("aiv"),
                                    m::Or(m::Not(m::Name("outputs_valid")),
                                          m::Name("outputs_ready")))));
}

TEST_F(IdleInsertionPassTest, MultiStagePipeline) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  BValue in = sbb.InputPort("in", p->GetBitsType(32));

  // Stage 0
  sbb.StartStage(
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"p0_inputs_valid"),
      sbb.Literal(UBits(1, 1), SourceInfo(),
                  /*name=*/"p0_outputs_ready"));
  BValue p0 = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"p0_active_inputs_valid"),
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"p0_outputs_valid"));

  // Stage 1
  sbb.StartStage(
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"p1_inputs_valid"),
      sbb.Literal(UBits(1, 1), SourceInfo(),
                  /*name=*/"p1_outputs_ready"));
  BValue out = sbb.UMul(p0, sbb.Literal(UBits(2, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"p1_active_inputs_valid"),
      sbb.Literal(UBits(1, 1), SourceInfo(), /*name=*/"p1_outputs_valid"));

  XLS_ASSERT_OK(sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.add_idle_output(true);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));
  XLS_ASSERT_OK_AND_ASSIGN(Node * idle_output, block->GetOutputPort("idle"));

  // Check structure:
  // stage_idle =
  //   NAND(inputs_valid, active_inputs_valid, OR(NOT(outputs_valid),
  //                                              outputs_ready))
  // idle = AND(stage_0_idle, stage_1_idle)
  EXPECT_THAT(idle_output,
              m::OutputPort(m::And(
                  AllOf(m::Name("stage_0_idle"),
                        m::Nand(m::Name("p0_inputs_valid"),
                                m::Name("p0_active_inputs_valid"),
                                m::Or(m::Not(m::Name("p0_outputs_valid")),
                                      m::Name("p0_outputs_ready")))),
                  AllOf(m::Name("stage_1_idle"),
                        m::Nand(m::Name("p1_inputs_valid"),
                                m::Name("p1_active_inputs_valid"),
                                m::Or(m::Not(m::Name("p1_outputs_valid")),
                                      m::Name("p1_outputs_ready")))))));
}

}  // namespace
}  // namespace xls::codegen
