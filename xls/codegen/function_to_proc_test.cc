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

#include "xls/codegen/function_to_proc.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace m = xls::op_matchers;

namespace xls {
namespace verilog {
namespace {

class FunctionToProcTest : public IrTestBase {
 protected:
  // Returns the unique output port of the proc (send over a port
  // channel). Check fails if no such unique send exists.
  Send* GetOutputNode(Proc* proc) {
    Send* send = nullptr;
    for (Node* node : proc->nodes()) {
      if (node->Is<Send>() && node->package()
                                  ->GetChannel(node->As<Send>()->channel_id())
                                  .value()
                                  ->IsPort()) {
        XLS_CHECK(send == nullptr);
        send = node->As<Send>();
      }
    }
    XLS_CHECK(send != nullptr);
    return send;
  }
};

class TestDelayEstimator : public DelayEstimator {
 public:
  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kConcat:
        return 0;
      default:
        return 1;
    }
  }
};

TEST_F(FunctionToProcTest, SimpleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           FunctionToProc(f, "SimpleFunctionProc"));

  EXPECT_EQ(proc->name(), "SimpleFunctionProc");
  EXPECT_EQ(proc->StateType(), p->GetTupleType({}));
  EXPECT_EQ(p->channels().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_ch, p->GetChannel("out"));
  EXPECT_TRUE(out_ch->IsPort());
  EXPECT_EQ(out_ch->supported_ops(), ChannelOps::kSendOnly);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * x_ch, p->GetChannel("x"));
  EXPECT_TRUE(x_ch->IsPort());
  EXPECT_EQ(x_ch->supported_ops(), ChannelOps::kReceiveOnly);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * y_ch, p->GetChannel("y"));
  EXPECT_TRUE(y_ch->IsPort());
  EXPECT_EQ(y_ch->supported_ops(), ChannelOps::kReceiveOnly);

  EXPECT_THAT(GetOutputNode(proc),
              m::OutputPort(m::Add(m::InputPort("x"), m::InputPort("y"))));
}

TEST_F(FunctionToProcTest, ZeroInputs) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Literal(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, FunctionToProc(f, "ZeroInputsProc"));

  EXPECT_EQ(p->channels().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_ch, p->GetChannel("out"));
  EXPECT_TRUE(out_ch->IsPort());
  EXPECT_EQ(out_ch->supported_ops(), ChannelOps::kSendOnly);

  EXPECT_THAT(GetOutputNode(proc), m::OutputPort(m::Literal(42)));
}

TEST_F(FunctionToProcTest, ZeroWidthInputsAndOutput) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetTupleType({}));
  BValue y = fb.Param("y", p->GetBitsType(0));
  fb.Param("z", p->GetBitsType(1234));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({x, y})));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           FunctionToProc(f, "SimpleFunctionProc"));

  EXPECT_EQ(proc->name(), "SimpleFunctionProc");
  EXPECT_EQ(proc->StateType(), p->GetTupleType({}));
  EXPECT_EQ(p->channels().size(), 4);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * z_ch, p->GetChannel("z"));
  EXPECT_TRUE(z_ch->IsPort());
  EXPECT_EQ(z_ch->supported_ops(), ChannelOps::kReceiveOnly);
}

TEST_F(FunctionToProcTest, SimplePipelinedFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Negate(fb.Not(fb.Add(x, y)))));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, FunctionToPipelinedProc(schedule, f, "SimpleFunctionProc"));

  EXPECT_THAT(GetOutputNode(proc),
              m::OutputPort(m::Neg(m::Register(
                  m::Not(m::Register(m::Add(m::Param("x"), m::Param("y"))))))));
}

TEST_F(FunctionToProcTest, TrivialPipelinedFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Negate(fb.Not(fb.Add(x, y)))));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, FunctionToPipelinedProc(schedule, f, "SimpleFunctionProc"));

  EXPECT_THAT(GetOutputNode(proc),
              m::OutputPort(m::Neg(m::Register(
                  m::Not(m::Register(m::Add(m::Param("x"), m::Param("y"))))))));
}

TEST_F(FunctionToProcTest, ZeroWidthPipeline) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetTupleType({}));
  BValue y = fb.Param("y", p->GetBitsType(0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({x, y})));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));
  XLS_ASSERT_OK(
      FunctionToPipelinedProc(schedule, f, "ZeroWidthPipelineProc").status());

  EXPECT_EQ(p->channels().size(), 3);
}

}  // namespace
}  // namespace verilog
}  // namespace xls
