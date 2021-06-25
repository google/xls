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

#include "xls/codegen/block_conversion.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
// #include "xls/codegen/block_generator.h"
// #include "xls/codegen/codegen_options.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/verifier.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace m = xls::op_matchers;

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using testing::Pair;
using testing::UnorderedElementsAre;

class BlockConversionTest : public IrTestBase {
 protected:
  // Returns the unique output port of the block (send over a port
  // channel). Check fails if no such unique send exists.
  OutputPort* GetOutputPort(Block* block) {
    OutputPort* output_port = nullptr;
    for (Node* node : block->nodes()) {
      if (node->Is<OutputPort>()) {
        output_port = node->As<OutputPort>();
      }
    }
    XLS_CHECK(output_port != nullptr);
    return output_port;
  }
};

class TestDelayEstimator : public DelayEstimator {
 public:
  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kInputPort:
      case Op::kOutputPort:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kConcat:
        return 0;
      default:
        return 1;
    }
  }
};

TEST_F(BlockConversionTest, SimpleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           FunctionToBlock(f, "SimpleFunctionBlock"));

  EXPECT_EQ(block->name(), "SimpleFunctionBlock");
  EXPECT_EQ(block->GetPorts().size(), 3);

  EXPECT_THAT(GetOutputPort(block),
              m::OutputPort(m::Add(m::InputPort("x"), m::InputPort("y"))));
}

TEST_F(BlockConversionTest, ZeroInputs) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Literal(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           FunctionToBlock(f, "ZeroInputsBlock"));

  EXPECT_EQ(block->GetPorts().size(), 1);

  EXPECT_THAT(GetOutputPort(block), m::OutputPort("out", m::Literal(42)));
}

TEST_F(BlockConversionTest, ZeroWidthInputsAndOutput) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetTupleType({}));
  BValue y = fb.Param("y", p->GetBitsType(0));
  fb.Param("z", p->GetBitsType(1234));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({x, y})));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           FunctionToBlock(f, "SimpleFunctionBlock"));

  EXPECT_EQ(block->GetPorts().size(), 4);
}

TEST_F(BlockConversionTest, SimplePipelinedFunction) {
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
      Block * block,
      FunctionToPipelinedBlock(schedule, f, "SimpleFunctionBlock"));

  EXPECT_THAT(GetOutputPort(block),
              m::OutputPort(m::Neg(m::Register(m::Not(m::Register(
                  m::Add(m::InputPort("x"), m::InputPort("y"))))))));
}

TEST_F(BlockConversionTest, TrivialPipelinedFunction) {
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
      Block * block,
      FunctionToPipelinedBlock(schedule, f, "SimpleFunctionBlock"));

  EXPECT_THAT(GetOutputPort(block),
              m::OutputPort(m::Neg(m::Register(m::Not(m::Register(
                  m::Add(m::InputPort("x"), m::InputPort("y"))))))));
}

TEST_F(BlockConversionTest, ZeroWidthPipeline) {
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
  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block,
      FunctionToPipelinedBlock(schedule, f, "ZeroWidthPipelineBlock"));

  EXPECT_EQ(block->GetRegisters().size(), 4);
}

// Verifies that an implicit token, as generated by the DSLX IR converter, is
// appropriately plumbed into the wrapping block during conversion.
TEST_F(BlockConversionTest, ImplicitToken) {
  const std::string kIrText = R"(
package implicit_token

fn __itok__implicit_token__main(__token: token, __activated: bits[1]) ->
(token, ()) {
  after_all.7: token = after_all(__token, id=7)
  tuple.6: () = tuple(id=6)
  ret tuple.8: (token, ()) = tuple(after_all.7, tuple.6, id=8)
}

fn __implicit_token__main() -> () {
  after_all.9: token = after_all(id=9)
  literal.10: bits[1] = literal(value=1, id=10)
  invoke.11: (token, ()) = invoke(after_all.9, literal.10,
  to_apply=__itok__implicit_token__main, id=11) tuple_index.12: token =
  tuple_index(invoke.11, index=0, id=12) invoke.13: (token, ()) =
  invoke(tuple_index.12, literal.10, to_apply=__itok__implicit_token__main,
  id=13) ret tuple_index.14: () = tuple_index(invoke.13, index=1, id=14)
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, p->GetFunction("__implicit_token__main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto block,
                           FunctionToBlock(f, "ImplicitTokenBlock"));
  XLS_ASSERT_OK(VerifyBlock(block));
}

TEST_F(BlockConversionTest, TwoToOneProc) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_dir,
      package.CreateSingleValueChannel("dir", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue dir = pb.Receive(ch_dir);
  BValue a = pb.ReceiveIf(ch_a, dir);
  BValue b = pb.ReceiveIf(ch_b, pb.Not(dir));
  pb.Send(ch_out, pb.Select(dir, {b, a}));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, "the_proc"));

  // Input B selected, input valid and output ready asserted.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(block, {{"dir", 0},
                                                 {"a", 123},
                                                 {"b", 42},
                                                 {"a_vld", 1},
                                                 {"b_vld", 1},
                                                 {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 1), Pair("b_rdy", 1),
                                        Pair("out", 42), Pair("a_rdy", 0))));

  // Input A selected, input valid and output ready asserted.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(block, {{"dir", 1},
                                                 {"a", 123},
                                                 {"b", 42},
                                                 {"a_vld", 1},
                                                 {"b_vld", 0},
                                                 {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 1), Pair("b_rdy", 0),
                                        Pair("out", 123), Pair("a_rdy", 1))));

  // Input A selected, input valid asserted, and output ready *not*
  // asserted. Input ready should be zero.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(block, {{"dir", 1},
                                                 {"a", 123},
                                                 {"b", 42},
                                                 {"a_vld", 1},
                                                 {"b_vld", 1},
                                                 {"out_rdy", 0}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 1), Pair("b_rdy", 0),
                                        Pair("out", 123), Pair("a_rdy", 0))));

  // Input A selected, input valid *not* asserted, and output ready
  // asserted. Output valid should be zero.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(block, {{"dir", 1},
                                                 {"a", 123},
                                                 {"b", 42},
                                                 {"a_vld", 0},
                                                 {"b_vld", 1},
                                                 {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 0), Pair("b_rdy", 0),
                                        Pair("out", 123), Pair("a_rdy", 1))));
}

TEST_F(BlockConversionTest, OneToTwoProc) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_dir,
      package.CreateSingleValueChannel("dir", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a,
      package.CreateStreamingChannel("a", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b,
      package.CreateStreamingChannel("b", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue dir = pb.Receive(ch_dir);
  BValue in = pb.Receive(ch_in);
  pb.SendIf(ch_a, dir, in);
  pb.SendIf(ch_b, pb.Not(dir), in);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, "the_proc"));

  // Output B selected. Input valid and output readies asserted.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(
          block,
          {{"dir", 0}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 1),
                                        Pair("in_rdy", 1), Pair("a_vld", 0),
                                        Pair("b", 123))));

  // Output A selected. Input valid and output readies asserted.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 1), Pair("a_vld", 1),
                                        Pair("b", 123))));

  // Output A selected. Input *not* valid and output readies asserted.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 0}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 1), Pair("a_vld", 0),
                                        Pair("b", 123))));

  // Output A selected. Input valid and output ready *not* asserted.
  EXPECT_THAT(
      BlockInterpreter::RunCombinational(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 0}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 0), Pair("a_vld", 1),
                                        Pair("b", 123))));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
