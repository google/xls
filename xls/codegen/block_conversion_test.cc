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

TEST_F(BlockConversionTest, SimpleProc) {
  const std::string ir_text = R"(package test

chan in(bits[32], id=0, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 1 }""")
chan out(bits[32], id=1, kind=single_value, ops=send_only,
         metadata="""module_port { flopped: false,  port_order: 0 }""")

proc my_proc(my_token: token, my_state: (), init=()) {
  rcv: (token, bits[32]) = receive(my_token, channel_id=0)
  data: bits[32] = tuple_index(rcv, index=1)
  negate: bits[32] = neg(data)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, negate, channel_id=1)
  next (send, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, TestName()));
  EXPECT_THAT(FindNode("out", block),
              m::OutputPort("out", m::Neg(m::InputPort("in"))));
}

TEST_F(BlockConversionTest, ProcWithMultipleInputChannels) {
  const std::string ir_text = R"(package test

chan in0(bits[32], id=0, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 0 }""")
chan in1(bits[32], id=1, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 2 }""")
chan in2(bits[32], id=2, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 1 }""")
chan out(bits[32], id=3, kind=single_value, ops=send_only,
         metadata="""module_port { flopped: false,  port_order: 0 }""")

proc my_proc(my_token: token, my_state: (), init=()) {
  rcv0: (token, bits[32]) = receive(my_token, channel_id=0)
  rcv0_token: token = tuple_index(rcv0, index=0)
  rcv1: (token, bits[32]) = receive(rcv0_token, channel_id=1)
  rcv1_token: token = tuple_index(rcv1, index=0)
  rcv2: (token, bits[32]) = receive(rcv1_token, channel_id=2)
  rcv2_token: token = tuple_index(rcv2, index=0)
  data0: bits[32] = tuple_index(rcv0, index=1)
  data1: bits[32] = tuple_index(rcv1, index=1)
  data2: bits[32] = tuple_index(rcv2, index=1)
  neg_data1: bits[32] = neg(data1)
  two: bits[32] = literal(value=2)
  data2_times_two: bits[32] = umul(data2, two)
  tmp: bits[32] = add(neg_data1, data2_times_two)
  sum: bits[32] = add(tmp, data0)
  send: token = send(rcv2_token, sum, channel_id=3)
  next (send, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, TestName()));
  EXPECT_THAT(
      FindNode("out", block),
      m::OutputPort("out",
                    m::Add(m::Add(m::Neg(m::InputPort("in1")),
                                  m::UMul(m::InputPort("in2"), m::Literal(2))),
                           m::InputPort("in0"))));
}

TEST_F(BlockConversionTest, OnlyFIFOOutProc) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=single_value, ops=receive_only, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

proc my_proc(tkn: token, st: (), init=()) {
  receive.13: (token, bits[32]) = receive(tkn, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  literal.21: bits[1] = literal(value=1, id=21, pos=1,8,3)
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15, predicate=literal.21, channel_id=1, id=20, pos=1,5,1)
  next (send.20, st)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, TestName()));
  EXPECT_THAT(FindNode("out", block), m::OutputPort("out", m::InputPort("in")));
  EXPECT_THAT(FindNode("out_vld", block),
              m::OutputPort("out_vld", m::And(m::Literal(1), m::Literal(1))));
}

TEST_F(BlockConversionTest, OnlyFIFOInProc) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only,
flow_control=ready_valid, metadata="""module_port { flopped: false
port_order: 0 }""") chan out(bits[32], id=1, kind=single_value,
ops=send_only, metadata="""module_port { flopped: false port_order: 1 }""")

proc my_proc(tkn: token, st: (), init=()) {
  literal.21: bits[1] = literal(value=1, id=21, pos=1,8,3)
  receive.13: (token, bits[32]) = receive(tkn, predicate=literal.21, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15,
  channel_id=1, id=20, pos=1,5,1) next (send.20, st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, TestName()));

  EXPECT_THAT(FindNode("out", block), m::OutputPort("out", m::InputPort("in")));
  EXPECT_THAT(FindNode("in", block), m::InputPort("in"));
  EXPECT_THAT(FindNode("in_vld", block), m::InputPort("in_vld"));
  EXPECT_THAT(FindNode("in_rdy", block),
              m::OutputPort("in_rdy", m::And(m::Literal(1), m::Literal(1))));
}

TEST_F(BlockConversionTest, UnconditionalSendRdyVldProc) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=single_value, ops=receive_only, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

proc my_proc(tkn: token, st: (), init=()) {
  receive.13: (token, bits[32]) = receive(tkn, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15, channel_id=1, id=20) next (send.20, st)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, TestName()));

  EXPECT_THAT(FindNode("out", block), m::OutputPort("out", m::InputPort("in")));
  EXPECT_THAT(FindNode("out_vld", block),
              m::OutputPort("out_vld", m::Literal(1)));
  EXPECT_THAT(FindNode("out_rdy", block), m::InputPort("out_rdy"));
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
      InterpretCombinationalBlock(block, {{"dir", 0},
                                          {"a", 123},
                                          {"b", 42},
                                          {"a_vld", 1},
                                          {"b_vld", 1},
                                          {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 1), Pair("b_rdy", 1),
                                        Pair("out", 42), Pair("a_rdy", 0))));

  // Input A selected, input valid and output ready asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(block, {{"dir", 1},
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
      InterpretCombinationalBlock(block, {{"dir", 1},
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
      InterpretCombinationalBlock(block, {{"dir", 1},
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
      InterpretCombinationalBlock(
          block,
          {{"dir", 0}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 1),
                                        Pair("in_rdy", 1), Pair("a_vld", 0),
                                        Pair("b", 123))));

  // Output A selected. Input valid and output readies asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 1), Pair("a_vld", 1),
                                        Pair("b", 123))));

  // Output A selected. Input *not* valid and output readies asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 0}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 1), Pair("a_vld", 0),
                                        Pair("b", 123))));

  // Output A selected. Input valid and output ready *not* asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 0}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 0), Pair("a_vld", 1),
                                        Pair("b", 123))));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
