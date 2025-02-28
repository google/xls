// Copyright 2020 The XLS Authors
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

#include "xls/passes/dfe_pass.h"

#include <memory>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

namespace m = xls::op_matchers;

class DeadFunctionEliminationPassTest : public IrTestBase {
 protected:
  DeadFunctionEliminationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return DeadFunctionEliminationPass().Run(p, OptimizationPassOptions(),
                                             &results, context);
  }

  absl::StatusOr<Function*> MakeFunction(std::string_view name, Package* p) {
    FunctionBuilder fb(name, p);
    fb.Param("arg", p->GetBitsType(32));
    return fb.Build();
  }

  absl::StatusOr<Proc*> CreateNewStyleAccumProc(std::string_view proc_name,
                                                Package* package) {
    TokenlessProcBuilder pb(NewStyleProc(), proc_name, "tkn", package);
    BValue accum = pb.StateElement("accum", Value(UBits(0, 32)));
    XLS_ASSIGN_OR_RETURN(ReceiveChannelReference * in_channel,
                         pb.AddInputChannel("in_ch", package->GetBitsType(32)));
    BValue input = pb.Receive(in_channel);
    BValue next_accum = pb.Add(accum, input);
    XLS_ASSIGN_OR_RETURN(
        SendChannelReference * out_channel,
        pb.AddOutputChannel("out_ch", package->GetBitsType(32)));
    pb.Send(out_channel, next_accum);
    return pb.Build({next_accum});
  }
};

TEST_F(DeadFunctionEliminationPassTest, NoDeadFunctions) {
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * b, MakeFunction("b", p.get()));
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, b));
  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("the_entry"));

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, OneDeadFunction) {
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK(MakeFunction("dead", p.get()).status());
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, a));
  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("the_entry"));

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(p->functions().size(), 2);
}

TEST_F(DeadFunctionEliminationPassTest, OneDeadFunctionButNoEntry) {
  // If no entry function is specified, then DFS cannot happen as all functions
  // are live.
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK(MakeFunction("dead", p.get()).status());
  FunctionBuilder fb("blah", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, a));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, ProcCallingFunction) {
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           MakeFunction("called_by_proc", p.get()));
  XLS_ASSERT_OK(MakeFunction("not_called_by_proc", p.get()).status());

  TokenlessProcBuilder b(TestName(), "tkn", p.get());
  b.StateElement("st", Value(UBits(0, 32)));
  BValue invoke = b.Invoke({b.GetStateParam(0)}, f);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({invoke}));
  XLS_ASSERT_OK(p->SetTop(proc));

  EXPECT_EQ(p->functions().size(), 2);
  XLS_EXPECT_OK(p->GetFunction("not_called_by_proc").status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(p->functions().size(), 1);
  EXPECT_THAT(p->GetFunction("not_called_by_proc"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DeadFunctionEliminationPassTest, MultipleProcs) {
  auto p = std::make_unique<Package>(TestName());
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in_a,
      p->CreateStreamingChannel("in_a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out_a,
      p->CreateStreamingChannel("out_a", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in_c,
      p->CreateStreamingChannel("in_c", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out_c,
      p->CreateStreamingChannel("out_c", ChannelOps::kSendReceive, u32));

  {
    TokenlessProcBuilder b("A", "tkn", p.get());
    b.Send(ch_a_to_b, b.Receive(ch_in_a));
    b.Send(ch_out_a, b.Receive(ch_b_to_a));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * a, b.Build({}));
    XLS_ASSERT_OK(p->SetTop(a));
  }
  {
    TokenlessProcBuilder b("B", "tkn", p.get());
    b.Send(ch_b_to_a, b.Receive(ch_a_to_b));
    XLS_ASSERT_OK(b.Build({}).status());
  }
  {
    TokenlessProcBuilder b("C", "tkn", p.get());
    b.Send(ch_out_c, b.Receive(ch_in_c));
    XLS_ASSERT_OK(b.Build({}).status());
  }
  {
    TokenlessProcBuilder b("D", "tkn", p.get());
    b.Send(ch_in_c, b.Receive(ch_out_c));
    XLS_ASSERT_OK(b.Build({}).status());
  }

  // Procs "C" and "D" should be removed as well as their channels.
  EXPECT_EQ(p->procs().size(), 4);
  XLS_EXPECT_OK(p->GetProc("C").status());
  XLS_EXPECT_OK(p->GetProc("D").status());
  EXPECT_EQ(p->channels().size(), 6);
  XLS_EXPECT_OK(p->GetChannel("in_c").status());
  XLS_EXPECT_OK(p->GetChannel("out_c").status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->procs(), UnorderedElementsAre(m::Proc("A"), m::Proc("B")));
  EXPECT_THAT(p->channels(), Not(AnyOf(Contains(m::Channel("in_c")),
                                       Contains(m::Channel("out_c")))));
}

TEST_F(DeadFunctionEliminationPassTest, MapAndCountedFor) {
  // If no entry function is specified, then DFS cannot happen as all functions
  // are live.
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  Function* body;
  {
    FunctionBuilder fb("jesse_the_loop_body", p.get());
    fb.Param("i", p->GetBitsType(32));
    fb.Param("arg", p->GetBitsType(32));
    fb.Literal(UBits(123, 32));
    XLS_ASSERT_OK_AND_ASSIGN(body, fb.Build());
  }
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue ar = fb.Param("ar", p->GetArrayType(42, p->GetBitsType(32)));
  BValue mapped_ar = fb.Map(ar, a);
  BValue for_loop = fb.CountedFor(x, /*trip_count=*/42, /*stride=*/1, body);
  fb.Tuple({mapped_ar, for_loop});

  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("the_entry"));

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, MapAndDynamicCountedFor) {
  // If no entry function is specified, then DFS cannot happen as all functions
  // are live.
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  Function* body;
  {
    FunctionBuilder fb("jesse_the_loop_body", p.get());
    fb.Param("i", p->GetBitsType(32));
    fb.Param("arg", p->GetBitsType(32));
    fb.Literal(UBits(123, 32));
    XLS_ASSERT_OK_AND_ASSIGN(body, fb.Build());
  }
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue ar = fb.Param("ar", p->GetArrayType(42, p->GetBitsType(32)));
  BValue mapped_ar = fb.Map(ar, a);
  BValue for_loop =
      fb.DynamicCountedFor(x, /*trip_count=*/fb.Literal(UBits(42, 10)),
                           /*stride=*/fb.Literal(UBits(1, 10)), body);
  fb.Tuple({mapped_ar, for_loop});

  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("the_entry"));

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, BlockWithInstantiation) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  auto build_subblock = [&](std::string_view name) -> absl::StatusOr<Block*> {
    BlockBuilder bb(name, p.get());
    bb.OutputPort("out", bb.InputPort("in", u32));
    return bb.Build();
  };

  XLS_ASSERT_OK_AND_ASSIGN(Block * used_subblock,
                           build_subblock("used_subblock"));
  XLS_ASSERT_OK(build_subblock("unused_subblock").status());

  BlockBuilder bb("my_block", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * instantiation,
      bb.block()->AddBlockInstantiation("inst", used_subblock));
  BValue in = bb.InputPort("in0", u32);
  bb.InstantiationInput(instantiation, "in", in);
  BValue inst_out = bb.InstantiationOutput(instantiation, "out");
  bb.OutputPort("out", inst_out);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, bb.Build());
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_EXPECT_OK(p->GetBlock("unused_subblock").status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(p->blocks().size(), 2);
  EXPECT_THAT(p->GetBlock("unused_subblock"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DeadFunctionEliminationPassTest, ProcsUsingExternalChannels) {
  // The following IR shows procs 'connected' only by their communication on
  // external channels. test_proc0 and test_proc1 use channel a, and test_proc1
  // and test_proc2 both use channel b. test_proc2 also calls a function
  // (negate). All of these should *not* be removed by DFE. test_proc3, however,
  // is the only proc that doesn't use an external channel (channel d is
  // internal), so it should be removed by DFE.
  constexpr std::string_view ir_text = R"(package text
chan a(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan b(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan c(bits[32], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan d(bits[32], id=3, kind=streaming, ops=send_receive, flow_control=ready_valid)

top proc test_proc0(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=a)
  next (state)
}

proc test_proc1(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=a)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  send_token: token = send(rcv_token, rcv_data, channel=b)
  next (state)
}

fn negate(in: bits[32]) -> bits[32] {
  ret negate: bits[32] = neg(in)
}

proc test_proc2(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=c)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  send_data: bits[32] = invoke(rcv_data, to_apply=negate)
  send_token: token = send(rcv_token, rcv_data, channel=b)
  next (state)
}

proc test_proc3(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=d)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  send_token: token = send(rcv_token, rcv_data, channel=d)
  next (state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(ir_text));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->procs(),
              UnorderedElementsAre(m::Proc("test_proc0"), m::Proc("test_proc1"),
                                   m::Proc("test_proc2")));

  EXPECT_THAT(
      p->channels(),
      UnorderedElementsAre(m::Channel("a"), m::Channel("b"), m::Channel("c")));

  EXPECT_THAT(p->functions(), UnorderedElementsAre(m::Function("negate")));
}

TEST_F(DeadFunctionEliminationPassTest, TopProcWithNoChannelsWork) {
  // All the other procs are talking over internal channels and are therefore
  // not observable and can be removed.
  constexpr std::string_view ir_text = R"(package text
chan a(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=ready_valid)
chan b(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=ready_valid)

top proc test_proc0(state:(), init={()}) {
  next (state)
}

proc test_proc1(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=a)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  send_token: token = send(rcv_token, rcv_data, channel=b)
  next (state)
}

fn negate(in: bits[32]) -> bits[32] {
  ret negate: bits[32] = neg(in)
}

proc test_proc2(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=b)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  send_data: bits[32] = invoke(rcv_data, to_apply=negate)
  send_token: token = send(rcv_token, rcv_data, channel=a)
  next (state)
}

proc test_proc3(state:(), init={()}) {
  tkn: token = literal(value=token)
  literal0: bits[32] = literal(value=0)
  send_token: token = send(tkn, literal0, channel=a)
  next (state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(ir_text));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->GetFunctionBases(),
              UnorderedElementsAre(m::Proc("test_proc0")));
  EXPECT_THAT(p->channels(), IsEmpty());
}

TEST_F(DeadFunctionEliminationPassTest, ProcsWithTopFnRemovesAllProcs) {
  constexpr std::string_view ir_text = R"(package text
chan a(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan b(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan c(bits[32], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan d(bits[32], id=3, kind=streaming, ops=send_only, flow_control=ready_valid)

proc test_proc0(state:(), init={()}) {
  next (state)
}

proc test_proc1(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=a)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  send_token: token = send(rcv_token, rcv_data, channel=b)
  next (state)
}

top fn negate(in: bits[32]) -> bits[32] {
  ret negate: bits[32] = neg(in)
}

proc test_proc2(state:(), init={()}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=c)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  send_data: bits[32] = invoke(rcv_data, to_apply=negate)
  send_token: token = send(rcv_token, rcv_data, channel=b)
  next (state)
}

proc test_proc3(state:(), init={()}) {
  tkn: token = literal(value=token)
  literal0: bits[32] = literal(value=0)
  send_token: token = send(tkn, literal0, channel=d)
  next (state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(ir_text));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->GetFunctionBases(),
              UnorderedElementsAre(m::Function("negate")));
  EXPECT_THAT(p->channels(), IsEmpty());
}

TEST_F(DeadFunctionEliminationPassTest, SingleNewStyleProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK(CreateNewStyleAccumProc("my_proc", p.get()).status());

  EXPECT_THAT(p->GetFunctionBases(), UnorderedElementsAre(m::Proc("my_proc")));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_THAT(p->GetFunctionBases(), UnorderedElementsAre(m::Proc("my_proc")));
}

TEST_F(DeadFunctionEliminationPassTest, MultipleNewStyleProcs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK(CreateNewStyleAccumProc("my_proc1", p.get()).status());
  XLS_ASSERT_OK(CreateNewStyleAccumProc("my_proc2", p.get()).status());
  XLS_ASSERT_OK(CreateNewStyleAccumProc("my_proc3", p.get()).status());
  XLS_ASSERT_OK(p->SetTopByName("my_proc2"));

  EXPECT_THAT(p->GetFunctionBases(),
              UnorderedElementsAre(m::Proc("my_proc1"), m::Proc("my_proc2"),
                                   m::Proc("my_proc3")));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->GetFunctionBases(), UnorderedElementsAre(m::Proc("my_proc2")));
}

TEST_F(DeadFunctionEliminationPassTest, NewStyleProcWithInstantiations) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Proc * my_proc1,
                           CreateNewStyleAccumProc("my_proc1", p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * my_proc2,
                           CreateNewStyleAccumProc("my_proc2", p.get()));
  XLS_ASSERT_OK(CreateNewStyleAccumProc("my_proc3", p.get()).status());

  TokenlessProcBuilder pb(NewStyleProc(), "top_proc", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(ChannelReferences the_channel,
                           pb.AddChannel("the_channel", p->GetBitsType(32)));
  XLS_ASSERT_OK(
      pb.InstantiateProc("inst0", my_proc1,
                         std::vector<ChannelReference*>{the_channel.receive_ref,
                                                        the_channel.send_ref}));
  XLS_ASSERT_OK(
      pb.InstantiateProc("inst1", my_proc2,
                         std::vector<ChannelReference*>{the_channel.receive_ref,
                                                        the_channel.send_ref}));
  XLS_ASSERT_OK(pb.Build({}).status());

  XLS_ASSERT_OK(p->SetTopByName("top_proc"));

  EXPECT_THAT(p->GetFunctionBases(),
              UnorderedElementsAre(m::Proc("top_proc"), m::Proc("my_proc1"),
                                   m::Proc("my_proc2"), m::Proc("my_proc3")));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->GetFunctionBases(),
              UnorderedElementsAre(m::Proc("top_proc"), m::Proc("my_proc1"),
                                   m::Proc("my_proc2")));
}

}  // namespace
}  // namespace xls
