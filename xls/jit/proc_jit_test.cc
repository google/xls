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

#include "xls/jit/proc_jit.h"

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::ElementsAre;

void EnqueueU32(ChannelQueue* queue, uint32_t data) {
  XLS_CHECK_OK(queue->Enqueue(Value(UBits(data, 32))));
}

uint32_t DequeueU32(ChannelQueue* queue) {
  std::optional<Value> value_opt = queue->Dequeue();
  XLS_CHECK(value_opt.has_value());
  return value_opt->bits().ToUint64().value();
}

class ProcJitTest : public IrTestBase {
 protected:
  // Creates a queue manager and ProcJit for the given proc.
  std::pair<std::unique_ptr<JitChannelQueueManager>, std::unique_ptr<ProcJit>>
  CreateQueueManagerAndJit(Proc* proc) {
    std::unique_ptr<OrcJit> orc_jit = OrcJit::Create().value();
    auto queue_mgr =
        JitChannelQueueManager::CreateThreadSafe(proc->package(), orc_jit.get())
            .value();
    std::unique_ptr<ProcJit> jit =
        ProcJit::Create(proc, queue_mgr.get(), std ::move(orc_jit)).value();
    return {std::move(queue_mgr), std::move(jit)};
  }
};

TEST_F(ProcJitTest, SendOnly) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_out,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tok", package.get());
  pb.Send(channel_out, pb.Literal(UBits(42, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  auto [queue_mgr, jit] = CreateQueueManagerAndJit(proc);

  auto* out_queue = queue_mgr->GetQueueById(0).value();
  EXPECT_TRUE(out_queue->IsEmpty());

  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();
  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_FALSE(out_queue->IsEmpty());
  EXPECT_EQ(DequeueU32(out_queue), 42);
  EXPECT_TRUE(out_queue->IsEmpty());

  XLS_ASSERT_OK(jit->Tick(*continuation));
  XLS_ASSERT_OK(jit->Tick(*continuation));
  XLS_ASSERT_OK(jit->Tick(*continuation));
  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_FALSE(out_queue->IsEmpty());
  EXPECT_EQ(DequeueU32(out_queue), 42);
  EXPECT_EQ(DequeueU32(out_queue), 42);
  EXPECT_EQ(DequeueU32(out_queue), 42);
  EXPECT_EQ(DequeueU32(out_queue), 42);
  EXPECT_TRUE(out_queue->IsEmpty());
}

TEST_F(ProcJitTest, CanCompileProcs) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan c_o(bits[32], id=1, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc the_proc(my_token: token, state: (), init={()}) {
  literal.1: bits[32] = literal(value=3)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(literal.1, tuple_index.4)
  send.6: token = send(tuple_index.3, umul.5, channel_id=1)
  next (send.6, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(kIrText));

  auto [queue_mgr, jit] =
      CreateQueueManagerAndJit(FindProc("the_proc", package.get()));

  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();
  {
    EnqueueU32(queue_mgr->GetQueueById(0).value(), 7);
    XLS_ASSERT_OK(jit->Tick(*continuation));
    EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), 21);
  }

  // Let's make sure we can call it 2x!
  {
    EnqueueU32(queue_mgr->GetQueueById(0).value(), 7);
    XLS_ASSERT_OK(jit->Tick(*continuation));
    EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), 21);
  }
}

TEST_F(ProcJitTest, RecvIf) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none,
metadata="") chan c_o(bits[32], id=1, kind=streaming, ops=send_only,
flow_control=none, metadata="")

proc the_proc(my_token: token, state: bits[1], init={0}) {
  receive.2: (token, bits[32]) = receive(my_token, predicate=state,channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  send.5: token = send(tuple_index.3, tuple_index.4, channel_id=1)
  next_state: bits[1] = not(state)
  next (send.5, next_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(kIrText));

  constexpr uint32_t kQueueData = 0xbeef;

  auto [queue_mgr, jit] =
      CreateQueueManagerAndJit(FindProc("the_proc", package.get()));

  EnqueueU32(queue_mgr->GetQueueById(0).value(), kQueueData);

  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();
  {
    // First: initial state is zero; receive predicate is false. Should produce
    // zero.
    XLS_ASSERT_OK(jit->Tick(*continuation));
    EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), 0);
  }

  {
    // Next: state is now 1; receive predice is true, should produce non-zero
    // data.
    XLS_ASSERT_OK(jit->Tick(*continuation));
    EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), kQueueData);
  }
}

TEST_F(ProcJitTest, ConditionalSend) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none,
metadata="") chan c_o(bits[32], id=1, kind=streaming, ops=send_only,
flow_control=none, metadata="")

proc the_proc(my_token: token, state: bits[1], init={0}) {
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  send.5: token = send(tuple_index.3, tuple_index.4, predicate=state, channel_id=1)
  next_state: bits[1] = not(state)
  next (send.5, next_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(kIrText));

  constexpr uint32_t kQueueData = 0xbeef;
  auto [queue_mgr, jit] =
      CreateQueueManagerAndJit(FindProc("the_proc", package.get()));

  EnqueueU32(queue_mgr->GetQueueById(0).value(), kQueueData);
  EnqueueU32(queue_mgr->GetQueueById(0).value(), kQueueData + 1);

  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();
  {
    // First: with state 0, make sure no send occurred (i.e., our output queue
    // is empty).
    XLS_ASSERT_OK(jit->Tick(*continuation));
    EXPECT_TRUE(queue_mgr->GetQueueById(1).value()->IsEmpty());
  }

  {
    // Second: with state 1, make sure we've now got output data.
    XLS_ASSERT_OK(jit->Tick(*continuation));
    EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), kQueueData + 1);
  }
}

TEST_F(ProcJitTest, SingleValueChannel) {
  const std::string kIrText = R"(
package p

chan c_sv(bits[32], id=0, kind=single_value, ops=receive_only, metadata="")
chan c_i(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=none,
metadata="") chan c_o(bits[32], id=2, kind=streaming, ops=send_only,
flow_control=none, metadata="")

proc the_proc(my_token: token, state: (), init={()}) {
  recv_sv: (token, bits[32]) = receive(my_token, channel_id=0)
  tkn0: token = tuple_index(recv_sv, index=0)
  single_value: bits[32] = tuple_index(recv_sv, index=1)

  recv_streaming: (token, bits[32]) = receive(tkn0, channel_id=1)
  tkn1: token = tuple_index(recv_streaming, index=0)
  streaming_value: bits[32] = tuple_index(recv_streaming, index=1)

  sum: bits[32] = add(single_value, streaming_value)
  tkn2: token = send(tkn1, sum, channel_id=2)
  next (tkn2, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(kIrText));

  auto [queue_mgr, jit] =
      CreateQueueManagerAndJit(FindProc("the_proc", package.get()));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * single_value_input,
                           queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * streaming_input,
                           queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * streaming_output,
                           queue_mgr->GetQueueById(2));

  EnqueueU32(single_value_input, 7);
  EnqueueU32(streaming_input, 42);
  EnqueueU32(streaming_input, 123);

  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();

  XLS_EXPECT_OK(jit->Tick(*continuation));
  XLS_EXPECT_OK(jit->Tick(*continuation));
  EXPECT_EQ(DequeueU32(streaming_output), 49);
  EXPECT_EQ(DequeueU32(streaming_output), 130);

  EnqueueU32(single_value_input, 10);
  EnqueueU32(streaming_input, 42);
  EnqueueU32(streaming_input, 123);

  XLS_EXPECT_OK(jit->Tick(*continuation));
  XLS_EXPECT_OK(jit->Tick(*continuation));
  EXPECT_EQ(DequeueU32(streaming_output), 52);
  EXPECT_EQ(DequeueU32(streaming_output), 133);
}

TEST_F(ProcJitTest, StatelessProc) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_in,
      package->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_out,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  ProcBuilder pb("stateless", /*token_name=*/"tok", package.get());
  BValue receive = pb.Receive(channel_in, pb.GetTokenParam());
  BValue send =
      pb.Send(channel_out, pb.TupleIndex(receive, 0),
              pb.Add(pb.TupleIndex(receive, 1), pb.Literal(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(send, std::vector<BValue>()));

  auto [queue_mgr, jit] = CreateQueueManagerAndJit(proc);

  EnqueueU32(queue_mgr->GetQueueById(0).value(), 7);
  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();
  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), 49);
}

TEST_F(ProcJitTest, MultistateProc) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_in,
      package->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_out,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  // Proc pseudocode:
  //
  // x = 1
  // y = 42
  // while (true):
  //   tmp = rcv(in)
  //   send(out, tmp * x + y)
  //   x += 1
  //   y += 10
  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tok", package.get());
  BValue x = pb.StateElement("x", Value(UBits(1, 32)));
  BValue y = pb.StateElement("y", Value(UBits(42, 32)));
  BValue in = pb.Receive(channel_in);
  pb.Send(channel_out, pb.Add(pb.UMul(in, x), y));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build({pb.Add(x, pb.Literal(UBits(1, 32))),
                                     pb.Add(y, pb.Literal(UBits(10, 32)))}));

  auto [queue_mgr, jit] = CreateQueueManagerAndJit(proc);

  EnqueueU32(queue_mgr->GetQueueById(0).value(), 7);
  EnqueueU32(queue_mgr->GetQueueById(0).value(), 10);
  EnqueueU32(queue_mgr->GetQueueById(0).value(), 14);

  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();
  XLS_ASSERT_OK_AND_ASSIGN(TickResult result, jit->Tick(*continuation));
  EXPECT_THAT(continuation->GetState(),
              ElementsAre(Value(UBits(2, 32)), Value(UBits(52, 32))));

  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_TRUE(continuation->AtStartOfTick());
  EXPECT_THAT(continuation->GetState(),
              ElementsAre(Value(UBits(3, 32)), Value(UBits(62, 32))));

  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_TRUE(continuation->AtStartOfTick());
  EXPECT_THAT(continuation->GetState(),
              ElementsAre(Value(UBits(4, 32)), Value(UBits(72, 32))));

  // 7 * 1 + 42
  EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), 49);

  // 10 * 2 + 52
  EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), 72);

  // 14 * 3 + 62
  EXPECT_EQ(DequeueU32(queue_mgr->GetQueueById(1).value()), 104);

  EXPECT_TRUE(queue_mgr->GetQueueById(1).value()->IsEmpty());
}

TEST_F(ProcJitTest, NonBlockingReceivesProc) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Channel * in0, package->CreateStreamingChannel(
                                              "in0", ChannelOps::kReceiveOnly,
                                              package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * in1, package->CreateStreamingChannel(
                                              "in1", ChannelOps::kReceiveOnly,
                                              package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * in2, package->CreateSingleValueChannel(
                                              "in2", ChannelOps::kReceiveOnly,
                                              package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * out0, package->CreateStreamingChannel(
                                               "out0", ChannelOps::kSendOnly,
                                               package->GetBitsType(32)));

  ProcBuilder pb("nb_recv", /*token_name=*/"tok", package.get());

  BValue in0_data_and_valid = pb.ReceiveNonBlocking(in0, pb.GetTokenParam());
  BValue in1_data_and_valid = pb.ReceiveNonBlocking(in1, pb.GetTokenParam());
  BValue in2_data_and_valid = pb.ReceiveNonBlocking(in2, pb.GetTokenParam());

  BValue sum = pb.Literal(UBits(0, 32));

  BValue in0_tok = pb.TupleIndex(in0_data_and_valid, 0);
  BValue in0_data = pb.TupleIndex(in0_data_and_valid, 1);
  BValue in0_valid = pb.TupleIndex(in0_data_and_valid, 2);
  BValue add_sum_in0 = pb.Add(sum, in0_data);
  BValue sum0 = pb.Select(in0_valid, {sum, add_sum_in0});

  BValue in1_tok = pb.TupleIndex(in1_data_and_valid, 0);
  BValue in1_data = pb.TupleIndex(in1_data_and_valid, 1);
  BValue in1_valid = pb.TupleIndex(in1_data_and_valid, 2);
  BValue add_sum0_in1 = pb.Add(sum0, in1_data);
  BValue sum1 = pb.Select(in1_valid, {sum0, add_sum0_in1});

  BValue in2_tok = pb.TupleIndex(in2_data_and_valid, 0);
  BValue in2_data = pb.TupleIndex(in2_data_and_valid, 1);
  BValue in2_valid = pb.TupleIndex(in2_data_and_valid, 2);
  BValue add_sum1_in2 = pb.Add(sum1, in2_data);
  BValue sum2 = pb.Select(in2_valid, {sum1, add_sum1_in2});

  BValue after_in_tok = pb.AfterAll({in0_tok, in1_tok, in2_tok});
  BValue tok_fin = pb.Send(out0, after_in_tok, sum2);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(tok_fin, {}));
  XLS_VLOG_LINES(3, proc->DumpIr());

  auto [queue_mgr, jit] = CreateQueueManagerAndJit(proc);

  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in0_queue,
                           queue_mgr->GetQueueById(in0->id()));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in1_queue,
                           queue_mgr->GetQueueById(in1->id()));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in2_queue,
                           queue_mgr->GetQueueById(in2->id()));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * out0_queue,
                           queue_mgr->GetQueueById(out0->id()));

  // Initialize the single value queue.
  EnqueueU32(in2_queue, 10);

  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();

  // All other channels are non-blocking, so run even if the queues are empty.
  EXPECT_TRUE(in0_queue->IsEmpty());
  EXPECT_TRUE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_TRUE(out0_queue->IsEmpty());
  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_TRUE(in0_queue->IsEmpty());
  EXPECT_TRUE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_FALSE(out0_queue->IsEmpty());

  EXPECT_EQ(DequeueU32(out0_queue), 10);

  // Run with only in1 (and in2) having data.
  EnqueueU32(in1_queue, 5);

  EXPECT_TRUE(in0_queue->IsEmpty());
  EXPECT_FALSE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_TRUE(out0_queue->IsEmpty());
  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_TRUE(in0_queue->IsEmpty());
  EXPECT_TRUE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_FALSE(out0_queue->IsEmpty());

  EXPECT_EQ(DequeueU32(out0_queue), 15);

  // Run with only in0 (and in2) having data.
  EnqueueU32(in0_queue, 7);

  EXPECT_FALSE(in0_queue->IsEmpty());
  EXPECT_TRUE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_TRUE(out0_queue->IsEmpty());
  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_TRUE(in0_queue->IsEmpty());
  EXPECT_TRUE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_FALSE(out0_queue->IsEmpty());

  EXPECT_EQ(DequeueU32(out0_queue), 17);

  // Run with all channels having data.
  EnqueueU32(in0_queue, 11);
  EnqueueU32(in1_queue, 22);

  EXPECT_FALSE(in0_queue->IsEmpty());
  EXPECT_FALSE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_TRUE(out0_queue->IsEmpty());
  XLS_ASSERT_OK(jit->Tick(*continuation));
  EXPECT_TRUE(in0_queue->IsEmpty());
  EXPECT_TRUE(in1_queue->IsEmpty());
  EXPECT_FALSE(in2_queue->IsEmpty());
  EXPECT_FALSE(out0_queue->IsEmpty());

  EXPECT_EQ(DequeueU32(out0_queue), 43);
}

TEST_F(ProcJitTest, MultipleReceives) {
  auto package = CreatePackage();
  TokenlessProcBuilder pb("prev", /*token_name=*/"tok", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in0,
      package->CreateStreamingChannel("in0", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in1,
      package->CreateStreamingChannel("in1", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in2,
      package->CreateStreamingChannel("in2", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_out, package->CreateStreamingChannel(
                                                 "out", ChannelOps::kSendOnly,
                                                 package->GetBitsType(32)));

  // Build the following proc which returns the sum of ch0 and ch2 inputs. ch2
  // is only read if the input from ch1 is true.
  BValue in0 = pb.Receive(ch_in0);
  BValue in1 = pb.Receive(ch_in1);
  BValue in2 =
      pb.ReceiveIf(ch_in2, /*pred=*/pb.Ne(in1, pb.Literal(UBits(0, 32))));
  pb.Send(ch_out, pb.Add(in0, in2));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  auto [queue_mgr, jit] = CreateQueueManagerAndJit(proc);

  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in0_queue,
                           queue_mgr->GetQueueById(ch_in0->id()));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in1_queue,
                           queue_mgr->GetQueueById(ch_in1->id()));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in2_queue,
                           queue_mgr->GetQueueById(ch_in2->id()));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * output_queue,
                           queue_mgr->GetQueueById(ch_out->id()));

  // Initially should be blocked on in0.
  std::unique_ptr<ProcContinuation> continuation = jit->NewContinuation();
  EXPECT_THAT(jit->Tick(*continuation),
              IsOkAndHolds(TickResult{.tick_complete = false,
                                      .progress_made = true,
                                      .blocked_channel = ch_in0,
                                      .sent_channels = {}}));
  EXPECT_FALSE(continuation->AtStartOfTick());

  EnqueueU32(in0_queue, 10);

  // Then should be blocked on in1.
  EXPECT_THAT(jit->Tick(*continuation),
              IsOkAndHolds(TickResult{.tick_complete = false,
                                      .progress_made = true,
                                      .blocked_channel = ch_in1,
                                      .sent_channels = {}}));
  EXPECT_FALSE(continuation->AtStartOfTick());

  EnqueueU32(in1_queue, 1);

  // Then should be blocked on in2.
  EXPECT_THAT(jit->Tick(*continuation),
              IsOkAndHolds(TickResult{.tick_complete = false,
                                      .progress_made = true,
                                      .blocked_channel = ch_in2,
                                      .sent_channels = {}}));
  EXPECT_FALSE(continuation->AtStartOfTick());

  EnqueueU32(in2_queue, 42);

  // Finally, should run to completion.
  EXPECT_THAT(jit->Tick(*continuation),
              IsOkAndHolds(TickResult{.tick_complete = true,
                                      .progress_made = true,
                                      .blocked_channel = std::nullopt,
                                      .sent_channels = {}}));
  EXPECT_TRUE(continuation->AtStartOfTick());

  EXPECT_THAT(DequeueU32(output_queue), 52);

  // Next, only enqueue data on ch0 and ch1. ch2 should not be read because it's
  // predicate is false.
  EnqueueU32(in0_queue, 123);
  EnqueueU32(in1_queue, 0);
  EXPECT_THAT(jit->Tick(*continuation),
              IsOkAndHolds(TickResult{.tick_complete = true,
                                      .progress_made = true,
                                      .blocked_channel = std::nullopt,
                                      .sent_channels = {}}));
  EXPECT_TRUE(continuation->AtStartOfTick());

  EXPECT_THAT(DequeueU32(output_queue), 123);
}

}  // namespace
}  // namespace xls
