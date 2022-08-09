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

namespace xls {
namespace {

using ::testing::ElementsAre;

void EnqueueData(JitChannelQueue* queue, uint32_t data) {
  queue->Send(absl::bit_cast<uint8_t*>(&data), sizeof(uint32_t));
}

uint32_t DequeueData(JitChannelQueue* queue) {
  uint32_t data;
  queue->Recv(absl::bit_cast<uint8_t*>(&data), sizeof(uint32_t));
  return data;
}

class ProcJitTest : public IrTestBase {};

// Recv/Send functions for the "CanCompileProcs" test.
bool CanCompileProcs_recv(JitChannelQueue* queue_ptr, Receive* recv_ptr,
                          uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  return queue->Recv(data_ptr, data_sz);
}

void CanCompileProcs_send(JitChannelQueue* queue_ptr, Send* send_ptr,
                          uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  queue->Send(data_ptr, data_sz);
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
  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(FindProc("the_proc", package.get()), queue_mgr.get(),
                      CanCompileProcs_recv, CanCompileProcs_send));

  {
    EnqueueData(queue_mgr->GetQueueById(0).value(), 7);
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<std::vector<Value>> result,
                             jit->Run({Value::Tuple({})}));
    EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 21);
  }

  // Let's make sure we can call it 2x!
  {
    EnqueueData(queue_mgr->GetQueueById(0).value(), 7);
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<std::vector<Value>> result,
                             jit->Run({Value::Tuple({})}));
    EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 21);
  }
}

TEST_F(ProcJitTest, RecvIf) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none,
metadata="") chan c_o(bits[32], id=1, kind=streaming, ops=send_only,
flow_control=none, metadata="")

proc the_proc(my_token: token, state: bits[1], init={0}) {
  receive.2: (token, bits[32]) = receive(my_token, predicate=state,
  channel_id=0) tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  send.5: token = send(tuple_index.3, tuple_index.4, channel_id=1)
  next (send.5, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(kIrText));

  constexpr uint32_t kQueueData = 0xbeef;
  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(FindProc("the_proc", package.get()), queue_mgr.get(),
                      CanCompileProcs_recv, CanCompileProcs_send));

  EnqueueData(queue_mgr->GetQueueById(0).value(), kQueueData);

  {
    // First: set state to 0; see that recv_if returns 0.
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<std::vector<Value>> result,
                             jit->Run({Value(UBits(0, 1))}));
    EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 0);
  }

  {
    // First: set state to 0; see that recv_if returns 0.
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<std::vector<Value>> result,
                             jit->Run({Value(UBits(1, 1))}));
    EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), kQueueData);
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
  next (send.5, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(kIrText));

  constexpr uint32_t kQueueData = 0xbeef;
  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(FindProc("the_proc", package.get()), queue_mgr.get(),
                      CanCompileProcs_recv, CanCompileProcs_send));

  EnqueueData(queue_mgr->GetQueueById(0).value(), kQueueData);
  EnqueueData(queue_mgr->GetQueueById(0).value(), kQueueData + 1);

  {
    // First: with state 0, make sure no send occurred (i.e., our output queue
    // is empty).
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<std::vector<Value>> result,
                             jit->Run({Value(UBits(0, 1))}));
    EXPECT_THAT(result.value, ElementsAre(Value(UBits(0, 1))));
    EXPECT_TRUE(queue_mgr->GetQueueById(1).value()->Empty());
  }

  {
    // Second: with state 1, make sure we've now got output data.
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<std::vector<Value>> result,
                             jit->Run({Value(UBits(1, 1))}));
    EXPECT_THAT(result.value, ElementsAre(Value(UBits(1, 1))));
    EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), kQueueData + 1);
  }
}

// Recv/Send functions for the "GetsUserData" test.
bool GetsUserData_recv(JitChannelQueue* queue_ptr, Receive* recv_ptr,
                       uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  uint64_t* int_data = absl::bit_cast<uint64_t*>(user_data);
  *int_data = *int_data * 2;
  return queue->Recv(data_ptr, data_sz);
}

void GetsUserData_send(JitChannelQueue* queue_ptr, Send* send_ptr,
                       uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  uint64_t* int_data = absl::bit_cast<uint64_t*>(user_data);
  *int_data = *int_data * 3;
  queue->Send(data_ptr, data_sz);
}

// Verifies that the "user data" pointer is properly passed into proc callbacks.
TEST_F(ProcJitTest, GetsUserData) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none,
metadata="") chan c_o(bits[32], id=1, kind=streaming, ops=send_only,
flow_control=none, metadata="")

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

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(FindProc("the_proc", package.get()), queue_mgr.get(),
                      GetsUserData_recv, GetsUserData_send));

  EnqueueData(queue_mgr->GetQueueById(0).value(), 7);

  {
    uint64_t user_data = 7;
    XLS_ASSERT_OK_AND_ASSIGN(
        InterpreterResult<std::vector<Value>> result,
        jit->Run({Value::Tuple({})}, absl::bit_cast<void*>(&user_data)));
    EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 21);
    EXPECT_EQ(user_data, 7 * 2 * 3);
  }

  {
    // Let's make sure we can call it 2x!
    uint64_t user_data = 7;
    EnqueueData(queue_mgr->GetQueueById(0).value(), 7);
    XLS_ASSERT_OK_AND_ASSIGN(
        InterpreterResult<std::vector<Value>> result,
        jit->Run({Value::Tuple({})}, absl::bit_cast<void*>(&user_data)));
    EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 21);
    EXPECT_EQ(user_data, 7 * 2 * 3);
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

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(FindProc("the_proc", package.get()), queue_mgr.get(),
                      CanCompileProcs_recv, CanCompileProcs_send));

  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueue * single_value_input,
                           queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueue * streaming_input,
                           queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueue * streaming_output,
                           queue_mgr->GetQueueById(2));

  EnqueueData(single_value_input, 7);
  EnqueueData(streaming_input, 42);
  EnqueueData(streaming_input, 123);

  auto tick = [&]() { XLS_EXPECT_OK(jit->Run({Value::Tuple({})}).status()); };

  tick();
  tick();
  EXPECT_EQ(DequeueData(streaming_output), 49);
  EXPECT_EQ(DequeueData(streaming_output), 130);

  EnqueueData(single_value_input, 10);
  EnqueueData(streaming_input, 42);
  EnqueueData(streaming_input, 123);

  tick();
  tick();
  EXPECT_EQ(DequeueData(streaming_output), 52);
  EXPECT_EQ(DequeueData(streaming_output), 133);
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

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));

  ProcBuilder pb("stateless", /*token_name=*/"tok", package.get());
  BValue receive = pb.Receive(channel_in, pb.GetTokenParam());
  BValue send =
      pb.Send(channel_out, pb.TupleIndex(receive, 0),
              pb.Add(pb.TupleIndex(receive, 1), pb.Literal(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(send, std::vector<BValue>()));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(proc, queue_mgr.get(), CanCompileProcs_recv,
                      CanCompileProcs_send));

  EnqueueData(queue_mgr->GetQueueById(0).value(), 7);
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<std::vector<Value>> result,
                           jit->Run({Value::Tuple({})}));
  EXPECT_TRUE(result.value.empty());
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 49);
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

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));

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

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(proc, queue_mgr.get(), CanCompileProcs_recv,
                      CanCompileProcs_send));

  EnqueueData(queue_mgr->GetQueueById(0).value(), 7);
  EnqueueData(queue_mgr->GetQueueById(0).value(), 10);
  EnqueueData(queue_mgr->GetQueueById(0).value(), 14);

  std::vector<Value> state = {Value(UBits(1, 32)), Value(UBits(42, 32))};
  InterpreterResult<std::vector<Value>> result;
  XLS_ASSERT_OK_AND_ASSIGN(result, jit->Run(state));
  EXPECT_THAT(result.value,
              ElementsAre(Value(UBits(2, 32)), Value(UBits(52, 32))));
  state = result.value;

  XLS_ASSERT_OK_AND_ASSIGN(result, jit->Run(state));
  EXPECT_THAT(result.value,
              ElementsAre(Value(UBits(3, 32)), Value(UBits(62, 32))));
  state = result.value;

  XLS_ASSERT_OK_AND_ASSIGN(result, jit->Run(state));
  EXPECT_THAT(result.value,
              ElementsAre(Value(UBits(4, 32)), Value(UBits(72, 32))));

  // 7 * 1 + 42
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 49);

  // 10 * 2 + 52
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 72);

  // 14 * 3 + 62
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 104);

  EXPECT_TRUE(queue_mgr->GetQueueById(1).value()->Empty());
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

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(proc, queue_mgr.get(), CanCompileProcs_recv,
                      CanCompileProcs_send));

  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueue * in0_queue,
                           queue_mgr->GetQueueById(in0->id()));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueue * in1_queue,
                           queue_mgr->GetQueueById(in1->id()));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueue * in2_queue,
                           queue_mgr->GetQueueById(in2->id()));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueue * out0_queue,
                           queue_mgr->GetQueueById(out0->id()));

  // Initialize the single value queue.
  EnqueueData(in2_queue, 10);

  // All other channels are non-blocking, so run even if the queues are empty.
  EXPECT_TRUE(in0_queue->Empty());
  EXPECT_TRUE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_TRUE(out0_queue->Empty());
  XLS_ASSERT_OK(jit->Run({}));
  EXPECT_TRUE(in0_queue->Empty());
  EXPECT_TRUE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_FALSE(out0_queue->Empty());

  EXPECT_EQ(DequeueData(out0_queue), 10);

  // Run with only in1 (and in2) having data.
  EnqueueData(in1_queue, 5);

  EXPECT_TRUE(in0_queue->Empty());
  EXPECT_FALSE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_TRUE(out0_queue->Empty());
  XLS_ASSERT_OK(jit->Run({}));
  EXPECT_TRUE(in0_queue->Empty());
  EXPECT_TRUE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_FALSE(out0_queue->Empty());

  EXPECT_EQ(DequeueData(out0_queue), 15);

  // Run with only in0 (and in2) having data.
  EnqueueData(in0_queue, 7);

  EXPECT_FALSE(in0_queue->Empty());
  EXPECT_TRUE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_TRUE(out0_queue->Empty());
  XLS_ASSERT_OK(jit->Run({}));
  EXPECT_TRUE(in0_queue->Empty());
  EXPECT_TRUE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_FALSE(out0_queue->Empty());

  EXPECT_EQ(DequeueData(out0_queue), 17);

  // Run with all channels having data.
  EnqueueData(in0_queue, 11);
  EnqueueData(in1_queue, 22);

  EXPECT_FALSE(in0_queue->Empty());
  EXPECT_FALSE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_TRUE(out0_queue->Empty());
  XLS_ASSERT_OK(jit->Run({}));
  EXPECT_TRUE(in0_queue->Empty());
  EXPECT_TRUE(in1_queue->Empty());
  EXPECT_FALSE(in2_queue->Empty());
  EXPECT_FALSE(out0_queue->Empty());

  EXPECT_EQ(DequeueData(out0_queue), 43);
}

}  // namespace
}  // namespace xls
