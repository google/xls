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
#include "xls/jit/serial_proc_runtime.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

template <typename T>
void EnqueueData(JitChannelQueue* queue, T data) {
  queue->Send(absl::bit_cast<uint8_t*>(&data), sizeof(T));
}

template <typename T>
T DequeueData(JitChannelQueue* queue) {
  T data;
  queue->Recv(absl::bit_cast<uint8_t*>(&data), sizeof(T));
  return data;
}

// This test verifies functionality of a simple X -> A -> B -> Y network without
// internal state. Passes a constant into two procs, with the result that the
// input should be multiplied by 6.
TEST(SerialProcRuntimeTest, SimpleNetwork) {
  constexpr int kNumCycles = 4;
  const std::string kIrText = R"(
package p

chan a_in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan a_to_b(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan b_out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc a(my_token: token, state: (), init=()) {
  literal.1: bits[32] = literal(value=2)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(literal.1, tuple_index.4)
  send.6: token = send(tuple_index.3, umul.5, channel_id=1)
  next (send.6, state)
}

proc b(my_token: token, state: (), init=()) {
  literal.100: bits[32] = literal(value=3)
  receive.200: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.300: token = tuple_index(receive.200, index=0)
  tuple_index.400: bits[32] = tuple_index(receive.200, index=1)
  umul.500: bits[32] = umul(literal.100, tuple_index.400)
  send.600: token = send(tuple_index.300, umul.500, channel_id=2)
  next (send.600, state)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  auto queue_mgr = runtime->queue_mgr();
  XLS_ASSERT_OK_AND_ASSIGN(auto input_queue, queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(auto internal_queue, queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(auto output_queue, queue_mgr->GetQueueById(2));

  // Prepopulate the non-output queues.
  for (int i = 0; i < kNumCycles; i++) {
    EnqueueData(input_queue, i);
  }

  int dummy = 0;
  EnqueueData(internal_queue, dummy);

  // Run the runtime for those four cycles...
  for (int i = 0; i < kNumCycles; i++) {
    XLS_ASSERT_OK(runtime->Tick());
  }

  // Then verify the output queue contains the right info. We drop one output,
  // since "b" doesn't get the actual input data until cycle 1.
  DequeueData<int>(output_queue);
  for (int i = 0; i < kNumCycles - 1; i++) {
    int result = DequeueData<int>(output_queue);
    EXPECT_EQ(result, i * 6);
  }
}

// Test verifies that an "X"-shaped network can be modeled correctly, i.e.,
// a network that looks like:
//  A   B
//   \ /
//    C
//   / \
//  D   E
//
// Where A and B receive inputs from "outside", and D and E produce outputs.
TEST(SerialProcRuntimeTest, XNetwork) {
  constexpr int kNumCycles = 32;
  const std::string kIrText = R"(
package p

chan i_a(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan i_b(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan a_c(bits[32], id=2, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan b_c(bits[32], id=3, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan c_d(bits[32], id=4, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan c_e(bits[32], id=5, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan d_o(bits[32], id=6, kind=streaming, ops=send_only, flow_control=none, metadata="")
chan e_o(bits[32], id=7, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc a(my_token: token, state: (), init=()) {
  literal.1: bits[32] = literal(value=1)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(literal.1, tuple_index.4)
  send.6: token = send(tuple_index.3, umul.5, channel_id=2)
  next (send.6, state)
}

proc b(my_token: token, state: (), init=()) {
  literal.101: bits[32] = literal(value=2)
  receive.102: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.103: token = tuple_index(receive.102, index=0)
  tuple_index.104: bits[32] = tuple_index(receive.102, index=1)
  umul.105: bits[32] = umul(literal.101, tuple_index.104)
  send.106: token = send(tuple_index.103, umul.105, channel_id=3)
  next (send.106, state)
}

proc c(my_token: token, state: (), init=()) {
  literal.201: bits[32] = literal(value=3)
  receive.202: (token, bits[32]) = receive(my_token, channel_id=2)
  tuple_index.203: token = tuple_index(receive.202, index=0)
  tuple_index.204: bits[32] = tuple_index(receive.202, index=1)
  receive.205: (token, bits[32]) = receive(tuple_index.203, channel_id=3)
  tuple_index.206: token = tuple_index(receive.205, index=0)
  tuple_index.207: bits[32] = tuple_index(receive.205, index=1)
  umul.208: bits[32] = umul(literal.201, tuple_index.204)
  umul.209: bits[32] = umul(literal.201, tuple_index.207)
  send.210: token = send(tuple_index.206, umul.208, channel_id=4)
  send.211: token = send(send.210, umul.209, channel_id=5)
  next (send.211, state)
}

proc d(my_token: token, state: (), init=()) {
  literal.301: bits[32] = literal(value=4)
  receive.302: (token, bits[32]) = receive(my_token, channel_id=4)
  tuple_index.303: token = tuple_index(receive.302, index=0)
  tuple_index.304: bits[32] = tuple_index(receive.302, index=1)
  umul.305: bits[32] = umul(literal.301, tuple_index.304)
  send.306: token = send(tuple_index.303, umul.305, channel_id=6)
  next (send.306, state)
}

proc e(my_token: token, state: (), init=()) {
  literal.401: bits[32] = literal(value=5)
  receive.402: (token, bits[32]) = receive(my_token, channel_id=5)
  tuple_index.403: token = tuple_index(receive.402, index=0)
  tuple_index.404: bits[32] = tuple_index(receive.402, index=1)
  umul.405: bits[32] = umul(literal.401, tuple_index.404)
  send.406: token = send(tuple_index.403, umul.405, channel_id=7)
  next (send.406, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  auto queue_mgr = runtime->queue_mgr();
  XLS_ASSERT_OK_AND_ASSIGN(auto i_a, queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(auto i_b, queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(auto d_o, queue_mgr->GetQueueById(6));
  XLS_ASSERT_OK_AND_ASSIGN(auto e_o, queue_mgr->GetQueueById(7));

  // "Prime" the internal queues with garbage data (since it'll be one or two
  // cycles until they get real data).

  for (int i = 0; i < kNumCycles; i++) {
    EnqueueData(i_a, i);
    EnqueueData(i_b, i + 10);
    XLS_ASSERT_OK(runtime->Tick());
  }

  // Now, cut out the garbage data from the output queues, and then verify their
  // contents.
  for (int i = 0; i < kNumCycles - 2; i++) {
    int result = DequeueData<int>(d_o);
    ASSERT_EQ(result, i * 1 * 3 * 4);

    result = DequeueData<int>(e_o);
    ASSERT_EQ(result, (i + 10) * 2 * 3 * 5);
  }
}

// This test verify that state is indeed carried correctly between cycles.
// "a" starts with a "0" state and increments it every time, using it as a
// factor in its umul.
TEST(SerialProcRuntimeTest, CarriesState) {
  constexpr int kNumCycles = 16000;
  const std::string kIrText = R"(
package p

chan a_in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan a_to_b(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan b_out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc a(my_token: token, state: (bits[32]), init=(1)) {
  tuple_index.1: bits[32] = tuple_index(state, index=0)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(tuple_index.1, tuple_index.4)
  send.6: token = send(tuple_index.3, umul.5, channel_id=1)
  literal.7: bits[32] = literal(value=1)
  add.8: bits[32] = add(tuple_index.1, literal.7)
  tuple.9: (bits[32]) = tuple(add.8)
  next (send.6, tuple.9)
}

proc b(my_token: token, state: (bits[32]), init=()) {
  literal.100: bits[32] = literal(value=3)
  receive.200: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.300: token = tuple_index(receive.200, index=0)
  tuple_index.400: bits[32] = tuple_index(receive.200, index=1)
  umul.500: bits[32] = umul(literal.100, tuple_index.400)
  send.600: token = send(tuple_index.300, umul.500, channel_id=2)
  next (send.600, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  auto* queue_mgr = runtime->queue_mgr();

  XLS_ASSERT_OK_AND_ASSIGN(auto input_queue, queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(auto internal_queue, queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(auto output_queue, queue_mgr->GetQueueById(2));

  int dummy = 0;
  EnqueueData(internal_queue, dummy);

  for (int i = 0; i < kNumCycles; i++) {
    EnqueueData(input_queue, i);
    XLS_ASSERT_OK(runtime->Tick());
  }

  // Drop the output from the first cycle; it's not real/valid output.
  DequeueData<int>(output_queue);
  for (int i = 0; i < kNumCycles - 1; i++) {
    int actual = DequeueData<int>(output_queue);
    ASSERT_EQ(actual, i * (i + 1) * 3);
  }
}

// This test verifies that SerialProcRuntime can detect when a network has
// deadlocked (when it's waiting on more data that's not coming).
TEST(SerialProcRuntimeTest, DetectsDeadlock) {
  // Proc A sends one pieces of data to B, but B expects two - the second will
  // never arrive.
  const std::string kIrText = R"(
package p

chan first(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan second(bits[32], id=2, kind=streaming, ops=send_receive, flow_control=none, metadata="")

proc a(my_token: token, state: bits[1], init=0) {
  literal.1: bits[32] = literal(value=1)
  send.3: token = send(my_token, literal.1, channel_id=1)
  send.4: token = send(send.3, literal.1, predicate=state, channel_id=2)
  next (send.4, state)
}

proc b(my_token: token, state: (), init=()) {
  receive.101: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.102: token = tuple_index(receive.101, index=0)
  receive.103: (token, bits[32]) = receive(tuple_index.102, channel_id=2)
  tuple_index.104: token = tuple_index(receive.103, index=0)
  next (tuple_index.104, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  ASSERT_THAT(runtime->Tick(),
              status_testing::StatusIs(absl::StatusCode::kAborted));
}

// Tests that a proc can be blocked (by missing recv data) and then become
// unblocked when data is available.
TEST(SerialProcRuntimeTest, FinishesDelayedCycle) {
  const std::string kIrText = R"(
package p

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan a_to_b(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=none, metadata="")
chan output(bits[32], id=2, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc a(my_token: token, state: (), init=()) {
  receive.1: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tuple_index.2, tuple_index.3, channel_id=1)
  next (send.4, state)
}

proc b(my_token: token, state: (), init=()) {
  receive.101: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.102: token = tuple_index(receive.101, index=0)
  tuple_index.103: bits[32] = tuple_index(receive.101, index=1)
  send.104: token = send(tuple_index.102, tuple_index.103, channel_id=2)
  next (send.104, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(auto input_queue,
                           runtime->queue_mgr()->GetQueueById(0));
  Thread thread([input_queue]() {
    // Give enough time for the network to block, then send in the missing data.
    sleep(1);
    int32_t data = 42;
    input_queue->Send(absl::bit_cast<uint8_t*>(&data), sizeof(data));
  });
  XLS_ASSERT_OK(runtime->Tick());
  XLS_ASSERT_OK_AND_ASSIGN(auto output_queue,
                           runtime->queue_mgr()->GetQueueById(2));

  int32_t data;
  output_queue->Recv(absl::bit_cast<uint8_t*>(&data), sizeof(data));
  EXPECT_EQ(data, 42);
  thread.Join();
}

// This test verifies that wide types may be passed via send/receive.
TEST(SerialProcRuntimeTest, WideTypes) {
  const std::string kIrText = R"(
package p

chan in((bits[132], bits[217]), id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan out((bits[132], bits[217]), id=1, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc a(my_token: token, state: (), init=()) {
  rcv: (token, (bits[132], bits[217])) = receive(my_token, channel_id=0)
  rcv_tkn: token = tuple_index(rcv, index=0)
  rcv_data: (bits[132], bits[217]) = tuple_index(rcv, index=1)
  elem_0: bits[132] = tuple_index(rcv_data, index=0)
  elem_1: bits[217] = tuple_index(rcv_data, index=1)
  one: bits[132] = literal(value=1)
  two: bits[217] = literal(value=2)
  mod_elem_0: bits[132] = add(elem_0, one)
  mod_elem_1: bits[217] = add(elem_1, two)
  to_send: (bits[132], bits[217]) = tuple(mod_elem_0, mod_elem_1)
  snd: token = send(rcv_tkn, to_send, channel_id=1)

  next (snd, state)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      Value input,
      Parser::ParseTypedValue(
          "(bits[132]: 0xf_abcd_1234_9876_1010_aaaa_beeb_c12c_defd, "
          "bits[217]: 0x1111_2222_3333_4444_abcd_4321_4444_2468_3579)"));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * input_channel, p->GetChannel(0));
  XLS_ASSERT_OK(runtime->EnqueueValueToChannel(input_channel, input));

  XLS_ASSERT_OK(runtime->Tick());

  XLS_ASSERT_OK_AND_ASSIGN(Channel * output_channel, p->GetChannel(1));
  XLS_ASSERT_OK_AND_ASSIGN(Value output,
                           runtime->DequeueValueFromChannel(output_channel));

  EXPECT_EQ(output.ToString(),
            "(bits[132]:0xf_abcd_1234_9876_1010_aaaa_beeb_c12c_defe, "
            "bits[217]:0x1111_2222_3333_4444_abcd_4321_4444_2468_357b)");
}

// TODO(meheff): This test is a duplicate of one in
// proc_network_interpreter_test. Unify the set of tests in one location.
TEST(SerialProcRuntimeTest, ChannelInitValues) {
  auto p = absl::make_unique<Package>("init_value");
  // Create an iota proc which uses a channel to convey the state rather than
  // using the explicit proc state. However, the state channel has multiple
  // initial values which results in interleaving of difference sequences of
  // iota values.
  ProcBuilder pb("backedge_proc", /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tok", /*state_name=*/"nil_state", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * state_channel,
      p->CreateStreamingChannel(
          "state", ChannelOps::kSendReceive, p->GetBitsType(32),
          // Initial value of iotas are 42, 55, 100. Three sequences of
          // interleaved numbers will be generated starting at these
          // values.
          {Value(UBits(42, 32)), Value(UBits(55, 32)), Value(UBits(100, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_channel,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));

  BValue state_receive = pb.Receive(state_channel, pb.GetTokenParam());
  BValue receive_token = pb.TupleIndex(state_receive, /*idx=*/0);
  BValue state = pb.TupleIndex(state_receive, /*idx=*/1);
  BValue next_state = pb.Add(state, pb.Literal(UBits(1, 32)));
  BValue out_send = pb.Send(output_channel, pb.GetTokenParam(), state);
  BValue state_send = pb.Send(state_channel, receive_token, next_state);
  XLS_ASSERT_OK(
      pb.Build(pb.AfterAll({out_send, state_send}), pb.GetStateParam())
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));

  for (int64_t i = 0; i < 9; ++i) {
    XLS_ASSERT_OK(runtime->Tick());
  }

  auto get_output = [&]() -> absl::StatusOr<Value> {
    return runtime->DequeueValueFromChannel(output_channel);
  };

  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(55, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(100, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(43, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(56, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(101, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(44, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(57, 32))));
  EXPECT_THAT(get_output(), IsOkAndHolds(Value(UBits(102, 32))));
}

}  // namespace
}  // namespace xls
