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

#include "xls/interpreter/proc_network_interpreter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class ProcNetworkInterpreterTest : public IrTestBase {};

// Creates a proc which has a single send operation using the given channel
// which sends a sequence of U32 values starting at 'starting_value' and
// increasing byte 'step' each tick.
absl::StatusOr<Proc*> CreateIotaProc(absl::string_view proc_name,
                                     int64_t starting_value, int64_t step,
                                     Channel* channel, Package* package) {
  ProcBuilder pb(proc_name, /*init_value=*/Value(UBits(starting_value, 32)),
                 /*token_name=*/"tok", /*state_name=*/"prev", package);
  BValue send_token = pb.Send(channel, pb.GetTokenParam(), pb.GetStateParam());

  BValue new_value = pb.Add(pb.GetStateParam(), pb.Literal(UBits(step, 32)));
  return pb.Build(send_token, new_value);
}

// Creates a proc which keeps a running sum of all values read through the input
// channel. The sum is sent via an output chanel each iteration.
absl::StatusOr<Proc*> CreateAccumProc(absl::string_view proc_name,
                                      Channel* in_channel, Channel* out_channel,
                                      Package* package) {
  ProcBuilder pb(proc_name, /*init_value=*/Value(UBits(0, 32)),
                 /*token_name=*/"tok", /*state_name=*/"prev", package);
  BValue token_input = pb.Receive(in_channel, pb.GetTokenParam());
  BValue recv_token = pb.TupleIndex(token_input, 0);
  BValue input = pb.TupleIndex(token_input, 1);
  BValue accum = pb.Add(pb.GetStateParam(), input);
  BValue send_token = pb.Send(out_channel, recv_token, accum);
  return pb.Build(send_token, accum);
}

// Creates a proc which simply passes through a received value to a send.
absl::StatusOr<Proc*> CreatePassThroughProc(absl::string_view proc_name,
                                            Channel* in_channel,
                                            Channel* out_channel,
                                            Package* package) {
  ProcBuilder pb(proc_name, /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tok", /*state_name=*/"state", package);
  BValue token_input = pb.Receive(in_channel, pb.GetTokenParam());
  BValue recv_token = pb.TupleIndex(token_input, 0);
  BValue input = pb.TupleIndex(token_input, 1);
  BValue send_token = pb.Send(out_channel, recv_token, input);
  return pb.Build(send_token, pb.GetStateParam());
}

// Create a proc which reads tuples of (count: u32, char: u8) from in_channel,
// run-length decodes them, and sends the resulting char stream to
// out_channel. Run lengths of zero are allowed.
absl::StatusOr<Proc*> CreateRunLengthDecoderProc(absl::string_view proc_name,
                                                 Channel* in_channel,
                                                 Channel* out_channel,
                                                 Package* package) {
  // Proc state is a two-tuple containing: character to write and remaining
  // number of times to write the character.
  ProcBuilder pb(
      proc_name,
      /*init_value=*/Value::Tuple({Value(UBits(0, 8)), Value(UBits(0, 32))}),
      /*token_name=*/"tok", /*state_name=*/"state", package);
  BValue last_char = pb.TupleIndex(pb.GetStateParam(), 0);
  BValue num_remaining = pb.TupleIndex(pb.GetStateParam(), 1);
  BValue receive_next = pb.Eq(num_remaining, pb.Literal(UBits(0, 32)));
  BValue receive_if =
      pb.ReceiveIf(in_channel, pb.GetTokenParam(), receive_next);
  BValue receive_if_data = pb.TupleIndex(receive_if, 1);
  BValue run_length =
      pb.Select(receive_next,
                /*cases=*/{num_remaining, pb.TupleIndex(receive_if_data, 0)});
  BValue this_char = pb.Select(
      receive_next, /*cases=*/{last_char, pb.TupleIndex(receive_if_data, 1)});
  BValue run_length_is_nonzero = pb.Ne(run_length, pb.Literal(UBits(0, 32)));
  BValue send = pb.SendIf(out_channel, pb.TupleIndex(receive_if, 0),
                          run_length_is_nonzero, this_char);
  BValue next_state = pb.Tuple(
      {this_char,
       pb.Select(
           run_length_is_nonzero,
           /*cases=*/{pb.Literal(UBits(0, 32)),
                      pb.Subtract(run_length, pb.Literal(UBits(1, 32)))})});

  return pb.Build(send, next_state);
}

TEST_F(ProcNetworkInterpreterTest, ProcIota) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package->CreateStreamingChannel("iota_out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK(CreateIotaProc("iota", /*starting_value=*/5, /*step=*/10,
                               channel, package.get())
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcNetworkInterpreter> interpreter,
                           ProcNetworkInterpreter::Create(
                               package.get(), /*user_defined_queues*/ {}));

  ChannelQueue& queue = interpreter->queue_manager().GetQueue(channel);

  EXPECT_TRUE(queue.empty());
  XLS_ASSERT_OK(interpreter->Tick());
  EXPECT_EQ(queue.size(), 1);

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(5, 32))));

  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());

  EXPECT_EQ(queue.size(), 3);

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(15, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(25, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(35, 32))));
}

TEST_F(ProcNetworkInterpreterTest, IotaFeedingAccumulator) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * iota_accum_channel,
      package->CreateStreamingChannel("iota_accum", ChannelOps::kSendReceive,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_channel,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK(CreateIotaProc("iota", /*starting_value=*/0, /*step=*/1,
                               iota_accum_channel, package.get())
                    .status());
  XLS_ASSERT_OK(
      CreateAccumProc("accum", iota_accum_channel, out_channel, package.get())
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcNetworkInterpreter> interpreter,
                           ProcNetworkInterpreter::Create(
                               package.get(), /*user_defined_queues*/ {}));

  ChannelQueue& queue = interpreter->queue_manager().GetQueue(out_channel);

  EXPECT_TRUE(queue.empty());

  XLS_ASSERT_OK(interpreter->Tick());

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(0, 32))));

  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());

  EXPECT_EQ(queue.size(), 3);

  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(1, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(3, 32))));
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(6, 32))));
}

TEST_F(ProcNetworkInterpreterTest, DegenerateProc) {
  // Tests interpreting a proc with no send of receive nodes.
  auto package = CreatePackage();
  ProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tok", /*state_name=*/"prev", package.get());
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), pb.GetStateParam()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcNetworkInterpreter> interpreter,
                           ProcNetworkInterpreter::Create(
                               package.get(), /*user_defined_queues*/ {}));

  // Ticking the proc has no observable effect, but it should not hang or crash.
  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());
}

TEST_F(ProcNetworkInterpreterTest, WrappedProc) {
  // Create a proc which receives a value, sends it the accumulator proc, and
  // sends the result.
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_channel,
      package->CreateStreamingChannel("input", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_accum_channel,
      package->CreateStreamingChannel("accum_in", ChannelOps::kSendReceive,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_accum_channel,
      package->CreateStreamingChannel("accum_out", ChannelOps::kSendReceive,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_channel,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  ProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tok", /*state_name=*/"prev", package.get());
  BValue recv_input = pb.Receive(in_channel, pb.GetTokenParam());
  BValue send_to_accum =
      pb.Send(in_accum_channel, /*token=*/pb.TupleIndex(recv_input, 0),
              /*data_operands=*/{pb.TupleIndex(recv_input, 1)});
  BValue recv_from_accum = pb.Receive(out_accum_channel, send_to_accum);
  BValue send_output =
      pb.Send(out_channel, /*token=*/pb.TupleIndex(recv_from_accum, 0),
              /*data_operands=*/{pb.TupleIndex(recv_from_accum, 1)});
  XLS_ASSERT_OK(pb.Build(send_output, pb.Tuple({})));

  XLS_ASSERT_OK(CreateAccumProc("accum", /*in_channel=*/in_accum_channel,
                                /*out_channel=*/out_accum_channel,
                                package.get())
                    .status());

  std::vector<std::unique_ptr<ChannelQueue>> queues;
  std::vector<Value> inputs = {
      {Value(UBits(10, 32))}, {Value(UBits(20, 32))}, {Value(UBits(30, 32))}};
  queues.push_back(
      absl::make_unique<FixedChannelQueue>(in_channel, package.get(), inputs));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcNetworkInterpreter> interpreter,
      ProcNetworkInterpreter::Create(package.get(), std::move(queues)));

  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());
  XLS_ASSERT_OK(interpreter->Tick());

  ChannelQueue& output_queue =
      interpreter->queue_manager().GetQueue(out_channel);
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(10, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(30, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(60, 32))));
}

TEST_F(ProcNetworkInterpreterTest, DeadlockedProc) {
  // Test a trivial deadlocked proc network. A single proc with a feedback edge
  // from its send operation to its receive.
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package->CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK(CreatePassThroughProc("feedback", /*in_channel=*/channel,
                                      /*out_channel=*/channel, package.get())
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcNetworkInterpreter> interpreter,
                           ProcNetworkInterpreter::Create(
                               package.get(), /*user_defined_queues*/ {}));

  // The interpreter can tick once without deadlocking because some instructions
  // can actually execute initially (e.g., the parameters). A subsequent call to
  // Tick() will detect the deadlock.
  XLS_ASSERT_OK(interpreter->Tick());
  EXPECT_THAT(
      interpreter->Tick(),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Proc network is deadlocked. Blocked channels: my_channel")));
}

TEST_F(ProcNetworkInterpreterTest, RunLengthDecoding) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_channel,
      package->CreateStreamingChannel(
          "in", ChannelOps::kReceiveOnly,
          package->GetTupleType(
              {package->GetBitsType(32), package->GetBitsType(8)})));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_channel,
      package->CreateStreamingChannel("output", ChannelOps::kSendOnly,
                                      package->GetBitsType(8)));

  XLS_ASSERT_OK(CreateRunLengthDecoderProc("decoder", input_channel,
                                           output_channel, package.get())
                    .status());

  std::vector<std::unique_ptr<ChannelQueue>> queues;
  std::vector<Value> inputs = {
      Value::Tuple({Value(UBits(1, 32)), Value(UBits(42, 8))}),
      Value::Tuple({Value(UBits(3, 32)), Value(UBits(123, 8))}),
      Value::Tuple({Value(UBits(0, 32)), Value(UBits(55, 8))}),
      Value::Tuple({Value(UBits(0, 32)), Value(UBits(66, 8))}),
      Value::Tuple({Value(UBits(2, 32)), Value(UBits(20, 8))})};
  queues.push_back(absl::make_unique<FixedChannelQueue>(input_channel,
                                                        package.get(), inputs));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcNetworkInterpreter> interpreter,
      ProcNetworkInterpreter::Create(package.get(), std::move(queues)));

  ChannelQueue& output_queue =
      interpreter->queue_manager().GetQueue(output_channel);
  while (output_queue.size() < 6) {
    XLS_ASSERT_OK(interpreter->Tick());
  }

  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 8))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(123, 8))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(123, 8))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(123, 8))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(20, 8))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(20, 8))));
}

TEST_F(ProcNetworkInterpreterTest, RunLengthDecodingFilter) {
  // Connect a run-length decoding proc to a proc which only passes through even
  // values.
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_channel,
      package->CreateStreamingChannel(
          "in", ChannelOps::kReceiveOnly,
          package->GetTupleType(
              {package->GetBitsType(32), package->GetBitsType(8)})));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * decoded_channel,
      package->CreateStreamingChannel("decoded", ChannelOps::kSendReceive,
                                      package->GetBitsType(8)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_channel,
      package->CreateStreamingChannel("output", ChannelOps::kSendOnly,
                                      package->GetBitsType(8)));

  XLS_ASSERT_OK(CreateRunLengthDecoderProc("decoder", input_channel,
                                           decoded_channel, package.get())
                    .status());
  ProcBuilder pb("filter", /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tok", /*state_name=*/"nil", package.get());
  BValue receive = pb.Receive(decoded_channel, pb.GetTokenParam());
  BValue rx_token = pb.TupleIndex(receive, 0);
  BValue rx_value = pb.TupleIndex(receive, 1);
  BValue rx_value_even =
      pb.Not(pb.BitSlice(rx_value, /*start=*/0, /*width=*/1));
  BValue send_if = pb.SendIf(output_channel, rx_token, rx_value_even, rx_value);
  XLS_ASSERT_OK(pb.Build(send_if, pb.GetStateParam()));

  std::vector<std::unique_ptr<ChannelQueue>> queues;
  std::vector<Value> inputs = {
      Value::Tuple({Value(UBits(1, 32)), Value(UBits(42, 8))}),
      Value::Tuple({Value(UBits(3, 32)), Value(UBits(123, 8))}),
      Value::Tuple({Value(UBits(0, 32)), Value(UBits(55, 8))}),
      Value::Tuple({Value(UBits(0, 32)), Value(UBits(66, 8))}),
      Value::Tuple({Value(UBits(2, 32)), Value(UBits(20, 8))})};
  queues.push_back(absl::make_unique<FixedChannelQueue>(input_channel,
                                                        package.get(), inputs));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcNetworkInterpreter> interpreter,
      ProcNetworkInterpreter::Create(package.get(), std::move(queues)));

  ChannelQueue& output_queue =
      interpreter->queue_manager().GetQueue(output_channel);
  while (output_queue.size() < 3) {
    XLS_ASSERT_OK(interpreter->Tick());
  }

  // Only even values should make it through the filter.
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 8))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(20, 8))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(20, 8))));
}

TEST_F(ProcNetworkInterpreterTest, IotaWithChannelBackedge) {
  // Create an iota proc which uses a channel to convey the state rather than
  // using the explicit proc state. The state channel has an initial value, just
  // like a proc's state.
  auto package = CreatePackage();
  ProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tok", /*state_name=*/"nil_state",
                 package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * state_channel,
      package->CreateStreamingChannel(
          "state", ChannelOps::kSendReceive, package->GetBitsType(32),
          /*initial_values=*/{Value(UBits(42, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_channel,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  BValue state_receive = pb.Receive(state_channel, pb.GetTokenParam());
  BValue receive_token = pb.TupleIndex(state_receive, /*idx=*/0);
  BValue state = pb.TupleIndex(state_receive, /*idx=*/1);
  BValue next_state = pb.Add(state, pb.Literal(UBits(1, 32)));
  BValue out_send = pb.Send(output_channel, pb.GetTokenParam(), state);
  BValue state_send = pb.Send(state_channel, receive_token, next_state);
  XLS_ASSERT_OK(
      pb.Build(pb.AfterAll({out_send, state_send}), pb.GetStateParam())
          .status());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcNetworkInterpreter> interpreter,
                           ProcNetworkInterpreter::Create(package.get(), {}));

  ChannelQueue& output_queue =
      interpreter->queue_manager().GetQueue(output_channel);
  while (output_queue.size() < 3) {
    XLS_ASSERT_OK(interpreter->Tick());
  }

  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(43, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(44, 32))));
}

TEST_F(ProcNetworkInterpreterTest, IotaWithChannelBackedgeAndTwoInitialValues) {
  auto package = CreatePackage();
  // Create an iota proc which uses a channel to convey the state rather than
  // using the explicit proc state. However, the state channel has multiple
  // initial values which results in interleaving of difference sequences of
  // iota values.
  ProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tok", /*state_name=*/"nil_state",
                 package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * state_channel,
      package->CreateStreamingChannel(
          "state", ChannelOps::kSendReceive, package->GetBitsType(32),
          // Initial value of iotas are 42, 55, 100. Three sequences of
          // interleaved numbers will be generated starting at these
          // values.
          {Value(UBits(42, 32)), Value(UBits(55, 32)), Value(UBits(100, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_channel,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  BValue state_receive = pb.Receive(state_channel, pb.GetTokenParam());
  BValue receive_token = pb.TupleIndex(state_receive, /*idx=*/0);
  BValue state = pb.TupleIndex(state_receive, /*idx=*/1);
  BValue next_state = pb.Add(state, pb.Literal(UBits(1, 32)));
  BValue out_send = pb.Send(output_channel, pb.GetTokenParam(), state);
  BValue state_send = pb.Send(state_channel, receive_token, next_state);
  XLS_ASSERT_OK(
      pb.Build(pb.AfterAll({out_send, state_send}), pb.GetStateParam())
          .status());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcNetworkInterpreter> interpreter,
                           ProcNetworkInterpreter::Create(package.get(), {}));

  ChannelQueue& output_queue =
      interpreter->queue_manager().GetQueue(output_channel);
  while (output_queue.size() < 9) {
    XLS_ASSERT_OK(interpreter->Tick());
  }

  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(55, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(100, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(43, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(56, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(101, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(44, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(57, 32))));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(102, 32))));
}

}  // namespace
}  // namespace xls
