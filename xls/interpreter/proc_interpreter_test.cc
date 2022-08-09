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

#include "xls/interpreter/proc_interpreter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using testing::ElementsAre;
class ProcInterpreterTest : public IrTestBase {};

TEST_F(ProcInterpreterTest, ProcIota) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package->CreateStreamingChannel("iota_out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  // Create an output-only proc which counts up by 7 starting at 42.
  ProcBuilder pb("iota", /*token_name=*/"tok", package.get());
  BValue counter = pb.StateElement("cnt", Value(UBits(42, 32)));
  BValue send_token = pb.Send(channel, pb.GetTokenParam(), counter);
  BValue new_value = pb.Add(counter, pb.Literal(UBits(7, 32)));
  XLS_ASSERT_OK(pb.Build(send_token, {new_value}).status());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, package.get()));
  ProcInterpreter interpreter(FindProc("iota", package.get()),
                              queue_manager.get());
  ChannelQueue& ch0_queue = queue_manager->GetQueue(channel);

  ASSERT_TRUE(ch0_queue.empty());

  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  EXPECT_TRUE(interpreter.IsIterationComplete());
  EXPECT_EQ(ch0_queue.size(), 1);
  EXPECT_FALSE(ch0_queue.empty());
  EXPECT_THAT(ch0_queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 32))));
  EXPECT_EQ(ch0_queue.size(), 0);
  EXPECT_TRUE(ch0_queue.empty());

  // Run three times. Should enqueue three values in the output queue.
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));

  EXPECT_EQ(ch0_queue.size(), 3);

  EXPECT_THAT(ch0_queue.Dequeue(), IsOkAndHolds(Value(UBits(49, 32))));
  EXPECT_THAT(ch0_queue.Dequeue(), IsOkAndHolds(Value(UBits(56, 32))));
  EXPECT_THAT(ch0_queue.Dequeue(), IsOkAndHolds(Value(UBits(63, 32))));

  EXPECT_TRUE(ch0_queue.empty());
}

TEST_F(ProcInterpreterTest, ProcWhichReturnsPreviousResults) {
  Package package(TestName());
  ProcBuilder pb("prev", /*token_name=*/"tok", &package);
  BValue prev_input = pb.StateElement("prev_in", Value(UBits(55, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_in, package.CreateStreamingChannel(
                                                "in", ChannelOps::kSendReceive,
                                                package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_out, package.CreateStreamingChannel(
                                                 "out", ChannelOps::kSendOnly,
                                                 package.GetBitsType(32)));

  // Build a proc which receives a value and saves it, and sends the value
  // received in the previous iteration.
  BValue token_input = pb.Receive(ch_in, pb.GetTokenParam());
  BValue recv_token = pb.TupleIndex(token_input, 0);
  BValue input = pb.TupleIndex(token_input, 1);
  BValue send_token = pb.Send(ch_out, recv_token, prev_input);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(send_token, {input}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, &package));

  ProcInterpreter interpreter(proc, queue_manager.get());
  ChannelQueue& input_queue = queue_manager->GetQueue(ch_in);
  ChannelQueue& output_queue = queue_manager->GetQueue(ch_out);

  ASSERT_TRUE(input_queue.empty());
  ASSERT_TRUE(output_queue.empty());

  // First invocation of RunIterationUntilCompleteOrBlocked should block on
  // waiting for input on the "in" channel.
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = false,
                                              .progress_made = true,
                                              .blocked_channels = {ch_in}}));
  EXPECT_FALSE(interpreter.IsIterationComplete());

  // Blocked on the receive so no progress should be made if you try to resume
  // execution again.
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = false,
                                              .progress_made = false,
                                              .blocked_channels = {ch_in}}));
  EXPECT_FALSE(interpreter.IsIterationComplete());

  // Enqueue something into the input queue.
  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(42, 32))}));
  EXPECT_EQ(input_queue.size(), 1);
  EXPECT_TRUE(output_queue.empty());

  // It can now continue until complete.
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  EXPECT_TRUE(interpreter.IsIterationComplete());

  EXPECT_TRUE(input_queue.empty());
  EXPECT_EQ(output_queue.size(), 1);

  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(55, 32))));
  EXPECT_TRUE(output_queue.empty());

  // Now run the next iteration. It should spit out the value we fed in during
  // the last iteration (42).
  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(123, 32))}));
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 32))));
}

TEST_F(ProcInterpreterTest, ConditionalReceiveProc) {
  // Create a proc which has a receive_if which fires every other
  // iteration. Receive_if value is unconditionally sent over a different
  // channel.
  Package package(TestName());
  ProcBuilder pb("conditional_send", /*token_name=*/"tok", &package);
  BValue st = pb.StateElement("st", Value(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_in, package.CreateStreamingChannel(
                                                "in", ChannelOps::kSendReceive,
                                                package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_out, package.CreateStreamingChannel(
                                                 "out", ChannelOps::kSendOnly,
                                                 package.GetBitsType(32)));

  BValue receive_if = pb.ReceiveIf(ch_in, /*token=*/pb.GetTokenParam(),
                                   /*pred=*/st);
  BValue rx_token = pb.TupleIndex(receive_if, 0);
  BValue rx_data = pb.TupleIndex(receive_if, 1);
  BValue send = pb.Send(ch_out, rx_token, {rx_data});
  // Next state value is the inverse of the current state value.
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(send, {pb.Not(st)}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, &package));

  ProcInterpreter interpreter(proc, queue_manager.get());
  ChannelQueue& input_queue = queue_manager->GetQueue(ch_in);
  ChannelQueue& output_queue = queue_manager->GetQueue(ch_out);

  ASSERT_TRUE(input_queue.empty());
  ASSERT_TRUE(output_queue.empty());

  // Enqueue a single value into the input queue.
  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(42, 32))}));

  // In the first iteration, the receive_if should dequeue a value because the
  // proc state value (which is the receive_if predicate) is initialized to
  // true.
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(42, 32))));

  // The second iteration should not dequeue anything as the receive predicate
  // is now false. The data value of the receive (which is sent over the output
  // channel) should be zeros.
  ASSERT_TRUE(input_queue.empty());
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(0, 32))));

  // The third iteration should again dequeue a value.
  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(123, 32))}));
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(123, 32))));
}

TEST_F(ProcInterpreterTest, ConditionalSendProc) {
  // Create an output-only proc with a by-one-counter which sends only
  // even values over a conditional send.
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("even_out", ChannelOps::kSendOnly,
                                     package.GetBitsType(32)));

  ProcBuilder pb("even", /*token_name=*/"tok", &package);
  BValue prev = pb.StateElement("prev", Value(UBits(0, 32)));
  BValue is_even = pb.Eq(pb.BitSlice(prev, /*start=*/0, /*width=*/1),
                         pb.Literal(UBits(0, 1)));
  BValue send_if = pb.SendIf(channel, pb.GetTokenParam(), is_even, prev);
  BValue new_value = pb.Add(prev, pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK(pb.Build(send_if, {new_value}).status());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, &package));
  ProcInterpreter interpreter(FindProc("even", &package), queue_manager.get());

  ChannelQueue& queue = queue_manager->GetQueue(channel);

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_EQ(queue.size(), 1);
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(0, 32))));

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_TRUE(queue.empty());

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(2, 32))));

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_TRUE(queue.empty());

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_THAT(queue.Dequeue(), IsOkAndHolds(Value(UBits(4, 32))));
}

TEST_F(ProcInterpreterTest, OneToTwoDemux) {
  // Build a proc which acts as a one-to-two demux. Data channels are streaming,
  // and the selector is a single-value channel.
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

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue dir = pb.Receive(ch_dir);
  BValue in = pb.Receive(ch_in);
  pb.SendIf(ch_a, dir, in);
  pb.SendIf(ch_b, pb.Not(dir), in);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, &package));
  ProcInterpreter interpreter(proc, queue_manager.get());

  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * dir_queue,
                           queue_manager->GetQueueByName("dir"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in_queue,
                           queue_manager->GetQueueByName("in"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * a_queue,
                           queue_manager->GetQueueByName("a"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * b_queue,
                           queue_manager->GetQueueByName("b"));

  // Set the direction to output B and enqueue a bunch of values.
  XLS_ASSERT_OK(dir_queue->Enqueue(Value(UBits(0, 1))));
  XLS_ASSERT_OK(in_queue->Enqueue(Value(UBits(1, 32))));
  XLS_ASSERT_OK(in_queue->Enqueue(Value(UBits(2, 32))));
  XLS_ASSERT_OK(in_queue->Enqueue(Value(UBits(3, 32))));
  XLS_ASSERT_OK(in_queue->Enqueue(Value(UBits(4, 32))));

  // Tick twice and verify that the expected outputs appear at output B.
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_TRUE(a_queue->empty());
  EXPECT_FALSE(b_queue->empty());
  EXPECT_THAT(b_queue->Dequeue(), IsOkAndHolds(Value(UBits(1, 32))));

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_THAT(b_queue->Dequeue(), IsOkAndHolds(Value(UBits(2, 32))));
  EXPECT_TRUE(a_queue->empty());
  EXPECT_TRUE(b_queue->empty());

  // Switch direction to output A.
  XLS_ASSERT_OK(dir_queue->Enqueue(Value(UBits(1, 1))));

  // Tick twice and verify that the expectedoutputs appear at output A.
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_FALSE(a_queue->empty());
  EXPECT_TRUE(b_queue->empty());

  EXPECT_THAT(a_queue->Dequeue(), IsOkAndHolds(Value(UBits(3, 32))));

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());

  EXPECT_THAT(a_queue->Dequeue(), IsOkAndHolds(Value(UBits(4, 32))));
  EXPECT_TRUE(a_queue->empty());
  EXPECT_TRUE(b_queue->empty());
}

TEST_F(ProcInterpreterTest, StateReset) {
  // Build a proc which outputs its input state and then iterates by adding 3.
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue st = pb.StateElement("st", Value(SBits(11, 32)));
  BValue send_token = pb.Send(ch_out, pb.GetTokenParam(), st);
  BValue add_lit = pb.Literal(SBits(3, 32));
  BValue next_int = pb.Add(st, add_lit);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(send_token, {next_int}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, &package));
  ProcInterpreter interpreter(proc, queue_manager.get());

  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * out_queue,
                           queue_manager->GetQueueByName("out"));

  // Tick twice and verify that the expected outputs appear at output B.
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  ASSERT_FALSE(out_queue->empty());
  EXPECT_THAT(out_queue->Dequeue(), IsOkAndHolds(Value(UBits(11, 32))));
  EXPECT_TRUE(out_queue->empty());

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  ASSERT_FALSE(out_queue->empty());
  EXPECT_THAT(out_queue->Dequeue(), IsOkAndHolds(Value(UBits(14, 32))));
  EXPECT_TRUE(out_queue->empty());

  interpreter.ResetState();

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  ASSERT_FALSE(out_queue->empty());
  EXPECT_THAT(out_queue->Dequeue(), IsOkAndHolds(Value(UBits(11, 32))));
  EXPECT_TRUE(out_queue->empty());
}

TEST_F(ProcInterpreterTest, StatelessProc) {
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

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, package.get()));
  ProcInterpreter interpreter(proc, queue_manager.get());
  ChannelQueue& input_queue = queue_manager->GetQueue(channel_in);
  ChannelQueue& output_queue = queue_manager->GetQueue(channel_out);

  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(1, 32))}));
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  EXPECT_TRUE(interpreter.IsIterationComplete());
  EXPECT_THAT(output_queue.Dequeue(), IsOkAndHolds(Value(UBits(43, 32))));

  // The next state should be empty.
  EXPECT_THAT(interpreter.ResolveState(), IsOkAndHolds(ElementsAre()));
}

TEST_F(ProcInterpreterTest, MultiStateElementProc) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_in,
      package->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));

  // Proc has two state elements:
  //  a: accumulator starting at 0
  //  b: de-cumulator starting at 100
  ProcBuilder pb("multistate", /*token_name=*/"tok", package.get());
  BValue a_state = pb.StateElement("a", Value(UBits(0, 32)));
  BValue b_state = pb.StateElement("b", Value(UBits(100, 32)));
  BValue receive = pb.Receive(channel_in, pb.GetTokenParam());
  BValue data = pb.TupleIndex(receive, 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.TupleIndex(receive, 0),
                                                 {pb.Add(a_state, data),
                                                  pb.Subtract(b_state, data)}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, package.get()));
  ProcInterpreter interpreter(proc, queue_manager.get());
  ChannelQueue& input_queue = queue_manager->GetQueue(channel_in);
  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(1, 32))}));
  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(10, 32))}));
  XLS_ASSERT_OK(input_queue.Enqueue({Value(UBits(20, 32))}));

  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_THAT(
      interpreter.ResolveState(),
      IsOkAndHolds(ElementsAre(Value(UBits(1, 32)), Value(UBits(99, 32)))));
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_THAT(
      interpreter.ResolveState(),
      IsOkAndHolds(ElementsAre(Value(UBits(11, 32)), Value(UBits(89, 32)))));
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_THAT(
      interpreter.ResolveState(),
      IsOkAndHolds(ElementsAre(Value(UBits(31, 32)), Value(UBits(69, 32)))));
}

TEST_F(ProcInterpreterTest, NonBlockingReceives) {
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

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, package.get()));

  ProcInterpreter interpreter(proc, queue_manager.get());

  ChannelQueue& in0_queue = queue_manager->GetQueue(in0);
  ChannelQueue& in1_queue = queue_manager->GetQueue(in1);
  ChannelQueue& in2_queue = queue_manager->GetQueue(in2);
  ChannelQueue& out0_queue = queue_manager->GetQueue(out0);

  Value out_v;

  // Initialize the single value queue.
  XLS_ASSERT_OK(in2_queue.Enqueue(Value(UBits(10, 32))));

  // All other channels are non-blocking, so run even if the queues are empty.
  EXPECT_TRUE(in0_queue.empty());
  EXPECT_TRUE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_TRUE(in0_queue.empty());
  EXPECT_TRUE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  EXPECT_FALSE(out0_queue.empty());

  XLS_ASSERT_OK_AND_ASSIGN(out_v, out0_queue.Dequeue());
  EXPECT_EQ(out_v, Value(UBits(10, 32)));

  // Run with only in1 (and in2) having data.
  XLS_ASSERT_OK(in1_queue.Enqueue(Value(UBits(5, 32))));
  EXPECT_TRUE(in0_queue.empty());
  EXPECT_FALSE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_TRUE(in0_queue.empty());
  EXPECT_TRUE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  EXPECT_FALSE(out0_queue.empty());

  XLS_ASSERT_OK_AND_ASSIGN(out_v, out0_queue.Dequeue());
  EXPECT_EQ(out_v, Value(UBits(15, 32)));

  // Run with only in0 (and in2) having data.
  XLS_ASSERT_OK(in0_queue.Enqueue(Value(UBits(7, 32))));
  EXPECT_FALSE(in0_queue.empty());
  EXPECT_TRUE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_TRUE(in0_queue.empty());
  EXPECT_TRUE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  EXPECT_FALSE(out0_queue.empty());

  XLS_ASSERT_OK_AND_ASSIGN(out_v, out0_queue.Dequeue());
  EXPECT_EQ(out_v, Value(UBits(17, 32)));

  // Run with all channels having data.
  XLS_ASSERT_OK(in0_queue.Enqueue(Value(UBits(11, 32))));
  XLS_ASSERT_OK(in1_queue.Enqueue(Value(UBits(22, 32))));
  EXPECT_FALSE(in0_queue.empty());
  EXPECT_FALSE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  XLS_ASSERT_OK(interpreter.RunIterationUntilCompleteOrBlocked().status());
  EXPECT_TRUE(in0_queue.empty());
  EXPECT_TRUE(in1_queue.empty());
  EXPECT_FALSE(in2_queue.empty());
  EXPECT_FALSE(out0_queue.empty());

  XLS_ASSERT_OK_AND_ASSIGN(out_v, out0_queue.Dequeue());
  EXPECT_EQ(out_v, Value(UBits(43, 32)));
}

}  // namespace
}  // namespace xls
