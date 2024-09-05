// Copyright 2022 The XLS Authors
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

#include "xls/interpreter/proc_evaluator_test_base.h"

#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Optional;

TEST_P(ProcEvaluatorTestBase, EmptyProc) {
  auto package = CreatePackage();

  ProcBuilder pb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
}

TEST_P(ProcEvaluatorTestBase, ProcIota) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package->CreateStreamingChannel("iota_out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  // Create an output-only proc which counts up by 7 starting at 42.
  ProcBuilder pb("iota", package.get());
  BValue counter = pb.StateElement("cnt", Value(UBits(42, 32)));
  pb.Send(channel, pb.Literal(Value::Token()), counter);
  BValue new_value = pb.Add(counter, pb.Literal(UBits(7, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({new_value}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator = GetParam().CreateEvaluator(
      FindProc("iota", package.get()), queue_manager.get());
  ChannelQueue& ch0_queue = queue_manager->GetQueue(channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_instance,
      queue_manager->elaboration().GetUniqueInstance(channel));

  ASSERT_TRUE(ch0_queue.IsEmpty());

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  // Before running, the state should be the initial value.
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(42, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(49, 32))));

  EXPECT_EQ(ch0_queue.GetSize(), 1);
  EXPECT_FALSE(ch0_queue.IsEmpty());
  EXPECT_THAT(ch0_queue.Read(), Optional(Value(UBits(42, 32))));
  EXPECT_EQ(ch0_queue.GetSize(), 0);
  EXPECT_TRUE(ch0_queue.IsEmpty());

  // Run three times. Should write three values in the output queue.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(56, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(63, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(70, 32))));

  ASSERT_EQ(ch0_queue.GetSize(), 3);

  EXPECT_THAT(ch0_queue.Read(), Optional(Value(UBits(49, 32))));
  EXPECT_THAT(ch0_queue.Read(), Optional(Value(UBits(56, 32))));
  EXPECT_THAT(ch0_queue.Read(), Optional(Value(UBits(63, 32))));

  EXPECT_TRUE(ch0_queue.IsEmpty());
}

TEST_P(ProcEvaluatorTestBase, ProcWhichReturnsPreviousResults) {
  Package package(TestName());
  ProcBuilder pb("prev", &package);
  BValue prev_input = pb.StateElement("prev_in", Value(UBits(55, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_in, package.CreateStreamingChannel(
                                                "in", ChannelOps::kSendReceive,
                                                package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_out, package.CreateStreamingChannel(
                                                 "out", ChannelOps::kSendOnly,
                                                 package.GetBitsType(32)));

  // Build a proc which receives a value and saves it, and sends the value
  // received in the previous iteration.
  BValue tok = pb.Literal(Value::Token());
  BValue token_input = pb.Receive(ch_in, tok);
  BValue recv_token = pb.TupleIndex(token_input, 0);
  BValue input = pb.TupleIndex(token_input, 1);
  pb.Send(ch_out, recv_token, prev_input);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({input}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());
  ChannelQueue& input_queue = queue_manager->GetQueue(ch_in);
  ChannelQueue& output_queue = queue_manager->GetQueue(ch_out);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_in_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_in));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_out_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_out));

  XLS_ASSERT_OK(input_queue.Write({Value(UBits(42, 32))}));
  XLS_ASSERT_OK(input_queue.Write({Value(UBits(123, 32))}));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(continuation->AtStartOfTick());

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(continuation->AtStartOfTick());

  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(55, 32))));
  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(42, 32))));

  // Ticking once more should block waiting for the input queue. The state
  // should not change because the proc is blocked.
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(123, 32))));
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kBlockedOnReceive,
                  .channel_instance = ch_in_instance,
                  .progress_made = true}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(123, 32))));
  EXPECT_FALSE(continuation->AtStartOfTick());
  EXPECT_TRUE(output_queue.IsEmpty());

  // Feeding another input should unblock it.
  XLS_ASSERT_OK(input_queue.Write({Value(UBits(111, 32))}));
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(continuation->AtStartOfTick());
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(111, 32))));
}

TEST_P(ProcEvaluatorTestBase, MultipleReceives) {
  Package package(TestName());
  TokenlessProcBuilder pb("prev", /*token_name=*/"tok", &package);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in0,
      package.CreateStreamingChannel("in0", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in1,
      package.CreateStreamingChannel("in1", ChannelOps::kSendReceive,
                                     package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in2,
      package.CreateStreamingChannel("in2", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_out, package.CreateStreamingChannel(
                                                 "out", ChannelOps::kSendOnly,
                                                 package.GetBitsType(32)));

  // Build the following proc which returns the sum of ch0 and ch2 inputs. ch2
  // is only read if the input from ch1 is true.
  BValue in0 = pb.Receive(ch_in0);
  BValue in1 = pb.Receive(ch_in1);
  BValue in2 = pb.ReceiveIf(ch_in2, /*pred=*/in1);
  pb.Send(ch_out, pb.Add(in0, in2));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());
  ChannelQueue& in0_queue = queue_manager->GetQueue(ch_in0);
  ChannelQueue& in1_queue = queue_manager->GetQueue(ch_in1);
  ChannelQueue& in2_queue = queue_manager->GetQueue(ch_in2);
  ChannelQueue& output_queue = queue_manager->GetQueue(ch_out);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_in0_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_in0));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_in1_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_in1));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_in2_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_in2));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_out_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_out));

  // Initially should be blocked on in0.
  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kBlockedOnReceive,
                  .channel_instance = ch_in0_instance,
                  .progress_made = true}));
  EXPECT_FALSE(continuation->AtStartOfTick());

  XLS_ASSERT_OK(in0_queue.Write({Value(UBits(10, 32))}));

  // Then should be blocked on in1.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kBlockedOnReceive,
                  .channel_instance = ch_in1_instance,
                  .progress_made = true}));
  EXPECT_FALSE(continuation->AtStartOfTick());

  XLS_ASSERT_OK(in1_queue.Write({Value(UBits(1, 1))}));

  // Then should be blocked on in2.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kBlockedOnReceive,
                  .channel_instance = ch_in2_instance,
                  .progress_made = true}));
  EXPECT_FALSE(continuation->AtStartOfTick());

  XLS_ASSERT_OK(in2_queue.Write({Value(UBits(42, 32))}));

  // Then should send on ch_out.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  // Finally, should run to completion.
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(continuation->AtStartOfTick());

  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(52, 32))));

  // Next, only write data on ch0 and ch1. ch2 should not be read because it's
  // predicate is false.
  XLS_ASSERT_OK(in0_queue.Write({Value(UBits(123, 32))}));
  XLS_ASSERT_OK(in1_queue.Write({Value(UBits(0, 1))}));
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(continuation->AtStartOfTick());

  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(123, 32))));
}

TEST_P(ProcEvaluatorTestBase, ConditionalReceiveProc) {
  // Create a proc which has a receive_if which fires every other
  // iteration. Receive_if value is unconditionally sent over a different
  // channel.
  Package package(TestName());
  ProcBuilder pb("conditional_send", &package);
  BValue st = pb.StateElement("st", Value(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_in, package.CreateStreamingChannel(
                                                "in", ChannelOps::kSendReceive,
                                                package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_out, package.CreateStreamingChannel(
                                                 "out", ChannelOps::kSendOnly,
                                                 package.GetBitsType(32)));

  BValue tok = pb.Literal(Value::Token());
  BValue receive_if = pb.ReceiveIf(ch_in, /*token=*/tok,
                                   /*pred=*/st);
  BValue rx_token = pb.TupleIndex(receive_if, 0);
  BValue rx_data = pb.TupleIndex(receive_if, 1);
  pb.Send(ch_out, rx_token, {rx_data});
  // Next state value is the inverse of the current state value.
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Not(st)}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());
  ChannelQueue& input_queue = queue_manager->GetQueue(ch_in);
  ChannelQueue& output_queue = queue_manager->GetQueue(ch_out);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_out_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_out));

  ASSERT_TRUE(input_queue.IsEmpty());
  ASSERT_TRUE(output_queue.IsEmpty());

  // Write a single value into the input queue.
  XLS_ASSERT_OK(input_queue.Write({Value(UBits(42, 32))}));

  // In the first iteration, the receive_if should read a value because the
  // proc state value (which is the receive_if predicate) is initialized to
  // true.
  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(42, 32))));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(0, 1))));

  // The second iteration should not read anything as the receive predicate
  // is now false. The data value of the receive (which is sent over the output
  // channel) should be zeros.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(0, 32))));

  // The third iteration should again read a value.
  XLS_ASSERT_OK(input_queue.Write({Value(UBits(123, 32))}));
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(123, 32))));

  EXPECT_TRUE(output_queue.IsEmpty());
}

TEST_P(ProcEvaluatorTestBase, ConditionalSendProc) {
  // Create an output-only proc with a by-one-counter which sends only
  // even values over a conditional send.
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("even_out", ChannelOps::kSendOnly,
                                     package.GetBitsType(32)));

  ProcBuilder pb("even", &package);
  BValue prev = pb.StateElement("prev", Value(UBits(0, 32)));
  BValue is_even = pb.Eq(pb.BitSlice(prev, /*start=*/0, /*width=*/1),
                         pb.Literal(UBits(0, 1)));
  pb.SendIf(channel, pb.Literal(Value::Token()), is_even, prev);
  BValue new_value = pb.Add(prev, pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({new_value}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator = GetParam().CreateEvaluator(
      FindProc("even", &package), queue_manager.get());

  ChannelQueue& queue = queue_manager->GetQueue(channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_instance,
      queue_manager->elaboration().GetUniqueInstance(channel));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(0, 32))));

  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(queue.IsEmpty());

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(2, 32))));

  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(queue.IsEmpty());

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(4, 32))));
}

TEST_P(ProcEvaluatorTestBase, UnconditionalNextProc) {
  // Create an output-only proc which increments its counter value each
  // iteration, using explicit next_value nodes.
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("counter_out", ChannelOps::kSendOnly,
                                     package.GetBitsType(32)));

  ProcBuilder pb("counter", &package);
  BValue counter = pb.StateElement("counter", Value(UBits(0, 32)));
  pb.Send(channel, pb.Literal(Value::Token()), counter);
  BValue incremented_counter = pb.Add(counter, pb.Literal(UBits(1, 32)));
  pb.Next(/*param=*/counter, /*value=*/incremented_counter);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator = GetParam().CreateEvaluator(
      FindProc("counter", &package), queue_manager.get());

  ChannelQueue& queue = queue_manager->GetQueue(channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_instance,
      queue_manager->elaboration().GetUniqueInstance(channel));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(0, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(1, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(2, 32))));
}

TEST_P(ProcEvaluatorTestBase, ConditionalNextProc) {
  // Create an output-only proc which increments its counter value only every
  // other iteration.
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("slow_counter_out", ChannelOps::kSendOnly,
                                     package.GetBitsType(32)));

  ProcBuilder pb("slow_counter", &package);
  BValue counter = pb.StateElement("counter", Value(UBits(0, 32)));
  BValue iteration = pb.StateElement("iteration", Value(UBits(0, 32)));
  pb.Send(channel, pb.Literal(Value::Token()), counter);
  BValue incremented_counter = pb.Add(counter, pb.Literal(UBits(1, 32)));
  BValue odd_iteration = pb.Eq(pb.BitSlice(iteration, /*start=*/0, /*width=*/1),
                               pb.Literal(UBits(1, 1)));
  pb.Next(/*param=*/counter, /*value=*/incremented_counter,
          /*pred=*/odd_iteration);
  pb.Next(/*param=*/iteration,
          /*value=*/pb.Add(iteration, pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator = GetParam().CreateEvaluator(
      FindProc("slow_counter", &package), queue_manager.get());

  ChannelQueue& queue = queue_manager->GetQueue(channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_instance,
      queue_manager->elaboration().GetUniqueInstance(channel));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(0, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(0, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(1, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(1, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(2, 32))));
}

TEST_P(ProcEvaluatorTestBase, CollidingNextValuesProc) {
  // Create an output-only proc which increments its counter value only every
  // other iteration - but also tries to set the counter value to a different
  // value.
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package.CreateStreamingChannel("slow_counter_out", ChannelOps::kSendOnly,
                                     package.GetBitsType(32)));

  ProcBuilder pb("slow_counter", &package);
  BValue counter = pb.StateElement("counter", Value(UBits(0, 32)));
  BValue iteration = pb.StateElement("iteration", Value(UBits(0, 32)));
  pb.Send(channel, pb.Literal(Value::Token()), counter);
  BValue incremented_counter = pb.Add(counter, pb.Literal(UBits(1, 32)));
  BValue odd_iteration = pb.Eq(pb.BitSlice(iteration, /*start=*/0, /*width=*/1),
                               pb.Literal(UBits(1, 1)));
  pb.Next(/*param=*/counter, /*value=*/incremented_counter,
          /*pred=*/odd_iteration);
  pb.Next(/*param=*/counter, /*value=*/pb.Literal(UBits(0, 32)));
  pb.Next(/*param=*/iteration,
          /*value=*/pb.Add(iteration, pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator = GetParam().CreateEvaluator(
      FindProc("slow_counter", &package), queue_manager.get());

  ChannelQueue& queue = queue_manager->GetQueue(channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_instance,
      queue_manager->elaboration().GetUniqueInstance(channel));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_EQ(queue.GetSize(), 1);
  EXPECT_THAT(queue.Read(), Optional(Value(UBits(0, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(evaluator->Tick(*continuation),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("Multiple active next values for param "
                                 "\"counter\" in a single activation")));
}

TEST_P(ProcEvaluatorTestBase, OneToTwoDemux) {
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

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(&package);
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());

  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * dir_queue,
                           queue_manager->GetQueueByName("dir"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * in_queue,
                           queue_manager->GetQueueByName("in"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * a_queue,
                           queue_manager->GetQueueByName("a"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * b_queue,
                           queue_manager->GetQueueByName("b"));

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_a_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_a));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * ch_b_instance,
      queue_manager->elaboration().GetUniqueInstance(ch_b));

  // Set the direction to output B and write a bunch of values.
  XLS_ASSERT_OK(dir_queue->Write(Value(UBits(0, 1))));
  XLS_ASSERT_OK(in_queue->Write(Value(UBits(1, 32))));
  XLS_ASSERT_OK(in_queue->Write(Value(UBits(2, 32))));
  XLS_ASSERT_OK(in_queue->Write(Value(UBits(3, 32))));
  XLS_ASSERT_OK(in_queue->Write(Value(UBits(4, 32))));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  // Tick twice and verify that the expected outputs appear at output B.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_b_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_TRUE(a_queue->IsEmpty());
  EXPECT_FALSE(b_queue->IsEmpty());
  EXPECT_THAT(b_queue->Read(), Optional(Value(UBits(1, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_b_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(b_queue->Read(), Optional(Value(UBits(2, 32))));
  EXPECT_TRUE(a_queue->IsEmpty());
  EXPECT_TRUE(b_queue->IsEmpty());

  // Switch direction to output A.
  XLS_ASSERT_OK(dir_queue->Write(Value(UBits(1, 1))));

  // Tick twice and verify that the expectedoutputs appear at output A.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_a_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_FALSE(a_queue->IsEmpty());
  EXPECT_TRUE(b_queue->IsEmpty());

  EXPECT_THAT(a_queue->Read(), Optional(Value(UBits(3, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = ch_a_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_THAT(a_queue->Read(), Optional(Value(UBits(4, 32))));
  EXPECT_TRUE(a_queue->IsEmpty());
  EXPECT_TRUE(b_queue->IsEmpty());
}

TEST_P(ProcEvaluatorTestBase, StatelessProc) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_in,
      package->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_out,
      package->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  ProcBuilder pb("stateless", package.get());
  BValue receive = pb.Receive(channel_in, pb.Literal(Value::Token()));
  pb.Send(channel_out, pb.TupleIndex(receive, 0),
          pb.Add(pb.TupleIndex(receive, 1), pb.Literal(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());
  ChannelQueue& input_queue = queue_manager->GetQueue(channel_in);
  ChannelQueue& output_queue = queue_manager->GetQueue(channel_out);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_out_instance,
      queue_manager->elaboration().GetUniqueInstance(channel_out));

  XLS_ASSERT_OK(input_queue.Write({Value(UBits(1, 32))}));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_out_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(output_queue.Read(), Optional(Value(UBits(43, 32))));

  // The next state should be empty.
  EXPECT_THAT(continuation->GetState(), ElementsAre());
}

TEST_P(ProcEvaluatorTestBase, MultiStateElementProc) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_in,
      package->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                      package->GetBitsType(32)));

  // Proc has two state elements:
  //  a: accumulator starting at 0
  //  b: de-cumulator starting at 100
  ProcBuilder pb("multistate", package.get());
  BValue a_state = pb.StateElement("a", Value(UBits(0, 32)));
  BValue b_state = pb.StateElement("b", Value(UBits(100, 32)));
  BValue receive = pb.Receive(channel_in, pb.Literal(Value::Token()));
  BValue data = pb.TupleIndex(receive, 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Add(a_state, data),
                                                  pb.Subtract(b_state, data)}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());
  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  ChannelQueue& input_queue = queue_manager->GetQueue(channel_in);
  XLS_ASSERT_OK(input_queue.Write({Value(UBits(1, 32))}));
  XLS_ASSERT_OK(input_queue.Write({Value(UBits(10, 32))}));
  XLS_ASSERT_OK(input_queue.Write({Value(UBits(20, 32))}));

  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(),
              ElementsAre(Value(UBits(1, 32)), Value(UBits(99, 32))));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(),
              ElementsAre(Value(UBits(11, 32)), Value(UBits(89, 32))));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(),
              ElementsAre(Value(UBits(31, 32)), Value(UBits(69, 32))));
}

TEST_P(ProcEvaluatorTestBase, NonBlockingReceives) {
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

  TokenlessProcBuilder pb("nb_recv", /*token_name=*/"tok", package.get());

  auto [in0_data, in0_valid] = pb.ReceiveNonBlocking(in0);
  auto [in1_data, in1_valid] = pb.ReceiveNonBlocking(in1);
  auto [in2_data, in2_valid] = pb.ReceiveNonBlocking(in2);

  BValue sum = pb.Literal(UBits(0, 32));

  BValue add_sum_in0 = pb.Add(sum, in0_data);
  BValue sum0 = pb.Select(in0_valid, {sum, add_sum_in0});

  BValue add_sum0_in1 = pb.Add(sum0, in1_data);
  BValue sum1 = pb.Select(in1_valid, {sum0, add_sum0_in1});

  BValue add_sum1_in2 = pb.Add(sum1, in2_data);
  BValue sum2 = pb.Select(in2_valid, {sum1, add_sum1_in2});

  pb.Send(out0, sum2);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());
  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  ChannelQueue& in0_queue = queue_manager->GetQueue(in0);
  ChannelQueue& in1_queue = queue_manager->GetQueue(in1);
  ChannelQueue& in2_queue = queue_manager->GetQueue(in2);
  ChannelQueue& out0_queue = queue_manager->GetQueue(out0);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * out0_instance,
      queue_manager->elaboration().GetUniqueInstance(out0));

  // Initialize the single value queue.
  XLS_ASSERT_OK(in2_queue.Write(Value(UBits(10, 32))));

  // All other channels are non-blocking, so run even if the queues are empty.
  EXPECT_TRUE(in0_queue.IsEmpty());
  EXPECT_TRUE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = out0_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_TRUE(in0_queue.IsEmpty());
  EXPECT_TRUE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_FALSE(out0_queue.IsEmpty());

  EXPECT_THAT(out0_queue.Read(), Optional(Value(UBits(10, 32))));

  // Run with only in1 (and in2) having data.
  XLS_ASSERT_OK(in1_queue.Write(Value(UBits(5, 32))));
  EXPECT_TRUE(in0_queue.IsEmpty());
  EXPECT_FALSE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = out0_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_TRUE(in0_queue.IsEmpty());
  EXPECT_TRUE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_FALSE(out0_queue.IsEmpty());

  EXPECT_THAT(out0_queue.Read(), Optional(Value(UBits(15, 32))));

  // Run with only in0 (and in2) having data.
  XLS_ASSERT_OK(in0_queue.Write(Value(UBits(7, 32))));
  EXPECT_FALSE(in0_queue.IsEmpty());
  EXPECT_TRUE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = out0_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_TRUE(in0_queue.IsEmpty());
  EXPECT_TRUE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_FALSE(out0_queue.IsEmpty());

  EXPECT_THAT(out0_queue.Read(), Optional(Value(UBits(17, 32))));

  // Run with all channels having data.
  XLS_ASSERT_OK(in0_queue.Write(Value(UBits(11, 32))));
  XLS_ASSERT_OK(in1_queue.Write(Value(UBits(22, 32))));
  EXPECT_FALSE(in0_queue.IsEmpty());
  EXPECT_FALSE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = out0_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_TRUE(in0_queue.IsEmpty());
  EXPECT_TRUE(in1_queue.IsEmpty());
  EXPECT_FALSE(in2_queue.IsEmpty());
  EXPECT_FALSE(out0_queue.IsEmpty());

  EXPECT_THAT(out0_queue.Read(), Optional(Value(UBits(43, 32))));
}

TEST_P(ProcEvaluatorTestBase, NonBlockingReceivesZeroRecv) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Channel * in0, package->CreateStreamingChannel(
                                              "in0", ChannelOps::kReceiveOnly,
                                              package->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * out0, package->CreateStreamingChannel(
                                               "out0", ChannelOps::kSendOnly,
                                               package->GetBitsType(32)));

  TokenlessProcBuilder pb("nb_recv", /*token_name=*/"tok", package.get());

  auto [in0_data, in0_valid] = pb.ReceiveNonBlocking(in0);
  pb.Send(out0, in0_data);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());
  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  ChannelQueue& in0_queue = queue_manager->GetQueue(in0);
  ChannelQueue& out0_queue = queue_manager->GetQueue(out0);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * out0_instance,
      queue_manager->elaboration().GetUniqueInstance(out0));

  EXPECT_TRUE(in0_queue.IsEmpty());
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = out0_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  // Reads on an empty channel should return default value of zero.
  EXPECT_THAT(out0_queue.Read(), Optional(Value(UBits(0, 32))));
}

TEST_P(ProcEvaluatorTestBase, ProcSetState) {
  auto package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      package->CreateStreamingChannel("iota_out", ChannelOps::kSendOnly,
                                      package->GetBitsType(32)));

  // Create an output-only proc which counts up by 7 starting at 42.
  ProcBuilder pb("iota", package.get());
  BValue counter = pb.StateElement("cnt", Value(UBits(42, 32)));
  pb.Send(channel, pb.Literal(Value::Token()), counter);
  BValue new_value = pb.Add(counter, pb.Literal(UBits(7, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({new_value}));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator = GetParam().CreateEvaluator(
      FindProc("iota", package.get()), queue_manager.get());
  ChannelQueue& ch0_queue = queue_manager->GetQueue(channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_instance,
      queue_manager->elaboration().GetUniqueInstance(channel));

  ASSERT_TRUE(ch0_queue.IsEmpty());

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  // Before running, the state should be the initial value.
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(42, 32))));

  // Override state.
  XLS_ASSERT_OK(
      continuation->SetState(std::vector<Value>{Value(UBits(20, 32))}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(20, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(27, 32))));

  // Run again
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(34, 32))));

  // Set state and run again
  XLS_ASSERT_OK(
      continuation->SetState(std::vector<Value>{Value(UBits(100, 32))}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(100, 32))));

  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));
  EXPECT_THAT(continuation->GetState(), ElementsAre(Value(UBits(107, 32))));

  // Check that each tick sent the right value on the output port.
  ASSERT_EQ(ch0_queue.GetSize(), 3);

  EXPECT_THAT(ch0_queue.Read(), Optional(Value(UBits(20, 32))));
  EXPECT_THAT(ch0_queue.Read(), Optional(Value(UBits(27, 32))));
  EXPECT_THAT(ch0_queue.Read(), Optional(Value(UBits(100, 32))));

  EXPECT_TRUE(ch0_queue.IsEmpty());
}

}  // namespace
}  // namespace xls
