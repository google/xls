// Copyright 2025 The XLS Authors
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

#include "xls/codegen/state_channel_conversion.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"

namespace xls {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Optional;
using verilog::State2ChannelConversionPass;

// Test suite ready to run against different ProcEvaluator implementations.
class TestParam {
 public:
  TestParam(
      std::function<std::unique_ptr<ProcEvaluator>(Proc*, ChannelQueueManager*)>
          evaluator_factory,
      std::function<std::unique_ptr<ChannelQueueManager>(Package*)>
          queue_manager_factory)
      : evaluator_factory_(std::move(evaluator_factory)),
        queue_manager_factory_(std::move(queue_manager_factory)) {}

  std::unique_ptr<ChannelQueueManager> CreateQueueManager(
      Package* package) const {
    return queue_manager_factory_(package);
  }

  std::unique_ptr<ProcEvaluator> CreateEvaluator(
      Proc* proc, ChannelQueueManager* queue_manager) const {
    return evaluator_factory_(proc, queue_manager);
  }

 private:
  std::function<std::unique_ptr<ProcEvaluator>(Proc*, ChannelQueueManager*)>
      evaluator_factory_;
  std::function<std::unique_ptr<ChannelQueueManager>(Package*)>
      queue_manager_factory_;
};

class StateChannelConversionTest
    : public IrTestBase,
      public testing::WithParamInterface<TestParam> {};

TEST_P(StateChannelConversionTest, CounterProc) {
  // A counter proc that starts with start value and emits a new value every
  // round kDeltaEachRound higher.
  constexpr uint32_t kStartValue = 123;
  constexpr uint32_t kDeltaEachRound = 7;

  auto package = CreatePackage();

  // Building a proc the traditional way with a state ...
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * channel,
        package->CreateStreamingChannel("iota_out", ChannelOps::kSendOnly,
                                        package->GetBitsType(32)));

    ProcBuilder pb("iota", package.get());
    BValue counter = pb.StateElement("count", Value(UBits(kStartValue, 32)));
    pb.Send(channel, pb.Literal(Value::Token()), counter);
    BValue new_value = pb.Add(counter, pb.Literal(UBits(kDeltaEachRound, 32)));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({new_value}));
    XLS_EXPECT_OK(package->SetTop(proc));

    // ... and then converting state to a state channel with loop-back
    auto converted = State2ChannelConversionPass(proc, ChannelConfig());
    XLS_ASSERT_OK(converted);
    EXPECT_TRUE(*converted) << "Expected some change to have occured";

    {  // Test idempotency: running it the second time: no change.
      auto twice = State2ChannelConversionPass(proc, ChannelConfig());
      XLS_EXPECT_OK(twice) << "Expected successful outcome second time around";
      EXPECT_FALSE(*twice) << "No change expected second time around";
    }
  }

  Proc* proc = FindProc("iota", package.get());

  // We have two channels: the state channel, and the output of the proc
  Channel* state_channel;
  Channel* iota_out_channel;

  XLS_ASSERT_OK_AND_ASSIGN(state_channel,
                           package->GetChannel("iotacount_channel"));
  XLS_ASSERT_OK_AND_ASSIGN(iota_out_channel, package->GetChannel("iota_out"));

  // We expect the state channel to have as initial value as the init of state.
  EXPECT_THAT(state_channel->initial_values(),
              ElementsAre(Value(UBits(kStartValue, 32))));

  std::unique_ptr<ChannelQueueManager> queue_manager =
      GetParam().CreateQueueManager(package.get());
  std::unique_ptr<ProcEvaluator> evaluator =
      GetParam().CreateEvaluator(proc, queue_manager.get());

  ChannelQueue& iota_out_queue = queue_manager->GetQueue(iota_out_channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * channel_instance,
      queue_manager->elaboration().GetUniqueInstance(iota_out_channel));

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * state_channel_instance,
      queue_manager->elaboration().GetUniqueInstance(state_channel));

  std::unique_ptr<ProcContinuation> continuation = evaluator->NewContinuation(
      queue_manager->elaboration().GetUniqueInstance(proc).value());

  // Prime our state channel with the initial value enqueued.
  ChannelQueue& state_channel_queue = queue_manager->GetQueue(state_channel);
  XLS_ASSERT_OK(state_channel_queue.Write(state_channel->initial_values()[0]));

  // Expected initial state.
  EXPECT_EQ(state_channel_queue.GetSize(), 1);
  EXPECT_TRUE(iota_out_queue.IsEmpty());

  uint32_t expected_count = kStartValue;

  // New data out.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = channel_instance,
                  .progress_made = true}));

  // Updated state.
  EXPECT_THAT(evaluator->Tick(*continuation),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = state_channel_instance,
                  .progress_made = true}));

  // Done this round.
  EXPECT_THAT(
      evaluator->Tick(*continuation),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  // Would be neat if there was state_channel_queue.Peek() to inspect state.
  EXPECT_EQ(state_channel_queue.GetSize(), 1);  // state channel looped back.

  // output has one result.
  EXPECT_EQ(iota_out_queue.GetSize(), 1);

  // ... with the expected value.
  EXPECT_THAT(iota_out_queue.Read(),
              Optional(Value(UBits(expected_count, 32))));
  EXPECT_TRUE(iota_out_queue.IsEmpty());  // consumed it.

  // Let's do this a few more times and empty the channel-queue in the end.
  constexpr int kRounds = 5;
  for (int i = 0; i < kRounds; ++i) {
    // Data out.
    EXPECT_THAT(evaluator->Tick(*continuation),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kSentOnChannel,
                    .channel_instance = channel_instance,
                    .progress_made = true}))
        << "Round " << i;

    // Updated state.
    EXPECT_THAT(evaluator->Tick(*continuation),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kSentOnChannel,
                    .channel_instance = state_channel_instance,
                    .progress_made = true}))
        << "Round " << i;

    // Done this round.
    EXPECT_THAT(evaluator->Tick(*continuation),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kCompleted,
                    .channel_instance = std::nullopt,
                    .progress_made = true}))
        << "Round " << i;
  }

  ASSERT_EQ(iota_out_queue.GetSize(), kRounds);

  for (int i = 0; i < kRounds; ++i) {
    expected_count += kDeltaEachRound;
    EXPECT_THAT(iota_out_queue.Read(),
                Optional(Value(UBits(expected_count, 32))));
  }

  EXPECT_TRUE(iota_out_queue.IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    StateChannelConversionProcEvaluator, StateChannelConversionTest,
    testing::Values(TestParam(
        [](Proc* proc, ChannelQueueManager* queue_manager)
            -> std::unique_ptr<ProcEvaluator> {
          return std::make_unique<ProcInterpreter>(proc, queue_manager);
        },
        [](Package* package) -> std::unique_ptr<ChannelQueueManager> {
          return ChannelQueueManager::Create(package).value();
        })));

}  // namespace
}  // namespace xls
