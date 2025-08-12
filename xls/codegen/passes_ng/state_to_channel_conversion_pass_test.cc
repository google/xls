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

#include "xls/codegen/passes_ng/state_to_channel_conversion_pass.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
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
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Optional;

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
      public testing::WithParamInterface<TestParam> {
 public:
  void SetUp() final { package_ = CreatePackage(); }

  std::unique_ptr<Package> package_;
};

// A helper that creates and wraps a proc which simply counts a state value
// called "count" up by a configured delta on every activation. A subclass can
// override `Init()`, `BuildProc()`, and/or `NextRound()` to implement more
// elaborate variations of the proc.
class CounterProcHelper {
 public:
  CounterProcHelper(Package* package, const TestParam& param,
                    std::string_view proc_name)
      : package_(package), param_(param), proc_name_(proc_name) {}

  virtual ~CounterProcHelper() = default;

  absl::Status Init(uint32_t start_value, uint32_t delta_each_round) {
    // Build a proc the traditional way with a state.
    const std::string out_channel_name = absl::StrCat(proc_name_, "_out");
    XLS_ASSIGN_OR_RETURN(Channel * out_channel,
                         package_->CreateStreamingChannel(
                             out_channel_name, ChannelOps::kSendOnly,
                             package_->GetBitsType(32)));

    ProcBuilder pb(proc_name_, package_);
    XLS_ASSIGN_OR_RETURN(
        Proc * proc, BuildProc(pb, out_channel, start_value, delta_each_round));
    XLS_EXPECT_OK(package_->SetTop(proc));

    // Convert state to a state channel with loop-back.
    XLS_ASSIGN_OR_RETURN(bool converted, Run());
    EXPECT_TRUE(converted) << "Expected some change to have occured";

    {  // Test idempotency: running it the second time: no change.
      absl::StatusOr<bool> twice = Run();
      XLS_EXPECT_OK(twice) << "Expected successful outcome second time around";
      EXPECT_FALSE(*twice) << "No change expected second time around";
    }
    proc_ = IrTestBase::FindProc(proc_name_, package_);

    XLS_ASSIGN_OR_RETURN(
        count_state_channel_,
        package_->GetChannel(GetStateChannelName(kCounterStateElementName)));
    XLS_ASSIGN_OR_RETURN(out_channel_, package_->GetChannel(out_channel_name));

    queue_manager_ = param_.CreateQueueManager(package_);
    evaluator_ = param_.CreateEvaluator(proc, queue_manager_.get());

    // We expect the state channel to have as initial value as the init of
    // state.
    EXPECT_THAT(count_state_channel_->initial_values(),
                ElementsAre(Value(UBits(start_value, 32))));

    XLS_ASSIGN_OR_RETURN(
        count_state_channel_instance_,
        queue_manager_->elaboration().GetUniqueInstance(count_state_channel_));

    XLS_ASSIGN_OR_RETURN(
        out_channel_instance_,
        queue_manager_->elaboration().GetUniqueInstance(out_channel_));

    continuation_ = evaluator_->NewContinuation(
        queue_manager_->elaboration().GetUniqueInstance(proc).value());

    // Prime our state channel with the initial value enqueued.
    XLS_RETURN_IF_ERROR(count_state_channel_queue().Write(
        count_state_channel_->initial_values()[0]));

    // Expected initial state.
    EXPECT_EQ(count_state_channel_queue().GetSize(), 1);
    EXPECT_TRUE(out_queue().IsEmpty());
    return absl::OkStatus();
  }

  // Executes a "round" of ticks of the proc. This can be overridden to match
  // the behavior of an overridden `BuildProc()` function.
  virtual void NextRound() {
    // New data out.
    EXPECT_THAT(Tick(),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kSentOnChannel,
                    .channel_instance = out_channel_instance(),
                    .progress_made = true}));

    // Updated state.
    EXPECT_THAT(Tick(),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kSentOnChannel,
                    .channel_instance = count_state_channel_instance(),
                    .progress_made = true}));

    // Done this round.
    EXPECT_THAT(Tick(), IsOkAndHolds(TickResult{
                            .execution_state = TickExecutionState::kCompleted,
                            .channel_instance = std::nullopt,
                            .progress_made = true}));
  }

  absl::StatusOr<TickResult> Tick() const {
    return evaluator_->Tick(*continuation_);
  }

  ChannelInstance* count_state_channel_instance() {
    return count_state_channel_instance_;
  }

  ChannelInstance* out_channel_instance() { return out_channel_instance_; }
  ChannelQueue& out_queue() { return queue_manager_->GetQueue(out_channel_); }

  ChannelQueue& count_state_channel_queue() {
    return queue_manager_->GetQueue(count_state_channel_);
  }

  std::string GetStateChannelName(std::string_view state_element_name) {
    return absl::StrCat(proc_name_, state_element_name, "_channel");
  }

  Package* package() { return package_; }
  ChannelQueueManager* queue_manager() { return queue_manager_.get(); }

 protected:
  static constexpr std::string_view kCounterStateElementName = "count";

  // Builds the proc using the given builder and returns it. This is called
  // during `Init()` invocation.
  virtual absl::StatusOr<Proc*> BuildProc(ProcBuilder& pb, Channel* out_channel,
                                          uint32_t start_value,
                                          uint32_t delta_each_round) {
    BValue counter = pb.StateElement(kCounterStateElementName,
                                     Value(UBits(start_value, 32)));
    pb.Send(out_channel, pb.Literal(Value::Token()), counter);
    BValue new_value = pb.Add(counter, pb.Literal(UBits(delta_each_round, 32)));
    return pb.Build({new_value});
  }

 private:
  absl::StatusOr<bool> Run() {
    PassResults results;
    CodegenContext context;
    return StateToChannelConversionPass().Run(package_, CodegenPassOptions(),
                                              &results, context);
  }

  Package* const package_;
  const TestParam& param_;
  const std::string_view proc_name_;

  Proc* proc_ = nullptr;
  Channel* count_state_channel_ = nullptr;
  Channel* out_channel_ = nullptr;
  ChannelInstance* count_state_channel_instance_ = nullptr;
  ChannelInstance* out_channel_instance_ = nullptr;
  std::unique_ptr<ChannelQueueManager> queue_manager_;
  std::unique_ptr<ProcEvaluator> evaluator_;
  std::unique_ptr<ProcContinuation> continuation_;
};

// A variation on `CounterProcHelper` that uses predicated state reads and
// writes to disable further counting when a configured max value is reached.
class CounterProcWithMaxHelper : public CounterProcHelper {
 public:
  using CounterProcHelper::CounterProcHelper;

  absl::Status Init(uint32_t start_value, uint32_t delta_each_round,
                    uint32_t max) {
    max_ = max;
    XLS_RETURN_IF_ERROR(CounterProcHelper::Init(start_value, delta_each_round));
    XLS_ASSIGN_OR_RETURN(
        enable_state_channel_,
        package()->GetChannel(GetStateChannelName(kEnableStateElementName)));
    XLS_ASSIGN_OR_RETURN(enable_state_channel_instance_,
                         queue_manager()->elaboration().GetUniqueInstance(
                             enable_state_channel_));

    EXPECT_THAT(enable_state_channel_->initial_values(),
                ElementsAre(Value(UBits(1, 1))));

    XLS_RETURN_IF_ERROR(enable_state_channel_queue().Write(
        enable_state_channel_->initial_values()[0]));

    return absl::OkStatus();
  }

  void NextRound() final {
    // New data out.
    EXPECT_THAT(Tick(),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kSentOnChannel,
                    .channel_instance = out_channel_instance(),
                    .progress_made = true}));

    // Updated enable flag.
    EXPECT_THAT(Tick(),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kSentOnChannel,
                    .channel_instance = count_state_channel_instance(),
                    .progress_made = true}));

    // Updated count.
    EXPECT_THAT(Tick(),
                IsOkAndHolds(TickResult{
                    .execution_state = TickExecutionState::kSentOnChannel,
                    .channel_instance = enable_state_channel_instance(),
                    .progress_made = true}));

    // Done this round.
    EXPECT_THAT(Tick(), IsOkAndHolds(TickResult{
                            .execution_state = TickExecutionState::kCompleted,
                            .channel_instance = std::nullopt,
                            .progress_made = true}));
  }

  ChannelInstance* enable_state_channel_instance() {
    return enable_state_channel_instance_;
  }

  ChannelQueue& enable_state_channel_queue() {
    return queue_manager()->GetQueue(enable_state_channel_);
  }

 protected:
  absl::StatusOr<Proc*> BuildProc(ProcBuilder& pb, Channel* out_channel,
                                  uint32_t start_value,
                                  uint32_t delta_each_round) final {
    BValue enable = pb.StateElement("enable", Value(UBits(1, 1)));
    BValue counter = pb.StateElement(kCounterStateElementName,
                                     Value(UBits(start_value, 32)), enable);
    pb.SendIf(out_channel, pb.Literal(Value::Token()), enable, counter);
    pb.Next(counter, pb.Add(counter, pb.Literal(UBits(delta_each_round, 32))),
            enable);
    pb.Next(enable, pb.ULt(counter, pb.Literal(UBits(max_, 32))));
    return pb.Build();
  }

 private:
  static constexpr std::string_view kEnableStateElementName = "enable";

  uint32_t max_;
  Channel* enable_state_channel_;
  ChannelInstance* enable_state_channel_instance_ = nullptr;
};

TEST_P(StateChannelConversionTest, CounterProc) {
  constexpr uint32_t kStartValue = 123;
  constexpr uint32_t kDeltaEachRound = 7;

  CounterProcHelper helper(package_.get(), GetParam(), "iota");
  XLS_ASSERT_OK(helper.Init(kStartValue, kDeltaEachRound));

  uint32_t expected_count = kStartValue;

  helper.NextRound();

  // Would be neat if there was count_state_channel_queue.Peek() to inspect
  // state.
  EXPECT_EQ(helper.count_state_channel_queue().GetSize(),
            1);  // state channel looped back.

  // output has one result.
  EXPECT_EQ(helper.out_queue().GetSize(), 1);

  // ... with the expected value.
  EXPECT_THAT(helper.out_queue().Read(),
              Optional(Value(UBits(expected_count, 32))));
  EXPECT_TRUE(helper.out_queue().IsEmpty());  // consumed it.

  // Let's do this a few more times and empty the channel-queue in the end.
  constexpr int kRounds = 5;
  for (int i = 0; i < kRounds; ++i) {
    helper.NextRound();
  }

  ASSERT_EQ(helper.out_queue().GetSize(), kRounds);

  for (int i = 0; i < kRounds; ++i) {
    expected_count += kDeltaEachRound;
    EXPECT_THAT(helper.out_queue().Read(),
                Optional(Value(UBits(expected_count, 32))));
  }

  EXPECT_TRUE(helper.out_queue().IsEmpty());
}

TEST_P(StateChannelConversionTest, CounterProcWithStatePredicates) {
  constexpr uint32_t kStartValue = 123;
  constexpr uint32_t kDeltaEachRound = 7;
  constexpr uint32_t kMax = 130;

  CounterProcWithMaxHelper helper(package_.get(), GetParam(), "iota");
  XLS_ASSERT_OK(helper.Init(kStartValue, kDeltaEachRound, kMax));

  // First round...
  helper.NextRound();

  EXPECT_EQ(helper.count_state_channel_queue().GetSize(), 1);
  EXPECT_EQ(helper.enable_state_channel_queue().GetSize(), 1);

  // output has one result.
  EXPECT_EQ(helper.out_queue().GetSize(), 1);

  // ... with the expected value.
  EXPECT_THAT(helper.out_queue().Read(),
              Optional(Value(UBits(kStartValue, 32))));
  EXPECT_TRUE(helper.out_queue().IsEmpty());  // consumed it.

  // Second round...
  helper.NextRound();

  EXPECT_EQ(helper.count_state_channel_queue().GetSize(), 1);
  EXPECT_EQ(helper.enable_state_channel_queue().GetSize(), 1);

  // output has one result.
  EXPECT_EQ(helper.out_queue().GetSize(), 1);

  // ... with the expected value.
  EXPECT_THAT(helper.out_queue().Read(), Optional(Value(UBits(kMax, 32))));
  EXPECT_TRUE(helper.out_queue().IsEmpty());  // consumed it.

  // A round after hitting max. Now the predicates for reading and writing the
  // counter should be false.

  // Updated enable flag (this is unconditional).
  EXPECT_THAT(helper.Tick(),
              IsOkAndHolds(TickResult{
                  .execution_state = TickExecutionState::kSentOnChannel,
                  .channel_instance = helper.enable_state_channel_instance(),
                  .progress_made = true}));

  // Done this round.
  EXPECT_THAT(
      helper.Tick(),
      IsOkAndHolds(TickResult{.execution_state = TickExecutionState::kCompleted,
                              .channel_instance = std::nullopt,
                              .progress_made = true}));

  // It should not have read the count value that was already in the queue.
  EXPECT_EQ(helper.count_state_channel_queue().GetSize(), 1);

  EXPECT_EQ(helper.enable_state_channel_queue().GetSize(), 1);
  EXPECT_EQ(helper.out_queue().GetSize(), 0);
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
}  // namespace xls::verilog
