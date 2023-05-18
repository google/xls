// Copyright 2023 The XLS Authors
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

#include "xls/scheduling/channel_legalization_pass.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/scheduling/scheduling_pass_pipeline.h"

namespace xls {
namespace {

using status_testing::IsOk;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::Eq;
using testing::HasSubstr;
using testing::Optional;

SchedulingPass* DefaultSchedulingPassPipeline() {
  static SchedulingPass* singleton = CreateSchedulingPassPipeline().release();
  return singleton;
}

SchedulingPass* ChannelLegalizationPassOnly() {
  static SchedulingPass* singleton = new ChannelLegalizationPass();
  return singleton;
}

struct TestParam {
  using evaluation_function = std::function<absl::Status(
      SerialProcRuntime*,
      std::optional<MultipleChannelOpsLegalizationStrictness>)>;
  std::string_view test_name;
  std::string_view ir_text;
  absl::flat_hash_map<MultipleChannelOpsLegalizationStrictness,
                      ::testing::Matcher<absl::StatusOr<bool>>>
      builder_matcher = {};
  evaluation_function evaluate =
      [](SerialProcRuntime* interpreter,
         std::optional<MultipleChannelOpsLegalizationStrictness> strictness)
      -> absl::Status {
    GTEST_MESSAGE_("Evaluation is not implemented for this test!",
                   ::testing::TestPartResult::kSkip);
    return absl::OkStatus();
  };
};

class ChannelLegalizationPassTest
    : public testing::TestWithParam<
          std::tuple<TestParam, SchedulingPass*,
                     MultipleChannelOpsLegalizationStrictness>> {
 protected:
  absl::StatusOr<bool> Run(Package* package) {
    SchedulingPass* pass = std::get<1>(GetParam());
    SchedulingPassOptions options;
    XLS_ASSIGN_OR_RETURN(options.delay_estimator, GetDelayEstimator("unit"));
    options.scheduling_options.clock_period_ps(100);
    options.scheduling_options.multiple_channel_ops_legalization_strictness(
        std::get<2>(GetParam()));
    SchedulingPassResults results;
    SchedulingUnit<Package*> unit;
    unit.ir = package;
    return pass->Run(&unit, options, &results);
  }
};

TestParam kTestParameters[] = {
    TestParam{
        .test_name = "SingleProcBackToBackDataSwitchingOps",
        .ir_text = R"(package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc my_proc(tok: token, init={}) {
  recv0: (token, bits[32]) = receive(tok, channel_id=0)
  recv0_tok: token = tuple_index(recv0, index=0)
  recv0_data: bits[32] = tuple_index(recv0, index=1)
  recv1: (token, bits[32]) = receive(recv0_tok, channel_id=0)
  recv1_tok: token = tuple_index(recv1, index=0)
  recv1_data: bits[32] = tuple_index(recv1, index=1)
  send0: token = send(recv1_tok, recv1_data, channel_id=1)
  send1: token = send(send0, recv0_data, channel_id=1)
  next(send1)
}
    )",
        .builder_matcher =
            {
                {MultipleChannelOpsLegalizationStrictness::
                     kProvenMutuallyExclusive,
                 StatusIs(absl::StatusCode::kInvalidArgument,
                          HasSubstr("Could not prove"))},
                {MultipleChannelOpsLegalizationStrictness::kTotalOrder,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered,
                 IsOkAndHolds(true)},
                // Build should be OK, but will fail at runtime.
                {MultipleChannelOpsLegalizationStrictness::
                     kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::
                     kArbitraryStaticOrder,
                 IsOkAndHolds(true)},
            },
        .evaluate = [](SerialProcRuntime* interpreter,
                       std::optional<MultipleChannelOpsLegalizationStrictness>
                           strictness) -> absl::Status {
          constexpr int64_t kMaxTicks = 1000;
          constexpr int64_t kNumInputs = 32;

          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * inq,
              interpreter->queue_manager().GetQueueByName("in"));
          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * outq,
              interpreter->queue_manager().GetQueueByName("out"));

          for (int64_t i = 0; i < kNumInputs; ++i) {
            XLS_RETURN_IF_ERROR(inq->Write(Value(UBits(i, /*bit_count=*/32))));
          }
          absl::flat_hash_map<Channel*, int64_t> output_count{
              {outq->channel(), kNumInputs}};
          absl::Status interpreter_status =
              interpreter->TickUntilOutput(output_count, kMaxTicks).status();
          if (strictness.has_value() &&
              strictness.value() == MultipleChannelOpsLegalizationStrictness::
                                        kRuntimeMutuallyExclusive) {
            EXPECT_THAT(interpreter_status,
                        StatusIs(absl::StatusCode::kAborted,
                                 HasSubstr("was not mutually exclusive.")));
            // Return early, we have no output to check.
            return absl::OkStatus();
          }
          XLS_EXPECT_OK(interpreter_status);
          for (int64_t i = 0; i < kNumInputs; ++i) {
            EXPECT_FALSE(outq->IsEmpty());
            int64_t flip_evens_and_odds = i;
            if (i % 2 == 0) {
              flip_evens_and_odds++;
            } else {
              flip_evens_and_odds--;
            }
            EXPECT_THAT(outq->Read(),
                        Optional(Eq(Value(
                            UBits(flip_evens_and_odds, /*bit_count=*/32)))));
          }

          return absl::OkStatus();
        },
    },
    TestParam{
        .test_name = "TwoProcsMutuallyExclusive",
        .ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc proc_a(tok: token, pred: bits[1], init={1}) {
  recv: (token, bits[32]) = receive(tok, predicate=pred, channel_id=0)
  recv_tok: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  send: token = send(recv_tok, recv_data, predicate=pred, channel_id=1)
  next_pred: bits[1] = not(pred)
  next(send, next_pred)
}

proc proc_b(tok: token, pred: bits[1], init={0}) {
  recv: (token, bits[32]) = receive(tok, predicate=pred, channel_id=0)
  recv_tok: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  send: token = send(recv_tok, recv_data, predicate=pred, channel_id=1)
  next_pred: bits[1] = not(pred)
  next(send, next_pred)
}
      )",
        .builder_matcher =
            {
                {MultipleChannelOpsLegalizationStrictness::
                     kProvenMutuallyExclusive,
                 StatusIs(absl::StatusCode::kInvalidArgument,
                          HasSubstr("Could not prove"))},
                {MultipleChannelOpsLegalizationStrictness::kTotalOrder,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::
                     kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::
                     kArbitraryStaticOrder,
                 IsOkAndHolds(true)},
            },
        .evaluate = [](SerialProcRuntime* interpreter,
                       std::optional<MultipleChannelOpsLegalizationStrictness>
                           strictness) -> absl::Status {
          constexpr int64_t kMaxTicks = 1000;
          constexpr int64_t kNumInputs = 32;

          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * inq,
              interpreter->queue_manager().GetQueueByName("in"));
          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * outq,
              interpreter->queue_manager().GetQueueByName("out"));

          for (int64_t i = 0; i < kNumInputs; ++i) {
            XLS_RETURN_IF_ERROR(inq->Write(Value(UBits(i, /*bit_count=*/32))));
          }
          absl::flat_hash_map<Channel*, int64_t> output_count{
              {outq->channel(), kNumInputs}};
          XLS_RETURN_IF_ERROR(
              interpreter->TickUntilOutput(output_count, kMaxTicks).status());
          for (int64_t i = 0; i < kNumInputs; ++i) {
            EXPECT_FALSE(outq->IsEmpty());
            EXPECT_THAT(outq->Read(),
                        Optional(Eq(Value(UBits(i, /*bit_count=*/32)))));
          }

          return absl::OkStatus();
        },
    },
    TestParam{
        .test_name = "TwoProcsAlwaysFiringCausesError",
        .ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc proc_a(tok: token, init={}) {
  recv: (token, bits[32]) = receive(tok, channel_id=0)
  recv_tok: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  send: token = send(recv_tok, recv_data, channel_id=1)
  next(send)
}

proc proc_b(tok: token, init={}) {
  recv: (token, bits[32]) = receive(tok, channel_id=0)
  recv_tok: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  send: token = send(recv_tok, recv_data, channel_id=1)
  next(send)
}
      )",
        .builder_matcher =
            {
                {MultipleChannelOpsLegalizationStrictness::
                     kProvenMutuallyExclusive,
                 StatusIs(absl::StatusCode::kInvalidArgument,
                          HasSubstr("Could not prove"))},
                {MultipleChannelOpsLegalizationStrictness::kTotalOrder,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::
                     kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::
                     kArbitraryStaticOrder,
                 IsOkAndHolds(true)},
            },
        .evaluate = [](SerialProcRuntime* interpreter,
                       std::optional<MultipleChannelOpsLegalizationStrictness>
                           strictness) -> absl::Status {
          if (!strictness.has_value()) {
            // Skip evaluation before adding the adapter, the test will deadlock
            // without an adapter.
            return absl::OkStatus();
          }
          constexpr int64_t kMaxTicks = 1000;
          constexpr int64_t kNumInputs = 32;

          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * outq,
              interpreter->queue_manager().GetQueueByName("out"));

          absl::flat_hash_map<Channel*, int64_t> output_count{
              {outq->channel(), kNumInputs}};
          EXPECT_THAT(
              interpreter->TickUntilOutput(output_count, kMaxTicks).status(),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("Activation for node recv was not mutually "
                                 "exclusive.")));

          return absl::OkStatus();
        },
    },
    TestParam{
        .test_name = "SingleProcWithPartialOrder",
        .ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan pred(bits[2], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

top proc my_proc(tok: token, init={}) {
  pred_recv: (token, bits[2]) = receive(tok, channel_id=2)
  pred_token: token = tuple_index(pred_recv, index=0)
  pred_data: bits[2] = tuple_index(pred_recv, index=1)
  pred0: bits[1] = bit_slice(pred_data, start=0, width=1)
  pred1: bits[1] = bit_slice(pred_data, start=1, width=1)
  recv0: (token, bits[32]) = receive(pred_token, channel_id=0)
  recv0_tok: token = tuple_index(recv0, index=0)
  recv0_data: bits[32] = tuple_index(recv0, index=1)
  recv1: (token, bits[32]) = receive(recv0_tok, channel_id=0, predicate=pred0)
  recv1_tok: token = tuple_index(recv1, index=0)
  recv1_data: bits[32] = tuple_index(recv1, index=1)
  recv2: (token, bits[32]) = receive(recv0_tok, channel_id=0, predicate=pred1)
  recv2_tok: token = tuple_index(recv2, index=0)
  recv2_data: bits[32] = tuple_index(recv2, index=1)
  all_recv_tok: token = after_all(recv0_tok, recv1_tok, recv2_tok)
  send0: token = send(all_recv_tok, recv0_data, channel_id=1)
  send1: token = send(send0, recv1_data, predicate=pred0, channel_id=1)
  send2: token = send(send0, recv2_data, predicate=pred1, channel_id=1)
  all_send_tok: token = after_all(send0, send1, send2)
  next(all_send_tok)
}
      )",
        .builder_matcher =
            {
                {MultipleChannelOpsLegalizationStrictness::
                     kProvenMutuallyExclusive,
                 StatusIs(absl::StatusCode::kInvalidArgument,
                          HasSubstr("Could not prove"))},
                {MultipleChannelOpsLegalizationStrictness::kTotalOrder,
                 StatusIs(absl::StatusCode::kInternal,
                          HasSubstr("is not totally ordered"))},
                {MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::
                     kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {MultipleChannelOpsLegalizationStrictness::
                     kArbitraryStaticOrder,
                 IsOkAndHolds(true)},
            },
        .evaluate = [](SerialProcRuntime* interpreter,
                       std::optional<MultipleChannelOpsLegalizationStrictness>
                           strictness) -> absl::Status {
          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * outq,
              interpreter->queue_manager().GetQueueByName("out"));
          auto run_with_pred = [interpreter, outq](bool fire0,
                                                   bool fire1) -> absl::Status {
            constexpr int64_t kMaxTicks = 1000;
            constexpr int64_t kNumInputs = 3;
            XLS_ASSIGN_OR_RETURN(
                ChannelQueue * inq,
                interpreter->queue_manager().GetQueueByName("in"));
            XLS_ASSIGN_OR_RETURN(
                ChannelQueue * predq,
                interpreter->queue_manager().GetQueueByName("pred"));

            // Clear queues from previous runs.
            while (!inq->IsEmpty()) {
              inq->Read();
            }
            while (!outq->IsEmpty()) {
              outq->Read();
            }
            for (int64_t i = 0; i < kNumInputs; ++i) {
              XLS_RETURN_IF_ERROR(
                  inq->Write(Value(UBits(i, /*bit_count=*/32))));
            }
            int64_t num_outputs = 1;  // first recv fires unconditionally
            int64_t pred = 0;
            if (fire0) {
              num_outputs++;
              pred |= 1;
            }
            if (fire1) {
              num_outputs++;
              pred |= 2;
            }
            XLS_RETURN_IF_ERROR(
                predq->Write(Value(UBits(pred, /*bit_count=*/2))));
            absl::flat_hash_map<Channel*, int64_t> output_count{
                {outq->channel(), num_outputs}};
            return interpreter->TickUntilOutput(output_count, kMaxTicks)
                .status();
          };

          absl::Status run_status;
          run_status = run_with_pred(false, false);
          EXPECT_THAT(run_status, IsOk());
          EXPECT_EQ(outq->GetSize(), 1);
          std::optional<Value> read_value = outq->Read();
          EXPECT_TRUE(read_value.has_value());
          EXPECT_EQ(read_value.value(), Value(UBits(0, 32)));

          run_status = run_with_pred(true, false);
          if (strictness.has_value() &&
              strictness.value() == MultipleChannelOpsLegalizationStrictness::
                                        kRuntimeMutuallyExclusive) {
            EXPECT_THAT(run_status,
                        StatusIs(absl::StatusCode::kAborted,
                                 HasSubstr("was not mutually exclusive")));
          } else {
            EXPECT_THAT(run_status, IsOk());
            read_value = outq->Read();
            EXPECT_TRUE(read_value.has_value());
            EXPECT_EQ(read_value.value(), Value(UBits(0, 32)));
            read_value = outq->Read();
            EXPECT_TRUE(read_value.has_value());
            EXPECT_EQ(read_value.value(), Value(UBits(1, 32)));
          }

          run_status = run_with_pred(false, true);
          if (strictness.has_value() &&
              strictness.value() == MultipleChannelOpsLegalizationStrictness::
                                        kRuntimeMutuallyExclusive) {
            EXPECT_THAT(run_status,
                        StatusIs(absl::StatusCode::kAborted,
                                 HasSubstr("was not mutually exclusive")));
          } else {
            EXPECT_THAT(run_status, IsOk());
            read_value = outq->Read();
            EXPECT_TRUE(read_value.has_value());
            EXPECT_EQ(read_value.value(), Value(UBits(0, 32)));
            read_value = outq->Read();
            EXPECT_TRUE(read_value.has_value());
            EXPECT_EQ(read_value.value(), Value(UBits(1, 32)));
          }

          run_status = run_with_pred(true, true);
          if (strictness.has_value() &&
              (strictness.value() == MultipleChannelOpsLegalizationStrictness::
                                         kRuntimeMutuallyExclusive ||
               strictness.value() ==
                   MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered)) {
            EXPECT_THAT(run_status,
                        StatusIs(absl::StatusCode::kAborted,
                                 HasSubstr("was not mutually exclusive")));
          } else {
            EXPECT_THAT(run_status, IsOk());
            read_value = outq->Read();
            EXPECT_TRUE(read_value.has_value());
            EXPECT_EQ(read_value.value(), Value(UBits(0, 32)));
            read_value = outq->Read();
            EXPECT_TRUE(read_value.has_value());
            // When both predicates are true, they are unordered with respect to
            // each other and any order is legal.
            EXPECT_TRUE(read_value.value() == Value(UBits(1, 32)) ||
                        read_value.value() == Value(UBits(2, 32)));
            Value prev_value = read_value.value();
            read_value = outq->Read();
            EXPECT_TRUE(read_value.has_value());
            EXPECT_TRUE(read_value.value() == Value(UBits(1, 32)) ||
                        read_value.value() == Value(UBits(2, 32)));
            EXPECT_NE(read_value.value(), prev_value);
          }

          return absl::OkStatus();
        },
    },
};

TEST_P(ChannelLegalizationPassTest, PassRuns) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> p,
      Parser::ParsePackage(std::get<0>(GetParam()).ir_text));

  MultipleChannelOpsLegalizationStrictness strictness = std::get<2>(GetParam());
  auto matchers = std::get<0>(GetParam()).builder_matcher;
  auto itr = matchers.find(strictness);
  if (itr == matchers.end()) {
    GTEST_SKIP();
  }
  ::testing::Matcher<absl::StatusOr<bool>> matcher = itr->second;
  EXPECT_THAT(Run(p.get()), matcher);
}

TEST_P(ChannelLegalizationPassTest, WillInlineAndVerify) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> p,
      Parser::ParsePackage(std::get<0>(GetParam()).ir_text));
  absl::StatusOr<bool> run_status;
  MultipleChannelOpsLegalizationStrictness strictness = std::get<2>(GetParam());
  auto matchers = std::get<0>(GetParam()).builder_matcher;
  auto itr = matchers.find(strictness);
  if (itr == matchers.end()) {
    GTEST_SKIP();
  }
  ::testing::Matcher<absl::StatusOr<bool>> matcher = itr->second;
  EXPECT_THAT((run_status = Run(p.get())), matcher);
  // If we expect the pass to complete, the result should be codegen'able.
  if (run_status.ok()) {
    EXPECT_THAT(VerifyPackage(p.get(), /*codegen=*/true), IsOk());
  }
}

TEST_P(ChannelLegalizationPassTest, EvaluatesCorrectly) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> p,
      Parser::ParsePackage(std::get<0>(GetParam()).ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SerialProcRuntime> interpreter,
                           CreateInterpreterSerialProcRuntime(p.get()));

  // Don't pass in strictness because the pass hasn't been run yet.
  XLS_EXPECT_OK(std::get<0>(GetParam())
                    .evaluate(interpreter.get(), /*strictness=*/std::nullopt));

  absl::StatusOr<bool> run_status = Run(p.get());
  if (!run_status.ok()) {
    GTEST_SKIP();
  }

  XLS_ASSERT_OK_AND_ASSIGN(interpreter,
                           CreateInterpreterSerialProcRuntime(p.get()));
  XLS_EXPECT_OK(std::get<0>(GetParam())
                    .evaluate(interpreter.get(), std::get<2>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(
    ChannelLegalizationPassTestInstantiation, ChannelLegalizationPassTest,
    ::testing::Combine(
        testing::ValuesIn(kTestParameters),
        testing::Values(DefaultSchedulingPassPipeline(),
                        ChannelLegalizationPassOnly()),
        testing::Values(
            MultipleChannelOpsLegalizationStrictness::kProvenMutuallyExclusive,
            MultipleChannelOpsLegalizationStrictness::kRuntimeMutuallyExclusive,
            MultipleChannelOpsLegalizationStrictness::kTotalOrder,
            MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered,
            MultipleChannelOpsLegalizationStrictness::kArbitraryStaticOrder)),
    [](const auto& info) {
      return absl::StrCat(std::get<0>(info.param).test_name, "_",
                          std::get<1>(info.param)->short_name(), "_",
                          AbslUnparseFlag(std::get<2>(info.param)));
    });

}  // namespace
}  // namespace xls
