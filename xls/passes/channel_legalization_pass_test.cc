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

#include "xls/passes/channel_legalization_pass.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/passes.h"
#include "xls/passes/standard_pipeline.h"

namespace xls {
namespace {

using status_testing::IsOk;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::Eq;
using testing::HasSubstr;
using testing::Optional;

Pass* StandardPipelinePass() {
  static Pass* singleton = CreateStandardPassPipeline(3).release();
  return singleton;
}

Pass* ChannelLegalizationPassOnly() {
  static Pass* singleton = new ChannelLegalizationPass();
  return singleton;
}

enum class PassVariant {
  RunStandardPipelineNoInlineProcs,
  RunStandardPipelineInlineProcs,
  RunChannelLegalizationPassOnly,
};

std::string_view PassVariantName(PassVariant pass_variant) {
  switch (pass_variant) {
    case PassVariant::RunStandardPipelineNoInlineProcs:
      return "RunStandardPipelineNoInlineProcs";
    case PassVariant::RunStandardPipelineInlineProcs:
      return "RunStandardPipelineInlineProcs";
    case PassVariant::RunChannelLegalizationPassOnly:
      return "RunChannelLegalizationPassOnly";
  }
  XLS_LOG(ERROR) << absl::StreamFormat("Unexpected value for PassVariant: %d",
                                       static_cast<int>(pass_variant));
  return "<unknown>";
}

bool PassVariantInlinesProcs(PassVariant pass_variant) {
  return pass_variant == PassVariant::RunStandardPipelineInlineProcs;
}

struct TestParam {
  using evaluation_function = std::function<absl::Status(
      SerialProcRuntime*, std::optional<ChannelStrictness>)>;
  std::string_view test_name;
  std::string_view ir_text;
  absl::flat_hash_map<ChannelStrictness,
                      ::testing::Matcher<absl::StatusOr<bool>>>
      builder_matcher = {};
  evaluation_function evaluate =
      [](SerialProcRuntime* interpreter,
         std::optional<ChannelStrictness> strictness) -> absl::Status {
    GTEST_MESSAGE_("Evaluation is not implemented for this test!",
                   ::testing::TestPartResult::kSkip);
    return absl::OkStatus();
  };
};

class ChannelLegalizationPassTest
    : public testing::TestWithParam<
          std::tuple<TestParam, PassVariant, ChannelStrictness>> {
 protected:
  absl::StatusOr<bool> Run(Package* package) {
    PassVariant pass_variant = std::get<1>(GetParam());
    Pass* pass;
    switch (pass_variant) {
      case PassVariant::RunStandardPipelineNoInlineProcs:
      case PassVariant::RunStandardPipelineInlineProcs: {
        pass = StandardPipelinePass();
        break;
      }
      case PassVariant::RunChannelLegalizationPassOnly: {
        pass = ChannelLegalizationPassOnly();
        break;
      }
    }

    PassOptions options{.inline_procs = PassVariantInlinesProcs(pass_variant)};
    PassResults results;
    return pass->Run(package, options, &results);
  }
};

TestParam kTestParameters[] = {
    TestParam{
        .test_name = "SingleProcBackToBackDataSwitchingOps",
        .ir_text = R"(package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=$0, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=$0, metadata="""""")

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
                // Mutually exclusive OK- channel legalization pass skips them.
                // They are ultimately handled them in scheduling.
                {ChannelStrictness::kProvenMutuallyExclusive, IsOk()},
                {ChannelStrictness::kTotalOrder, IsOkAndHolds(true)},
                {ChannelStrictness::kRuntimeOrdered, IsOkAndHolds(true)},
                // Build should be OK, but will fail at runtime.
                {ChannelStrictness::kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {ChannelStrictness::kArbitraryStaticOrder, IsOkAndHolds(true)},
            },
        .evaluate =
            [](SerialProcRuntime* interpreter,
               std::optional<ChannelStrictness> strictness) -> absl::Status {
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
              strictness.value() ==
                  ChannelStrictness::kRuntimeMutuallyExclusive) {
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
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=$0, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=$0, metadata="""""")

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
                {ChannelStrictness::kProvenMutuallyExclusive, IsOk()},
                {ChannelStrictness::kTotalOrder, IsOkAndHolds(true)},
                {ChannelStrictness::kRuntimeOrdered, IsOkAndHolds(true)},
                {ChannelStrictness::kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {ChannelStrictness::kArbitraryStaticOrder, IsOkAndHolds(true)},
            },
        .evaluate =
            [](SerialProcRuntime* interpreter,
               std::optional<ChannelStrictness> strictness) -> absl::Status {
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
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=$0, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=$0, metadata="""""")

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
                {ChannelStrictness::kProvenMutuallyExclusive, IsOk()},
                {ChannelStrictness::kTotalOrder, IsOkAndHolds(true)},
                {ChannelStrictness::kRuntimeOrdered, IsOkAndHolds(true)},
                {ChannelStrictness::kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {ChannelStrictness::kArbitraryStaticOrder, IsOkAndHolds(true)},
            },
        .evaluate =
            [](SerialProcRuntime* interpreter,
               std::optional<ChannelStrictness> strictness) -> absl::Status {
          constexpr int64_t kMaxTicks = 1000;
          constexpr int64_t kNumInputs = 32;

          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * inq,
              interpreter->queue_manager().GetQueueByName("in"));
          for (int64_t i = 0; i < kNumInputs; ++i) {
            XLS_RET_CHECK_OK(inq->Write(Value(UBits(i, /*bit_count=*/32))));
          }

          XLS_ASSIGN_OR_RETURN(
              ChannelQueue * outq,
              interpreter->queue_manager().GetQueueByName("out"));

          absl::flat_hash_map<Channel*, int64_t> output_count{
              {outq->channel(), kNumInputs}};
          // Adapters assert that only one proc fires on a channel per adapter
          // proc tick. The 'proven mutually exclusive' case doesn't insert an
          // adapter, so exclude that case.
          if (strictness.has_value() &&
              strictness.value() !=
                  ChannelStrictness::kProvenMutuallyExclusive) {
            EXPECT_THAT(
                interpreter->TickUntilOutput(output_count, kMaxTicks).status(),
                StatusIs(absl::StatusCode::kAborted,
                         HasSubstr("Activation for node recv was not mutually "
                                   "exclusive.")));
            return absl::OkStatus();
          }
          EXPECT_THAT(
              interpreter->TickUntilOutput(output_count, kMaxTicks).status(),
              IsOk());

          for (int64_t i = 0; i < kNumInputs; ++i) {
            EXPECT_EQ(outq->Read(), Value(UBits(i, /*bit_count=*/32)));
          }

          return absl::OkStatus();
        },
    },
    TestParam{
        .test_name = "SingleProcWithPartialOrder",
        .ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=$0, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=$0, metadata="""""")
chan pred(bits[2], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=$0, metadata="""""")

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
                {ChannelStrictness::kProvenMutuallyExclusive, IsOk()},
                {ChannelStrictness::kTotalOrder,
                 StatusIs(absl::StatusCode::kInternal,
                          HasSubstr("is not totally ordered"))},
                {ChannelStrictness::kRuntimeOrdered, IsOkAndHolds(true)},
                {ChannelStrictness::kRuntimeMutuallyExclusive,
                 IsOkAndHolds(true)},
                {ChannelStrictness::kArbitraryStaticOrder, IsOkAndHolds(true)},
            },
        .evaluate =
            [](SerialProcRuntime* interpreter,
               std::optional<ChannelStrictness> strictness) -> absl::Status {
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
              strictness.value() ==
                  ChannelStrictness::kRuntimeMutuallyExclusive) {
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
              strictness.value() ==
                  ChannelStrictness::kRuntimeMutuallyExclusive) {
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
              (strictness.value() ==
                   ChannelStrictness::kRuntimeMutuallyExclusive ||
               strictness.value() == ChannelStrictness::kRuntimeOrdered)) {
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
  ChannelStrictness strictness = std::get<2>(GetParam());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(absl::Substitute(
                               std::get<0>(GetParam()).ir_text,
                               ChannelStrictnessToString(strictness))));

  absl::StatusOr<bool> run_status;
  auto matchers = std::get<0>(GetParam()).builder_matcher;
  auto itr = matchers.find(strictness);
  if (itr == matchers.end()) {
    GTEST_SKIP();
  }
  ::testing::Matcher<absl::StatusOr<bool>> matcher = itr->second;
  EXPECT_THAT((run_status = Run(p.get())), matcher);
  // If we expect the pass to complete, the result should be codegen'able.
  bool inline_procs = PassVariantInlinesProcs(std::get<1>(GetParam()));
  if (run_status.ok()) {
    EXPECT_THAT(VerifyPackage(p.get(), /*codegen=*/inline_procs), IsOk());
  }
}

TEST_P(ChannelLegalizationPassTest, EvaluatesCorrectly) {
  ChannelStrictness strictness = std::get<2>(GetParam());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(absl::Substitute(
                               std::get<0>(GetParam()).ir_text,
                               ChannelStrictnessToString(strictness))));
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
        testing::Values(
            // TODO(google/xls#1018): Enable proc inlining variant when cycle
            // problems are solved.
            // PassVariant::RunStandardPipelineInlineProcs,
            PassVariant::RunStandardPipelineNoInlineProcs,
            PassVariant::RunChannelLegalizationPassOnly),
        testing::Values(ChannelStrictness::kProvenMutuallyExclusive,
                        ChannelStrictness::kRuntimeMutuallyExclusive,
                        ChannelStrictness::kTotalOrder,
                        ChannelStrictness::kRuntimeOrdered,
                        ChannelStrictness::kArbitraryStaticOrder)),
    [](const auto& info) {
      return absl::StrCat(std::get<0>(info.param).test_name, "_",
                          PassVariantName(std::get<1>(info.param)), "_",
                          ChannelStrictnessToString(std::get<2>(info.param)));
    });

}  // namespace
}  // namespace xls
