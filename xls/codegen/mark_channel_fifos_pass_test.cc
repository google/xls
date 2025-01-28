// Copyright 2024 The XLS Authors
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

#include "xls/codegen/mark_channel_fifos_pass.h"

#include <initializer_list>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::verilog {
namespace {
using ::absl_testing::IsOkAndHolds;

class MarkChannelFifosPassTest : public IrTestBase {
 public:
  absl::StatusOr<std::pair<FlopKind, FlopKind>> GetFlopResult(
      bool input_flops, CodegenOptions::IOKind input_kind, bool output_flops,
      CodegenOptions::IOKind output_kind) {
    Package pkg("test_pkg");
    ProcBuilder pb("test_proc", &pkg);
    XLS_ASSIGN_OR_RETURN(
        auto* chan,
        pkg.CreateStreamingChannel("chan", ChannelOps::kReceiveOnly,
                                   pkg.GetBitsType(1), {}, ChannelConfig()));
    BValue recv = pb.Receive(chan, pb.Literal(Value::Token()));
    pb.Trace(pb.TupleIndex(recv, 0), pb.Literal(UBits(1, 1)),
             {pb.TupleIndex(recv, 1)}, "foo {}");
    XLS_RETURN_IF_ERROR(pb.Build().status());

    BlockBuilder bb("bb", &pkg, /*should_verify=*/false);
    XLS_ASSIGN_OR_RETURN(Block * b, bb.Build());
    MarkChannelFifosPass mcfp;
    CodegenPassUnit pu(&pkg, b);
    CodegenPassOptions opt;
    opt.codegen_options.flop_inputs(input_flops)
        .flop_outputs(output_flops)
        .flop_inputs_kind(input_kind)
        .flop_outputs_kind(output_kind);
    CodegenPassResults res;

    XLS_RETURN_IF_ERROR(mcfp.Run(&pu, opt, &res).status());
    XLS_RET_CHECK(chan->channel_config().input_flop_kind()) << "No input kind";
    XLS_RET_CHECK(chan->channel_config().output_flop_kind())
        << "no output kind";
    return std::make_pair(*chan->channel_config().input_flop_kind(),
                          *chan->channel_config().output_flop_kind());
  }
};

TEST_F(MarkChannelFifosPassTest, SetsInputFlopOutputFlop) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kFlop, true,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kFlop, FlopKind::kFlop)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputFlopOutputZeroLatency) {
  EXPECT_THAT(
      GetFlopResult(true, CodegenOptions::IOKind::kFlop, true,
                    CodegenOptions::IOKind::kZeroLatencyBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kFlop, FlopKind::kZeroLatency)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputFlopOutputSkid) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kFlop, true,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kFlop, FlopKind::kSkid)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputZeroLatencyOutputFlop) {
  EXPECT_THAT(
      GetFlopResult(true, CodegenOptions::IOKind::kZeroLatencyBuffer, true,
                    CodegenOptions::IOKind::kFlop),
      IsOkAndHolds(testing::Pair(FlopKind::kZeroLatency, FlopKind::kFlop)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputZeroLatencyOutputZeroLatency) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kZeroLatencyBuffer,
                            true, CodegenOptions::IOKind::kZeroLatencyBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kZeroLatency,
                                         FlopKind::kZeroLatency)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputZeroLatencyOutputSkid) {
  EXPECT_THAT(
      GetFlopResult(true, CodegenOptions::IOKind::kZeroLatencyBuffer, true,
                    CodegenOptions::IOKind::kSkidBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kZeroLatency, FlopKind::kSkid)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputSkidOutputFlop) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kSkidBuffer, true,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kSkid, FlopKind::kFlop)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputSkidOutputZeroLatency) {
  EXPECT_THAT(
      GetFlopResult(true, CodegenOptions::IOKind::kSkidBuffer, true,
                    CodegenOptions::IOKind::kZeroLatencyBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kSkid, FlopKind::kZeroLatency)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputSkidOutputSkid) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kSkidBuffer, true,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kSkid, FlopKind::kSkid)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputFlopNoOutputFlop) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kFlop, false,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kFlop, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputFlopNoOutputZeroLatency) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kFlop, false,
                            CodegenOptions::IOKind::kZeroLatencyBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kFlop, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputFlopNoOutputSkid) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kFlop, false,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kFlop, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputZeroLatencyNoOutputFlop) {
  EXPECT_THAT(
      GetFlopResult(true, CodegenOptions::IOKind::kZeroLatencyBuffer, false,
                    CodegenOptions::IOKind::kFlop),
      IsOkAndHolds(testing::Pair(FlopKind::kZeroLatency, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputZeroLatencyNoOutputZeroLatency) {
  EXPECT_THAT(
      GetFlopResult(true, CodegenOptions::IOKind::kZeroLatencyBuffer, false,
                    CodegenOptions::IOKind::kZeroLatencyBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kZeroLatency, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputZeroLatencyNoOutputSkid) {
  EXPECT_THAT(
      GetFlopResult(true, CodegenOptions::IOKind::kZeroLatencyBuffer, false,
                    CodegenOptions::IOKind::kSkidBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kZeroLatency, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputSkidNoOutputFlop) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kSkidBuffer, false,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kSkid, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputSkidNoOutputZeroLatency) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kSkidBuffer, false,
                            CodegenOptions::IOKind::kZeroLatencyBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kSkid, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsInputSkidNoOutputSkid) {
  EXPECT_THAT(GetFlopResult(true, CodegenOptions::IOKind::kSkidBuffer, false,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kSkid, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputFlopOutputFlop) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kFlop, true,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kFlop)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputFlopOutputZeroLatency) {
  EXPECT_THAT(
      GetFlopResult(false, CodegenOptions::IOKind::kFlop, true,
                    CodegenOptions::IOKind::kZeroLatencyBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kZeroLatency)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputFlopOutputSkid) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kFlop, true,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kSkid)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputZeroLatencyOutputFlop) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kZeroLatencyBuffer,
                            true, CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kFlop)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputZeroLatencyOutputZeroLatency) {
  EXPECT_THAT(
      GetFlopResult(false, CodegenOptions::IOKind::kZeroLatencyBuffer, true,
                    CodegenOptions::IOKind::kZeroLatencyBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kZeroLatency)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputZeroLatencyOutputSkid) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kZeroLatencyBuffer,
                            true, CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kSkid)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputSkidOutputFlop) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kSkidBuffer, true,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kFlop)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputSkidOutputZeroLatency) {
  EXPECT_THAT(
      GetFlopResult(false, CodegenOptions::IOKind::kSkidBuffer, true,
                    CodegenOptions::IOKind::kZeroLatencyBuffer),
      IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kZeroLatency)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputSkidOutputSkid) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kSkidBuffer, true,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kSkid)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputFlopNoOutputFlop) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kFlop, false,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputFlopNoOutputZeroLatency) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kFlop, false,
                            CodegenOptions::IOKind::kZeroLatencyBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputFlopNoOutputSkid) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kFlop, false,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputZeroLatencyNoOutputFlop) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kZeroLatencyBuffer,
                            false, CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputZeroLatencyNoOutputZeroLatency) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kZeroLatencyBuffer,
                            false, CodegenOptions::IOKind::kZeroLatencyBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputZeroLatencyNoOutputSkid) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kZeroLatencyBuffer,
                            false, CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputSkidNoOutputFlop) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kSkidBuffer, false,
                            CodegenOptions::IOKind::kFlop),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputSkidNoOutputZeroLatency) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kSkidBuffer, false,
                            CodegenOptions::IOKind::kZeroLatencyBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

TEST_F(MarkChannelFifosPassTest, SetsNoInputSkidNoOutputSkid) {
  EXPECT_THAT(GetFlopResult(false, CodegenOptions::IOKind::kSkidBuffer, false,
                            CodegenOptions::IOKind::kSkidBuffer),
              IsOkAndHolds(testing::Pair(FlopKind::kNone, FlopKind::kNone)));
}

void IgnoresManuallySet(bool input_flops, CodegenOptions::IOKind input_kind,
                        bool output_flops, CodegenOptions::IOKind output_kind) {
  Package pkg("test_pkg");
  ProcBuilder pb("test_proc", &pkg);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* chan, pkg.CreateStreamingChannel(
                      "chan", ChannelOps::kReceiveOnly, pkg.GetBitsType(1), {},
                      ChannelConfig()
                          .WithInputFlopKind(FlopKind::kFlop)
                          .WithOutputFlopKind(FlopKind::kSkid)));
  BValue recv = pb.Receive(chan, pb.Literal(Value::Token()));
  pb.Trace(pb.TupleIndex(recv, 0), pb.Literal(UBits(1, 1)),
           {pb.TupleIndex(recv, 1)}, "foo {}");
  XLS_ASSERT_OK(pb.Build().status());

  BlockBuilder bb("bb", &pkg, /*should_verify=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  MarkChannelFifosPass mcfp;
  CodegenPassUnit pu(&pkg, b);
  CodegenPassOptions opt;
  opt.codegen_options.flop_inputs(input_flops)
      .flop_outputs(output_flops)
      .flop_inputs_kind(input_kind)
      .flop_outputs_kind(output_kind);
  CodegenPassResults res;

  EXPECT_THAT(mcfp.Run(&pu, opt, &res), absl_testing::IsOkAndHolds(false));
}

void IgnoresNonStreaming(bool input_flops, CodegenOptions::IOKind input_kind,
                         bool output_flops,
                         CodegenOptions::IOKind output_kind) {
  Package pkg("test_pkg");
  ProcBuilder pb("test_proc", &pkg);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* chan, pkg.CreateSingleValueChannel("chan", ChannelOps::kReceiveOnly,
                                               pkg.GetBitsType(1), {}));
  BValue recv = pb.Receive(chan, pb.Literal(Value::Token()));
  pb.Trace(pb.TupleIndex(recv, 0), pb.Literal(UBits(1, 1)),
           {pb.TupleIndex(recv, 1)}, "foo {}");
  XLS_ASSERT_OK(pb.Build().status());

  BlockBuilder bb("bb", &pkg, /*should_verify=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  MarkChannelFifosPass mcfp;
  CodegenPassUnit pu(&pkg, b);
  CodegenPassOptions opt;
  opt.codegen_options.flop_inputs(input_flops)
      .flop_outputs(output_flops)
      .flop_inputs_kind(input_kind)
      .flop_outputs_kind(output_kind);
  CodegenPassResults res;

  EXPECT_THAT(mcfp.Run(&pu, opt, &res), absl_testing::IsOkAndHolds(false));
}

constexpr std::initializer_list<CodegenOptions::IOKind> kIoKinds = {
    CodegenOptions::IOKind::kFlop, CodegenOptions::IOKind::kSkidBuffer,
    CodegenOptions::IOKind::kZeroLatencyBuffer};
FUZZ_TEST(MarkChannelFifosPassFuzzTest, IgnoresManuallySet)
    .WithDomains(fuzztest::Arbitrary<bool>(), fuzztest::ElementOf(kIoKinds),
                 fuzztest::Arbitrary<bool>(), fuzztest::ElementOf(kIoKinds));

FUZZ_TEST(MarkChannelFifosPassFuzzTest, IgnoresNonStreaming)
    .WithDomains(fuzztest::Arbitrary<bool>(), fuzztest::ElementOf(kIoKinds),
                 fuzztest::Arbitrary<bool>(), fuzztest::ElementOf(kIoKinds));

}  // namespace
}  // namespace xls::verilog
