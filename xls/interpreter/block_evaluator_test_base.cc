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

#include "xls/interpreter/block_evaluator_test_base.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

TEST_P(BlockEvaluatorTest, EmptyBlock) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  EXPECT_THAT(evaluator().EvaluateCombinationalBlock(
                  block, absl::flat_hash_map<std::string, Value>()),
              IsOkAndHolds(UnorderedElementsAre()));
}

TEST_P(BlockEvaluatorTest, OutputOnlyBlock) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  b.OutputPort("out", b.Literal(Value(UBits(123, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  EXPECT_THAT(
      evaluator().EvaluateCombinationalBlock(
          block, absl::flat_hash_map<std::string, Value>()),
      IsOkAndHolds(UnorderedElementsAre(Pair("out", Value(UBits(123, 32))))));

  EXPECT_THAT(evaluator().EvaluateCombinationalBlock(
                  block, absl::flat_hash_map<std::string, uint64_t>()),
              IsOkAndHolds(UnorderedElementsAre(Pair("out", 123))));
}

TEST_P(BlockEvaluatorTest, SumAndDifferenceBlock) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue y = b.InputPort("y", package->GetBitsType(32));
  b.OutputPort("sum", b.Add(x, y));
  b.OutputPort("diff", b.Subtract(x, y));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  EXPECT_THAT(
      evaluator().EvaluateCombinationalBlock(
          block, {{"x", Value(UBits(42, 32))}, {"y", Value(UBits(10, 32))}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("sum", Value(UBits(52, 32))),
                                        Pair("diff", Value(UBits(32, 32))))));

  EXPECT_THAT(
      evaluator().EvaluateCombinationalBlock(block, {{"x", 42}, {"y", 10}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("sum", 52), Pair("diff", 32))));
}

TEST_P(BlockEvaluatorTest, InputErrors) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  b.InputPort("x", package->GetBitsType(32));
  b.InputPort("y", package->GetTupleType({}));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  EXPECT_THAT(evaluator().EvaluateCombinationalBlock(
                  block, {{"a", Value(UBits(42, 32))},
                          {"x", Value(UBits(42, 32))},
                          {"y", Value::Tuple({})}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Block has no input port 'a'")));

  EXPECT_THAT(evaluator().EvaluateCombinationalBlock(
                  block, {{"x", Value(UBits(10, 32))}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing input for port 'y'")));

  EXPECT_THAT(
      evaluator().EvaluateCombinationalBlock(block, {{"x", 3}, {"y", 42}}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Block has non-Bits-typed input port 'y' of type: ()")));
}

TEST_P(BlockEvaluatorTest, RunWithUInt64Errors) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  b.InputPort("in", package->GetBitsType(8));
  b.OutputPort("out", b.Literal(Value(Bits::AllOnes(100))));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  EXPECT_THAT(evaluator().EvaluateCombinationalBlock(block, {{"in", 500}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input value 500 for input port 'in' does not "
                                 "fit in type: bits[8]")));

  EXPECT_THAT(
      evaluator().EvaluateCombinationalBlock(block, {{"in", 10}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Output value 'out' does not fit in a uint64_t: "
                         "bits[100]:0xf_ffff_ffff_ffff_ffff_ffff_ffff")));
}

TEST_P(BlockEvaluatorTest, PipelinedAdder) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue y = b.InputPort("y", package->GetBitsType(32));

  BValue x_d = b.InsertRegister("x_d", x);
  BValue y_d = b.InsertRegister("y_d", y);

  BValue x_plus_y = b.Add(x_d, y_d);

  BValue x_plus_y_d = b.InsertRegister("x_plus_y_d", x_plus_y);

  b.OutputPort("out", x_plus_y_d);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"x", 1}, {"y", 2}},
      {{"x", 42}, {"y", 100}},
      {{"x", 0}, {"y", 0}},
      {{"x", 0}, {"y", 0}},
      {{"x", 0}, {"y", 0}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           evaluator().EvaluateSequentialBlock(block, inputs));

  ASSERT_EQ(outputs.size(), 5);
  EXPECT_THAT(outputs.at(0), UnorderedElementsAre(Pair("out", 0)));
  EXPECT_THAT(outputs.at(1), UnorderedElementsAre(Pair("out", 0)));
  EXPECT_THAT(outputs.at(2), UnorderedElementsAre(Pair("out", 3)));
  EXPECT_THAT(outputs.at(3), UnorderedElementsAre(Pair("out", 142)));
  EXPECT_THAT(outputs.at(4), UnorderedElementsAre(Pair("out", 0)));
}

TEST_P(BlockEvaluatorTest, RegisterWithReset) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue rst = b.InputPort("rst", package->GetBitsType(1));

  BValue x_d =
      b.InsertRegister("x_d", x, rst,
                       Reset{Value(UBits(42, 32)), /*asynchronous=*/false,
                             /*active_low=*/false});

  b.OutputPort("out", x_d);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"rst", 0}, {"x", 1}},
      {{"rst", 1}, {"x", 2}},
      {{"rst", 1}, {"x", 3}},
      {{"rst", 0}, {"x", 4}},
      {{"rst", 0}, {"x", 5}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           evaluator().EvaluateSequentialBlock(block, inputs));

  ASSERT_EQ(outputs.size(), 5);
  EXPECT_THAT(outputs.at(0), UnorderedElementsAre(Pair("out", 0)));
  EXPECT_THAT(outputs.at(1), UnorderedElementsAre(Pair("out", 1)));
  EXPECT_THAT(outputs.at(2), UnorderedElementsAre(Pair("out", 42)));
  EXPECT_THAT(outputs.at(3), UnorderedElementsAre(Pair("out", 42)));
  EXPECT_THAT(outputs.at(4), UnorderedElementsAre(Pair("out", 4)));
}

TEST_P(BlockEvaluatorTest, RegisterWithLoadEnable) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue le = b.InputPort("le", package->GetBitsType(1));

  BValue x_d = b.InsertRegister("x_d", x, le);

  b.OutputPort("out", x_d);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"le", 0}, {"x", 1}},
      {{"le", 1}, {"x", 2}},
      {{"le", 1}, {"x", 3}},
      {{"le", 0}, {"x", 4}},
      {{"le", 0}, {"x", 5}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           evaluator().EvaluateSequentialBlock(block, inputs));

  ASSERT_EQ(outputs.size(), 5);
  EXPECT_THAT(outputs.at(0), UnorderedElementsAre(Pair("out", 0)));
  EXPECT_THAT(outputs.at(1), UnorderedElementsAre(Pair("out", 0)));
  EXPECT_THAT(outputs.at(2), UnorderedElementsAre(Pair("out", 2)));
  EXPECT_THAT(outputs.at(3), UnorderedElementsAre(Pair("out", 3)));
  EXPECT_THAT(outputs.at(4), UnorderedElementsAre(Pair("out", 3)));
}

TEST_P(BlockEvaluatorTest, RegisterWithResetAndLoadEnable) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue rst_n = b.InputPort("rst_n", package->GetBitsType(1));
  BValue le = b.InputPort("le", package->GetBitsType(1));

  BValue x_d =
      b.InsertRegister("x_d", x, rst_n,
                       Reset{Value(UBits(42, 32)), /*asynchronous=*/false,
                             /*active_low=*/true},
                       le);

  b.OutputPort("out", x_d);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"rst_n", 1}, {"le", 0}, {"x", 1}},
      {{"rst_n", 0}, {"le", 0}, {"x", 2}},
      {{"rst_n", 0}, {"le", 1}, {"x", 3}},
      {{"rst_n", 1}, {"le", 1}, {"x", 4}},
      {{"rst_n", 1}, {"le", 0}, {"x", 5}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           evaluator().EvaluateSequentialBlock(block, inputs));

  ASSERT_EQ(outputs.size(), 5);
  EXPECT_THAT(outputs.at(0), UnorderedElementsAre(Pair("out", 0)));
  EXPECT_THAT(outputs.at(1), UnorderedElementsAre(Pair("out", 0)));
  EXPECT_THAT(outputs.at(2), UnorderedElementsAre(Pair("out", 42)));
  EXPECT_THAT(outputs.at(3), UnorderedElementsAre(Pair("out", 42)));
  EXPECT_THAT(outputs.at(4), UnorderedElementsAre(Pair("out", 4)));
}

TEST_P(BlockEvaluatorTest, AccumulatorRegister) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg,
      b.block()->AddRegister("accum", package->GetBitsType(32)));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue accum = b.RegisterRead(reg);
  BValue next_accum = b.Add(x, accum);
  b.RegisterWrite(reg, next_accum);
  b.OutputPort("out", next_accum);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"x", 1}}, {{"x", 2}}, {{"x", 3}}, {{"x", 4}}, {{"x", 5}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           evaluator().EvaluateSequentialBlock(block, inputs));

  ASSERT_EQ(outputs.size(), 5);
  EXPECT_THAT(outputs.at(0), UnorderedElementsAre(Pair("out", 1)));
  EXPECT_THAT(outputs.at(1), UnorderedElementsAre(Pair("out", 3)));
  EXPECT_THAT(outputs.at(2), UnorderedElementsAre(Pair("out", 6)));
  EXPECT_THAT(outputs.at(3), UnorderedElementsAre(Pair("out", 10)));
  EXPECT_THAT(outputs.at(4), UnorderedElementsAre(Pair("out", 15)));
}

TEST_P(BlockEvaluatorTest, ChannelizedAccumulatorRegister) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg,
      b.block()->AddRegister("accum", package->GetBitsType(32)));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue x_vld = b.InputPort("x_vld", package->GetBitsType(1));
  BValue out_rdy = b.InputPort("out_rdy", package->GetBitsType(1));

  BValue input_valid_and_output_ready = b.And(x_vld, out_rdy);
  BValue accum = b.RegisterRead(reg);
  BValue x_add_accum = b.Add(x, accum);
  BValue next_accum =
      b.Select(input_valid_and_output_ready, {accum, x_add_accum});

  b.RegisterWrite(reg, next_accum);
  b.OutputPort("x_rdy", out_rdy);
  b.OutputPort("out", next_accum);
  b.OutputPort("out_vld", x_vld);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  // Check that we can simulate the block without any input sequence.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 0.5, block)};
    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 0.1, block),
    };

    std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
    inputs.resize(100);

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io,
        evaluator().EvaluateChannelizedSequentialBlockWithUint64(
            block, absl::MakeSpan(sources), absl::MakeSpan(sinks), inputs));
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_EQ(output_sequence.size(), 0);
  }

  // Provide an input sequence and simulate.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 0.5, block)};
    XLS_ASSERT_OK(
        sources.at(0).SetDataSequence(std::vector<uint64_t>{1, 2, 3, 4, 5}));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 0.1, block),
    };

    std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
    inputs.resize(100);

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io,
        evaluator().EvaluateChannelizedSequentialBlockWithUint64(
            block, absl::MakeSpan(sources), absl::MakeSpan(sinks), inputs));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_GT(block_io.outputs.size(), output_sequence.size());
    EXPECT_THAT(output_sequence, ElementsAre(1, 3, 6, 10, 15));
  }
}

TEST_P(BlockEvaluatorTest, ChannelizedResetHandling) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  verilog::ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(false);
  reset.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg,
      b.block()->AddRegister("accum", package->GetBitsType(32),
                             Reset{
                                 .reset_value = Value(UBits(0, 32)),
                                 .asynchronous = reset.asynchronous(),
                                 .active_low = reset.active_low(),
                             }));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue x_vld = b.InputPort("x_vld", package->GetBitsType(1));
  BValue out_rdy = b.InputPort("out_rdy", package->GetBitsType(1));
  BValue rst = b.InputPort("rst", package->GetBitsType(1));

  BValue input_valid_and_output_ready = b.And(x_vld, out_rdy);
  BValue accum = b.RegisterRead(reg);
  BValue x_add_accum = b.Add(x, accum);
  BValue next_accum =
      b.Select(input_valid_and_output_ready, {accum, x_add_accum});

  b.RegisterWrite(reg, next_accum, /*load_enable=*/std::nullopt, /*reset=*/rst);
  b.OutputPort("x_rdy", out_rdy);
  b.OutputPort("out", next_accum);
  b.OutputPort("out_vld", x_vld);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs(15,
                                                                 {{"rst", 0}});
  for (size_t cycle = 0; cycle < 5; ++cycle) {
    inputs[cycle]["rst"] = 1;
  }

  // Provide an input sequence and simulate, sending input & receiving output
  // during reset.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 1.0, block,
                      /*reset_behavior=*/ChannelSource::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 1.0, block,
                    /*reset_behavior=*/ChannelSink::kAttendValid),
    };

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io, evaluator().EvaluateChannelizedSequentialBlockWithUint64(
                      block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
                      inputs, reset));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_GT(block_io.outputs.size(), output_sequence.size());
    absl::Span<const uint64_t> outputs = absl::MakeConstSpan(output_sequence);

    // During reset, the block is a pass-through for input to output.
    EXPECT_THAT(outputs.first(5), ElementsAre(1, 2, 3, 4, 5));

    // After reset, the block accumulates as designed.
    EXPECT_THAT(outputs.subspan(5), ElementsAre(6, 13, 21, 30, 40));
  }

  // Provide an input sequence and simulate, sending input but receiving no
  // output during reset.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 1.0, block,
                      /*reset_behavior=*/ChannelSource::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 1.0, block,
                    /*reset_behavior=*/ChannelSink::kIgnoreValid),
    };

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io, evaluator().EvaluateChannelizedSequentialBlockWithUint64(
                      block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
                      inputs, reset));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_GT(block_io.outputs.size(), output_sequence.size());

    // We only see the block's results after reset, when it accumulates as
    // designed.
    EXPECT_THAT(output_sequence, ElementsAre(6, 13, 21, 30, 40));
  }

  // Provide an input sequence and simulate, sending no input & ignoring output
  // during reset.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 1.0, block,
                      /*reset_behavior=*/ChannelSource::kIgnoreReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 1.0, block,
                    /*reset_behavior=*/ChannelSink::kIgnoreValid),
    };

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io, evaluator().EvaluateChannelizedSequentialBlockWithUint64(
                      block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
                      inputs, reset));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_GT(block_io.outputs.size(), output_sequence.size());

    // We send & receive no data during reset, so we only see the block
    // accumulate.
    EXPECT_THAT(output_sequence,
                ElementsAre(1, 3, 6, 10, 15, 21, 28, 36, 45, 55));
  }
}

TEST_P(BlockEvaluatorTest, ChannelizedResetHandlingActiveLow) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  verilog::ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(false);
  reset.set_active_low(true);

  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg,
      b.block()->AddRegister("accum", package->GetBitsType(32),
                             Reset{
                                 .reset_value = Value(UBits(0, 32)),
                                 .asynchronous = reset.asynchronous(),
                                 .active_low = reset.active_low(),
                             }));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue x_vld = b.InputPort("x_vld", package->GetBitsType(1));
  BValue out_rdy = b.InputPort("out_rdy", package->GetBitsType(1));
  BValue rst = b.InputPort("rst", package->GetBitsType(1));

  BValue input_valid_and_output_ready = b.And(x_vld, out_rdy);
  BValue accum = b.RegisterRead(reg);
  BValue x_add_accum = b.Add(x, accum);
  BValue next_accum =
      b.Select(input_valid_and_output_ready, {accum, x_add_accum});

  b.RegisterWrite(reg, next_accum, /*load_enable=*/std::nullopt, /*reset=*/rst);
  b.OutputPort("x_rdy", out_rdy);
  b.OutputPort("out", next_accum);
  b.OutputPort("out_vld", x_vld);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs(15,
                                                                 {{"rst", 1}});
  for (size_t cycle = 0; cycle < 5; ++cycle) {
    inputs[cycle]["rst"] = 0;
  }

  // Provide an input sequence and simulate, sending input & receiving output
  // during reset.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 1.0, block,
                      /*reset_behavior=*/ChannelSource::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 1.0, block,
                    /*reset_behavior=*/ChannelSink::kAttendValid),
    };

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io, evaluator().EvaluateChannelizedSequentialBlockWithUint64(
                      block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
                      inputs, reset));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_GT(block_io.outputs.size(), output_sequence.size());
    absl::Span<const uint64_t> outputs = absl::MakeConstSpan(output_sequence);

    // During reset, the block is a pass-through for input to output.
    EXPECT_THAT(outputs.first(5), ElementsAre(1, 2, 3, 4, 5));

    // After reset, the block accumulates as designed.
    EXPECT_THAT(outputs.subspan(5), ElementsAre(6, 13, 21, 30, 40));
  }

  // Provide an input sequence and simulate, sending input but receiving no
  // output during reset.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 1.0, block,
                      /*reset_behavior=*/ChannelSource::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 1.0, block,
                    /*reset_behavior=*/ChannelSink::kIgnoreValid),
    };

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io, evaluator().EvaluateChannelizedSequentialBlockWithUint64(
                      block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
                      inputs, reset));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_GT(block_io.outputs.size(), output_sequence.size());

    // We only see the block's results after reset, when it accumulates as
    // designed.
    EXPECT_THAT(output_sequence, ElementsAre(6, 13, 21, 30, 40));
  }

  // Provide an input sequence and simulate, sending no input & ignoring output
  // during reset.
  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 1.0, block,
                      /*reset_behavior=*/ChannelSource::kIgnoreReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 1.0, block,
                    /*reset_behavior=*/ChannelSink::kIgnoreValid),
    };

    BlockIOResultsAsUint64 block_io;
    XLS_ASSERT_OK_AND_ASSIGN(
        block_io, evaluator().EvaluateChannelizedSequentialBlockWithUint64(
                      block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
                      inputs, reset));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             sinks.at(0).GetOutputSequenceAsUint64());
    EXPECT_GT(block_io.outputs.size(), output_sequence.size());

    // We send & receive no data during reset, so we only see the block
    // accumulate.
    EXPECT_THAT(output_sequence,
                ElementsAre(1, 3, 6, 10, 15, 21, 28, 36, 45, 55));
  }
}

TEST_P(BlockEvaluatorTest, InterpreterEventsCaptured) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue tkn = b.Literal(Value::Token());
  BValue assertion =
      b.Assert(tkn, b.UGt(x, b.Literal(Value(UBits(5, 32)))), "foo");
  b.Trace(assertion, b.Literal(Value(UBits(1, 1))), {x},
          {"x is ", FormatPreference::kDefault});

  b.OutputPort("y", x);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockRunResult result,
      evaluator().EvaluateBlock({{"x", Value(UBits(10, 32))}}, {}, block));

  EXPECT_THAT(result.interpreter_events.trace_msgs, ElementsAre("x is 10"));
  EXPECT_THAT(result.interpreter_events.assert_msgs, IsEmpty());

  XLS_ASSERT_OK_AND_ASSIGN(
      result,
      evaluator().EvaluateBlock({{"x", Value(UBits(3, 32))}}, {}, block));

  EXPECT_THAT(result.interpreter_events.trace_msgs, ElementsAre("x is 3"));
  EXPECT_THAT(result.interpreter_events.assert_msgs, ElementsAre("foo"));
}

}  // namespace
}  // namespace xls
