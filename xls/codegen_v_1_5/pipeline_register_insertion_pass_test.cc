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

#include "xls/codegen_v_1_5/pipeline_register_insertion_pass.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_finalization_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/register.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"

namespace m = xls::op_matchers;

namespace m2 {
MATCHER_P2(Register, name, type,
           absl::StrCat("Register with name ",
                        testing::DescribeMatcher<std::string>(name, negation),
                        " and type ",
                        testing::DescribeMatcher<xls::Type*>(type, negation))) {
  if (!testing::ExplainMatchResult(name, arg->name(), result_listener)) {
    *result_listener << "where register name is " << arg->name();
    return false;
  }
  if (!testing::ExplainMatchResult(type, arg->type(), result_listener)) {
    *result_listener << "where register type is " << arg->type()->ToString();
    return false;
  }
  return true;
}
}  // namespace m2

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class PipelineRegisterInsertionPassTest : public IrTestBase {
 protected:
  PipelineRegisterInsertionPassTest() = default;

  absl::StatusOr<bool> Run(Package* p,
                           const BlockConversionPassOptions& options =
                               BlockConversionPassOptions()) {
    PassResults results;
    return PipelineRegisterInsertionPass().Run(p, options, &results);
  }
};

TEST_F(PipelineRegisterInsertionPassTest, TestNoCrossStageUses) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  BValue x = sbb.InputPort("x", p->GetBitsType(32));
  BValue y = sbb.InputPort("y", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue x_plus_1 = sbb.Add(x, sbb.Literal(UBits(1, 32)));
  sbb.OutputPort("x_plus_1", x_plus_1);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue y_plus_1 = sbb.Add(y, sbb.Literal(UBits(1, 32)));
  sbb.OutputPort("y_plus_1", y_plus_1);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(sb->GetRegisters(), IsEmpty());
}

TEST_F(PipelineRegisterInsertionPassTest, TestSingleStageForwarding) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));
  BValue y = sbb.InputPort("y", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue v0 = sbb.Add(x, y, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              ElementsAre(m2::Register("p0_v0", m::Type("bits[32]"))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestMultiStageForwarding) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));
  BValue y = sbb.InputPort("y", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue v0 = sbb.Add(x, y, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              ElementsAre(m2::Register("p0_v0", m::Type("bits[32]")),
                          m2::Register("p1_v0", m::Type("bits[32]"))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestFanoutToDifferentStages) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));
  BValue y = sbb.InputPort("y", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue v0 = sbb.Add(x, y, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0_s1 = sbb.Negate(v0);
  sbb.OutputPort("out1", neg_v0_s1);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0_s2 = sbb.Negate(v0);
  sbb.OutputPort("out2", neg_v0_s2);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              ElementsAre(m2::Register("p0_v0", m::Type("bits[32]")),
                          m2::Register("p1_v0", m::Type("bits[32]"))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestTupleSplitting) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));
  BValue y = sbb.InputPort("y", p->GetBitsType(8));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue v0 = sbb.Tuple({x, sbb.Literal(UBits(0, 0)), y}, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue index0 = sbb.TupleIndex(v0, 0);
  BValue index2 = sbb.TupleIndex(v0, 2);
  BValue add = sbb.Add(index0, sbb.ZeroExtend(index2, 32));
  sbb.OutputPort("out", add);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              ElementsAre(m2::Register("p0_v0_index0", m::Type("bits[32]")),
                          m2::Register("p0_v0_index1", m::Type("bits[0]")),
                          m2::Register("p0_v0_index2", m::Type("bits[8]"))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestPipelineSimulation) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue mul4 = sbb.UMul(in, sbb.Literal(UBits(4, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue add1 = sbb.Add(mul4, sbb.Literal(UBits(1, 32)));
  sbb.OutputPort("out", add1);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK(sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  PassResults results;
  XLS_ASSERT_OK(BlockFinalizationPass().Run(
      p.get(), BlockConversionPassOptions(), &results));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(
      block_results,
      InterpretSequentialBlock(block, {{{"in", Value(UBits(10, 32))}},
                                       {{"in", Value(UBits(20, 32))}},
                                       {{"in", Value(UBits(0, 32))}}}));
  EXPECT_THAT(
      block_results,
      ElementsAre(UnorderedElementsAre(Pair("out", Value(UBits(1, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(41, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(81, 32))))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestResetSimulation) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK(sbb.block()->AddResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false}));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue in_plus_1 = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.OutputPort("out", in_plus_1);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK(sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.reset("rst", /*asynchronous=*/false,
                                /*active_low=*/false,
                                /*reset_data_path=*/true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));

  PassResults results;
  XLS_ASSERT_OK(BlockFinalizationPass().Run(p.get(), options, &results));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(
      block_results,
      InterpretSequentialBlock(
          block, {{{"in", Value(UBits(100, 32))}, {"rst", Value(UBits(1, 1))}},
                  {{"in", Value(UBits(5, 32))}, {"rst", Value(UBits(0, 1))}},
                  {{"in", Value(UBits(6, 32))}, {"rst", Value(UBits(0, 1))}},
                  {{"in", Value(UBits(0, 32))}, {"rst", Value(UBits(0, 1))}}}));
  EXPECT_THAT(
      block_results,
      ElementsAre(UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(6, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(7, 32))))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestStalledPipeline) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  BValue s0r_in = sbb.InputPort("s0r_in", p->GetBitsType(1));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), s0r_in);
  BValue s0v_in = sbb.InputPort("s0v_in", p->GetBitsType(1));
  BValue v = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), s0v_in);

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.OutputPort("out", v);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK(sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  PassResults results;
  XLS_ASSERT_OK(BlockFinalizationPass().Run(
      p.get(), BlockConversionPassOptions(), &results));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(
      block_results,
      InterpretSequentialBlock(block, {{{"in", Value(UBits(10, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(20, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(0, 1))}},
                                       {{"in", Value(UBits(30, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(0, 1))}},
                                       {{"in", Value(UBits(40, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(50, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(60, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}}}));
  EXPECT_THAT(
      block_results,
      ElementsAre(UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(11, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(11, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(11, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(41, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(51, 32))))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestPipelineBubble) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  BValue s0r_in = sbb.InputPort("s0r_in", p->GetBitsType(1));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), s0r_in);
  BValue s0v_in = sbb.InputPort("s0v_in", p->GetBitsType(1));
  BValue v = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), s0v_in);

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.OutputPort("out", v);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK(sbb.Build());

  ScopedRecordIr sri(p.get());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  PassResults results;
  XLS_ASSERT_OK(BlockFinalizationPass().Run(
      p.get(), BlockConversionPassOptions(), &results));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(
      block_results,
      InterpretSequentialBlock(block, {{{"in", Value(UBits(10, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(20, 32))},
                                        {"s0v_in", Value(UBits(0, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(30, 32))},
                                        {"s0v_in", Value(UBits(0, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(40, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(50, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}},
                                       {{"in", Value(UBits(60, 32))},
                                        {"s0v_in", Value(UBits(1, 1))},
                                        {"s0r_in", Value(UBits(1, 1))}}}));
  EXPECT_THAT(
      block_results,
      ElementsAre(UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(11, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(11, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(11, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(41, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(51, 32))))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestCombinedRegisters) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  sbb.AddStateReadToCurrentStage(acc);
  BValue v0 = sbb.Add(x, acc, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.Next(acc, neg_v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              ElementsAre(m2::Register("p0_v0", m::Type("bits[32]"))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestRegistersDontCombine) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  sbb.AddStateReadToCurrentStage(acc);
  BValue v0 = sbb.Add(x, acc, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.Next(acc, neg_v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kDontMerge);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              ElementsAre(m2::Register("p0_v0", m::Type("bits[32]")),
                          m2::Register("p1_v0", m::Type("bits[32]"))));
}

TEST_F(PipelineRegisterInsertionPassTest, TestCombinedRegistersWithState) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  // Stage 0 - starts mutex region
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  sbb.AddStateReadToCurrentStage(acc);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue v0 = sbb.Add(x, acc, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 3
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 4 - ends mutex region
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.Next(acc, neg_v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));

  // Since the value of `acc` can't change until stage 4, stage 2 can directly
  // read from `acc` rather than needing any pipeline registers. The result (v0)
  // can be stored in a pipeline register & then used directly in stage 4, since
  // stages 2-4 are mutually exclusive.
  EXPECT_THAT(sb->GetRegisters(),
              UnorderedElementsAre(m2::Register("p2_v0", m::Type("bits[32]"))));
}

TEST_F(PipelineRegisterInsertionPassTest,
       TestCombinedRegistersWithPredicatedWrites) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  // Stage 0 - starts mutex region
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  sbb.AddStateReadToCurrentStage(acc);
  BValue v0 = sbb.Add(x, acc, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 2 - ends mutex region
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Next(acc, v0, /*pred=*/sbb.Eq(v0, sbb.Literal(UBits(0, 32))));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 3
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 4
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.Next(acc, neg_v0, /*pred=*/sbb.Ne(v0, sbb.Literal(UBits(0, 32))));
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              ElementsAre(m2::Register("p0_v0", m::Type("bits[32]")),
                          m2::Register("p2_v0", m::Type("bits[32]")),
                          m2::Register("p3_v0", m::Type("bits[32]"))))
      << "actual registers: "
      << absl::StrJoin(sb->GetRegisters(), ", ",
                       [](std::string* out, const Register* reg) {
                         absl::StrAppend(out, reg->ToString());
                       });
}

TEST_F(PipelineRegisterInsertionPassTest,
       TestCombinedRegistersWithCloselyOverlappingMutexRegions) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  // Stage 0 - starts mutex region 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  sbb.AddStateReadToCurrentStage(acc);
  BValue v0 = sbb.Add(x, acc, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 1 - starts mutex region 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_z, source->AppendStateElement("z", Value(UBits(0, 32))));
  BValue z = sbb.SourceNode(source_z);
  sbb.AddStateReadToCurrentStage(z);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 2 - ends mutex region 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Next(acc, v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 3
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 4
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 5 - ends mutex region 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Next(z, sbb.Literal(UBits(1, 32)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              UnorderedElementsAre(m2::Register("p0_v0", m::Type("bits[32]")),
                                   m2::Register("p2_v0", m::Type("bits[32]"))))
      << "actual registers: "
      << absl::StrJoin(sb->GetRegisters(), ", ",
                       [](std::string* out, const Register* reg) {
                         absl::StrAppend(out, reg->ToString());
                       });
}

TEST_F(PipelineRegisterInsertionPassTest,
       TestCombinedRegistersWithDistantOverlappingMutexRegions) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  // Stage 0 - starts mutex region 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  sbb.AddStateReadToCurrentStage(acc);
  BValue v0 = sbb.Add(x, acc, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 2 - ends mutex region 1, starts mutex region 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_z, source->AppendStateElement("z", Value(UBits(0, 32))));
  BValue z = sbb.SourceNode(source_z);
  sbb.AddStateReadToCurrentStage(z);
  sbb.Next(acc, v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 3
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 4
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 5 - ends mutex region 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Next(z, sbb.Literal(UBits(1, 32)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              UnorderedElementsAre(m2::Register("p0_v0", m::Type("bits[32]")),
                                   m2::Register("p2_v0", m::Type("bits[32]"))))
      << "actual registers: "
      << absl::StrJoin(sb->GetRegisters(), ", ",
                       [](std::string* out, const Register* reg) {
                         absl::StrAppend(out, reg->ToString());
                       });
}

TEST_F(PipelineRegisterInsertionPassTest,
       TestCombinedRegistersWithDisjointMutexRegions) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  // Stage 0 - starts mutex region 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  sbb.AddStateReadToCurrentStage(acc);
  BValue v0 = sbb.Add(x, acc, SourceInfo(), "v0");
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 1
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 2 - ends mutex region 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Next(acc, v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 3 - starts mutex region 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_z, source->AppendStateElement("z", Value(UBits(0, 32))));
  BValue z = sbb.SourceNode(source_z);
  sbb.AddStateReadToCurrentStage(z);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 4
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 5 - ends mutex region 2
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Next(z, sbb.Literal(UBits(1, 32)));
  BValue neg_v0 = sbb.Negate(v0);
  sbb.OutputPort("out", neg_v0);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  EXPECT_THAT(sb->GetRegisters(),
              UnorderedElementsAre(m2::Register("p0_v0", m::Type("bits[32]")),
                                   m2::Register("p2_v0", m::Type("bits[32]")),
                                   m2::Register("p3_v0", m::Type("bits[32]"))))
      << "actual registers: "
      << absl::StrJoin(sb->GetRegisters(), ", ",
                       [](std::string* out, const Register* reg) {
                         absl::StrAppend(out, reg->ToString());
                       });
}

TEST_F(PipelineRegisterInsertionPassTest,
       TestRAWStateHazardWithMutualExclusion) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  Proc* source;
  {
    std::unique_ptr<Proc> owned_source = std::make_unique<Proc>(
        absl::StrCat("__", TestName(), "_source"), p.get());
    source = owned_source.get();
    sbb.SetSource(std::move(owned_source));
  }
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue x = sbb.InputPort("x", p->GetBitsType(32));

  // Stage 0 - read from both `acc` and `z`, update `acc`
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_acc, source->AppendStateElement(
                                                  "acc", Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_z, source->AppendStateElement("z", Value(UBits(0, 32))));
  BValue acc = sbb.SourceNode(source_acc);
  BValue z = sbb.SourceNode(source_z);
  sbb.AddStateReadToCurrentStage(acc);
  sbb.AddStateReadToCurrentStage(z);
  BValue next_acc = sbb.Add(x, acc, SourceInfo(), "next_acc");
  sbb.Next(acc, next_acc);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  // Stage 1 - update `z`, ending a mutex region on stages [0, 1]
  // Also uses `acc`, requiring a pipeline register for `acc` if we behave
  // correctly, as the write in stage 0 would otherwise clobber its value.
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  sbb.Next(z, sbb.Add(z, sbb.Literal(UBits(1, 32))));
  BValue out = sbb.Add(acc, z, SourceInfo(), "out");
  sbb.OutputPort("result", out);

  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));

  // Verify the correct set of pipeline registers. `z` doesn't need one, even
  // though the read is in stage 0, because we can extend its lifetime to stage
  // 1 due to no earlier writes. `acc` *does* need a pipeline register; its
  // write in stage 0 would otherwise clobber the value needed for the addition
  // in stage 1.
  EXPECT_THAT(sb->GetRegisters(),
              testing::Contains(m2::Register("p0_acc", m::Type("bits[32]"))));
}

}  // namespace
}  // namespace xls::codegen
