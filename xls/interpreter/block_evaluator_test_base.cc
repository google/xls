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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {
namespace {

using OutputPortSampleTime = BlockEvaluator::OutputPortSampleTime;

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::A;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

TEST_P(BlockEvaluatorTest, ObserverSeesValues) {
  if (!SupportsObserver()) {
    GTEST_SKIP() << "Observers unsupported";
  }
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  BValue foo_inp = bb.InputPort("foo", p->GetBitsType(32));
  BValue enable = bb.InputPort("enable", p->GetBitsType(1));
  BValue rhs_inp = bb.InputPort("rhs", p->GetBitsType(32));
  BValue delay = bb.InsertRegister("delay", foo_inp, enable);
  BValue add_res = bb.Add(delay, rhs_inp);
  BValue out = bb.OutputPort("res", add_res);
  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  CollectingEvaluationObserver observer;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cont,
      evaluator().NewContinuation(
          b, BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock));
  XLS_ASSERT_OK(cont->SetObserver(&observer));
  // nb evaluator forces ports to zero at the start.
  XLS_ASSERT_OK(cont->RunOneCycle({{"foo", Value(UBits(1, 32))},
                                   {"enable", Value(UBits(0, 1))},
                                   {"rhs", Value(UBits(2, 32))}}));
  XLS_ASSERT_OK(cont->RunOneCycle({{"foo", Value(UBits(2, 32))},
                                   {"enable", Value(UBits(1, 1))},
                                   {"rhs", Value(UBits(3, 32))}}));
  XLS_ASSERT_OK(cont->RunOneCycle({{"foo", Value(UBits(0, 32))},
                                   {"enable", Value(UBits(0, 1))},
                                   {"rhs", Value(UBits(4, 32))}}));
  // NB The observer sees both the rising edge start value and the post falling
  // edge value so there are 6 entries despite only being 3 cycles.
  EXPECT_THAT(
      observer.values(),
      UnorderedElementsAre(
          Pair(foo_inp.node(),
               ElementsAre(Value(UBits(1, 32)), Value(UBits(2, 32)),
                           Value(UBits(0, 32)))),
          Pair(enable.node(),
               ElementsAre(Value(UBits(0, 1)), Value(UBits(1, 1)),
                           Value(UBits(0, 1)))),
          Pair(rhs_inp.node(),
               ElementsAre(Value(UBits(2, 32)), Value(UBits(3, 32)),
                           Value(UBits(4, 32)))),
          Pair(add_res.node(),
               ElementsAre(Value(UBits(2, 32)), Value(UBits(3, 32)),
                           Value(UBits(6, 32)))),
          Pair(out.node(), ElementsAre(Value::Tuple({}), Value::Tuple({}),
                                       Value::Tuple({}))),
          Pair(A<Node*>(), ElementsAre(Value(UBits(0, 32)), Value(UBits(0, 32)),
                                       Value(UBits(2, 32)))),
          Pair(A<Node*>(),
               testing::SizeIs(3))  // The write side of the register
          ));
}
TEST_P(BlockEvaluatorTest, ObserverSeesValuesOnBothEdges) {
  if (!SupportsObserver()) {
    GTEST_SKIP() << "Observers unsupported";
  }
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  BValue foo_inp = bb.InputPort("foo", p->GetBitsType(32));
  BValue enable = bb.InputPort("enable", p->GetBitsType(1));
  BValue rhs_inp = bb.InputPort("rhs", p->GetBitsType(32));
  BValue delay = bb.InsertRegister("delay", foo_inp, enable);
  BValue add_res = bb.Add(delay, rhs_inp);
  BValue out = bb.OutputPort("res", add_res);
  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  CollectingEvaluationObserver observer;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cont, evaluator().NewContinuation(
                     b, BlockEvaluator::OutputPortSampleTime::kAfterLastClock));
  XLS_ASSERT_OK(cont->SetObserver(&observer));
  // nb evaluator forces ports to zero at the start.
  XLS_ASSERT_OK(cont->RunOneCycle({{"foo", Value(UBits(1, 32))},
                                   {"enable", Value(UBits(0, 1))},
                                   {"rhs", Value(UBits(2, 32))}}));
  XLS_ASSERT_OK(cont->RunOneCycle({{"foo", Value(UBits(2, 32))},
                                   {"enable", Value(UBits(1, 1))},
                                   {"rhs", Value(UBits(3, 32))}}));
  XLS_ASSERT_OK(cont->RunOneCycle({{"foo", Value(UBits(0, 32))},
                                   {"enable", Value(UBits(0, 1))},
                                   {"rhs", Value(UBits(4, 32))}}));
  // NB The observer sees both the rising edge start value and the post falling
  // edge value so there are 6 entries despite only being 3 cycles.
  EXPECT_THAT(
      observer.values(),
      UnorderedElementsAre(
          Pair(foo_inp.node(),
               ElementsAre(Value(UBits(1, 32)), Value(UBits(1, 32)),
                           Value(UBits(2, 32)), Value(UBits(2, 32)),
                           Value(UBits(0, 32)), Value(UBits(0, 32)))),
          Pair(enable.node(),
               ElementsAre(Value(UBits(0, 1)), Value(UBits(0, 1)),
                           Value(UBits(1, 1)), Value(UBits(1, 1)),
                           Value(UBits(0, 1)), Value(UBits(0, 1)))),
          Pair(rhs_inp.node(),
               ElementsAre(Value(UBits(2, 32)), Value(UBits(2, 32)),
                           Value(UBits(3, 32)), Value(UBits(3, 32)),
                           Value(UBits(4, 32)), Value(UBits(4, 32)))),
          Pair(add_res.node(),
               ElementsAre(Value(UBits(2, 32)),  // rhs + 0-value
                           Value(UBits(2, 32)),  // cycle 1 result
                           Value(UBits(3, 32)), Value(UBits(5, 32)),
                           Value(UBits(6, 32)), Value(UBits(6, 32)))),
          Pair(out.node(), ElementsAre(Value::Tuple({}), Value::Tuple({}),
                                       Value::Tuple({}), Value::Tuple({}),
                                       Value::Tuple({}), Value::Tuple({}))),
          Pair(A<Node*>(),
               ElementsAre(Value(UBits(0, 32)), Value(UBits(0, 32)),
                           Value(UBits(0, 32)), Value(UBits(2, 32)),
                           Value(UBits(2, 32)), Value(UBits(2, 32)))),
          Pair(A<Node*>(),
               testing::SizeIs(6))  // The write side of the register
          ));
}

TEST_P(BlockEvaluatorTest, PackagesAreIndependent) {
  auto p1 = CreatePackage();
  auto p2 = CreatePackage();
  BlockBuilder b1(TestName(), p1.get());
  BlockBuilder b2(TestName(), p2.get());
  b1.OutputPort("out", b1.Add(b1.Literal(UBits(1, 32)),
                              b1.InputPort("inp", p1->GetBitsType(32))));
  b2.OutputPort("out", b2.Subtract(b2.InputPort("inp", p2->GetBitsType(32)),
                                   b2.Literal(UBits(1, 32))));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block1, b1.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Block * block2, b2.Build());
  auto v = [](int64_t v) { return Value(UBits(v, 32)); };
  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      {{"inp", v(1)}},  {{"inp", v(10)}}, {{"inp", v(12)}}, {{"inp", v(22)}},
      {{"inp", v(33)}}, {{"inp", v(44)}}, {{"inp", v(55)}},
  };
  auto oracle_1 = [](auto v) { return v.bits().ToUint64().value() + 1; };
  auto oracle_2 = [](auto v) { return v.bits().ToUint64().value() - 1; };
  XLS_ASSERT_OK_AND_ASSIGN(auto e1,
                           evaluator().NewContinuation(
                               block1, OutputPortSampleTime::kAfterLastClock));
  XLS_ASSERT_OK_AND_ASSIGN(auto e2,
                           evaluator().NewContinuation(
                               block2, OutputPortSampleTime::kAfterLastClock));
  for (const auto& inp : inputs) {
    XLS_EXPECT_OK(e1->RunOneCycle(inp));
    XLS_EXPECT_OK(e2->RunOneCycle(inp));
    EXPECT_THAT(e1->output_ports().at("out").bits().ToUint64(),
                IsOkAndHolds(oracle_1(inp.at("inp"))));
    EXPECT_THAT(e2->output_ports().at("out").bits().ToUint64(),
                IsOkAndHolds(oracle_2(inp.at("inp"))));
  }
}

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

TEST_P(BlockEvaluatorTest, StatelessInstantiatedPassthrough) {
  auto package = CreatePackage();
  Block* inner;
  {
    BlockBuilder b(absl::StrCat(TestName(), "_inner"), package.get());
    b.OutputPort("out2", b.InputPort("in2", package->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(inner, b.Build());
  }
  Block* outer;
  {
    BlockBuilder b(TestName(), package.get());
    XLS_ASSERT_OK_AND_ASSIGN(BlockInstantiation * inst,
                             b.block()->AddBlockInstantiation("inst", inner));
    b.InstantiationInput(inst, "in2",
                         b.InputPort("in", package->GetBitsType(32)));
    b.OutputPort("out", b.InstantiationOutput(inst, "out2"));
    XLS_ASSERT_OK_AND_ASSIGN(outer, b.Build());
  }

  EXPECT_THAT(
      evaluator().EvaluateCombinationalBlock(
          outer, absl::flat_hash_map<std::string, Value>(
                     {{"in", Value(UBits(123, 32))}})),
      IsOkAndHolds(UnorderedElementsAre(Pair("out", Value(UBits(123, 32))))));
}

TEST_P(BlockEvaluatorTest, PipelinedHierarchicalRotate) {
  auto package = CreatePackage();
  auto rot = [&](int64_t n, int64_t r) -> absl::StatusOr<Block*> {
    BlockBuilder b(absl::StrCat(TestName(), "_rot", n, "_", r), package.get());
    XLS_RETURN_IF_ERROR(b.AddClockPort("clk"));
    std::vector<BValue> inputs;
    inputs.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
      inputs.push_back(
          b.InputPort(absl::StrCat("in", i), package->GetBitsType(8)));
    }
    for (int64_t i = 0; i < n; ++i) {
      int64_t rot_idx = (i + r) % n;
      b.OutputPort(absl::StrCat("out", i),
                   b.InsertRegister(absl::StrCat("reg", i), inputs[rot_idx]));
    }
    return b.Build();
  };
  auto multi_rot = [&](int64_t n) -> absl::StatusOr<Block*> {
    BlockBuilder b(absl::StrCat(TestName(), "multirot", n), package.get());
    XLS_RETURN_IF_ERROR(b.AddClockPort("clk"));
    std::vector<BValue> inputs;
    inputs.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
      inputs.push_back(
          b.InputPort(absl::StrCat("in", i), package->GetBitsType(8)));
    }
    for (int64_t i = 0; i < n; ++i) {
      XLS_ASSIGN_OR_RETURN(Block * rot_block, rot(n, i));
      XLS_ASSIGN_OR_RETURN(BlockInstantiation * inst,
                           b.block()->AddBlockInstantiation(
                               absl::StrFormat("rot%d_inst", i), rot_block));
      for (int64_t j = 0; j < n; ++j) {
        b.InstantiationInput(inst, absl::StrCat("in", j), inputs[j]);
        inputs[j] = b.InstantiationOutput(inst, absl::StrCat("out", j));
      }
    }
    for (int64_t i = 0; i < n; ++i) {
      b.OutputPort(absl::StrCat("out", i), inputs[i]);
    }
    return b.Build();
  };
  XLS_ASSERT_OK_AND_ASSIGN(Block * outer, multi_rot(4));

  auto in_t = [](int64_t timestep) -> absl::flat_hash_map<std::string, Value> {
    absl::flat_hash_map<std::string, Value> inputs;
    inputs.reserve(4);
    for (int64_t i = 0; i < 4; ++i) {
      inputs[absl::StrCat("in", i)] = Value(UBits(timestep + i, 8));
    }
    return inputs;
  };
  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      in_t(0), in_t(1), in_t(2), in_t(3), in_t(4), in_t(5), in_t(6), in_t(7),
  };

  // Net result is to rotate inputs by 2 delayed by 4 cycles.
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  outer, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(
                  // t = 0
                  UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                                       Pair("out1", Value(UBits(0, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(0, 8)))),
                  // t = 1
                  UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                                       Pair("out1", Value(UBits(0, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(0, 8)))),
                  // t = 2
                  UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                                       Pair("out1", Value(UBits(0, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(0, 8)))),
                  // t = 3
                  UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                                       Pair("out1", Value(UBits(0, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(0, 8)))),
                  // t = 4
                  UnorderedElementsAre(Pair("out0", Value(UBits(2, 8))),
                                       Pair("out1", Value(UBits(3, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(1, 8)))),
                  // t = 5
                  UnorderedElementsAre(Pair("out0", Value(UBits(3, 8))),
                                       Pair("out1", Value(UBits(4, 8))),
                                       Pair("out2", Value(UBits(1, 8))),
                                       Pair("out3", Value(UBits(2, 8)))),

                  // t = 6
                  UnorderedElementsAre(Pair("out0", Value(UBits(4, 8))),
                                       Pair("out1", Value(UBits(5, 8))),
                                       Pair("out2", Value(UBits(2, 8))),
                                       Pair("out3", Value(UBits(3, 8)))),
                  // t = 7
                  UnorderedElementsAre(Pair("out0", Value(UBits(5, 8))),
                                       Pair("out1", Value(UBits(6, 8))),
                                       Pair("out2", Value(UBits(3, 8))),
                                       Pair("out3", Value(UBits(4, 8)))))));
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  outer, inputs, OutputPortSampleTime::kAfterLastClock),
              IsOkAndHolds(ElementsAre(
                  // t = 0 (before the first tick).
                  // UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                  //                      Pair("out1", Value(UBits(0, 8))),
                  //                      Pair("out2", Value(UBits(0, 8))),
                  //                      Pair("out3", Value(UBits(0, 8)))),
                  // t = 1
                  UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                                       Pair("out1", Value(UBits(0, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(0, 8)))),
                  // t = 2
                  UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                                       Pair("out1", Value(UBits(0, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(0, 8)))),
                  // t = 3
                  UnorderedElementsAre(Pair("out0", Value(UBits(0, 8))),
                                       Pair("out1", Value(UBits(0, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(0, 8)))),
                  // t = 4
                  UnorderedElementsAre(Pair("out0", Value(UBits(2, 8))),
                                       Pair("out1", Value(UBits(3, 8))),
                                       Pair("out2", Value(UBits(0, 8))),
                                       Pair("out3", Value(UBits(1, 8)))),
                  // t = 5
                  UnorderedElementsAre(Pair("out0", Value(UBits(3, 8))),
                                       Pair("out1", Value(UBits(4, 8))),
                                       Pair("out2", Value(UBits(1, 8))),
                                       Pair("out3", Value(UBits(2, 8)))),

                  // t = 6
                  UnorderedElementsAre(Pair("out0", Value(UBits(4, 8))),
                                       Pair("out1", Value(UBits(5, 8))),
                                       Pair("out2", Value(UBits(2, 8))),
                                       Pair("out3", Value(UBits(3, 8)))),
                  // t = 7
                  UnorderedElementsAre(Pair("out0", Value(UBits(5, 8))),
                                       Pair("out1", Value(UBits(6, 8))),
                                       Pair("out2", Value(UBits(3, 8))),
                                       Pair("out3", Value(UBits(4, 8)))),
                  // t = 8
                  UnorderedElementsAre(Pair("out0", Value(UBits(6, 8))),
                                       Pair("out1", Value(UBits(7, 8))),
                                       Pair("out2", Value(UBits(4, 8))),
                                       Pair("out3", Value(UBits(5, 8)))))));
}

absl::flat_hash_map<std::string, Value> PushAndPopInputs(int64_t data) {
  return absl::flat_hash_map<std::string, Value>{
      {"push_data", Value(UBits(data, 32))},
      {"push_valid", Value(UBits(1, 1))},
      {"pop_ready", Value(UBits(1, 1))},
  };
}

absl::flat_hash_map<std::string, Value> PushNoPopInputs(int64_t data) {
  return absl::flat_hash_map<std::string, Value>{
      {"push_data", Value(UBits(data, 32))},
      {"push_valid", Value(UBits(1, 1))},
      {"pop_ready", Value(UBits(0, 1))},
  };
}

absl::flat_hash_map<std::string, Value> PopOnlyInputs() {
  return absl::flat_hash_map<std::string, Value>{
      {"push_data", Value(UBits(0, 32))},
      {"push_valid", Value(UBits(0, 1))},
      {"pop_ready", Value(UBits(1, 1))},
  };
}

absl::flat_hash_map<std::string, Value> NoopInputs() {
  return absl::flat_hash_map<std::string, Value>{
      {"push_data", Value(UBits(0, 32))},
      {"push_valid", Value(UBits(0, 1))},
      {"pop_ready", Value(UBits(0, 1))},
  };
}

MATCHER_P(TryPopValue, matcher, "") {
  if (arg.at("pop_valid").IsAllZeros()) {
    *result_listener << "No valid pop!";
    return false;
  }
  return ExplainMatchResult(IsOkAndHolds(matcher),
                            arg.at("pop_data").bits().ToInt64(),
                            result_listener);
}

MATCHER(NoTryPopValue, "") {
  if (arg.at("pop_valid").IsAllOnes()) {
    *result_listener << absl::StreamFormat("Unexpected pop! Saw value %s.",
                                           arg.at("pop_data").ToString());
    return false;
  }
  return true;
}

TEST_P(BlockEvaluatorTest, SingleElementFifoInstantiationNoBypassWorks) {
  // TODO(rigge): add instantiation support to block jit and remove this guard.
  if (!SupportsFifos()) {
    GTEST_SKIP();
    return;
  }
  auto p = CreatePackage();
  Type* u1 = p->GetBitsType(1);
  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("fifo_wrapper", p.get());
  FifoConfig fifo_config(/*depth=*/1, /*bypass=*/false,
                         /*register_push_outputs=*/true,
                         /*register_pop_outputs=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fifo_config, u32));

  bb.OutputPort("pop_data", bb.InstantiationOutput(fifo_inst, "pop_data"));
  bb.OutputPort("pop_valid", bb.InstantiationOutput(fifo_inst, "pop_valid"));
  bb.OutputPort("push_ready", bb.InstantiationOutput(fifo_inst, "push_ready"));

  BValue reset = bb.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.InstantiationInput(fifo_inst, "rst", reset);

  // Make push side.
  bb.InstantiationInput(fifo_inst, "push_data", bb.InputPort("push_data", u32));
  bb.InstantiationInput(fifo_inst, "push_valid",
                        bb.InputPort("push_valid", u1));
  bb.InstantiationInput(fifo_inst, "pop_ready", bb.InputPort("pop_ready", u1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      // Try pushing more than the capacity (do not pop).
      PushNoPopInputs(0),
      PushNoPopInputs(1),
      PushNoPopInputs(2),
      // Noop should change nothing.
      NoopInputs(),
      NoopInputs(),
      NoopInputs(),
      // Pop and try to push
      PushAndPopInputs(3),
      PushAndPopInputs(4),
      PushAndPopInputs(5),
      // Flush for a while
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      // Push and pop a few times
      PushAndPopInputs(6),
      PushAndPopInputs(7),
      PushAndPopInputs(8),
      PushAndPopInputs(9),
  };

  // Add reset signals to inputs.
  for (absl::flat_hash_map<std::string, Value>& input_values : inputs) {
    input_values["rst"] = Value(UBits(0, 1));
  }

  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(NoTryPopValue(),  //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   // actually pops
                                       NoTryPopValue(),  //
                                       TryPopValue(4),   // actually pops
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       TryPopValue(6),   // actually pops
                                       NoTryPopValue(),  //
                                       TryPopValue(8)    // actually pops
                                       )));
}

TEST_P(BlockEvaluatorTest, SingleElementFifoInstantiationWithBypassWorks) {
  // TODO(rigge): add instantiation support to block jit and remove this guard.
  if (!SupportsFifos()) {
    GTEST_SKIP();
    return;
  }
  auto p = CreatePackage();
  Type* u1 = p->GetBitsType(1);
  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("fifo_wrapper", p.get());
  FifoConfig fifo_config(/*depth=*/1, /*bypass=*/true,
                         /*register_push_outputs=*/false,
                         /*register_pop_outputs=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fifo_config, u32));

  bb.OutputPort("pop_data", bb.InstantiationOutput(fifo_inst, "pop_data"));
  bb.OutputPort("pop_valid", bb.InstantiationOutput(fifo_inst, "pop_valid"));
  bb.OutputPort("push_ready", bb.InstantiationOutput(fifo_inst, "push_ready"));

  BValue reset = bb.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.InstantiationInput(fifo_inst, "rst", reset);

  // Make push side.
  bb.InstantiationInput(fifo_inst, "push_data", bb.InputPort("push_data", u32));
  bb.InstantiationInput(fifo_inst, "push_valid",
                        bb.InputPort("push_valid", u1));
  bb.InstantiationInput(fifo_inst, "pop_ready", bb.InputPort("pop_ready", u1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      // Try pushing more than the capacity (do not pop).
      PushNoPopInputs(0),
      PushNoPopInputs(1),
      PushNoPopInputs(2),
      // Noop should change nothing.
      NoopInputs(),
      NoopInputs(),
      NoopInputs(),
      // Pop and try to push
      PushAndPopInputs(3),
      PushNoPopInputs(4),
      // Flush for a while
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      // Push and pop a few times
      PushAndPopInputs(6),
      PushAndPopInputs(7),
      PushAndPopInputs(8),
      PushAndPopInputs(9),
  };

  // Add reset signals to inputs.
  for (absl::flat_hash_map<std::string, Value>& input_values : inputs) {
    input_values["rst"] = Value(UBits(0, 1));
  }

  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   // actually pops
                                       TryPopValue(3),   //
                                       TryPopValue(3),   // actually pops
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       TryPopValue(6),   //
                                       TryPopValue(7),   // actually pops
                                       TryPopValue(8),   // actually pops
                                       TryPopValue(9)    // actually pops
                                       )));
}

TEST_P(BlockEvaluatorTest, FifoInstantiationNoBypassWorks) {
  // TODO(rigge): add instantiation support to block jit and remove this guard.
  if (!SupportsFifos()) {
    GTEST_SKIP();
    return;
  }
  auto p = CreatePackage();
  Type* u1 = p->GetBitsType(1);
  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("fifo_wrapper", p.get());
  FifoConfig fifo_config(/*depth=*/5, /*bypass=*/false,
                         /*register_push_outputs=*/true,
                         /*register_pop_outputs=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fifo_config, u32));

  bb.OutputPort("pop_data", bb.InstantiationOutput(fifo_inst, "pop_data"));
  bb.OutputPort("pop_valid", bb.InstantiationOutput(fifo_inst, "pop_valid"));
  bb.OutputPort("push_ready", bb.InstantiationOutput(fifo_inst, "push_ready"));

  BValue reset = bb.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.InstantiationInput(fifo_inst, "rst", reset);

  // Make push side.
  bb.InstantiationInput(fifo_inst, "push_data", bb.InputPort("push_data", u32));
  bb.InstantiationInput(fifo_inst, "push_valid",
                        bb.InputPort("push_valid", u1));
  bb.InstantiationInput(fifo_inst, "pop_ready", bb.InputPort("pop_ready", u1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      // Try pushing more than the capacity (do not pop).
      PushNoPopInputs(0),
      PushNoPopInputs(1),
      PushNoPopInputs(2),
      PushNoPopInputs(3),
      PushNoPopInputs(4),
      PushNoPopInputs(5),
      PushNoPopInputs(6),
      // Noop should change nothing.
      NoopInputs(),
      NoopInputs(),
      NoopInputs(),
      // Pop and try to push
      PushAndPopInputs(7),
      PushAndPopInputs(8),
      PushAndPopInputs(9),
      PushAndPopInputs(10),
      PushAndPopInputs(11),
      PushAndPopInputs(12),
      // Flush for a while
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      // Push and pop steady state
      PushAndPopInputs(13),
      PushAndPopInputs(14),
      PushAndPopInputs(15),
      PushAndPopInputs(16),
      PushAndPopInputs(17),
      PushAndPopInputs(18),
      PushAndPopInputs(19),
      PushAndPopInputs(20),
  };
  // Add reset signals to inputs.
  for (absl::flat_hash_map<std::string, Value>& input_values : inputs) {
    input_values["rst"] = Value(UBits(0, 1));
  }

  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(NoTryPopValue(),  //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   // actually pops
                                       TryPopValue(1),   // actually pops
                                       TryPopValue(2),   // actually pops
                                       TryPopValue(3),   // actually pops
                                       TryPopValue(4),   // actually pops
                                       TryPopValue(8),   // actually pops
                                       TryPopValue(9),   // actually pops
                                       TryPopValue(10),  // actually pops
                                       TryPopValue(11),  // actually pops
                                       TryPopValue(12),  // actually pops
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       NoTryPopValue(),  //
                                       TryPopValue(13),  // actually pops
                                       TryPopValue(14),  // actually pops
                                       TryPopValue(15),  // actually pops
                                       TryPopValue(16),  // actually pops
                                       TryPopValue(17),  // actually pops
                                       TryPopValue(18),  // actually pops
                                       TryPopValue(19)   // actually pops
                                       )));
}

TEST_P(BlockEvaluatorTest, FifoInstantiationWithBypassWorks) {
  // TODO(rigge): add instantiation support to block jit and remove this guard.
  if (!SupportsFifos()) {
    GTEST_SKIP();
    return;
  }
  auto p = CreatePackage();
  Type* u1 = p->GetBitsType(1);
  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("fifo_wrapper", p.get());
  FifoConfig fifo_config(/*depth=*/5, /*bypass=*/true,
                         /*register_push_outputs=*/false,
                         /*register_pop_outputs=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fifo_config, u32));

  BValue reset = bb.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.InstantiationInput(fifo_inst, "rst", reset);

  bb.OutputPort("pop_data", bb.InstantiationOutput(fifo_inst, "pop_data"));
  bb.OutputPort("pop_valid", bb.InstantiationOutput(fifo_inst, "pop_valid"));
  bb.OutputPort("push_ready", bb.InstantiationOutput(fifo_inst, "push_ready"));

  // Make push side.
  bb.InstantiationInput(fifo_inst, "push_data", bb.InputPort("push_data", u32));
  bb.InstantiationInput(fifo_inst, "push_valid",
                        bb.InputPort("push_valid", u1));
  bb.InstantiationInput(fifo_inst, "pop_ready", bb.InputPort("pop_ready", u1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      // Try pushing more than the capacity (do not pop).
      PushNoPopInputs(0),
      PushNoPopInputs(1),
      PushNoPopInputs(2),
      PushNoPopInputs(3),
      PushNoPopInputs(4),
      PushNoPopInputs(5),
      PushNoPopInputs(6),
      // Noop should change nothing.
      NoopInputs(),
      NoopInputs(),
      NoopInputs(),
      // Pop and try to push
      PushAndPopInputs(7),
      PushAndPopInputs(8),
      PushAndPopInputs(9),
      PushAndPopInputs(10),
      PushAndPopInputs(11),
      PushAndPopInputs(12),
      // Flush for a while
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      PopOnlyInputs(),
      // Push and pop steady state
      PushAndPopInputs(13),
      PushAndPopInputs(14),
      PushAndPopInputs(15),
      PushAndPopInputs(16),
      PushAndPopInputs(17),
      PushAndPopInputs(18),
      PushAndPopInputs(19),
      PushAndPopInputs(20),
  };
  // Add reset signals to inputs.
  for (absl::flat_hash_map<std::string, Value>& input_values : inputs) {
    input_values["rst"] = Value(UBits(0, 1));
  }

  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   //
                                       TryPopValue(0),   // actually pops
                                       TryPopValue(1),   // actually pops
                                       TryPopValue(2),   // actually pops
                                       TryPopValue(3),   // actually pops
                                       TryPopValue(4),   // actually pops
                                       TryPopValue(7),   // actually pops
                                       TryPopValue(8),   // actually pops
                                       TryPopValue(9),   // actually pops
                                       TryPopValue(10),  // actually pops
                                       TryPopValue(11),  // actually pops
                                       TryPopValue(12),  // actually pops
                                       NoTryPopValue(),  //
                                       TryPopValue(13),  // actually pops
                                       TryPopValue(14),  // actually pops
                                       TryPopValue(15),  // actually pops
                                       TryPopValue(16),  // actually pops
                                       TryPopValue(17),  // actually pops
                                       TryPopValue(18),  // actually pops
                                       TryPopValue(19),  // actually pops
                                       TryPopValue(20)   // actually pops
                                       )));
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

void PipelinedAdderCommon(BlockEvaluatorTest& bet,
                          OutputPortSampleTime sample_time) {
  auto package = bet.CreatePackage();
  BlockBuilder b(bet.TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue y = b.InputPort("y", package->GetBitsType(32));

  BValue x_d = b.InsertRegister("x_d", x);
  BValue y_d = b.InsertRegister("y_d", y);

  BValue x_plus_y = b.Add(x_d, y_d);

  BValue x_plus_y_d = b.InsertRegister("x_plus_y_d", x_plus_y);

  if (sample_time == OutputPortSampleTime::kAfterLastClock) {
    // Put a flop in the design manually.
    b.FloppedOutputPort("out", x_plus_y_d);
  } else {
    // rely on the synthetic flop.
    b.OutputPort("out", x_plus_y_d);
  }

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"x", 1}, {"y", 2}},
      {{"x", 42}, {"y", 100}},
      {{"x", 0}, {"y", 0}},
      {{"x", 0}, {"y", 0}},
      {{"x", 0}, {"y", 0}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  EXPECT_THAT(
      bet.evaluator().EvaluateSequentialBlock(block, inputs, sample_time),
      IsOkAndHolds(ElementsAre(UnorderedElementsAre(Pair("out", 0)),
                               UnorderedElementsAre(Pair("out", 0)),
                               UnorderedElementsAre(Pair("out", 3)),
                               UnorderedElementsAre(Pair("out", 142)),
                               UnorderedElementsAre(Pair("out", 0)))));
}

TEST_P(BlockEvaluatorTest, PipelinedAdderRaw) {
  PipelinedAdderCommon(*this, OutputPortSampleTime::kAfterLastClock);
}
TEST_P(BlockEvaluatorTest, PipelinedAdderClocked) {
  PipelinedAdderCommon(*this, OutputPortSampleTime::kAtLastPosEdgeClock);
}

TEST_P(BlockEvaluatorTest, RegisterWithReset) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue rst = b.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});

  BValue x_d = b.InsertRegister("x_d", x, rst, Value(UBits(42, 32)));

  b.OutputPort("out", x_d);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"rst", 0}, {"x", 1}},
      {{"rst", 1}, {"x", 2}},
      {{"rst", 1}, {"x", 3}},
      {{"rst", 0}, {"x", 4}},
      {{"rst", 0}, {"x", 5}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAfterLastClock),
              IsOkAndHolds(ElementsAre(UnorderedElementsAre(Pair("out", 1)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 4)),
                                       UnorderedElementsAre(Pair("out", 5)))));
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(UnorderedElementsAre(Pair("out", 0)),
                                       UnorderedElementsAre(Pair("out", 1)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 4)))));
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
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAfterLastClock),
              IsOkAndHolds(ElementsAre(UnorderedElementsAre(Pair("out", 0)),
                                       UnorderedElementsAre(Pair("out", 2)),
                                       UnorderedElementsAre(Pair("out", 3)),
                                       UnorderedElementsAre(Pair("out", 3)),
                                       UnorderedElementsAre(Pair("out", 3)))));
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(UnorderedElementsAre(Pair("out", 0)),
                                       UnorderedElementsAre(Pair("out", 0)),
                                       UnorderedElementsAre(Pair("out", 2)),
                                       UnorderedElementsAre(Pair("out", 3)),
                                       UnorderedElementsAre(Pair("out", 3)))));
}

TEST_P(BlockEvaluatorTest, RegisterWithResetAndLoadEnable) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue rst_n = b.ResetPort(
      "rst_n", ResetBehavior{.asynchronous = false, .active_low = true});
  BValue le = b.InputPort("le", package->GetBitsType(1));

  BValue x_d = b.InsertRegister("x_d", x, rst_n, Value(UBits(42, 32)), le);

  b.OutputPort("out", x_d);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"rst_n", 1}, {"le", 0}, {"x", 1}},
      {{"rst_n", 0}, {"le", 0}, {"x", 2}},
      {{"rst_n", 0}, {"le", 1}, {"x", 3}},
      {{"rst_n", 1}, {"le", 1}, {"x", 4}},
      {{"rst_n", 1}, {"le", 0}, {"x", 5}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAfterLastClock),
              IsOkAndHolds(ElementsAre(UnorderedElementsAre(Pair("out", 0)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 4)),
                                       UnorderedElementsAre(Pair("out", 4)))));
  EXPECT_THAT(evaluator().EvaluateSequentialBlock(
                  block, inputs, OutputPortSampleTime::kAtLastPosEdgeClock),
              IsOkAndHolds(ElementsAre(UnorderedElementsAre(Pair("out", 0)),
                                       UnorderedElementsAre(Pair("out", 0)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 42)),
                                       UnorderedElementsAre(Pair("out", 4)))));
}

void AccumulatorRegisterCommon(BlockEvaluatorTest& bet,
                               OutputPortSampleTime sample_time) {
  auto package = bet.CreatePackage();
  BlockBuilder b(bet.TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg,
      b.block()->AddRegister("accum", package->GetBitsType(32)));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue accum = b.RegisterRead(reg);
  BValue next_accum = b.Add(x, accum);
  b.RegisterWrite(reg, next_accum);
  if (sample_time == OutputPortSampleTime::kAfterLastClock) {
    // Manually flop.
    b.FloppedOutputPort("out", next_accum);
  } else {
    b.OutputPort("out", next_accum);
  }

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"x", 1}}, {{"x", 2}}, {{"x", 3}}, {{"x", 4}}, {{"x", 5}}};
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, bet.evaluator().EvaluateSequentialBlock(
                                        block, inputs, sample_time));

  ASSERT_EQ(outputs.size(), 5);
  EXPECT_THAT(outputs.at(0), UnorderedElementsAre(Pair("out", 1)));
  EXPECT_THAT(outputs.at(1), UnorderedElementsAre(Pair("out", 3)));
  EXPECT_THAT(outputs.at(2), UnorderedElementsAre(Pair("out", 6)));
  EXPECT_THAT(outputs.at(3), UnorderedElementsAre(Pair("out", 10)));
  EXPECT_THAT(outputs.at(4), UnorderedElementsAre(Pair("out", 15)));
}
TEST_P(BlockEvaluatorTest, AccumulatorRegisterClocked) {
  AccumulatorRegisterCommon(*this, OutputPortSampleTime::kAtLastPosEdgeClock);
}
TEST_P(BlockEvaluatorTest, AccumulatorRegisterRaw) {
  AccumulatorRegisterCommon(*this, OutputPortSampleTime::kAfterLastClock);
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
  // NB Channelized function uses synthetic flops for taps.
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

  BValue rst = b.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});
  verilog::ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(false);
  reset.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg, b.block()->AddRegister("accum", package->GetBitsType(32),
                                             Value(UBits(0, 32))));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue x_vld = b.InputPort("x_vld", package->GetBitsType(1));
  BValue out_rdy = b.InputPort("out_rdy", package->GetBitsType(1));

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
    std::vector<ChannelSource> sources{ChannelSource(
        "x", "x_vld", "x_rdy", 1.0, block,
        /*reset_behavior=*/ChannelSource::BehaviorDuringReset::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink(
            "out", "out_vld", "out_rdy", 1.0, block,
            /*reset_behavior=*/ChannelSink::BehaviorDuringReset::kAttendValid),
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
    std::vector<ChannelSource> sources{ChannelSource(
        "x", "x_vld", "x_rdy", 1.0, block,
        /*reset_behavior=*/ChannelSource::BehaviorDuringReset::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink(
            "out", "out_vld", "out_rdy", 1.0, block,
            /*reset_behavior=*/ChannelSink::BehaviorDuringReset::kIgnoreValid),
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
    std::vector<ChannelSource> sources{ChannelSource(
        "x", "x_vld", "x_rdy", 1.0, block,
        /*reset_behavior=*/ChannelSource::BehaviorDuringReset::kIgnoreReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink(
            "out", "out_vld", "out_rdy", 1.0, block,
            /*reset_behavior=*/ChannelSink::BehaviorDuringReset::kIgnoreValid),
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

  BValue rst = b.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = true});
  verilog::ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(false);
  reset.set_active_low(true);

  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg, b.block()->AddRegister("accum", package->GetBitsType(32),
                                             Value(UBits(0, 32))));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue x_vld = b.InputPort("x_vld", package->GetBitsType(1));
  BValue out_rdy = b.InputPort("out_rdy", package->GetBitsType(1));

  BValue input_valid_and_output_ready = b.And(x_vld, out_rdy);
  BValue accum = b.RegisterRead(reg);
  BValue x_add_accum = b.Add(x, accum);
  BValue next_accum =
      b.Select(input_valid_and_output_ready, {accum, x_add_accum});

  // NB Channelized uses clocked taps.
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
    std::vector<ChannelSource> sources{ChannelSource(
        "x", "x_vld", "x_rdy", 1.0, block,
        /*reset_behavior=*/ChannelSource::BehaviorDuringReset::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink(
            "out", "out_vld", "out_rdy", 1.0, block,
            /*reset_behavior=*/ChannelSink::BehaviorDuringReset::kAttendValid),
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
    std::vector<ChannelSource> sources{ChannelSource(
        "x", "x_vld", "x_rdy", 1.0, block,
        /*reset_behavior=*/ChannelSource::BehaviorDuringReset::kAttendReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink(
            "out", "out_vld", "out_rdy", 1.0, block,
            /*reset_behavior=*/ChannelSink::BehaviorDuringReset::kIgnoreValid),
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
    std::vector<ChannelSource> sources{ChannelSource(
        "x", "x_vld", "x_rdy", 1.0, block,
        /*reset_behavior=*/ChannelSource::BehaviorDuringReset::kIgnoreReady)};
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    std::vector<ChannelSink> sinks{
        ChannelSink(
            "out", "out_vld", "out_rdy", 1.0, block,
            /*reset_behavior=*/ChannelSink::BehaviorDuringReset::kIgnoreValid),
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
  BValue first_trace = b.Trace(assertion, b.Literal(Value(UBits(1, 1))), {x},
                               {"x is ", FormatPreference::kDefault});
  b.Trace(first_trace, b.Literal(Value(UBits(1, 1))), {x},
          {"I'm emphasizing that x is ", FormatPreference::kDefault},
          /*verbosity=*/3);

  b.FloppedOutputPort("y", x);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        auto cont,
        evaluator().NewContinuation(
            block, BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock));
    XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(10, 32))}}));
    BlockRunResult result{
        .outputs = cont->output_ports(),
        .reg_state = cont->registers(),
        .interpreter_events = cont->events(),
    };

    EXPECT_THAT(result.interpreter_events.GetTraceMessageStrings(),
                ElementsAre("x is 10", "I'm emphasizing that x is 10"));
    EXPECT_THAT(result.interpreter_events.GetAssertMessages(), IsEmpty());
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        auto cont,
        evaluator().NewContinuation(
            block, BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock));
    XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(3, 32))}}));
    BlockRunResult result{
        .outputs = cont->output_ports(),
        .reg_state = cont->registers(),
        .interpreter_events = cont->events(),
    };

    EXPECT_THAT(result.interpreter_events.GetTraceMessageStrings(),
                ElementsAre("x is 3", "I'm emphasizing that x is 3"));
    EXPECT_THAT(result.interpreter_events.GetAssertMessages(),
                ElementsAre("foo"));
  }
}

TEST_P(BlockEvaluatorTest, InterpreterEventsCapturedByChannelizedInterface) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue x_vld = b.InputPort("x_vld", package->GetBitsType(1));
  BValue y_rdy = b.InputPort("y_rdy", package->GetBitsType(1));
  b.OutputPort("y", x);
  b.OutputPort("x_rdy", y_rdy);
  b.OutputPort("y_vld", x_vld);

  BValue tkn = b.Literal(Value::Token());
  BValue fire = b.And({x_vld, y_rdy});
  BValue assert_cond = b.UGt(x, b.Literal(Value(UBits(5, 32))));
  BValue fire_implies_cond = b.Or(b.Not(fire), assert_cond);
  BValue assertion = b.Assert(tkn, fire_implies_cond, "foo");
  b.Trace(assertion, fire, {x},
          {"I'm emphasizing that x is ", FormatPreference::kDefault},
          /*verbosity=*/3);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  {
    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 0.5, block)};
    XLS_ASSERT_OK(
        sources.at(0).SetDataSequence(std::vector<uint64_t>{8, 7, 6, 5, 4}));

    std::vector<ChannelSink> sinks{
        ChannelSink("y", "y_vld", "y_rdy", 0.1, block),
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
    EXPECT_THAT(output_sequence, ElementsAre(8, 7, 6, 5, 4));
    EXPECT_THAT(block_io.interpreter_events.GetAssertMessages(),
                // Assertion fails for inputs 5 and 4.
                ElementsAre("foo", "foo"));
    EXPECT_THAT(block_io.interpreter_events.GetTraceMessageStrings(),
                Contains(HasSubstr("I'm emphasizing that x is ")).Times(5));
  }
}

TEST_P(BlockEvaluatorTest, TupleInputOutput) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());

  Type* type =
      package->GetTupleType({package->GetTupleType({package->GetBitsType(1),
                                                    package->GetBitsType(2)}),
                             package->GetTupleType({package->GetBitsType(4),
                                                    package->GetBitsType(8)})});
  BValue x = b.InputPort("x", type);
  b.OutputPort("o0", b.TupleIndex(x, 0));
  b.OutputPort("o1", b.TupleIndex(x, 1));
  b.OutputPort("o00", b.TupleIndex(b.TupleIndex(x, 0), 0));
  b.OutputPort("o01", b.TupleIndex(b.TupleIndex(x, 0), 1));
  b.OutputPort("o10", b.TupleIndex(b.TupleIndex(x, 1), 0));
  b.OutputPort("o11", b.TupleIndex(b.TupleIndex(x, 1), 1));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               block, OutputPortSampleTime::kAfterLastClock));
  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value::Tuple(
                 {Value::Tuple({Value(UBits(0, 1)), Value(UBits(1, 2))}),
                  Value::Tuple({Value(UBits(2, 4)), Value(UBits(3, 8))})})}}));

  EXPECT_THAT(
      cont->output_ports(),
      UnorderedElementsAre(
          Pair("o0", Value::Tuple({Value(UBits(0, 1)), Value(UBits(1, 2))})),
          Pair("o00", Value(UBits(0, 1))), Pair("o01", Value(UBits(1, 2))),
          Pair("o1", Value::Tuple({Value(UBits(2, 4)), Value(UBits(3, 8))})),
          Pair("o10", Value(UBits(2, 4))), Pair("o11", Value(UBits(3, 8)))));
}

TEST_P(BlockEvaluatorTest, TupleRegister) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());

  Value all_ones = Value::Tuple(
      {Value::Tuple({Value(UBits(0b1, 1)), Value(UBits(0b11, 2))}),
       Value::Tuple({Value(UBits(0b1111, 4)), Value(UBits(0xff, 8))})});
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));
  auto x = b.InsertRegister("x", b.Literal(all_ones));
  b.InsertRegister("o0", b.TupleIndex(x, 0));
  b.InsertRegister("o00", b.TupleIndex(b.TupleIndex(x, 0), 0));
  b.InsertRegister("o01", b.TupleIndex(b.TupleIndex(x, 0), 1));
  b.InsertRegister("o1", b.TupleIndex(x, 1));
  b.InsertRegister("o10", b.TupleIndex(b.TupleIndex(x, 1), 0));
  b.InsertRegister("o11", b.TupleIndex(b.TupleIndex(x, 1), 1));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());
  RecordProperty("ir", package->DumpIr());

  XLS_ASSERT_OK_AND_ASSIGN(auto cont, evaluator().NewContinuation(block));
  XLS_ASSERT_OK(cont->SetRegisters({
      {"x",
       Value::Tuple({Value::Tuple({Value(UBits(0, 1)), Value(UBits(1, 2))}),
                     Value::Tuple({Value(UBits(2, 4)), Value(UBits(3, 8))})})},
      {"o0", all_ones.element(0)},
      {"o1", all_ones.element(1)},
      {"o00", all_ones.element(0).element(0)},
      {"o01", all_ones.element(0).element(1)},
      {"o10", all_ones.element(1).element(0)},
      {"o11", all_ones.element(1).element(1)},
  }));
  XLS_ASSERT_OK(cont->RunOneCycle({}));

  EXPECT_THAT(
      cont->registers(),
      UnorderedElementsAre(
          Pair("x", all_ones),
          Pair("o0", Value::Tuple({Value(UBits(0, 1)), Value(UBits(1, 2))})),
          Pair("o00", Value(UBits(0, 1))), Pair("o01", Value(UBits(1, 2))),
          Pair("o1", Value::Tuple({Value(UBits(2, 4)), Value(UBits(3, 8))})),
          Pair("o10", Value(UBits(2, 4))), Pair("o11", Value(UBits(3, 8)))));
}

TEST_P(BlockEvaluatorTest, TypeChecksInputs) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  b.InputPort("test", package->GetBitsType(32));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont, evaluator().NewContinuation(block));
  auto result = cont->RunOneCycle(
      {{"test",
        Value::Tuple(
            {Value::Tuple({Value(UBits(0, 1)), Value(UBits(1, 2))}),
             Value::Tuple({Value(UBits(2, 4)), Value(UBits(3, 8))})})}});
  RecordProperty("error", result.ToString());
  EXPECT_THAT(result, Not(IsOk()));
}

TEST_P(BlockEvaluatorTest, TypeChecksRegister) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));
  b.InsertRegister("test", b.Literal(UBits(0, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont, evaluator().NewContinuation(block));

  auto result = cont->SetRegisters(
      {{"test",
        Value::Tuple(
            {Value::Tuple({Value(UBits(0, 1)), Value(UBits(1, 2))}),
             Value::Tuple({Value(UBits(2, 4)), Value(UBits(3, 8))})})}});
  RecordProperty("error", result.ToString());
  EXPECT_THAT(result, Not(IsOk()));
}

// TODO(allight): This is the current explicit behavior of the block-evaluator
// api and to reduce surprise also the direct JIT apis. We might want to
// consider if we should change it however since really the values are Xs and
// the difference can be visible in some circumstances.
TEST_P(BlockEvaluatorTest, InitialRegisterValueIsZero) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto r1,
                           bb.block()->AddRegister("test1", p->GetBitsType(16),
                                                   Value(UBits(1234, 16))));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue reset = bb.ResetPort(
      "reset", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.OutputPort("out", bb.RegisterRead(r1));
  bb.RegisterWrite(r1, bb.Literal(UBits(0xbeef, 16)),
                   /*load_enable=*/bb.Literal(UBits(0, 1)), reset);
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               blk, OutputPortSampleTime::kAtLastPosEdgeClock));
  XLS_ASSERT_OK(cont->RunOneCycle({{"reset", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("test1", Value(UBits(0, 16)))));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(0, 16)))));
}

// TODO(allight): This is the current explicit behavior of the block-evaluator
// api and to reduce surprise also the direct JIT apis. We might want to
// consider if we should change it however since really the values are Xs and
// the difference can be visible in some circumstances.
TEST_P(BlockEvaluatorTest, OutputPortsGetStartValue) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto r1,
                           bb.block()->AddRegister("test1", p->GetBitsType(16),
                                                   Value(UBits(1234, 16))));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue reset_port = bb.ResetPort(
      "reset", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.OutputPort("out", bb.RegisterRead(r1));
  bb.RegisterWrite(r1, bb.InputPort("in", p->GetBitsType(16)),
                   /*load_enable=*/std::nullopt, reset_port);
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               blk, OutputPortSampleTime::kAtLastPosEdgeClock));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("test1", Value(UBits(0, 16)))));
  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"in", Value(UBits(0, 16))}, {"reset", Value(UBits(1, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(0, 16)))));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("test1", Value(UBits(1234, 16)))));
}

TEST_P(BlockEvaluatorTest, SetRegistersContinuation) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue rd1 = b.InsertRegister("s1", x);
  BValue rd2 = b.InsertRegister("s2", rd1);
  BValue rd3 = b.InsertRegister("s3", rd2);
  BValue rd4 = b.InsertRegister("s4", rd3);
  b.OutputPort("x_out", x);
  b.OutputPort("v1", rd1);
  b.OutputPort("v2", rd2);
  b.OutputPort("v3", rd3);
  b.OutputPort("v4", rd4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<int32_t> inputs{1, 2,  3,  4,  5,  6,  7, 8,
                              9, 10, 11, 12, 13, 14, 15};
  struct Output {
    int32_t x_out;  // also value of register s1
    int32_t v1;     // also value of register s2
    int32_t v2;     // also value of register s3
    int32_t v3;     // also value of register s4
    int32_t v4;
  };
  std::vector<Output> outputs{
      {1, 1, 16, 16, 16},    // after cycle 1
      {2, 2, 1, 16, 16},     // after cycle 2
      {3, 3, 2, 1, 16},      // after cycle 3
      {4, 4, 3, 2, 1},       // after cycle 4
      {5, 5, 4, 3, 2},       // after cycle 5
      {6, 6, 5, 4, 3},       // after cycle 6
      {7, 7, 6, 5, 4},       // after cycle 7
      {8, 8, 7, 6, 5},       // after cycle 8
      {9, 9, 8, 7, 6},       // after cycle 9
      {10, 10, 9, 8, 7},     // after cycle 10
      {11, 11, 10, 9, 8},    // after cycle 11
      {12, 12, 11, 10, 9},   // after cycle 12
      {13, 13, 12, 11, 10},  // after cycle 13
      {14, 14, 13, 12, 11},  // after cycle 14
      {15, 15, 14, 13, 12},  // after cycle 15
  };
  auto in_it = inputs.cbegin();
  auto out_it = outputs.cbegin();
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               block, OutputPortSampleTime::kAfterLastClock));
  Value sixteen = Value(UBits(16, 32));
  XLS_ASSERT_OK(cont->SetRegisters(
      {{"s1", sixteen}, {"s2", sixteen}, {"s3", sixteen}, {"s4", sixteen}}));
  int64_t i = 0;
  for (; in_it != inputs.cend(); ++in_it, ++out_it, ++i) {
    auto expected = *out_it;
    XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(*in_it, 32))}}));
    EXPECT_THAT(
        cont->output_ports(),
        UnorderedElementsAre(Pair("x_out", Value(UBits(expected.x_out, 32))),
                             Pair("v1", Value(UBits(expected.v1, 32))),
                             Pair("v2", Value(UBits(expected.v2, 32))),
                             Pair("v3", Value(UBits(expected.v3, 32))),
                             Pair("v4", Value(UBits(expected.v4, 32)))));
    EXPECT_THAT(
        cont->registers(),
        UnorderedElementsAre(Pair("s1", Value(UBits(expected.v1, 32))),
                             Pair("s2", Value(UBits(expected.v2, 32))),
                             Pair("s3", Value(UBits(expected.v3, 32))),
                             Pair("s4", Value(UBits(expected.v4, 32)))));
  }
}

TEST_P(BlockEvaluatorTest, DelaysContinuation) {
  auto package = CreatePackage();
  BlockBuilder b(TestName(), package.get());
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", package->GetBitsType(32));
  BValue rd1 = b.InsertRegister("s1", x);
  BValue rd2 = b.InsertRegister("s2", rd1);
  BValue rd3 = b.InsertRegister("s3", rd2);
  BValue rd4 = b.InsertRegister("s4", rd3);
  b.FloppedOutputPort("x_out", x);
  b.FloppedOutputPort("v1", rd1);
  b.FloppedOutputPort("v2", rd2);
  b.FloppedOutputPort("v3", rd3);
  b.FloppedOutputPort("v4", rd4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  std::vector<int32_t> inputs{1, 2,  3,  4,  5,  6,  7, 8,
                              9, 10, 11, 12, 13, 14, 15};
  struct Output {
    int32_t x_out;  // also value of register s1
    int32_t v1;     // also value of register s2
    int32_t v2;     // also value of register s3
    int32_t v3;     // also value of register s4
    int32_t v4;
  };
  std::vector<Output> outputs{
      {1, 0, 0, 0, 0},       // after cycle 1
      {2, 1, 0, 0, 0},       // after cycle 2
      {3, 2, 1, 0, 0},       // after cycle 3
      {4, 3, 2, 1, 0},       // after cycle 4
      {5, 4, 3, 2, 1},       // after cycle 5
      {6, 5, 4, 3, 2},       // after cycle 6
      {7, 6, 5, 4, 3},       // after cycle 7
      {8, 7, 6, 5, 4},       // after cycle 8
      {9, 8, 7, 6, 5},       // after cycle 9
      {10, 9, 8, 7, 6},      // after cycle 10
      {11, 10, 9, 8, 7},     // after cycle 11
      {12, 11, 10, 9, 8},    // after cycle 12
      {13, 12, 11, 10, 9},   // after cycle 13
      {14, 13, 12, 11, 10},  // after cycle 14
      {15, 14, 13, 12, 11},  // after cycle 15
  };
  auto in_it = inputs.cbegin();
  auto out_it = outputs.cbegin();
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               block, OutputPortSampleTime::kAfterLastClock));
  for (; in_it != inputs.cend(); ++in_it, ++out_it) {
    auto expected = *out_it;
    XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(*in_it, 32))}}));
    EXPECT_THAT(
        cont->output_ports(),
        UnorderedElementsAre(Pair("x_out", Value(UBits(expected.x_out, 32))),
                             Pair("v1", Value(UBits(expected.v1, 32))),
                             Pair("v2", Value(UBits(expected.v2, 32))),
                             Pair("v3", Value(UBits(expected.v3, 32))),
                             Pair("v4", Value(UBits(expected.v4, 32)))));
    EXPECT_THAT(cont->registers(),
                AllOf(SizeIs(9),  // Output flops.
                      Contains(Pair("s1", Value(UBits(expected.x_out, 32)))),
                      Contains(Pair("s2", Value(UBits(expected.v1, 32)))),
                      Contains(Pair("s3", Value(UBits(expected.v2, 32)))),
                      Contains(Pair("s4", Value(UBits(expected.v3, 32))))));
  }
}

TEST_P(BlockEvaluatorTest, RegUpdateHappensBeforeWireUpdate) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  BValue in_1 = bb.InputPort("in_1", p->GetBitsType(16));
  BValue in_2 = bb.InputPort("in_2", p->GetBitsType(16));
  BValue reg1 = bb.InsertRegister("reg1", in_1);
  BValue reg2 = bb.InsertRegister("reg2", in_2);
  BValue sum = bb.Add(reg1, reg2);
  bb.OutputPort("out_1", sum);
  BValue reg3 = bb.InsertRegister("reg3", sum);
  bb.OutputPort("out_2", reg3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               block, OutputPortSampleTime::kAfterLastClock));
  XLS_ASSERT_OK(cont->SetRegisters({
      {"reg1", Value(UBits(0xbe00, 16))},
      {"reg2", Value(UBits(0x00ef, 16))},
      {"reg3", Value(UBits(0xdead, 16))},
  }));
  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"in_1", Value(UBits(1, 16))}, {"in_2", Value(UBits(2, 16))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out_1", Value(UBits(3, 16))),
                                   Pair("out_2", Value(UBits(0xbeef, 16)))));
}

TEST_P(FifoTest, FifosReset) {
  // TODO(rigge): add instantiation support to block jit and remove this guard.
  if (!SupportsFifos()) {
    GTEST_SKIP();
    return;
  }
  auto p = CreatePackage();
  Type* u1 = p->GetBitsType(1);
  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("fifo_wrapper", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fifo_config(), u32));

  bb.OutputPort("pop_data", bb.InstantiationOutput(fifo_inst, "pop_data"));
  bb.OutputPort("pop_valid", bb.InstantiationOutput(fifo_inst, "pop_valid"));
  bb.OutputPort("push_ready", bb.InstantiationOutput(fifo_inst, "push_ready"));

  // Make reset.
  BValue reset = bb.ResetPort(
      "reset", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.InstantiationInput(fifo_inst, "rst", reset);

  // Make push side.
  bb.InstantiationInput(fifo_inst, "push_data", bb.InputPort("push_data", u32));
  bb.InstantiationInput(fifo_inst, "push_valid",
                        bb.InputPort("push_valid", u1));
  bb.InstantiationInput(fifo_inst, "pop_ready", bb.InputPort("pop_ready", u1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BlockContinuation> eval,
      evaluator().NewContinuation(
          block, BlockEvaluator::OutputPortSampleTime::kAfterLastClock));
  // Reset first.
  XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(1, 1))},
                                   {"push_data", ZeroOfType(u32)},
                                   {"push_valid", ZeroOfType(u1)},
                                   {"pop_ready", ZeroOfType(u1)}}));
  // Fill the FIFO completely.
  for (int i = 0; i < fifo_config().depth() + 1; ++i) {
    XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                     {"push_data", ZeroOfType(u32)},
                                     {"push_valid", Value(UBits(1, 1))},
                                     {"pop_ready", Value(UBits(0, 1))}}));
  }
  EXPECT_THAT(eval->output_ports(),
              UnorderedElementsAre(
                  // We only pushed 0s, so we should get 0s out.
                  Pair("pop_data", Value(UBits(0, 32))),
                  // Fifo is full, pop should be valid.
                  Pair("pop_valid", Value(UBits(1, 1))),
                  // Fifo is full, push should not be ready.
                  Pair("push_ready", Value(UBits(0, 1)))));

  // Reset and check that the fifo is empty.
  XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(1, 1))},
                                   {"push_data", ZeroOfType(u32)},
                                   {"push_valid", Value(UBits(1, 1))},
                                   {"pop_ready", Value(UBits(0, 1))}}));
  XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                   {"push_data", ZeroOfType(u32)},
                                   {"push_valid", Value(UBits(0, 1))},
                                   {"pop_ready", Value(UBits(1, 1))}}));
  EXPECT_THAT(eval->output_ports(),
              AllOf(Contains(Pair("pop_valid", Value(UBits(0, 1)))),
                    Contains(Pair("push_ready", Value(UBits(1, 1))))));
}

void CutThroughLatencyCorrectCommon(
    FifoTest& fft, BlockEvaluator::OutputPortSampleTime time_step) {
  // TODO(rigge): add instantiation support to block jit and remove this guard.
  if (!fft.SupportsFifos()) {
    GTEST_SKIP();
    return;
  }
  auto p = fft.CreatePackage();
  Type* u1 = p->GetBitsType(1);
  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("fifo_wrapper", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fft.fifo_config(), u32));
  XLS_ASSERT_OK(bb.AddClockPort("clk"));

  auto out_port = [&](std::string_view name, BValue val) -> BValue {
    if (time_step == BlockEvaluator::OutputPortSampleTime::kAfterLastClock) {
      return bb.FloppedOutputPort(name, val);
    }
    return bb.OutputPort(name, val);
  };
  out_port("pop_data", bb.InstantiationOutput(fifo_inst, "pop_data"));
  out_port("pop_valid", bb.InstantiationOutput(fifo_inst, "pop_valid"));
  out_port("push_ready", bb.InstantiationOutput(fifo_inst, "push_ready"));

  // Make reset.
  BValue reset = bb.ResetPort(
      "reset", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.InstantiationInput(fifo_inst, "rst", reset);

  // Make push side.
  bb.InstantiationInput(fifo_inst, "push_data", bb.InputPort("push_data", u32));
  bb.InstantiationInput(fifo_inst, "push_valid",
                        bb.InputPort("push_valid", u1));
  bb.InstantiationInput(fifo_inst, "pop_ready", bb.InputPort("pop_ready", u1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BlockContinuation> eval,
                           fft.evaluator().NewContinuation(block, time_step));
  XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                   {"push_data", ZeroOfType(u32)},
                                   {"push_valid", Value(UBits(1, 1))},
                                   {"pop_ready", Value(UBits(1, 1))}}));
  int64_t cycles = -1;
  constexpr int64_t kMaxCycles = 5;
  for (int64_t i = 0; i < kMaxCycles; ++i) {
    if (eval->output_ports().at("pop_valid").IsAllOnes()) {
      EXPECT_EQ(cycles, -1) << "Saw unexpected multiple pops.";
      EXPECT_THAT(eval->output_ports(),
                  Contains(Pair("pop_data", Value(UBits(0, 32)))));
      cycles = i;
    }
    XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                     {"push_data", ZeroOfType(u32)},
                                     {"push_valid", Value(UBits(0, 1))},
                                     {"pop_ready", Value(UBits(1, 1))}}));
  }
  EXPECT_NE(cycles, -1) << "Did not see a pop.";
  EXPECT_EQ(cycles,
            1 - (fft.GetParam().fifo_config.bypass() ? 1 : 0) +
                (fft.GetParam().fifo_config.register_pop_outputs() ? 1 : 0));
}
TEST_P(FifoTest, CutThroughLatencyCorrectPreFallingEdge) {
  CutThroughLatencyCorrectCommon(
      *this, BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock);
}
TEST_P(FifoTest, CutThroughLatencyCorrectPostFallingEdge) {
  CutThroughLatencyCorrectCommon(
      *this, BlockEvaluator::OutputPortSampleTime::kAfterLastClock);
}

void BackpressureLatencyCorrectCommon(
    FifoTest& fft, BlockEvaluator::OutputPortSampleTime time_step) {
  // TODO(rigge): add instantiation support to block jit and remove this guard.
  if (!fft.SupportsFifos()) {
    GTEST_SKIP();
    return;
  }
  auto p = fft.CreatePackage();
  Type* u1 = p->GetBitsType(1);
  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("fifo_wrapper", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fft.fifo_config(), u32));
  XLS_ASSERT_OK(bb.AddClockPort("clk"));

  auto out_port = [&](std::string_view name, BValue val) -> BValue {
    if (time_step == BlockEvaluator::OutputPortSampleTime::kAfterLastClock) {
      return bb.FloppedOutputPort(name, val);
    }
    return bb.OutputPort(name, val);
  };
  out_port("pop_data", bb.InstantiationOutput(fifo_inst, "pop_data"));
  out_port("pop_valid", bb.InstantiationOutput(fifo_inst, "pop_valid"));
  out_port("push_ready", bb.InstantiationOutput(fifo_inst, "push_ready"));

  // Make reset.
  BValue reset = bb.ResetPort(
      "reset", ResetBehavior{.asynchronous = false, .active_low = false});
  bb.InstantiationInput(fifo_inst, "rst", reset);

  // Make push side.
  bb.InstantiationInput(fifo_inst, "push_data", bb.InputPort("push_data", u32));
  bb.InstantiationInput(fifo_inst, "push_valid",
                        bb.InputPort("push_valid", u1));
  bb.InstantiationInput(fifo_inst, "pop_ready", bb.InputPort("pop_ready", u1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BlockContinuation> eval,
                           fft.evaluator().NewContinuation(block, time_step));
  // Reset first.
  XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(1, 1))},
                                   {"push_data", ZeroOfType(u32)},
                                   {"push_valid", Value(UBits(0, 1))},
                                   {"pop_ready", Value(UBits(0, 1))}}));
  // Push until the fifo is full.
  int64_t attempts = 0;
  int64_t pushed = 0;
  while (pushed < fft.fifo_config().depth()) {
    if (attempts++ > 10000) {
      FAIL() << "Failed to reach end after " << attempts << " cycles";
    }
    XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                     {"push_data", ZeroOfType(u32)},
                                     {"push_valid", Value(UBits(1, 1))},
                                     {"pop_ready", Value(UBits(0, 1))}}));
    if (eval->output_ports().at("push_ready").IsAllOnes()) {
      ++pushed;
    }
  }
  XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                   {"push_data", ZeroOfType(u32)},
                                   {"push_valid", Value(UBits(0, 1))},
                                   {"pop_ready", Value(UBits(0, 1))}}));
  EXPECT_THAT(eval->output_ports(),
              // Cannot push more.
              Contains(Pair("push_ready", Value(UBits(0, 1)))));
  // Pop an output.
  XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                   {"push_data", ZeroOfType(u32)},
                                   {"push_valid", Value(UBits(0, 1))},
                                   {"pop_ready", Value(UBits(1, 1))}}));
  int64_t push_ready = 1;
  if (fft.GetParam().fifo_config.register_push_outputs()) {
    push_ready = 0;
  }
  EXPECT_THAT(eval->output_ports(),
              Contains(Pair("push_ready", Value(UBits(push_ready, 1)))));
  if (!push_ready) {
    push_ready = 1;
    XLS_ASSERT_OK(eval->RunOneCycle({{"reset", Value(UBits(0, 1))},
                                     {"push_data", ZeroOfType(u32)},
                                     {"push_valid", Value(UBits(0, 1))},
                                     {"pop_ready", Value(UBits(0, 1))}}));
    EXPECT_THAT(eval->output_ports(),
                Contains(Pair("push_ready", Value(UBits(push_ready, 1)))));
  }
}

TEST_P(BlockEvaluatorTest, MultipleRegisterWrites) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto r1, bb.block()->AddRegister("r1", u32, Value(UBits(1234, 32))));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue reset = bb.ResetPort(
      "reset", ResetBehavior{.asynchronous = false, .active_low = false});

  BValue x = bb.InputPort("x", p->GetBitsType(32));
  BValue x_en = bb.InputPort("x_en", p->GetBitsType(1));
  BValue y = bb.InputPort("y", p->GetBitsType(32));
  BValue y_en = bb.InputPort("y_en", p->GetBitsType(1));
  bb.RegisterRead(r1);
  bb.RegisterWrite(r1, x, /*load_enable=*/x_en, reset);
  bb.RegisterWrite(r1, y, /*load_enable=*/y_en, reset);
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               blk, OutputPortSampleTime::kAtLastPosEdgeClock));

  XLS_ASSERT_OK(cont->SetRegisters({{"r1", Value(UBits(12, 32))}}));

  // No inputs enables. Should preserve previous value.
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(42, 32))},
                                   {"x_en", Value(UBits(0, 1))},
                                   {"y", Value(UBits(100, 32))},
                                   {"y_en", Value(UBits(0, 1))},
                                   {"reset", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("r1", Value(UBits(12, 32)))));

  // Input x is enabled.
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(42, 32))},
                                   {"x_en", Value(UBits(1, 1))},
                                   {"y", Value(UBits(100, 32))},
                                   {"y_en", Value(UBits(0, 1))},
                                   {"reset", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("r1", Value(UBits(42, 32)))));

  // Input y is enabled.
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(42, 32))},
                                   {"x_en", Value(UBits(0, 1))},
                                   {"y", Value(UBits(100, 32))},
                                   {"y_en", Value(UBits(1, 1))},
                                   {"reset", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("r1", Value(UBits(100, 32)))));

  // Both inputs enabled.
  EXPECT_THAT(
      cont->RunOneCycle({{"x", Value(UBits(42, 32))},
                         {"x_en", Value(UBits(1, 1))},
                         {"y", Value(UBits(100, 32))},
                         {"y_en", Value(UBits(1, 1))},
                         {"reset", Value(UBits(0, 1))}}),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Multiple writes of register `r1` activated")));

  // Reset with both inputs enabled should be fine.
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(42, 32))},
                                   {"x_en", Value(UBits(1, 1))},
                                   {"y", Value(UBits(100, 32))},
                                   {"y_en", Value(UBits(1, 1))},
                                   {"reset", Value(UBits(1, 1))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("r1", Value(UBits(1234, 32)))));
}

TEST_P(BlockEvaluatorTest, HundredRegisterWrites) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto r1, bb.block()->AddRegister("r1", u32));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));

  BValue x = bb.InputPort("x", p->GetBitsType(32));
  bb.RegisterRead(r1);
  for (int64_t i = 0; i < 100; ++i) {
    bb.RegisterWrite(r1, bb.Add(x, x),
                     /*load_enable=*/bb.Eq(x, bb.Literal(UBits(i, 32))));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               blk, OutputPortSampleTime::kAtLastPosEdgeClock));

  XLS_ASSERT_OK(cont->SetRegisters({{"r1", Value(UBits(0, 32))}}));

  // The block has 1000 register writes. Each load enable is x == C for C in
  // [0, 1000).
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(42, 32))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("r1", Value(UBits(2 * 42, 32)))));
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(55, 32))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("r1", Value(UBits(2 * 55, 32)))));
  // No load-enable should fire so register values are unchanged.
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(2000, 32))}}));
  EXPECT_THAT(cont->registers(),
              UnorderedElementsAre(Pair("r1", Value(UBits(2 * 55, 32)))));
}

TEST_P(BlockEvaluatorTest, MultipleRegistersWithMultipleWrites) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto r1, bb.block()->AddRegister("r1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(auto r2, bb.block()->AddRegister("r2", u32));
  XLS_ASSERT_OK_AND_ASSIGN(auto r3, bb.block()->AddRegister("r3", u32));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));

  BValue en1 = bb.InputPort("en1", p->GetBitsType(1));
  BValue en2 = bb.InputPort("en2", p->GetBitsType(1));
  BValue en3 = bb.InputPort("en3", p->GetBitsType(1));
  bb.RegisterRead(r1);
  bb.RegisterRead(r2);
  bb.RegisterRead(r3);

  bb.RegisterWrite(r1, bb.Literal(UBits(1, 32)), en1);

  bb.RegisterWrite(r2, bb.Literal(UBits(1, 32)), en1);
  bb.RegisterWrite(r2, bb.Literal(UBits(2, 32)), en2);

  bb.RegisterWrite(r3, bb.Literal(UBits(1, 32)), en1);
  bb.RegisterWrite(r3, bb.Literal(UBits(2, 32)), en2);
  bb.RegisterWrite(r3, bb.Literal(UBits(3, 32)), en3);

  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               blk, OutputPortSampleTime::kAtLastPosEdgeClock));

  XLS_ASSERT_OK(cont->SetRegisters({{"r1", Value(UBits(10, 32))},
                                    {"r2", Value(UBits(20, 32))},
                                    {"r3", Value(UBits(30, 32))}}));

  // Initially don't set any load-enables.
  XLS_ASSERT_OK(cont->RunOneCycle({{"en1", Value(UBits(0, 1))},
                                   {"en2", Value(UBits(0, 1))},
                                   {"en3", Value(UBits(0, 1))}}));

  XLS_ASSERT_OK(cont->SetRegisters({{"r1", Value(UBits(10, 32))},
                                    {"r2", Value(UBits(20, 32))},
                                    {"r3", Value(UBits(30, 32))}}));

  // Set en1.
  XLS_ASSERT_OK(cont->RunOneCycle({{"en1", Value(UBits(1, 1))},
                                   {"en2", Value(UBits(0, 1))},
                                   {"en3", Value(UBits(0, 1))}}));

  XLS_ASSERT_OK(cont->SetRegisters({{"r1", Value(UBits(1, 32))},
                                    {"r2", Value(UBits(1, 32))},
                                    {"r3", Value(UBits(1, 32))}}));

  // Set en2.
  XLS_ASSERT_OK(cont->RunOneCycle({{"en1", Value(UBits(0, 1))},
                                   {"en2", Value(UBits(1, 1))},
                                   {"en3", Value(UBits(0, 1))}}));

  XLS_ASSERT_OK(cont->SetRegisters({{"r1", Value(UBits(1, 32))},
                                    {"r2", Value(UBits(2, 32))},
                                    {"r3", Value(UBits(2, 32))}}));

  // Set en3.
  XLS_ASSERT_OK(cont->RunOneCycle({{"en1", Value(UBits(0, 1))},
                                   {"en2", Value(UBits(0, 1))},
                                   {"en3", Value(UBits(1, 1))}}));

  XLS_ASSERT_OK(cont->SetRegisters({{"r1", Value(UBits(1, 32))},
                                    {"r2", Value(UBits(2, 32))},
                                    {"r3", Value(UBits(3, 32))}}));
}

TEST_P(FifoTest, BackpressureLatencyCorrectPostFallingEdge) {
  BackpressureLatencyCorrectCommon(
      *this, BlockEvaluator::OutputPortSampleTime::kAfterLastClock);
}
TEST_P(FifoTest, BackpressureLatencyCorrectPreFallingEdge) {
  BackpressureLatencyCorrectCommon(
      *this, BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock);
}

TEST_P(BlockEvaluatorTest, DelayLine) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  bb.ResetPort("rst",
               ResetBehavior{.asynchronous = false, .active_low = false});
  BValue x = bb.InputPort("x", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Block::DelayLineInstantiationAndConnections inst,
      block->AddAndConnectDelayLineInstantiation("delay3_inst", 3, x.node()));
  XLS_ASSERT_OK(block->AddOutputPort("out", inst.data_output).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cont, evaluator().NewContinuation(
                     block, OutputPortSampleTime::kAtLastPosEdgeClock));

  // Assert reset for one cycle.
  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(0, 32))}, {"rst", Value(UBits(1, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))));

  // Deassert reset and start running the block.
  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(42, 32))}, {"rst", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))));

  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(43, 32))}, {"rst", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))));

  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(44, 32))}, {"rst", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))));

  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(45, 32))}, {"rst", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(42, 32)))));

  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(46, 32))}, {"rst", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(43, 32)))));
}

TEST_P(BlockEvaluatorTest, DelayLineWithAfterLastClockSampling) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  bb.ResetPort("rst",
               ResetBehavior{.asynchronous = false, .active_low = false});
  BValue x = bb.InputPort("x", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Block::DelayLineInstantiationAndConnections inst,
      block->AddAndConnectDelayLineInstantiation("delay1_inst", 1, x.node()));
  XLS_ASSERT_OK(block->AddOutputPort("out", inst.data_output).status());
  XLS_ASSERT_OK_AND_ASSIGN(auto cont,
                           evaluator().NewContinuation(
                               block, OutputPortSampleTime::kAfterLastClock));

  // Assert reset for one cycle.
  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(0, 32))}, {"rst", Value(UBits(1, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))));

  // Deassert reset and start running the block. Input value should be
  // immediately available with kAfterLastClock sampling.
  XLS_ASSERT_OK(cont->RunOneCycle(
      {{"x", Value(UBits(42, 32))}, {"rst", Value(UBits(0, 1))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(42, 32)))));
}

TEST_P(BlockEvaluatorTest, ZeroLatencyDelayLine) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder bb(TestName(), p.get());
  BValue x = bb.InputPort("x", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Block::DelayLineInstantiationAndConnections inst,
      block->AddAndConnectDelayLineInstantiation("delay0_inst", 0, x.node()));
  XLS_ASSERT_OK(block->AddOutputPort("out", inst.data_output).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cont, evaluator().NewContinuation(
                     block, OutputPortSampleTime::kAtLastPosEdgeClock));

  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(42, 32))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(42, 32)))));
  XLS_ASSERT_OK(cont->RunOneCycle({{"x", Value(UBits(100, 32))}}));
  EXPECT_THAT(cont->output_ports(),
              UnorderedElementsAre(Pair("out", Value(UBits(100, 32)))));
}

}  // namespace
}  // namespace xls
