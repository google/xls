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

#include "xls/ir/benchmark_support.h"

#include <array>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/package.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace benchmark_support {

namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::Not;

MATCHER_P(SameNode, bn, "") { return bn.node() == arg.node(); }
MATCHER_P(AsBValue, inner, "") {
  return testing::ExplainMatchResult(inner, arg.node(), result_listener);
}
TEST(NullaryNodeStrategy, Shared) {
  Package p("p");
  FunctionBuilder fb("test", &p);
  strategy::SharedLiteral strategy;
  XLS_ASSERT_OK_AND_ASSIGN(BValue v1, strategy.GenerateNullaryNode(fb));
  EXPECT_THAT(strategy.GenerateNullaryNode(fb), IsOkAndHolds(SameNode(v1)));
  EXPECT_THAT(strategy.GenerateNullaryNode(fb), IsOkAndHolds(SameNode(v1)));
  EXPECT_THAT(strategy.GenerateNullaryNode(fb), IsOkAndHolds(SameNode(v1)));
  EXPECT_THAT(v1.node(), m::Literal());
}
TEST(NullaryNodeStrategy, Distinct) {
  Package p("p");
  FunctionBuilder fb("test", &p);
  strategy::DistinctLiteral strategy;
  XLS_ASSERT_OK_AND_ASSIGN(BValue v1, strategy.GenerateNullaryNode(fb));
  EXPECT_THAT(strategy.GenerateNullaryNode(fb),
              IsOkAndHolds(Not(SameNode(v1))));
  EXPECT_THAT(strategy.GenerateNullaryNode(fb),
              IsOkAndHolds(Not(SameNode(v1))));
  EXPECT_THAT(strategy.GenerateNullaryNode(fb),
              IsOkAndHolds(Not(SameNode(v1))));
  EXPECT_THAT(v1.node(), m::Literal());
}
TEST(LayerGraph, GenerateFullSelect) {
  Package p("p");
  // 1 bit leaf so no else branch.
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, GenerateFullyConnectedLayerGraph(
                        &p, /*depth=*/1, /*width=*/3, strategy::FullSelect(),
                        strategy::DistinctLiteral(UBits(0, 1))));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Select(m::Literal(), {m::Literal(), m::Literal()}),
                {m::Select(m::Literal(), {m::Literal(), m::Literal()}),
                 m::Select(m::Literal(), {m::Literal(), m::Literal()})}));
  RecordProperty("program", f->DumpIr());
}
TEST(LayerGraph, GenerateFullSelect2) {
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, GenerateFullyConnectedLayerGraph(
                        &p, /*depth=*/2, /*width=*/4, strategy::FullSelect(),
                        strategy::DistinctLiteral()));
  auto leaf = m::Literal();
  auto level_one = m::Select(leaf, {leaf, leaf}, leaf);
  auto level_two = m::Select(level_one, {level_one, level_one}, level_one);
  EXPECT_THAT(f->return_value(),
              m::Select(level_two, {level_two, level_two}, level_two));
  RecordProperty("program", f->DumpIr());
}

TEST(LayerGraph, GenerateCaseSelect) {
  Package p("p");
  strategy::DistinctLiteral selector_leaf_strategy(UBits(0, 2));
  strategy::CaseSelect sls(selector_leaf_strategy);
  strategy::DistinctLiteral case_leaf_strategy(UBits(42, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, GenerateFullyConnectedLayerGraph(
                        &p, /*depth=*/2, /*width=*/4, sls, case_leaf_strategy));
  auto leaf = m::Literal(UBits(42, 8));
  auto selector = m::Literal(UBits(0, 2));
  auto level_one = m::Select(selector, {leaf, leaf, leaf, leaf});
  auto level_two =
      m::Select(selector, {level_one, level_one, level_one, level_one});
  EXPECT_THAT(f->return_value(), m::Select(selector, {level_two, level_two,
                                                      level_two, level_two}));
  EXPECT_EQ(f->return_value()->GetType()->GetFlatBitCount(), 8);
  RecordProperty("program", f->DumpIr());
}

TEST(LayerGraph, AddLayer) {
  Package p("p");
  FunctionBuilder fb("function", &p);
  std::array<BValue, 3> args{
      fb.Literal(UBits(20, 8)),
      fb.Literal(UBits(21, 8)),
      fb.Literal(UBits(22, 8)),
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<BValue> res,
      AddFullyConnectedGraphLayer(fb, 4, strategy::FullSelect(),
                                  absl::MakeSpan(args)));
  EXPECT_THAT(
      res, testing::ElementsAre(AsBValue(m::Select(m::Literal(UBits(20, 8)),
                                                   {m::Literal(UBits(22, 8))},
                                                   m::Literal(UBits(21, 8)))),
                                AsBValue(m::Select(m::Literal(UBits(20, 8)),
                                                   {m::Literal(UBits(22, 8))},
                                                   m::Literal(UBits(21, 8)))),
                                AsBValue(m::Select(m::Literal(UBits(20, 8)),
                                                   {m::Literal(UBits(22, 8))},
                                                   m::Literal(UBits(21, 8)))),
                                AsBValue(m::Select(m::Literal(UBits(20, 8)),
                                                   {m::Literal(UBits(22, 8))},
                                                   m::Literal(UBits(21, 8))))));
}

TEST(BalancedTree, BinaryAdd) {
  Package p("p");
  strategy::DistinctLiteral leaf_strategy(UBits(42, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, GenerateBalancedTree(
                        &p, /*depth=*/4, /*fan_out=*/2, strategy::BinaryAdd(),
                        strategy::DistinctLiteral(UBits(42, 8))));
  auto leaf = m::Literal(UBits(42, 8));
  auto level_one = m::Add(leaf, leaf);
  auto level_two = m::Add(level_one, level_one);
  auto level_three = m::Add(level_two, level_two);
  EXPECT_THAT(f->return_value(), m::Add(level_three, level_three));
  RecordProperty("program", f->DumpIr());
}

TEST(BalancedTree, FullSelect) {
  Package p("p");
  strategy::FullSelect fsts;
  strategy::DistinctLiteral leaf_strategy(UBits(42, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, GenerateBalancedTree(&p, /*depth=*/4, /*fan_out=*/5, fsts,
                                         leaf_strategy));
  auto leaf = m::Literal(UBits(42, 8));
  auto level_one = m::Select(leaf, {leaf, leaf, leaf}, leaf);
  auto level_two =
      m::Select(level_one, {level_one, level_one, level_one}, level_one);
  auto level_three =
      m::Select(level_two, {level_two, level_two, level_two}, level_two);
  EXPECT_THAT(f->return_value(),
              m::Select(level_three, {level_three, level_three, level_three},
                        level_three));
  RecordProperty("program", f->DumpIr());
}

TEST(BalancedTree, CaseSelect) {
  Package p("p");
  strategy::DistinctLiteral selector_strategy(UBits(0, 2));
  strategy::CaseSelect csts(selector_strategy);
  strategy::DistinctLiteral leaf_strategy(UBits(42, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, GenerateBalancedTree(&p, /*depth=*/4, /*fan_out=*/4, csts,
                                         leaf_strategy));
  auto selector = m::Literal(UBits(0, 2));
  auto leaf = m::Literal(UBits(42, 8));
  auto level_one = m::Select(selector, {leaf, leaf, leaf, leaf});
  auto level_two =
      m::Select(selector, {level_one, level_one, level_one, level_one});
  auto level_three =
      m::Select(selector, {level_two, level_two, level_two, level_two});
  EXPECT_THAT(f->return_value(),
              m::Select(selector,
                        {level_three, level_three, level_three, level_three}));
  RecordProperty("program", f->DumpIr());
}

TEST(BalancedTree, Reduce) {
  Package p("p");
  FunctionBuilder fb("function", &p);
  std::array<BValue, 9> args{
      fb.Literal(UBits(20, 8)), fb.Literal(UBits(21, 8)),
      fb.Literal(UBits(22, 8)), fb.Literal(UBits(23, 8)),
      fb.Literal(UBits(24, 8)), fb.Literal(UBits(25, 8)),
      fb.Literal(UBits(26, 8)), fb.Literal(UBits(27, 8)),
      fb.Literal(UBits(28, 8)),
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<BValue> res,
      BalancedTreeReduce(fb, 3, strategy::FullSelect(), absl::MakeSpan(args)));
  EXPECT_THAT(
      res, testing::ElementsAre(AsBValue(m::Select(m::Literal(UBits(20, 8)),
                                                   {m::Literal(UBits(22, 8))},
                                                   m::Literal(UBits(21, 8)))),
                                AsBValue(m::Select(m::Literal(UBits(23, 8)),
                                                   {m::Literal(UBits(25, 8))},
                                                   m::Literal(UBits(24, 8)))),
                                AsBValue(m::Select(m::Literal(UBits(26, 8)),
                                                   {m::Literal(UBits(28, 8))},
                                                   m::Literal(UBits(27, 8))))));
}

TEST(Chain, BinaryAdd) {
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      GenerateChain(&p, /*depth=*/4, /*num_children=*/2, strategy::BinaryAdd(),
                    strategy::DistinctLiteral(UBits(42, 8))));
  auto leaf = m::Literal(UBits(42, 8));
  auto level_one = m::Add(leaf, leaf);
  auto level_two = m::Add(level_one, leaf);
  auto level_three = m::Add(level_two, leaf);
  EXPECT_THAT(f->return_value(), m::Add(level_three, leaf));
  RecordProperty("program", f->DumpIr());
}

TEST(Chain, Reduce) {
  Package p("p");
  FunctionBuilder fb("function", &p);
  XLS_ASSERT_OK_AND_ASSIGN(BValue res,
                           ChainReduce(fb, 8, strategy::FullSelect(),
                                       strategy::DistinctLiteral(UBits(42, 8)),
                                       fb.Literal(UBits(22, 8))));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(res.node(),
              m::Select(m::Literal(UBits(22, 8)),
                        {m::Literal(UBits(42, 8)), m::Literal(UBits(42, 8)),
                         m::Literal(UBits(42, 8)), m::Literal(UBits(42, 8)),
                         m::Literal(UBits(42, 8)), m::Literal(UBits(42, 8))},
                        m::Literal(UBits(42, 8))));
}
}  // namespace
}  // namespace benchmark_support
}  // namespace xls
