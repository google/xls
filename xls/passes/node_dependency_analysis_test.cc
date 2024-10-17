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

#include "xls/passes/node_dependency_analysis.h"

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/benchmark_support.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

class NodeDependencyAnalysisTest : public IrTestBase {};

MATCHER_P2(DependentOn, nda, source,
           absl::StrFormat("%spath from %s in analysis", negation ? "no " : "",
                           source.node()->ToString())) {
  auto res = nda.IsDependent(source.node(), arg.node());
  return testing::ExplainMatchResult(absl_testing::IsOkAndHolds(true), res,
                                     result_listener);
}

MATCHER_P2(DependedOnBy, nda, destination,
           absl::StrFormat("%spath to %s in analysis", negation ? "no " : "",
                           destination.node()->ToString())) {
  auto res = nda.IsDependent(arg.node(), destination.node());
  return testing::ExplainMatchResult(absl_testing::IsOkAndHolds(true), res,
                                     result_listener);
}

TEST_F(NodeDependencyAnalysisTest, BackwardsUninterestingIsRemoved) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  std::array<BValue, 8> layer0{fb.Literal(UBits(0, 8)),   // influences target1
                               fb.Literal(UBits(1, 8)),   // influences target1
                               fb.Literal(UBits(2, 8)),   // influences target1
                               fb.Literal(UBits(3, 8)),   // influences target1
                               fb.Literal(UBits(4, 8)),   // influences target2
                               fb.Literal(UBits(5, 8)),   // influences target2
                               fb.Literal(UBits(6, 8)),   // influences target2
                               fb.Literal(UBits(7, 8))};  // influences target2
  std::array<BValue, 4> layer1{
      fb.Add(layer0[0], layer0[1]),   // influences target1
      fb.Add(layer0[2], layer0[3]),   // influences target1
      fb.Add(layer0[4], layer0[5]),   // influences target2
      fb.Add(layer0[6], layer0[7])};  // influences target2
  BValue target1 = fb.Add(layer1[0], layer1[1]);
  BValue target2 = fb.Add(layer1[2], layer1[3]);
  std::array<BValue, 2> layer2{target1,   // influences target1
                               target2};  // influences target2
  // influences none
  std::array<BValue, 1> layer3{fb.Add(layer2[0], layer2[1])};
  // First and last input.
  auto targets = std::array{target1.node(), target2.node()};
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  NodeDependencyAnalysis nda(
      NodeDependencyAnalysis::BackwardDependents(f, targets));
  EXPECT_THAT(nda.GetDependents(layer1[1].node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda.GetDependents(layer3[0].node()),
              testing::Not(absl_testing::IsOk()));
}

TEST_F(NodeDependencyAnalysisTest, BasicBackwards) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  std::array<BValue, 8> layer0{fb.Literal(UBits(0, 8)),   // influences target1
                               fb.Literal(UBits(1, 8)),   // influences target1
                               fb.Literal(UBits(2, 8)),   // influences target1
                               fb.Literal(UBits(3, 8)),   // influences target1
                               fb.Literal(UBits(4, 8)),   // influences target2
                               fb.Literal(UBits(5, 8)),   // influences target2
                               fb.Literal(UBits(6, 8)),   // influences target2
                               fb.Literal(UBits(7, 8))};  // influences target2
  std::array<BValue, 4> layer1{
      fb.Add(layer0[0], layer0[1]),   // influences target1
      fb.Add(layer0[2], layer0[3]),   // influences target1
      fb.Add(layer0[4], layer0[5]),   // influences target2
      fb.Add(layer0[6], layer0[7])};  // influences target2
  BValue target1 = fb.Add(layer1[0], layer1[1]);
  BValue target2 = fb.Add(layer1[2], layer1[3]);
  std::array<BValue, 2> layer2{target1,                    // influences target1
                               target2};                   // influences target2
  std::array<BValue, 1> layer3{fb.Add(target1, target2)};  // influences none
  // First and last input.
  auto targets = std::array{target1.node(), target2.node()};
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  NodeDependencyAnalysis nda(
      NodeDependencyAnalysis::BackwardDependents(f, targets));
  auto depends_on1 = DependentOn(nda, target1);
  auto depends_on2 = DependentOn(nda, target2);
  auto depends_on_none = testing::Not(testing::AnyOf(depends_on1, depends_on2));
  EXPECT_THAT(layer3, testing::ElementsAre(depends_on_none));
  EXPECT_THAT(layer2, testing::ElementsAre(depends_on1, depends_on2));
  EXPECT_THAT(layer1, testing::ElementsAre(depends_on1, depends_on1,
                                           depends_on2, depends_on2));
  EXPECT_THAT(layer0, testing::ElementsAre(
                          depends_on1, depends_on1, depends_on1, depends_on1,
                          depends_on2, depends_on2, depends_on2, depends_on2));
}

TEST_F(NodeDependencyAnalysisTest, BasicForwards) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue target1 = fb.Literal(UBits(0, 8));
  BValue target2 = fb.Literal(UBits(7, 8));
  std::array<BValue, 8> layer0{target1,
                               fb.Literal(UBits(1, 8)),
                               fb.Literal(UBits(2, 8)),
                               fb.Literal(UBits(3, 8)),
                               fb.Literal(UBits(4, 8)),
                               fb.Literal(UBits(5, 8)),
                               fb.Literal(UBits(6, 8)),
                               target2};
  std::array<BValue, 4> layer1{fb.Add(layer0[0], layer0[1]),   // uses target1
                               fb.Add(layer0[2], layer0[3]),   // uses nothing
                               fb.Add(layer0[4], layer0[5]),   // uses nothing
                               fb.Add(layer0[6], layer0[7])};  // uses target2
  std::array<BValue, 2> layer2{fb.Add(layer1[0], layer1[1]),   // uses target1
                               fb.Add(layer1[2], layer1[3])};  // uses target1
  std::array<BValue, 1> layer3{fb.Add(layer2[0], layer2[1])};  // uses both
  // First and last input.
  auto targets = std::array{target1.node(), target2.node()};
  // Should have all except layer1[1] & layer1[2] in cone.
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  NodeDependencyAnalysis nda(
      NodeDependencyAnalysis::ForwardDependents(f, targets));
  auto depends_on1 = DependentOn(nda, target1);
  auto depends_on2 = DependentOn(nda, target2);
  auto depends_on_both = testing::AllOf(depends_on1, depends_on2);
  auto depends_on_none = testing::Not(testing::AnyOf(depends_on1, depends_on2));
  EXPECT_THAT(layer3, testing::ElementsAre(depends_on_both));
  EXPECT_THAT(layer2, testing::ElementsAre(depends_on1, depends_on2));
  EXPECT_THAT(layer1, testing::ElementsAre(depends_on1, depends_on_none,
                                           depends_on_none, depends_on2));
  EXPECT_THAT(layer0, testing::ElementsAre(depends_on1, depends_on_none,
                                           depends_on_none, depends_on_none,
                                           depends_on_none, depends_on_none,
                                           depends_on_none, depends_on2));
}

TEST_F(NodeDependencyAnalysisTest, SeparateForwards) {
  // x -> a1 -> a2 -> finish
  // x -> b1 -> b2 -> finish
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Literal(UBits(0, 8));
  BValue a1 = fb.Not(x);
  BValue a2 = fb.Not(a1);
  BValue b1 = fb.Not(x);
  BValue b2 = fb.Not(b1);
  BValue finish = fb.Add(a2, b2);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  NodeDependencyAnalysis nda(NodeDependencyAnalysis::ForwardDependents(f));
  auto all_nodes = std::array{x, a1, b1, a2, b2, finish};
  EXPECT_THAT(all_nodes, testing::Each(DependedOnBy(nda, finish)));
  auto a_nodes = std::array{a1, a2};
  EXPECT_THAT(a_nodes, testing::AllOf(
                           testing::Each(testing::Not(DependentOn(nda, b1))),
                           testing::Each(testing::Not(DependentOn(nda, b2)))));
  auto b_nodes = std::array{b1, b2};
  EXPECT_THAT(b_nodes, testing::AllOf(
                           testing::Each(testing::Not(DependentOn(nda, a1))),
                           testing::Each(testing::Not(DependentOn(nda, a2)))));
}

TEST_F(NodeDependencyAnalysisTest, SeparateBackwards) {
  // x -> a1 -> a2 -> finish
  // x -> b1 -> b2 -> finish
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Literal(UBits(0, 8));
  BValue a1 = fb.Not(x);
  BValue a2 = fb.Not(a1);
  BValue b1 = fb.Not(x);
  BValue b2 = fb.Not(b1);
  BValue finish = fb.Add(a2, b2);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  NodeDependencyAnalysis nda(NodeDependencyAnalysis::BackwardDependents(f));
  auto all_nodes = std::array{x, a1, b1, a2, b2, finish};
  EXPECT_THAT(all_nodes, testing::Each(DependentOn(nda, finish)));
  auto a_nodes = std::array{a1, a2};
  EXPECT_THAT(a_nodes, testing::AllOf(
                           testing::Each(testing::Not(DependedOnBy(nda, b1))),
                           testing::Each(testing::Not(DependedOnBy(nda, b2)))));
  auto b_nodes = std::array{b1, b2};
  EXPECT_THAT(b_nodes, testing::AllOf(
                           testing::Each(testing::Not(DependedOnBy(nda, a1))),
                           testing::Each(testing::Not(DependedOnBy(nda, a2)))));
}

TEST_F(NodeDependencyAnalysisTest, InputUsedMultipleTimesForwards) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue multisel = fb.Select(x, {x, x, x, x}, x);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  NodeDependencyAnalysis nda(NodeDependencyAnalysis::ForwardDependents(f));
  EXPECT_THAT(multisel, DependentOn(nda, x));
}
TEST_F(NodeDependencyAnalysisTest, InputUsedMultipleTimesBackwards) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue multisel = fb.Select(x, {x, x, x, x}, x);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  NodeDependencyAnalysis nda(NodeDependencyAnalysis::BackwardDependents(f));
  EXPECT_THAT(x, DependentOn(nda, multisel));
}
TEST_F(NodeDependencyAnalysisTest, WorksWithNodeIdGaps) {
  auto p = CreatePackage();
  FunctionBuilder fb_1(TestName() + "_first", p.get());
  FunctionBuilder fb_2(TestName() + "_second", p.get());
  FunctionBuilder fb_3(TestName() + "_third", p.get());
  FunctionBuilder fb_4(TestName() + "_fourth", p.get());
  // Just make each of them a 256 element select
  std::array<BValue, 256> args_1{};
  std::array<BValue, 256> args_2{};
  std::array<BValue, 256> args_3{};
  std::array<BValue, 256> args_4{};
  BValue cond1 = fb_1.Param("cond1", p->GetBitsType(8));
  BValue cond2 = fb_2.Param("cond2", p->GetBitsType(8));
  BValue cond3 = fb_3.Param("cond3", p->GetBitsType(8));
  BValue cond4 = fb_4.Param("cond4", p->GetBitsType(8));
  for (int i = 0; i < 256; ++i) {
    args_1[i] = fb_1.Literal(UBits(i, 8));
    args_2[i] = fb_2.Literal(UBits(i, 8));
    args_3[i] = fb_3.Literal(UBits(i, 8));
    args_4[i] = fb_4.Literal(UBits(i, 8));
  }
  BValue ret1 = fb_1.Select(cond1, args_1);
  BValue ret2 = fb_2.Select(cond2, args_2);
  BValue ret3 = fb_3.Select(cond3, args_3);
  BValue ret4 = fb_4.Select(cond4, args_4);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f1, fb_1.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto* f2, fb_2.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto* f3, fb_3.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto* f4, fb_4.Build());
  NodeDependencyAnalysis nda1(NodeDependencyAnalysis::BackwardDependents(f1));
  NodeDependencyAnalysis nda2(NodeDependencyAnalysis::BackwardDependents(f2));
  NodeDependencyAnalysis nda3(NodeDependencyAnalysis::BackwardDependents(f3));
  NodeDependencyAnalysis nda4(NodeDependencyAnalysis::BackwardDependents(f4));
  EXPECT_THAT(nda1.GetDependents(ret1.node()), absl_testing::IsOk());
  EXPECT_THAT(nda2.GetDependents(ret2.node()), absl_testing::IsOk());
  EXPECT_THAT(nda3.GetDependents(ret3.node()), absl_testing::IsOk());
  EXPECT_THAT(nda4.GetDependents(ret4.node()), absl_testing::IsOk());
  EXPECT_THAT(nda1.GetDependents(ret1.node())->IsDependent(cond1.node()),
              absl_testing::IsOk());
  EXPECT_THAT(nda2.GetDependents(ret2.node())->IsDependent(cond2.node()),
              absl_testing::IsOk());
  EXPECT_THAT(nda3.GetDependents(ret3.node())->IsDependent(cond3.node()),
              absl_testing::IsOk());
  EXPECT_THAT(nda4.GetDependents(ret4.node())->IsDependent(cond4.node()),
              absl_testing::IsOk());
  // Cross function.
  EXPECT_THAT(nda2.GetDependents(ret1.node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda3.GetDependents(ret2.node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda4.GetDependents(ret3.node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda1.GetDependents(ret4.node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda1.GetDependents(ret1.node())->IsDependent(cond2.node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda2.GetDependents(ret2.node())->IsDependent(cond3.node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda3.GetDependents(ret3.node())->IsDependent(cond4.node()),
              testing::Not(absl_testing::IsOk()));
  EXPECT_THAT(nda4.GetDependents(ret4.node())->IsDependent(cond1.node()),
              testing::Not(absl_testing::IsOk()));
}

template <typename Iter>
Node* NodeAt(Iter nodes, int64_t off) {
  return *std::next(nodes.begin(), off);
}

template <typename Analyze>
void BM_NDABinaryTree(Analyze analyze, benchmark::State& state) {
  std::unique_ptr<VerifiedPackage> p =
      std::make_unique<VerifiedPackage>("balanced_tree_pkg");
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* f, benchmark_support::GenerateBalancedTree(
                   p.get(), /*depth=*/state.range(0),
                   /*fan_out=*/2, benchmark_support::strategy::BinaryAdd(),
                   benchmark_support::strategy::DistinctLiteral()));
  for (auto _ : state) {
    auto v = analyze(f);
    benchmark::DoNotOptimize(v);
  }
}
void BM_NDABinaryTreeForward(benchmark::State& state) {
  BM_NDABinaryTree(
      [](auto f) { return NodeDependencyAnalysis::ForwardDependents(f); },
      state);
}
void BM_NDABinaryTreeBackward(benchmark::State& state) {
  BM_NDABinaryTree(
      [](auto f) { return NodeDependencyAnalysis::BackwardDependents(f); },
      state);
}
void BM_NDABinaryTreeForwardReturnOnly(benchmark::State& state) {
  BM_NDABinaryTree(
      [](auto f) {
        return NodeDependencyAnalysis::ForwardDependents(
            f, std::array{f->AsFunctionOrDie()->return_value()});
      },
      state);
}
void BM_NDABinaryTreeBackwardReturnOnly(benchmark::State& state) {
  BM_NDABinaryTree(
      [](auto f) {
        return NodeDependencyAnalysis::BackwardDependents(
            f, std::array{f->AsFunctionOrDie()->return_value()});
      },
      state);
}
void BM_NDABinaryTreeForwardMidOnly(benchmark::State& state) {
  BM_NDABinaryTree(
      [](auto f) {
        return NodeDependencyAnalysis::ForwardDependents(
            f,
            std::array{NodeAt(f->nodes(), f->package()->next_node_id() / 2)});
      },
      state);
}
void BM_NDABinaryTreeBackwardMidOnly(benchmark::State& state) {
  BM_NDABinaryTree(
      [](auto f) {
        return NodeDependencyAnalysis::BackwardDependents(
            f,
            std::array{NodeAt(f->nodes(), f->package()->next_node_id() / 2)});
      },
      state);
}

// Wide and dense
// Fully connected layers of a given width.
template <typename Analyze>
void BM_NDADense(Analyze analyze, benchmark::State& state) {
  std::unique_ptr<VerifiedPackage> p =
      std::make_unique<VerifiedPackage>("dense_tree_pkg");
  FunctionBuilder fb("dense_tree", p.get());
  int64_t depth = state.range(0);
  int64_t width = state.range(1);
  benchmark_support::strategy::DistinctLiteral selector;
  benchmark_support::strategy::CaseSelect csts(selector);
  benchmark_support::strategy::DistinctLiteral leaf;
  XLS_ASSERT_OK_AND_ASSIGN(auto* f,
                           benchmark_support::GenerateFullyConnectedLayerGraph(
                               p.get(), depth, width, csts, leaf));
  for (auto _ : state) {
    auto v = analyze(f);
    benchmark::DoNotOptimize(v);
  }
}

void BM_NDADenseForward(benchmark::State& state) {
  BM_NDADense(
      [](auto f) { return NodeDependencyAnalysis::ForwardDependents(f); },
      state);
}
void BM_NDADenseBackward(benchmark::State& state) {
  BM_NDADense(
      [](auto f) { return NodeDependencyAnalysis::BackwardDependents(f); },
      state);
}
void BM_NDADenseForwardReturnOnly(benchmark::State& state) {
  BM_NDADense(
      [](auto f) {
        return NodeDependencyAnalysis::ForwardDependents(
            f, std::array{f->AsFunctionOrDie()->return_value()});
      },
      state);
}
void BM_NDADenseBackwardReturnOnly(benchmark::State& state) {
  BM_NDADense(
      [](auto f) {
        return NodeDependencyAnalysis::BackwardDependents(
            f, std::array{f->AsFunctionOrDie()->return_value()});
      },
      state);
}
void BM_NDADenseForwardMidOnly(benchmark::State& state) {
  BM_NDADense(
      [](auto f) {
        return NodeDependencyAnalysis::ForwardDependents(
            f,
            std::array{NodeAt(f->nodes(), f->package()->next_node_id() / 2)});
      },
      state);
}
void BM_NDADenseBackwardMidOnly(benchmark::State& state) {
  BM_NDADense(
      [](auto f) {
        return NodeDependencyAnalysis::BackwardDependents(
            f,
            std::array{NodeAt(f->nodes(), f->package()->next_node_id() / 2)});
      },
      state);
}
// Just a very deep ladder structure
// ...
// x_{n-2} : x_{n-3} + 1
// x_n := x_{n-1} + 1
// ...
template <typename Analyze>
void BM_NDALadder(Analyze analyze, benchmark::State& state) {
  std::unique_ptr<VerifiedPackage> p =
      std::make_unique<VerifiedPackage>("ladder_tree_pkg");
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* f,
      benchmark_support::GenerateChain(
          p.get(), state.range(0), 2, benchmark_support::strategy::BinaryAdd(),
          benchmark_support::strategy::DistinctLiteral()));
  for (auto _ : state) {
    auto v = analyze(f);
    benchmark::DoNotOptimize(v);
  }
}
void BM_NDALadderForward(benchmark::State& state) {
  BM_NDALadder(
      [](auto f) { return NodeDependencyAnalysis::ForwardDependents(f); },
      state);
}
void BM_NDALadderBackward(benchmark::State& state) {
  BM_NDALadder(
      [](auto f) { return NodeDependencyAnalysis::BackwardDependents(f); },
      state);
}
void BM_NDALadderForwardReturnOnly(benchmark::State& state) {
  BM_NDALadder(
      [](FunctionBase* f) {
        return NodeDependencyAnalysis::ForwardDependents(
            f, std::array{f->AsFunctionOrDie()->return_value()});
      },
      state);
}
void BM_NDALadderBackwardReturnOnly(benchmark::State& state) {
  BM_NDALadder(
      [](auto f) {
        return NodeDependencyAnalysis::BackwardDependents(
            f, std::array{f->AsFunctionOrDie()->return_value()});
      },
      state);
}
void BM_NDALadderForwardMidOnly(benchmark::State& state) {
  BM_NDALadder(
      [](auto f) {
        return NodeDependencyAnalysis::ForwardDependents(
            f,
            std::array{NodeAt(f->nodes(), f->package()->next_node_id() / 2)});
      },
      state);
}
void BM_NDALadderBackwardMidOnly(benchmark::State& state) {
  BM_NDALadder(
      [](FunctionBase* f) {
        return NodeDependencyAnalysis::BackwardDependents(
            f,
            std::array{NodeAt(f->nodes(), f->package()->next_node_id() / 2)});
      },
      state);
}

BENCHMARK(BM_NDABinaryTreeForward)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDABinaryTreeBackward)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDABinaryTreeForwardReturnOnly)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDABinaryTreeBackwardReturnOnly)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDABinaryTreeForwardMidOnly)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDABinaryTreeBackwardMidOnly)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDALadderForward)->Range(2, 1024);
BENCHMARK(BM_NDALadderBackward)->Range(2, 1024);
BENCHMARK(BM_NDALadderForwardReturnOnly)->Range(2, 1024);
BENCHMARK(BM_NDALadderBackwardReturnOnly)->Range(2, 1024);
BENCHMARK(BM_NDALadderForwardMidOnly)->Range(2, 1024);
BENCHMARK(BM_NDALadderBackwardMidOnly)->Range(2, 1024);
BENCHMARK(BM_NDADenseForward)->RangePair(2, 512, 3, 32);
BENCHMARK(BM_NDADenseBackward)->RangePair(2, 512, 3, 32);
BENCHMARK(BM_NDADenseForwardReturnOnly)->RangePair(2, 512, 3, 32);
BENCHMARK(BM_NDADenseBackwardReturnOnly)->RangePair(2, 512, 3, 32);
BENCHMARK(BM_NDADenseForwardMidOnly)->RangePair(2, 512, 3, 32);
BENCHMARK(BM_NDADenseBackwardMidOnly)->RangePair(2, 512, 3, 32);

}  // namespace
}  // namespace xls
