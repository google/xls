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
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/benchmark_support.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

class NodeDependencyAnalysisTest : public IrTestBase {};

MATCHER_P2(DependentOn, nda, source,
           absl::StrFormat("%spath from %s in analysis", negation ? "no " : "",
                           source.node()->ToString())) {
  auto res = nda->IsDependent(source.node(), arg.node());
  return testing::ExplainMatchResult(true, res, result_listener);
}

MATCHER_P2(DependedOnBy, nda, destination,
           absl::StrFormat("%spath to %s in analysis", negation ? "no " : "",
                           destination.node()->ToString())) {
  auto res = nda->IsDependent(arg.node(), destination.node());
  return testing::ExplainMatchResult(true, res, result_listener);
}

MATCHER_P2(DependentOnNode, nda, source,
           absl::StrFormat("%spath from %s in analysis", negation ? "no " : "",
                           source->ToString())) {
  auto res = nda->IsDependent(source, arg);
  return testing::ExplainMatchResult(true, res, result_listener);
}

MATCHER_P2(DependedOnByNode, nda, destination,
           absl::StrFormat("%spath to %s in analysis", negation ? "no " : "",
                           destination->ToString())) {
  auto res = nda->IsDependent(arg, destination);
  return testing::ExplainMatchResult(true, res, result_listener);
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
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  NodeBackwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  EXPECT_EQ(nda.NodesDependingOn(layer3[0].node()).size(), 1);
  EXPECT_TRUE(
      nda.NodesDependingOn(layer3[0].node()).contains(layer3[0].node()));
  EXPECT_EQ(nda.NodesDependingOn(target1.node()).size(), 2);
  EXPECT_EQ(nda.NodesDependingOn(target2.node()).size(), 2);
  for (auto& bval : layer1) {
    EXPECT_EQ(nda.NodesDependingOn(bval.node()).size(), 3);
  }
  for (auto& bval : layer0) {
    EXPECT_EQ(nda.NodesDependingOn(bval.node()).size(), 4);
  }
  auto depends_on1 = DependedOnBy(&nda, target1);
  auto depends_on2 = DependedOnBy(&nda, target2);
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
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  auto depends_on1 = DependentOn(&nda, target1);
  auto depends_on2 = DependentOn(&nda, target2);
  auto depends_on_both = testing::AllOf(depends_on1, depends_on2);
  auto depends_on_none = testing::Not(testing::AnyOf(depends_on1, depends_on2));
  EXPECT_EQ(nda.NodesDependedOnBy(layer3[0].node()).size(), 15);
  for (auto& bval : layer2) {
    EXPECT_EQ(nda.NodesDependedOnBy(bval.node()).size(), 7);
  }
  for (auto& bval : layer1) {
    EXPECT_EQ(nda.NodesDependedOnBy(bval.node()).size(), 3);
  }
  for (auto& bval : layer0) {
    EXPECT_EQ(nda.NodesDependedOnBy(bval.node()).size(), 1);
  }
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
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  auto all_nodes = std::array{x, a1, b1, a2, b2, finish};
  EXPECT_THAT(all_nodes, testing::Each(DependedOnBy(&nda, finish)));
  auto a_nodes = std::array{a1, a2};
  EXPECT_THAT(a_nodes, testing::AllOf(
                           testing::Each(testing::Not(DependentOn(&nda, b1))),
                           testing::Each(testing::Not(DependentOn(&nda, b2)))));
  auto b_nodes = std::array{b1, b2};
  EXPECT_THAT(b_nodes, testing::AllOf(
                           testing::Each(testing::Not(DependentOn(&nda, a1))),
                           testing::Each(testing::Not(DependentOn(&nda, a2)))));
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
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  NodeBackwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  auto all_nodes = std::array{x, a1, b1, a2, b2, finish};
  EXPECT_THAT(all_nodes, testing::Each(DependedOnBy(&nda, finish)));
  auto a_nodes = std::array{a1, a2};
  EXPECT_THAT(
      a_nodes,
      testing::AllOf(testing::Each(testing::Not(DependedOnBy(&nda, b1))),
                     testing::Each(testing::Not(DependedOnBy(&nda, b2)))));
  auto b_nodes = std::array{b1, b2};
  EXPECT_THAT(
      b_nodes,
      testing::AllOf(testing::Each(testing::Not(DependedOnBy(&nda, a1))),
                     testing::Each(testing::Not(DependedOnBy(&nda, a2)))));
}

TEST_F(NodeDependencyAnalysisTest, InputUsedMultipleTimesForwards) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue multisel = fb.Select(x, {x, x, x, x}, x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  EXPECT_THAT(multisel, DependentOn(&nda, x));
}

TEST_F(NodeDependencyAnalysisTest, InputUsedMultipleTimesBackwards) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue multisel = fb.Select(x, {x, x, x, x}, x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  NodeBackwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  EXPECT_THAT(x, DependedOnBy(&nda, multisel));
}

TEST_F(NodeDependencyAnalysisTest, Invalidation) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue lit1 = fb.Literal(UBits(1, 8));
  BValue add1 = fb.Add(x, lit1);
  BValue add2 = fb.Add(add1, lit1);
  BValue addY = fb.Add(add2, y);
  BValue addZ = fb.Add(addY, z);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    NodeForwardDependencyAnalysis nda;
    XLS_ASSERT_OK(nda.Attach(f).status());
    EXPECT_THAT(add1, DependedOnBy(&nda, addZ))
        << "initial forward dep missing: add1 -> addZ";
    addZ.node()->ReplaceOperand(addY.node(), lit1.node());
    EXPECT_THAT(add1, testing::Not(DependedOnBy(&nda, addZ)))
        << "forward dep incorrectly present: add1 -> addZ";
    addZ.node()->ReplaceOperand(lit1.node(), addY.node());
    EXPECT_THAT(add1, DependedOnBy(&nda, addZ))
        << "new forward dep missing: add1 -> addZ";
  }
  {
    NodeBackwardDependencyAnalysis nda;
    XLS_ASSERT_OK(nda.Attach(f).status());
    EXPECT_THAT(add1, DependedOnBy(&nda, addZ))
        << "initial backward dep missing: add1 -> addZ";
    addZ.node()->ReplaceOperand(addY.node(), lit1.node());
    EXPECT_THAT(add1, testing::Not(DependedOnBy(&nda, addZ)))
        << "backward dep incorrectly present: add1 -> addZ";
    addZ.node()->ReplaceOperand(lit1.node(), addY.node());
    EXPECT_THAT(add1, DependedOnBy(&nda, addZ))
        << "new backward dep missing: add1 -> addZ";

    XLS_ASSERT_OK_AND_ASSIGN(
        Node * addZ2,
        f->MakeNode<BinOp>(addZ.loc(), addY.node(), lit1.node(), Op::kAdd));
    EXPECT_THAT(add1.node(), DependedOnByNode(&nda, addZ2))
        << "backward dep missing: add1 -> addZ2";
    EXPECT_EQ(nda.GetInfo(add1.node())->size(), 5)
        << "backward dep count mismatch for 'add1' post-addZ2-created";
    XLS_ASSERT_OK(f->RemoveNode(addZ2));
    EXPECT_EQ(nda.GetInfo(add1.node())->size(), 4)
        << "backward dep count mismatch for 'add1' post-addZ2-removed";
  }
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
      [](auto f) {
        NodeForwardDependencyAnalysis nda;
        CHECK_OK(nda.Attach(f).status());
        nda.GetInfo(f->return_value());
        return nda;
      },
      state);
}
void BM_NDABinaryTreeBackward(benchmark::State& state) {
  BM_NDABinaryTree(
      [](auto f) {
        NodeBackwardDependencyAnalysis nda;
        CHECK_OK(nda.Attach(f).status());
        for (Node* n : f->nodes()) {
          nda.GetInfo(n);
        }
        return nda;
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
      [](auto f) {
        NodeForwardDependencyAnalysis nda;
        CHECK_OK(nda.Attach(f).status());
        nda.GetInfo(f->return_value());
        return nda;
      },
      state);
}
void BM_NDADenseBackward(benchmark::State& state) {
  BM_NDADense(
      [](auto f) {
        NodeBackwardDependencyAnalysis nda;
        CHECK_OK(nda.Attach(f).status());
        for (Node* n : f->params()) {
          nda.GetInfo(n);
        }
        return nda;
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
      [](auto f) {
        NodeForwardDependencyAnalysis nda;
        CHECK_OK(nda.Attach(f).status());
        nda.GetInfo(f->return_value());
        return nda;
      },
      state);
}
void BM_NDALadderBackward(benchmark::State& state) {
  BM_NDALadder(
      [](auto f) {
        NodeBackwardDependencyAnalysis nda;
        CHECK_OK(nda.Attach(f).status());
        for (Node* n : f->params()) {
          nda.GetInfo(n);
        }
        return nda;
      },
      state);
}

BENCHMARK(BM_NDABinaryTreeForward)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDABinaryTreeBackward)->DenseRange(2, 12, 2);
BENCHMARK(BM_NDALadderForward)->Range(2, 1024);
BENCHMARK(BM_NDALadderBackward)->Range(2, 1024);
BENCHMARK(BM_NDADenseForward)->RangePair(2, 32, 3, 32);
BENCHMARK(BM_NDADenseBackward)->RangePair(2, 32, 3, 32);

}  // namespace
}  // namespace xls
