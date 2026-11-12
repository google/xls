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
#include "xls/passes/resource_sharing_pass.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/bit_provenance_analysis.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/folding_graph.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/resource_sharing_equivalence.h"
#include "xls/passes/resource_sharing_pass_test_base.h"
#include "xls/passes/visibility_analysis.h"
#include "xls/passes/visibility_expr_builder.h"
#include "xls/solvers/ir_equivalence_testutils.h"

namespace xls {

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;
using ::xls::solvers::ScopedVerifyEquivalence;

namespace m = ::xls::op_matchers;

using NodeVisibility = NaryFoldingAction::NodeVisibility;

struct ResourceSharingPassRunner {
  static absl::StatusOr<bool> RunPass(Function* f) {
    PassResults results;
    OptimizationContext context;

    OptimizationPassOptions opts{};
    opts.enable_resource_sharing = true;
    opts.force_resource_sharing = true;
    XLS_ASSIGN_OR_RETURN(opts.area_estimator, GetAreaEstimator("asap7"));
    XLS_ASSIGN_OR_RETURN(opts.delay_estimator, GetDelayEstimator("asap7"));

    return ResourceSharingPass().RunOnFunctionBase(f, opts, &results, context);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Baseline, ResourceSharingPassTestBase,
                               ::testing::Types<ResourceSharingPassRunner>);

class ResourceSharingPassTest : public IrTestBase {
 protected:
  absl::StatusOr<VisibilityBuilder*> CreateVisibilityBuilder(
      FunctionBase* f, int64_t next_node_id) {
    XLS_RETURN_IF_ERROR(bdd_engine_.Populate(f).status());
    XLS_RETURN_IF_ERROR(nda_.Attach(f).status());
    XLS_RETURN_IF_ERROR(bpa_.Populate(f));
    visibility_builder_.emplace(next_node_id, &bdd_engine_, nda_, bpa_);
    return &*visibility_builder_;
  }

  BddQueryEngine bdd_engine_;
  NodeForwardDependencyAnalysis nda_;
  BitProvenanceAnalysis bpa_;
  std::optional<VisibilityBuilder> visibility_builder_;
};

using VisibilityEdges =
    absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>;

TEST_F(ResourceSharingPassTest, CatchesCyclesBeforeTransforming) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  BValue i = fb.Param("i", u8);
  BValue X = fb.UMul(i, i, 8, SourceInfo(), "X");
  BValue Y = fb.UMul(X, i, 8, SourceInfo(), "D");
  BValue cond = fb.Param("cond", p->GetBitsType(2));
  BValue sel =
      fb.Select(cond, {X, Y, fb.Literal(UBits(0, 8)), fb.Literal(UBits(0, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));
  int64_t next_node_id = 10;
  XLS_ASSERT_OK_AND_ASSIGN(VisibilityBuilder * visibility_builder,
                           CreateVisibilityBuilder(f, next_node_id));

  // Exercise producing a cycle by requesting to fold X into Y; yes, they are
  // not mutually exclusive, but the transformation doesn't have a feasible way
  // to check whether analyses were correct; however, it should detect the cycle
  // that would result from this transformation.
  VisibilityEdges edges = {
      OperandVisibilityAnalysis::OperandNode(X.node(), sel.node()),
      OperandVisibilityAnalysis::OperandNode(Y.node(), sel.node())};
  std::vector<NodeVisibility> from_X = {
      {.node = X.node(), .edges = edges, .sinks = {sel.node()}}};
  auto fold_X_into_Y = std::make_unique<NaryFoldingAction>(
      std::move(from_X), Y.node(), edges, /*area_saved=*/0.0);
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  folding_actions_to_perform.push_back(std::move(fold_X_into_Y));

  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_ASSERT_OK(nda_backwards.Attach(f));
  EXPECT_THAT(
      ResourceSharingPass::PerformFoldingActions(
          f, next_node_id, visibility_builder, nda_backwards,
          folding_actions_to_perform),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("would create a cycle")));
}

TEST_F(ResourceSharingPassTest,
       PrecomputedVisibilityAvoidsStaleAllOnesOnSequentialFoldings) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = fb.Param("cond", p->GetBitsType(2));
  BValue A0 = fb.Add(x, y, SourceInfo(), "a0");
  BValue B0 = fb.Add(A0, x, SourceInfo(), "b0");
  BValue A1 = fb.Add(y, x, SourceInfo(), "a1");
  BValue B1 = fb.Add(A1, y, SourceInfo(), "b1");
  BValue A2 = fb.Add(x, x, SourceInfo(), "t");
  BValue sel = fb.Select(cond, {B0, B1, A2, fb.Literal(UBits(0, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));
  int64_t next_node_id = f->node_count() + 1;
  XLS_ASSERT_OK_AND_ASSIGN(VisibilityBuilder * visibility_builder,
                           CreateVisibilityBuilder(f, next_node_id));

  // Action 1: Fold A1 into A0.
  VisibilityEdges edges_A1 = {
      OperandVisibilityAnalysis::OperandNode(B1.node(), sel.node())};
  VisibilityEdges edges_A0 = {
      OperandVisibilityAnalysis::OperandNode(B0.node(), sel.node())};
  std::vector<NodeVisibility> from_1 = {
      {.node = A1.node(), .edges = edges_A1, .sinks = {sel.node()}}};
  auto fold_1 = std::make_unique<NaryFoldingAction>(
      std::move(from_1), A0.node(), edges_A0, /*area_saved=*/1.0);

  // Action 2: Fold A0 into A2.
  // Without precomputing visibility expressions, the visibility of A0, now
  // replaced by a folded selection between A0 and A1, would be corrupted by
  // the previous folding action.
  VisibilityEdges edges_A2 = {
      OperandVisibilityAnalysis::OperandNode(A2.node(), sel.node())};
  std::vector<NodeVisibility> from_2 = {
      {.node = A0.node(), .edges = edges_A0, .sinks = {sel.node()}}};
  auto fold_2 = std::make_unique<NaryFoldingAction>(
      std::move(from_2), A2.node(), edges_A2, /*area_saved=*/1.0);

  ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(1));
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  folding_actions_to_perform.push_back(std::move(fold_1));
  folding_actions_to_perform.push_back(std::move(fold_2));
  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_ASSERT_OK(nda_backwards.Attach(f));
  XLS_EXPECT_OK(ResourceSharingPass::PerformFoldingActions(
      f, next_node_id, visibility_builder, nda_backwards,
      folding_actions_to_perform));
}

TEST_F(ResourceSharingPassTest,
       PerformFoldingActionsSkipsFoldingIntoAlreadyFoldedNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = fb.Param("cond", p->GetBitsType(2));
  BValue A0 = fb.Add(x, y, SourceInfo(), "a0");
  BValue B0 = fb.Add(A0, x, SourceInfo(), "b0");
  BValue A1 = fb.Add(y, x, SourceInfo(), "a1");
  BValue B1 = fb.Add(A1, y, SourceInfo(), "b1");
  BValue A2 = fb.Add(x, x, SourceInfo(), "t");
  BValue sel = fb.Select(cond, {B0, B1, A2, fb.Literal(UBits(0, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));
  int64_t next_node_id = f->node_count() + 1;
  XLS_ASSERT_OK_AND_ASSIGN(VisibilityBuilder * visibility_builder,
                           CreateVisibilityBuilder(f, next_node_id));

  // Action 1: Fold A1 into A0.
  VisibilityEdges edges_A1{
      OperandVisibilityAnalysis::OperandNode(B1.node(), sel.node())};
  VisibilityEdges edges_A0 = {
      OperandVisibilityAnalysis::OperandNode(B0.node(), sel.node())};
  std::vector<NodeVisibility> from_1 = {
      {.node = A1.node(), .edges = edges_A1, .sinks = {sel.node()}}};
  auto fold_1 = std::make_unique<NaryFoldingAction>(
      std::move(from_1), A0.node(), edges_A0, /*area_saved=*/1.0);

  // Action 2: Fold A2 into A1, which action 1 has already folded into A0, so
  // action 2 is skipped.
  VisibilityEdges edges_A2 = {
      OperandVisibilityAnalysis::OperandNode(A2.node(), sel.node())};
  std::vector<NodeVisibility> from_2 = {
      {.node = A2.node(), .edges = edges_A2, .sinks = {sel.node()}}};
  auto fold_2 = std::make_unique<NaryFoldingAction>(
      std::move(from_2), A1.node(), edges_A1, /*area_saved=*/1.0);

  ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(1));
  const NaryFoldingAction* performed_folding = fold_1.get();
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  folding_actions_to_perform.push_back(std::move(fold_1));
  folding_actions_to_perform.push_back(std::move(fold_2));
  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_ASSERT_OK(nda_backwards.Attach(f));
  EXPECT_THAT(ResourceSharingPass::PerformFoldingActions(
                  f, next_node_id, visibility_builder, nda_backwards,
                  folding_actions_to_perform),
              IsOkAndHolds(ElementsAre(performed_folding)));
}

TEST_F(ResourceSharingPassTest, ReplaceOperandsIfChanged_Tests) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue add_node = fb.Add(a, b, SourceInfo(), "to_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add_node));
  (void)f;

  EXPECT_EQ(add_node.node()->operand(0), a.node());
  EXPECT_EQ(add_node.node()->operand(1), b.node());

  // Replace operand 1 with `c`, keeping operand 0 as `a`.
  std::vector<Node*> new_operands = {a.node(), c.node()};
  XLS_ASSERT_OK(ReplaceOperandsIfChanged(add_node.node(), new_operands));

  EXPECT_EQ(add_node.node()->operand(0), a.node());
  EXPECT_EQ(add_node.node()->operand(1), c.node());
}

std::unique_ptr<NaryFoldingAction> MakeFoldingAction(
    Node* target_node, absl::Span<Node* const> source_nodes = {}) {
  std::vector<NodeVisibility> sources;
  absl::c_transform(
      source_nodes, std::back_inserter(sources),
      [](Node* source) { return NodeVisibility{.node = source}; });
  return std::make_unique<NaryFoldingAction>(
      std::move(sources), target_node, VisibilityEdges{}, /*area_saved=*/0.0);
}

template <typename T, typename... Args>
std::vector<std::unique_ptr<T>> make_unique_vector(Args&&... args) {
  std::vector<std::unique_ptr<T>> vec;
  vec.reserve(sizeof...(Args));
  (vec.emplace_back(std::forward<Args>(args)), ...);
  return vec;
}

TEST_F(
    ResourceSharingPassTest,
    SortFoldingActionsInDescendingOrderOfTheirAreaSavings_SpecialNodePriority) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));

  // Construct two Add nodes (special in comparator) and two Sub nodes (normal).
  // ID order: id(add_low) < id(sub_mid) < id(add_high) < id(sub_highest).
  BValue add_low = fb.Add(x, y, SourceInfo(), "add_low");
  BValue sub_mid = fb.Subtract(x, y, SourceInfo(), "sub_mid");
  BValue add_high = fb.Add(x, y, SourceInfo(), "add_high");
  BValue sub_highest = fb.Subtract(x, y, SourceInfo(), "sub_highest");

  // Create helper nodes for intermediate and high delays.
  BValue from_mid = fb.Add(add_low, x, SourceInfo(), "from_mid");
  BValue mul1 = fb.UMul(x, y);
  BValue mul2 = fb.UMul(mul1, y);
  BValue from_high = fb.Add(mul2, x, SourceInfo(), "from_high");

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Tuple(
          {add_low, sub_mid, add_high, sub_highest, from_mid, from_high})));

  // The comparator doesn't care that we specify no source nodes, and this
  // allows us to conveniently express a folding of no delay spread / increase.
  auto make_add_low = [&] { return MakeFoldingAction(add_low.node()); };
  auto make_sub_mid = [&] {
    return MakeFoldingAction(sub_mid.node(), {from_mid.node()});
  };
  auto make_add_high = [&] {
    return MakeFoldingAction(add_high.node(), {from_mid.node()});
  };
  auto make_sub_highest = [&] {
    return MakeFoldingAction(sub_highest.node(), {from_mid.node()});
  };

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  CriticalPathDelayAnalysis delay_analysis(delay_estimator);
  XLS_ASSERT_OK(delay_analysis.Attach(f));

  auto sort_actions = [&]<typename... Folds>(Folds&&... folds) {
    std::vector<std::unique_ptr<NaryFoldingAction>> actions =
        make_unique_vector<NaryFoldingAction>(std::forward<Folds>(folds)...);
    TimingAnalysis ta(actions, delay_analysis);
    SortFoldingActionsInDescendingOrderOfTheirAreaSavings(actions, ta);
    return actions;
  };

  // In the sorting comparator:
  // 1. Special nodes (Add, DynamicBitSlice) are prioritized over normal nodes
  //    when area savings are equal. So both add_low and add_high come before
  //    both sub_mid and sub_highest.
  // 2. For special nodes, delay information is ignored, and we tie-break by
  //    larger node ID, so add_high > add_low despite the add_high fold having
  //    worse delay characteristics.
  // 3. For normal nodes, we tie-break by smaller delay total and smaller ID.
  //    Since both sub_mid and sub_highest have the same delay characteristics,
  //    smaller ID wins, so sub_mid > sub_highest.
  auto expected_targets = ElementsAre(
      Pointee(Property(&NaryFoldingAction::GetTo, Eq(add_high.node()))),
      Pointee(Property(&NaryFoldingAction::GetTo, Eq(add_low.node()))),
      Pointee(Property(&NaryFoldingAction::GetTo, Eq(sub_mid.node()))),
      Pointee(Property(&NaryFoldingAction::GetTo, Eq(sub_highest.node()))));

  // Verify a few input permutations to expose weak comparator hardening bugs.
  EXPECT_THAT(sort_actions(make_add_low(), make_sub_mid(), make_add_high(),
                           make_sub_highest()),
              expected_targets);
  EXPECT_THAT(sort_actions(make_add_low(), make_add_high(), make_sub_mid(),
                           make_sub_highest()),
              expected_targets);
  EXPECT_THAT(sort_actions(make_sub_mid(), make_add_high(), make_add_low(),
                           make_sub_highest()),
              expected_targets);
  EXPECT_THAT(sort_actions(make_sub_highest(), make_sub_mid(), make_add_high(),
                           make_add_low()),
              expected_targets);
}

TEST_F(ResourceSharingPassTest, SortFoldingActionsTieBreaksOnSourceIds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue s0 = fb.Subtract(x, y, SourceInfo(), "s0");
  BValue s1 = fb.Subtract(x, y, SourceInfo(), "s1");
  BValue s2 = fb.Subtract(x, y, SourceInfo(), "s2");
  BValue to = fb.Subtract(x, y, SourceInfo(), "to");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({s0, s1, s2, to})));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  CriticalPathDelayAnalysis delay_analysis(delay_estimator);
  XLS_ASSERT_OK(delay_analysis.Attach(f));

  auto sort_actions = [&]<typename... Folds>(Folds&&... folds) {
    std::vector<std::unique_ptr<NaryFoldingAction>> actions =
        make_unique_vector<NaryFoldingAction>(std::forward<Folds>(folds)...);
    TimingAnalysis ta(actions, delay_analysis);
    SortFoldingActionsInDescendingOrderOfTheirAreaSavings(actions, ta);
    return actions;
  };

  auto froms_are = [](auto... nodes) {
    return ElementsAre(Field(&NodeVisibility::node, Eq(nodes))...);
  };
  auto expected = ElementsAre(
      Pointee(Property(&NaryFoldingAction::GetFrom, froms_are(s0.node()))),
      Pointee(Property(&NaryFoldingAction::GetFrom,
                       froms_are(s0.node(), s2.node()))),
      Pointee(Property(&NaryFoldingAction::GetFrom, froms_are(s1.node()))));
  EXPECT_THAT(sort_actions(MakeFoldingAction(to.node(), {s1.node()}),
                           MakeFoldingAction(to.node(), {s0.node(), s2.node()}),
                           MakeFoldingAction(to.node(), {s0.node()})),
              expected);
  EXPECT_THAT(sort_actions(MakeFoldingAction(to.node(), {s0.node(), s2.node()}),
                           MakeFoldingAction(to.node(), {s1.node()}),
                           MakeFoldingAction(to.node(), {s0.node()})),
              expected);
}

TEST_F(ResourceSharingPassTest,
       SelectFoldingActionsAvoidsGroupingIncompatibleEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  BValue x = fb.Param("x", u8);
  BValue y = fb.Param("y", u8);
  BValue cond = fb.Param("cond", p->GetBitsType(2));
  BValue a = fb.Add(x, y, SourceInfo(), "a");
  BValue b = fb.Add(x, y, SourceInfo(), "b");
  BValue c = fb.Add(x, y, SourceInfo(), "c");
  BValue d = fb.Add(x, y, SourceInfo(), "d");
  BValue sel = fb.Select(cond, {a, b, c, d});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));

  XLS_ASSERT_OK_AND_ASSIGN(ResourceSharingPass::VisibilityAnalyses visibilities,
                           ResourceSharingPass::VisibilityAnalyses::Create(
                               f, ResourceSharingPass::kDefaultConfig));

  // All pairs of a, b, c, and d are mutually exclusive.
  absl::btree_set<ResourceSharingPass::MutuallyExclPair> mutual_exclusivity = {
      {a.node(), b.node()}, {a.node(), c.node()}, {a.node(), d.node()},
      {b.node(), c.node()}, {b.node(), d.node()}, {c.node(), d.node()}};

  // Folding graph contains all edges to d.
  auto make_folding_graph = [&]() {
    auto make_binary_fold = [&](Node* from, Node* to) {
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>> mappings;
      mappings[from] = std::make_unique<IdentityEquivalenceMapping>(from, to);
      return std::make_unique<BinaryFoldingAction>(
          from, to, /*from_edges=*/FoldingAction::VisibilityEdges(),
          /*to_edges=*/FoldingAction::VisibilityEdges(), /*area_saved=*/10.0,
          std::move(mappings));
    };
    std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions;
    foldable_actions.push_back(make_binary_fold(a.node(), d.node()));
    foldable_actions.push_back(make_binary_fold(b.node(), d.node()));
    foldable_actions.push_back(make_binary_fold(c.node(), d.node()));
    return FoldingGraph(f, std::move(foldable_actions));
  };

  OptimizationPassOptions opts{};
  opts.enable_resource_sharing = true;
  opts.force_resource_sharing = true;
  XLS_ASSERT_OK_AND_ASSIGN(opts.area_estimator, GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(opts.delay_estimator, GetDelayEstimator("unit"));
  OptimizationContext context;

  // Equivalence mapper that only supports mapping up to 2 sources to a dest.
  // An artificial way to test whether invalid nary foldings are avoided.
  NodeEquivalenceMapper mapper;
  mapper.Register([](absl::Span<Node* const> sources, Node* dst)
                      -> absl::StatusOr<std::optional<absl::flat_hash_map<
                          Node*, std::unique_ptr<EquivalenceMapping>>>> {
    if (sources.size() > 2) {
      return std::nullopt;
    }
    return ComputeMappingsSourcesToDest<IdentityEquivalenceMapping>(sources,
                                                                    dst);
  });

  // Confirm with the above restrictive eq mapping, only ab -> d is chosen.
  ResourceSharingPass pass;
  FoldingGraph graph1 = make_folding_graph();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<NaryFoldingAction>> actions1,
      pass.SelectFoldingActionsForGraph(f, &graph1, mutual_exclusivity,
                                        visibilities, mapper, opts, context));
  ASSERT_EQ(actions1.size(), 1);
  EXPECT_EQ(actions1[0]->GetTo(), d.node());
  EXPECT_EQ(actions1[0]->GetNumberOfFroms(), 2);
  EXPECT_THAT(actions1[0]->GetFrom(),
              UnorderedElementsAre(Field(&NodeVisibility::node, a.node()),
                                   Field(&NodeVisibility::node, b.node())));

  // Add the identity equivalence mapping so the full nary folding is allowed.
  mapper.Register<IdentityEquivalenceMapping>();

  // Confirm with the above flexible eq mapping, abc -> d is chosen.
  FoldingGraph graph2 = make_folding_graph();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<NaryFoldingAction>> actions2,
      pass.SelectFoldingActionsForGraph(f, &graph2, mutual_exclusivity,
                                        visibilities, mapper, opts, context));
  ASSERT_EQ(actions2.size(), 1);
  EXPECT_EQ(actions2[0]->GetTo(), d.node());
  EXPECT_EQ(actions2[0]->GetNumberOfFroms(), 3);
  EXPECT_THAT(actions2[0]->GetFrom(),
              UnorderedElementsAre(Field(&NodeVisibility::node, a.node()),
                                   Field(&NodeVisibility::node, b.node()),
                                   Field(&NodeVisibility::node, c.node())));
}

class BitWidthMockAreaEstimator : public AreaEstimator {
 public:
  BitWidthMockAreaEstimator() : AreaEstimator("bit_width_mock") {}

  absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const override {
    return static_cast<double>(node->GetType()->GetFlatBitCount());
  }

  absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons() const override {
    return 1.0;
  }
};

TEST_F(ResourceSharingPassTest, SelectSubsetOfFoldsEachInputMuxEstimated) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u64 = p->GetBitsType(64);
  Type* u6 = p->GetBitsType(6);
  BValue x = fb.Param("x", u64);
  BValue sa0 = fb.Param("sa0", u6);
  BValue sa1 = fb.Param("sa1", u6);
  BValue shift0 = fb.Shll(x, sa0, SourceInfo(), "shift0");
  BValue shift1 = fb.Shll(x, sa1, SourceInfo(), "shift1");
  BValue cond = fb.Param("cond", p->GetBitsType(1));
  BValue sel = fb.Select(cond, {shift0, shift1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));

  BitWidthMockAreaEstimator area_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  // Estimate area for selecting operand 0 (shifted value, 64 bits) vs
  // operand 1 (shift amount, 6 bits).
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_select_value,
      EstimateAreaForSelectingASingleInput(shift1.node(), 0, area_estimator));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_select_amount,
      EstimateAreaForSelectingASingleInput(shift1.node(), 1, area_estimator));
  EXPECT_DOUBLE_EQ(area_select_value, 64.0);
  EXPECT_DOUBLE_EQ(area_select_amount, 6.0);

  CriticalPathDelayAnalysis critical_path_delay(delay_estimator);
  XLS_ASSERT_OK(critical_path_delay.Attach(f));
  BddQueryEngine bdd_engine;
  XLS_ASSERT_OK(bdd_engine.Populate(f));
  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK(bpa.Populate(f));
  VisibilityEstimator visibility_estimator(
      f->node_count(), &bdd_engine, nda, bpa, &area_estimator, delay_estimator);

  // The shifted value is the same, so only the 6-bit shift amount needs muxing.
  absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>> mappings;
  mappings[shift0.node()] = std::make_unique<IdentityEquivalenceMapping>(
      shift0.node(), shift1.node());
  BinaryFoldingAction fold(shift0.node(), shift1.node(),
                           /*from_edges=*/FoldingAction::VisibilityEdges(),
                           /*to_edges=*/FoldingAction::VisibilityEdges(),
                           /*area_saved=*/0.0, std::move(mappings));
  std::vector<BinaryFoldingAction*> folds = {&fold};

  XLS_ASSERT_OK_AND_ASSIGN(
      VisibilityEstimator::AreaDelay selector_cost,
      visibility_estimator.GetAreaAndDelayOfVisibilityExpr(
          fold.GetFrom(), fold.GetFromVisibilityEdges(), fold.GetSinks()));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_shift0,
      area_estimator.GetOperationAreaInSquareMicrons(shift0.node()));
  double expected_area_saved =
      area_shift0 - (area_select_amount + selector_cost.area);

  // When min_area is less or equal to expected area saved, the folding is kept.
  XLS_ASSERT_OK_AND_ASSIGN(
      NaryFoldEstimate estimate_accepted,
      SelectSubsetOfFolds(
          folds, area_estimator, critical_path_delay, &visibility_estimator,
          /*min_area=*/expected_area_saved,
          /*max_delay_spread=*/std::numeric_limits<int64_t>::max(),
          /*max_delay_increase=*/std::numeric_limits<uint64_t>::max(),
          ResourceSharingPass::kDefaultConfig));
  EXPECT_EQ(estimate_accepted.folds.size(), 1);
  EXPECT_DOUBLE_EQ(estimate_accepted.area_saved, expected_area_saved);

  // When min_area exceeds the expected area saved, the folding is filtered out.
  XLS_ASSERT_OK_AND_ASSIGN(
      NaryFoldEstimate estimate_rejected,
      SelectSubsetOfFolds(
          folds, area_estimator, critical_path_delay, &visibility_estimator,
          /*min_area=*/expected_area_saved + 1.0,
          /*max_delay_spread=*/std::numeric_limits<int64_t>::max(),
          /*max_delay_increase=*/std::numeric_limits<uint64_t>::max(),
          ResourceSharingPass::kDefaultConfig));
  EXPECT_TRUE(estimate_rejected.folds.empty());
  EXPECT_DOUBLE_EQ(estimate_rejected.area_saved, 0.0);
}

TEST_F(ResourceSharingPassTest,
       IgnoresComplexIrrelevantConditionsForMutualExclusivity) {
  // Create two shifts to share.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue opcode = fb.Param("opcode", p->GetBitsType(6));
  BValue mode_1 = fb.Param("mode_1", p->GetBitsType(4));
  BValue mode_2 = fb.Param("mode_2", p->GetBitsType(4));
  BValue mode_3 = fb.Param("mode_3", p->GetBitsType(4));
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue shll_1 = fb.Shll(a, b);
  BValue shll_2 = fb.Shll(b, a);

  // Gate usage of the shifts by high BDD path count xor operations and only one
  // operation on a shared variable `opcode` to exercise pairwise visibility's
  // strength. Then verify only pairwise visibility proves mutual exclusivity.
  BValue sh1_gated =
      fb.Gate(fb.XorReduce(mode_3), fb.Gate(fb.XorReduce(mode_1), shll_1));
  BValue sh2_gated = fb.Gate(fb.XorReduce(mode_2), shll_2);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(
          fb.Select(fb.XorReduce(opcode), /*cases=*/{sh2_gated, sh1_gated})));

  // Verify PairwiseVisibilityAnalysis enables folding the shifts.
  XLS_ASSERT_OK_AND_ASSIGN(ResourceSharingPass::VisibilityAnalyses visibilities,
                           ResourceSharingPass::VisibilityAnalyses::Create(
                               f, ResourceSharingPass::kDefaultConfig));
  EXPECT_TRUE(
      visibilities.pairwise.IsMutuallyExclusive(shll_1.node(), shll_2.node()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));
  EXPECT_EQ(absl::c_count_if(
                f->nodes(), [](Node* node) { return node->op() == Op::kShll; }),
            1);
}

// Verify that sink nodes from one source's folding action are not applied when
// evaluating visibility for another source. Here, mul_c has a second use in
// aux_out, so prematurely stopping visibility traversal at main_out's select
// would corrupt mul_c's visibility and cause a miscompile.
TEST_F(ResourceSharingPassTest, NaryFoldDoesNotApplyOtherSourcesSinks) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(3));
  BValue a = fb.Param("a", p->GetBitsType(2));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue zero = fb.Literal(UBits(0, 2));
  BValue mul_a = fb.UMul(a, a, /*result_width=*/2);
  BValue mul_b = fb.UMul(b, b, /*result_width=*/2);
  BValue mul_c = fb.UMul(c, c, /*result_width=*/2);
  BValue main_out = fb.PrioritySelect(s, {mul_a, mul_b, mul_c}, zero);
  BValue aux_out = fb.PrioritySelect(s, {zero, zero, zero}, mul_c);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({main_out, aux_out})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfMultiplications(f), 1);
}

// Verify that when an n-ary fold falls back from single-select binary edges to
// general visibility edges, binary sinks are dropped. Otherwise, traversal
// stops at s1 and misses downstream conditions (e.g. the and-gate selector),
// under-constraining visibility and causing a miscompile.
TEST_F(ResourceSharingPassTest,
       NaryFoldDoesNotPreserveBinarySinksWhenFallingBackToGeneralEdges) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(3));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(3));
  BValue mul_a = fb.UMul(a, a, /*result_width=*/2);
  BValue mul_b = fb.UMul(b, b, /*result_width=*/2);
  BValue mul_c = fb.UMul(c, c, /*result_width=*/2);
  BValue nx = fb.Not(x);
  BValue nx_ext = fb.SignExtend(nx, /*new_bit_count=*/2);
  BValue x_ext = fb.SignExtend(x, /*new_bit_count=*/2);
  BValue s1 = fb.PrioritySelect(s, {mul_a}, mul_c);
  BValue r1 = fb.And(s1, x_ext);
  BValue r2 = fb.And(mul_b, nx_ext);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({r1, r2})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfMultiplications(f), 1);
}

// Verify that the reverse action constructed in ComputeFoldableActions uses its
// own source's select node as its sink rather than copying the forward
// action's sink.
TEST_F(ResourceSharingPassTest,
       ReverseBinaryFoldingActionUsesItsOwnSourceSelectAsSink) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue zero = fb.Literal(UBits(0, 4));
  BValue mul_a = fb.UMul(a, a, /*result_width=*/4);
  BValue mul_b = fb.UMul(b, b, /*result_width=*/4);
  BValue out_1 = fb.PrioritySelect(s, {zero, mul_a}, zero);
  BValue out_2 = fb.PrioritySelect(s, {mul_b, zero}, zero);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({out_1, out_2})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfMultiplications(f), 1);
}

TEST_F(ResourceSharingPassTest,
       ComputeFoldingGraphIsDeterministicAcrossIrRoundTrip) {
  std::unique_ptr<VerifiedPackage> package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  Type* u1 = package->GetBitsType(1);
  Type* u32 = package->GetBitsType(32);

  static constexpr std::array<std::string_view, 5> kXNames = {"x0", "x1", "x2",
                                                              "x3", "x4"};
  static constexpr std::array<std::string_view, 5> kYNames = {"y0", "y1", "y2",
                                                              "y3", "y4"};
  std::vector<BValue> x;
  std::vector<BValue> y;
  std::vector<BValue> interleaved_terms;
  x.reserve(kXNames.size());
  y.reserve(kYNames.size());
  interleaved_terms.reserve(kXNames.size());
  for (size_t i = 0; i < kXNames.size(); ++i) {
    x.push_back(fb.Param(kXNames[i], u1));
    y.push_back(fb.Param(kYNames[i], u1));
    interleaved_terms.push_back(fb.And(x.back(), y.back()));
  }
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);

  // Build mul_interleaved_* before mul_split_* so they get lower node IDs,
  // while placing sel_split first in the return tuple so IR dumping places
  // mul_split_* earlier in topological order after round-tripping.
  // With 5 pairs of (x_i, y_i), `cond = Or_i(x_i & y_i)` stays within the
  // OperandVisibilityAnalysis edge path limit under interleaved BDD variable
  // order (x0, y0, ..., x4, y4) and exceeds it under split order
  // (x0..x4, y0..y4) triggered when `split_cond = Xor(And(x), Or(y))` is
  // evaluated first.
  BValue cond = fb.Or(interleaved_terms);
  BValue mul_interleaved_0 = fb.UMul(a, b, SourceInfo(), "mul_interleaved_0");
  BValue mul_interleaved_1 = fb.UMul(b, a, SourceInfo(), "mul_interleaved_1");
  BValue sel_interleaved =
      fb.Select(cond, {mul_interleaved_1, mul_interleaved_0});

  BValue split_cond = fb.Xor(fb.And(x), fb.Or(y));
  BValue mul_split_0 = fb.UMul(a, a, SourceInfo(), "mul_split_0");
  BValue mul_split_1 = fb.UMul(b, b, SourceInfo(), "mul_split_1");
  BValue sel_split = fb.Select(split_cond, {mul_split_1, mul_split_0});

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Tuple({sel_split, sel_interleaved})));

  using EdgeIdPair = std::pair<int64_t, int64_t>;
  auto get_edge_id_pairs =
      [](FunctionBase* fn) -> absl::StatusOr<std::vector<EdgeIdPair>> {
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(
        ResourceSharingPass::FunctionFoldingGraph folding_graph,
        ResourceSharingPass::ComputeFoldingGraph(
            fn, context, [](Node* n) { return n->op() == Op::kUMul; }));
    std::vector<EdgeIdPair> edge_ids;
    for (BinaryFoldingAction* edge : folding_graph.graph->GetEdges()) {
      edge_ids.emplace_back(edge->GetFrom()->id(), edge->GetTo()->id());
    }
    return edge_ids;
  };

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<EdgeIdPair> in_memory_edges,
                           get_edge_id_pairs(f));
  EXPECT_EQ(in_memory_edges.size(), 4);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> round_tripped_pkg,
                           ParsePackage(package->DumpIr()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * round_tripped_f,
                           round_tripped_pkg->GetFunction(TestName()));
  EXPECT_THAT(get_edge_id_pairs(round_tripped_f),
              IsOkAndHolds(in_memory_edges));
}

// Estimates per-source marginal area saved by `SelectSubsetOfFolds` for a
// batch of binary folds `sources -> to` with unconstrained budgets.
absl::StatusOr<std::vector<double>> EstimateBinaryFoldBatchSourceAreaSaved(
    Function* f, absl::Span<Node* const> sources, Node* to,
    AreaEstimator* area_estimator, DelayEstimator* delay_estimator) {
  CriticalPathDelayAnalysis critical_path_delay(delay_estimator);
  XLS_RETURN_IF_ERROR(critical_path_delay.Attach(f).status());
  BddQueryEngine bdd_engine;
  XLS_RETURN_IF_ERROR(bdd_engine.Populate(f).status());
  NodeForwardDependencyAnalysis nda;
  XLS_RETURN_IF_ERROR(nda.Attach(f).status());
  BitProvenanceAnalysis bpa;
  XLS_RETURN_IF_ERROR(bpa.Populate(f));
  VisibilityEstimator visibility_estimator(
      f->node_count(), &bdd_engine, nda, bpa, area_estimator, delay_estimator);

  using NodeToMappings =
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>;
  std::vector<BinaryFoldingAction> owned_folds;
  owned_folds.reserve(sources.size());
  std::vector<BinaryFoldingAction*> fold_ptrs;
  fold_ptrs.reserve(sources.size());
  for (Node* from : sources) {
    XLS_ASSIGN_OR_RETURN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({from}, to));
    XLS_RET_CHECK(mappings.has_value());
    owned_folds.emplace_back(from, to,
                             /*from_edges=*/FoldingAction::VisibilityEdges(),
                             /*to_edges=*/FoldingAction::VisibilityEdges(),
                             /*area_saved=*/0.0, std::move(*mappings));
    fold_ptrs.push_back(&owned_folds.back());
  }
  XLS_ASSIGN_OR_RETURN(
      NaryFoldEstimate estimate,
      SelectSubsetOfFolds(
          fold_ptrs, *area_estimator, critical_path_delay,
          &visibility_estimator,
          /*min_area=*/-1000.0,
          /*max_delay_spread=*/std::numeric_limits<int64_t>::max(),
          /*max_delay_increase=*/std::numeric_limits<uint64_t>::max(),
          ResourceSharingPass::kDefaultConfig));
  XLS_RET_CHECK_EQ(estimate.folds.size(), sources.size());
  return std::move(estimate.source_area_saved);
}

// Estimates the area saved by `SelectSubsetOfFolds` for a single binary fold
// `from -> to` in `f` with unconstrained area and delay budgets.
absl::StatusOr<double> EstimateSingleBinaryFoldAreaSaved(
    Function* f, Node* from, Node* to, AreaEstimator* area_estimator,
    DelayEstimator* delay_estimator) {
  XLS_ASSIGN_OR_RETURN(std::vector<double> source_areas,
                       EstimateBinaryFoldBatchSourceAreaSaved(
                           f, {from}, to, area_estimator, delay_estimator));
  XLS_RET_CHECK_EQ(source_areas.size(), 1);
  return source_areas[0];
}

TEST_F(ResourceSharingPassTest,
       SelectSubsetOfFoldsChargesZeroMuxesWhenSwappedComparatorAligns) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue ugt_ba = fb.UGt(b, a);
  BValue ult_ab = fb.ULt(a, b);
  BValue cond = fb.Param("cond", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Select(cond, {ugt_ba, ult_ab})));
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * area_estimator,
                           GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_ugt,
      area_estimator->GetOperationAreaInSquareMicrons(ugt_ba.node()));

  EXPECT_THAT(
      EstimateSingleBinaryFoldAreaSaved(f, ugt_ba.node(), ult_ab.node(),
                                        area_estimator, delay_estimator),
      IsOkAndHolds(DoubleEq(area_ugt)));
}

TEST_F(ResourceSharingPassTest,
       SelectSubsetOfFoldsChargesTwoMuxesWhenSwappedComparatorMisaligns) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue ugt_ab = fb.UGt(a, b);
  BValue ult_ab = fb.ULt(a, b);
  BValue cond = fb.Param("cond", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Select(cond, {ugt_ab, ult_ab})));
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * area_estimator,
                           GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_ugt,
      area_estimator->GetOperationAreaInSquareMicrons(ugt_ab.node()));
  XLS_ASSERT_OK_AND_ASSIGN(double area_mux,
                           EstimateAreaForSelectingASingleInput(
                               ult_ab.node(), /*op_idx=*/0, *area_estimator));

  EXPECT_THAT(
      EstimateSingleBinaryFoldAreaSaved(f, ugt_ab.node(), ult_ab.node(),
                                        area_estimator, delay_estimator),
      IsOkAndHolds(DoubleEq(area_ugt - 2.0 * area_mux)));
}

TEST_F(ResourceSharingPassTest,
       SelectSubsetOfFoldsChargesTwoMuxesForCrossSignednessMsbSwap) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue sgt_ba = fb.SGt(b, a);
  BValue ult_ab = fb.ULt(a, b);
  BValue cond = fb.Param("cond", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Select(cond, {sgt_ba, ult_ab})));
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * area_estimator,
                           GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_sgt,
      area_estimator->GetOperationAreaInSquareMicrons(sgt_ba.node()));
  XLS_ASSERT_OK_AND_ASSIGN(double area_mux,
                           EstimateAreaForSelectingASingleInput(
                               ult_ab.node(), /*op_idx=*/0, *area_estimator));

  EXPECT_THAT(
      EstimateSingleBinaryFoldAreaSaved(f, sgt_ba.node(), ult_ab.node(),
                                        area_estimator, delay_estimator),
      IsOkAndHolds(DoubleEq(area_sgt - 2.0 * area_mux)));
}

TEST_F(ResourceSharingPassTest,
       SelectSubsetOfFoldsChargesZeroInputMuxesWhenSwappedCompareFoldsIntoSub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue ugt_ba = fb.UGt(b, a);
  BValue sub_ab = fb.Subtract(a, b);
  BValue cond = fb.Param("cond", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Select(
                        cond, {ugt_ba, fb.BitSlice(sub_ab, /*start=*/0, 1)})));
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * area_estimator,
                           GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_ugt,
      area_estimator->GetOperationAreaInSquareMicrons(ugt_ba.node()));
  using NodeToMappings =
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>;
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                           GetNodeEquivalenceMapper().ComputeMappings(
                               {ugt_ba.node()}, sub_ab.node()));
  ASSERT_TRUE(mappings.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      double ugt_to_sub_overhead,
      mappings->at(ugt_ba.node())
          ->EstimateAreaOverhead(*area_estimator, ugt_ba.node()->operands(),
                                 ugt_ba.node()));

  EXPECT_THAT(
      EstimateSingleBinaryFoldAreaSaved(f, ugt_ba.node(), sub_ab.node(),
                                        area_estimator, delay_estimator),
      IsOkAndHolds(DoubleNear(area_ugt - ugt_to_sub_overhead, 1e-6)));
}

// Builds an `NaryFoldingAction` with `NodeVisibility::area_saved` left unset.
absl::StatusOr<std::unique_ptr<NaryFoldingAction>>
MakeNaryFoldWithoutPerSourceArea(absl::Span<Node* const> froms, Node* to,
                                 double area_saved) {
  using NodeToMappings =
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>;
  XLS_ASSIGN_OR_RETURN(std::optional<NodeToMappings> mappings,
                       GetNodeEquivalenceMapper().ComputeMappings(froms, to));
  XLS_RET_CHECK(mappings.has_value());
  std::vector<NodeVisibility> from_visibilities;
  from_visibilities.reserve(froms.size());
  for (Node* from : froms) {
    from_visibilities.push_back(NodeVisibility{.node = from});
  }
  return std::make_unique<NaryFoldingAction>(std::move(from_visibilities), to,
                                             /*to_edges=*/VisibilityEdges{},
                                             area_saved, std::move(*mappings));
}

TEST_F(ResourceSharingPassTest,
       SelectSubsetOfFoldsRecordsPerSourceMarginalAreaSaved) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue ugt_ba = fb.UGt(b, a);
  BValue ugt_bc = fb.UGt(b, c);
  BValue ult_ab = fb.ULt(a, b);
  BValue cond = fb.Param("cond", p->GetBitsType(2));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(
                        fb.Select(cond, {ugt_ba, ugt_bc, ult_ab},
                                  /*default_value=*/fb.Literal(UBits(0, 1)))));
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * area_estimator,
                           GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_ugt,
      area_estimator->GetOperationAreaInSquareMicrons(ugt_ba.node()));
  XLS_ASSERT_OK_AND_ASSIGN(double area_mux,
                           EstimateAreaForSelectingASingleInput(
                               ult_ab.node(), /*op_idx=*/0, *area_estimator));

  EXPECT_THAT(EstimateBinaryFoldBatchSourceAreaSaved(
                  f, {ugt_ba.node(), ugt_bc.node()}, ult_ab.node(),
                  area_estimator, delay_estimator),
              IsOkAndHolds(ElementsAre(DoubleEq(area_ugt),
                                       DoubleEq(area_ugt - area_mux))));
}

TEST_F(ResourceSharingPassTest,
       LegalizeSequenceDeductsEstimatorAreaWhenPerSourceAreaSavedIsUnset) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue claimed_src = fb.Add(a, b);
  BValue surviving_src = fb.Add(a, b);
  BValue first_dst = fb.Add(a, b);
  BValue second_dst = fb.Add(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple(
                        {claimed_src, surviving_src, first_dst, second_dst})));
  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_ASSERT_OK(nda_backwards.Attach(f));
  BitWidthMockAreaEstimator area_estimator;

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NaryFoldingAction> first_fold,
      MakeNaryFoldWithoutPerSourceArea({claimed_src.node()}, first_dst.node(),
                                       /*area_saved=*/50.0));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NaryFoldingAction> second_fold,
      MakeNaryFoldWithoutPerSourceArea(
          {claimed_src.node(), surviving_src.node()}, second_dst.node(),
          /*area_saved=*/100.0));
  std::vector<std::unique_ptr<NaryFoldingAction>> sequence;
  sequence.push_back(std::move(first_fold));
  sequence.push_back(std::move(second_fold));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto legalized_and_modified,
      ResourceSharingPass::LegalizeSequenceOfFolding(
          sequence, /*mutual_exclusivity=*/{}, nda_backwards, &area_estimator,
          ResourceSharingPass::kDefaultConfig));
  EXPECT_TRUE(legalized_and_modified.second);
  ASSERT_EQ(legalized_and_modified.first.size(), 2);
  EXPECT_DOUBLE_EQ(legalized_and_modified.first[1]->area_saved(), 100.0 - 32.0);
}

TEST_F(
    ResourceSharingPassTest,
    LegalizeSequencePrunesPriorDestinationWhenPriorSourceConflictsWithTarget) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue conflicting_prior_src = fb.Add(a, b);
  BValue chained_mid = fb.Add(a, b);
  BValue legal_src = fb.Add(a, b);
  BValue final_dst = fb.Add(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Tuple(
          {conflicting_prior_src, chained_mid, legal_src, final_dst})));
  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_ASSERT_OK(nda_backwards.Attach(f));

  // Every pair is mutually exclusive except `{conflicting_prior_src,
  // final_dst}`, exercising non-transitivity across
  // `conflicting_prior_src -> chained_mid` and `chained_mid -> final_dst`.
  absl::btree_set<ResourceSharingPass::MutuallyExclPair> mutual_exclusivity = {
      {conflicting_prior_src.node(), chained_mid.node()},
      {conflicting_prior_src.node(), legal_src.node()},
      {chained_mid.node(), legal_src.node()},
      {chained_mid.node(), final_dst.node()},
      {legal_src.node(), final_dst.node()},
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NaryFoldingAction> prior_fold,
      MakeNaryFoldWithoutPerSourceArea({conflicting_prior_src.node()},
                                       chained_mid.node(),
                                       /*area_saved=*/10.0));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NaryFoldingAction> downstream_fold,
      MakeNaryFoldWithoutPerSourceArea({chained_mid.node(), legal_src.node()},
                                       final_dst.node(),
                                       /*area_saved=*/20.0));
  std::vector<std::unique_ptr<NaryFoldingAction>> sequence;
  sequence.push_back(std::move(prior_fold));
  sequence.push_back(std::move(downstream_fold));

  XLS_ASSERT_OK_AND_ASSIGN(auto legalized_and_modified,
                           ResourceSharingPass::LegalizeSequenceOfFolding(
                               sequence, mutual_exclusivity, nda_backwards,
                               /*area_estimator=*/std::nullopt,
                               ResourceSharingPass::kDefaultConfig));
  EXPECT_TRUE(legalized_and_modified.second);
  ASSERT_EQ(legalized_and_modified.first.size(), 2);
  EXPECT_THAT(legalized_and_modified.first[1]->GetFrom(),
              ElementsAre(Field(&NodeVisibility::node, legal_src.node())));
}

TEST_F(ResourceSharingPassTest,
       PassFoldsSwappedComparatorWithZeroMuxesAndRejectsUnswapped) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u34 = p->GetBitsType(34);
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue a = fb.Param("a", u34);
  BValue b = fb.Param("b", u34);
  BValue ult_ab = fb.ULt(a, b);
  BValue ugt_ba = fb.UGt(b, a);
  BValue ugt_ab = fb.UGt(a, b);
  BValue out = fb.PrioritySelect(sel, /*cases=*/{ugt_ba, ugt_ab},
                                 /*default_value=*/ult_ab);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(10));

  OptimizationPassOptions opts{};
  opts.enable_resource_sharing = true;
  XLS_ASSERT_OK_AND_ASSIGN(opts.area_estimator, GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(opts.delay_estimator, GetDelayEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_ugt,
      opts.area_estimator->GetOperationAreaInSquareMicrons(ugt_ba.node()));
  XLS_ASSERT_OK_AND_ASSIGN(
      double area_mux, EstimateAreaForSelectingASingleInput(
                           ult_ab.node(), /*op_idx=*/0, *opts.area_estimator));
  PassResults results;
  OptimizationContext context;
  ResourceSharingPass pass(ResourceSharingPass::ProfitabilityGuard::kInDegree,
                           /*min_area_savings=*/area_ugt - area_mux);

  EXPECT_THAT(pass.RunOnFunctionBase(f, opts, &results, context),
              IsOkAndHolds(true));
  auto shared_cmp = AnyOf(m::ULt(m::Param("a"), m::Param("b")),
                          m::UGt(m::Param("b"), m::Param("a")));
  ASSERT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Param("sel"),
                  /*cases=*/{shared_cmp, m::UGt(m::Param("a"), m::Param("b"))},
                  /*default_value=*/shared_cmp));
  const PrioritySelect* ret_sel = f->return_value()->As<PrioritySelect>();
  EXPECT_EQ(ret_sel->get_case(0), ret_sel->default_value());
}

TEST_F(ResourceSharingPassTest,
       PassFoldsSwappedCompareIntoSubWithZeroMuxesAndRejectsUnswapped) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u34 = p->GetBitsType(34);
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue a = fb.Param("a", u34);
  BValue b = fb.Param("b", u34);
  BValue sub_ab = fb.Subtract(a, b);
  BValue ugt_ba = fb.UGt(b, a);
  BValue ugt_ab = fb.UGt(a, b);
  BValue out = fb.PrioritySelect(
      sel, /*cases=*/{fb.ZeroExtend(ugt_ba, 34), fb.ZeroExtend(ugt_ab, 34)},
      /*default_value=*/sub_ab);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(10));

  OptimizationPassOptions opts{};
  opts.enable_resource_sharing = true;
  XLS_ASSERT_OK_AND_ASSIGN(opts.area_estimator, GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(opts.delay_estimator, GetDelayEstimator("asap7"));
  PassResults results;
  OptimizationContext context;
  ResourceSharingPass pass(ResourceSharingPass::ProfitabilityGuard::kInDegree,
                           /*min_area_savings=*/2.0);

  EXPECT_THAT(pass.RunOnFunctionBase(f, opts, &results, context),
              IsOkAndHolds(true));
  ASSERT_THAT(f->return_value(), m::PrioritySelect());
  const Node* shared_sub =
      f->return_value()->As<PrioritySelect>()->default_value();
  auto folded_ugt_ba = m::Select(
      m::Xor(m::BitSlice(m::Param("a"), /*start=*/33, /*width=*/1),
             m::BitSlice(m::Param("b"), /*start=*/33, /*width=*/1)),
      /*cases=*/{m::BitSlice(Eq(shared_sub), /*start=*/33, /*width=*/1),
                 m::BitSlice(m::Param("b"), /*start=*/33, /*width=*/1)});
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Param("sel"),
                  /*cases=*/
                  {m::ZeroExt(folded_ugt_ba),
                   m::ZeroExt(m::UGt(m::Param("a"), m::Param("b")))},
                  /*default_value=*/m::Sub(m::Param("a"), m::Param("b"))));
}

TEST_F(ResourceSharingPassTest,
       EndToEndPrunedMappedDestinationAndRecomputedMappings) {
  OptimizationPassOptions opts{};
  opts.enable_resource_sharing = true;
  XLS_ASSERT_OK_AND_ASSIGN(opts.area_estimator, GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(opts.delay_estimator, GetDelayEstimator("asap7"));

  // 1. Pruning a replaced destination preserves remaining downstream folds:
  // `ult_512 -> sub_a` runs first (sole comparator, zero muxes) and replaces
  // `sub_a` with `sub[512]`. Pruning `sub_a` from `{sub_a, sub_c} -> sub_b`
  // during legalization allows `sub_c -> sub_b` to execute, leaving `{64, 512}`
  // subtractors instead of aborting the batch and leaving `{64, 64, 512}`.
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    Type* u64 = p->GetBitsType(64);
    BValue op = fb.Param("op", p->GetBitsType(2));
    BValue a64 = fb.Param("a64", u64);
    BValue b64 = fb.Param("b64", u64);
    BValue c64 = fb.Param("c64", u64);

    BValue sub_a = fb.Subtract(a64, b64);
    BValue sub_b = fb.Subtract(a64, c64);
    BValue sub_c = fb.Subtract(a64, c64);
    BValue ult_512 = fb.ULt(fb.ZeroExtend(a64, 512), fb.ZeroExtend(b64, 512));

    BValue out =
        fb.Select(op,
                  /*cases=*/{sub_a, sub_b, sub_c, fb.ZeroExtend(ult_512, 64)});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
    ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(10));

    PassResults results;
    OptimizationContext context;
    ResourceSharingPass pass;
    EXPECT_THAT(pass.RunOnFunctionBase(f, opts, &results, context),
                IsOkAndHolds(true));

    std::vector<int64_t> sub_widths;
    for (Node* node : f->nodes()) {
      if (!node->IsDead() && node->op() == Op::kSub) {
        sub_widths.push_back(node->BitCountOrDie());
      }
    }
    absl::c_sort(sub_widths);
    EXPECT_THAT(sub_widths, ElementsAre(64, 512));
  }

  // 2. Pruning a claimed widening source recomputes narrow destination
  // mappings: `shll_wide_amt` (`u62 << u7`) and `shll_narrow_amt`
  // (`u63 << u6`) have incomparable operand widths so neither folds into the
  // other, while `{shll_wide_amt, shll_narrow_amt} -> shra_dst2` (`u64 >>> u6`)
  // widens `shra_dst2`'s shift-amount operand to 7 bits. Once
  // `{shll_wide_amt, shrl_for_dst1} -> shra_dst1` claims `shll_wide_amt`,
  // recomputing mappings for `shll_narrow_amt -> shra_dst2` keeps `shra_dst2`'s
  // shift-amount operand at 6 bits (`{6, 7}`) rather than retaining the stale
  // 7-bit widening (`{7, 7}`).
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue mode = fb.Param("mode", p->GetBitsType(2));
    BValue v62 = fb.Param("v62", p->GetBitsType(62));
    BValue v63 = fb.Param("v63", p->GetBitsType(63));
    BValue amt6 = fb.Param("amt6", p->GetBitsType(6));
    BValue amt7 = fb.Param("amt7", p->GetBitsType(7));

    BValue v64_from_62 = fb.ZeroExtend(v62, 64);
    BValue shll_wide_amt = fb.Shll(v62, amt7);
    BValue shll_narrow_amt = fb.Shll(v63, amt6);
    BValue shra_dst1 = fb.Shra(v64_from_62, amt7);
    BValue shra_dst2 = fb.Shra(v64_from_62, amt6);
    BValue shrl_for_dst1 = fb.Shrl(v62, amt7);

    BValue arm0 =
        fb.Tuple({fb.ZeroExtend(shll_wide_amt, 64), fb.Literal(UBits(0, 64))});
    BValue arm1 = fb.Tuple({shra_dst1, fb.ZeroExtend(shll_narrow_amt, 64)});
    BValue arm2 = fb.Tuple({shra_dst1, shra_dst2});
    BValue arm3 = fb.Tuple({fb.ZeroExtend(shrl_for_dst1, 64), shra_dst2});
    BValue out = fb.Select(mode, /*cases=*/{arm0, arm1, arm2, arm3});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
    ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(10));

    PassResults results;
    OptimizationContext context;
    ResourceSharingPass pass;
    EXPECT_THAT(pass.RunOnFunctionBase(f, opts, &results, context),
                IsOkAndHolds(true));

    std::vector<int64_t> shra_amt_widths;
    for (Node* node : f->nodes()) {
      if (!node->IsDead() && node->op() == Op::kShra) {
        shra_amt_widths.push_back(node->operand(1)->BitCountOrDie());
      }
    }
    absl::c_sort(shra_amt_widths);
    EXPECT_THAT(shra_amt_widths, ElementsAre(6, 7));
  }
}

TEST_F(ResourceSharingPassTest,
       MakeNaryAndLegalizeSequencePruningAccountingAndEdgeCases) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u16 = p->GetBitsType(16);
  Type* u32 = p->GetBitsType(32);
  BValue op = fb.Param("op", p->GetBitsType(3));
  BValue a16 = fb.Param("a16", u16);
  BValue b16 = fb.Param("b16", u16);
  BValue a32 = fb.Param("a32", u32);
  BValue b32 = fb.Param("b32", u32);

  BValue ult_32 = fb.ULt(a32, b32);
  BValue sge_16 = fb.SGe(a16, b16);
  BValue sub_16_leaf = fb.Subtract(a16, b16);
  BValue sub_16_mid = fb.Subtract(a16, b16);
  BValue sub_16_dst1 = fb.Subtract(a16, b16);
  BValue sub_16_dst2 = fb.Subtract(a16, b16);
  BValue sub_32_dst = fb.Subtract(a32, b32);

  BValue c0 = fb.ZeroExtend(ult_32, 32);
  BValue c1 = fb.ZeroExtend(sge_16, 32);
  BValue c2 = fb.ZeroExtend(sub_16_leaf, 32);
  BValue c3 = fb.ZeroExtend(sub_16_mid, 32);
  BValue c4 = fb.ZeroExtend(sub_16_dst1, 32);
  BValue c5 = fb.ZeroExtend(sub_16_dst2, 32);
  BValue c6 = sub_32_dst;

  BValue result = fb.Select(op, {c0, c1, c2, c3, c4, c5, c6},
                            /*default_value=*/fb.Literal(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * area_estimator,
                           GetAreaEstimator("asap7"));
  XLS_ASSERT_OK_AND_ASSIGN(ResourceSharingPass::VisibilityAnalyses visibility,
                           ResourceSharingPass::VisibilityAnalyses::Create(
                               f, ResourceSharingPass::kDefaultConfig));

  absl::btree_set<ResourceSharingPass::MutuallyExclPair> mutual_exclusivity;
  const std::array<Node*, 7> nodes = {ult_32.node(),      sge_16.node(),
                                      sub_16_leaf.node(), sub_16_mid.node(),
                                      sub_16_dst1.node(), sub_16_dst2.node(),
                                      sub_32_dst.node()};
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (size_t j = i + 1; j < nodes.size(); ++j) {
      mutual_exclusivity.emplace(nodes[i], nodes[j]);
    }
  }

  using NodeToMappings =
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>;
  auto edge_of = [&](BValue case_val) {
    return FoldingAction::VisibilityEdges{
        OperandVisibilityAnalysis::OperandNode(case_val.node(), result.node())};
  };

  // 1. `NaryFoldingAction::Clone` preserves `NodeVisibility::area_saved`, and
  // `LegalizeSequenceOfFolding` retains unmapped prior destinations while
  // reporting `modified = true`, subtracting per-source net `area_saved` (not
  // gross op area), and recomputing mappings when a source is pruned.
  {
    absl::flat_hash_map<Node*, Node*> identity_map;
    for (Node* n : f->nodes()) {
      identity_map[n] = n;
    }
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> m_leaf,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {sub_16_leaf.node()}, sub_16_mid.node()));
    auto fold_into_mid = std::make_unique<NaryFoldingAction>(
        std::vector<NaryFoldingAction::NodeVisibility>{{
            .node = sub_16_leaf.node(),
            .edges = edge_of(c2),
            .sinks = {result.node()},
            .area_saved = 10.0,
        }},
        sub_16_mid.node(), edge_of(c3), /*area_saved=*/10.0,
        std::move(*m_leaf));
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NaryFoldingAction> cloned_mid,
        NaryFoldingAction::Clone(*fold_into_mid, identity_map));
    ASSERT_EQ(cloned_mid->GetFrom().size(), 1);
    EXPECT_EQ(cloned_mid->GetFrom()[0].area_saved, 10.0);

    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> m_mid_to_dst1,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {sub_16_mid.node()}, sub_16_dst1.node()));
    auto fold_mid_into_dst1 = std::make_unique<NaryFoldingAction>(
        std::vector<NaryFoldingAction::NodeVisibility>{{
            .node = sub_16_mid.node(),
            .edges = edge_of(c3),
            .sinks = {result.node()},
            .area_saved = 10.0,
        }},
        sub_16_dst1.node(), edge_of(c4), /*area_saved=*/10.0,
        std::move(*m_mid_to_dst1));

    // `ult_32` is claimed by `consume_ult32`, so pruning `ult_32` from
    // `{ult_32 (3.0), sge_16 (5.0)} -> sub_16_dst2` deducts `3.0` (leaving net
    // `5.0`, whereas subtracting `ult_32`'s gross ASAP7 area would go negative)
    // and drops the 32-bit widening mapping on `sub_16_dst2`.
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> m_consume_ult32,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {ult_32.node()}, sub_32_dst.node()));
    auto consume_ult32 = std::make_unique<NaryFoldingAction>(
        std::vector<NaryFoldingAction::NodeVisibility>{{
            .node = ult_32.node(),
            .edges = edge_of(c0),
            .sinks = {result.node()},
            .area_saved = 20.0,
        }},
        sub_32_dst.node(), edge_of(c6), /*area_saved=*/20.0,
        std::move(*m_consume_ult32));

    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> m_pair1,
        GetNodeEquivalenceMapper().ComputeMappings(
            {ult_32.node(), sge_16.node()}, sub_16_dst2.node()));
    ASSERT_TRUE(m_pair1->contains(sub_16_dst2.node()));
    auto keep_sge_after_net_sub = std::make_unique<NaryFoldingAction>(
        std::vector<NaryFoldingAction::NodeVisibility>{
            {.node = ult_32.node(),
             .edges = edge_of(c0),
             .sinks = {result.node()},
             .area_saved = 3.0},
            {.node = sge_16.node(),
             .edges = edge_of(c1),
             .sinks = {result.node()},
             .area_saved = 5.0},
        },
        sub_16_dst2.node(), edge_of(c5), /*area_saved=*/8.0,
        std::move(*m_pair1));

    std::vector<std::unique_ptr<NaryFoldingAction>> seq;
    seq.push_back(std::move(consume_ult32));
    seq.push_back(std::move(cloned_mid));
    seq.push_back(std::move(fold_mid_into_dst1));
    seq.push_back(std::move(keep_sge_after_net_sub));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto legalized_and_modified,
        ResourceSharingPass::LegalizeSequenceOfFolding(
            seq, mutual_exclusivity, *visibility.nda_backwards, area_estimator,
            ResourceSharingPass::kDefaultConfig));
    EXPECT_TRUE(legalized_and_modified.second);
    ASSERT_EQ(legalized_and_modified.first.size(), 4);
    EXPECT_NEAR(legalized_and_modified.first[3]->area_saved(), 5.0, 1e-6);
    EXPECT_FALSE(legalized_and_modified.first[3]->mappings().contains(
        sub_16_dst2.node()));

    // An incompatible mapper cleanly drops the pruned action.
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> m_unset,
        GetNodeEquivalenceMapper().ComputeMappings(
            {ult_32.node(), sge_16.node()}, sub_16_dst2.node()));
    auto unset_area_fold = std::make_unique<NaryFoldingAction>(
        std::vector<NaryFoldingAction::NodeVisibility>{
            {.node = ult_32.node(),
             .edges = edge_of(c0),
             .sinks = {result.node()}},
            {.node = sge_16.node(),
             .edges = edge_of(c1),
             .sinks = {result.node()}},
        },
        sub_16_dst2.node(), edge_of(c5), /*area_saved=*/0.0,
        std::move(*m_unset));
    std::vector<std::unique_ptr<NaryFoldingAction>> unset_seq;
    unset_seq.push_back(std::move(legalized_and_modified.first[0]));
    unset_seq.push_back(std::move(unset_area_fold));
    NodeEquivalenceMapper empty_mapper;
    XLS_ASSERT_OK_AND_ASSIGN(
        auto dropped,
        ResourceSharingPass::LegalizeSequenceOfFolding(
            unset_seq, mutual_exclusivity, *visibility.nda_backwards,
            area_estimator, ResourceSharingPass::kDefaultConfig, empty_mapper));
    EXPECT_TRUE(dropped.second);
    EXPECT_EQ(dropped.first.size(), 1);
  }

  // 2. `MakeNaryFoldingAction` recomputes mappings and deducts dropped
  // `source_area_saved` when visibility extraction drops a widening source.
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> m_ult32,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {ult_32.node()}, sub_16_dst1.node()));
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> m_sge16,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {sge_16.node()}, sub_16_dst1.node()));
    BinaryFoldingAction fold_ult32(ult_32.node(), sub_16_dst1.node(),
                                   /*from_edges=*/{}, edge_of(c4),
                                   /*area_saved=*/0.0, std::move(*m_ult32),
                                   /*sinks=*/{});
    BinaryFoldingAction fold_sge16(
        sge_16.node(), sub_16_dst1.node(), edge_of(c1), edge_of(c4),
        /*area_saved=*/0.0, std::move(*m_sge16), {result.node()});
    std::vector<BinaryFoldingAction*> folds = {&fold_ult32, &fold_sge16};
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> nary_map,
                             ResourceSharingPass::CompatibleNaryMappings(
                                 folds, GetNodeEquivalenceMapper()));
    ASSERT_TRUE(nary_map->contains(sub_16_dst1.node()));

    const ResourceSharingPass::Config zero_edge_cfg = {
        .min_area_savings = ResourceSharingPass::kDefaultMinAreaSavings,
        .max_delay_spread_squared =
            ResourceSharingPass::kDefaultMaxDelaySpreadSquared,
        .max_delay_increase = ResourceSharingPass::kDefaultMaxDelayIncrease,
        .max_delay_increase_per_fold =
            ResourceSharingPass::kDefaultMaxDelayIncreasePerFold,
        .max_edges_to_handle = 0,
        .max_path_count_for_edge_in_general_visibility_analysis =
            ResourceSharingPass::
                kDefaultMaxPathCountForEdgeInGeneralVisibilityAnalysis,
        .max_path_count_for_bdd_engine =
            ResourceSharingPass::kDefaultMaxPathCountForBddEngine,
    };
    const std::array<double, 2> src_areas = {12.0, 7.5};
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NaryFoldingAction> nary_action,
        ResourceSharingPass::MakeNaryFoldingAction(
            folds, /*area_saved=*/19.5, visibility, std::move(*nary_map),
            zero_edge_cfg, src_areas));
    EXPECT_THAT(nary_action->GetFrom(),
                ElementsAre(Field(&NaryFoldingAction::NodeVisibility::node,
                                  sge_16.node())));
    EXPECT_FALSE(nary_action->mappings().contains(sub_16_dst1.node()));
    EXPECT_NEAR(nary_action->area_saved(), 7.5, 1e-6);

    NodeEquivalenceMapper empty_mapper;
    EXPECT_THAT(ResourceSharingPass::MakeNaryFoldingAction(
                    folds, /*area_saved=*/19.5, visibility,
                    /*mappings=*/{}, zero_edge_cfg, src_areas, empty_mapper),
                StatusIs(absl::StatusCode::kInternal));
  }
}

void IrFuzzResourceSharing(FuzzPackageWithArgs fuzz_package_with_args) {
  ResourceSharingPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzResourceSharing)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace

}  // namespace xls
