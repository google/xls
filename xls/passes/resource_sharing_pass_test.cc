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
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
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
using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Key;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::xls::solvers::ScopedVerifyEquivalence;

using NodeVisibility = NaryFoldingAction::NodeVisibility;

struct ResourceSharingPassRunner {
  static absl::StatusOr<OptimizationPassOptions> DefaultOptions() {
    OptimizationPassOptions opts{};
    opts.enable_resource_sharing = true;
    opts.force_resource_sharing = true;
    XLS_ASSIGN_OR_RETURN(opts.area_estimator, GetAreaEstimator("asap7"));
    XLS_ASSIGN_OR_RETURN(opts.delay_estimator, GetDelayEstimator("asap7"));
    return opts;
  }

  static absl::StatusOr<bool> RunPass(Function* f) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(OptimizationPassOptions opts, DefaultOptions());
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
    visibility_builder_.emplace(next_node_id - 1, &bdd_engine_, nda_, bpa_);
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

TEST_F(ResourceSharingPassTest,
       RejectsFoldingWhenFoldedNodeDependsOnVisibilityEdgeNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue zero = fb.Literal(UBits(0, 4));
  BValue mul_a = fb.UMul(a, a, /*result_width=*/4, SourceInfo(), "mul_a");
  BValue sel_a = fb.PrioritySelect(s, {mul_a}, zero);
  // mul_b depends on sel_a (mul_a's visibility edge node), even though mul_a
  // and mul_b are only visible under mutually exclusive values of `s`.
  BValue mul_b = fb.UMul(sel_a, b, /*result_width=*/4, SourceInfo(), "mul_b");
  BValue sel_b = fb.PrioritySelect(s, {zero}, mul_b);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({sel_a, sel_b})));

  XLS_ASSERT_OK_AND_ASSIGN(ResourceSharingPass::VisibilityAnalyses visibilities,
                           ResourceSharingPass::VisibilityAnalyses::Create(
                               f, ResourceSharingPass::kDefaultConfig));
  ASSERT_TRUE(visibilities.single_select.IsMutuallyExclusive(mul_a.node(),
                                                             mul_b.node()));

  // Check both directions to exercise dependency analysis.
  EXPECT_THAT(ResourceSharingPass::GetFoldableActionForMutuallyExclusiveNodes(
                  mul_a.node(), mul_b.node(), visibilities,
                  ResourceSharingPass::kDefaultConfig.max_edges_to_handle),
              IsOkAndHolds(IsNull()));
  EXPECT_THAT(ResourceSharingPass::GetFoldableActionForMutuallyExclusiveNodes(
                  mul_b.node(), mul_a.node(), visibilities,
                  ResourceSharingPass::kDefaultConfig.max_edges_to_handle),
              IsOkAndHolds(IsNull()));
  absl::btree_set<ResourceSharingPass::MutuallyExclPair> mutual_exclusivity = {
      {mul_a.node(), mul_b.node()}};
  EXPECT_THAT(
      ResourceSharingPass::ComputeFoldableActions(
          f, mutual_exclusivity, visibilities, [](Node*) { return true; },
          ResourceSharingPass::kDefaultConfig),
      IsOkAndHolds(IsEmpty()));
}

// Independently cloned actions targeting distinct parts of the same decomposed
// node have distinct equivalence mapping instances, yet remain compatible in
// legalization and realize the decomposition only once in transformation.
TEST_F(ResourceSharingPassTest,
       LegalizeAndPerformClonedDecomposedActionsRealizeOnce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue y = fb.Param("y", p->GetBitsType(2));
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue mul8 = fb.UMul(a, b, /*result_width=*/8);
  BValue dot0 = fb.UMul(x, x, /*result_width=*/4);
  BValue dot1 = fb.UMul(y, y, /*result_width=*/4);
  BValue dots_sum = fb.ZeroExtend(fb.Add(dot0, dot1), /*new_bit_count=*/8);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Select(
                                             sel, /*cases=*/{mul8, dots_sum})));

  OptimizationContext context;
  XLS_ASSERT_OK_AND_ASSIGN(
      ResourceSharingPass::FunctionFoldingGraph fg,
      ResourceSharingPass::ComputeFoldingGraph(
          f, context, [](Node* n) { return n->op() == Op::kUMul; }));
  XLS_ASSERT_OK_AND_ASSIGN(OptimizationPassOptions opts,
                           ResourceSharingPassRunner::DefaultOptions());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<NaryFoldingAction>> actions,
      ResourceSharingPass().SelectFoldingActionsForGraph(
          f, fg.graph.get(), fg.mutual_exclusivity, fg.visibility,
          GetNodeEquivalenceMapper(), opts, context));

  absl::flat_hash_map<Node*, Node*> identity_node_map;
  for (Node* n : f->nodes()) {
    identity_node_map[n] = n;
  }
  std::vector<std::unique_ptr<NaryFoldingAction>> cloned_actions;
  for (const auto& action : actions) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NaryFoldingAction> cloned,
        NaryFoldingAction::Clone(*action, identity_node_map));
    cloned_actions.push_back(std::move(cloned));
  }
  ASSERT_NE(cloned_actions[0]->to_composition()->mapping,
            cloned_actions[1]->to_composition()->mapping);

  XLS_ASSERT_OK_AND_ASSIGN(
      (auto [legalized, modified]),
      ResourceSharingPass::LegalizeSequenceOfFolding(
          cloned_actions, fg.mutual_exclusivity, *fg.visibility.nda_backwards,
          std::nullopt, ResourceSharingPass::kDefaultConfig));
  EXPECT_FALSE(modified);
  ASSERT_THAT(legalized, SizeIs(2));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(
      ResourceSharingPass::PerformFoldingActions(f, fg.visibility, legalized),
      IsOkAndHolds(SizeIs(2)));
  EXPECT_EQ(NumberOfMultiplications(f), 4);
}

// Legalization accepts both `{cmp} -> sub36` (which widens & renames `sub36`)
// and `{sub36} -> umul64:add_part`. Skipping the latter when transforming,
// neither decomposes `umul64` nor leaves dead precomputed visibility nodes.
TEST_F(ResourceSharingPassTest,
       RunPassDoesNotDecomposeTargetOrLeakVisibilityOfSkippedFold) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u36 = p->GetBitsType(36);
  Type* u37 = p->GetBitsType(37);
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue cmp = fb.ULt(fb.Param("a", u37), fb.Param("b", u37));
  BValue sub36 = fb.Subtract(fb.Param("s0", u36), fb.Param("s1", u36));
  BValue umul64 = fb.UMul(fb.Param("m0", p->GetBitsType(32)),
                          fb.Param("m1", p->GetBitsType(32)),
                          /*result_width=*/64);
  BValue out = fb.PrioritySelect(
      sel, {fb.ZeroExtend(cmp, 64), fb.ZeroExtend(sub36, 64)}, umul64);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));

  ASSERT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));

  EXPECT_EQ(NumberOfMultiplications(f), 1);
  EXPECT_THAT(f->nodes(), Each(Property(&Node::IsDead, false)));
}

// ComputeFoldingGraph applies `should_target` to decomposition parts.
TEST_F(ResourceSharingPassTest,
       ComputeFoldingGraphFiltersDecompositionPartsWithShouldTarget) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u2 = p->GetBitsType(2);
  Type* u4 = p->GetBitsType(4);
  BValue sel = fb.Param("sel", u2);
  BValue a = fb.Param("a", u4);
  BValue b = fb.Param("b", u4);
  BValue smul8 = fb.SMul(a, b, /*result_width=*/8);
  BValue umul4 =
      fb.UMul(fb.Param("c", u2), fb.Param("d", u2), /*result_width=*/4);
  BValue ult4 = fb.ULt(a, b);
  BValue out = fb.Select(
      sel, /*cases=*/{smul8, fb.ZeroExtend(umul4, 8), fb.ZeroExtend(ult4, 8),
                      fb.Literal(UBits(0, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));

  auto should_target = [](Node* n) {
    return !(n->op() == Op::kSub ||
             (n->op() == Op::kAdd && n->BitCountOrDie() < 8));
  };
  OptimizationContext context;
  XLS_ASSERT_OK_AND_ASSIGN(
      ResourceSharingPass::FunctionFoldingGraph fg,
      ResourceSharingPass::ComputeFoldingGraph(f, context, should_target));

  // `smul8`'s 8-bit Add parts pass `should_target` (pairing `smul8` with
  // `ult4`), whereas `umul4`'s 4-bit Add parts do not.
  EXPECT_THAT(
      fg.mutual_exclusivity,
      ElementsAre(
          ResourceSharingPass::MutuallyExclPair(smul8.node(), umul4.node()),
          ResourceSharingPass::MutuallyExclPair(smul8.node(), ult4.node())));
  // `smul8`'s 8-bit Sub part is rejected by `should_target`, leaving only the
  // 5 `umul4 -> smul8` edges (1 undecomposed + 4 decomposed smul parts).
  EXPECT_THAT(fg.graph->GetEdges(),
              AllOf(SizeIs(5), Each(Property(&BinaryFoldingAction::GetToOrPart,
                                             Property(&Node::op, Op::kSMul)))));
}

absl::StatusOr<CompositionPart> DecomposeFirstPart(Node* node, Op op) {
  XLS_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<EquivalenceMapping>> decomps,
                       GetNodeEquivalenceMapper().ComputeDecompositions(node));
  for (const std::shared_ptr<EquivalenceMapping>& decomp : decomps) {
    for (Node* part : decomp->dst()->function_base()->nodes()) {
      if (part->op() == op) {
        return CompositionPart{.mapping = decomp, .part = part};
      }
    }
  }
  return absl::NotFoundError("Decomposition part not found");
}

// Skipping conflicting actions (targeting an already decomposed destination or
// decomposing an already folded source) neither decomposes the skipped source
// nor pollutes accumulated visibility for subsequent actions.
TEST_F(
    ResourceSharingPassTest,
    PerformFoldingActionsSkipsConflictingActionsWithoutCorruptingVisibility) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue cond = fb.Param("cond", p->GetBitsType(2));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue u = fb.Param("u", p->GetBitsType(2));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue mul_a = fb.UMul(u, u, /*result_width=*/4);
  BValue mul_b = fb.UMul(x, x, /*result_width=*/8);
  BValue mul_c = fb.UMul(c, d, /*result_width=*/2);
  BValue mul_d = fb.UMul(d, d, /*result_width=*/2);
  BValue case0 = fb.ZeroExtend(fb.Concat({mul_a, mul_d}), 8);
  BValue case2 = fb.ZeroExtend(mul_c, 8);
  BValue sel =
      fb.Select(cond, /*cases=*/{case0, mul_b, case2, fb.Literal(UBits(0, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));
  XLS_ASSERT_OK_AND_ASSIGN(ResourceSharingPass::VisibilityAnalyses visibility,
                           ResourceSharingPass::VisibilityAnalyses::Create(
                               f, ResourceSharingPass::kDefaultConfig));
  XLS_ASSERT_OK_AND_ASSIGN(CompositionPart a_part,
                           DecomposeFirstPart(mul_a.node(), Op::kUMul));
  XLS_ASSERT_OK_AND_ASSIGN(CompositionPart b_part,
                           DecomposeFirstPart(mul_b.node(), Op::kUMul));

  NodeVisibility from_a = {.node = mul_a.node(),
                           .edges = {{case0.node(), sel.node()}},
                           .sinks = {sel.node()}};
  NodeVisibility from_a_part = {.node = mul_a.node(),
                                .edges = from_a.edges,
                                .sinks = {sel.node()},
                                .composition = a_part};
  NodeVisibility from_c = {.node = mul_c.node(),
                           .edges = {{case2.node(), sel.node()}},
                           .sinks = {sel.node()}};
  VisibilityEdges edges_b = {{mul_b.node(), sel.node()}};

  // Action 0 runs: `mul_a -> mul_b:b_part` (decomposes `mul_b`, folds `mul_a`).
  auto action0 = std::make_unique<NaryFoldingAction>(
      std::vector<NodeVisibility>{from_a}, mul_b.node(), edges_b,
      /*area_saved=*/0.0, ResourceSharingPass::NodeToMappings{}, b_part);
  // Action 1 is skipped: `mul_c -> mul_b` (`mul_b` was already decomposed).
  auto action1 = std::make_unique<NaryFoldingAction>(
      std::vector<NodeVisibility>{from_c}, mul_b.node(), edges_b);
  // Action 2 is skipped: `mul_a:a_part -> mul_c` (`mul_a` was already folded).
  auto action2 = std::make_unique<NaryFoldingAction>(
      std::vector<NodeVisibility>{from_a_part}, mul_c.node(), from_c.edges);
  // Action 3 runs: `mul_c -> mul_d` with uncorrupted visibility.
  auto action3 = std::make_unique<NaryFoldingAction>(
      std::vector<NodeVisibility>{from_c}, mul_d.node(), from_a.edges);

  const NaryFoldingAction* expected0 = action0.get();
  const NaryFoldingAction* expected3 = action3.get();
  auto actions = make_unique_vector<NaryFoldingAction>(
      std::move(action0), std::move(action1), std::move(action2),
      std::move(action3));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(
      ResourceSharingPass::PerformFoldingActions(f, visibility, actions),
      IsOkAndHolds(ElementsAre(expected0, expected3)));
  EXPECT_EQ(NumberOfMultiplications(f), 5);
}

// PerformFoldingActions returns an error if asked to decompose a destination
// node that was already mutated in-place by an earlier undecomposed fold.
TEST_F(ResourceSharingPassTest,
       PerformFoldingActionsErrorsOnDecomposingAlreadyMutatedTarget) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue cond = fb.Param("cond", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(2));
  BValue b = fb.Param("b", p->GetBitsType(2));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue mul4_dst = fb.UMul(a, b, /*result_width=*/4);
  BValue mul4_src = fb.UMul(b, a, /*result_width=*/4);
  BValue mul2_src = fb.UMul(c, c, /*result_width=*/2);
  BValue case1 = fb.Add(mul4_src, fb.ZeroExtend(mul2_src, 4));
  BValue sel = fb.Select(cond, {mul4_dst, case1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));
  XLS_ASSERT_OK_AND_ASSIGN(ResourceSharingPass::VisibilityAnalyses visibility,
                           ResourceSharingPass::VisibilityAnalyses::Create(
                               f, ResourceSharingPass::kDefaultConfig));
  XLS_ASSERT_OK_AND_ASSIGN(CompositionPart dst_part,
                           DecomposeFirstPart(mul4_dst.node(), Op::kUMul));

  VisibilityEdges edges_dst = {{mul4_dst.node(), sel.node()}};
  VisibilityEdges edges_src = {{case1.node(), sel.node()}};
  auto action0 = std::make_unique<NaryFoldingAction>(
      std::vector<NodeVisibility>{
          {.node = mul4_src.node(), .edges = edges_src, .sinks = {sel.node()}}},
      mul4_dst.node(), edges_dst);
  auto action1 = std::make_unique<NaryFoldingAction>(
      std::vector<NodeVisibility>{
          {.node = mul2_src.node(), .edges = edges_src, .sinks = {sel.node()}}},
      mul4_dst.node(), edges_dst, /*area_saved=*/0.0,
      ResourceSharingPass::NodeToMappings{}, dst_part);
  auto actions = make_unique_vector<NaryFoldingAction>(std::move(action0),
                                                       std::move(action1));

  EXPECT_THAT(
      ResourceSharingPass::PerformFoldingActions(f, visibility, actions),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("already mutated")));
}

// legalization and transformation support mixed `from`-side and `to`-side
// decompositions without mutating the input NaryFoldingAction.
TEST_F(ResourceSharingPassTest,
       PerformFoldingActionsRealizesFromAndToDecompositionsWithoutMutating) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue a = fb.Param("a", p->GetBitsType(2));
  BValue b = fb.Param("b", p->GetBitsType(2));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue from_mul4 = fb.UMul(a, b, /*result_width=*/4);
  BValue from_smul2 = fb.SMul(c, c, /*result_width=*/2);
  BValue to_mul4 = fb.UMul(b, a, /*result_width=*/4);
  BValue case1 = fb.ZeroExtend(from_smul2, 4);
  BValue out = fb.PrioritySelect(sel, {from_mul4, case1}, to_mul4);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  XLS_ASSERT_OK_AND_ASSIGN(ResourceSharingPass::VisibilityAnalyses visibility,
                           ResourceSharingPass::VisibilityAnalyses::Create(
                               f, ResourceSharingPass::kDefaultConfig));

  XLS_ASSERT_OK_AND_ASSIGN(CompositionPart from_comp,
                           DecomposeFirstPart(from_mul4.node(), Op::kUMul));
  XLS_ASSERT_OK_AND_ASSIGN(CompositionPart to_comp,
                           DecomposeFirstPart(to_mul4.node(), Op::kUMul));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<ResourceSharingPass::NodeToMappings> mappings,
      GetNodeEquivalenceMapper().ComputeMappings(
          {from_comp.part, from_smul2.node()}, to_comp.part));
  ASSERT_THAT(mappings, Optional(UnorderedElementsAre(Key(from_comp.part),
                                                      Key(from_smul2.node()),
                                                      Key(to_comp.part))));

  Node* orig_from_mul = from_mul4.node();
  Node* orig_from_smul = from_smul2.node();
  Node* orig_to_mul = to_mul4.node();
  auto make_actions = [&]() -> std::vector<std::unique_ptr<NaryFoldingAction>> {
    auto action = std::make_unique<NaryFoldingAction>(
        std::vector<NodeVisibility>{{.node = orig_from_mul,
                                     .edges = {{orig_from_mul, out.node()}},
                                     .sinks = {out.node()},
                                     .composition = from_comp},
                                    {.node = orig_from_smul,
                                     .edges = {{case1.node(), out.node()}},
                                     .sinks = {out.node()}}},
        orig_to_mul, VisibilityEdges{{orig_to_mul, out.node()}},
        /*area_saved=*/10.0, std::move(*mappings), to_comp);
    return make_unique_vector<NaryFoldingAction>(std::move(action));
  };

  std::vector<std::unique_ptr<NaryFoldingAction>> actions = make_actions();
  absl::btree_set<ResourceSharingPass::MutuallyExclPair> mutual_exclusivity = {
      {orig_from_mul, orig_from_smul},
      {orig_from_mul, orig_to_mul},
      {orig_from_smul, orig_to_mul},
  };
  EXPECT_THAT(ResourceSharingPass::LegalizeSequenceOfFolding(
                  actions, mutual_exclusivity, *visibility.nda_backwards,
                  std::nullopt, ResourceSharingPass::kDefaultConfig),
              IsOkAndHolds(Pair(SizeIs(1), false)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(
      ResourceSharingPass::PerformFoldingActions(f, visibility, actions),
      IsOkAndHolds(ElementsAre(actions[0].get())));
  // There are 9 muls after both decompositions and 7 after the 2->1 nary fold.
  EXPECT_EQ(NumberOfMultiplications(f), 7);

  // Expect that the nary folds are not mutated.
  std::vector<std::unique_ptr<NaryFoldingAction>> expected_actions =
      make_actions();
  EXPECT_EQ(actions[0]->GetFrom(), expected_actions[0]->GetFrom());
  EXPECT_EQ(actions[0]->GetTo(), expected_actions[0]->GetTo());
}

TEST_F(ResourceSharingPassTest, MergeLargeUMulWithFourSmallerUMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(16));
  BValue b = fb.Param("b", p->GetBitsType(16));
  BValue mul32 = fb.UMul(a, b, /*result_width=*/32);
  Type* u8 = p->GetBitsType(8);
  BValue dot0 =
      fb.UMul(fb.Param("x0", u8), fb.Param("y0", u8), /*result_width=*/16);
  BValue dot1 =
      fb.UMul(fb.Param("x1", u8), fb.Param("y1", u8), /*result_width=*/16);
  BValue dot2 =
      fb.UMul(fb.Param("x2", u8), fb.Param("y2", u8), /*result_width=*/16);
  BValue dot3 =
      fb.UMul(fb.Param("x3", u8), fb.Param("y3", u8), /*result_width=*/16);
  BValue dots_sum = fb.ZeroExtend(
      fb.Add(fb.Add(dot0, dot1), fb.Add(dot2, dot3)), /*new_bit_count=*/32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Select(sel, /*cases=*/{mul32, dots_sum})));

  EXPECT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));

  EXPECT_EQ(NumberOfMultiplications(f), 4);
  EXPECT_EQ(absl::c_count_if(f->nodes(),
                             [](Node* n) {
                               return n->op() == Op::kUMul &&
                                      n->BitCountOrDie() == 16;
                             }),
            4);

  // Verify 16x16 multiplication path (sel = 0): 0x1234 * 0x5678 = 0x06260060.
  InterpretAndCheck(f, {0, 0x1234, 0x5678, 2, 3, 4, 5, 6, 7, 8, 9},
                    {1, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8},
                    /*input_signs=*/std::vector<bool>(11, false), 0x06260060,
                    32);
  // Verify 4x 8x8 dot-product path (sel = 1): 2*3 + 4*5 + 6*7 + 8*9 = 140.
  InterpretAndCheck(f, {1, 0x1234, 0x5678, 2, 3, 4, 5, 6, 7, 8, 9},
                    {1, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8},
                    /*input_signs=*/std::vector<bool>(11, false), 140, 32);
}

TEST_F(ResourceSharingPassTest, MergeSmallUMulWithFourSmallerUMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue mul8 = fb.UMul(a, b, /*result_width=*/8);
  Type* u2 = p->GetBitsType(2);
  BValue dot0 =
      fb.UMul(fb.Param("x0", u2), fb.Param("y0", u2), /*result_width=*/4);
  BValue dot1 =
      fb.UMul(fb.Param("x1", u2), fb.Param("y1", u2), /*result_width=*/4);
  BValue dot2 =
      fb.UMul(fb.Param("x2", u2), fb.Param("y2", u2), /*result_width=*/4);
  BValue dot3 =
      fb.UMul(fb.Param("x3", u2), fb.Param("y3", u2), /*result_width=*/4);
  BValue dots_sum = fb.ZeroExtend(
      fb.Add(fb.Add(dot0, dot1), fb.Add(dot2, dot3)), /*new_bit_count=*/8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Select(
                                             sel, /*cases=*/{mul8, dots_sum})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));

  EXPECT_EQ(NumberOfMultiplications(f), 4);
  EXPECT_EQ(absl::c_count_if(f->nodes(),
                             [](Node* n) {
                               return n->op() == Op::kUMul &&
                                      n->BitCountOrDie() == 4;
                             }),
            4);
}

TEST_F(ResourceSharingPassTest, MergeSMulWithSmallerUMulSMulAndAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue p0 = fb.Param("p0", p->GetBitsType(32));
  BValue p1 = fb.Param("p1", p->GetBitsType(32));
  BValue p2 = fb.Param("p2", p->GetBitsType(32));
  BValue p3 = fb.Param("p3", p->GetBitsType(32));
  BValue smul32 = fb.SMul(fb.BitSlice(p0, 0, 16), fb.BitSlice(p1, 0, 16),
                          /*result_width=*/32);
  BValue umul16 = fb.UMul(fb.BitSlice(p2, 0, 8), fb.BitSlice(p3, 0, 8),
                          /*result_width=*/16);
  BValue smul16 = fb.SMul(fb.BitSlice(p1, 0, 8), fb.BitSlice(p3, 0, 8),
                          /*result_width=*/16);
  BValue add0 = fb.Add(p0, p1);
  BValue add1 = fb.Add(p0, p2);
  BValue add2 = fb.Add(p1, p2);
  BValue add3 = fb.Add(p0, p3);
  BValue case1 = fb.Xor({fb.ZeroExtend(umul16, 32), fb.SignExtend(smul16, 32),
                         add0, add1, add2, add3});

  // case2 adds compete to fold both with case1 adds and smul32 decomposed adds.
  BValue add4 = fb.Add(p1, p3);
  BValue add5 = fb.Add(p2, p3);
  BValue case2 = fb.Xor(add4, add5);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(
                        fb.Select(sel, /*cases=*/{smul32, case1, case2},
                                  /*default_value=*/fb.Literal(UBits(0, 32)))));

  EXPECT_THAT(ResourceSharingPassRunner::RunPass(f), IsOkAndHolds(true));

  // Decomposed smul32 has 4 quarter smuls (2 shared with umul16 and smul16)
  // and 6 32-bit adders (5 adds + 1 sub). It absorbs add2 and chained
  // add4 -> add3, while add0 folds into add5 and add1/add5 remain unshared due
  // to profitability (max_delay_increase) and legalization conflicts, leaving
  // 8 adders total down from 12.
  EXPECT_EQ(NumberOfMultiplications(f), 4);
  EXPECT_EQ(NumberOfNodes(f, {Op::kSMul}), 4);
  EXPECT_EQ(NumberOfNodes(f, {Op::kUMul}), 0);
  EXPECT_EQ(NumberOfAdders(f), 8);

  auto eval = [&](uint64_t sel_val) -> absl::StatusOr<Value> {
    XLS_ASSIGN_OR_RETURN(
        InterpreterResult<Value> res,
        InterpretFunction(f, {Value(UBits(sel_val, 2)), Value(SBits(-4, 32)),
                              Value(SBits(-3, 32)), Value(UBits(128, 32)),
                              Value(UBits(2, 32))}));
    return res.value;
  };
  // sel = 0: (-4) * (-3) = 12.
  EXPECT_THAT(eval(0), IsOkAndHolds(Value(SBits(12, 32))));
  // sel = 1: (128*2) ^ (-3*2) ^ (-7) ^ 124 ^ 125 ^ (-2)
  //   = 0x0100 ^ 0xFFFA ^ 0xFFF9 ^ 0x007C ^ 0x007D ^ 0xFFFE = 0xFEFC (-260)
  EXPECT_THAT(eval(1), IsOkAndHolds(Value(SBits(-260, 32))));
  // sel = 2: (-1) ^ 130 = -131.
  EXPECT_THAT(eval(2), IsOkAndHolds(Value(SBits(-131, 32))));
}

void IrFuzzResourceSharing(FuzzPackageWithArgs fuzz_package_with_args) {
  ResourceSharingPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzResourceSharing)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace

}  // namespace xls
