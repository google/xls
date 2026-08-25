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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
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
#include "xls/passes/resource_sharing_pass_test_base.h"
#include "xls/passes/visibility_expr_builder.h"
#include "xls/solvers/ir_equivalence_testutils.h"

namespace xls {

namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::xls::solvers::ScopedVerifyEquivalence;

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

class ResourceSharingPassTest : public IrTestBase {};

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

  BddQueryEngine bdd_engine;
  XLS_ASSERT_OK(bdd_engine.Populate(f));
  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK(bpa.Populate(f));
  VisibilityBuilder visibility_builder(next_node_id, &bdd_engine, nda, bpa);

  // Exercise producing a cycle by requesting to fold X into Y; yes, they are
  // not mutually exclusive, but the transformation doesn't have a feasible way
  // to check whether analyses were correct; however, it should detect the cycle
  // that would result from this transformation.
  VisibilityEdges edges = {
      OperandVisibilityAnalysis::OperandNode(X.node(), sel.node()),
      OperandVisibilityAnalysis::OperandNode(Y.node(), sel.node())};
  std::vector<std::pair<Node*, VisibilityEdges>> from_X = {
      std::make_pair(X.node(), edges)};
  auto fold_X_into_Y = std::make_unique<NaryFoldingAction>(
      std::move(from_X), Y.node(), edges, /*area_saved=*/0.0,
      /*sinks=*/absl::flat_hash_set<Node*>{sel.node()});
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  folding_actions_to_perform.push_back(std::move(fold_X_into_Y));

  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_ASSERT_OK(nda_backwards.Attach(f));
  EXPECT_THAT(
      ResourceSharingPass::PerformFoldingActions(
          f, next_node_id, &visibility_builder, nda_backwards,
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

  BddQueryEngine bdd_engine;
  XLS_ASSERT_OK(bdd_engine.Populate(f));
  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK(bpa.Populate(f));
  VisibilityBuilder visibility_builder(next_node_id, &bdd_engine, nda, bpa);

  // Action 1: Fold A1 into A0.
  VisibilityEdges edges_A1 = {
      OperandVisibilityAnalysis::OperandNode(B1.node(), sel.node())};
  VisibilityEdges edges_A0 = {
      OperandVisibilityAnalysis::OperandNode(B0.node(), sel.node())};
  std::vector<std::pair<Node*, VisibilityEdges>> from_1 = {
      std::make_pair(A1.node(), edges_A1)};
  auto fold_1 = std::make_unique<NaryFoldingAction>(
      std::move(from_1), A0.node(), edges_A0, /*area_saved=*/1.0,
      /*sinks=*/absl::flat_hash_set<Node*>{sel.node()});

  // Action 2: Fold A0 into A2.
  // Without precomputing visibility expressions, the visibility of A0, now
  // replaced by a folded selection between A0 and A1, would be corrupted by
  // the previous folding action.
  VisibilityEdges edges_A2 = {
      OperandVisibilityAnalysis::OperandNode(A2.node(), sel.node())};
  std::vector<std::pair<Node*, VisibilityEdges>> from_2 = {
      std::make_pair(A0.node(), edges_A0)};
  auto fold_2 = std::make_unique<NaryFoldingAction>(
      std::move(from_2), A2.node(), edges_A2, /*area_saved=*/1.0,
      /*sinks=*/absl::flat_hash_set<Node*>{sel.node()});

  ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(1));
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  folding_actions_to_perform.push_back(std::move(fold_1));
  folding_actions_to_perform.push_back(std::move(fold_2));
  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_ASSERT_OK(nda_backwards.Attach(f));
  XLS_EXPECT_OK(ResourceSharingPass::PerformFoldingActions(
      f, next_node_id, &visibility_builder, nda_backwards,
      folding_actions_to_perform));
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

  auto make_action = [&](Node* target, Node* source = nullptr) {
    std::vector<std::pair<Node*, VisibilityEdges>> sources;
    if (source != nullptr) {
      sources.push_back({source, {}});
    }
    return std::make_unique<NaryFoldingAction>(
        std::move(sources), target, VisibilityEdges{}, /*area_saved=*/0.0);
  };

  // The comparator doesn't care that we specify no source nodes, and this
  // allows us to conveniently express a folding of no delay spread / increase.
  auto make_add_low = [&] { return make_action(add_low.node()); };
  auto make_sub_mid = [&] {
    return make_action(sub_mid.node(), from_mid.node());
  };
  auto make_add_high = [&] {
    return make_action(add_high.node(), from_mid.node());
  };
  auto make_sub_highest = [&] {
    return make_action(sub_highest.node(), from_mid.node());
  };

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  CriticalPathDelayAnalysis delay_analysis(delay_estimator);
  XLS_ASSERT_OK(delay_analysis.Attach(f));

  auto sort_and_get_names = [&](std::unique_ptr<NaryFoldingAction> a,
                                std::unique_ptr<NaryFoldingAction> b,
                                std::unique_ptr<NaryFoldingAction> c,
                                std::unique_ptr<NaryFoldingAction> d) {
    std::vector<std::unique_ptr<NaryFoldingAction>> actions;
    actions.push_back(std::move(a));
    actions.push_back(std::move(b));
    actions.push_back(std::move(c));
    actions.push_back(std::move(d));
    TimingAnalysis ta(actions, delay_analysis);
    SortFoldingActionsInDescendingOrderOfTheirAreaSavings(actions, ta);
    return std::vector<std::string>{
        actions[0]->GetTo()->GetName(), actions[1]->GetTo()->GetName(),
        actions[2]->GetTo()->GetName(), actions[3]->GetTo()->GetName()};
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
  const std::vector<std::string> kExpected = {"add_high", "add_low", "sub_mid",
                                              "sub_highest"};

  // Verify a few input permutations to expose weak comparator hardening bugs.
  EXPECT_EQ(sort_and_get_names(make_add_low(), make_sub_mid(), make_add_high(),
                               make_sub_highest()),
            kExpected);
  EXPECT_EQ(sort_and_get_names(make_add_low(), make_add_high(), make_sub_mid(),
                               make_sub_highest()),
            kExpected);
  EXPECT_EQ(sort_and_get_names(make_sub_mid(), make_add_high(), make_add_low(),
                               make_sub_highest()),
            kExpected);
  EXPECT_EQ(sort_and_get_names(make_sub_highest(), make_sub_mid(),
                               make_add_high(), make_add_low()),
            kExpected);
}

void IrFuzzResourceSharing(FuzzPackageWithArgs fuzz_package_with_args) {
  ResourceSharingPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzResourceSharing)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace

}  // namespace xls
