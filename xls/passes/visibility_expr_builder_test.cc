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

#include "xls/passes/visibility_expr_builder.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/visibility_analysis.h"
#include "xls/visualization/math_notation.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

class VisibilityExprBuilderTest : public IrTestBase {
 protected:
  absl::StatusOr<std::pair<Node*, VisibilityEstimator::AreaDelay>>
  BuildDefaultVisibilityExpr(
      Function* f, Node* node,
      std::vector<Node*> mutually_exclusive_requirements) {
    NodeForwardDependencyAnalysis nda;
    XLS_RETURN_IF_ERROR(nda.Attach(f).status());
    LazyPostDominatorAnalysis post_dom;
    XLS_RETURN_IF_ERROR(post_dom.Attach(f).status());
    std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
    XLS_RETURN_IF_ERROR(bdd_engine->Populate(f).status());
    XLS_ASSIGN_OR_RETURN(
        auto operand_visibility,
        OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
    XLS_ASSIGN_OR_RETURN(auto visibility,
                         VisibilityAnalysis::Create(
                             &operand_visibility, bdd_engine.get(), &post_dom));
    XLS_ASSIGN_OR_RETURN(auto conditional_edges,
                         visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(
                             node, mutually_exclusive_requirements, -1));
    VLOG(3) << "node: " << node->ToString() << "\n";
    VLOG(3) << "conditional_edges: " << conditional_edges.size() << "\n";
    for (auto edge : conditional_edges) {
      VLOG(3) << "edge: " << edge.operand->ToString() << " -> "
              << edge.node->ToString() << "\n";
    }
    auto last_node_id = f->nodes_reversed().begin()->id();
    XLS_ASSIGN_OR_RETURN(AreaEstimator * ae, GetAreaEstimator("unit"));
    XLS_ASSIGN_OR_RETURN(DelayEstimator * de, GetDelayEstimator("unit"));
    VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, ae, de);
    XLS_ASSIGN_OR_RETURN(Node * expr, estimator.BuildVisibilityIRExpr(
                                          f, node, conditional_edges));
    XLS_ASSIGN_OR_RETURN(
        auto area_delay,
        estimator.GetAreaAndDelayOfVisibilityExpr(node, conditional_edges));
    return std::make_pair(expr, area_delay);
  }
};

TEST_F(VisibilityExprBuilderTest, ExampleInFunctionHeaderComment) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));
  BValue op1 = fb.Param("op1", p->GetBitsType(4));
  BValue op2 = fb.Param("op2", p->GetBitsType(4));
  BValue select1 = fb.Select(op1, {x, y, x}, y);
  BValue lt1 = fb.ULt(op2, fb.Literal(UBits(5, 4)));
  BValue and1 = fb.And(x, fb.SignExtend(lt1, 32));
  BValue select2 = fb.Select(op1, {y, z, y}, and1);
  BValue ret = fb.Tuple({select1, select2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_x_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_x_used,
                           BuildDefaultVisibilityExpr(f, x.node(), {}));
  std::pair<Node*, VisibilityEstimator::AreaDelay> is_y_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_y_used,
                           BuildDefaultVisibilityExpr(f, y.node(), {}));
  VLOG(3) << f->DumpIr();
  VLOG(3) << "is 'x' used:\n" << ToMathNotation(is_x_used.first);
  VLOG(3) << "is 'y' used:\n" << ToMathNotation(is_y_used.first);

  EXPECT_THAT(is_x_used.first,
              m::Or(m::Or(m::Eq(m::Param("op1"), m::Literal(0)),
                          m::Eq(m::Param("op1"), m::Literal(2))),
                    m::And(m::ULt(m::Param("op2"), m::Literal(5)),
                           m::UGe(m::Param("op1"), m::Literal(3)))));
  // 6 instead of 7 because of the re-use of the ult.
  EXPECT_EQ(is_x_used.second.area, 6);
  EXPECT_EQ(is_x_used.second.delay, 3);
  EXPECT_THAT(is_y_used.first,
              m::Or(m::Or(m::Eq(m::Param("op1"), m::Literal(1)),
                          m::UGe(m::Param("op1"), m::Literal(3))),
                    m::Or(m::Eq(m::Param("op1"), m::Literal(0)),
                          m::Eq(m::Param("op1"), m::Literal(2)))));
  EXPECT_EQ(is_y_used.second.area, 7);
  EXPECT_EQ(is_y_used.second.delay, 3);

  // Now that the returned expression must be mutually exclusive with z's
  // visibility, it must condition on the selection criteria of 'select2'.
  std::pair<Node*, VisibilityEstimator::AreaDelay> is_x_used_and_z_not;
  XLS_ASSERT_OK_AND_ASSIGN(is_x_used_and_z_not,
                           BuildDefaultVisibilityExpr(f, x.node(), {z.node()}));
  VLOG(3) << "is 'x' used and 'z' not used:\n"
          << ToMathNotation(is_x_used_and_z_not.first);
  EXPECT_THAT(is_x_used_and_z_not.first,
              m::Or(m::Or(m::Eq(m::Param("op1"), m::Literal(0)),
                          m::Eq(m::Param("op1"), m::Literal(2))),
                    m::UGe(m::Param("op1"), m::Literal(3))));
  EXPECT_EQ(is_x_used_and_z_not.second.area, 5);
  EXPECT_EQ(is_x_used_and_z_not.second.delay, 3);
}

// TODO: add test to confirm IR expressions are never a function of the mutually
// exclusive nodes in question

}  // namespace
}  // namespace xls
