// Copyright 2026 The XLS Authors
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

#include "xls/passes/visibility_contracted_dag.h"

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class VisibilityContractedDagTest : public IrTestBase {};

TEST_F(VisibilityContractedDagTest,
       BuildContractedVisibilityDagSeparatesJoinedAndAdjacent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Full def-use DAG:
  // A -> B -> D
  // |--> C ---^
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Not(a);
  BValue c = fb.Negate(a);
  BValue d = fb.Or(b, c);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(d).status());

  BinaryDecisionDiagram bdd;
  BddNodeIndex b_to_d_vis = bdd.NewVariable();
  auto edge_vis = [&](Node* operand, Node* user) -> BddNodeIndex {
    if (operand == b.node() && user == d.node()) {
      return b_to_d_vis;
    }
    return bdd.one();
  };

  // `one` is A, `edges` contains B -> D.
  // Order: [D (0), B (1), A (2)].
  // `adjacent` should be {B: {D}}; `joined` should be {A: {B, D}}.
  std::vector<ContractedDagNode> dag =
      BuildContractedVisibilityDag(a.node(), {{b.node(), d.node()}}, edge_vis);
  EXPECT_THAT(
      dag,
      ElementsAre(
          FieldsAre(IsEmpty(), IsEmpty()),
          FieldsAre(UnorderedElementsAre(ContractedDagEdge{
                        .user_idx = 0, .edge_idx = 0, .edge_vis = b_to_d_vis}),
                    IsEmpty()),
          FieldsAre(IsEmpty(), UnorderedElementsAre(0, 1))));

  // If A -> B is also in `edges`, then A -> B is in `adjacent` while A -> D
  // (via C) remains in `joined`.
  // Order: [D (0), B (1), A (2)].
  std::vector<ContractedDagNode> dag_with_ab = BuildContractedVisibilityDag(
      a.node(), {{a.node(), b.node()}, {b.node(), d.node()}}, edge_vis);
  EXPECT_THAT(
      dag_with_ab,
      ElementsAre(
          FieldsAre(IsEmpty(), IsEmpty()),
          FieldsAre(UnorderedElementsAre(ContractedDagEdge{
                        .user_idx = 0, .edge_idx = 1, .edge_vis = b_to_d_vis}),
                    IsEmpty()),
          FieldsAre(UnorderedElementsAre(ContractedDagEdge{
                        .user_idx = 1, .edge_idx = 0, .edge_vis = bdd.one()}),
                    UnorderedElementsAre(0))));
}

TEST_F(VisibilityContractedDagTest,
       BuildContractedVisibilityDagUnconditionalPathSupersedesAdjacentEdge) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Full def-use DAG:
  // A ------> D
  // |--> C ---^
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue c = fb.Negate(a);
  BValue d = fb.Or(a, c);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(d).status());

  BinaryDecisionDiagram bdd;
  auto always_one = [&](Node*, Node*) { return bdd.one(); };

  // Even though A -> D is in `edges`, A also has an unconditional path to D
  // via C, so D should be in `joined` for A and not `adjacent`.
  std::vector<ContractedDagNode> dag = BuildContractedVisibilityDag(
      a.node(), {{a.node(), d.node()}}, always_one);
  EXPECT_THAT(dag, ElementsAre(FieldsAre(IsEmpty(), IsEmpty()),
                               FieldsAre(IsEmpty(), UnorderedElementsAre(0))));
}

TEST_F(VisibilityContractedDagTest,
       BuildContractedVisibilityDagHandlesTerminal) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Full def-use DAG:
  // A -> B -> D
  // |--> C -> E
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Not(a);
  BValue c = fb.Negate(a);
  BValue d = fb.Identity(b);
  BValue e = fb.Identity(c);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(fb.Tuple({d, e})).status());

  BinaryDecisionDiagram bdd;
  auto always_one = [&](Node*, Node*) { return bdd.one(); };

  // A -> C -> E reaches the function return value without hitting any node in
  // `dag_nodes`, making A terminal in the contracted DAG.
  std::vector<ContractedDagNode> dag = BuildContractedVisibilityDag(
      a.node(), {{b.node(), d.node()}}, always_one);
  EXPECT_THAT(dag, ElementsAre(FieldsAre(IsEmpty(), IsEmpty())));
}

TEST_F(VisibilityContractedDagTest,
       BuildContractedVisibilityDagHandlesRepeatNodeEncounters) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Full def-use DAG:
  // A -> M ----> B -> D
  // |    |-> C --^    |
  // |-----------------|
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue m = fb.Not(a);
  BValue b = fb.Or(m, m);
  BValue c = fb.Negate(m);
  XLS_ASSERT_OK(b.node()->ReplaceOperandNumber(1, c.node()));
  BValue d = fb.Or(a, b);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(d).status());

  BinaryDecisionDiagram bdd;
  BddNodeIndex a_to_d_vis = bdd.NewVariable();
  BddNodeIndex b_to_d_vis = bdd.NewVariable();
  auto edge_vis = [&](Node* operand, Node* user) -> BddNodeIndex {
    if (operand == a.node() && user == d.node()) {
      return a_to_d_vis;
    }
    if (operand == b.node() && user == d.node()) {
      return b_to_d_vis;
    }
    return bdd.one();
  };

  // Order: [D (0), B (1), A (2)].  A is joined to B.
  std::vector<ContractedDagNode> dag = BuildContractedVisibilityDag(
      a.node(), {{a.node(), d.node()}, {b.node(), d.node()}}, edge_vis);
  EXPECT_THAT(
      dag,
      ElementsAre(
          FieldsAre(IsEmpty(), IsEmpty()),
          FieldsAre(UnorderedElementsAre(ContractedDagEdge{
                        .user_idx = 0, .edge_idx = 1, .edge_vis = b_to_d_vis}),
                    IsEmpty()),
          FieldsAre(UnorderedElementsAre(ContractedDagEdge{
                        .user_idx = 0, .edge_idx = 0, .edge_vis = a_to_d_vis}),
                    UnorderedElementsAre(1))));
}

}  // namespace
}  // namespace xls
