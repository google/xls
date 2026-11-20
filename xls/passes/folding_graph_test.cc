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

#include "xls/passes/folding_graph.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/resource_sharing_equivalence.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

class FoldingGraphTest : public IrTestBase {
 protected:
  FoldingGraphTest() = default;
};

TEST_F(FoldingGraphTest, FoldingGraphIsOrdered) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue literal_0 = fb.Literal(UBits(0, 32));
  BValue literal_1 = fb.Literal(UBits(1, 32));
  BValue add = fb.Add(i, j);
  BValue sub = fb.Subtract(i, j);
  BValue mul = fb.UMul(i, j);
  BValue select = fb.PrioritySelect(
      fb.Concat({fb.Eq(op, literal_1), fb.Eq(op, literal_0)}), {add, sub}, mul);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));
  std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions;
  // NOTE: node order is reversed to ensure FoldingGraph does sorting.
  for (Node* one : {mul.node(), sub.node(), add.node()}) {
    for (Node* other : {sub.node(), mul.node(), add.node()}) {
      if (one != other) {
        foldable_actions.push_back(std::make_unique<BinaryFoldingAction>(
            one, other, /*from_edges=*/FoldingAction::VisibilityEdges(),
            /*to_edges=*/FoldingAction::VisibilityEdges(), 0.0));
      }
    }
  }
  FoldingGraph graph(f, std::move(foldable_actions));

  EXPECT_EQ(graph.GetNodes().size(), 3);
  EXPECT_THAT(graph.GetNodes(),
              ElementsAre(add.node(), sub.node(), mul.node()));
  EXPECT_EQ(graph.GetEdges().size(), 6);
  EXPECT_EQ(graph.GetEdges()[0]->GetFrom(), add.node());
  EXPECT_EQ(graph.GetEdges()[0]->GetTo(), sub.node());
  EXPECT_EQ(graph.GetEdges()[1]->GetFrom(), add.node());
  EXPECT_EQ(graph.GetEdges()[1]->GetTo(), mul.node());
  EXPECT_EQ(graph.GetEdges()[2]->GetFrom(), sub.node());
  EXPECT_EQ(graph.GetEdges()[2]->GetTo(), add.node());
  EXPECT_EQ(graph.GetEdges()[3]->GetFrom(), sub.node());
  EXPECT_EQ(graph.GetEdges()[3]->GetTo(), mul.node());
  EXPECT_EQ(graph.GetEdges()[4]->GetFrom(), mul.node());
  EXPECT_EQ(graph.GetEdges()[4]->GetTo(), add.node());
  EXPECT_EQ(graph.GetEdges()[5]->GetFrom(), mul.node());
  EXPECT_EQ(graph.GetEdges()[5]->GetTo(), sub.node());
}

TEST_F(FoldingGraphTest, WouldCommittingFoldingActionCreateDataCycle) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));
  BValue w = fb.Param("w", p->GetBitsType(32));
  BValue add0 = fb.Add(x, y);
  BValue add1 = fb.Add(z, w);
  BValue shared_op_add = fb.Add(x, z);
  BValue dep = fb.Add(add0, z);
  BValue from_with_dep_lhs = fb.Add(dep, w);
  BValue from_with_dep_rhs = fb.Add(w, dep);
  BValue to_dependent_on_from = fb.Add(add1, z);
  BValue not_x = fb.Not(x);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.Build());

  FunctionBuilder fb2("other_func", p.get());
  BValue a = fb2.Param("a", p->GetBitsType(32));
  BValue b = fb2.Param("b", p->GetBitsType(32));
  BValue other_add = fb2.Add(a, b);
  XLS_ASSERT_OK(fb2.Build().status());

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));

  {
    // No cycle created when folding two independent nodes.
    BinaryFoldingAction fold(add0.node(), add1.node(), /*from_edges=*/{},
                             /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(false));
  }
  {
    // No cycle created when to node depends on from node, but operands of from
    // node do not depend on to node.
    BinaryFoldingAction fold(add1.node(), to_dependent_on_from.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(false));
  }
  {
    // No cycle created when from and to share an operand.
    BinaryFoldingAction fold(shared_op_add.node(), add0.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(false));
  }
  {
    // Cycle created when from node depends directly on to node.
    BinaryFoldingAction fold(to_dependent_on_from.node(), add1.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when left operand of from node depends on to node.
    BinaryFoldingAction fold(from_with_dep_lhs.node(), add0.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when right operand of from node depends on to node.
    BinaryFoldingAction fold(from_with_dep_rhs.node(), add0.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // No cycle created when to node is from a different function.
    BinaryFoldingAction fold(add0.node(), other_add.node(), /*from_edges=*/{},
                             /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(false));
  }
  {
    // No cycle created when from node is from a different function.
    BinaryFoldingAction fold(other_add.node(), add0.node(), /*from_edges=*/{},
                             /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                IsOkAndHolds(false));
  }
  {
    // Error returned when from and to have different operand counts.
    BinaryFoldingAction fold(not_x.node(), add0.node(), /*from_edges=*/{},
                             /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateDataCycle(nda, fold),
                StatusIs(absl::StatusCode::kInternal));
  }
}

TEST_F(FoldingGraphTest, WouldCommittingFoldingActionCreateVisibilityCycle) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue sel_not = fb.Not(sel);
  BValue t = fb.Param("t", p->GetBitsType(32));
  BValue f_param = fb.Param("f", p->GetBitsType(32));
  BValue select = fb.Select(sel_not, t, f_param);
  BValue unrelated = fb.Param("unrelated", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.Build());

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));

  {
    // No cycle created when visibility edges don't depend on from/to nodes.
    BinaryFoldingAction fold(
        t.node(), unrelated.node(),
        /*from_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())},
        /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(false));
  }
  {
    // Cycle created when from node is in from_edges dependency path.
    BinaryFoldingAction fold(
        sel.node(), unrelated.node(),
        /*from_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())},
        /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when to node is in from_edges dependency path.
    BinaryFoldingAction fold(
        unrelated.node(), sel.node(),
        /*from_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())},
        /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when from node is in to_edges dependency path.
    BinaryFoldingAction fold(
        sel.node(), unrelated.node(),
        /*from_edges=*/{},
        /*to_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when to node is in to_edges dependency path.
    BinaryFoldingAction fold(
        unrelated.node(), sel.node(),
        /*from_edges=*/{},
        /*to_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
}

auto EqMappingBetween(Node* src, Node* dst) {
  return Pointee(AllOf(Property(&EquivalenceMapping::src, src),
                       Property(&EquivalenceMapping::dst, dst)));
}

auto CompositionIs(Node* src, Node* dst, Node* part) {
  return AllOf(Field(&CompositionPart::mapping, EqMappingBetween(src, dst)),
               Field(&CompositionPart::part, part));
}

absl::StatusOr<std::vector<CompositionPart>> DecomposeParts(Node* node, Op op) {
  XLS_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<EquivalenceMapping>> decomps,
                       GetNodeEquivalenceMapper().ComputeDecompositions(node));
  XLS_RET_CHECK(!decomps.empty());
  std::vector<CompositionPart> parts;
  for (Node* part : decomps[0]->dst()->function_base()->nodes()) {
    if (part->op() == op) {
      parts.push_back(CompositionPart{.mapping = decomps[0], .part = part});
    }
  }
  std::sort(parts.begin(), parts.end(),
            [](const CompositionPart& a, const CompositionPart& b) {
              return a.part->id() < b.part->id();
            });
  XLS_RET_CHECK(!parts.empty());
  return parts;
}

TEST_F(FoldingGraphTest, FoldingGraphWithDecomposedParts) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(2));
  BValue from_mul = fb.UMul(a, a, /*result_width=*/4);
  BValue to_mul = fb.SMul(a, a, /*result_width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({from_mul, to_mul})));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<CompositionPart> from_umuls,
                           DecomposeParts(from_mul.node(), Op::kUMul));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<CompositionPart> to_smuls,
                           DecomposeParts(to_mul.node(), Op::kSMul));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<CompositionPart> to_adds,
                           DecomposeParts(to_mul.node(), Op::kAdd));
  ASSERT_GE(from_umuls.size(), 2);
  ASSERT_GE(to_smuls.size(), 2);
  ASSERT_LT(to_smuls[0].part->id(), to_adds[0].part->id());

  std::vector<std::unique_ptr<BinaryFoldingAction>> actions;
  auto add_action = [&](std::optional<CompositionPart> fc,
                        std::optional<CompositionPart> tc) {
    actions.push_back(std::make_unique<BinaryFoldingAction>(BinaryFoldingAction(
        from_mul.node(), to_mul.node(), /*from_edges=*/{}, /*to_edges=*/{},
        /*area_saved=*/0.0, /*mappings=*/{}, /*sinks=*/{}, std::move(fc),
        std::move(tc))));
    return actions.back().get();
  };
  // Insert out of sorted (from_part_id, to_part_id) order
  BinaryFoldingAction* e_p1_smul1 = add_action(from_umuls[1], to_smuls[1]);
  BinaryFoldingAction* e_p0_smul0 = add_action(from_umuls[0], to_smuls[0]);
  BinaryFoldingAction* e_p0_mul = add_action(from_umuls[0], std::nullopt);
  BinaryFoldingAction* e_mul_add = add_action(std::nullopt, to_adds[0]);
  BinaryFoldingAction* e_mul_smul0 = add_action(std::nullopt, to_smuls[0]);

  FoldingGraph graph(f, std::move(actions));

  std::string from_name = from_mul.node()->GetName();
  std::string to_name = to_mul.node()->GetName();
  auto part_name = [](const std::string& base, const CompositionPart& comp) {
    return base + ":" + comp.part->GetName();
  };
  auto edge_is = [](BinaryFoldingAction* expected, Node* from, Node* to,
                    bool is_signed, const std::string& name) {
    return AllOf(expected, Property(&BinaryFoldingAction::GetFromOrPart, from),
                 Property(&BinaryFoldingAction::GetToOrPart, to),
                 Property(&BinaryFoldingAction::IsSigned, is_signed),
                 Property(&BinaryFoldingAction::GetName, name));
  };

  EXPECT_THAT(graph.GetNodes(),
              ElementsAre(from_mul.node(), to_smuls[0].part, to_adds[0].part,
                          from_umuls[0].part, to_mul.node(), from_umuls[1].part,
                          to_smuls[1].part));
  EXPECT_THAT(
      graph.GetEdges(),
      ElementsAre(
          edge_is(e_mul_smul0, from_mul.node(), to_smuls[0].part, true,
                  from_name + "->" + part_name(to_name, to_smuls[0])),
          edge_is(e_mul_add, from_mul.node(), to_adds[0].part, false,
                  from_name + "->" + part_name(to_name, to_adds[0])),
          edge_is(e_p0_mul, from_umuls[0].part, to_mul.node(), true,
                  part_name(from_name, from_umuls[0]) + "->" + to_name),
          edge_is(e_p0_smul0, from_umuls[0].part, to_smuls[0].part, true,
                  part_name(from_name, from_umuls[0]) + "->" +
                      part_name(to_name, to_smuls[0])),
          edge_is(e_p1_smul1, from_umuls[1].part, to_smuls[1].part, true,
                  part_name(from_name, from_umuls[1]) + "->" +
                      part_name(to_name, to_smuls[1]))));

  EXPECT_EQ(graph.GetEdge(from_umuls[0].part, to_smuls[0].part), e_p0_smul0);
  EXPECT_EQ(graph.GetEdge(from_mul.node(), to_mul.node()), nullptr);
  EXPECT_EQ(graph.GetEdge(a.node(), to_mul.node()), nullptr);
  EXPECT_THAT(graph.GetEdgesTo(to_smuls[0].part),
              ElementsAre(e_mul_smul0, e_p0_smul0));
  EXPECT_EQ(graph.GetInDegree(to_smuls[0].part), 2);
  EXPECT_EQ(graph.GetOutDegree(from_umuls[0].part), 2);
  EXPECT_THAT(graph.GetConnectedComponents(),
              UnorderedElementsAre(UnorderedElementsAre(e_mul_smul0, e_mul_add,
                                                        e_p0_mul, e_p0_smul0),
                                   UnorderedElementsAre(e_p1_smul1)));
}

TEST_F(FoldingGraphTest, CloneNaryFoldingActionWithCompositions) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(2));
  BValue from_mul4 = fb.UMul(a, a, /*result_width=*/4);
  BValue to_mul4 = fb.UMul(a, a, /*result_width=*/4);
  BValue from_mul2 = fb.UMul(a, a, /*result_width=*/2);
  BValue clone_from_mul4 = fb.UMul(a, a, /*result_width=*/4);
  BValue clone_to_mul4 = fb.UMul(a, a, /*result_width=*/4);
  BValue clone_from_mul2 = fb.UMul(a, a, /*result_width=*/2);
  XLS_ASSERT_OK(fb.Build().status());

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<CompositionPart> from_parts,
                           DecomposeParts(from_mul4.node(), Op::kUMul));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<CompositionPart> to_parts,
                           DecomposeParts(to_mul4.node(), Op::kUMul));
  const CompositionPart& from_comp = from_parts[0];
  const CompositionPart& to_comp = to_parts[0];

  absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>> mappings;
  mappings[from_comp.part] = std::make_unique<IdentityEquivalenceMapping>(
      from_comp.part, to_comp.part);
  mappings[from_mul2.node()] = std::make_unique<IdentityEquivalenceMapping>(
      from_mul2.node(), to_comp.part);
  mappings[to_comp.part] =
      std::make_unique<IdentityEquivalenceMapping>(to_comp.part, to_comp.part);

  EXPECT_THAT(
      NaryFoldingAction::Clone(
          {/*froms=*/{{.node = from_mul4.node(), .composition = from_comp},
                      {.node = from_mul2.node()}},
           to_mul4.node(),
           /*to_edges=*/{},
           /*area_saved=*/0.0, std::move(mappings),
           /*to_composition=*/to_comp},
          {{from_mul4.node(), clone_from_mul4.node()},
           {to_mul4.node(), clone_to_mul4.node()},
           {from_mul2.node(), clone_from_mul2.node()}}),
      IsOkAndHolds(Pointee(AllOf(
          Property(&NaryFoldingAction::GetTo, clone_to_mul4.node()),
          Property(&NaryFoldingAction::GetToOrPart, to_comp.part),
          Property(
              &NaryFoldingAction::to_composition,
              Optional(CompositionIs(clone_to_mul4.node(),
                                     to_comp.mapping->dst(), to_comp.part))),
          Property(
              &NaryFoldingAction::GetFrom,
              ElementsAre(
                  AllOf(Field(&NaryFoldingAction::NodeVisibility::node,
                              clone_from_mul4.node()),
                        Field(&NaryFoldingAction::NodeVisibility::composition,
                              Optional(CompositionIs(clone_from_mul4.node(),
                                                     from_comp.mapping->dst(),
                                                     from_comp.part)))),
                  NaryFoldingAction::NodeVisibility{
                      .node = clone_from_mul2.node()})),
          Property(
              &NaryFoldingAction::mappings,
              UnorderedElementsAre(
                  Pair(from_comp.part,
                       EqMappingBetween(from_comp.part, to_comp.part)),
                  Pair(clone_from_mul2.node(),
                       EqMappingBetween(clone_from_mul2.node(), to_comp.part)),
                  Pair(to_comp.part,
                       EqMappingBetween(to_comp.part, to_comp.part))))))));
}

}  // namespace
}  // namespace xls
