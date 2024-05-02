// Copyright 2024 The XLS Authors
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

#include "xls/ir/elaborated_block_dfs_visitor.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

MATCHER_P(NodeIs, m, "") {
  return ::testing::ExplainMatchResult(m, arg.node, result_listener);
}

using ::testing::HasSubstr;
using ::xls::status_testing::StatusIs;

namespace xls {
namespace {

absl::StatusOr<BlockElaboration> ParseAndElaborate(std::string_view ir,
                                                   Package* p) {
  XLS_ASSIGN_OR_RETURN(Block * b, Parser::ParseBlock(ir, p));
  return BlockElaboration::Elaborate(b);
}

// A testing visitor which records which nodes it has visited.
class TestVisitor : public ElaboratedBlockDfsVisitorWithDefault {
 public:
  absl::Status DefaultHandler(const ElaboratedNode& node) override {
    VLOG(1) << "Visiting " << node.ToString();
    visited_.push_back(node);
    visited_set_.insert(node);
    return absl::OkStatus();
  }

  // Returns the ordered set of visited nodes.
  const std::vector<ElaboratedNode>& visited() const { return visited_; }

  int64_t visited_count() const { return visited_.size(); }

  // Returns the total number of unique nodes visited.
  int64_t unique_visited_count() const { return visited_set_.size(); }

 private:
  std::vector<ElaboratedNode> visited_;
  absl::flat_hash_set<ElaboratedNode> visited_set_;
};

using ElaboratedBlockDfsVisitor = IrTestBase;

TEST_F(ElaboratedBlockDfsVisitor, VisitSingleNode) {
  constexpr std::string_view input = R"(
block single_node() {
   literal.1: bits[32] = literal(value=42)
})";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           ParseAndElaborate(input, p.get()));

  TestVisitor v;
  ElaboratedNode node = ElaboratedNode{
      .node =
          elab.top()->block().value()->GetNode("literal.1").value_or(nullptr),
      .instance = elab.top()};
  EXPECT_FALSE(v.IsVisited(node));
  XLS_ASSERT_OK(node.Accept(v));
  EXPECT_TRUE(v.IsVisited(node));
  EXPECT_EQ(1, v.visited_count());
  EXPECT_THAT(v.visited(), ::testing::ElementsAre(node));

  // Calling it again should not revisit the node.
  XLS_ASSERT_OK(node.Accept(v));
  EXPECT_EQ(1, v.visited_count());
}

TEST_F(ElaboratedBlockDfsVisitor, VisitGraph) {
  constexpr std::string_view input = R"(
block graph(p: bits[42], q: bits[42], r: bits[42]) {
  p: bits[42] = input_port(name=p)
  q: bits[42] = input_port(name=q)
  and.3: bits[42] = and(p, q)
  add.4: bits[42] = add(and.3, q)
  sub.5: bits[42] = sub(add.4, add.4)
  r: () = output_port(sub.5, name=r)
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           ParseAndElaborate(input, p.get()));
  Block* top = elab.top()->block().value();

  {
    // Visit from the root.
    TestVisitor v;
    ElaboratedNode r =
        ElaboratedNode{.node = FindNode("r", top), .instance = elab.top()};
    XLS_ASSERT_OK(r.Accept(v));
    EXPECT_EQ(6, v.visited_count());
    EXPECT_EQ(6, v.unique_visited_count());
    EXPECT_THAT(v.visited().back(), NodeIs(FindNode("r", top)));
    EXPECT_THAT(
        v.visited(),
        ::testing::UnorderedElementsAre(
            NodeIs(FindNode("r", top)), NodeIs(FindNode("sub.5", top)),
            NodeIs(FindNode("add.4", top)), NodeIs(FindNode("and.3", top)),
            NodeIs(FindNode("p", top)), NodeIs(FindNode("q", top))));
  }

  {
    // Visit from an interior node.
    TestVisitor v;
    ElaboratedNode and_1 =
        ElaboratedNode{.node = FindNode("and.3", top), .instance = elab.top()};
    XLS_ASSERT_OK(and_1.Accept(v));
    EXPECT_EQ(3, v.visited_count());
    EXPECT_THAT(v.visited(),
                ::testing::UnorderedElementsAre(NodeIs(FindNode("and.3", top)),
                                                NodeIs(FindNode("p", top)),
                                                NodeIs(FindNode("q", top))));
    EXPECT_FALSE(v.IsVisited(ElaboratedNode{.node = FindNode("add.4", top),
                                            .instance = elab.top()}));
    EXPECT_FALSE(v.IsVisited(ElaboratedNode{.node = FindNode("sub.5", top),
                                            .instance = elab.top()}));
  }
}

TEST_F(ElaboratedBlockDfsVisitor, ElabVisit) {
  // The umul operation is dead.
  constexpr std::string_view input = R"(
block graph(p: bits[42], q: bits[42], r: bits[42]) {
  p: bits[42] = input_port(name=p)
  q: bits[42] = input_port(name=q)
  and.3: bits[42] = and(p, q)
  add.4: bits[42] = add(and.3, q)
  umul.5: bits[42] = umul(add.4, p)
  sub.6: bits[42] = sub(add.4, add.4)
  r: () = output_port(sub.6, name=r)
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           ParseAndElaborate(input, p.get()));
  Block* top = elab.top()->block().value();

  {
    // Visit from the root should only visit 5 nodes, not the dead umul.
    TestVisitor v;
    ElaboratedNode r{.node = FindNode("r", top), .instance = elab.top()};
    XLS_ASSERT_OK(r.Accept(v));
    EXPECT_EQ(6, v.visited_count());
    EXPECT_FALSE(v.IsVisited(ElaboratedNode{.node = FindNode("umul.5", top),
                                            .instance = elab.top()}));
  }

  {
    // Calling Function::Accept should visit all nodes in the graph including
    // the dead multiply.
    TestVisitor v;
    XLS_ASSERT_OK(elab.Accept(v));
    EXPECT_EQ(7, v.visited_count());
    EXPECT_TRUE(v.IsVisited(ElaboratedNode{.node = FindNode("umul.5", top),
                                           .instance = elab.top()}));
  }
}

TEST_F(ElaboratedBlockDfsVisitor, DetectSimpleCycle) {
  Package p(TestName());
  // Make a cycle between two identity nodes.
  BlockBuilder b(TestName(), &p, /*should_verify=*/false);
  BValue temp = b.Literal(UBits(0, 1));
  BValue identity0 = b.Identity(temp, SourceInfo(), "identity0");
  BValue identity1 = b.Identity(identity0, SourceInfo(), "identity1");
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, b.Build());
  XLS_ASSERT_OK(identity0.node()->ReplaceOperandNumber(0, identity1.node()));
  XLS_ASSERT_OK(top->RemoveNode(temp.node()));

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(top));

  TestVisitor v;
  EXPECT_THAT(elab.Accept(v), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("Cycle detected:")));
  EXPECT_LT(v.visited_count(), top->node_count());
}

TEST_F(ElaboratedBlockDfsVisitor, HierarchicalCycleDetected) {
  constexpr std::string_view input = R"(
package has_cycle

block bar(a: bits[42], b: bits[42], c: bits[42], d: bits[42]) {
  a: bits[42] = input_port(name=a)
  b: bits[42] = input_port(name=b)
  c: () = output_port(b, name=c)
  d: () = output_port(a, name=d)
}

block foo(p: bits[42], q: bits[42], r: bits[42]) {
  instantiation inst(kind=block, block=bar)
  p: bits[42] = input_port(name=p)
  q: bits[42] = input_port(name=q)
  bar_c: bits[42] = instantiation_output(instantiation=inst, port_name=c)
  bar_d: bits[42] = instantiation_output(instantiation=inst, port_name=d)
  p_plus_c: bits[42] = add(p, bar_c)
  q_plus_d: bits[42] = add(q, bar_d)
  sum: bits[42] = add(p_plus_c, q_plus_d)
  bar_a: () = instantiation_input(sum, instantiation=inst, port_name=a)
  bar_b: () = instantiation_input(sum, instantiation=inst, port_name=b)
  r: () = output_port(sum, name=r)
}
)";
  // ParsePackageNoVerify because there's a cycle.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(
      BlockElaboration elab,
      BlockElaboration::Elaborate(p->GetBlock("foo").value()));

  TestVisitor v;
  EXPECT_THAT(
      elab.Accept(v),
      StatusIs(absl::StatusCode::kInternal,
               AllOf(HasSubstr("Cycle detected:"), HasSubstr("sum"),
                     HasSubstr("p_plus_c"), HasSubstr("bar_c"), HasSubstr("c"),
                     HasSubstr("b"), HasSubstr("bar_b"), HasSubstr("sum"))));
}

}  // namespace
}  // namespace xls
