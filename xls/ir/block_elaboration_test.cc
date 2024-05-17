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

#include "xls/ir/block_elaboration.h"

#include <algorithm>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/elaborated_block_dfs_visitor.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace m = xls::op_matchers;

using ElaborationTest = IrTestBase;

MATCHER_P(BlockInstanceFor, value, "") { return arg->block() == value; }
MATCHER_P(BlockInstanceName, value, "") {
  return arg->block().value()->name() == value;
}
MATCHER_P(InstantiationName, value, "") {
  if (!arg->instantiation().has_value()) {
    return false;
  }
  return arg->instantiation().value()->name() == value;
}

MATCHER_P(NodeIs, node_matcher, "") {
  return ExplainMatchResult(node_matcher, arg.node, result_listener);
}

MATCHER_P2(NodeAndInst, node_matcher, inst_matcher, "") {
  return ExplainMatchResult(node_matcher, arg.node, result_listener) &&
         ExplainMatchResult(inst_matcher, arg.instance->ToString(),
                            result_listener);
}

absl::StatusOr<Block*> AddBlock(Package& p) {
  Type* u32 = p.GetBitsType(32);

  BlockBuilder bb("adder", &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("c", bb.Add(a, b));
  return bb.Build();
}

TEST_F(ElaborationTest, ElaborateSingleBlock) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, AddBlock(*p));

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(block));

  EXPECT_EQ(elab.package(), p.get());
  EXPECT_THAT(elab.top(), BlockInstanceFor(block));
  EXPECT_EQ(elab.top()->instantiation(),
            std::nullopt);  // top is not an instantiation
  EXPECT_EQ(elab.top()->path().top, elab.top()->block());
  EXPECT_THAT(elab.top()->path().path, IsEmpty());
  EXPECT_THAT(elab.top()->child_instances(), IsEmpty());
  EXPECT_THAT(elab.top()->instantiation_to_instance(), IsEmpty());
  EXPECT_THAT(elab.blocks(), UnorderedElementsAre(block));
  EXPECT_THAT(elab.instances(), UnorderedElementsAre(BlockInstanceFor(block)));
  EXPECT_THAT(elab.GetInstance("adder"), IsOkAndHolds(BlockInstanceFor(block)));
  EXPECT_THAT(
      elab.GetInstance("subtraction"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Path top `subtraction` does not match name of top proc")));
  EXPECT_THAT(elab.GetInstance("adder::subtraction"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid component of path `subtraction`.")));
  EXPECT_THAT(
      elab.GetInstance("adder::subtraction->sub"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("does not have an instantiation named `subtraction`")));

  EXPECT_THAT(elab.GetInstances(block),
              UnorderedElementsAre(BlockInstanceFor(block)));
}

absl::StatusOr<Block*> MultipleAddInstantiations(Package& p) {
  Type* u32 = p.GetBitsType(32);
  XLS_ASSIGN_OR_RETURN(Block * adder, AddBlock(p));

  BlockBuilder bb("multi_adder", &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  BValue c = bb.InputPort("c", u32);
  BValue d = bb.InputPort("d", u32);

  XLS_ASSIGN_OR_RETURN(BlockInstantiation * adder0_inst,
                       bb.block()->AddBlockInstantiation("adder0_inst", adder));
  XLS_ASSIGN_OR_RETURN(BlockInstantiation * adder1_inst,
                       bb.block()->AddBlockInstantiation("adder1_inst", adder));
  XLS_ASSIGN_OR_RETURN(BlockInstantiation * adder2_inst,
                       bb.block()->AddBlockInstantiation("adder2_inst", adder));

  bb.InstantiationInput(adder0_inst, "a", a);
  bb.InstantiationInput(adder0_inst, "b", b);
  bb.InstantiationInput(adder1_inst, "a", c);
  bb.InstantiationInput(adder1_inst, "b", d);
  BValue partial0 = bb.InstantiationOutput(adder0_inst, "c");
  BValue partial1 = bb.InstantiationOutput(adder1_inst, "c");
  bb.InstantiationInput(adder2_inst, "a", partial0);
  bb.InstantiationInput(adder2_inst, "b", partial1);
  bb.OutputPort("out", bb.InstantiationOutput(adder2_inst, "c"));
  return bb.Build();
}

TEST_F(ElaborationTest, ElaborateMultipleBlockInstantiations) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, MultipleAddInstantiations(*p));

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(block));

  EXPECT_EQ(elab.package(), p.get());

  ASSERT_THAT(elab.top(), BlockInstanceFor(block));
  EXPECT_EQ(elab.top()->block().value()->name(), "multi_adder");
  EXPECT_EQ(elab.top()->instantiation(),
            std::nullopt);  // top is not an instantiation
  EXPECT_EQ(elab.top()->path().top, elab.top()->block());
  EXPECT_THAT(elab.top()->path().path, IsEmpty());
  ASSERT_EQ(elab.top()->child_instances().size(), 3);
  EXPECT_THAT(elab.top()->child_instances(), Each(BlockInstanceName("adder")));
  EXPECT_THAT(
      elab.top()->instantiation_to_instance(),
      UnorderedElementsAre(
          Pair(m::Instantiation("adder0_inst"), BlockInstanceName("adder")),
          Pair(m::Instantiation("adder1_inst"), BlockInstanceName("adder")),
          Pair(m::Instantiation("adder2_inst"), BlockInstanceName("adder"))));

  Block* child_block = *elab.top()->child_instances().front()->block();

  EXPECT_THAT(elab.blocks(), UnorderedElementsAre(block, child_block));
  EXPECT_THAT(elab.instances(),
              UnorderedElementsAre(BlockInstanceFor(block),
                                   AllOf(BlockInstanceFor(child_block),
                                         InstantiationName("adder0_inst")),
                                   AllOf(BlockInstanceFor(child_block),
                                         InstantiationName("adder1_inst")),
                                   AllOf(BlockInstanceFor(child_block),
                                         InstantiationName("adder2_inst"))));

  EXPECT_THAT(elab.GetInstance("multi_adder"),
              IsOkAndHolds(BlockInstanceFor(block)));
  EXPECT_THAT(elab.GetInstance("multi_adder::adder0_inst->adder"),
              IsOkAndHolds(AllOf(BlockInstanceFor(child_block),
                                 InstantiationName("adder0_inst"))));
  EXPECT_THAT(elab.GetInstance("multi_adder::adder1_inst->adder"),
              IsOkAndHolds(AllOf(BlockInstanceFor(child_block),
                                 InstantiationName("adder1_inst"))));
  EXPECT_THAT(elab.GetInstance("multi_adder::adder2_inst->adder"),
              IsOkAndHolds(AllOf(BlockInstanceFor(child_block),
                                 InstantiationName("adder2_inst"))));
  EXPECT_THAT(
      elab.GetInstance("multi_adder::adder3_inst->adder"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("does not have an instantiation named `adder3_inst`")));

  EXPECT_THAT(elab.GetInstances(block),
              UnorderedElementsAre(BlockInstanceFor(block)));
  EXPECT_THAT(elab.GetInstances(child_block),
              UnorderedElementsAre(InstantiationName("adder0_inst"),
                                   InstantiationName("adder1_inst"),
                                   InstantiationName("adder2_inst")));

  EXPECT_THAT(elab.GetUniqueInstance(block),
              IsOkAndHolds(BlockInstanceFor(block)));
  EXPECT_THAT(elab.GetUniqueInstance(child_block),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       Eq("There is not exactly 1 instance of `adder`, "
                          "instance count: 3")));
}

TEST_F(ElaborationTest, ElaborateFifoInstantiation) {
  auto p = CreatePackage();

  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("adder_with_fifo", p.get());
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);

  XLS_ASSERT_OK_AND_ASSIGN(FifoInstantiation * fifo_inst,
                           bb.block()->AddFifoInstantiation(
                               "fifo_inst",
                               FifoConfig(/*depth=*/1, /*bypass=*/true,
                                          /*register_push_outputs=*/true,
                                          /*register_pop_outputs=*/false),
                               u32));

  BValue lit1 = bb.Literal(UBits(1, 1));
  bb.InstantiationInput(fifo_inst, "push_data", a);
  bb.InstantiationInput(fifo_inst, "push_valid", lit1);
  bb.InstantiationInput(fifo_inst, "pop_ready", lit1);
  BValue pop_data = bb.InstantiationOutput(fifo_inst, "pop_data");
  bb.OutputPort("out", bb.Add(pop_data, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(block));

  EXPECT_EQ(elab.package(), p.get());
  EXPECT_THAT(elab.top(), BlockInstanceFor(block));
  EXPECT_EQ(elab.top()->instantiation(),
            std::nullopt);  // top is not an instantiation
  EXPECT_EQ(elab.top()->path().top, elab.top()->block());
  EXPECT_THAT(elab.top()->path().path, IsEmpty());
  EXPECT_THAT(elab.top()->child_instances(),
              UnorderedElementsAre(InstantiationName("fifo_inst")));
  EXPECT_THAT(elab.top()->instantiation_to_instance(),
              UnorderedElementsAre(Pair(m::Instantiation("fifo_inst"),
                                        InstantiationName("fifo_inst"))));
  EXPECT_THAT(elab.blocks(), UnorderedElementsAre(block));
  EXPECT_THAT(elab.instances(),
              UnorderedElementsAre(BlockInstanceFor(block),
                                   InstantiationName("fifo_inst")));
  EXPECT_THAT(elab.GetInstance("adder_with_fifo"),
              IsOkAndHolds(BlockInstanceFor(block)));
  ASSERT_THAT(elab.GetInstance("adder_with_fifo::fifo_inst->fifo"),
              IsOkAndHolds(InstantiationName("fifo_inst")));
  EXPECT_EQ(
      elab.GetInstance("adder_with_fifo::fifo_inst->fifo").value()->block(),
      std::nullopt);
  EXPECT_THAT(
      elab.GetInstance(
          "adder_with_fifo::fifo_inst->fifo::some_inst->some_block"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("is not a block and has no instantiation `some_inst`")));
}

class RecordingVisitor : public ElaboratedBlockDfsVisitorWithDefault {
 public:
  absl::Status DefaultHandler(
      const ElaboratedNode& node_and_instance) override {
    ordered_.push_back(node_and_instance);
    return absl::OkStatus();
  }
  absl::Span<ElaboratedNode const> ordered() const { return ordered_; }

 private:
  std::vector<ElaboratedNode> ordered_;
};

TEST_F(ElaborationTest, ElaborateFifoInstantiationNoBypassImposesNoOrder) {
  auto p = CreatePackage();

  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("adder_with_fifo", p.get());
  FifoConfig fifo_config(/*depth=*/1, /*bypass=*/false,
                         /*register_push_outputs=*/true,
                         /*register_pop_outputs=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fifo_config, u32));

  // Make pop side of FIFO before the push side. No-bypass FIFO will not impose
  // an order wrt to the FIFO, so the pop side will come out before the inputs.
  BValue pop_data = bb.InstantiationOutput(fifo_inst, "pop_data");
  BValue pop_valid = bb.InstantiationOutput(fifo_inst, "pop_valid");
  BValue push_ready = bb.InstantiationOutput(fifo_inst, "push_ready");
  bb.OutputPort("out", pop_data);
  bb.OutputPort("out_valid", pop_valid);
  bb.OutputPort("in_ready", push_ready);

  // Make push side.
  BValue a = bb.InputPort("a", u32);
  BValue lit1 = bb.Literal(UBits(1, 1));
  bb.InstantiationInput(fifo_inst, "push_data", a);
  bb.InstantiationInput(fifo_inst, "push_valid", lit1);
  bb.InstantiationInput(fifo_inst, "pop_ready", lit1);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(block));

  RecordingVisitor visitor;
  XLS_ASSERT_OK(elab.Accept(visitor));

  EXPECT_THAT(
      visitor.ordered(),
      ElementsAre(NodeIs(m::InstantiationOutput("pop_data")),
                  NodeIs(m::OutputPort("out")),
                  NodeIs(m::InstantiationOutput("pop_valid")),
                  NodeIs(m::OutputPort("out_valid")),
                  NodeIs(m::InstantiationOutput("push_ready")),
                  NodeIs(m::OutputPort("in_ready")), NodeIs(m::InputPort("a")),
                  NodeIs(m::InstantiationInput(_, "push_data")),
                  NodeIs(m::Literal()),
                  NodeIs(m::InstantiationInput(_, "push_valid")),
                  NodeIs(m::InstantiationInput(_, "pop_ready"))));
  EXPECT_THAT(ElaboratedTopoSort(elab),
              ElementsAre(NodeIs(m::InstantiationOutput("pop_data")),
                          NodeIs(m::InstantiationOutput("pop_valid")),
                          NodeIs(m::InstantiationOutput("push_ready")),
                          NodeIs(m::InputPort("a")), NodeIs(m::Literal()),
                          NodeIs(m::OutputPort("out")),
                          NodeIs(m::OutputPort("out_valid")),
                          NodeIs(m::OutputPort("in_ready")),
                          NodeIs(m::InstantiationInput(_, "push_data")),
                          NodeIs(m::InstantiationInput(_, "push_valid")),
                          NodeIs(m::InstantiationInput(_, "pop_ready"))));
}

TEST_F(ElaborationTest, ElaborateFifoInstantiationBypassImposesOrder) {
  auto p = CreatePackage();

  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb("adder_with_fifo", p.get());
  FifoConfig fifo_config(/*depth=*/1, /*bypass=*/true,
                         /*register_push_outputs=*/false,
                         /*register_pop_outputs=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(
      FifoInstantiation * fifo_inst,
      bb.block()->AddFifoInstantiation("fifo_inst", fifo_config, u32));

  // Make pop side of FIFO before the push side. No-bypass FIFO will not impose
  // an order wrt to the FIFO, so the pop side will come out before the inputs.
  BValue pop_data = bb.InstantiationOutput(fifo_inst, "pop_data");
  BValue pop_valid = bb.InstantiationOutput(fifo_inst, "pop_valid");
  BValue push_ready = bb.InstantiationOutput(fifo_inst, "push_ready");
  bb.OutputPort("out", pop_data);
  bb.OutputPort("out_valid", pop_valid);
  bb.OutputPort("in_ready", push_ready);

  // Make push side.
  BValue a = bb.InputPort("a", u32);
  BValue lit1 = bb.Literal(UBits(1, 1));
  bb.InstantiationInput(fifo_inst, "push_data", a);
  bb.InstantiationInput(fifo_inst, "push_valid", lit1);
  bb.InstantiationInput(fifo_inst, "pop_ready", lit1);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(block));

  RecordingVisitor visitor;
  XLS_ASSERT_OK(elab.Accept(visitor));

  EXPECT_THAT(visitor.ordered(),
              ElementsAre(NodeIs(m::InputPort("a")),
                          NodeIs(m::InstantiationInput(_, "push_data")),
                          NodeIs(m::Literal()),
                          NodeIs(m::InstantiationInput(_, "push_valid")),
                          NodeIs(m::InstantiationOutput("pop_data")),
                          NodeIs(m::OutputPort("out")),
                          NodeIs(m::InstantiationOutput("pop_valid")),
                          NodeIs(m::OutputPort("out_valid")),
                          NodeIs(m::InstantiationInput(_, "pop_ready")),
                          NodeIs(m::InstantiationOutput("push_ready")),
                          NodeIs(m::OutputPort("in_ready"))));

  EXPECT_THAT(ElaboratedTopoSort(elab),
              ElementsAre(NodeIs(m::Literal()), NodeIs(m::InputPort("a")),
                          NodeIs(m::InstantiationInput(_, "push_valid")),
                          NodeIs(m::InstantiationInput(_, "push_data")),
                          NodeIs(m::InstantiationInput(_, "pop_ready")),
                          NodeIs(m::InstantiationOutput("pop_data")),
                          NodeIs(m::InstantiationOutput("pop_valid")),
                          NodeIs(m::InstantiationOutput("push_ready")),
                          NodeIs(m::OutputPort("out")),
                          NodeIs(m::OutputPort("out_valid")),
                          NodeIs(m::OutputPort("in_ready"))));
}

TEST_F(ElaborationTest, TopoSortSingleInstantiation) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, AddBlock(*p));

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(block));

  EXPECT_THAT(ElaboratedTopoSort(elab),
              ElementsAre(NodeIs(m::InputPort("a")), NodeIs(m::InputPort("b")),
                          NodeIs(m::Add()), NodeIs(m::OutputPort("c"))));
}

TEST_F(ElaborationTest, TopoSortMultipleInstantiations) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, MultipleAddInstantiations(*p));

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(block));

  EXPECT_THAT(
      ElaboratedTopoSort(elab),
      ElementsAre(
          NodeAndInst(m::InputPort("a"), HasSubstr("multi_adder ")),
          NodeAndInst(m::InputPort("b"), HasSubstr("multi_adder ")),
          NodeAndInst(m::InputPort("c"), HasSubstr("multi_adder ")),
          NodeAndInst(m::InputPort("d"), HasSubstr("multi_adder ")),
          NodeAndInst(m::InstantiationInput(m::InputPort("a"), "a"),
                      HasSubstr("multi_adder ")),
          NodeAndInst(m::InstantiationInput(m::InputPort("b"), "b"),
                      HasSubstr("multi_adder ")),
          NodeAndInst(m::InstantiationInput(m::InputPort("c"), "a"),
                      HasSubstr("multi_adder ")),
          NodeAndInst(m::InstantiationInput(m::InputPort("d"), "b"),
                      HasSubstr("multi_adder ")),
          NodeAndInst(m::InputPort("a"), HasSubstr("adder0_inst->adder")),
          NodeAndInst(m::InputPort("b"), HasSubstr("adder0_inst->adder")),
          NodeAndInst(m::InputPort("a"), HasSubstr("adder1_inst->adder")),
          NodeAndInst(m::InputPort("b"), HasSubstr("adder1_inst->adder")),
          NodeAndInst(m::Add(), HasSubstr("adder0_inst->adder")),
          NodeAndInst(m::Add(), HasSubstr("adder1_inst->adder")),
          NodeAndInst(m::OutputPort("c"), HasSubstr("adder0_inst->adder")),
          NodeAndInst(m::OutputPort("c"), HasSubstr("adder1_inst->adder")),
          NodeAndInst(m::InstantiationOutput("c"), HasSubstr("multi_adder ")),
          NodeAndInst(m::InstantiationOutput("c"), HasSubstr("multi_adder ")),
          NodeAndInst(m::InstantiationInput(m::InstantiationOutput(), "a"),
                      HasSubstr("multi_adder ")),
          NodeAndInst(m::InstantiationInput(m::InstantiationOutput(), "b"),
                      HasSubstr("multi_adder ")),
          NodeAndInst(m::InputPort("a"), HasSubstr("adder2_inst->adder")),
          NodeAndInst(m::InputPort("b"), HasSubstr("adder2_inst->adder")),
          NodeAndInst(m::Add(), HasSubstr("adder2_inst->adder")),
          NodeAndInst(m::OutputPort("c"), HasSubstr("adder2_inst->adder")),
          NodeAndInst(m::InstantiationOutput("c"), HasSubstr("multi_adder ")),
          NodeAndInst(m::OutputPort("out"), HasSubstr("multi_adder "))));
}

// Note that the topo sort on FunctionBase is intended to have the same order
// (modulo instantiations) as this topo sort. Tests should be duplicated here
// and there to the extend that it is possible.
//
// LINT.IfChange
TEST(NodeIteratorTest, ReordersViaDependencies) {
  Package p("p");
  Block f("f", &p);
  SourceInfo loc;
  XLS_ASSERT_OK_AND_ASSIGN(Node * literal,
                           f.MakeNode<Literal>(loc, Value(UBits(3, 2))));
  XLS_ASSERT_OK_AND_ASSIGN(Node * neg,
                           f.MakeNode<UnOp>(loc, literal, Op::kNeg));

  XLS_ASSERT_OK_AND_ASSIGN(Node * out, f.MakeNode<OutputPort>(loc, neg));

  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(&f));

  // Literal should precede the negation in RPO although we added those nodes in
  // the opposite order.
  std::vector<ElaboratedNode> rni = ElaboratedTopoSort(elab);
  auto it = rni.begin();
  EXPECT_EQ(it->node, literal);
  ++it;
  EXPECT_EQ(it->node, neg);
  ++it;
  EXPECT_EQ(it->node, out);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

TEST(NodeIteratorTest, Diamond) {
  constexpr std::string_view program = R"(
  block diamond(x: bits[32], y: bits[32]) {
    x: bits[32] = input_port(name=x)
    neg.2: bits[32] = neg(x)
    neg.3: bits[32] = neg(x)
    add.4: bits[32] = add(neg.2, neg.3)
    y: () = output_port(add.4, name=y)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Block * f, Parser::ParseBlock(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(f));

  std::vector<ElaboratedNode> rni = ElaboratedTopoSort(elab);
  auto it = rni.begin();
  EXPECT_EQ(it->node->GetName(), "x");
  ++it;
  EXPECT_EQ(it->node->GetName(), "neg.2");
  ++it;
  EXPECT_EQ(it->node->GetName(), "neg.3");
  ++it;
  EXPECT_EQ(it->node->GetName(), "add.4");
  ++it;
  EXPECT_EQ(it->node->GetName(), "y");
  ++it;
  EXPECT_EQ(rni.end(), it);
}

// Constructs a test as follows:
//
//        A
//      /   \
//      \    B
//       \  /
//        \/
//         C
//
// Topological order: A B C
TEST(NodeIteratorTest, PostOrderNotPreOrder) {
  Package p("p");
  Block f("f", &p);
  SourceInfo loc;
  XLS_ASSERT_OK_AND_ASSIGN(Node * a,
                           f.MakeNode<Literal>(loc, Value(UBits(0, 2))));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b, f.MakeNode<BinOp>(loc, a, a, Op::kAdd));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c, f.MakeNode<BinOp>(loc, a, b, Op::kAdd));

  XLS_ASSERT_OK_AND_ASSIGN(Node * output, f.MakeNode<OutputPort>(loc, c));
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(&f));

  std::vector<ElaboratedNode> rni = ElaboratedTopoSort(elab);
  auto it = rni.begin();
  EXPECT_EQ(it->node, a);
  ++it;
  EXPECT_EQ(it->node, b);
  ++it;
  EXPECT_EQ(it->node, c);
  ++it;
  EXPECT_EQ(it->node, output);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

// Constructs a test as follows:
//
//         A --
//        / \  \
//        | |   \
//        \ /   |
//         B    C
//          \  /
//            D
//
// Topo: D B C A =(reverse)=> A C B D
//                              2 1 3
TEST(NodeIteratorTest, TwoOfSameOperandLinks) {
  constexpr std::string_view program = R"(
  block computation(a: bits[32], e: bits[32]) {
    a: bits[32] = input_port(name=a)
    b: bits[32] = add(a, a)
    c: bits[32] = neg(a)
    d: bits[32] = add(b, c)
    e: () = output_port(d, name=e)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Block * f, Parser::ParseBlock(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(f));

  std::vector<ElaboratedNode> rni = ElaboratedTopoSort(elab);
  auto it = rni.begin();
  EXPECT_EQ(it->node->GetName(), "a");
  ++it;
  EXPECT_EQ(it->node->GetName(), "b");
  ++it;
  EXPECT_EQ(it->node->GetName(), "c");
  ++it;
  EXPECT_EQ(it->node->GetName(), "d");
  ++it;
  EXPECT_EQ(it->node->GetName(), "e");
  ++it;
  EXPECT_EQ(rni.end(), it);
}

TEST(NodeIteratorTest, UselessParamsUnrelatedReturn) {
  constexpr std::string_view program = R"(
  block computation(a: bits[32], b: bits[32], c: bits[32]) {
    a: bits[32] = input_port(name=a)
    b: bits[32] = input_port(name=b)
    r: bits[32] = literal(value=2)
    c: () = output_port(r, name=c)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Block * f, Parser::ParseBlock(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(f));

  std::vector<ElaboratedNode> rni = ElaboratedTopoSort(elab);
  auto it = rni.begin();
  // Note that this order differs from topo_sort_test.cc- this is because blocks
  // input/outputs and function param/retvals are different.
  EXPECT_EQ(it->node->GetName(), "r");
  ++it;
  EXPECT_EQ(it->node->GetName(), "a");
  ++it;
  EXPECT_EQ(it->node->GetName(), "b");
  ++it;
  EXPECT_EQ(it->node->GetName(), "c");
  ++it;
  EXPECT_EQ(rni.end(), it);
}

// Constructs a test as follows:
//
//      A
//     / \
//    T   C
//     \ / \
//      B   E
//       \ /
//        D
TEST(NodeIteratorTest, ExtendedDiamond) {
  constexpr std::string_view program = R"(
  block computation(a: bits[32], f: bits[32]) {
    a: bits[32] = input_port(name=a)
    t: bits[32] = neg(a)
    c: bits[32] = neg(a)
    b: bits[32] = add(t, c)
    e: bits[32] = neg(c)
    d: bits[32] = add(b, e)
    f: () = output_port(d, name=f)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Block * f, Parser::ParseBlock(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(f));

  std::vector<ElaboratedNode> rni = ElaboratedTopoSort(elab);
  auto it = rni.begin();
  EXPECT_EQ(it->node->GetName(), "a");
  ++it;
  EXPECT_EQ(it->node->GetName(), "t");
  ++it;
  EXPECT_EQ(it->node->GetName(), "c");
  ++it;
  EXPECT_EQ(it->node->GetName(), "b");
  ++it;
  EXPECT_EQ(it->node->GetName(), "e");
  ++it;
  EXPECT_EQ(it->node->GetName(), "d");
  ++it;
  EXPECT_EQ(it->node->GetName(), "f");
  ++it;
  EXPECT_EQ(rni.end(), it);
}

TEST(NodeIteratorTest, ExtendedDiamondReverse) {
  constexpr std::string_view program = R"(
  block computation(a: bits[32], f: bits[32]) {
    a: bits[32] = input_port(name=a)
    t: bits[32] = neg(a)
    c: bits[32] = neg(a)
    b: bits[32] = add(t, c)
    e: bits[32] = neg(c)
    d: bits[32] = add(b, e)
    f: () = output_port(d, name=f)
  })";
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Block * f, Parser::ParseBlock(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(f));

  // ReverseTopoSort should produce the same order but in reverse.
  std::vector<ElaboratedNode> fwd_it = ElaboratedTopoSort(elab);
  std::vector<ElaboratedNode> rev_it = ElaboratedReverseTopoSort(elab);
  std::vector<ElaboratedNode> fwd(fwd_it.begin(), fwd_it.end());
  std::vector<ElaboratedNode> rev(rev_it.begin(), rev_it.end());
  std::reverse(fwd.begin(), fwd.end());
  EXPECT_EQ(fwd, rev);
}

// Constructs a test as follows:
//
//      D
//      | \
//      C  \
//      |   \
//      B    T
//       \  /
//        \/
//         A
//
// Given that we know we visit operands in left-to-right order, this example
// points out the discrepancy between the RPO ordering and what our algorithm
// produces. The depth-first traversal RPO necessitates would have us visit the
// whole D,C,B chain before T.
//
// Post-Order:     D C B T A =(rev)=> A T B C D
//                                      1 2 3 4
// Our topo order: D T C B A =(rev)=> A B C T D
//                                      2 3 1 4
TEST(NodeIteratorTest, RpoVsTopo) {
  constexpr std::string_view program = R"(
  block computation(a: bits[32], e: bits[32]) {
    a: bits[32] = input_port(name=a)
    t: bits[32] = neg(a)
    b: bits[32] = neg(a)
    c: bits[32] = neg(b)
    d: bits[32] = add(c, t)
    e: () = output_port(d, name=e)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Block * f, Parser::ParseBlock(program, &p));
  XLS_ASSERT_OK_AND_ASSIGN(BlockElaboration elab,
                           BlockElaboration::Elaborate(f));

  std::vector<ElaboratedNode> rni = ElaboratedTopoSort(elab);
  auto it = rni.begin();
  EXPECT_EQ(it->node->GetName(), "a");
  ++it;
  EXPECT_EQ(it->node->GetName(), "b");
  ++it;
  EXPECT_EQ(it->node->GetName(), "c");
  ++it;
  EXPECT_EQ(it->node->GetName(), "t");
  ++it;
  EXPECT_EQ(it->node->GetName(), "d");
  ++it;
  EXPECT_EQ(it->node->GetName(), "e");
  ++it;
  EXPECT_EQ(rni.end(), it);
}

// LINT.ThenChange(//xls/ir/topo_sort_test.cc)

}  // namespace
}  // namespace xls
