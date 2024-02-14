// Copyright 2021 The XLS Authors
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

#include "xls/ir/block.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class BlockTest : public IrTestBase {
 protected:
  // Returns the names of the given nodes as a vector of strings. Templated to
  // enable accepting spans of Node derived types.
  template <typename NodeT>
  std::vector<std::string> NodeNames(absl::Span<NodeT* const> nodes) {
    std::vector<std::string> names;
    for (Node* node : nodes) {
      names.push_back(node->GetName());
    }
    return names;
  }

  std::vector<std::string> GetPortNames(Block* block) {
    std::vector<std::string> names;
    for (const Block::Port& port : block->GetPorts()) {
      names.push_back(
          absl::visit(Visitor{[](InputPort* p) { return p->GetName(); },
                              [](OutputPort* p) { return p->GetName(); },
                              [](Block::ClockPort* p) { return p->name; }},
                      port));
    }
    return names;
  }

  std::vector<std::string> GetInputPortNames(Block* block) {
    return NodeNames(block->GetInputPorts());
  }

  std::vector<std::string> GetOutputPortNames(Block* block) {
    return NodeNames(block->GetOutputPorts());
  }
};

TEST_F(BlockTest, SimpleBlock) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue out = bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_FALSE(block->IsFunction());
  EXPECT_FALSE(block->IsProc());
  EXPECT_TRUE(block->IsBlock());
  EXPECT_THAT(GetPortNames(block), ElementsAre("a", "b", "out"));

  EXPECT_THAT(block->GetInputPorts(), ElementsAre(a.node(), b.node()));
  EXPECT_THAT(GetInputPortNames(block), ElementsAre("a", "b"));

  EXPECT_THAT(block->GetOutputPorts(), ElementsAre(out.node()));
  EXPECT_THAT(GetOutputPortNames(block), ElementsAre("out"));

  EXPECT_EQ(block->DumpIr(),
            R"(block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  add.3: bits[32] = add(a, b, id=3)
  out: () = output_port(add.3, name=out, id=4)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(clone->DumpIr(),
            R"(block cloned(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}
)");
}

TEST_F(BlockTest, RemovePorts) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue out = bb.OutputPort("out", bb.Literal(Value(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(GetInputPortNames(block), ElementsAre("a", "b"));
  EXPECT_THAT(GetOutputPortNames(block), ElementsAre("out"));
  EXPECT_THAT(GetPortNames(block), ElementsAre("a", "b", "out"));

  XLS_ASSERT_OK(block->RemoveNode(b.node()));
  EXPECT_THAT(GetInputPortNames(block), ElementsAre("a"));
  EXPECT_THAT(GetPortNames(block), ElementsAre("a", "out"));

  XLS_ASSERT_OK(block->RemoveNode(a.node()));
  EXPECT_THAT(GetInputPortNames(block), ElementsAre());
  EXPECT_THAT(GetPortNames(block), ElementsAre("out"));

  XLS_ASSERT_OK(block->RemoveNode(out.node()));
  EXPECT_THAT(GetOutputPortNames(block), ElementsAre());
  EXPECT_THAT(GetPortNames(block), ElementsAre());
}

TEST_F(BlockTest, PortOrder) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("out", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
  XLS_ASSERT_OK(block->AddClockPort("clk"));

  EXPECT_THAT(
      block->DumpIr(),
      HasSubstr("block my_block(in: bits[32], out: bits[32], clk: clock)"));

  XLS_ASSERT_OK(block->ReorderPorts({"in", "clk", "out"}));

  EXPECT_THAT(
      block->DumpIr(),
      HasSubstr("block my_block(in: bits[32], clk: clock, out: bits[32])"));

  EXPECT_THAT(block->ReorderPorts({"in", "out"}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Port order missing port \"clk\"")));
  EXPECT_THAT(block->ReorderPorts({"in", "out", "clk", "in"}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Port order has duplicate names")));
  EXPECT_THAT(block->ReorderPorts({"in", "out", "clk", "blarg"}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Port order includes invalid port names")));
}

TEST_F(BlockTest, MultiOutputBlock) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue a_out = bb.OutputPort("a_out", a);
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue b_out = bb.OutputPort("b_out", b);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(GetPortNames(block), ElementsAre("a", "a_out", "b", "b_out"));

  EXPECT_THAT(block->GetInputPorts(), ElementsAre(a.node(), b.node()));
  EXPECT_THAT(GetInputPortNames(block), ElementsAre("a", "b"));

  EXPECT_THAT(block->GetOutputPorts(), ElementsAre(a_out.node(), b_out.node()));
  EXPECT_THAT(GetOutputPortNames(block), ElementsAre("a_out", "b_out"));

  EXPECT_EQ(
      block->DumpIr(),
      R"(block my_block(a: bits[32], a_out: bits[32], b: bits[32], b_out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=3)
  a_out: () = output_port(a, name=a_out, id=2)
  b_out: () = output_port(b, name=b_out, id=4)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(
      clone->DumpIr(),
      R"(block cloned(a: bits[32], a_out: bits[32], b: bits[32], b_out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  a_out: () = output_port(a, name=a_out, id=7)
  b_out: () = output_port(b, name=b_out, id=8)
}
)");
}

TEST_F(BlockTest, ErrorConditions) {
  // Don't use CreatePackage because that creates a package which verifies on
  // test completion and the errors may leave the block in a invalid state.
  Package p(TestName());
  BlockBuilder bb("my_block", &p);
  Type* u32 = p.GetBitsType(32);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("out", bb.Add(a, b, SourceInfo(), /*name=*/"foo"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Add a port with a name that already exists.
  EXPECT_THAT(
      block->AddInputPort("b", u32),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Block my_block already contains a port named b")));
  EXPECT_THAT(
      block->AddOutputPort("b", b.node()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Block my_block already contains a port named b")));

  // It should be ok to add a port named `foo` even though there is a non-port
  // node already named `foo`. The non-port node will be renamed.
  Node* orig_foo = FindNode("foo", block);
  EXPECT_EQ(orig_foo->GetName(), "foo");
  XLS_ASSERT_OK(block->AddInputPort("foo", u32));

  // The node named `foo` should now be an input port and the node previously
  // known as `foo` should have a different name.
  EXPECT_TRUE(FindNode("foo", block)->Is<InputPort>());
  EXPECT_NE(orig_foo->GetName(), "foo");
  EXPECT_THAT(orig_foo->GetName(), testing::StartsWith("foo_"));
}

TEST_F(BlockTest, TrivialBlock) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_EQ(block->DumpIr(), R"(block my_block() {
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(clone->DumpIr(), R"(block cloned() {
}
)");
}

TEST_F(BlockTest, BlockWithRegisters) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(Register * state_reg,
                           bb.block()->AddRegister("state", u32));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));

  BValue state = bb.RegisterRead(state_reg, SourceInfo(), "state");
  BValue x = bb.InputPort("x", u32);

  BValue x_d = bb.InsertRegister("x_d", x);
  BValue sum = bb.Add(x_d, state, SourceInfo(), "sum");

  BValue sum_d = bb.InsertRegister("sum_d", sum);
  BValue out = bb.OutputPort("out", sum_d);

  bb.RegisterWrite(state_reg, sum_d,
                   /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
                   SourceInfo(), "state_write");

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_FALSE(block->IsFunction());
  EXPECT_FALSE(block->IsProc());
  EXPECT_TRUE(block->IsBlock());
  EXPECT_THAT(GetPortNames(block), ElementsAre("clk", "x", "out"));

  EXPECT_THAT(block->GetInputPorts(), ElementsAre(x.node()));
  EXPECT_THAT(GetInputPortNames(block), ElementsAre("x"));

  EXPECT_THAT(block->GetOutputPorts(), ElementsAre(out.node()));
  EXPECT_THAT(GetOutputPortNames(block), ElementsAre("out"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * x_port, block->GetInputPort("x"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * out_port, block->GetOutputPort("out"));
  EXPECT_EQ(x_port, x.node());
  EXPECT_EQ(out_port, out.node());

  ASSERT_TRUE(block->GetClockPort().has_value());
  EXPECT_THAT(block->GetClockPort()->name, "clk");

  XLS_VLOG_LINES(1, block->DumpIr());
  EXPECT_EQ(block->DumpIr(),
            R"(block my_block(clk: clock, x: bits[32], out: bits[32]) {
  reg state(bits[32])
  reg x_d(bits[32])
  reg sum_d(bits[32])
  x: bits[32] = input_port(name=x, id=2)
  x_d_write: () = register_write(x, register=x_d, id=3)
  state: bits[32] = register_read(register=state, id=1)
  x_d: bits[32] = register_read(register=x_d, id=4)
  sum: bits[32] = add(x_d, state, id=5)
  sum_d_write: () = register_write(sum, register=sum_d, id=6)
  sum_d: bits[32] = register_read(register=sum_d, id=7)
  state_write: () = register_write(sum_d, register=state, id=9)
  out: () = output_port(sum_d, name=out, id=8)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(clone->DumpIr(),
            R"(block cloned(clk: clock, x: bits[32], out: bits[32]) {
  reg state(bits[32])
  reg x_d(bits[32])
  reg sum_d(bits[32])
  x: bits[32] = input_port(name=x, id=12)
  x_d_write: () = register_write(x, register=x_d, id=15)
  x_d: bits[32] = register_read(register=x_d, id=10)
  state: bits[32] = register_read(register=state, id=11)
  sum: bits[32] = add(x_d, state, id=13)
  sum_d_write: () = register_write(sum, register=sum_d, id=16)
  sum_d: bits[32] = register_read(register=sum_d, id=14)
  state_write: () = register_write(sum_d, register=state, id=18)
  out: () = output_port(sum_d, name=out, id=17)
}
)");
}

TEST_F(BlockTest, BlockClone) {
  auto p = CreatePackage();
  std::unique_ptr<Package> p2 =
      std::make_unique<VerifiedPackage>("second_package");
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(Register * state_reg,
                           bb.block()->AddRegister("state", u32));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));

  BValue state = bb.RegisterRead(state_reg, SourceInfo(), "state");
  BValue x = bb.InputPort("x", u32);

  BValue x_d = bb.InsertRegister("x_d", x);
  BValue sum = bb.Add(x_d, state, SourceInfo(), "sum");

  BValue sum_d = bb.InsertRegister("sum_d", sum);
  BValue out = bb.OutputPort("out", sum_d);

  bb.RegisterWrite(state_reg, sum_d,
                   /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
                   SourceInfo(), "state_write");

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_FALSE(block->IsFunction());
  EXPECT_FALSE(block->IsProc());
  EXPECT_TRUE(block->IsBlock());
  EXPECT_THAT(GetPortNames(block), ElementsAre("clk", "x", "out"));

  EXPECT_THAT(block->GetInputPorts(), ElementsAre(x.node()));
  EXPECT_THAT(GetInputPortNames(block), ElementsAre("x"));

  EXPECT_THAT(block->GetOutputPorts(), ElementsAre(out.node()));
  EXPECT_THAT(GetOutputPortNames(block), ElementsAre("out"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * x_port, block->GetInputPort("x"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * out_port, block->GetOutputPort("out"));
  EXPECT_EQ(x_port, x.node());
  EXPECT_EQ(out_port, out.node());

  ASSERT_TRUE(block->GetClockPort().has_value());
  EXPECT_THAT(block->GetClockPort()->name, "clk");

  XLS_VLOG_LINES(1, block->DumpIr());
  EXPECT_EQ(block->DumpIr(),
            R"(block my_block(clk: clock, x: bits[32], out: bits[32]) {
  reg state(bits[32])
  reg x_d(bits[32])
  reg sum_d(bits[32])
  x: bits[32] = input_port(name=x, id=2)
  x_d_write: () = register_write(x, register=x_d, id=3)
  state: bits[32] = register_read(register=state, id=1)
  x_d: bits[32] = register_read(register=x_d, id=4)
  sum: bits[32] = add(x_d, state, id=5)
  sum_d_write: () = register_write(sum, register=sum_d, id=6)
  sum_d: bits[32] = register_read(register=sum_d, id=7)
  state_write: () = register_write(sum_d, register=state, id=9)
  out: () = output_port(sum_d, name=out, id=8)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(
      Block * clone,
      block->Clone("cloned", p2.get(),
                   {{"sum_d", "not_a_sum_d"}, {"state", "not_a_state"}}));

  EXPECT_EQ(clone->DumpIr(),
            R"(block cloned(clk: clock, x: bits[32], out: bits[32]) {
  reg not_a_state(bits[32])
  reg x_d(bits[32])
  reg not_a_sum_d(bits[32])
  x: bits[32] = input_port(name=x, id=3)
  x_d_write: () = register_write(x, register=x_d, id=6)
  x_d: bits[32] = register_read(register=x_d, id=1)
  state: bits[32] = register_read(register=not_a_state, id=2)
  sum: bits[32] = add(x_d, state, id=4)
  sum_d_write: () = register_write(sum, register=not_a_sum_d, id=7)
  sum_d: bits[32] = register_read(register=not_a_sum_d, id=5)
  state_write: () = register_write(sum_d, register=not_a_state, id=9)
  out: () = output_port(sum_d, name=out, id=8)
}
)");
}

TEST_F(BlockTest, BlockWithTrivialFeedbackDumpOrderTest) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(Register * x_reg, bb.block()->AddRegister("x", u32));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));

  bb.RegisterWrite(x_reg, bb.RegisterRead(x_reg));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_VLOG_LINES(1, block->DumpIr());
  EXPECT_EQ(block->DumpIr(),
            R"(block my_block(clk: clock) {
  reg x(bits[32])
  register_read.1: bits[32] = register_read(register=x, id=1)
  register_write.2: () = register_write(register_read.1, register=x, id=2)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(clone->DumpIr(),
            R"(block cloned(clk: clock) {
  reg x(bits[32])
  register_read_1: bits[32] = register_read(register=x, id=3)
  register_write_2: () = register_write(register_read_1, register=x, id=4)
}
)");
}

TEST_F(BlockTest, MultipleInputsAndOutputsDumpOrderTest) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);

  // Intentially create the input ports in the wrong order then order them
  // later. This makes the id order not match the port order for a better test.
  bb.InputPort("y", u32);
  bb.InputPort("z", u32);
  BValue x = bb.InputPort("x", u32);
  bb.OutputPort("b", x);
  bb.OutputPort("a", x);
  bb.OutputPort("c", x);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->ReorderPorts({"x", "y", "z", "a", "b", "c"}));

  XLS_VLOG_LINES(1, block->DumpIr());
  EXPECT_EQ(
      block->DumpIr(),
      R"(block my_block(x: bits[32], y: bits[32], z: bits[32], a: bits[32], b: bits[32], c: bits[32]) {
  x: bits[32] = input_port(name=x, id=3)
  y: bits[32] = input_port(name=y, id=1)
  z: bits[32] = input_port(name=z, id=2)
  a: () = output_port(x, name=a, id=5)
  b: () = output_port(x, name=b, id=4)
  c: () = output_port(x, name=c, id=6)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(
      clone->DumpIr(),
      R"(block cloned(x: bits[32], y: bits[32], z: bits[32], a: bits[32], b: bits[32], c: bits[32]) {
  x: bits[32] = input_port(name=x, id=7)
  y: bits[32] = input_port(name=y, id=8)
  z: bits[32] = input_port(name=z, id=9)
  a: () = output_port(x, name=a, id=11)
  b: () = output_port(x, name=b, id=10)
  c: () = output_port(x, name=c, id=12)
}
)");
}

TEST_F(BlockTest, DeepPipelineDumpOrderTest) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue x = bb.InputPort("x", u32);
  BValue y = bb.InputPort("y", u32);

  BValue p1_x = bb.InsertRegister("p1_x", x);
  BValue p1_y = bb.InsertRegister("p1_y", y);
  BValue p1_add = bb.Add(p1_x, p1_y, SourceInfo(), "p1_add");

  BValue p2_add = bb.InsertRegister("p2_add", p1_add);
  BValue p2_y = bb.InsertRegister("p2_y", p1_y);
  BValue p2_sub = bb.Subtract(p2_add, p2_y, SourceInfo(), "p2_sub");

  BValue p3_sub = bb.InsertRegister("p3_sub", p2_sub);
  BValue p3_zero = bb.Literal(UBits(0, 32), SourceInfo(), "p3_zero");
  BValue p3_mul = bb.UMul(p3_sub, p3_zero, SourceInfo(), "p3_mul");

  bb.OutputPort("result", p3_mul);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_VLOG_LINES(1, block->DumpIr());
  EXPECT_EQ(
      block->DumpIr(),
      R"(block my_block(clk: clock, x: bits[32], y: bits[32], result: bits[32]) {
  reg p1_x(bits[32])
  reg p1_y(bits[32])
  reg p2_add(bits[32])
  reg p2_y(bits[32])
  reg p3_sub(bits[32])
  x: bits[32] = input_port(name=x, id=1)
  y: bits[32] = input_port(name=y, id=2)
  p1_x_write: () = register_write(x, register=p1_x, id=3)
  p1_y_write: () = register_write(y, register=p1_y, id=5)
  p1_x: bits[32] = register_read(register=p1_x, id=4)
  p1_y: bits[32] = register_read(register=p1_y, id=6)
  p1_add: bits[32] = add(p1_x, p1_y, id=7)
  p2_add_write: () = register_write(p1_add, register=p2_add, id=8)
  p2_y_write: () = register_write(p1_y, register=p2_y, id=10)
  p2_add: bits[32] = register_read(register=p2_add, id=9)
  p2_y: bits[32] = register_read(register=p2_y, id=11)
  p2_sub: bits[32] = sub(p2_add, p2_y, id=12)
  p3_sub_write: () = register_write(p2_sub, register=p3_sub, id=13)
  p3_sub: bits[32] = register_read(register=p3_sub, id=14)
  p3_zero: bits[32] = literal(value=0, id=15)
  p3_mul: bits[32] = umul(p3_sub, p3_zero, id=16)
  result: () = output_port(p3_mul, name=result, id=17)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(
      clone->DumpIr(),
      R"(block cloned(clk: clock, x: bits[32], y: bits[32], result: bits[32]) {
  reg p1_x(bits[32])
  reg p1_y(bits[32])
  reg p2_add(bits[32])
  reg p2_y(bits[32])
  reg p3_sub(bits[32])
  x: bits[32] = input_port(name=x, id=24)
  y: bits[32] = input_port(name=y, id=25)
  p1_x_write: () = register_write(x, register=p1_x, id=29)
  p1_y_write: () = register_write(y, register=p1_y, id=30)
  p1_x: bits[32] = register_read(register=p1_x, id=18)
  p1_y: bits[32] = register_read(register=p1_y, id=19)
  p1_add: bits[32] = add(p1_x, p1_y, id=26)
  p2_add_write: () = register_write(p1_add, register=p2_add, id=31)
  p2_y_write: () = register_write(p1_y, register=p2_y, id=32)
  p2_add: bits[32] = register_read(register=p2_add, id=20)
  p2_y: bits[32] = register_read(register=p2_y, id=21)
  p2_sub: bits[32] = sub(p2_add, p2_y, id=27)
  p3_sub_write: () = register_write(p2_sub, register=p3_sub, id=33)
  p3_sub: bits[32] = register_read(register=p3_sub, id=22)
  p3_zero: bits[32] = literal(value=0, id=23)
  p3_mul: bits[32] = umul(p3_sub, p3_zero, id=28)
  result: () = output_port(p3_mul, name=result, id=34)
}
)");
}

TEST_F(BlockTest, RemoveRegisters) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg,
                           bb.block()->AddRegister("my_reg", u32));
  BValue sum = bb.Add(a, b);
  BValue sum_q = bb.RegisterWrite(reg, sum);
  BValue sum_d = bb.RegisterRead(reg);
  bb.OutputPort("out", sum_d);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_EQ(block->GetRegisters().size(), 1);
  EXPECT_EQ(block->GetRegisters()[0], reg);
  EXPECT_THAT(block->GetRegister("my_reg"), IsOkAndHolds(reg));

  XLS_ASSERT_OK(sum_d.node()->ReplaceUsesWith(sum.node()));
  XLS_ASSERT_OK(block->RemoveNode(sum_q.node()));
  XLS_ASSERT_OK(block->RemoveNode(sum_d.node()));
  XLS_ASSERT_OK(block->RemoveRegister(reg));

  EXPECT_TRUE(block->GetRegisters().empty());
  EXPECT_THAT(
      block->GetRegister("my_reg").status(),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("Block my_block has no register named my_reg")));
}

TEST_F(BlockTest, RegisterWithInvalidResetValue) {
  auto p = CreatePackage();
  Block* blk = p->AddBlock(std::make_unique<Block>("block1", p.get()));
  EXPECT_THAT(
      blk->AddRegister("my_reg", p->GetBitsType(32),
                       Reset{.reset_value = Value(UBits(0, 8)),
                             .asynchronous = false,
                             .active_low = false})
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Reset value bits[8]:0 for register my_reg is not of "
                         "type bits[32]")));
}

TEST_F(BlockTest, AddDuplicateRegisters) {
  // Don't use CreatePackage because that creates a package which verifies on
  // test completion and the errors may leave the block in a invalid state.
  Package p(TestName());
  Block* blk = p.AddBlock(std::make_unique<Block>("block1", &p));
  XLS_ASSERT_OK(blk->AddClockPort("clk"));
  static constexpr std::string_view kRegName = "my_reg";
  XLS_ASSERT_OK(blk->AddRegister(kRegName, p.GetBitsType(32)).status());
  // Adding a duplicate will succeed but the duplicated register will have a
  // different name.
  XLS_ASSERT_OK_AND_ASSIGN(Register * r,
                           blk->AddRegister(kRegName, p.GetBitsType(32)));
  EXPECT_THAT(r->name(), testing::Not(testing::StrEq(kRegName)));
  EXPECT_THAT(r->name(), testing::StartsWith(kRegName));
}

TEST_F(BlockTest, RemoveRegisterNotOwnedByBlock) {
  // Don't use CreatePackage because that creates a package which verifies on
  // test completion and the errors may leave the block in a invalid state.
  Package p(TestName());
  Block* blk1 = p.AddBlock(std::make_unique<Block>("block1", &p));
  XLS_ASSERT_OK(blk1->AddClockPort("clk"));
  Block* blk2 = p.AddBlock(std::make_unique<Block>("block2", &p));
  XLS_ASSERT_OK(blk2->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg1,
                           blk1->AddRegister("reg", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg2,
                           blk2->AddRegister("reg", p.GetBitsType(32)));

  EXPECT_TRUE(blk1->IsOwned(reg1));
  EXPECT_FALSE(blk2->IsOwned(reg1));
  EXPECT_FALSE(blk1->IsOwned(reg2));
  EXPECT_TRUE(blk2->IsOwned(reg2));

  EXPECT_THAT(blk2->RemoveRegister(reg1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Register is not owned by block")));
}

TEST_F(BlockTest, BlockRegisterNodes) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);

  XLS_ASSERT_OK_AND_ASSIGN(Register * a_reg,
                           bb.block()->AddRegister("a_reg", u32));
  BValue a_write = bb.RegisterWrite(a_reg, a);
  BValue a_read = bb.RegisterRead(a_reg);

  XLS_ASSERT_OK_AND_ASSIGN(Register * b_reg,
                           bb.block()->AddRegister("b_reg", u32));
  BValue b_write = bb.RegisterWrite(b_reg, b);
  BValue b_read = bb.RegisterRead(b_reg);

  bb.OutputPort("out", bb.Add(a_read, b_read));

  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(block->GetRegisterRead(a_reg), IsOkAndHolds(a_read.node()));
  EXPECT_THAT(block->GetRegisterWrite(a_reg), IsOkAndHolds(a_write.node()));

  EXPECT_THAT(block->GetRegisterRead(b_reg), IsOkAndHolds(b_read.node()));
  EXPECT_THAT(block->GetRegisterWrite(b_reg), IsOkAndHolds(b_write.node()));
}

TEST_F(BlockTest, GetRegisterReadWrite) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = bb.InputPort("a", u32);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Register * reg, block->AddRegister("a_reg", u32));

  EXPECT_THAT(
      block->GetRegisterRead(reg),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Block my_block has no read operation for register a_reg")));
  EXPECT_THAT(
      block->GetRegisterWrite(reg),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Block my_block has no write operation for register a_reg")));

  XLS_ASSERT_OK_AND_ASSIGN(RegisterRead * reg_read,
                           block->MakeNode<RegisterRead>(SourceInfo(), reg));

  EXPECT_THAT(block->GetRegisterRead(reg), IsOkAndHolds(reg_read));
  EXPECT_THAT(
      block->GetRegisterWrite(reg),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Block my_block has no write operation for register a_reg")));

  XLS_ASSERT_OK_AND_ASSIGN(
      RegisterWrite * reg_write,
      block->MakeNode<RegisterWrite>(SourceInfo(), a.node(),
                                     /*load_enable=*/std::nullopt,
                                     /*reset=*/std::nullopt, reg));

  EXPECT_THAT(block->GetRegisterRead(reg), IsOkAndHolds(reg_read));
  EXPECT_THAT(block->GetRegisterWrite(reg), IsOkAndHolds(reg_write));

  XLS_ASSERT_OK_AND_ASSIGN(RegisterRead * dup_reg_read,
                           block->MakeNode<RegisterRead>(SourceInfo(), reg));
  XLS_ASSERT_OK_AND_ASSIGN(
      RegisterWrite * dup_reg_write,
      block->MakeNode<RegisterWrite>(SourceInfo(), a.node(),
                                     /*load_enable=*/std::nullopt,
                                     /*reset=*/std::nullopt, reg));

  EXPECT_THAT(block->GetRegisterRead(reg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Block my_block has multiple read operation "
                                 "for register a_reg")));
  EXPECT_THAT(block->GetRegisterWrite(reg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Block my_block has multiple write operation "
                                 "for register a_reg")));

  // Remove duplicate register operations to avoid test failure when the block
  // is verified on destruction of the VerifiedPackage containing the block.
  XLS_ASSERT_OK(block->RemoveNode(dup_reg_read));
  XLS_ASSERT_OK(block->RemoveNode(dup_reg_write));

  // Removing register should fail because of existing reads and writes.
  EXPECT_THAT(
      block->RemoveRegister(reg),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Register a_reg can't be removed because a register read "
                    "or write operation for this register still exists")));
}

TEST_F(BlockTest, BlockInstantiation) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder sub_bb("sub_block", p.get());
  BValue a = sub_bb.InputPort("a", u32);
  BValue b = sub_bb.InputPort("b", u32);
  sub_bb.OutputPort("x", a);
  sub_bb.OutputPort("y", b);
  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block, sub_bb.Build());

  BlockBuilder bb("my_block", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * instantiation,
      bb.block()->AddBlockInstantiation("inst", sub_block));
  BValue in0 = bb.InputPort("in0", u32);
  BValue in1 = bb.InputPort("in1", u32);
  BValue inst_in0 = bb.InstantiationInput(instantiation, "a", in0);
  BValue out0 = bb.InstantiationOutput(instantiation, "x");
  BValue inst_in1 = bb.InstantiationInput(instantiation, "b", in1);
  BValue out1 = bb.InstantiationOutput(instantiation, "y");
  bb.OutputPort("out0", out0);
  bb.OutputPort("out1", out1);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(block->GetInstantiations(), ElementsAre(instantiation));
  EXPECT_THAT(block->GetInstantiation("inst"), IsOkAndHolds(instantiation));
  EXPECT_THAT(
      block->GetInstantiation("not_an_inst"),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr("Block my_block has no instantiation named not_an_inst")));
  EXPECT_THAT(block->GetInstantiationInputs(instantiation),
              ElementsAre(inst_in0.node(), inst_in1.node()));
  EXPECT_THAT(block->GetInstantiationOutputs(instantiation),
              ElementsAre(out0.node(), out1.node()));
  EXPECT_TRUE(block->IsOwned(instantiation));
  EXPECT_FALSE(sub_block->IsOwned(instantiation));

  XLS_VLOG_LINES(1, block->DumpIr());
  EXPECT_EQ(
      block->DumpIr(),
      R"(block my_block(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32]) {
  instantiation inst(block=sub_block, kind=block)
  in0: bits[32] = input_port(name=in0, id=5)
  in1: bits[32] = input_port(name=in1, id=6)
  instantiation_input.7: () = instantiation_input(in0, instantiation=inst, port_name=a, id=7)
  instantiation_output.8: bits[32] = instantiation_output(instantiation=inst, port_name=x, id=8)
  instantiation_input.9: () = instantiation_input(in1, instantiation=inst, port_name=b, id=9)
  instantiation_output.10: bits[32] = instantiation_output(instantiation=inst, port_name=y, id=10)
  out0: () = output_port(instantiation_output.8, name=out0, id=11)
  out1: () = output_port(instantiation_output.10, name=out1, id=12)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));

  EXPECT_EQ(
      clone->DumpIr(),
      R"(block cloned(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32]) {
  instantiation inst(block=sub_block, kind=block)
  in0: bits[32] = input_port(name=in0, id=13)
  in1: bits[32] = input_port(name=in1, id=14)
  instantiation_output_8: bits[32] = instantiation_output(instantiation=inst, port_name=x, id=15)
  instantiation_output_10: bits[32] = instantiation_output(instantiation=inst, port_name=y, id=16)
  instantiation_input_7: () = instantiation_input(in0, instantiation=inst, port_name=a, id=17)
  instantiation_input_9: () = instantiation_input(in1, instantiation=inst, port_name=b, id=18)
  out0: () = output_port(instantiation_output_8, name=out0, id=19)
  out1: () = output_port(instantiation_output_10, name=out1, id=20)
}
)");
}

}  // namespace
}  // namespace xls
