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
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/golden_files.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/function.h"
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

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

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

  void ExpectIr(std::string_view got, std::string_view test_name) {
    ExpectEqualToGoldenFile(
        absl::StrFormat("xls/ir/testdata/block_test_%s.ir", test_name), got);
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

  // Intentionally create the input ports in the wrong order then order them
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

  ForeignFunctionData ffi_data;
  ffi_data.set_code_template("my_ffi_{fn}");
  FunctionBuilder extern_fb("extern_func", p.get());
  extern_fb.SetForeignFunctionData(ffi_data);
  extern_fb.Add(extern_fb.Literal(UBits(32, 32)), extern_fb.Param("abc", u32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * extern_func, extern_fb.Build());

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
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * fifo_inst,
      bb.block()->AddFifoInstantiation(
          "my_fifo", FifoConfig(4, false, false, false), u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * extern_inst,
      bb.block()->AddInstantiation("extern_func_inst",
                                   std::make_unique<ExternInstantiation>(
                                       "extern_func_inst", extern_func)));
  BValue in0 = bb.InputPort("in0", u32);
  BValue in1 = bb.InputPort("in1", u32);
  BValue inst_in0 = bb.InstantiationInput(instantiation, "a", in0);
  BValue out0 = bb.InstantiationOutput(instantiation, "x");
  BValue inst_in1 = bb.InstantiationInput(instantiation, "b", in1);
  BValue out1 = bb.InstantiationOutput(instantiation, "y");
  bb.OutputPort("out0", out0);
  bb.OutputPort("out1", out1);
  bb.InstantiationInput(fifo_inst, FifoInstantiation::kPushDataPortName,
                        bb.InputPort("fifo_in", u32));
  bb.OutputPort(
      "fifo_out",
      bb.InstantiationOutput(fifo_inst, FifoInstantiation::kPopDataPortName));
  bb.InstantiationInput(extern_inst,
                        absl::StrFormat("%s.0", extern_func->name()),
                        bb.InputPort("extern_port", u32));
  bb.OutputPort("extern_out", bb.InstantiationOutput(extern_inst, "return"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(block->GetInstantiations(),
              UnorderedElementsAre(instantiation, fifo_inst, extern_inst));
  EXPECT_THAT(block->GetInstantiation("inst"), IsOkAndHolds(instantiation));
  EXPECT_THAT(block->GetInstantiation("my_fifo"), IsOkAndHolds(fifo_inst));
  EXPECT_THAT(block->GetInstantiation("extern_func_inst"),
              IsOkAndHolds(extern_inst));
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

  RecordProperty("block", block->DumpIr());
  XLS_VLOG_LINES(1, block->DumpIr());
  EXPECT_EQ(
      block->DumpIr(),
      R"(block my_block(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32], fifo_in: bits[32], fifo_out: bits[32], extern_port: bits[32], extern_out: bits[32]) {
  instantiation inst(block=sub_block, kind=block)
  instantiation my_fifo(data_type=bits[32], depth=4, bypass=false, register_push_outputs=false, register_pop_outputs=false, kind=fifo)
  instantiation extern_func_inst(foreign_function=extern_func, kind=extern)
  in0: bits[32] = input_port(name=in0, id=8)
  in1: bits[32] = input_port(name=in1, id=9)
  fifo_in: bits[32] = input_port(name=fifo_in, id=16)
  extern_port: bits[32] = input_port(name=extern_port, id=20)
  instantiation_input.10: () = instantiation_input(in0, instantiation=inst, port_name=a, id=10)
  instantiation_output.11: bits[32] = instantiation_output(instantiation=inst, port_name=x, id=11)
  instantiation_input.12: () = instantiation_input(in1, instantiation=inst, port_name=b, id=12)
  instantiation_output.13: bits[32] = instantiation_output(instantiation=inst, port_name=y, id=13)
  instantiation_input.17: () = instantiation_input(fifo_in, instantiation=my_fifo, port_name=push_data, id=17)
  instantiation_output.18: bits[32] = instantiation_output(instantiation=my_fifo, port_name=pop_data, id=18)
  instantiation_input.21: () = instantiation_input(extern_port, instantiation=extern_func_inst, port_name=extern_func.0, id=21)
  instantiation_output.22: bits[32] = instantiation_output(instantiation=extern_func_inst, port_name=return, id=22)
  out0: () = output_port(instantiation_output.11, name=out0, id=14)
  out1: () = output_port(instantiation_output.13, name=out1, id=15)
  fifo_out: () = output_port(instantiation_output.18, name=fifo_out, id=19)
  extern_out: () = output_port(instantiation_output.22, name=extern_out, id=23)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Block * clone, block->Clone("cloned"));
  RecordProperty("clone", clone->DumpIr());

  EXPECT_EQ(
      clone->DumpIr(),
      R"(block cloned(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32], fifo_in: bits[32], fifo_out: bits[32], extern_port: bits[32], extern_out: bits[32]) {
  instantiation inst(block=sub_block, kind=block)
  instantiation my_fifo(data_type=bits[32], depth=4, bypass=false, register_push_outputs=false, register_pop_outputs=false, kind=fifo)
  instantiation extern_func_inst(foreign_function=extern_func, kind=extern)
  in0: bits[32] = input_port(name=in0, id=24)
  in1: bits[32] = input_port(name=in1, id=25)
  fifo_in: bits[32] = input_port(name=fifo_in, id=28)
  extern_port: bits[32] = input_port(name=extern_port, id=30)
  instantiation_output_11: bits[32] = instantiation_output(instantiation=inst, port_name=x, id=26)
  instantiation_output_13: bits[32] = instantiation_output(instantiation=inst, port_name=y, id=27)
  instantiation_output_18: bits[32] = instantiation_output(instantiation=my_fifo, port_name=pop_data, id=29)
  instantiation_output_22: bits[32] = instantiation_output(instantiation=extern_func_inst, port_name=return, id=31)
  instantiation_input_10: () = instantiation_input(in0, instantiation=inst, port_name=a, id=32)
  instantiation_input_12: () = instantiation_input(in1, instantiation=inst, port_name=b, id=33)
  instantiation_input_17: () = instantiation_input(fifo_in, instantiation=my_fifo, port_name=push_data, id=36)
  instantiation_input_21: () = instantiation_input(extern_port, instantiation=extern_func_inst, port_name=extern_func.0, id=38)
  out0: () = output_port(instantiation_output_11, name=out0, id=34)
  out1: () = output_port(instantiation_output_13, name=out1, id=35)
  fifo_out: () = output_port(instantiation_output_18, name=fifo_out, id=37)
  extern_out: () = output_port(instantiation_output_22, name=extern_out, id=39)
}
)");
}

TEST_F(BlockTest, ReplaceInstantiation) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder sub_bb("sub_block", p.get());
  {
    BValue a = sub_bb.InputPort("a", u32);
    BValue b = sub_bb.InputPort("b", u32);
    sub_bb.OutputPort("x", a);
    sub_bb.OutputPort("y", b);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block, sub_bb.Build());
  BlockBuilder add_bb("add_block", p.get());
  {
    BValue a = add_bb.InputPort("a", u32);
    BValue b = add_bb.InputPort("b", u32);
    add_bb.OutputPort("x", add_bb.Add(a, b));
    add_bb.OutputPort("y", b);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * add_block, add_bb.Build());

  BlockBuilder add2_bb("add2_block", p.get());
  {
    BValue a = add2_bb.InputPort("a2", u32);
    BValue b = add2_bb.InputPort("b2", u32);
    add2_bb.OutputPort("x", add2_bb.Add(a, b));
    add2_bb.OutputPort("y", b);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * add2_block, add2_bb.Build());

  BlockBuilder bb("my_block", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * instantiation,
      bb.block()->AddBlockInstantiation("inst", sub_block));
  {
    BValue in0 = bb.InputPort("in0", u32);
    BValue in1 = bb.InputPort("in1", u32);
    bb.InstantiationInput(instantiation, "a", in0);
    BValue out0 = bb.InstantiationOutput(instantiation, "x");
    bb.InstantiationInput(instantiation, "b", in1);
    BValue out1 = bb.InstantiationOutput(instantiation, "y");
    bb.OutputPort("out0", out0);
    bb.OutputPort("out1", out1);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto inst_add,
                           block->AddBlockInstantiation("inst_add", add_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto inst2_add, block->AddBlockInstantiation("inst2_add", add2_block));

  EXPECT_THAT(block->ReplaceInstantiationWith(instantiation, inst2_add),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::ContainsRegex("Type mismatch")));
  XLS_ASSERT_OK(block->RemoveInstantiation(inst2_add));
  EXPECT_THAT(block->ReplaceInstantiationWith(instantiation, inst_add),
              absl_testing::IsOk());
  EXPECT_EQ(p->DumpIr(), R"(package ReplaceInstantiation

block sub_block(a: bits[32], b: bits[32], x: bits[32], y: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  x: () = output_port(a, name=x, id=3)
  y: () = output_port(b, name=y, id=4)
}

block add2_block(a2: bits[32], b2: bits[32], x: bits[32], y: bits[32]) {
  a2: bits[32] = input_port(name=a2, id=10)
  b2: bits[32] = input_port(name=b2, id=11)
  add.12: bits[32] = add(a2, b2, id=12)
  x: () = output_port(add.12, name=x, id=13)
  y: () = output_port(b2, name=y, id=14)
}

block add_block(a: bits[32], b: bits[32], x: bits[32], y: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  x: () = output_port(add.7, name=x, id=8)
  y: () = output_port(b, name=y, id=9)
}

block my_block(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32]) {
  instantiation inst_add(block=add_block, kind=block)
  in0: bits[32] = input_port(name=in0, id=15)
  in1: bits[32] = input_port(name=in1, id=16)
  instantiation_input.23: () = instantiation_input(in0, instantiation=inst_add, port_name=a, id=23)
  instantiation_input.24: () = instantiation_input(in1, instantiation=inst_add, port_name=b, id=24)
  instantiation_output.25: bits[32] = instantiation_output(instantiation=inst_add, port_name=x, id=25)
  instantiation_output.26: bits[32] = instantiation_output(instantiation=inst_add, port_name=y, id=26)
  out0: () = output_port(instantiation_output.25, name=out0, id=21)
  out1: () = output_port(instantiation_output.26, name=out1, id=22)
}
)");
}

TEST_F(BlockTest, ReplaceInstantiationWithRename) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  BlockBuilder sub_bb("sub_block", p.get());
  {
    BValue a = sub_bb.InputPort("a", u32);
    BValue b = sub_bb.InputPort("b", u32);
    sub_bb.OutputPort("x", a);
    sub_bb.OutputPort("y", b);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block, sub_bb.Build());
  BlockBuilder add_bb("add_block", p.get());
  {
    BValue a_renamed = add_bb.InputPort("a_renamed", u32);
    BValue b = add_bb.InputPort("b", u32);
    add_bb.OutputPort("x_renamed", add_bb.Add(a_renamed, b));
    add_bb.OutputPort("y", b);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * add_block, add_bb.Build());

  BlockBuilder bb("my_block", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * instantiation,
      bb.block()->AddBlockInstantiation("inst", sub_block));
  {
    BValue in0 = bb.InputPort("in0", u32);
    BValue in1 = bb.InputPort("in1", u32);
    bb.InstantiationInput(instantiation, "a", in0);
    BValue out0 = bb.InstantiationOutput(instantiation, "x");
    bb.InstantiationInput(instantiation, "b", in1);
    BValue out1 = bb.InstantiationOutput(instantiation, "y");
    bb.OutputPort("out0", out0);
    bb.OutputPort("out1", out1);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto inst_add,
                           block->AddBlockInstantiation("inst_add", add_block));

  // Replacing without rename should fail since port names do not match:
  EXPECT_THAT(block->ReplaceInstantiationWith(instantiation, inst_add),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::ContainsRegex("Type mismatch")));

  // Replacing with rename of non-existent port should fail:
  EXPECT_THAT(block->ReplaceInstantiationWith(instantiation, inst_add,
                                              {{"does_not_exist", "a2"}}),
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::ContainsRegex("rename port that does not exist")));

  // Replacing with rename of port to used name should fail:
  EXPECT_THAT(
      block->ReplaceInstantiationWith(instantiation, inst_add, {{"a", "b"}}),
      absl_testing::StatusIs(absl::StatusCode::kInternal,
                             testing::ContainsRegex("name already exists")));

  // Replacing with acceptable rename:
  EXPECT_THAT(
      block->ReplaceInstantiationWith(instantiation, inst_add,
                                      {{"a", "a_renamed"}, {"x", "x_renamed"}}),

      absl_testing::IsOk());
  ExpectIr(p->DumpIr(), TestName());
}

}  // namespace
}  // namespace xls
