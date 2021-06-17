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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/visitor.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

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
}

TEST_F(BlockTest, ErrorConditions) {
  // Don't use CreatePackage because that creates a package which verifies on
  // test completion and the errors may leave the block in a invalid state.
  Package p(TestName());
  BlockBuilder bb("my_block", &p);
  Type* u32 = p.GetBitsType(32);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("out", bb.Add(a, b, /*loc=*/absl::nullopt, /*name=*/"foo"));
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
  EXPECT_THAT(block->AddInputPort("foo", u32),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("A node already exists with name foo")));
}

TEST_F(BlockTest, TrivialBlock) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_EQ(block->DumpIr(), R"(block my_block() {
}
)");
}

TEST_F(BlockTest, BlockWithRegisters) {
  auto p = CreatePackage();
  BlockBuilder bb("my_block", p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg,
                           bb.block()->AddRegister("my_reg", u32));
  bb.RegisterWrite(reg, bb.Add(a, b));
  BValue sum_d = bb.RegisterRead(reg);
  BValue out = bb.OutputPort("out", sum_d);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_FALSE(block->IsFunction());
  EXPECT_FALSE(block->IsProc());
  EXPECT_TRUE(block->IsBlock());
  EXPECT_THAT(GetPortNames(block), ElementsAre("a", "b", "out", "clk"));

  EXPECT_THAT(block->GetInputPorts(), ElementsAre(a.node(), b.node()));
  EXPECT_THAT(GetInputPortNames(block), ElementsAre("a", "b"));

  EXPECT_THAT(block->GetOutputPorts(), ElementsAre(out.node()));
  EXPECT_THAT(GetOutputPortNames(block), ElementsAre("out"));

  ASSERT_TRUE(block->GetClockPort().has_value());
  EXPECT_THAT(block->GetClockPort()->name, "clk");

  EXPECT_EQ(
      block->DumpIr(),
      R"(block my_block(a: bits[32], b: bits[32], out: bits[32], clk: clock) {
  reg my_reg(bits[32])
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  add.3: bits[32] = add(a, b, id=3)
  register_read.5: bits[32] = register_read(register=my_reg, id=5)
  register_write.4: bits[32] = register_write(add.3, register=my_reg, id=4)
  out: () = output_port(register_read.5, name=out, id=6)
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
  Block* blk = p->AddBlock(absl::make_unique<Block>("block1", p.get()));
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
  Block* blk = p.AddBlock(absl::make_unique<Block>("block1", &p));
  XLS_ASSERT_OK(blk->AddClockPort("clk"));
  XLS_ASSERT_OK(blk->AddRegister("my_reg", p.GetBitsType(32)).status());
  EXPECT_THAT(blk->AddRegister("my_reg", p.GetBitsType(32)).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Register already exists with name my_reg")));
}

TEST_F(BlockTest, RemoveRegisterNotOwnedByBlock) {
  // Don't use CreatePackage because that creates a package which verifies on
  // test completion and the errors may leave the block in a invalid state.
  Package p(TestName());
  Block* blk1 = p.AddBlock(absl::make_unique<Block>("block1", &p));
  XLS_ASSERT_OK(blk1->AddClockPort("clk"));
  Block* blk2 = p.AddBlock(absl::make_unique<Block>("block2", &p));
  XLS_ASSERT_OK(blk2->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg,
                           blk1->AddRegister("my_reg", p.GetBitsType(32)));

  EXPECT_THAT(blk2->RemoveRegister(reg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Register is not owned by block")));
}

}  // namespace
}  // namespace xls
