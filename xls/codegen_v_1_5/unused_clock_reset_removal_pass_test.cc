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

#include "xls/codegen_v_1_5/unused_clock_reset_removal_pass.h"

#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;

constexpr ResetBehavior kSyncActiveHigh = {.asynchronous = false,
                                           .active_low = false};

class UnusedClockResetRemovalPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    BlockConversionContext context;
    BlockConversionPassOptions options;
    return UnusedClockResetRemovalPass().Run(p, options, &results, context);
  }

  // Creates a block with clock and reset ports whose logic is purely
  // combinational: out = in.
  absl::StatusOr<Block*> CreateCombinationalLeaf(Package* p,
                                                 std::string_view name) {
    BlockBuilder bb(name, p);
    bb.AddClockPort("clk");
    bb.ResetPort("rst", kSyncActiveHigh);
    BValue in = bb.InputPort("in", p->GetBitsType(32));
    bb.OutputPort("out", in);
    return bb.Build();
  }

  // Creates a block with clock and reset ports containing a register with a
  // reset value.
  absl::StatusOr<Block*> CreateSequentialLeaf(Package* p,
                                              std::string_view name) {
    BlockBuilder bb(name, p);
    bb.AddClockPort("clk");
    BValue rst = bb.ResetPort("rst", kSyncActiveHigh);
    BValue in = bb.InputPort("in", p->GetBitsType(32));
    BValue in_d = bb.InsertRegister("in_reg", in, rst, Value(UBits(0, 32)));
    bb.OutputPort("out", in_d);
    return bb.Build();
  }

  // Creates a block with clock and reset ports which does nothing but
  // instantiate `child`, forwarding its own input/output ports.
  absl::StatusOr<Block*> CreateWrapper(Package* p, std::string_view name,
                                       Block* child) {
    BlockBuilder bb(name, p);
    bb.AddClockPort("clk");
    bb.ResetPort("rst", kSyncActiveHigh);
    BValue in = bb.InputPort("in", p->GetBitsType(32));
    XLS_ASSIGN_OR_RETURN(Block * block, bb.Build());
    XLS_ASSIGN_OR_RETURN(
        Block::BlockInstantiationAndConnections instantiation,
        block->AddAndConnectBlockInstantiation(
            absl::StrCat(name, "_inst"), child, {{"in", in.node()}}));
    XLS_RETURN_IF_ERROR(
        block->AddOutputPort("out", instantiation.outputs.at("out")).status());
    return block;
  }
};

TEST_F(UnusedClockResetRemovalPassTest, CombinationalLeafLosesClockAndReset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf,
                           CreateCombinationalLeaf(p.get(), "leaf"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, CreateWrapper(p.get(), "top_block", leaf));
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_FALSE(leaf->GetClockPort().has_value());
  EXPECT_FALSE(leaf->GetResetPort().has_value());
  // The top block keeps its interface.
  EXPECT_TRUE(top->GetClockPort().has_value());
  EXPECT_TRUE(top->GetResetPort().has_value());
  // The instantiation in the top block no longer drives the leaf's reset.
  for (::xls::Instantiation* instantiation : top->GetInstantiations()) {
    for (InstantiationInput* input :
         top->GetInstantiationInputs(instantiation)) {
      EXPECT_NE(input->port_name(), "rst");
    }
  }
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(UnusedClockResetRemovalPassTest, SequentialLeafKeepsClockAndReset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf, CreateSequentialLeaf(p.get(), "leaf"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, CreateWrapper(p.get(), "top_block", leaf));
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_TRUE(leaf->GetClockPort().has_value());
  EXPECT_TRUE(leaf->GetResetPort().has_value());
  EXPECT_TRUE(top->GetClockPort().has_value());
  EXPECT_TRUE(top->GetResetPort().has_value());
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(UnusedClockResetRemovalPassTest, TopBlockIsNeverChanged) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Block * top,
                           CreateCombinationalLeaf(p.get(), "top_block"));
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_TRUE(top->GetClockPort().has_value());
  EXPECT_TRUE(top->GetResetPort().has_value());
}

TEST_F(UnusedClockResetRemovalPassTest, RemovalPropagatesUpTheHierarchy) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf,
                           CreateCombinationalLeaf(p.get(), "leaf"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * middle,
                           CreateWrapper(p.get(), "middle", leaf));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, CreateWrapper(p.get(), "top_block", middle));
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  // Both the leaf and the middle wrapper lose clock and reset; only the top
  // keeps its interface.
  EXPECT_FALSE(leaf->GetClockPort().has_value());
  EXPECT_FALSE(leaf->GetResetPort().has_value());
  EXPECT_FALSE(middle->GetClockPort().has_value());
  EXPECT_FALSE(middle->GetResetPort().has_value());
  EXPECT_TRUE(top->GetClockPort().has_value());
  EXPECT_TRUE(top->GetResetPort().has_value());
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(UnusedClockResetRemovalPassTest, WrapperOfSequentialLeafKeepsPorts) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf, CreateSequentialLeaf(p.get(), "leaf"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * middle,
                           CreateWrapper(p.get(), "middle", leaf));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, CreateWrapper(p.get(), "top_block", middle));
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));

  // The middle wrapper must keep clock and reset: its child uses them.
  EXPECT_TRUE(middle->GetClockPort().has_value());
  EXPECT_TRUE(middle->GetResetPort().has_value());
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(UnusedClockResetRemovalPassTest, BlockWithExistingSignatureIsSkipped) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf,
                           CreateCombinationalLeaf(p.get(), "leaf"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top,
                           CreateWrapper(p.get(), "top_block", leaf));
  XLS_ASSERT_OK(p->SetTop(top));
  // A block with an already-generated signature is left alone: removing ports
  // would make the signature stale.
  verilog::ModuleSignatureProto proto;
  proto.set_module_name("leaf");
  leaf->SetSignature(proto);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_TRUE(leaf->GetClockPort().has_value());
  EXPECT_TRUE(leaf->GetResetPort().has_value());
}

TEST_F(UnusedClockResetRemovalPassTest, BlockWithAssertKeepsClockAndReset) {
  auto p = CreatePackage();
  Block* leaf;
  {
    BlockBuilder bb("leaf", p.get());
    bb.AddClockPort("clk");
    bb.ResetPort("rst", kSyncActiveHigh);
    BValue in = bb.InputPort("in", p->GetBitsType(1));
    bb.Assert(bb.AfterAll({}), in, "assertion failed");
    bb.OutputPort("out", in);
    XLS_ASSERT_OK_AND_ASSIGN(leaf, bb.Build());
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, [&]() -> absl::StatusOr<Block*> {
    BlockBuilder bb("top_block", p.get());
    bb.AddClockPort("clk");
    bb.ResetPort("rst", kSyncActiveHigh);
    BValue in = bb.InputPort("in", p->GetBitsType(1));
    XLS_ASSIGN_OR_RETURN(Block * block, bb.Build());
    XLS_ASSIGN_OR_RETURN(Block::BlockInstantiationAndConnections instantiation,
                         block->AddAndConnectBlockInstantiation(
                             "top_inst", leaf, {{"in", in.node()}}));
    XLS_RETURN_IF_ERROR(
        block->AddOutputPort("out", instantiation.outputs.at("out")).status());
    return block;
  }());
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));

  // Asserts are emitted differently depending on clock/reset presence, so the
  // leaf keeps both ports.
  EXPECT_TRUE(leaf->GetClockPort().has_value());
  EXPECT_TRUE(leaf->GetResetPort().has_value());
}

}  // namespace
}  // namespace xls::codegen
