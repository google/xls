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

#include "xls/codegen_v_1_5/channel_to_port_io_lowering_pass.h"

#include <memory>
#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

namespace m = ::xls::op_matchers;

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;

class ChannelToPortIoLoweringPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p, BlockConversionPassOptions options =
                                           BlockConversionPassOptions()) {
    PassResults results;

    // Set default options if not provided
    if (!options.codegen_options.clock_name().has_value()) {
      options.codegen_options.clock_name("clk");
    }
    if (!options.codegen_options.reset().has_value()) {
      options.codegen_options.reset("rst", false, false, false);
    }

    // 1. Convert scheduled_proc -> scheduled_block
    // We assume the input package has a scheduled proc as top.
    XLS_RETURN_IF_ERROR(
        ScheduledBlockConversionPass().Run(p, options, &results).status());

    // 2. Run the pass under test
    return ChannelToPortIoLoweringPass().Run(p, options, &results);
  }
};

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleInputChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->GetInputPort("a_in"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_in_vld"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), IsOk());
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleOutputChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_out: bits[32] out>(init={}) {
  chan_interface a_out(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)
    send.1: token = send(tkn, lit, channel=a_out)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_out_vld"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_out_rdy"), IsOk());
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerInternalLoopbackChannelOldStyle) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

chan internal_ch(bits[32], id=0, kind=streaming, flow_control=ready_valid, ops=send_receive, strictness=proven_mutually_exclusive, fifo_depth=1, bypass=false, register_push_outputs=true, input_flop_kind=none, output_flop_kind=none)

top scheduled_proc __test__main(init={}) {
  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)
    send.1: token = send(tkn, lit, channel=internal_ch)
    receive.2: (token, bits[32]) = receive(tkn, channel=internal_ch)
  }
}
)"));

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  // Should NOT have ports for internal channel
  EXPECT_THAT(block->GetInputPort("internal_ch"), Not(IsOk()));
  EXPECT_THAT(block->GetOutputPort("internal_ch"), Not(IsOk()));

  // Should have FIFO instantiation
  EXPECT_THAT(block->nodes(), Contains(AnyOf(m::InstantiationInput(),
                                             m::InstantiationOutput())));
}

TEST_F(ChannelToPortIoLoweringPassTest, MultiOutputWithOneShotLogic) {
  // Two output channels, not mutually exclusive. Should generate one-shot logic
  // (already_done registers).
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<ch_a: bits[32] out, ch_b: bits[32] out>(init={}) {
  chan_interface ch_a(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)
  chan_interface ch_b(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)
    send.1: token = send(tkn, lit, channel=ch_a)
    send.2: token = send(tkn, lit, channel=ch_b)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  // Check for registers named something like "*already_done_reg"
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("already_done_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest,
       MutuallyExclusiveOutputsNoOneShotLogic) {
  // Two output channels, strictly mutually exclusive. Should NOT generate
  // one-shot logic.
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<ch_in: bits[1] in, ch_a: bits[32] out, ch_b: bits[32] out>(init={}) {
  chan_interface ch_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)
  chan_interface ch_a(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)
  chan_interface ch_b(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)

    receive.1: (token, bits[1]) = receive(tkn, channel=ch_in)
    recv_tkn: token = tuple_index(receive.1, index=0)
    pred: bits[1] = tuple_index(receive.1, index=1)
    not_pred: bits[1] = not(pred)

    send.2: token = send(recv_tkn, lit, predicate=pred, channel=ch_a)
    send.3: token = send(recv_tkn, lit, predicate=not_pred, channel=ch_b)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  // Check for registers named something like "*already_done_reg"
  EXPECT_THAT(block->nodes(),
              Not(Contains(m::RegisterRead(HasSubstr("already_done_reg")))));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleValueInputChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=single_value, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->GetInputPort("a_in"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_in_vld"), Not(IsOk()));
  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), Not(IsOk()));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleValueOutputChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_out: bits[32] out>(init={}) {
  chan_interface a_out(direction=send, kind=single_value, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)
    send.1: token = send(tkn, lit, channel=a_out)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_out_vld"), Not(IsOk()));
  EXPECT_THAT(block->GetInputPort("a_out_rdy"), Not(IsOk()));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerMultipleSendsSameChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_out: bits[32] out>(init={}) {
  chan_interface a_out(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    lit1: bits[32] = literal(value=1)
    lit2: bits[32] = literal(value=2)
    p1: bits[1] = literal(value=1)
    p2: bits[1] = literal(value=0)
    send.1: token = send(tkn, lit1, predicate=p1, channel=a_out)
    send.2: token = send(tkn, lit2, predicate=p2, channel=a_out)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  // Check that we have the ports.
  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());

  // Check that the data port is driven by a OneHotSelect (multiplexer)
  Node* data_port_driver =
      block->GetOutputPort("a_out").value()->output_source();
  EXPECT_THAT(data_port_driver, m::OneHotSelect());
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerMultipleReceivesSameChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    p1: bits[1] = literal(value=1)
    p2: bits[1] = literal(value=0)
    receive.1: (token, bits[32]) = receive(tkn, predicate=p1, channel=a_in)
    receive.2: (token, bits[32]) = receive(tkn, predicate=p2, channel=a_in)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), IsOk());

  // Check that the ready port is driven by an OR of the individual receive
  // ready signals (which include predicates).
  Node* ready_port_driver =
      block->GetOutputPort("a_in_rdy").value()->output_source();
  // It should be an NaryOp(kOr) because we have multiple receives.
  EXPECT_THAT(ready_port_driver, m::Or());
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerInternalChannelNewStyle) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)
  chan internal_ch(bits[32], id=1000, kind=streaming, flow_control=ready_valid, ops=send_receive, strictness=proven_mutually_exclusive, fifo_depth=1, bypass=false, register_push_outputs=true)
  chan_interface internal_ch(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)
  chan_interface internal_ch(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
    // Receive from internal channel
    receive.2: (token, bits[32]) = receive(tkn, channel=internal_ch)
    // Send to internal channel
    lit: bits[32] = literal(value=123)
    send.3: token = send(tkn, lit, channel=internal_ch)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  // Should NOT have ports for internal channel
  EXPECT_THAT(block->GetInputPort("internal_ch"), Not(IsOk()));
  EXPECT_THAT(block->GetOutputPort("internal_ch"), Not(IsOk()));

  // Should have FIFO instantiation
  EXPECT_THAT(block->nodes(), Contains(AnyOf(m::InstantiationInput(),
                                             m::InstantiationOutput())));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingInputFlop) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=flop)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_valid_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingInputZeroLatency) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=zero_latency)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_skid_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_valid_skid_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingInputSkid) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=skid)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  // Expect skid registers
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_skid_reg"))));
  // Expect pipeline registers
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingOutputFlop) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_out: bits[32] out>(init={}) {
  chan_interface a_out(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=flop)

  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)
    send.1: token = send(tkn, lit, channel=a_out)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_valid_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingOutputSkid) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_out: bits[32] out>(init={}) {
  chan_interface a_out(direction=send, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=skid)

  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)
    send.1: token = send(tkn, lit, channel=a_out)
  }
}
)"));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_skid_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, SingleValueInputFlop) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=single_value, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.flop_inputs(true);
  options.codegen_options.flop_single_value_channels(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, SingleValueOutputFlop) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_out: bits[32] out>(init={}) {
  chan_interface a_out(direction=send, kind=single_value, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    lit: bits[32] = literal(value=123)
    send.1: token = send(tkn, lit, channel=a_out)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.flop_outputs(true);
  options.codegen_options.flop_single_value_channels(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsWithPredicate) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    p: bits[1] = literal(value=1)
    receive.1: (token, bits[32]) = receive(tkn, predicate=p, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(
      block->nodes(),
      Contains(m::Select(m::Literal(1),
                         /*cases=*/{m::Literal(0), m::InputPort("a_in")})));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsNonBlocking) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32], bits[1]) = receive(tkn, blocking=false, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(
      block->nodes(),
      Contains(m::Select(m::InputPort("a_in_vld"),
                         /*cases=*/{m::Literal(0), m::InputPort("a_in")})));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsNonBlockingWithPredicate) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    p: bits[1] = literal(value=1)
    receive.1: (token, bits[32], bits[1]) = receive(tkn, predicate=p, blocking=false, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(
      block->nodes(),
      Contains(m::Select(m::And(m::Literal(1), m::InputPort("a_in_vld")),
                         /*cases=*/{m::Literal(0), m::InputPort("a_in")})));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsBlockingNoPredicate) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32]) = receive(tkn, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(), Contains(m::Tuple(m::Literal(Value::Token()),
                                                m::InputPort("a_in"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, NoGateRecvsWithPredicate) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    p: bits[1] = literal(value=1)
    receive.1: (token, bits[32]) = receive(tkn, predicate=p, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(false);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(), Contains(m::Tuple(m::Literal(Value::Token()),
                                                m::InputPort("a_in"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, NoGateRecvsNonBlocking) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    receive.1: (token, bits[32], bits[1]) = receive(tkn, blocking=false, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(false);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(), Contains(m::Tuple(m::Literal(Value::Token()),
                                                m::InputPort("a_in"),
                                                m::InputPort("a_in_vld"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, NoGateRecvsNonBlockingWithPredicate) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

top scheduled_proc __test__main<a_in: bits[32] in>(init={}) {
  chan_interface a_in(direction=receive, kind=streaming, flow_control=ready_valid, strictness=proven_mutually_exclusive, flop_kind=none)

  stage {
    tkn: token = literal(value=token)
    p: bits[1] = literal(value=1)
    receive.1: (token, bits[32], bits[1]) = receive(tkn, predicate=p, blocking=false, channel=a_in)
  }
}
)"));

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(false);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("__test__main"));

  EXPECT_THAT(block->nodes(), Contains(m::Tuple(m::Literal(Value::Token()),
                                                m::InputPort("a_in"),
                                                m::InputPort("a_in_vld"))));
}

}  // namespace
}  // namespace xls::codegen
