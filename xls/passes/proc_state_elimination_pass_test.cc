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

#include "xls/passes/proc_state_elimination_pass.h"

#include <cstdint>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/base/optimization.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

enum class NextValueType : std::uint8_t {
  kNextStateVector,
  kNextValueNodes,
};

template <typename Sink>
void AbslStringify(Sink& sink, NextValueType e) {
  absl::Format(&sink, "%s",
               e == NextValueType::kNextStateVector ? "NextStateVector"
                                                    : "NextValueNodes");
}

class BaseProcStateEliminationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return ProcStateEliminationPass().Run(p, OptimizationPassOptions(),
                                           &results, context);
  }
};
class ProcStateEliminationPassTest
    : public BaseProcStateEliminationPassTest,
      public testing::WithParamInterface<NextValueType> {
 protected:
  ProcStateEliminationPassTest() = default;

  absl::StatusOr<Proc*> BuildProc(ProcBuilder& pb,
                                  absl::Span<const BValue> next_state) {
    switch (GetParam()) {
      case NextValueType::kNextStateVector:
        return pb.Build(next_state);
      case NextValueType::kNextValueNodes: {
        for (int64_t index = 0; index < next_state.size(); ++index) {
          BStateElement state_element = pb.GetStateElement(index);
          BValue next_value = next_state[index];
          pb.Next(state_element, next_value);
        }
        return pb.Build();
      }
    }
    ABSL_UNREACHABLE();
  }
  absl::StatusOr<Proc*> BuildProc(TokenlessProcBuilder& pb,
                                  absl::Span<const BValue> next_state) {
    switch (GetParam()) {
      case NextValueType::kNextStateVector:
        return pb.Build(next_state);
      case NextValueType::kNextValueNodes: {
        for (int64_t index = 0; index < next_state.size(); ++index) {
          BValue state_read = pb.GetStateParam(index);
          BValue next_value = next_state[index];
          pb.Next(state_read, next_value);
        }
        return pb.Build();
      }
    }
    ABSL_UNREACHABLE();
  }
};

TEST_P(ProcStateEliminationPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  XLS_ASSERT_OK(BuildProc(pb, {}).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(ProcStateEliminationPassTest, SimpleNonoptimizableStateProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.ReadStateElement("x", Value(UBits(0, 32)));
  BValue y = pb.ReadStateElement("y", Value(UBits(0, 32)));
  pb.Send(out, pb.Add(x, y));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {pb.Not(x), pb.Not(y)}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  ScopedRecordIr sri(p.get());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(proc->GetStateElementCount(), 2);
}

TEST_P(ProcStateEliminationPassTest, SimpleNonoptimizableTokenStateProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                              p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  ProcBuilder pb("p", p.get());
  BValue recvd = pb.Receive(in, pb.ReadStateElement("tok", Value::Token()));
  BValue recv_tok = pb.TupleIndex(recvd, 0);
  BValue recv_val = pb.TupleIndex(recvd, 1);
  BValue send_tok = pb.Send(out, recv_tok, recv_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {send_tok}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(proc->GetStateElementCount(), 1);
}

TEST_P(ProcStateEliminationPassTest, ProcWithDeadElements) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.ReadStateElement("x", Value(UBits(0, 32)));
  BValue y = pb.ReadStateElement("y", Value(UBits(0, 32)));
  BValue z = pb.ReadStateElement("z", Value(UBits(0, 32)));
  pb.Send(out, x);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           BuildProc(pb, {pb.Not(x), y, pb.Not(z)}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateElement(0)->name(), "x");
}

TEST_F(BaseProcStateEliminationPassTest, DecoupledDeadElements) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());

  // Register 'x': Live (read is sent to channel)
  BStateElement x_element = pb.StateElement("x", Value(UBits(0, 32)),
                                            /*non_synthesizable=*/false);
  BValue x_read = pb.StateRead(x_element);
  pb.Send(out, x_read);
  pb.Next(x_element, pb.Not(x_read));

  // Register 'y': Dead (has 1 read but it is unused, write is a constant)
  BStateElement y_element = pb.StateElement("y", Value(UBits(0, 32)),
                                            /*non_synthesizable=*/false);
  pb.StateRead(y_element);
  pb.Next(y_element, pb.Literal(UBits(5, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  // State element y gets cleaned up
  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "x");
}

TEST_P(ProcStateEliminationPassTest, MultipleStateReads) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BStateElement x_element =
      pb.StateElement("x", Value(UBits(42, 32)), /*non_synthesizable=*/false);
  BValue r1 = pb.StateRead(x_element);
  BValue r2 = pb.StateRead(x_element);
  BValue next_val = pb.Add(r1, r2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {next_val}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 0);
}

TEST_P(ProcStateEliminationPassTest, CrissCrossDeadElements) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.ReadStateElement("x", Value(UBits(0, 32)));
  BValue y = pb.ReadStateElement("y", Value(UBits(0, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {y, x}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 0);
}

TEST_P(ProcStateEliminationPassTest, CrissCrossDeadAndLiveElements) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue a = pb.ReadStateElement("a", Value(UBits(0, 32)));
  BValue b = pb.ReadStateElement("b", Value(UBits(0, 32)));
  BValue c = pb.ReadStateElement("c", Value(UBits(0, 32)));
  BValue x = pb.ReadStateElement("x", Value(UBits(0, 32)));
  BValue y = pb.ReadStateElement("y", Value(UBits(0, 32)));

  pb.Send(out, c);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {b, c, a, y, x}));

  EXPECT_EQ(proc->GetStateElementCount(), 5);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->StateElements(),
              ElementsAre(m::StateElement("a"), m::StateElement("b"),
                          m::StateElement("c")));
}

TEST_P(ProcStateEliminationPassTest, ProcWithZeroWidthElement) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), "p", "tkn", p.get());
  BValue x = pb.ReadStateElement("x", Value(UBits(0, 0)));
  BSendChannel out = pb.AddOutputChannel("out", p->GetBitsType(32));
  BValue y = pb.ReadStateElement("y", Value(UBits(0, 32)));
  BValue send = pb.Send(out, pb.Concat({x, y}));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {pb.Not(x), pb.Not(y)}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateElement(0)->name(), "y");
  EXPECT_THAT(send.node(),
              m::Send(m::Literal(Value::Token()),
                      m::Concat(m::Literal(UBits(0, 0)), m::StateRead("y"))));
}

TEST_P(ProcStateEliminationPassTest, StateElementsIntoTuplesAndOut) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.ReadStateElement("x", Value(UBits(0, 32)));
  BValue y = pb.ReadStateElement("y", Value(UBits(0, 32)));
  BValue z = pb.ReadStateElement("z", Value(UBits(0, 32)));

  BValue xy = pb.Tuple({x, y});
  BValue xy_z = pb.Tuple({xy, z});

  // Send element `y` from the tuple.
  pb.Send(out, pb.TupleIndex(xy, 1));

  BValue next_x = y;
  BValue next_y = pb.TupleIndex(pb.TupleIndex(xy_z, 0), 0);
  BValue next_z = pb.TupleIndex(xy_z, 1);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           BuildProc(pb, {next_x, next_y, next_z}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(proc->StateElements(),
              ElementsAre(m::StateElement("x"), m::StateElement("y")));
}

TEST_P(ProcStateEliminationPassTest, ProcWithPartiallyDeadStateElement) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  Value zero(UBits(0, 32));
  BValue dead_state = pb.ReadStateElement("dead", Value::Tuple({zero, zero}));
  BValue not_dead_state =
      pb.ReadStateElement("not_dead", Value::Tuple({zero, zero}));
  // Send only one tuple element of the `not_dead` state.
  pb.Send(out, pb.TupleIndex(not_dead_state, 0));
  // Modify the active part of the `not_dead` state so it can't be eliminated.
  BValue next_not_dead_state =
      pb.Tuple({pb.Not(pb.TupleIndex(not_dead_state, 0)),
                pb.TupleIndex(not_dead_state, 1)});

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           BuildProc(pb, {dead_state, next_not_dead_state}));

  EXPECT_THAT(proc->StateElements(),
              UnorderedElementsAre(m::StateElement("dead"),
                                   m::StateElement("not_dead")));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->StateElements(),
              UnorderedElementsAre(m::StateElement("not_dead")));
}

TEST_P(ProcStateEliminationPassTest, ProcWithConstantStateElement) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  Value zero(UBits(0, 32));
  Value one(UBits(1, 32));
  BValue constant_state =
      pb.ReadStateElement("constant", Value::Tuple({one, zero}));
  BValue not_constant_state =
      pb.ReadStateElement("not_constant", Value::Tuple({zero, zero}));
  // Use one tuple element of both states.
  BValue state_usage = pb.Add(pb.TupleIndex(constant_state, 0),
                              pb.TupleIndex(not_constant_state, 0));
  pb.Send(out, state_usage);
  // Modify the active part of the `not_constant` state so it can't be
  // eliminated.
  BValue next_not_constant_state =
      pb.Tuple({pb.Not(pb.TupleIndex(not_constant_state, 0)),
                pb.TupleIndex(not_constant_state, 1)});

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, BuildProc(pb, {constant_state, next_not_constant_state}));

  EXPECT_THAT(proc->StateElements(),
              UnorderedElementsAre(m::StateElement("constant"),
                                   m::StateElement("not_constant")));
  EXPECT_THAT(state_usage.node(),
              m::Add(m::TupleIndex(m::StateRead("constant")),
                     m::TupleIndex(m::StateRead("not_constant"))));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->StateElements(),
              UnorderedElementsAre(m::StateElement("not_constant")));
  // Verify the state element has been replaced with its initial value.
  EXPECT_THAT(state_usage.node(),
              m::Add(m::TupleIndex(m::Literal(Value::Tuple({one, zero}))),
                     m::TupleIndex(m::StateRead("not_constant"))));
}

TEST_P(ProcStateEliminationPassTest, ProcWithImplicitlyConstantStateElements) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  Value zero(UBits(0, 32));
  Value one(UBits(1, 32));
  BValue constant_state = pb.ReadStateElement("constant", zero);
  BValue not_constant_state =
      pb.ReadStateElement("not_constant", Value::Tuple({zero, zero}));
  BValue implicit_constant_state_1 =
      pb.ReadStateElement("implicit_constant_1", one);
  BValue implicit_constant_state_2 =
      pb.ReadStateElement("implicit_constant_2", one);
  // Use one element of each state.
  pb.Send(out,
          pb.Or(pb.Or(constant_state, pb.TupleIndex(not_constant_state, 0)),
                pb.Or(implicit_constant_state_1, implicit_constant_state_2)));
  // Modify just one part of the `not_constant` state so it can't be
  // eliminated.
  BValue next_not_constant_state =
      pb.Tuple({pb.Not(pb.TupleIndex(not_constant_state, 0)),
                pb.TupleIndex(not_constant_state, 1)});
  // Modify the implicitly-constant states based on the `constant` state.
  BValue next_implicit_constant_state_1 =
      pb.Xor(constant_state, pb.Literal(one));
  BValue next_implicit_constant_state_2 =
      pb.Xor(constant_state, implicit_constant_state_1);

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, BuildProc(pb, {constant_state, next_not_constant_state,
                                  next_implicit_constant_state_1,
                                  next_implicit_constant_state_2}));

  EXPECT_THAT(proc->StateElements(),
              UnorderedElementsAre(m::StateElement("constant"),
                                   m::StateElement("not_constant"),
                                   m::StateElement("implicit_constant_1"),
                                   m::StateElement("implicit_constant_2")));
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->StateElements(),
              UnorderedElementsAre(m::StateElement("not_constant")));
}

TEST_F(BaseProcStateEliminationPassTest, ProcWithWriteNoReads) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Channel * chan, p->CreateStreamingChannel(
                                               "chan", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));
  ProcBuilder pb(TestName(), p.get());
  BValue live = pb.ReadStateElement("live", UBits(0, 1));
  BValue dead = pb.Not(live);
  BValue val = pb.ReadStateElement("chan_val", UBits(0, 32));
  BValue nv = pb.ReceiveIf(chan, pb.Literal(Value::Token()), live);
  pb.Next(val, pb.TupleIndex(nv, 1), live);
  pb.Next(val, val, dead);
  pb.Next(live, dead);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ScopedRecordIr sri(p.get());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(proc->StateElements(), ElementsAre(m::StateElement("live")));
}

INSTANTIATE_TEST_SUITE_P(NextValueTypes, ProcStateEliminationPassTest,
                         testing::Values(NextValueType::kNextStateVector,
                                         NextValueType::kNextValueNodes),
                         testing::PrintToStringParamName());

void IrFuzzProcStateElimination(FuzzPackageWithArgs fuzz_package_with_args) {
  ProcStateEliminationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzProcStateElimination)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
