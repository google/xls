// Copyright 2022 The XLS Authors
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

#include "xls/passes/proc_state_fsm_pass.h"

#include <cstdint>
#include <utility>
#include <vector>

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

class BaseProcStateFSMPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return ProcStateFSMPass().Run(p, OptimizationPassOptions(), &results,
                                  context);
  }
};
class ProcStateFSMPassTest : public BaseProcStateFSMPassTest,
                             public testing::WithParamInterface<NextValueType> {
 protected:
  ProcStateFSMPassTest() = default;

  absl::StatusOr<Proc*> BuildProc(ProcBuilder& pb,
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

TEST_P(ProcStateFSMPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  XLS_ASSERT_OK(BuildProc(pb, {}).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(ProcStateFSMPassTest, SimpleNonoptimizableStateProc) {
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

TEST_P(ProcStateFSMPassTest, SimpleNonoptimizableTokenStateProc) {
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

TEST_P(ProcStateFSMPassTest, LiteralChainOfSize1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.ReadStateElement("x", Value(UBits(100, 32)));
  BValue lit = pb.Literal(Value(UBits(200, 32)));
  BValue send = pb.Send(out, x);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {lit}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_EQ(proc->GetStateElement(0)->type()->GetFlatBitCount(), 1);

  EXPECT_THAT(send.node(),
              m::Send(m::Literal(Value::Token()),
                      m::Select(m::StateRead("state_machine_x"),
                                /*cases=*/{m::Literal(100)},
                                /*default_value=*/m::Literal(200))));
}

TEST_F(BaseProcStateFSMPassTest, LiteralChainDecoupled) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc{}, TestName(), "tkn", p.get());
  BSendChannel out = pb.AddOutputChannel("out", p->GetBitsType(32));
  BStateElement x_elem = pb.StateElement("x", Value(UBits(100, 32)),
                                         /*non_synthesizable=*/false);
  BValue x = pb.StateRead(x_elem);
  BValue lit = pb.Literal(Value(UBits(200, 32)));
  BValue send = pb.Send(out, x);
  pb.Next(x_elem, lit);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_EQ(proc->GetStateElement(0)->type()->GetFlatBitCount(), 1);

  EXPECT_THAT(send.node(),
              m::Send(m::Literal(Value::Token()),
                      m::Select(m::StateRead("state_machine_x"),
                                /*cases=*/{m::Literal(100)},
                                /*default_value=*/m::Literal(200))));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(m::NextWithStateElement(
                  m::StateElement("state_machine_x"), ::testing::_)));
}

INSTANTIATE_TEST_SUITE_P(NextValueTypes, ProcStateFSMPassTest,
                         testing::Values(NextValueType::kNextStateVector,
                                         NextValueType::kNextValueNodes),
                         testing::PrintToStringParamName());

void IrFuzzProcStateOptimization(FuzzPackageWithArgs fuzz_package_with_args) {
  ProcStateFSMPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzProcStateOptimization)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
