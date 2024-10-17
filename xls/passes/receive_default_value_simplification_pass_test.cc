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

#include "xls/passes/receive_default_value_simplification_pass.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class ReceiveDefaultValueSimplificationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Proc* proc, int64_t opt_level = kMaxOptLevel) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         ReceiveDefaultValueSimplificationPass().RunOnProc(
                             proc, OptimizationPassOptions(), &results));
    return changed;
  }
};

TEST_F(ReceiveDefaultValueSimplificationPassTest,
       TransformableConditionalReceive) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue pred = pb.StateElement("pred", Value(UBits(1, 1)));
  pb.StateElement("s", Value(UBits(42, 32)));
  BValue receive = pb.ReceiveIf(out, pred);
  BValue select = pb.Select(pred, {pb.Literal(UBits(0, 32)), receive});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pred, select}));

  EXPECT_THAT(proc->GetNextStateElement(1),
              m::Select(m::Param("pred"),
                        {m::Literal(0), m::TupleIndex(m::Receive(), 1)}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->GetNextStateElement(1), m::TupleIndex(m::Receive(), 1));
}

TEST_F(ReceiveDefaultValueSimplificationPassTest,
       TransformableConditionalReceiveWithCompoundType) {
  auto p = CreatePackage();

  Type* tuple_type = p->GetTupleType({p->GetBitsType(32), p->GetBitsType(8)});
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, tuple_type));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue pred = pb.StateElement("pred", Value(UBits(1, 1)));
  pb.StateElement("s", ZeroOfType(tuple_type));
  BValue receive = pb.ReceiveIf(out, pred);
  BValue select =
      pb.Select(pred, {pb.Literal(ZeroOfType(tuple_type)), receive});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pred, select}));

  EXPECT_THAT(proc->GetNextStateElement(1),
              m::Select(m::Param("pred"),
                        {m::Literal(), m::TupleIndex(m::Receive(), 1)}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->GetNextStateElement(1), m::TupleIndex(m::Receive(), 1));
}

TEST_F(ReceiveDefaultValueSimplificationPassTest,
       TransformableNonblockingReceive) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  pb.StateElement("s", Value(UBits(42, 32)));
  auto [data, valid] = pb.ReceiveNonBlocking(out);
  BValue select = pb.Select(valid, {pb.Literal(UBits(0, 32)), data});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({select}));

  EXPECT_THAT(proc->GetNextStateElement(0),
              m::Select(m::TupleIndex(),
                        {m::Literal(0), m::TupleIndex(m::Receive(), 1)}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->GetNextStateElement(0), m::TupleIndex(m::Receive(), 1));
}

TEST_F(ReceiveDefaultValueSimplificationPassTest,
       TransformableConditionalNonblockingReceive) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue pred = pb.StateElement("pred", Value(UBits(1, 1)));
  pb.StateElement("s", Value(UBits(42, 32)));
  auto [data, valid] = pb.ReceiveIfNonBlocking(out, pred);
  BValue select = pb.Select(valid, {pb.Literal(UBits(0, 32)), data});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pred, select}));

  EXPECT_THAT(proc->GetNextStateElement(1),
              m::Select(m::TupleIndex(),
                        {m::Literal(0), m::TupleIndex(m::Receive(), 1)}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->GetNextStateElement(1), m::TupleIndex(m::Receive(), 1));
}

TEST_F(ReceiveDefaultValueSimplificationPassTest, SelectWithArmsSwitched) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue pred = pb.StateElement("pred", Value(UBits(1, 1)));
  pb.StateElement("s", Value(UBits(42, 32)));
  BValue receive = pb.ReceiveIf(out, pred);
  // Select arms are in the wrong position for transformation.
  BValue select = pb.Select(pred, {receive, pb.Literal(UBits(0, 32))});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pred, select}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ReceiveDefaultValueSimplificationPassTest,
       SelectWithUnconditionalReceive) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue pred = pb.StateElement("pred", Value(UBits(1, 1)));
  pb.StateElement("s", Value(UBits(42, 32)));
  BValue receive = pb.Receive(out);
  BValue select = pb.Select(pred, {pb.Literal(UBits(0, 32)), receive});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pred, select}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ReceiveDefaultValueSimplificationPassTest, SelectWithNonZeroCase) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue pred = pb.StateElement("pred", Value(UBits(1, 1)));
  pb.StateElement("s", Value(UBits(42, 32)));
  BValue receive = pb.ReceiveIf(out, pred);
  BValue select = pb.Select(pred, {pb.Literal(UBits(123, 32)), receive});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pred, select}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ReceiveDefaultValueSimplificationPassTest,
       SelectWithDifferentPredicate) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue pred0 = pb.StateElement("pred0", Value(UBits(1, 1)));
  BValue pred1 = pb.StateElement("pred1", Value(UBits(1, 1)));
  pb.StateElement("s", Value(UBits(42, 32)));
  BValue receive = pb.ReceiveIf(out, pred0);
  BValue select = pb.Select(pred1, {pb.Literal(UBits(123, 32)), receive});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pred0, pred1, select}));

  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
