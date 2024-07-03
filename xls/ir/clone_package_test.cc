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

#include "xls/ir/clone_package.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_testutils.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {
using solvers::z3::IsProvenTrue;
using solvers::z3::TryProveEquivalence;
using status_testing::IsOkAndHolds;

class ClonePackageTest : public IrTestBase {};

TEST_F(ClonePackageTest, BasicFunc) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  fb.Add(fb.Param("x", p->GetBitsType(32)), fb.Literal(UBits(32, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  FunctionBuilder fb2("func2", p.get());
  fb2.Invoke({fb2.Param("abc", p->GetBitsType(32))}, f);
  XLS_ASSERT_OK(fb2.Build().status());

  XLS_ASSERT_OK_AND_ASSIGN(auto p2, ClonePackage(p.get(), "foobar"));
  EXPECT_EQ(p2->name(), "foobar");
  XLS_ASSERT_OK_AND_ASSIGN(Function * cf, p2->GetFunction("func"));
  ASSERT_NE(cf, f);
  EXPECT_THAT(TryProveEquivalence(f, cf), IsOkAndHolds(IsProvenTrue()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * cf2, p2->GetFunction("func2"));
  ASSERT_THAT(cf2->return_value(), m::Invoke(m::Param("abc")));
  EXPECT_EQ(cf2->return_value()->As<Invoke>()->to_apply(), cf);
}

TEST_F(ClonePackageTest, BasicProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Channel * chan, p->CreateStreamingChannel(
                                               "chan", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));
  ProcBuilder pb("prc", p.get());
  auto st = pb.StateElement("foo", UBits(32, 32));
  auto nv = pb.TupleIndex(pb.Receive(chan, pb.Literal(Value::Token())), 1);
  auto nv_even = pb.BitSlice(nv, 0, 1);
  pb.Next(st, pb.Add(st, nv), nv_even);
  pb.Next(st, st, pb.Not(nv_even));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto p2, ClonePackage(p.get(), "foobar"));
  EXPECT_EQ(p2->name(), "foobar");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * cp, p2->GetProc("prc"));
  ASSERT_NE(cp, proc);
  XLS_ASSERT_OK_AND_ASSIGN(Function * original_f,
                           UnrollProcToFunction(proc, /*activation_count=*/10,
                                                /*include_state=*/true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * cloned_f,
                           UnrollProcToFunction(cp, /*activation_count=*/10,
                                                /*include_state=*/true));
  EXPECT_THAT(TryProveEquivalence(original_f, cloned_f),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(ClonePackageTest, BasicBlock) {
  auto p = CreatePackage();
  BlockBuilder bb("blk", p.get());
  bb.OutputPort("foo", bb.Add(bb.InputPort("bar", p->GetBitsType(32)),
                              bb.Literal(UBits(32, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());
  BlockBuilder bb2("blk2", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto inst,
                           bb2.block()->AddBlockInstantiation("foo", blk));
  bb2.InstantiationInput(inst, "bar",
                         bb2.InputPort("pass_in", p->GetBitsType(32)));
  bb2.OutputPort("pass_out", bb2.InstantiationOutput(inst, "foo"));
  XLS_ASSERT_OK(bb2.Build().status());

  XLS_ASSERT_OK_AND_ASSIGN(auto p2, ClonePackage(p.get(), "foobar"));
  EXPECT_EQ(p2->name(), "foobar");
  XLS_ASSERT_OK_AND_ASSIGN(Block * cb, p2->GetBlock("blk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * cb2, p2->GetBlock("blk2"));
  ASSERT_NE(cb, blk);
  // TODO(allight): It would be nice to have z3 provers for block equiv.
  EXPECT_THAT(cb->GetOutputPort("foo"),
              status_testing::IsOkAndHolds(m::OutputPort(
                  m::Add(m::InputPort("bar"), m::Literal(UBits(32, 32))))));
  EXPECT_THAT(cb2->GetOutputPort("pass_out"),
              status_testing::IsOkAndHolds(
                  m::OutputPort(m::InstantiationOutput("foo"))));
}

}  // namespace
}  // namespace xls
