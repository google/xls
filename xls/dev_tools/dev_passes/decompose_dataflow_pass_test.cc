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

#include "xls/dev_tools/dev_passes/decompose_dataflow_pass.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_test_helpers.h"
#include "xls/solvers/ir_equivalence_testutils.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {
using absl_testing::IsOkAndHolds;

using solvers::ScopedVerifyEquivalence;
using solvers::ScopedVerifyProcEquivalence;

class DecomposeDataflowPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(const std::unique_ptr<Package>& p) {
    return Run(p.get());
  }
  absl::StatusOr<bool> Run(Package* p) {
    OptimizationCompoundPass pass("test_pass", "test_pass");
    bool changed = false;
    pass.Add<RecordIfPassChanged<DecomposeDataflowPass>>(&changed);
    pass.Add<DeadCodeEliminationPass>();
    PassResults results;
    OptimizationContext context;
    XLS_RETURN_IF_ERROR(pass.Run(p, {}, &results, context).status());
    return changed;
  }
};

TEST_F(DecomposeDataflowPassTest, BasicTest) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(4, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.ArrayIndex(x, {y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("y"),
                        {m::ArrayIndex(m::Param("x"), {m::Literal(0)}),
                         m::ArrayIndex(m::Param("x"), {m::Literal(1)}),
                         m::ArrayIndex(m::Param("x"), {m::Literal(2)})},
                        m::ArrayIndex(m::Param("x"), {m::Literal(3)})));
}

TEST_F(DecomposeDataflowPassTest, ProcTest) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue idx = pb.StateElement("idx", Value(UBits(0, 32)));
  BValue array = pb.Literal(Value::ArrayOrDie({
      Value(UBits(10, 32)),
      Value(UBits(20, 32)),
      Value(UBits(30, 32)),
      Value(UBits(40, 32)),
  }));
  BValue val = pb.ArrayIndex(array, {idx});
  BValue next_idx = pb.Add(idx, pb.Literal(UBits(1, 32)));
  BValue send = pb.Send(out, val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({next_idx}));

  ScopedVerifyProcEquivalence svpe(proc, /*activation_count=*/3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(send.node()->As<Send>()->data(),
              m::Select(m::StateRead("idx"),
                        {m::ArrayIndex(m::Literal(), {m::Literal(0)}),
                         m::ArrayIndex(m::Literal(), {m::Literal(1)}),
                         m::ArrayIndex(m::Literal(), {m::Literal(2)})},
                        m::ArrayIndex(m::Literal(), {m::Literal(3)})));
}

TEST_F(DecomposeDataflowPassTest, DecoupledProcPassThroughTest) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), "p", "tkn", p.get());
  BStateElement state_element =
      pb.UnreadStateElement("st", Value(UBits(42, 32)),
                            /*non_synthesizable=*/false);
  BValue st = pb.StateRead(state_element);
  pb.Next(state_element, st);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  ASSERT_TRUE(proc->uses_decoupled_next());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(DecomposeDataflowPassTest, OneHotSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetArrayType(2, p->GetBitsType(32)));
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  fb.OneHotSelect(sel, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Array(m::OneHotSelect(), m::OneHotSelect()));
}

TEST_F(DecomposeDataflowPassTest, Gate) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue cond = fb.Param("cond", p->GetBitsType(1));
  fb.Gate(cond, x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Array(m::Gate(), m::Gate()));
}

TEST_F(DecomposeDataflowPassTest, ArrayUpdate) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue v = fb.Param("v", p->GetBitsType(32));
  BValue idx = fb.Param("idx", p->GetBitsType(1));
  fb.ArrayUpdate(x, v, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Array(m::Select(), m::Select()));
}

TEST_F(DecomposeDataflowPassTest, ArraySlice) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(4, p->GetBitsType(32)));
  BValue start = fb.Literal(UBits(1, 32));
  fb.ArraySlice(x, start, 2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Array(m::Select(), m::Select()));
}

TEST_F(DecomposeDataflowPassTest, ArrayConcat) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetArrayType(2, p->GetBitsType(32)));
  fb.ArrayConcat({x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Array(m::ArrayIndex(), m::ArrayIndex(),
                                          m::ArrayIndex(), m::ArrayIndex()));
}

TEST_F(DecomposeDataflowPassTest, DefaultHandlerTupleIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x =
      fb.Param("x", p->GetTupleType({p->GetBitsType(32), p->GetBitsType(8)}));
  fb.TupleIndex(x, 0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
}

TEST_F(DecomposeDataflowPassTest, Select) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetArrayType(2, p->GetBitsType(32)));
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  fb.Select(sel, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Array(m::Select(), m::Select()));
}

TEST_F(DecomposeDataflowPassTest, PrioritySelect) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetArrayType(2, p->GetBitsType(32)));
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  fb.PrioritySelect(sel, {x}, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Array(m::PrioritySelect(), m::PrioritySelect()));
}

TEST_F(DecomposeDataflowPassTest, Invoke) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb("sub", p.get());
  BValue sub_x = sub_fb.Param("sub_x", p->GetArrayType(2, p->GetBitsType(32)));
  sub_fb.ArrayIndex(sub_x, {sub_fb.Literal(UBits(0, 32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * sub, sub_fb.Build());

  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue inv = fb.Invoke({x}, sub);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(inv.node()->operand(0),
              m::Array(m::ArrayIndex(m::Param("x"), {m::Literal(0)}),
                       m::ArrayIndex(m::Param("x"), {m::Literal(1)})));
}

TEST_F(DecomposeDataflowPassTest, Map) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb("sub", p.get());
  sub_fb.Param("sub_x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * sub, sub_fb.Build());

  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue map_op = fb.Map(x, sub);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(map_op.node()->operand(0),
              m::Array(m::ArrayIndex(m::Param("x"), {m::Literal(0)}),
                       m::ArrayIndex(m::Param("x"), {m::Literal(1)})));
}

TEST_F(DecomposeDataflowPassTest, CountedFor) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb("sub", p.get());
  sub_fb.Param("i", p->GetBitsType(32));
  BValue sub_acc =
      sub_fb.Param("sub_acc", p->GetArrayType(2, p->GetBitsType(32)));
  sub_fb.Identity(sub_acc);
  XLS_ASSERT_OK_AND_ASSIGN(Function * sub, sub_fb.Build());

  FunctionBuilder fb("f", p.get());
  BValue acc = fb.Param("acc", p->GetArrayType(2, p->GetBitsType(32)));
  BValue for_op = fb.CountedFor(acc, 2, 1, sub);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(for_op.node()->operand(0),
              m::Array(m::ArrayIndex(m::Param("acc"), {m::Literal(0)}),
                       m::ArrayIndex(m::Param("acc"), {m::Literal(1)})));
}

TEST_F(DecomposeDataflowPassTest, Eq) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetArrayType(2, p->GetBitsType(32)));
  fb.Eq(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::And(m::Eq(m::ArrayIndex(), m::ArrayIndex()),
                     m::Eq(m::ArrayIndex(), m::ArrayIndex())));
}

TEST_F(DecomposeDataflowPassTest, Ne) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetArrayType(2, p->GetBitsType(32)));
  fb.Ne(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Or(m::Ne(m::ArrayIndex(), m::ArrayIndex()),
                    m::Ne(m::ArrayIndex(), m::ArrayIndex())));
}

TEST_F(DecomposeDataflowPassTest, ArrayIndexNarrowSelectorRegression) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetArrayType(100, p->GetBitsType(64)));
  BValue y = fb.Param("y", p->GetBitsType(1));
  fb.ArrayIndex(x, {y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
}

void IrFuzzDecomposeDataflow(FuzzPackageWithArgs fuzz_package_with_args) {
  OptimizationCompoundPass pass("test_pass", "test_pass");
  pass.Add<DecomposeDataflowPass>();
  pass.Add<DeadCodeEliminationPass>();
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzDecomposeDataflow)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
