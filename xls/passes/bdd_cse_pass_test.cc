// Copyright 2020 The XLS Authors
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

#include "xls/passes/bdd_cse_pass.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class BddCsePassTest : public IrTestBase {
 protected:
  BddCsePassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;
    return BddCsePass().RunOnFunctionBase(f, OptimizationPassOptions(),
                                          &results, context);
  }
};

TEST_F(BddCsePassTest, EqEquivalentToNotNe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue forty_two = fb.Literal(UBits(42, 16));
  BValue x_eq_42 = fb.Eq(x, forty_two);
  BValue forty_two_not_ne_x = fb.Not(fb.Ne(forty_two, x));
  fb.Tuple({x_eq_42, forty_two_not_ne_x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Eq(m::Param("x"), m::Literal(42)),
                       m::Eq(m::Param("x"), m::Literal(42))));
}

TEST_F(BddCsePassTest, DifferentExpressions) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(16));
  BValue forty_two = fb.Literal(UBits(42, 16));
  BValue x_eq_42 = fb.Eq(x, forty_two);
  BValue forty_two_not_ne_y = fb.Not(fb.Ne(forty_two, y));
  fb.Tuple({x_eq_42, forty_two_not_ne_y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BddCsePassTest, DecodeEquivalentToDeconstructedDecode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue decode_x = fb.Decode(x);
  BValue hacky_decode_x = fb.Concat(
      {fb.Eq(x, fb.Literal(UBits(3, 2))), fb.Eq(x, fb.Literal(UBits(2, 2))),
       fb.Eq(x, fb.Literal(UBits(1, 2))), fb.Eq(x, fb.Literal(UBits(0, 2)))});
  fb.Tuple({decode_x, hacky_decode_x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Tuple(m::Decode(), m::Decode()));
}

// Guarded conditional value propagation: `w` is only realized as arm 1 of the
// final select, and under that arm (p == 1, i.e. b == c) `w == a`.
// The arm-1 operand should be rewritten to `a`.
TEST_F(BddCsePassTest, GuardedValuePropagationXorEq) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue c = fb.Param("c", p->GetBitsType(4));
  BValue d = fb.Param("d", p->GetBitsType(4));
  BValue p_eq = fb.Eq(b, c);
  // w == a ^ (b ^ c). Under b == c, w == a. w is only used as arm 1 below.
  BValue w = fb.Xor(a, fb.Xor(b, c));
  BValue result = fb.Select(p_eq, /*cases=*/{d, w});
  fb.Tuple({result});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(m::Eq(m::Param("b"), m::Param("c")),
                                 /*cases=*/{m::Param("d"), m::Param("a")})));
}

// The guarded check folds a width-64 `eq` arm via the cheap per-bit surrogate
// (each `b[i]==c[i]` independently) instead of materializing the wide `eq` BDD,
// which would saturate the default 1024 path limit.
TEST_F(BddCsePassTest, GuardedValuePropagationWiderThanPathLimit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(64));
  BValue b = fb.Param("b", p->GetBitsType(64));
  BValue c = fb.Param("c", p->GetBitsType(64));
  BValue d = fb.Param("d", p->GetBitsType(64));
  BValue p_eq = fb.Eq(b, c);
  BValue w = fb.Xor(a, fb.Xor(b, c));
  BValue result = fb.Select(p_eq, /*cases=*/{d, w});
  fb.Tuple({result});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(m::Eq(m::Param("b"), m::Param("c")),
                                 /*cases=*/{m::Param("d"), m::Param("a")})));
}

// `Ne` counterpart of GuardedValuePropagationWiderThanPathLimit: `w` is arm 0,
// live when `ne(b,c)==0` i.e. `b==c`, so the per-bit surrogate folds it to `a`.
TEST_F(BddCsePassTest, GuardedValuePropagationNeArmZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue d = fb.Param("d", p->GetBitsType(8));
  BValue p_ne = fb.Ne(b, c);
  BValue w = fb.Xor(a, fb.Xor(b, c));
  // if p { d } else { w }: w is arm 0, live when ne==0 i.e. b==c.
  BValue result = fb.Select(p_ne, /*cases=*/{w, d});
  fb.Tuple({result});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(m::Ne(m::Param("b"), m::Param("c")),
                                 /*cases=*/{m::Param("a"), m::Param("d")})));
}

// Soundness (no false positive): `w` is arm 1 of `sel(ne(b,c), [d, w])`, live
// when `b != c`. The per-bit surrogate assumes `b == c`, the wrong predicate
// for this arm, so it must NOT fire; the arm must remain `w`.
TEST_F(BddCsePassTest,
       GuardedValuePropagationNeArmOneIsNegativeNotMisoptimized) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue d = fb.Param("d", p->GetBitsType(8));
  BValue p_ne = fb.Ne(b, c);
  BValue w = fb.Xor(a, fb.Xor(b, c));
  // if p { w } else { d }: w is arm 1, live when ne==1 i.e. b != c.
  BValue result = fb.Select(p_ne, /*cases=*/{d, w});
  fb.Tuple({result});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(
                  m::Ne(m::Param("b"), m::Param("c")),
                  /*cases=*/{m::Param("d"),
                             m::Xor(m::Param("a"),
                                    m::Xor(m::Param("b"), m::Param("c")))})));
}

// Soundness (no false positive): `w` is arm 0 of `sel(eq(b,c), [w, d])`, live
// when `b != c`. The per-bit surrogate assumes `b == c`, the wrong predicate
// for this arm, so it must NOT fire; the arm must remain `w`.
TEST_F(BddCsePassTest,
       GuardedValuePropagationEqArmZeroIsNegativeNotMisoptimized) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue d = fb.Param("d", p->GetBitsType(8));
  BValue p_eq = fb.Eq(b, c);
  BValue w = fb.Xor(a, fb.Xor(b, c));
  // if p { d } else { w }: w is arm 0, live when eq==0 i.e. b != c.
  BValue result = fb.Select(p_eq, /*cases=*/{w, d});
  fb.Tuple({result});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(
                  m::Eq(m::Param("b"), m::Param("c")),
                  /*cases=*/{m::Xor(m::Param("a"),
                                    m::Xor(m::Param("b"), m::Param("c"))),
                             m::Param("d")})));
}

// Soundness: even when `w` is also used unguarded elsewhere, the arm-1 edge may
// still fold to `a`, but the unguarded use must keep the original `w`.
TEST_F(BddCsePassTest, GuardedValuePropagationKeepsOtherUses) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue c = fb.Param("c", p->GetBitsType(4));
  BValue d = fb.Param("d", p->GetBitsType(4));
  BValue p_eq = fb.Eq(b, c);
  BValue w = fb.Xor(a, fb.Xor(b, c));
  BValue arm1 = fb.Select(p_eq, /*cases=*/{d, w});
  // `w` is also used unconditionally here; it must remain `w`.
  BValue other = fb.Add(w, fb.Literal(UBits(1, 4)));
  fb.Tuple({arm1, other});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(m::Eq(m::Param("b"), m::Param("c")),
                                 /*cases=*/{m::Param("d"), m::Param("a")}),
                       m::Add(m::Xor(m::Param("a"),
                                     m::Xor(m::Param("b"), m::Param("c"))),
                              m::Literal(1))));
}

// `w = a ^ ((b&c) ^ (b|c))` folds to `a` under `eq(b,c)`: the boolean noise
// term `(b&c) ^ (b|c)` is `0` when `b == c`.
TEST_F(BddCsePassTest, GuardedValuePropagationBooleanNoiseTerm) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue d = fb.Param("d", p->GetBitsType(8));
  BValue p_eq = fb.Eq(b, c);
  BValue noise = fb.Xor(fb.And(b, c), fb.Or(b, c));
  BValue w = fb.Xor(a, noise);
  BValue result = fb.Select(p_eq, /*cases=*/{d, w});
  fb.Tuple({result});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(m::Eq(m::Param("b"), m::Param("c")),
                                 /*cases=*/{m::Param("d"), m::Param("a")})));
}

// A guarded fold on a nested select: `sel_inner = (15 >> s)[0]` is 1 for every
// `s < 4`, the exact region where the outer `sel(or_reduce(s[2:]), [inner, x])`
// arm-0 is live, so `inner = sel(sel_inner, [A, B])` collapses to `B`.
TEST_F(BddCsePassTest, GuardedValuePropagationNestedSelectUnderOrReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue s = fb.Param("s", p->GetBitsType(8));
  // g == 0 iff s < 4.
  BValue g = fb.OrReduce(fb.BitSlice(s, /*start=*/2, /*width=*/6));
  // sel_inner == 1 for every s in {0,1,2,3} (the g == 0 region).
  BValue sel_inner = fb.BitSlice(fb.Shrl(fb.Literal(UBits(15, 8)), s),
                                 /*start=*/0, /*width=*/1);
  // inner == B whenever sel_inner == 1.
  BValue inner = fb.Select(sel_inner, /*cases=*/{a, b});
  BValue outer = fb.Select(g, /*cases=*/{inner, x});
  fb.Tuple({outer});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Tuple(m::Select(m::OrReduce(m::BitSlice(m::Param("s"), /*start=*/2,
                                                 /*width=*/6)),
                         /*cases=*/{m::Param("b"), m::Param("x")})));
}

// The per-bit surrogate only recognizes a `CompareOp` selector, so a guard
// equivalent to `eq(b,c)` but shaped as `and_reduce(not(xor(b,c)))` is driven
// by the exact path; the select must stay under the same guard (structural
// soundness).
TEST_F(BddCsePassTest, GuardedValuePropagationAndReduceGuardFoldsSoundly) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue d = fb.Param("d", p->GetBitsType(8));
  // 1-bit `and_reduce(not(xor(b,c)))` == `b == c`, but not a `CompareOp`.
  BValue guard = fb.AndReduce(fb.Not(fb.Xor(b, c)));
  BValue w = fb.Xor(a, fb.Xor(b, c));
  BValue result = fb.Select(guard, /*cases=*/{d, w});
  fb.Tuple({result});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // Must run cleanly (no crash) and keep the select under the same guard (no
  // miscompile), regardless of whether a fold fires.
  ASSERT_THAT(Run(f), ::absl_testing::IsOk());
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(
                  m::AndReduce(m::Not(m::Xor(m::Param("b"), m::Param("c")))),
                  /*cases=*/{::testing::_, ::testing::_})));
}

void IrFuzzBddCse(FuzzPackageWithArgs fuzz_package_with_args) {
  BddCsePass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzBddCse)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
