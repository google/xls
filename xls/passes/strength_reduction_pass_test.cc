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

#include "xls/passes/strength_reduction_pass.h"

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
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"
#include "xls/solvers/z3_ir_translator.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::z3::ScopedVerifyEquivalence;

using ::testing::_;
using ::testing::Each;
using ::testing::UnorderedElementsAre;
using ::testing::VariantWith;

class StrengthReductionPassTest : public IrTestBase {
 protected:
  StrengthReductionPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         StrengthReductionPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results, context));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results, context)
            .status());
    // Return whether strength reduction changed anything.
    return changed;
  }
};

class StrengthReductionPassSemanticsTest : public StrengthReductionPassTest {
 public:
  void TestReductionIsEquivalent(FunctionBuilder& fb) {
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TestReductionIsEquivalent(f);
  }
  void TestReductionIsEquivalent(Function* f) {
    EXPECT_THAT(solvers::z3::TryProveEquivalence(
                    f,
                    [&](auto package, auto function) {
                      return StrengthReductionPassTest::Run(function).status();
                    }),
                IsOkAndHolds(VariantWith<solvers::z3::ProvenTrue>(_)))
        << "Pass changed meaning of the function";
  }
};

TEST_F(StrengthReductionPassTest, ReducibleAdd) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[8], y: bits[8]) -> bits[24] {
       literal.1: bits[16] = literal(value=0)
       concat.2: bits[24] = concat(x, literal.1)
       concat.3: bits[24] = concat(literal.1, y)
       ret add.4: bits[24] = add(concat.2, concat.3)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kAdd);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Or(m::Concat(m::Param("x"), m::Literal(0)),
                    m::Concat(m::Literal(0), m::Param("y"))));
}

TEST_F(StrengthReductionPassTest, NotReducibleAdd) {
  auto p = CreatePackage();
  // Bit 7 of the add's operands both come from parameter inputs and so cannot
  // be determined to be zero.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[8], y: bits[8]) -> bits[15] {
       literal.1: bits[7] = literal(value=0)
       concat.2: bits[15] = concat(x, literal.1)
       concat.3: bits[15] = concat(literal.1, y)
       ret add.4: bits[15] = add(concat.2, concat.3)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kAdd);
  // The Add is narrowed, but cannot be completely replaced by an Or.
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat(m::Add(), m::BitSlice()));
}

TEST_F(StrengthReductionPassTest, OneBitAddToXor) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[1], y: bits[1]) -> bits[1] {
       ret add.3: bits[1] = add(x, y, pos=[(0,1,2)])
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kAdd);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::Param("x"), m::Param("y")));
}

TEST_F(StrengthReductionPassTest, ConcatZeroThenSignExt) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[1]) -> bits[3] {
       literal.2: bits[1] = literal(value=0)
       concat.3: bits[2] = concat(literal.2, s)
       ret sign_ext.4: bits[3] = sign_ext(concat.3, new_bit_count=3)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kSignExt);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(m::Concat(m::Literal(0), m::Param())));
}

TEST_F(StrengthReductionPassTest, AndWithMask) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[4]) -> bits[4] {
       literal.2: bits[4] = literal(value=0b0110)
       ret and.3: bits[4] = and(x, literal.2)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kAnd);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(0),
                        m::BitSlice(m::Param(), /*start=*/1, /*width=*/2),
                        m::Literal(0)));
}

TEST_F(StrengthReductionPassTest, AndWithEffectiveMaskToBitSliceConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[4]) -> bits[4] {
       literal.2: bits[4] = literal(value=0)
       literal.3: bits[4] = literal(value=0b0110)
       or.4: bits[4] = or(literal.2, literal.3)
       ret and.5: bits[4] = and(x, or.4)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kAnd);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(0),
                        m::BitSlice(m::Param(), /*start=*/1, /*width=*/2),
                        m::Literal(0)));
}

TEST_F(StrengthReductionPassTest, UGeWithMsbSet) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[10]) -> bits[1] {
       literal.1: bits[10] = literal(value=512)
       ret uge.2: bits[1] = uge(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kUGe);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Ne(m::BitSlice(m::Param(), /*start=*/9, /*width=*/1), m::Literal(0)));
}

TEST_F(StrengthReductionPassTest, ULtWithMsbSet) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[10]) -> bits[1] {
       literal.1: bits[10] = literal(value=512)
       ret ult.2: bits[1] = ult(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kULt);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Eq(m::BitSlice(m::Param(), /*start=*/9, /*width=*/1), m::Literal(0)));
}

TEST_F(StrengthReductionPassTest, UGeWithLeadingBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[10]) -> bits[1] {
       literal.1: bits[10] = literal(value=256)
       ret result: bits[1] = uge(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kUGe);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Ne(m::BitSlice(m::Param(), /*start=*/8, /*width=*/2), m::Literal(0)));
}

TEST_F(StrengthReductionPassTest, ULtWithLeadingBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[10]) -> bits[1] {
       literal.1: bits[10] = literal(value=256)
       ret ult.2: bits[1] = ult(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kULt);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Eq(m::BitSlice(m::Param(), /*start=*/8, /*width=*/2), m::Literal(0)));
}

TEST_F(StrengthReductionPassTest, TwoBitEq) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[2]) -> bits[1] {
       literal.1: bits[2] = literal(value=0)
       ret eq.2: bits[1] = eq(x, literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Not(m::Or(m::BitSlice(m::Param(), /*start=*/0, /*width=*/1),
                           m::BitSlice(m::Param(), /*start=*/1, /*width=*/1))));
}

TEST_F(StrengthReductionPassTest, NarrowAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(
      fb.Param("x", p->GetBitsType(32)),
      fb.Concat({fb.Param("y", p->GetBitsType(16)), fb.Literal(UBits(0, 16))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Add(m::BitSlice(m::Param("x"), /*start=*/16, /*width=*/16),
                       m::BitSlice(m::Concat(), /*start=*/16, /*width=*/16)),
                m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/16)));
}

TEST_F(StrengthReductionPassTest, NarrowAddsBothOperandsLsbZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(
      fb.Concat({fb.Param("x", p->GetBitsType(20)), fb.Literal(UBits(0, 12))}),
      fb.Concat({fb.Param("y", p->GetBitsType(10)), fb.Literal(UBits(0, 22))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Add(m::BitSlice(m::Concat(), /*start=*/22, /*width=*/10),
                       m::BitSlice(m::Concat(), /*start=*/22, /*width=*/10)),
                m::BitSlice()));
}

TEST_F(StrengthReductionPassTest, SignExtMux) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[1]) -> bits[16] {
       literal.2: bits[16] = literal(value=0)
       literal.3: bits[16] = literal(value=0xffff)
       ret sel.4: bits[16] = sel(s, cases=[literal.2, literal.3])
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kSel);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::SignExt(m::Param("s")));
}

TEST_F(StrengthReductionPassTest, SignExtMuxNegated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[1]) -> bits[16] {
       literal.2: bits[16] = literal(value=0)
       literal.3: bits[16] = literal(value=0xffff)
       ret sel.4: bits[16] = sel(s, cases=[literal.3, literal.2])
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kSel);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::SignExt(m::Not(m::Param("s"))));
}

TEST_F(StrengthReductionPassTest, GateKnownZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[8]) -> bits[8] {
       literal.2: bits[1] = literal(value=0)
       ret gate.3: bits[8] = gate(literal.2, s)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kGate);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_F(StrengthReductionPassTest, GateKnownOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[8]) -> bits[8] {
       literal.2: bits[1] = literal(value=1)
       ret gate.3: bits[8] = gate(literal.2, s)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kGate);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("s"));
}

TEST_F(StrengthReductionPassTest, GateKnownDataZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[1]) -> bits[8] {
       literal.2: bits[8] = literal(value=0)
       ret gate.3: bits[8] = gate(s, literal.2)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kGate);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_F(StrengthReductionPassSemanticsTest, ArithToSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("unknown", p->GetBitsType(1));
  // Value is 42 (0b101010) or 40 (0b101000)
  BValue big_unknown = fb.Concat(
      {fb.Literal(UBits(0b1010, 62)), param, fb.Literal(UBits(0, 1))});
  fb.Tuple({
      fb.UMul(big_unknown, fb.Literal(UBits(10, 64))),    // 400 or 420
      fb.UMul(fb.Literal(UBits(10, 64)), big_unknown),    // 400 or 420
      fb.SMul(big_unknown, fb.Literal(UBits(-10, 64))),   // -400 or -420
      fb.SMul(fb.Literal(UBits(-12, 64)), big_unknown),   // -400 or -420
      fb.UDiv(big_unknown, fb.Literal(UBits(2, 64))),     // 20 or 21
      fb.UDiv(fb.Literal(UBits(84, 64)), big_unknown),    // 2 or 1
      fb.SDiv(big_unknown, fb.Literal(UBits(-2, 64))),    // -20 or -21
      fb.SDiv(fb.Literal(UBits(-84, 64)), big_unknown),   // -2 or -1
      fb.UMod(big_unknown, fb.Literal(UBits(7, 64))),     // 0 or 5
      fb.UMod(fb.Literal(UBits(120, 64)), big_unknown),   // 36 or 0
      fb.SMod(big_unknown, fb.Literal(UBits(-7, 64))),    // 0 or -5
      fb.SMod(fb.Literal(UBits(-120, 64)), big_unknown),  // -36 or 0
  });
  TestReductionIsEquivalent(fb);
}

TEST_F(StrengthReductionPassTest, ArithToSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("unknown", p->GetBitsType(1));
  // Value is 42 (0b101010) or 40 (0b101000)
  BValue big_unknown = fb.Concat(
      {fb.Literal(UBits(0b1010, 62)), param, fb.Literal(UBits(0, 1))});
  fb.Tuple({
      fb.UMul(big_unknown, fb.Literal(UBits(10, 64))),    // 400 or 420
      fb.UMul(fb.Literal(UBits(10, 64)), big_unknown),    // 400 or 420
      fb.SMul(big_unknown, fb.Literal(UBits(-10, 64))),   // -400 or -420
      fb.SMul(fb.Literal(UBits(-12, 64)), big_unknown),   // -400 or -420
      fb.UDiv(big_unknown, fb.Literal(UBits(2, 64))),     // 20 or 21
      fb.UDiv(fb.Literal(UBits(84, 64)), big_unknown),    // 2 or 1
      fb.SDiv(big_unknown, fb.Literal(UBits(-2, 64))),    // -20 or -21
      fb.SDiv(fb.Literal(UBits(-84, 64)), big_unknown),   // -2 or -1
      fb.UMod(big_unknown, fb.Literal(UBits(7, 64))),     // 0 or 5
      fb.UMod(fb.Literal(UBits(120, 64)), big_unknown),   // 36 or 0
      fb.SMod(big_unknown, fb.Literal(UBits(-7, 64))),    // 0 or -5
      fb.SMod(fb.Literal(UBits(-120, 64)), big_unknown),  // -36 or 0
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // Actual verification of result is done by semantics test.
  EXPECT_THAT(f->return_value()->operands(),
              Each(m::Select(m::Eq(), {m::Literal(), m::Literal()})));
}

TEST_F(StrengthReductionPassTest, ArithToSelectOnlyWithOneBit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // 4 unknown bits.
  BValue param = fb.Param("unknown", p->GetBitsType(4));
  BValue big_unknown =
      fb.Concat({fb.Literal(UBits(0b101, 59)), param, fb.Literal(UBits(0, 1))});
  fb.UMul(big_unknown, fb.Literal(UBits(10, 64)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false))
      << "Optimization triggered unexpectedly. Got:\n"
      << f->DumpIr();
}

TEST_F(StrengthReductionPassTest, PushDownSelectValues) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Literal(UBits(3, 32)),
         fb.Select(fb.Param("selector", p->GetBitsType(1)),
                   {fb.Literal(UBits(1, 32)), fb.Literal(UBits(2, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param(),
                {
                    m::Add(m::Literal(UBits(3, 32)), m::Literal(UBits(1, 32))),
                    m::Add(m::Literal(UBits(3, 32)), m::Literal(UBits(2, 32))),
                }))
      << f->DumpIr();
}

TEST_F(StrengthReductionPassTest, DoNotPushDownCheapExtendingOps) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SignExtend(
      fb.Select(fb.Param("selector", p->GetBitsType(1)),
                {fb.Literal(UBits(0xFFFFFFFF, 32)), fb.Literal(UBits(2, 32))}),
      64);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false)) << f->DumpIr();
}

// This is something we might want to support at some point.
TEST_F(StrengthReductionPassTest, DoNotPushDownMultipleSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(
      fb.Select(fb.Param("selector", p->GetBitsType(1)),
                {fb.Literal(UBits(0xFFFFFFFF, 32)), fb.Literal(UBits(2, 32))}),
      fb.Select(fb.Param("selector2", p->GetBitsType(1)),
                {fb.Literal(UBits(33, 32)), fb.Literal(UBits(45, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false)) << f->DumpIr();
}

TEST_F(StrengthReductionPassTest, ReplaceWidth0Param) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue ret = fb.Or(fb.ZeroExtend(fb.Param("x", p->GetBitsType(0)), 1),
                     fb.Param("y", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  EXPECT_THAT(f->nodes(), UnorderedElementsAre(
                              m::Param("x"), m::Param("y"), m::ZeroExt(),
                              m::Or(m::ZeroExt(m::Param("x")), m::Param("y"))));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->nodes(),
              UnorderedElementsAre(m::Param("x"), m::Param("y"), m::Literal(),
                                   m::Or(m::Literal(0), m::Param("y"))));
}

TEST_F(StrengthReductionPassTest, DoNotReplaceUnusedWidth0Param) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(0));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Param("y", p->GetBitsType(1))));
  EXPECT_THAT(f->nodes(), UnorderedElementsAre(m::Param("x"), m::Param("y")));
  // Normally, the empty param would be replaced with a literal, but since it
  // is unused, it doesn't get replaced.
  // Replacing unused params with literals can lead to an infinite loop with
  // strength reduction adding the literal and DCE removing it.
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->nodes(), UnorderedElementsAre(m::Param("x"), m::Param("y")));
}

TEST_F(StrengthReductionPassTest, HandlesOneBitMuxWithDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func() -> bits[1] {
       literal.3: bits[1] = literal(value=0, id=3)
       ret x6__1: bits[1] = sel(literal.3, cases=[literal.3], default=literal.3, id=6, pos=[(0,12,27)])
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
}

TEST_F(StrengthReductionPassTest, AddOneOperandZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Literal(UBits(0b00000, 4));
  BValue rhs = fb.Param("p", p->GetBitsType(4));
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("p"));
}

TEST_F(StrengthReductionPassTest, AddOneOperandZeroWidth1) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Param("p", p->GetBitsType(1));
  BValue rhs = fb.Literal(UBits(0b0, 1));
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("p"));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarry) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(4));
  BValue r = fb.Param("r", p->GetBitsType(4));
  BValue lhs = fb.And(l, fb.Literal(UBits(0b0101, 4)));
  BValue rhs = fb.And(r, fb.Literal(UBits(0b1010, 4)));
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(lhs.node(), rhs.node()));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryOneBitSplit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(3));
  BValue r = fb.Param("r", p->GetBitsType(3));
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b1, 1))});
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b0, 1))});
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::BitSlice(lhs.node(), 1, 3),
                               m::BitSlice(rhs.node(), 1, 3)),
                        m::BitSlice(lhs.node(), 0, 1)));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryOneBitSplitBothZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(3));
  BValue r = fb.Param("r", p->GetBitsType(3));
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b0, 1))});
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b0, 1))});
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::BitSlice(lhs.node(), 1, 3),
                               m::BitSlice(rhs.node(), 1, 3)),
                        m::BitSlice(lhs.node(), 0, 1)));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryTwoBitSplit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(2));
  BValue r = fb.Param("r", p->GetBitsType(2));
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b00, 2))});
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b00, 2))});
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::BitSlice(lhs.node(), 2, 2),
                               m::BitSlice(rhs.node(), 2, 2)),
                        m::BitSlice(lhs.node(), 0, 2)));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryOneBitTernarySplit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Param("l", p->GetBitsType(4));
  BValue r = fb.Param("r", p->GetBitsType(3));
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b0, 1))});
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::BitSlice(lhs.node(), 1, 3),
                               m::BitSlice(rhs.node(), 1, 3)),
                        m::BitSlice(lhs.node(), 0, 1)));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryTwoBitTernarySplit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lp = fb.Param("lp", p->GetBitsType(4));
  BValue rp = fb.Param("rp", p->GetBitsType(4));
  BValue lhs = fb.Or(fb.And(lp, fb.Literal(UBits(0b0100, 4))),
                     fb.Literal(UBits(0b1000, 4)));
  BValue rhs = fb.Or(fb.And(rp, fb.Literal(UBits(0b0111, 4))),
                     fb.Literal(UBits(0b1000, 4)));
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::BitSlice(lhs.node(), 2, 2),
                               m::BitSlice(rhs.node(), 2, 2)),
                        m::BitSlice(rhs.node(), 0, 2)));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryAlmostFullSplit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(1));
  BValue r = fb.Param("r", p->GetBitsType(1));
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b00, 2))});
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b00, 2))});
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::BitSlice(lhs.node(), 2, 1),
                               m::BitSlice(rhs.node(), 2, 1)),
                        m::BitSlice(lhs.node(), 0, 2)));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryWideSplit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(64));
  BValue r = fb.Param("r", p->GetBitsType(64));
  BValue lit17 = fb.Literal(UBits(17, 64));
  BValue shll_l = fb.Shll(l, lit17);
  BValue shll_r = fb.Shll(r, lit17);
  fb.Add(shll_l, shll_r);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::BitSlice(m::Shll(), 17, 47),
                               m::BitSlice(m::Shll(), 17, 47)),
                        m::BitSlice(m::Shll(), 0, 17)));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryNoSplitLsbOne) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(3));
  BValue r = fb.Param("r", p->GetBitsType(3));
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b1, 1))});
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b1, 1))});
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryNoSplitLsbOneAndX) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(3));
  BValue r = fb.Param("r", p->GetBitsType(4));
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b1, 1))});
  BValue rhs = r;
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(StrengthReductionPassTest, AddTrailingNoCarryNoSplitLsbXAndX) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Param("l", p->GetBitsType(4));
  BValue rhs = fb.Param("r", p->GetBitsType(4));
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(StrengthReductionPassTest, SubTrailingNoBorrowOneBitSplitLHS) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(3));
  BValue r = fb.Param("r", p->GetBitsType(4));
  // Set lhs LSB to 1, guaranteeing no borrow to bit 0.
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b1, 1))});
  BValue rhs = r;
  fb.Subtract(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Sub(m::BitSlice(lhs.node(), 1, 3),
                               m::BitSlice(rhs.node(), 1, 3)),
                        m::Not(m::BitSlice(rhs.node(), 0, 1))));
}

TEST_F(StrengthReductionPassTest, SubTrailingNoBorrowOneBitSplitRHS) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(4));
  BValue r = fb.Param("r", p->GetBitsType(3));
  // Set rhs LSB to 0, guaranteeing no borrow to bit 0.
  BValue lhs = l;
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b0, 1))});
  fb.Subtract(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Sub(m::BitSlice(lhs.node(), 1, 3),
                               m::BitSlice(rhs.node(), 1, 3)),
                        m::BitSlice(lhs.node(), 0, 1)));
}

TEST_F(StrengthReductionPassTest, SubTrailingNoBorrowTwoBitSplitLHS) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(2));
  BValue r = fb.Param("r", p->GetBitsType(4));
  // Set lhs LSBs to 11, guaranteeing no borrow to bits 0 or 1.
  BValue lhs = fb.Concat({l, fb.Literal(UBits(0b11, 2))});
  BValue rhs = r;
  fb.Subtract(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Sub(m::BitSlice(lhs.node(), 2, 2),
                               m::BitSlice(rhs.node(), 2, 2)),
                        m::Not(m::BitSlice(rhs.node(), 0, 2))));
}

TEST_F(StrengthReductionPassTest, SubTrailingNoBorrowTwoBitSplitRHS) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(4));
  BValue r = fb.Param("r", p->GetBitsType(2));
  // Set rhs LSBs to 00, guaranteeing no borrow to bits 0 or 1.
  BValue lhs = l;
  BValue rhs = fb.Concat({r, fb.Literal(UBits(0b00, 2))});
  fb.Subtract(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Sub(m::BitSlice(lhs.node(), 2, 2),
                               m::BitSlice(rhs.node(), 2, 2)),
                        m::BitSlice(lhs.node(), 0, 2)));
}

TEST_F(StrengthReductionPassTest, SubTrailingBorrowNoSplitGeneric) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Param("l", p->GetBitsType(4));
  BValue rhs = fb.Param("r", p->GetBitsType(4));
  fb.Subtract(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

void IrFuzzStrengthReduction(FuzzPackageWithArgs fuzz_package_with_args) {
  StrengthReductionPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzStrengthReduction)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
