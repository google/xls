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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class StrengthReductionPassTest : public IrTestBase {
 protected:
  StrengthReductionPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(
        bool changed,
        StrengthReductionPass().RunOnFunctionBase(f, PassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .RunOnFunctionBase(f, PassOptions(), &results)
                            .status());
    // Return whether strength reduction changed anything.
    return changed;
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

}  // namespace
}  // namespace xls
