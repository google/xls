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

#include "xls/passes/canonicalization_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class CanonicalizePassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return CanonicalizationPass().Run(p, PassOptions(), &results);
  }
};

TEST_F(CanonicalizePassTest, Canonicalize) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_canon(x:bits[2]) -> bits[2] {
        one:bits[2] = literal(value=1)
        ret addval: bits[2] = add(one, x)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Param(), m::Literal()));
}

TEST_F(CanonicalizePassTest, SubToAddNegate) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_neg(x:bits[2]) -> bits[2] {
        one:bits[2] = literal(value=1)
        ret subval: bits[2] = sub(x, one)
     }
  )",
                                                       p.get()));

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Param(), m::Neg()));
}

TEST_F(CanonicalizePassTest, NopZeroExtend) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn nop_zero_ext(x:bits[16]) -> bits[16] {
        ret zero_ext: bits[16] = zero_ext(x, new_bit_count=16)
     }
  )",
                                                       p.get()));

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param());
}

TEST_F(CanonicalizePassTest, ZeroExtendReplacedWithConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn zero_ext(x:bits[33]) -> bits[42] {
        ret zero_ext: bits[42] = zero_ext(x, new_bit_count=42)
     }
  )",
                                                       p.get()));

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat(m::Literal(0), m::Param()));
}

TEST_F(CanonicalizePassTest, ComparisonWithLiteralCanonicalization) {
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    fb.ULt(fb.Literal(UBits(42, 32)), fb.Param("x", p->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::UGt(m::Param(), m::Literal(42)));
  }

  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    fb.UGe(fb.Literal(UBits(42, 32)), fb.Param("x", p->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::ULe(m::Param(), m::Literal(42)));
  }

  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    fb.SGt(fb.Literal(UBits(42, 32)), fb.Param("x", p->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::SLt(m::Param(), m::Literal(42)));
  }

  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    fb.Eq(fb.Literal(UBits(42, 32)), fb.Param("x", p->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::Eq(m::Param(), m::Literal(42)));
  }
}

}  // namespace
}  // namespace xls
