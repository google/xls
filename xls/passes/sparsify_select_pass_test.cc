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

#include "xls/passes/sparsify_select_pass.h"

#include <cstdint>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class SparsifySelectPassTest : public IrTestBase {
 protected:
  SparsifySelectPassTest() = default;

  absl::StatusOr<bool> Run(FunctionBase* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         SparsifySelectPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results));
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
            .status());
    return changed;
  }
};

TEST_F(SparsifySelectPassTest, CompoundType) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(1));
  auto x_big = fb.ZeroExtend(x, 3);
  auto complex_vs = [&](uint64_t v) {
    return fb.Literal(Value::Tuple({Value(UBits(v, 4)), Value(UBits(v, 5)),
                                    Value(UBits(v, 6)), Value(UBits(v, 5)),
                                    Value(UBits(v, 4))}));
  };
  auto complex_vs_match = [&](uint64_t v) {
    return m::Literal(Value::Tuple({Value(UBits(v, 4)), Value(UBits(v, 5)),
                                    Value(UBits(v, 6)), Value(UBits(v, 5)),
                                    Value(UBits(v, 4))}));
  };
  fb.Select(x_big,
            {complex_vs(1), complex_vs(2), complex_vs(3), complex_vs(4),
             complex_vs(5), complex_vs(6), complex_vs(7), complex_vs(8)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Sub(x_big.node(), m::Literal(UBits(0, 3))),
                        {complex_vs_match(1), complex_vs_match(2)},
                        complex_vs_match(0)));
}

TEST_F(SparsifySelectPassTest, Simple) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[2]) -> bits[4] {
       zero_ext.1: bits[4] = zero_ext(x, new_bit_count=4)
       literal.2: bits[4] = literal(value=8)
       add.3: bits[4] = add(zero_ext.1, literal.2)
       literal.4: bits[4] = literal(value=0)
       literal.5: bits[4] = literal(value=1)
       literal.6: bits[4] = literal(value=2)
       literal.7: bits[4] = literal(value=3)
       literal.8: bits[4] = literal(value=4)
       literal.9: bits[4] = literal(value=5)
       literal.10: bits[4] = literal(value=6)
       literal.11: bits[4] = literal(value=7)
       literal.12: bits[4] = literal(value=8)
       literal.13: bits[4] = literal(value=9)
       literal.14: bits[4] = literal(value=10)
       literal.15: bits[4] = literal(value=11)
       literal.16: bits[4] = literal(value=12)
       literal.17: bits[4] = literal(value=13)
       literal.18: bits[4] = literal(value=14)
       literal.19: bits[4] = literal(value=15)
       ret sel.20: bits[4] = sel(add.3, cases=[
         literal.4, literal.5, literal.6, literal.7, literal.8, literal.9,
         literal.10, literal.11, literal.12, literal.13, literal.14,
         literal.15, literal.16, literal.17, literal.18, literal.19
       ])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Sub(m::Add(m::ZeroExt(m::Param("x")),
                                      m::Literal("bits[4]:8")),
                               m::Literal("bits[4]:8")),
                        /*cases=*/
                        {
                            m::Literal("bits[4]:8"),
                            m::Literal("bits[4]:9"),
                            m::Literal("bits[4]:10"),
                            m::Literal("bits[4]:11"),
                        },
                        /*default_value=*/m::Literal("bits[4]:0")));
}

TEST_F(SparsifySelectPassTest, TwoIntervals) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[2], bool: bits[1]) -> bits[4] {
       zero_ext.1: bits[4] = zero_ext(x, new_bit_count=4)
       literal.2: bits[4] = literal(value=9)
       add.3: bits[4] = add(zero_ext.1, literal.2)
       literal.4: bits[4] = literal(value=2)
       add.5: bits[4] = add(zero_ext.1, literal.4)
       sel.6: bits[4] = sel(bool, cases=[add.3, add.5])
       literal.7: bits[4] = literal(value=0)
       literal.8: bits[4] = literal(value=1)
       literal.9: bits[4] = literal(value=2)
       literal.10: bits[4] = literal(value=3)
       literal.11: bits[4] = literal(value=4)
       literal.12: bits[4] = literal(value=5)
       literal.13: bits[4] = literal(value=6)
       literal.14: bits[4] = literal(value=7)
       literal.15: bits[4] = literal(value=8)
       literal.16: bits[4] = literal(value=9)
       literal.17: bits[4] = literal(value=10)
       literal.18: bits[4] = literal(value=11)
       literal.19: bits[4] = literal(value=12)
       literal.20: bits[4] = literal(value=13)
       literal.21: bits[4] = literal(value=14)
       literal.22: bits[4] = literal(value=15)
       ret sel.23: bits[4] = sel(sel.6, cases=[
         literal.7, literal.8, literal.9, literal.10, literal.11, literal.12,
         literal.13, literal.14, literal.15, literal.16, literal.17, literal.18,
         literal.19, literal.20, literal.21, literal.22
       ])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  testing::Matcher<const Node*> selector = m::Select(
      m::Param("bool"),
      /*cases=*/{m::Add(m::ZeroExt(m::Param("x")), m::Literal("bits[4]:9")),
                 m::Add(m::ZeroExt(m::Param("x")), m::Literal("bits[4]:2"))});
  EXPECT_THAT(
      f->return_value(),
      m::Select(
          m::And(m::UGe(selector, m::Literal("bits[4]:9")),
                 m::ULe(selector, m::Literal("bits[4]:12"))),
          /*cases=*/{m::Select(m::Sub(selector, m::Literal("bits[4]:2")),
                               /*cases=*/
                               {
                                   m::Literal("bits[4]:2"),
                                   m::Literal("bits[4]:3"),
                                   m::Literal("bits[4]:4"),
                                   m::Literal("bits[4]:5"),
                               },
                               /*default_value=*/m::Literal("bits[4]:0")),
                     m::Select(m::Sub(selector, m::Literal("bits[4]:9")),
                               /*cases=*/
                               {
                                   m::Literal("bits[4]:9"),
                                   m::Literal("bits[4]:10"),
                                   m::Literal("bits[4]:11"),
                                   m::Literal("bits[4]:12"),
                               },
                               /*default_value=*/m::Literal("bits[4]:0"))}));
}

TEST_F(SparsifySelectPassTest, FourIntervals) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[2]) -> bits[4] {
       literal.1: bits[4] = literal(value=5)
       literal.2: bits[4] = literal(value=8)
       literal.3: bits[4] = literal(value=11)
       literal.4: bits[4] = literal(value=14)
       sel.5: bits[4] = sel(x, cases=[literal.1, literal.2, literal.3, literal.4])
       literal.6: bits[4] = literal(value=0)
       literal.7: bits[4] = literal(value=1)
       literal.8: bits[4] = literal(value=2)
       literal.9: bits[4] = literal(value=3)
       literal.10: bits[4] = literal(value=4)
       literal.11: bits[4] = literal(value=5)
       literal.12: bits[4] = literal(value=6)
       literal.13: bits[4] = literal(value=7)
       literal.14: bits[4] = literal(value=8)
       literal.15: bits[4] = literal(value=9)
       literal.16: bits[4] = literal(value=10)
       literal.17: bits[4] = literal(value=11)
       literal.18: bits[4] = literal(value=12)
       literal.19: bits[4] = literal(value=13)
       literal.20: bits[4] = literal(value=14)
       literal.21: bits[4] = literal(value=15)
       ret sel.22: bits[4] = sel(sel.5, cases=[
         literal.6, literal.7, literal.8, literal.9, literal.10, literal.11,
         literal.12, literal.13, literal.14, literal.15, literal.16, literal.17,
         literal.18, literal.19, literal.20, literal.21
       ])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  testing::Matcher<const Node*> selector =
      m::Select(m::Param("x"),
                /*cases=*/{m::Literal("bits[4]:5"), m::Literal("bits[4]:8"),
                           m::Literal("bits[4]:11"), m::Literal("bits[4]:14")});
  EXPECT_THAT(
      f->return_value(),
      m::Select(
          m::And(m::UGe(selector, m::Literal("bits[4]:14")),
                 m::ULe(selector, m::Literal("bits[4]:14"))),
          /*cases=*/{
              m::Select(
                  m::And(m::UGe(selector, m::Literal("bits[4]:11")),
                         m::ULe(selector, m::Literal("bits[4]:11"))),
                  /*cases=*/
                  {m::Select(
                       m::And(m::UGe(selector, m::Literal("bits[4]:8")),
                              m::ULe(selector, m::Literal("bits[4]:8"))),
                       /*cases=*/
                       {m::Select(m::Sub(selector, m::Literal("bits[4]:5")),
                                  /*cases=*/{m::Literal("bits[4]:5")},
                                  /*default_value=*/m::Literal("bits[4]:0")),
                        m::Select(m::Sub(selector, m::Literal("bits[4]:8")),
                                  /*cases=*/
                                  {m::Literal("bits[4]:8")},
                                  /*default_value=*/
                                  m::Literal("bits[4]:0"))}),
                   m::Select(m::Sub(selector, m::Literal("bits[4]:11")),
                             /*cases=*/{m::Literal("bits[4]:11")},
                             /*default_value=*/m::Literal("bits[4]:0"))}),
              m::Select(m::Sub(selector, m::Literal("bits[4]:14")),
                        /*cases=*/{m::Literal("bits[4]:14")},
                        /*default_value=*/m::Literal("bits[4]:0"))}));
}

TEST_F(SparsifySelectPassTest, WideSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[2], bool: bits[1]) -> bits[4] {
       zero_ext.1: bits[4] = zero_ext(x, new_bit_count=4)
       literal.2: bits[4] = literal(value=9)
       add.3: bits[4] = add(zero_ext.1, literal.2)
       literal.4: bits[4] = literal(value=2)
       add.5: bits[4] = add(zero_ext.1, literal.4)
       sel.6: bits[4] = sel(bool, cases=[add.3, add.5])
       zero_ext.7: bits[100] = zero_ext(sel.6, new_bit_count=100)
       literal.8: bits[4] = literal(value=0)
       literal.9: bits[4] = literal(value=1)
       literal.10: bits[4] = literal(value=2)
       literal.11: bits[4] = literal(value=3)
       literal.12: bits[4] = literal(value=4)
       literal.13: bits[4] = literal(value=5)
       literal.14: bits[4] = literal(value=6)
       literal.15: bits[4] = literal(value=7)
       literal.16: bits[4] = literal(value=8)
       literal.17: bits[4] = literal(value=9)
       literal.18: bits[4] = literal(value=10)
       literal.19: bits[4] = literal(value=11)
       literal.20: bits[4] = literal(value=12)
       literal.21: bits[4] = literal(value=13)
       literal.22: bits[4] = literal(value=14)
       literal.23: bits[4] = literal(value=15)
       ret sel.24: bits[4] = sel(zero_ext.7, cases=[
         literal.8, literal.9, literal.10, literal.11, literal.12, literal.13,
         literal.14, literal.15, literal.16, literal.17, literal.18, literal.19,
         literal.20, literal.21, literal.22, literal.23
       ], default=literal.23)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  testing::Matcher<const Node*> selector = m::ZeroExt(m::Select(
      m::Param("bool"),
      /*cases=*/{m::Add(m::ZeroExt(m::Param("x")), m::Literal("bits[4]:9")),
                 m::Add(m::ZeroExt(m::Param("x")), m::Literal("bits[4]:2"))}));
  EXPECT_THAT(
      f->return_value(),
      m::Select(
          m::And(m::UGe(selector, m::Literal("bits[100]:9")),
                 m::ULe(selector, m::Literal("bits[100]:12"))),
          /*cases=*/{m::Select(m::Sub(selector, m::Literal("bits[100]:2")),
                               /*cases=*/
                               {
                                   m::Literal("bits[4]:2"),
                                   m::Literal("bits[4]:3"),
                                   m::Literal("bits[4]:4"),
                                   m::Literal("bits[4]:5"),
                               },
                               /*default_value=*/m::Literal("bits[4]:0")),
                     m::Select(m::Sub(selector, m::Literal("bits[100]:9")),
                               /*cases=*/
                               {
                                   m::Literal("bits[4]:9"),
                                   m::Literal("bits[4]:10"),
                                   m::Literal("bits[4]:11"),
                                   m::Literal("bits[4]:12"),
                               },
                               /*default_value=*/m::Literal("bits[4]:0"))}));
}

TEST_F(SparsifySelectPassTest, DefaultValue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[2], bool: bits[1]) -> bits[4] {
       zero_ext.1: bits[4] = zero_ext(x, new_bit_count=4)
       literal.2: bits[4] = literal(value=9)
       add.3: bits[4] = add(zero_ext.1, literal.2)
       literal.4: bits[4] = literal(value=2)
       add.5: bits[4] = add(zero_ext.1, literal.4)
       sel.6: bits[4] = sel(bool, cases=[add.3, add.5])
       literal.7: bits[4] = literal(value=0)
       literal.8: bits[4] = literal(value=1)
       literal.9: bits[4] = literal(value=2)
       literal.10: bits[4] = literal(value=3)
       literal.11: bits[4] = literal(value=4)
       literal.12: bits[4] = literal(value=5)
       literal.13: bits[4] = literal(value=6)
       literal.14: bits[4] = literal(value=7)
       literal.15: bits[4] = literal(value=8)
       literal.16: bits[4] = literal(value=9)
       literal.17: bits[4] = literal(value=10)
       literal.18: bits[4] = literal(value=11)
       literal.19: bits[4] = literal(value=12)
       literal.20: bits[4] = literal(value=13)
       literal.21: bits[4] = literal(value=14)
       literal.22: bits[4] = literal(value=15)
       ret sel.23: bits[4] = sel(sel.6, cases=[
         literal.7, literal.8, literal.9, literal.10, literal.11, literal.12,
         literal.13, literal.14, literal.15, literal.16, literal.17, literal.18,
         literal.19, literal.20, literal.21
       ], default=literal.22)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  testing::Matcher<const Node*> selector = m::Select(
      m::Param("bool"),
      /*cases=*/{m::Add(m::ZeroExt(m::Param("x")), m::Literal("bits[4]:9")),
                 m::Add(m::ZeroExt(m::Param("x")), m::Literal("bits[4]:2"))});
  EXPECT_THAT(
      f->return_value(),
      m::Select(
          m::And(m::UGe(selector, m::Literal("bits[4]:9")),
                 m::ULe(selector, m::Literal("bits[4]:12"))),
          /*cases=*/{m::Select(m::Sub(selector, m::Literal("bits[4]:2")),
                               /*cases=*/
                               {
                                   m::Literal("bits[4]:2"),
                                   m::Literal("bits[4]:3"),
                                   m::Literal("bits[4]:4"),
                                   m::Literal("bits[4]:5"),
                               },
                               /*default_value=*/m::Literal("bits[4]:0")),
                     m::Select(m::Sub(selector, m::Literal("bits[4]:9")),
                               /*cases=*/
                               {
                                   m::Literal("bits[4]:9"),
                                   m::Literal("bits[4]:10"),
                                   m::Literal("bits[4]:11"),
                                   m::Literal("bits[4]:12"),
                               },
                               /*default_value=*/m::Literal("bits[4]:0"))}));
}

TEST_F(SparsifySelectPassTest, LargeSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x1: bits[2], x2: bits[2]) -> bits[2] {
       literal.1: bits[2042] = literal(value=0x20_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000, id=1)
       literal.2: bits[2042] = literal(value=0x0, id=2)
       s: bits[2042] = one_hot_sel(x1, cases=[literal.1, literal.2], id=3)
       literal.4: bits[2] = literal(value=0, id=4)
       ret sel.5: bits[2] = sel(s, cases=[x2, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4, literal.4], default=literal.4, id=5)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  testing::Matcher<const Node*> selector =
      m::OneHotSelect(m::Param("x1"),
                      /*cases=*/{m::Literal(), m::Literal()});
  constexpr std::string_view kLargeLiteral =
      "bits[2042]:0x20_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_"
      "0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_"
      "0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_"
      "0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_"
      "0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000";
  EXPECT_THAT(
      f->return_value(),
      m::Select(
          m::And(m::UGe(selector, m::Literal(kLargeLiteral)),
                 m::ULe(selector, m::Literal(kLargeLiteral))),
          /*cases=*/{m::Select(m::Sub(selector, m::Literal("bits[2042]:0x0")),
                               /*cases=*/
                               {
                                   m::Param("x2"),
                               },
                               /*default_value=*/m::Literal("bits[2]:0x0")),
                     m::Select(m::Sub(selector, m::Literal(kLargeLiteral)),
                               /*cases=*/
                               {
                                   m::Literal("bits[2]:0x0"),
                               },
                               /*default_value=*/m::Literal("bits[2]:0x0"))}));
}

}  // namespace
}  // namespace xls
