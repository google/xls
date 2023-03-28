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

#include "xls/passes/select_simplification_pass.h"

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class SelectSimplificationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         SelectSimplificationPass().RunOnFunctionBase(
                             f, PassOptions(), &results));
    return changed;
  }
};

TEST_F(SelectSimplificationPassTest, BinaryTupleSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], x: bits[8], y: bits[8], z: bits[8]) -> (bits[8], bits[8]) {
        tuple.1: (bits[8], bits[8]) = tuple(x, y)
        tuple.2: (bits[8], bits[8]) = tuple(y, z)
        ret sel.3: (bits[8], bits[8]) = sel(p, cases=[tuple.2, tuple.1])
     }
  )",
                                                       p.get()));

  EXPECT_TRUE(f->return_value()->Is<Select>());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(m::Param("p"),
                                 /*cases=*/{m::TupleIndex(m::Tuple(), 0),
                                            m::TupleIndex(m::Tuple(), 0)}),
                       m::Select(m::Param("p"),
                                 /*cases=*/{m::TupleIndex(m::Tuple(), 1),
                                            m::TupleIndex(m::Tuple(), 1)})));
}

TEST_F(SelectSimplificationPassTest, BinaryTupleOneHotSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[8], y: bits[8], z: bits[8]) -> (bits[8], bits[8]) {
        tuple.1: (bits[8], bits[8]) = tuple(x, y)
        tuple.2: (bits[8], bits[8]) = tuple(y, z)
        ret result: (bits[8], bits[8]) = one_hot_sel(p, cases=[tuple.2, tuple.1])
     }
  )",
                                                       p.get()));

  EXPECT_TRUE(f->return_value()->Is<OneHotSelect>());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Tuple(m::OneHotSelect(m::Param("p"),
                               /*cases=*/{m::TupleIndex(m::Tuple(), 0),
                                          m::TupleIndex(m::Tuple(), 0)}),
               m::OneHotSelect(m::Param("p"),
                               /*cases=*/{m::TupleIndex(m::Tuple(), 1),
                                          m::TupleIndex(m::Tuple(), 1)})));
}

TEST_F(SelectSimplificationPassTest, FourWayTupleSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[8], y: bits[8], z: bits[8]) -> (bits[8], bits[8]) {
        tuple.1: (bits[8], bits[8]) = tuple(x, y)
        tuple.2: (bits[8], bits[8]) = tuple(y, z)
        tuple.3: (bits[8], bits[8]) = tuple(x, z)
        tuple.4: (bits[8], bits[8]) = tuple(z, z)
        ret sel.5: (bits[8], bits[8]) = sel(p, cases=[tuple.1, tuple.2, tuple.3, tuple.4])
     }
  )",
                                                       p.get()));

  EXPECT_TRUE(f->return_value()->Is<Select>());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Select(m::Param("p"),
                                 /*cases=*/{m::TupleIndex(m::Tuple(), 0),
                                            m::TupleIndex(m::Tuple(), 0),
                                            m::TupleIndex(m::Tuple(), 0),
                                            m::TupleIndex(m::Tuple(), 0)}),
                       m::Select(m::Param("p"),
                                 /*cases=*/{m::TupleIndex(m::Tuple(), 1),
                                            m::TupleIndex(m::Tuple(), 1),
                                            m::TupleIndex(m::Tuple(), 1),
                                            m::TupleIndex(m::Tuple(), 1)})));
}

TEST_F(SelectSimplificationPassTest, SelectWithConstantZeroSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42], y: bits[42]) -> bits[42] {
        literal.1: bits[1] = literal(value=0)
        ret sel.2: bits[42] = sel(literal.1, cases=[x, y])
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<Select>());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(SelectSimplificationPassTest, SelectWithConstantOneSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42], y: bits[42]) -> bits[42] {
        literal.1: bits[1] = literal(value=1)
        ret sel.2: bits[42] = sel(literal.1, cases=[x, y])
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<Select>());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("y"));
}

TEST_F(SelectSimplificationPassTest, OneHotSelectWithConstantSelector) {
  std::string tmpl = R"(
     fn f(x: bits[42], y: bits[42]) -> bits[42] {
        literal.1: bits[2] = literal(value=$0)
        ret result: bits[42] = one_hot_sel(literal.1, cases=[x, y])
     }
  )";
  {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, ParseFunction(absl::Substitute(tmpl, "0"), p.get()));
    EXPECT_THAT(Run(f), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::Literal("bits[42]:0"));
  }
  {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, ParseFunction(absl::Substitute(tmpl, "1"), p.get()));
    EXPECT_THAT(Run(f), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::Param("x"));
  }
  {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, ParseFunction(absl::Substitute(tmpl, "2"), p.get()));
    EXPECT_THAT(Run(f), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::Param("y"));
  }
  {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, ParseFunction(absl::Substitute(tmpl, "3"), p.get()));
    EXPECT_THAT(Run(f), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::Or());
  }
}

TEST_F(SelectSimplificationPassTest, UselessSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[1]) -> bits[32] {
        literal.1: bits[32] = literal(value=0)
        ret sel.2: bits[32] = sel(x, cases=[literal.1, literal.1])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal("bits[32]:0"));
}

TEST_F(SelectSimplificationPassTest, UselessOneHotSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[2]) -> bits[32] {
        literal.1: bits[32] = literal(value=42)
        ret result: bits[32] = one_hot_sel(x, cases=[literal.1, literal.1])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Eq(m::Param("x"), m::Literal("bits[2]:0")),
                        /*cases=*/{m::Literal("bits[32]:42"),
                                   m::Literal("bits[32]:0")}));
}

TEST_F(SelectSimplificationPassTest, MeaningfulSelect) {
  auto p = CreatePackage();
  const std::string program =
      R"(fn f(s: bits[1], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  add.2: bits[32] = add(x, z)
  ret sel.3: bits[32] = sel(s, cases=[add.1, add.2])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(SelectSimplificationPassTest, Useless3ArySelectWithDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[2]) -> bits[32] {
        literal.1: bits[32] = literal(value=0)
        ret sel.2: bits[32] = sel(x, cases=[literal.1, literal.1, literal.1], default=literal.1)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(), m::Literal("bits[32]:0"));
}

TEST_F(SelectSimplificationPassTest, Meaningful3ArySelectViaDefault) {
  auto p = CreatePackage();
  const std::string program = R"(fn f(x: bits[3]) -> bits[8] {
  literal.1: bits[8] = literal(value=0)
  literal.2: bits[8] = literal(value=129)
  ret sel.3: bits[8] = sel(x, cases=[literal.1, literal.1, literal.1], default=literal.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(SelectSimplificationPassTest, OneBitMux) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[1], a: bits[1]) -> bits[1] {
       ret sel.3: bits[1] = sel(s, cases=[s, a])
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kSel);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Or(m::And(m::Param("s"), m::Param("a")),
                    m::And(m::Not(m::Param("s")), m::Param("s"))));
}

TEST_F(SelectSimplificationPassTest, SelSqueezing) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(s: bits[2]) -> bits[3] {
  literal.1: bits[3] = literal(value=0b010)
  literal.2: bits[3] = literal(value=0b000)
  ret sel.3: bits[3] = sel(s, cases=[literal.1, literal.2], default=literal.1)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal("bits[1]:0"),
                        m::Select(m::Param("s"),
                                  /*cases=*/{m::BitSlice(), m::BitSlice()},
                                  /*default_value=*/m::BitSlice()),
                        m::Literal("bits[1]:0")));
}

TEST_F(SelectSimplificationPassTest, OneHotSelSqueezing) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(s: bits[3], x: bits[8]) -> bits[8] {
  literal.1: bits[8] = literal(value=0b01011001)
  literal.2: bits[8] = literal(value=0b01100101)
  ret one_hot_sel.3: bits[8] = one_hot_sel(s, cases=[x, literal.1, literal.2])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  // Should be decomposed into a concat of three OneHotSelects. The first and
  // last should select among two cases. The middle should select among three
  // cases.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(
          m::OneHotSelect(m::Param("s"),
                          /*cases=*/{m::BitSlice(/*start=*/6, /*width=*/2),
                                     m::BitSlice(/*start=*/6, /*width=*/2),
                                     m::BitSlice(/*start=*/6, /*width=*/2)}),
          m::OneHotSelect(m::Param("s"),
                          /*cases=*/{m::BitSlice(/*start=*/2, /*width=*/4),
                                     m::BitSlice(/*start=*/2, /*width=*/4),
                                     m::BitSlice(/*start=*/2, /*width=*/4)}),
          m::OneHotSelect(m::Param("s"),
                          /*cases=*/{m::BitSlice(/*start=*/0, /*width=*/2),
                                     m::BitSlice(/*start=*/0, /*width=*/2),
                                     m::BitSlice(/*start=*/0, /*width=*/2)})));
}

TEST_F(SelectSimplificationPassTest, OneHotSelectCommoning) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(s: bits[4], x: bits[3], y: bits[3]) -> bits[3] {
  ret one_hot_sel.2: bits[3] = one_hot_sel(s, cases=[x, y, x, y])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(m::Or(m::BitSlice(/*start=*/1, /*width=*/1),
                                      m::BitSlice(/*start=*/3, /*width=*/1)),
                                m::Or(m::BitSlice(/*start=*/0, /*width=*/1),
                                      m::BitSlice(/*start=*/2, /*width=*/1))),
                      /*cases=*/{m::Param("x"), m::Param("y")}));
}

TEST_F(SelectSimplificationPassTest, OneHotSelectFeedingOneHotSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p0: bits[2], p1: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot_sel.1: bits[32] = one_hot_sel(p0, cases=[x, y])
       ret one_hot_sel.2: bits[32] = one_hot_sel(p1, cases=[one_hot_sel.1, z])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(),
                      /*cases=*/{m::Param("x"), m::Param("y"), m::Param("z")}));
}

TEST_F(SelectSimplificationPassTest,
       OneHotSelectFeedingOneHotSelectWithMultipleUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p0: bits[2], p1: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot_sel.1: bits[32] = one_hot_sel(p0, cases=[x, y])
       neg.2: bits[32] = neg(one_hot_sel.1)
       one_hot_sel.3: bits[32] = one_hot_sel(p1, cases=[one_hot_sel.1, z])
       ret add.4: bits[32] = add(neg.2, one_hot_sel.3)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(SelectSimplificationPassTest, OneHotSelectChain) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p0: bits[3], p1: bits[2], p2: bits[4], v: bits[32], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot_sel.1: bits[32] = one_hot_sel(p0, cases=[w, x, y])
       one_hot_sel.2: bits[32] = one_hot_sel(p1, cases=[one_hot_sel.1, z])
       ret one_hot_sel.3: bits[32] = one_hot_sel(p2, cases=[v, one_hot_sel.2, z, w])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(
          m::Concat(),
          /*cases=*/{m::Param("v"), m::Param("w"), m::Param("x"), m::Param("y"),
                     m::Param("z"), m::Param("z"), m::Param("w")}));
}

TEST_F(SelectSimplificationPassTest, OneHotSelectTree) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p0: bits[2], p1: bits[2], p2: bits[3], v: bits[32], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot_sel.1: bits[32] = one_hot_sel(p0, cases=[v, w])
       one_hot_sel.2: bits[32] = one_hot_sel(p1, cases=[y, z])
       ret one_hot_sel.3: bits[32] = one_hot_sel(p2, cases=[one_hot_sel.1, x, one_hot_sel.2])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(),
                      /*cases=*/{m::Param("v"), m::Param("w"), m::Param("x"),
                                 m::Param("y"), m::Param("z")}));
}

TEST_F(SelectSimplificationPassTest, OneHotSelectWithLiteralZeroArms) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[6], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       literal.2: bits[32] = literal(value=0)
       literal.3: bits[32] = literal(value=0)
       ret one_hot_sel.4: bits[32] = one_hot_sel(p, cases=[literal.1, x, y, literal.2, literal.3, z])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Concat(),
                      /*cases=*/{m::Param("x"), m::Param("y"), m::Param("z")}));
}

TEST_F(SelectSimplificationPassTest, OneHotSelectWithOnlyLiteralZeroArms) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[3]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       literal.2: bits[32] = literal(value=0)
       literal.3: bits[32] = literal(value=0)
       ret one_hot_sel.4: bits[32] = one_hot_sel(p, cases=[literal.1, literal.2, literal.3])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal("bits[32]:0"));
}

TEST_F(SelectSimplificationPassTest, SelectWithZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret sel.2: bits[32] = sel(p, cases=[literal.1, x])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"), m::SignExt(m::Param("p"))));
}

TEST_F(SelectSimplificationPassTest, SelectWithZeroInCaseOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret sel.2: bits[32] = sel(p, cases=[x, literal.1])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"), m::SignExt(m::Not(m::Param("p")))));
}

TEST_F(SelectSimplificationPassTest, TwoWayOneHotSelectWhichIsNotOneHot) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32]) -> bits[32] {
       ret one_hot_sel.1: bits[32] = one_hot_sel(p, cases=[x, y])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(SelectSimplificationPassTest, OneBitOneHot) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1]) -> bits[2] {
       ret one_hot.1: bits[2] = one_hot(p, lsb_prio=true)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Not(m::Param("p")), m::Param("p")));
}

TEST_F(SelectSimplificationPassTest, SplittableOneHotSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[3], x: bits[8], y: bits[4]) -> bits[8] {
       literal.1: bits[4] = literal(value=0b1001)
       literal.2: bits[4] = literal(value=0b1000)
       bit_slice.3: bits[2] = bit_slice(x, start=0, width=2)
       concat.4: bits[8] = concat(literal.1, bit_slice.3, bit_slice.3)
       concat.5: bits[8] = concat(literal.2, y)
       ret result: bits[8] = one_hot_sel(p, cases=[concat.4, concat.5, x])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  // Should be split into a concat of three OneHotSelects. The first and
  // last should select among two cases. The middle should select among three
  // cases.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(
          m::OneHotSelect(m::Param("p"),
                          /*cases=*/{m::BitSlice(/*start=*/5, /*width=*/3),
                                     m::BitSlice(/*start=*/5, /*width=*/3),
                                     m::BitSlice(/*start=*/5, /*width=*/3)}),
          m::OneHotSelect(m::Param("p"),
                          /*cases=*/{m::BitSlice(/*start=*/2, /*width=*/3),
                                     m::BitSlice(/*start=*/2, /*width=*/3),
                                     m::BitSlice(/*start=*/2, /*width=*/3)}),
          m::OneHotSelect(m::Param("p"),
                          /*cases=*/{m::BitSlice(/*start=*/0, /*width=*/2),
                                     m::BitSlice(/*start=*/0, /*width=*/2),
                                     m::BitSlice(/*start=*/0, /*width=*/2)})));
}

TEST_F(SelectSimplificationPassTest, SelectsWithCommonCase0) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  Type* u1 = p->GetBitsType(1);
  BValue x = fb.Param("x", u32);
  BValue sel1 = fb.Select(fb.Param("p1", u1), {x, fb.Param("y", u32)});
  fb.Select(fb.Param("p0", u1), {x, sel1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Nand(m::Param("p0"), m::Param("p1")),
                        {m::Param("y"), m::Param("x")}));
}

TEST_F(SelectSimplificationPassTest, SelectsWithCommonCase1) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  Type* u1 = p->GetBitsType(1);
  BValue x = fb.Param("x", u32);
  BValue sel1 = fb.Select(fb.Param("p1", u1), {x, fb.Param("y", u32)});
  fb.Select(fb.Param("p0", u1), {sel1, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Or(m::Param("p0"), m::Not(m::Param("p1"))),
                        {m::Param("y"), m::Param("x")}));
}

TEST_F(SelectSimplificationPassTest, SelectsWithCommonCase2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  Type* u1 = p->GetBitsType(1);
  BValue x = fb.Param("x", u32);
  BValue sel1 = fb.Select(fb.Param("p1", u1), {fb.Param("y", u32), x});
  fb.Select(fb.Param("p0", u1), {x, sel1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Or(m::Not(m::Param("p0")), m::Param("p1")),
                        {m::Param("y"), m::Param("x")}));
}

TEST_F(SelectSimplificationPassTest, SelectsWithCommonCase3) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  Type* u1 = p->GetBitsType(1);
  BValue x = fb.Param("x", u32);
  BValue sel1 = fb.Select(fb.Param("p1", u1), {fb.Param("y", u32), x});
  fb.Select(fb.Param("p0", u1), {sel1, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Or(m::Param("p0"), m::Param("p1")),
                        {m::Param("y"), m::Param("x")}));
}

}  // namespace
}  // namespace xls
