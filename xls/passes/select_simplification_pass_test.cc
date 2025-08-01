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

#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::Eq;

enum class AnalysisType {
  kTernary,
  kRange,
};
std::ostream& operator<<(std::ostream& os, AnalysisType a) {
  switch (a) {
    case AnalysisType::kTernary:
      return os << "Ternary";
    case AnalysisType::kRange:
      return os << "Range";
  }
}
class SelectSimplificationPassTest
    : public IrTestBase,
      public testing::WithParamInterface<AnalysisType> {
 protected:
  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;
    if (GetParam() == AnalysisType::kTernary) {
      SelectSimplificationPass tern_pass;
      XLS_ASSIGN_OR_RETURN(
          bool changed, tern_pass.RunOnFunctionBase(
                            f, OptimizationPassOptions(), &results, context));
      return changed;
    }
    SelectRangeSimplificationPass range_pass;
    XLS_ASSIGN_OR_RETURN(
        bool changed, range_pass.RunOnFunctionBase(f, OptimizationPassOptions(),
                                                   &results, context));
    return changed;
  }
};

TEST_P(SelectSimplificationPassTest, BinaryTupleSelect) {
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

TEST_P(SelectSimplificationPassTest, BinaryTupleOneHotSelect) {
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

TEST_P(SelectSimplificationPassTest, BinaryTuplePrioritySelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[8], y: bits[8], z: bits[8]) -> (bits[8], bits[8]) {
        tuple.1: (bits[8], bits[8]) = tuple(x, y)
        tuple.2: (bits[8], bits[8]) = tuple(y, z)
        tuple.3: (bits[8], bits[8]) = tuple(x, z)
        ret result: (bits[8], bits[8]) = priority_sel(p, cases=[tuple.2, tuple.1], default=tuple.3)
     }
  )",
                                                       p.get()));

  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Tuple(m::PrioritySelect(
                   m::Param("p"),
                   /*cases=*/
                   {m::TupleIndex(m::Tuple(), 0), m::TupleIndex(m::Tuple(), 0)},
                   /*default_value=*/m::TupleIndex(m::Tuple(), 0)),
               m::PrioritySelect(
                   m::Param("p"),
                   /*cases=*/
                   {m::TupleIndex(m::Tuple(), 1), m::TupleIndex(m::Tuple(), 1)},
                   /*default_value=*/m::TupleIndex(m::Tuple(), 1))));
}

TEST_P(SelectSimplificationPassTest, FourWayTupleSelect) {
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

TEST_P(SelectSimplificationPassTest, SelectWithConstantZeroSelector) {
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

TEST_P(SelectSimplificationPassTest, SelectWithConstantOneSelector) {
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

TEST_P(SelectSimplificationPassTest, PrioritySelectWithConstantZeroSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=0)
        ret priority_sel.2: bits[42] = priority_sel(literal.1, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("d"));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithConstantOneSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=1)
        ret priority_sel.2: bits[42] = priority_sel(literal.1, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithConstantTwoSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=2)
        ret priority_sel.2: bits[42] = priority_sel(literal.1, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("y"));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithConstantThreeSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=3)
        ret priority_sel.2: bits[42] = priority_sel(literal.1, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithSelectorLowBitSet) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[3], x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=1)
        or.2: bits[3] = or(s, literal.1)
        ret priority_sel.3: bits[42] = priority_sel(or.2, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithSelectorMidBitSet) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[3], x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=2)
        or.2: bits[3] = or(s, literal.1)
        ret priority_sel.3: bits[42] = priority_sel(or.2, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::BitSlice(m::Or()), {m::Param("x")}, m::Param("y")));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithSelectorHighBitSet) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[3], x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=4)
        or.2: bits[3] = or(s, literal.1)
        ret priority_sel.3: bits[42] = priority_sel(or.2, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::BitSlice(m::Or()),
                                {m::Param("x"), m::Param("y")}, m::Param("z")));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithSelectorLowBitUnset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[3], x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=6)
        and.2: bits[3] = and(s, literal.1)
        ret priority_sel.3: bits[42] = priority_sel(and.2, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::BitSlice(m::And()),
                                {m::Param("y"), m::Param("z")}, m::Param("d")));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithSelectorMidBitUnset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[3], x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=5)
        and.2: bits[3] = and(s, literal.1)
        ret priority_sel.3: bits[42] = priority_sel(and.2, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Concat(m::BitSlice(m::And(), /*start=*/2, /*width=*/1),
                            m::BitSlice(m::And(), /*start=*/0, /*width=*/1)),
                  {m::Param("x"), m::Param("z")}, m::Param("d")));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithSelectorHighBitUnset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[3], x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=3)
        and.2: bits[3] = and(s, literal.1)
        ret priority_sel.3: bits[42] = priority_sel(and.2, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::BitSlice(m::And()),
                                {m::Param("x"), m::Param("y")}, m::Param("d")));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithSelectorBitsUnset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[3], x: bits[42], y: bits[42], z: bits[42], d: bits[42]) -> bits[42] {
        literal.1: bits[3] = literal(value=2)
        and.2: bits[3] = and(s, literal.1)
        ret priority_sel.3: bits[42] = priority_sel(and.2, cases=[x, y, z], default=d)
     }
  )",
                                                       p.get()));
  EXPECT_TRUE(f->return_value()->Is<PrioritySelect>());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::BitSlice(m::And()), {m::Param("y")}, m::Param("d")));
}

TEST_P(SelectSimplificationPassTest, OneHotSelectWithConstantSelector) {
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

TEST_P(SelectSimplificationPassTest, OneHotSelectWithIdenticalCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[2], x: bits[42]) -> bits[42] {
        ret result: bits[42] = one_hot_sel(s, cases=[x, x])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Ne(m::Param("s"), m::Literal("bits[2]:0")),
                        {m::Literal("bits[42]:0"), m::Param("x")}));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithTwoDistinctCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[2], x: bits[42], y: bits[42]) -> bits[42] {
        ret result: bits[42] = priority_sel(s, cases=[x, y], default=x)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::Eq(m::Param("s"), m::Literal("bits[2]:0b10")),
                        {m::Param("y")}, /*default_value=*/m::Param("x")));
}

TEST_P(SelectSimplificationPassTest,
       ComplexPrioritySelectWithTwoDistinctCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(s: bits[5], x: bits[42], y: bits[42]) -> bits[42] {
        ret result: bits[42] = priority_sel(s, cases=[y, x, y, x, y], default=x)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(
          m::Or(m::BitSlice(m::Param("s"), /*start=*/0, /*width=*/1),
                m::Eq(m::BitSlice(m::Param("s"), /*start=*/0, /*width=*/3),
                      m::Literal("bits[3]:0b100")),
                m::Eq(m::Param("s"), m::Literal("bits[5]:0b10000"))),
          {m::Param("y")}, /*default_value=*/m::Param("x")));
}

TEST_P(SelectSimplificationPassTest, UselessSelect) {
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

TEST_P(SelectSimplificationPassTest, UselessOneHotSelect) {
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
              m::Select(m::Ne(m::Param("x"), m::Literal("bits[2]:0")),
                        /*cases=*/{m::Literal("bits[32]:0"),
                                   m::Literal("bits[32]:42")}));
}

TEST_P(SelectSimplificationPassTest, MeaningfulSelect) {
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

TEST_P(SelectSimplificationPassTest, Useless3ArySelectWithDefault) {
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

TEST_P(SelectSimplificationPassTest, Meaningful3ArySelectViaDefault) {
  auto p = CreatePackage();
  const std::string program = R"(fn f(x: bits[3]) -> bits[8] {
  literal.1: bits[8] = literal(value=0)
  literal.2: bits[8] = literal(value=129)
  ret sel.3: bits[8] = sel(x, cases=[literal.1, literal.1, literal.1], default=literal.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::And(m::Literal("bits[8]:129"),
                     m::SignExt(m::UGe(m::Param("x"), m::Literal(3)))));
}

TEST_P(SelectSimplificationPassTest, MeaningfulArrayTyped3ArySelectViaDefault) {
  auto p = CreatePackage();
  const std::string program = R"(fn f(x: bits[3]) -> bits[4][2] {
  literal.1: bits[4][2] = literal(value=[0, 0])
  literal.2: bits[4][2] = literal(value=[3, 1])
  ret sel.3: bits[4][2] = sel(x, cases=[literal.1, literal.1, literal.1], default=literal.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(
      Value default_value,
      Value::Array({Value(UBits(3, 4)), Value(UBits(1, 4))}));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value zero_value, Value::Array({Value(UBits(0, 4)), Value(UBits(0, 4))}));
  EXPECT_THAT(f->return_value(), m::Select(m::UGe(m::Param("x"), m::Literal(3)),
                                           /*cases=*/
                                           {m::Literal(zero_value)},
                                           /*default_value=*/
                                           m::Literal(default_value)));
}

TEST_P(SelectSimplificationPassTest, OneBitMuxSel) {
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

TEST_P(SelectSimplificationPassTest, OneBitMuxPrioritySel) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[1], a: bits[1]) -> bits[1] {
       ret priority_sel.3: bits[1] = priority_sel(s, cases=[s], default=a)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kPrioritySel);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Or(m::And(m::Param("s"), m::Param("s")),
                    m::And(m::Not(m::Param("s")), m::Param("a"))));
}

TEST_P(SelectSimplificationPassTest, OneBitMuxOneHotSel) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(s: bits[1], a: bits[1]) -> bits[1] {
       ret one_hot_sel.3: bits[1] = one_hot_sel(s, cases=[a])
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kOneHotSel);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("s"), m::Param("a")));
}

TEST_P(SelectSimplificationPassTest, SelSqueezing) {
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

TEST_P(SelectSimplificationPassTest, OneHotSelSqueezing) {
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

TEST_P(SelectSimplificationPassTest, OneHotSelectCommoning) {
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

TEST_P(SelectSimplificationPassTest, PrioritySelectCommoning) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn f(s: bits[6], x: bits[3], y: bits[3], z: bits[3]) -> bits[3] {
  ret priority_sel.2: bits[3] = priority_sel(s, cases=[x, y, y, y, z, z], default=z)
}
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Concat(m::OrReduce(m::BitSlice(/*start=*/4, /*width=*/2)),
                            m::OrReduce(m::BitSlice(/*start=*/1, /*width=*/3)),
                            m::BitSlice(/*start=*/0, /*width=*/1)),
                  /*cases=*/{m::Param("x"), m::Param("y"), m::Param("z")},
                  /*default_value=*/m::Param("z")));
}

TEST_P(SelectSimplificationPassTest, OneHotSelectWithLiteralZeroArms) {
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

TEST_P(SelectSimplificationPassTest,
       OneHotSelectWithLiteralZeroArmAndZeroSelectorBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[5], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       zero0: bits[32] = literal(value=0)
       zero1: bits[32] = literal(value=0)
       mask: bits[5] = literal(value=0b10111)
       masked_p: bits[5] = and(p, mask)
       ret one_hot_sel.4: bits[32] = one_hot_sel(masked_p, cases=[zero0, x, zero1, y, z])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::OneHotSelect(m::Concat(),
                              /*cases=*/{m::Param("x"), m::Param("z")}));
}

TEST_P(SelectSimplificationPassTest,
       PrioritySelectWithLiteralZeroArmsAndZeroSelectorBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[6], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       zero: bits[32] = literal(value=0)
       mask: bits[6] = literal(value=0b110111)
       masked_p: bits[6] = and(p, mask)
       ret result: bits[32] = priority_sel(masked_p, cases=[zero, x, zero, y, z, zero], default=zero)
     }
  )",
                                                       p.get()));
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Concat(),
                  /*cases=*/
                  {m::Literal(0), m::Param("x"), m::Literal(0), m::Param("z")},
                  /*default_value=*/m::Literal(0)));
}

TEST_P(SelectSimplificationPassTest, OneHotSelectWithOnlyLiteralZeroArms) {
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

TEST_P(SelectSimplificationPassTest, PrioritySelectWithOnlyLiteralZeroArms) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[3]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       literal.2: bits[32] = literal(value=0)
       literal.3: bits[32] = literal(value=0)
       literal.4: bits[32] = literal(value=0)
       ret result: bits[32] = priority_sel(p, cases=[literal.1, literal.2, literal.3], default=literal.4)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal("bits[32]:0"));
}

TEST_P(SelectSimplificationPassTest,
       OneHotSelectWithOnlyLiteralZeroArmsAndZeroSelectorBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[5], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       zero0: bits[32] = literal(value=0)
       zero1: bits[32] = literal(value=0)
       mask: bits[5] = literal(value=0b00101)
       masked_p: bits[5] = and(p, mask)
       ret one_hot_sel.4: bits[32] = one_hot_sel(masked_p, cases=[zero0, x, zero1, y, z])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal("bits[32]:0"));
}

TEST_P(SelectSimplificationPassTest,
       PrioritySelectWithOnlyLiteralZeroArmsAndZeroSelectorBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[5], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       zero0: bits[32] = literal(value=0)
       zero1: bits[32] = literal(value=0)
       zero2: bits[32] = literal(value=0)
       mask: bits[5] = literal(value=0b00101)
       masked_p: bits[5] = and(p, mask)
       ret result: bits[32] = priority_sel(masked_p, cases=[zero0, x, zero1, y, z], default=zero2)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal("bits[32]:0"));
}

TEST_P(SelectSimplificationPassTest, SelectWithOnlyNonzeroCaseZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret sel.2: bits[32] = sel(p, cases=[x, literal.1])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"), m::SignExt(m::Not(m::Param("p")))));
}

TEST_P(SelectSimplificationPassTest, SelectWithOnlyNonzeroCaseOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret sel.2: bits[32] = sel(p, cases=[literal.1, x])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"), m::SignExt(m::Param("p"))));
}

TEST_P(SelectSimplificationPassTest, LargerSelectWithOnlyNonzeroCaseZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret sel.2: bits[32] = sel(p, cases=[x, literal.1, literal.1, literal.1])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::And(m::Param("x"), m::SignExt(m::Eq(m::Param("p"), m::Literal(0)))));
}

TEST_P(SelectSimplificationPassTest, LargerSelectWithOnlyNonzeroCaseTwo) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret sel.2: bits[32] = sel(p, cases=[literal.1, literal.1, x, literal.1])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::And(m::Param("x"), m::SignExt(m::Eq(m::Param("p"), m::Literal(2)))));
}

TEST_P(SelectSimplificationPassTest, LargerSelectWithOnlyNonzeroCaseDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret sel.2: bits[32] = sel(p, cases=[literal.1, literal.1], default=x)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::And(m::Param("x"), m::SignExt(m::UGe(m::Param("p"), m::Literal(2)))));
}

TEST_P(SelectSimplificationPassTest, OneHotSelectWithOnlyNonzeroCaseZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret one_hot_sel.2: bits[32] = one_hot_sel(p, cases=[x, literal.1])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"),
                     m::SignExt(m::BitSlice(m::Param("p"), /*start=*/Eq(0),
                                            /*width=*/Eq(1)))));
}

TEST_P(SelectSimplificationPassTest, OneHotSelectWithOnlyNonzeroCaseOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       ret one_hot_sel.2: bits[32] = one_hot_sel(p, cases=[literal.1, x])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"),
                     m::SignExt(m::BitSlice(m::Param("p"), /*start=*/Eq(1),
                                            /*width=*/Eq(1)))));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithOnlyNonzeroCaseZero) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       literal.2: bits[32] = literal(value=0)
       ret priority_sel.3: bits[32] = priority_sel(p, cases=[x, literal.1], default=literal.2)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"),
                     m::SignExt(m::BitSlice(m::Param("p"), /*start=*/Eq(0),
                                            /*width=*/Eq(1)))));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithOnlyNonzeroCaseOne) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       literal.2: bits[32] = literal(value=0)
       ret priority_sel.3: bits[32] = priority_sel(p, cases=[literal.1, x], default=literal.2)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"),
                     m::SignExt(m::Eq(m::Param("p"), m::Literal(0b10)))));
}

TEST_P(SelectSimplificationPassTest, PrioritySelectWithOnlyNonzeroCaseDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32]) -> bits[32] {
       literal.1: bits[32] = literal(value=0)
       literal.2: bits[32] = literal(value=0)
       ret priority_sel.3: bits[32] = priority_sel(p, cases=[literal.1, literal.2], default=x)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("x"),
                     m::SignExt(m::Eq(m::Param("p"), m::Literal(0b00)))));
}

TEST_P(SelectSimplificationPassTest, TwoWayOneHotSelectWhichIsNotOneHot) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32]) -> bits[32] {
       ret one_hot_sel.1: bits[32] = one_hot_sel(p, cases=[x, y])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_P(SelectSimplificationPassTest, LsbOneHotFeedingOneHotSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot.1: bits[3] = one_hot(p, lsb_prio=true)
       ret one_hot_sel.2: bits[32] = one_hot_sel(one_hot.1, cases=[x, y, z])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Param("p"),
                                /*cases=*/{m::Param("x"), m::Param("y")},
                                /*default_value=*/m::Param("z")));
}

TEST_P(SelectSimplificationPassTest,
       LsbOneHotFeedingOneHotSelectWithMultipleUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot.1: bits[3] = one_hot(p, lsb_prio=true)
       sign_ext.2: bits[32] = sign_ext(one_hot.1, new_bit_count=32)
       xor.3: bits[32] = xor(sign_ext.2, z)
       ret one_hot_sel.4: bits[32] = one_hot_sel(one_hot.1, cases=[x, y, xor.3])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_P(SelectSimplificationPassTest, LsbOneHotFeedingMultipleOneHotSelects) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> (bits[32], bits[32]) {
       one_hot.1: bits[3] = one_hot(p, lsb_prio=true)
       one_hot_sel.2: bits[32] = one_hot_sel(one_hot.1, cases=[x, y, z])
       one_hot_sel.3: bits[32] = one_hot_sel(one_hot.1, cases=[y, z, x])
       ret tuple.4: (bits[32], bits[32]) = tuple(one_hot_sel.2, one_hot_sel.3)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Tuple(m::PrioritySelect(m::Param("p"),
                                 /*cases=*/{m::Param("x"), m::Param("y")},
                                 /*default_value=*/m::Param("z")),
               m::PrioritySelect(m::Param("p"),
                                 /*cases=*/{m::Param("y"), m::Param("z")},
                                 /*default_value=*/m::Param("x"))));
}

TEST_P(SelectSimplificationPassTest, MsbOneHotFeedingOneHotSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot.1: bits[3] = one_hot(p, lsb_prio=false)
       ret one_hot_sel.2: bits[32] = one_hot_sel(one_hot.1, cases=[x, y, z])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Reverse(m::Param("p")),
                                /*cases=*/{m::Param("y"), m::Param("x")},
                                /*default_value=*/m::Param("z")));
}

TEST_P(SelectSimplificationPassTest,
       MsbOneHotFeedingOneHotSelectWithMultipleUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       one_hot.1: bits[3] = one_hot(p, lsb_prio=false)
       sign_ext.2: bits[32] = sign_ext(one_hot.1, new_bit_count=32)
       xor.3: bits[32] = xor(sign_ext.2, z)
       ret one_hot_sel.4: bits[32] = one_hot_sel(one_hot.1, cases=[x, y, xor.3])
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_P(SelectSimplificationPassTest, MsbOneHotFeedingMultipleOneHotSelects) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> (bits[32], bits[32]) {
       one_hot.1: bits[3] = one_hot(p, lsb_prio=false)
       one_hot_sel.2: bits[32] = one_hot_sel(one_hot.1, cases=[x, y, z])
       one_hot_sel.3: bits[32] = one_hot_sel(one_hot.1, cases=[y, z, x])
       ret tuple.4: (bits[32], bits[32]) = tuple(one_hot_sel.2, one_hot_sel.3)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Tuple(m::PrioritySelect(m::Reverse(m::Param("p")),
                                 /*cases=*/{m::Param("y"), m::Param("x")},
                                 /*default_value=*/m::Param("z")),
               m::PrioritySelect(m::Reverse(m::Param("p")),
                                 /*cases=*/{m::Param("z"), m::Param("y")},
                                 /*default_value=*/m::Param("x"))));
}

TEST_P(SelectSimplificationPassTest, OneBitOneHot) {
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

TEST_P(SelectSimplificationPassTest, SplittableOneHotSelect) {
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

TEST_P(SelectSimplificationPassTest, SplittablePrioritySelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2], x: bits[8], y: bits[4]) -> bits[8] {
       literal.1: bits[4] = literal(value=0b1001)
       literal.2: bits[4] = literal(value=0b1000)
       bit_slice.3: bits[2] = bit_slice(x, start=0, width=2)
       concat.4: bits[8] = concat(literal.1, bit_slice.3, bit_slice.3)
       concat.5: bits[8] = concat(literal.2, y)
       ret result: bits[8] = priority_sel(p, cases=[concat.4, concat.5], default=x)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  // Should be split into a concat of three PrioritySelects. The first and
  // last should select among two cases. The middle should select among three
  // cases.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::PrioritySelect(m::Param("p"),
                                  /*cases=*/
                                  {m::BitSlice(/*start=*/5, /*width=*/3),
                                   m::BitSlice(/*start=*/5, /*width=*/3)},
                                  /*default_value=*/
                                  m::BitSlice(/*start=*/5, /*width=*/3)),
                m::PrioritySelect(
                    m::Param("p"),
                    /*cases=*/
                    {m::BitSlice(/*start=*/2, /*width=*/3),
                     m::BitSlice(/*start=*/2, /*width=*/3)},
                    /*default_value=*/m::BitSlice(/*start=*/2, /*width=*/3)),
                m::PrioritySelect(
                    m::Param("p"),
                    /*cases=*/
                    {m::BitSlice(/*start=*/0, /*width=*/2),
                     m::BitSlice(/*start=*/0, /*width=*/2)},
                    /*default_value=*/m::BitSlice(/*start=*/0, /*width=*/2))));
}

TEST_P(SelectSimplificationPassTest, SelectsWithCommonCase0) {
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

TEST_P(SelectSimplificationPassTest, SelectsWithCommonCase1) {
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

TEST_P(SelectSimplificationPassTest, SelectsWithCommonCase2) {
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

TEST_P(SelectSimplificationPassTest, SelectsWithCommonCase3) {
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

// Performs the following:
//
//  fn f(x: bool) {
//    one_hot(1 ++ x, lsb_prio)
//  }
//
// Which can simplify to: if x { 0b001 } else { 0b010 }
TEST_P(SelectSimplificationPassTest, OneHotWithSingleUnknownBit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue x = fb.Param("x", u1);
  BValue concat = fb.Concat({fb.Literal(UBits(1, 1)), x});
  BValue one_hot = fb.OneHot(concat, LsbOrMsb::kLsb);
  (void)one_hot;  // retval
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // Note: this doesn't simplify the concat/slice but if it did the selector
  // would be 'x'.
  EXPECT_THAT(
      f->return_value(),
      m::Select(
          m::BitSlice(m::Concat(m::Literal("bits[1]:1"), m::Param("x")), 0, 1),
          {m::Literal("bits[3]:0b010"), m::Literal("bits[3]:0b001")}));
}

TEST_P(SelectSimplificationPassTest, ReorderableAffineSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  Type* u2 = p->GetBitsType(2);
  BValue selector = fb.Subtract(fb.Literal(UBits(2, 2)), fb.Param("p1", u2));
  fb.Select(selector, {fb.Param("a", u32), fb.Param("b", u32),
                       fb.Param("c", u32), fb.Param("d", u32)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("p1"), {m::Param("c"), m::Param("b"),
                                         m::Param("a"), m::Param("d")}));
}

TEST_P(SelectSimplificationPassTest, ReorderableSelectWithOperandReuse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  Type* u2 = p->GetBitsType(2);
  BValue p1 = fb.Param("p1", u2);
  BValue selector = fb.UMul(p1, p1);
  fb.Select(selector, {fb.Param("a", u32), fb.Param("b", u32),
                       fb.Param("c", u32), fb.Param("d", u32)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("p1"), {m::Param("a"), m::Param("b"),
                                         m::Param("a"), m::Param("b")}));
}

TEST_P(SelectSimplificationPassTest, UnchangedBitsSelSqueeze) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue high = fb.Param("high", p->GetBitsType(10));
  BValue left_a = fb.Param("left_a", p->GetBitsType(10));
  BValue right_a = fb.Param("right_a", p->GetBitsType(10));
  BValue mid = fb.Param("mid", p->GetBitsType(10));
  BValue left_b = fb.Param("left_b", p->GetBitsType(10));
  BValue right_b = fb.Param("right_b", p->GetBitsType(10));
  BValue low = fb.Param("low", p->GetBitsType(10));
  BValue left_concat =
      fb.Concat({high, left_a, mid, left_b, low}, SourceInfo(), "left_concat");
  BValue right_concat = fb.Concat({high, right_a, mid, right_b, low},
                                  SourceInfo(), "right_concat");
  fb.Tuple({
      fb.Select(fb.Param("sel_selector", p->GetBitsType(1)),
                {left_concat, right_concat}, std::nullopt, SourceInfo(),
                "choose_sel"),
      fb.PrioritySelect(fb.Param("prio", p->GetBitsType(1)), {left_concat},
                        right_concat, SourceInfo(), "choose_prio"),
      fb.OneHotSelect(fb.OneHot(fb.Param("ohs_selector", p->GetBitsType(1)),
                                LsbOrMsb::kLsb),
                      {left_concat, right_concat}, SourceInfo(), "choose_ohs"),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedRecordIr sri(p.get());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};

  ASSERT_THAT(Run(f), IsOkAndHolds(true));

  ASSERT_THAT(
      f->return_value(),
      m::Tuple(m::Concat(high.node(), m::TupleIndex(m::Select(), 1), mid.node(),
                         m::TupleIndex(m::Select(), 0), low.node()),
               m::Concat(high.node(), m::TupleIndex(m::PrioritySelect(), 1),
                         mid.node(), m::TupleIndex(m::PrioritySelect(), 0),
                         low.node()),
               m::Concat(high.node(), m::TupleIndex(m::OneHotSelect(), 1),
                         mid.node(), m::TupleIndex(m::OneHotSelect(), 0),
                         low.node())));
  EXPECT_EQ(f->return_value()->operand(0)->operand(1)->operand(
                0),  // first tuple-index
            f->return_value()->operand(0)->operand(3)->operand(
                0));  // second tuple-index
  EXPECT_EQ(f->return_value()->operand(1)->operand(1)->operand(
                0),  // first tuple-index
            f->return_value()->operand(1)->operand(3)->operand(
                0));  // second tuple-index
  EXPECT_EQ(f->return_value()->operand(2)->operand(1)->operand(
                0),  // first tuple-index
            f->return_value()->operand(2)->operand(3)->operand(
                0));  // second tuple-index
}

TEST_P(SelectSimplificationPassTest, NarrowWithRangeAnalysis) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u1 = p->GetBitsType(1);
  BValue inp = fb.Param("inp", u1);
  // exp is [[-1], [0]]
  BValue exp = fb.SignExtend(inp, 32);
  BValue s1 = fb.Select(fb.Param("a", u1), exp, fb.Literal(UBits(1, 32)));
  BValue s2 = fb.Select(fb.Param("b", u1), s1, fb.Literal(UBits(2, 32)));
  BValue s3 = fb.Select(fb.Param("c", u1), s2, fb.Literal(UBits(3, 32)));
  fb.Select(fb.Param("d", u1), s3, fb.Literal(UBits(3, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};

  if (GetParam() == AnalysisType::kTernary) {
    ASSERT_THAT(Run(f), IsOkAndHolds(false));
    return;
  }
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // NB other passes need to do a lot of cleanup to get rid of chained
  // '(bit-slice (sign-ext ...))'. Just do a basic check for the first level.
  EXPECT_THAT(f->return_value(),
              m::SignExt(m::Select(m::Param("d"),
                                   {m::BitSlice(m::Literal()),
                                    m::BitSlice(m::SignExt(m::Select()))})));
}

// Regression test for https://github.com/google/xls/issues/2673; checks that we
// work correctly (and don't crash) when processing a single-bit select with a
// default value. (This is usually canonicalized to two cases first, but we
// can handle it correctly either way... and with enough other noise in the
// program, it isn't always canonicalized.)
TEST_P(SelectSimplificationPassTest, SingleBitSelectWithDefaultRegression) {
  const std::string program = R"(
fn FuzzTest(p0: bits[1] id=1, p6: bits[64] id=12) -> bits[1] {
  literal.6: bits[1] = literal(value=0, id=6)
  eq.7: bits[1] = eq(p0, literal.6, id=7)
  concat.8: bits[1] = concat(eq.7, id=8)
  not.28: bits[1] = not(p0, id=28)
  literal.4: bits[1] = literal(value=0, id=4)
  not.32: bits[1] = not(concat.8, id=32)
  literal.2: bits[1] = literal(value=0, id=2)
  literal.13: bits[64] = literal(value=0, id=13)
  literal.14: bits[64] = literal(value=0, id=14)
  and.27: bits[1] = and(p0, p0, id=27)
  and.29: bits[1] = and(not.28, literal.4, id=29)
  and.31: bits[1] = and(concat.8, literal.6, id=31)
  and.33: bits[1] = and(not.32, p0, id=33)
  nand.3: bits[1] = nand(literal.2, id=3)
  sel.5: bits[1] = sel(p0, cases=[literal.4], default=p0, id=5)
  priority_sel.9: bits[1] = priority_sel(concat.8, cases=[literal.6], default=p0, id=9)
  zero_ext.10: bits[1] = zero_ext(p0, new_bit_count=1, id=10)
  sgt.11: bits[1] = sgt(p0, p0, id=11)
  priority_sel.15: bits[64] = priority_sel(p0, cases=[literal.13], default=literal.14, id=15)
  or.30: bits[1] = or(and.27, and.29, id=30)
  or.34: bits[1] = or(and.31, and.33, id=34)
  literal.35: bits[64] = literal(value=0, id=35)
  ret or_reduce.16: bits[1] = or_reduce(p0, id=16)
})";

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  solvers::z3::ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(f), IsOk());
}

INSTANTIATE_TEST_SUITE_P(SelectSimplificationPassTest,
                         SelectSimplificationPassTest,
                         testing::Values(AnalysisType::kTernary,
                                         AnalysisType::kRange),
                         testing::PrintToStringParamName());

void IrFuzzSelectSimplification(FuzzPackageWithArgs fuzz_package_with_args) {
  SelectSimplificationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzSelectSimplification)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
