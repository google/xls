// Copyright 2025 The XLS Authors
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

#include "xls/passes/select_merging_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Eq;

class SelectMergingPassTest : public IrTestBase {
 protected:
  SelectMergingPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;
    SelectMergingPass tern_pass;
    XLS_ASSIGN_OR_RETURN(
        bool changed, tern_pass.RunOnFunctionBase(f, OptimizationPassOptions(),
                                                  &results, context));
    return changed;
  }
};

TEST_F(SelectMergingPassTest, OneHotSelectFeedingOneHotSelect) {
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

TEST_F(SelectMergingPassTest, OneHotSelectFeedingOneHotSelectWithMultipleUses) {
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

TEST_F(SelectMergingPassTest, PrioritySelectFeedingPrioritySelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p0: bits[2], p1: bits[2], x: bits[32], y: bits[32], z: bits[32], d: bits[32]) -> bits[32] {
       priority_sel.1: bits[32] = priority_sel(p0, cases=[x, y], default=d)
       literal.2: bits[32] = literal(value=0)
       ret priority_sel.3: bits[32] = priority_sel(p1, cases=[priority_sel.1, z], default=literal.2)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(
          m::Concat(
              m::BitSlice(m::Param("p1"), /*start=*/Eq(1), /*width=*/Eq(1)),
              m::And(m::BitSlice(m::Param("p1"), /*start=*/Eq(0),
                                 /*width=*/Eq(1)),
                     m::Eq(m::Param("p0"), m::Literal(0))),
              m::And(m::SignExt(m::BitSlice(m::Param("p1"), /*start=*/Eq(0),
                                            /*width=*/Eq(1))),
                     m::Param("p0"))),
          /*cases=*/
          {m::Param("x"), m::Param("y"), m::Param("d"), m::Param("z")},
          /*default_value=*/m::Literal(0)));
}

TEST_F(SelectMergingPassTest,
       PrioritySelectFeedingPrioritySelectWithMultipleUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p0: bits[2], p1: bits[2], x: bits[32], y: bits[32], z: bits[32], d: bits[32]) -> bits[32] {
       priority_sel.1: bits[32] = priority_sel(p0, cases=[x, y], default=d)
       neg.2: bits[32] = neg(priority_sel.1)
       literal.3: bits[32] = literal(value=0)
       priority_sel.4: bits[32] = priority_sel(p1, cases=[priority_sel.1, z], default=literal.3)
       ret add.5: bits[32] = add(neg.2, priority_sel.4)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(SelectMergingPassTest, PrioritySelectFeedingPrioritySelectDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p0: bits[2], p1: bits[2], x: bits[32], y: bits[32], z: bits[32], d: bits[32]) -> bits[32] {
       priority_sel.1: bits[32] = priority_sel(p0, cases=[x, y], default=d)
       literal.2: bits[32] = literal(value=0)
       ret priority_sel.3: bits[32] = priority_sel(p1, cases=[literal.2, z], default=priority_sel.1)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Concat(m::Param("p0"), m::Param("p1")),
                  /*cases=*/
                  {m::Literal(0), m::Param("z"), m::Param("x"), m::Param("y")},
                  /*default_value=*/m::Param("d")));
}

TEST_F(SelectMergingPassTest, OneHotSelectChain) {
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

TEST_F(SelectMergingPassTest, OneHotSelectTree) {
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

TEST_F(SelectMergingPassTest, SimpleMergeablePrioritySelects) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], q: bits[1], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       priority_sel.1: bits[32] = priority_sel(q, cases=[y], default=z)
       ret priority_sel.2: bits[32] = priority_sel(p, cases=[x], default=priority_sel.1)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Concat(m::Param("q"), m::Param("p")),
                                /*cases=*/{m::Param("x"), m::Param("y")},
                                /*default_value=*/m::Param("z")));
}

TEST_F(SelectMergingPassTest, SimpleMergeablePrioritySelectsAfterSwap) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], q: bits[1], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
       priority_sel.1: bits[32] = priority_sel(q, cases=[y], default=z)
       ret priority_sel.2: bits[32] = priority_sel(p, cases=[priority_sel.1], default=x)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Concat(m::Param("q"), m::Not(m::Param("p"))),
                                /*cases=*/{m::Param("x"), m::Param("y")},
                                /*default_value=*/m::Param("z")));
}

TEST_F(SelectMergingPassTest, ComplexMergeablePrioritySelects) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], q: bits[2], x: bits[32], y: bits[32], z: bits[32], w: bits[32]) -> bits[32] {
       priority_sel.1: bits[32] = priority_sel(q, cases=[y, z], default=w)
       priority_sel.2: bits[32] = priority_sel(q, cases=[z, w], default=y)
       ret priority_sel.3: bits[32] = priority_sel(p, cases=[priority_sel.1], default=priority_sel.2)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(
          m::Concat(m::Param("q"),
                    m::And(m::Param("p"), m::Eq(m::Param("q"), m::Literal(0))),
                    m::And(m::SignExt(m::Param("p")), m::Param("q"))),
          /*cases=*/
          {m::Param("y"), m::Param("z"), m::Param("w"), m::Param("z"),
           m::Param("w")},
          /*default_value=*/m::Param("y")));
}

}  // namespace

}  // namespace xls
