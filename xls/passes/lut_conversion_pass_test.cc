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

#include "xls/passes/lut_conversion_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace xls {
namespace {

namespace m = ::xls::op_matchers;

using ::absl_testing::IsOkAndHolds;

class LutConversionPassTest : public IrTestBase {
 protected:
  LutConversionPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return LutConversionPass().Run(p, OptimizationPassOptions(), &results);
  }
};

TEST_F(LutConversionPassTest, SimpleSelectNoChange) {
  auto p = CreatePackage();
  XLS_ASSERT_OK(ParseFunction(R"(
     fn simple_select(x: bits[3]) -> bits[3] {
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        literal.4: bits[3] = literal(value=4)
        literal.5: bits[3] = literal(value=5)
        ret result: bits[3] = sel(x, cases=[literal.1, literal.2, literal.3, literal.4], default=literal.5)
     }
  )",
                              p.get())
                    .status());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(LutConversionPassTest, DoubledSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_select(x: bits[3]) -> bits[3] {
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        literal.4: bits[3] = literal(value=4)
        literal.5: bits[3] = literal(value=5)
        selector: bits[3] = add(x, x)
        ret result: bits[3] = sel(selector, cases=[literal.1, literal.2, literal.3, literal.4], default=literal.5)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("x"),
                {m::Literal(1), m::Literal(3), m::Literal(5), m::Literal(5),
                 m::Literal(1), m::Literal(3), m::Literal(5), m::Literal(5)}));
}

TEST_F(LutConversionPassTest, TripledSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_select(x: bits[3]) -> bits[3] {
        literal.0: bits[3] = literal(value=0)
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        doubled_x: bits[3] = add(x, x)
        selector: bits[3] = add(doubled_x, x)
        ret result: bits[3] = sel(selector, cases=[literal.0, literal.1, literal.2, literal.3], default=x)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("x"),
                {m::Literal(0), m::Literal(3), m::Param("x"), m::Literal(1),
                 m::Param("x"), m::Param("x"), m::Literal(2), m::Param("x")}));
}

TEST_F(LutConversionPassTest, AffineSelector) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_select(x: bits[3]) -> bits[3] {
        literal.0: bits[3] = literal(value=0)
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        half_x: bits[3] = shrl(x, literal.1)
        selector: bits[3] = sub(half_x, literal.1)
        ret result: bits[3] = sel(selector, cases=[literal.0, literal.1, literal.2, literal.3], default=x)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("x"),
                {m::Param("x"), m::Param("x"), m::Literal(0), m::Literal(0),
                 m::Literal(1), m::Literal(1), m::Literal(2), m::Literal(2)}));
}

TEST_F(LutConversionPassTest, IneligibleDominator) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_select(x: bits[3]) -> bits[3] {
        literal.0: bits[3] = literal(value=0)
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        truncated_x: bits[2] = bit_slice(x, start=0, width=2)
        short_one: bits[2] = literal(value=1)
        selector: bits[2] = sub(truncated_x, short_one)
        ret result: bits[3] = sel(selector, cases=[literal.0, literal.1, literal.2, literal.3])
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::BitSlice(), {m::Literal(3), m::Literal(0),
                                        m::Literal(1), m::Literal(2)}));
}

}  // namespace
}  // namespace xls
