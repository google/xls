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
#include "third_party/googlefuzztest/fuzztest_macros.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
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
    OptimizationContext context;
    return LutConversionPass().Run(p, OptimizationPassOptions(), &results,
                                   context);
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
      m::Select(m::BitSlice(m::Shrl(m::Param("x"), m::Literal(1)), /*start=*/0,
                            /*width=*/2),
                {m::Param("x"), m::Literal(0), m::Literal(1), m::Literal(2)}));
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

TEST_F(LutConversionPassTest, MultipleSources) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn test(x: bits[2], y: bits[1]) -> bits[3] {
        literal.0: bits[3] = literal(value=0)
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        literal.4: bits[3] = literal(value=4)
        literal.5: bits[3] = literal(value=5)
        literal.6: bits[3] = literal(value=6)
        literal.7: bits[3] = literal(value=7)
        six_x: bits[5] = umul(x, literal.6)
        seven_y: bits[5] = umul(y, literal.7)
        selector: bits[5] = sub(six_x, seven_y)
        ret result: bits[3] = sel(selector, cases=[literal.0, literal.1, literal.2, literal.3, literal.4, literal.5, literal.6, literal.7], default=literal.0)
     }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Concat(m::Param("y"), m::Param("x")),
                {m::Literal(0), m::Literal(6), m::Literal(0), m::Literal(0),
                 m::Literal(0), m::Literal(0), m::Literal(5), m::Literal(0)}));
}

// Found by minimizing a real example during development.
TEST_F(LutConversionPassTest, ComplexExample) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn test(x: bits[2]) -> bits[1] {
      bit_slice.2: bits[1] = bit_slice(x, start=0, width=1)
      bit_slice.3: bits[1] = bit_slice(x, start=0, width=1)
      literal.4: bits[2] = literal(value=0)
      zero_ext.5: bits[2] = zero_ext(bit_slice.2, new_bit_count=2)
      sel.6: bits[2] = sel(bit_slice.3, cases=[literal.4, zero_ext.5])
      literal.7: bits[1] = literal(value=0)
      ret sel.8: bits[1] = sel(sel.6, cases=[literal.7, literal.7], default=literal.7)
    }
  )",
                                                       p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("x"), {m::Literal(0), m::Literal(0),
                                        m::Literal(0), m::Literal(0)}));
}

void IrFuzzLutConversionPassTest(
    const PackageAndTestParams& paramaterized_package) {
  LutConversionPass pass;
  OptimizationPassChangesOutputs(paramaterized_package, pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzLutConversionPassTest)
    .WithDomains(IrFuzzDomainWithParams(/*param_set_count=*/10));

}  // namespace
}  // namespace xls
