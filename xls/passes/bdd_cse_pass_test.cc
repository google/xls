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

void IrFuzzBddCse(FuzzPackageWithArgs fuzz_package_with_args) {
  BddCsePass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzBddCse)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
