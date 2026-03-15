// Copyright 2026 The XLS Authors
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

#include "xls/passes/to_zero_ext_pass.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class ToZeroExtPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return ToZeroExtPass().Run(p, OptimizationPassOptions(), &results, context);
  }
};

TEST_F(ToZeroExtPassTest, SimplePrefixZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  fb.Concat({zero, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::Param("x")));
}

TEST_F(ToZeroExtPassTest, MultiplePrefixZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue zero8 = fb.Literal(UBits(0, 8));
  BValue zero1 = fb.Literal(UBits(0, 1));
  fb.Concat({zero8, zero1, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::Param("x")));
  EXPECT_EQ(f->return_value()->BitCountOrDie(), 17);
}

TEST_F(ToZeroExtPassTest, KnownZeroPrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  // (x & 0) is known to be zero.
  BValue zero = fb.And(x, fb.Literal(UBits(0, 8)));
  fb.Concat({zero, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // PartialInfoQueryEngine should be able to see that 'zero' is all zeros.
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::Param("x")));
}

TEST_F(ToZeroExtPassTest, NoChangeNonPrefixZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  fb.Concat({x, zero});
  XLS_ASSERT_OK(fb.Build().status());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ToZeroExtPassTest, NoChangeInMiddleZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  fb.Concat({x, zero, y});
  XLS_ASSERT_OK(fb.Build().status());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ToZeroExtPassTest, NoChangeNonZeroPrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue one = fb.Literal(UBits(1, 8));
  fb.Concat({one, x});
  XLS_ASSERT_OK(fb.Build().status());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
