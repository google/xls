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

#include "xls/passes/unroll_pass.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

TEST(UnrollPassTest, UnrollsCountedForWithInvariantArgsAndStride) {
  const std::string program = R"(
package some_package

fn body(i: bits[4], accum: bits[32], zero: bits[32]) -> bits[32] {
  zero_ext.3: bits[32] = zero_ext(i, new_bit_count=32)
  add.4: bits[32] = add(zero_ext.3, accum)
  ret add.5: bits[32] = add(add.4, zero)
}

fn unrollable() -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret counted_for.2: bits[32] = counted_for(literal.1, trip_count=2, stride=2, body=body, invariant_args=[literal.1])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("unrollable"));
  PassResults results;
  UnrollPass pass;
  EXPECT_THAT(pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results),
              IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Invoke(m::Literal(2),
                        m::Invoke(m::Literal(0), m::Literal(0), m::Literal(0)),
                        m::Literal(0)));
}

}  // namespace
}  // namespace xls
