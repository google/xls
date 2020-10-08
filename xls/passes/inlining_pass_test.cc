// Copyright 2020 Google LLC
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

#include "xls/passes/inlining_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/dce_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class InliningPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Inline(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(
        bool changed, InliningPass().RunOnFunction(f, PassOptions(), &results));
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .RunOnFunction(f, PassOptions(), &results)
                            .status());
    return changed;
  }
};

TEST_F(InliningPassTest, AddWrapper) {
  const std::string program = R"(
package some_package

fn callee(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn caller() -> bits[32] {
  literal.2: bits[32] = literal(value=2)
  ret invoke.3: bits[32] = invoke(literal.2, literal.2, to_apply=callee)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(program));
  Function* f = FindFunction("caller", package.get());
  ASSERT_THAT(Inline(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Literal(2), m::Literal(2)));
}

TEST_F(InliningPassTest, Transitive) {
  const std::string program = R"(
package some_package

fn callee2(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn callee1(x: bits[32], y: bits[32]) -> bits[32] {
  ret invoke.2: bits[32] = invoke(x, y, to_apply=callee2)
}

fn caller() -> bits[32] {
  literal.3: bits[32] = literal(value=2)
  ret invoke.4: bits[32] = invoke(literal.3, literal.3, to_apply=callee1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(program));
  Function* f = FindFunction("caller", package.get());
  ASSERT_THAT(Inline(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m ::Literal(2), m::Literal(2)));
}

}  // namespace
}  // namespace xls
