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

#include "xls/passes/identity_removal_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class IdentityRemovalPassTest : public IrTestBase {
 protected:
  IdentityRemovalPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(
        bool changed,
        IdentityRemovalPass().Run(p, OptimizationPassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .Run(p, OptimizationPassOptions(), &results)
                            .status());
    // Return whether cse changed anything.
    return changed;
  }
};

TEST_F(IdentityRemovalPassTest, IdentityChainRemoval) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_neg(x:bits[8]) -> bits[8] {
        one: bits[8] = literal(value=1)
        v1:  bits[8] = identity(x)
        add: bits[8] = add(v1, one)
        v2:  bits[8] = identity(add)
        v3:  bits[8] = identity(v2)
        v4:  bits[8] = identity(v3)
        v5:  bits[8] = identity(v4)
        v6:  bits[8] = identity(v5)
        v7:  bits[8] = identity(v6)
        v8:  bits[8] = identity(v7)
        v9:  bits[8] = identity(v8)
        ret add2:bits[8] = sub(v9, one)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Sub(m::Add(m::Param("x"), m::Literal(1)), m::Literal(1)));
}

TEST_F(IdentityRemovalPassTest, IdentityRemovalFromParam) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn simple_param(x:bits[8]) -> bits[8] {
        ret res: bits[8] = identity(x)
     }
  )",
                                                       p.get()));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

}  // namespace
}  // namespace xls
