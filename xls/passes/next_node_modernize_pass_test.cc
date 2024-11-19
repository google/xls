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

#include "xls/passes/next_node_modernize_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;
namespace xls {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class NextNodeModernizePassTest : public IrTestBase {};

TEST_F(NextNodeModernizePassTest, BasicModernize) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.StateElement("foo", UBits(0, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * pr,
                           pb.Build({pb.Add(st, pb.Literal(UBits(1, 32)))}));
  ScopedRecordIr sri(p.get());

  PassResults res;
  NextNodeModernizePass pass;
  ASSERT_THAT(pass.Run(p.get(), {}, &res), IsOkAndHolds(true));
  EXPECT_THAT(pr->NextState(), UnorderedElementsAre(m::StateRead("foo")));
  EXPECT_THAT(pr->next_values(),
              UnorderedElementsAre(m::Next(
                  m::StateRead("foo"),
                  m::Add(m::StateRead("foo"), m::Literal(UBits(1, 32))))));
}
TEST_F(NextNodeModernizePassTest, AlreadyModern) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.StateElement("foo", UBits(0, 32));
  pb.Next(st, pb.Add(st, pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK(pb.Build().status());
  ScopedRecordIr sri(p.get());

  PassResults res;
  NextNodeModernizePass pass;
  ASSERT_THAT(pass.Run(p.get(), {}, &res), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
