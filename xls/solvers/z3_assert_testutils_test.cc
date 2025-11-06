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

#include "xls/solvers/z3_assert_testutils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace xls::solvers::z3 {
namespace {

using ::testing::Not;

class AssertCleanTest : public IrTestBase {};

TEST_F(AssertCleanTest, CleanAssert) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Assert(fb.Literal(Value::Token()), fb.Eq(x, x), "foobar");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f, IsAssertClean());
}

TEST_F(AssertCleanTest, Clean2Asserts) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Assert(fb.Literal(Value::Token()), fb.Eq(x, x), "foo");
  fb.Assert(fb.Literal(Value::Token()),
            fb.Or(fb.Eq(y, fb.Literal(UBits(0, 32))), fb.Ne(fb.Add(x, y), x)),
            "bar");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f, IsAssertClean());
}

TEST_F(AssertCleanTest, TriggeredAssert) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue a = fb.Param("a", p->GetBitsType(3));
  BValue b = fb.Param("b", p->GetBitsType(3));
  BValue c = fb.Param("c", p->GetBitsType(3));
  BValue d = fb.Param("d", p->GetBitsType(3));
  BValue ohs = fb.OneHotSelect(x, {a, b, c});
  BValue pri = fb.PrioritySelect(x, {a, b, c}, d);
  fb.Assert(fb.Literal(Value::Token()), fb.Eq(ohs, pri), "foobar");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f, Not(IsAssertClean()));
}

}  // namespace
}  // namespace xls::solvers::z3
