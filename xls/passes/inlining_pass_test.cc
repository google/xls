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

#include "xls/passes/inlining_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/dce_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using testing::AnyOf;
using testing::Eq;

class InliningPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Inline(Package* package) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         InliningPass().Run(package, PassOptions(), &results));
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .Run(package, PassOptions(), &results)
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
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));
  Function* f = FindFunction("caller", package.get());
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
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));
  Function* f = FindFunction("caller", package.get());
  EXPECT_THAT(f->return_value(), m::Add(m ::Literal(2), m::Literal(2)));
}

TEST_F(InliningPassTest, NamePropagation) {
  const std::string program = R"(
package some_package

fn callee(x: bits[32], y: bits[32]) -> bits[32] {
  y_squared: bits[32] = umul(y, y)
  ret x_bamboozled: bits[32] = add(x, y_squared)
}

fn invoke_is_not_named(foo: bits[32], qux: bits[32]) -> bits[32] {
  ret invoke.111: bits[32] = invoke(foo, qux, to_apply=callee)
}

fn invoke_is_named(baz: bits[32], zub: bits[32]) -> bits[32] {
  ret special_name: bits[32] = invoke(baz, zub, to_apply=callee)
}

fn operands_not_named() -> bits[32] {
  literal.42: bits[32] = literal(value=232)
  literal.43: bits[32] = literal(value=222)
  ret invoke.333: bits[32] = invoke(literal.42, literal.43, to_apply=callee)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(program));
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));
  {
    // Inlined function result should get a name derived from names inside the
    // inlined function because the invoke instruction has no name.
    Node* ret =
        FindFunction("invoke_is_not_named", package.get())->return_value();
    EXPECT_EQ(ret->GetName(), "foo_bamboozled");
    EXPECT_THAT(ret, m::Add(m::Name("foo"), m::Name("qux_squared")));
  }

  {
    // Inlined function result should not get a name derived from names inside
    // the inlined function because the invoke instruction itself has a name.
    Node* ret = FindFunction("invoke_is_named", package.get())->return_value();
    EXPECT_EQ(ret->GetName(), "special_name");
    EXPECT_THAT(ret, m::Add(m::Name("baz"), m::Name("zub_squared")));
  }

  {
    // If the operands of the invoke do not have assigned names then copy the
    // name from inside the invoked function (if it has one).
    Node* ret =
        FindFunction("operands_not_named", package.get())->return_value();
    EXPECT_EQ(ret->GetName(), "x_bamboozled");
    EXPECT_FALSE(ret->operand(0)->HasAssignedName());
    EXPECT_EQ(ret->operand(1)->GetName(), "y_squared");
  }
}

TEST_F(InliningPassTest, NamePropagationWithPassThroughParam) {
  const std::string program = R"(
package some_package

fn callee(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}

fn f(foobar: bits[32]) -> bits[32] {
  ret invoke.42: bits[32] = invoke(foobar, to_apply=callee)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(program));
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));
  EXPECT_THAT(FindFunction("f", package.get())->return_value(),
              m::Name("foobar"));
}

// Verifies that Cover and Assert ops have their labels differentiated when
// "duplicated" via function inlining.
TEST_F(InliningPassTest, CoversAndAssertsDeduplicated) {
  const std::string kProgram = R"(
package some_package

fn callee(the_token: token, x: bits[32]) -> (token, bits[32]) {
  literal.10: bits[32] = literal(value=666)
  eq.20: bits[1] = eq(x, literal.10)
  ne.30: bits[1] = ne(x, literal.10)
  cover.40: token = cover(the_token, ne.30, label="cover_label")
  assert.50: token = assert(cover.40, eq.20, label="assert_label", message="derp")
  ret tuple.60: (token, bits[32]) = tuple(assert.50, x)
}

fn caller(the_token: token, x: bits[32]) -> (token, bits[32]) {
  invoke.110: (token, bits[32]) = invoke(the_token, x, to_apply=callee)
  tuple_index.120: token = tuple_index(invoke.110, index=0)
  tuple_index.130: bits[32] = tuple_index(invoke.110, index=1)
  ret invoke.140: (token, bits[32]) = invoke(tuple_index.120, tuple_index.130, to_apply=callee)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kProgram));
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("caller"));
  for (const auto* node : f->nodes()) {
    if (node->Is<Cover>()) {
      const Cover* cover = node->As<Cover>();
      EXPECT_THAT(cover->label(), AnyOf(Eq("caller_0_callee_cover_label"),
                                        Eq("caller_1_callee_cover_label")));
    } else if (node->Is<Assert>()) {
      const Assert* asrt = node->As<Assert>();
      EXPECT_THAT(asrt->label(), AnyOf(Eq("caller_0_callee_assert_label"),
                                       Eq("caller_1_callee_assert_label")));
    }
  }
}

}  // namespace
}  // namespace xls
