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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/unroll_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AnyOf;
using ::testing::Eq;

class InliningPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Inline(
      Package* package,
      InliningPass::InlineDepth depth = InliningPass::InlineDepth::kFull) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(bool changed, InliningPass(depth).Run(
                                           package, OptimizationPassOptions(),
                                           &results, context));
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .Run(package, OptimizationPassOptions(), &results, context)
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

TEST_F(InliningPassTest, FfiFunctionsNotInlined) {
  const std::string program = R"(
package some_package

#[ffi_proto("""code_template: "verilog_module {fn} (.a({x}), .b({y}), .out({return}));"
""")]
fn ffi_callee(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn caller() -> bits[32] {
  literal.2: bits[32] = literal(value=2)
  ret invoke.3: bits[32] = invoke(literal.2, literal.2, to_apply=ffi_callee)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(program));
  // No modification expected as the ffi-function is not inlined.
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(false));
  Function* f = FindFunction("caller", package.get());
  EXPECT_THAT(f->return_value(), m::Invoke(m::Literal(2), m::Literal(2)));
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

TEST_F(InliningPassTest, TransitiveLeaf) {
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
  ASSERT_THAT(Inline(package.get(), InliningPass::InlineDepth::kLeafOnly),
              IsOkAndHolds(true));
  Function* f = FindFunction("caller", package.get());
  Function* c1 = FindFunction("callee1", package.get());
  EXPECT_THAT(f->return_value(), m::Add());
  EXPECT_THAT(c1->return_value(), m::Add(m::Param("x"), m::Param("y")));
}
TEST_F(InliningPassTest, TransitiveLeafMultiUse) {
  const std::string program = R"(
package some_package

fn callee2(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn callee3(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.8: bits[32] = add(x, y)
}

fn callee1(x: bits[32], y: bits[32]) -> bits[32] {
  invoke.2: bits[32] = invoke(x, y, to_apply=callee2)
  invoke.9: bits[32] = invoke(y, x, to_apply=callee2)
  ret add.10: bits[32] = add(invoke.2, invoke.9)
}

fn caller() -> bits[32] {
  literal.3: bits[32] = literal(value=2)
  ret invoke.4: bits[32] = invoke(literal.3, literal.3, to_apply=callee1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(program));
  ScopedRecordIr sri(package.get());
  ASSERT_THAT(Inline(package.get(), InliningPass::InlineDepth::kLeafOnly),
              IsOkAndHolds(true));
  Function* f = FindFunction("caller", package.get());
  Function* c1 = FindFunction("callee1", package.get());
  EXPECT_THAT(f->return_value(), m::Invoke());
  EXPECT_THAT(c1->return_value(), m::Add(m::Add(m::Param("x"), m::Param("y")),
                                         m::Add(m::Param("y"), m::Param("x"))));
}

TEST_F(InliningPassTest, TransitiveWithFfiLeafFunction) {
  const std::string program = R"(
package some_package

#[ffi_proto("""code_template: "verilog_module {fn} (.a({x}), .b({y}), .out({return}));"
""")]
fn ffi_callee(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn callee1(x: bits[32], y: bits[32]) -> bits[32] {
  ret invoke.2: bits[32] = invoke(x, y, to_apply=ffi_callee)
}

fn caller() -> bits[32] {
  literal.3: bits[32] = literal(value=2)
  ret invoke.4: bits[32] = invoke(literal.3, literal.3, to_apply=callee1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(program));

  // One round of inlining expected
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));

  // Now there is still an invoke left, but it is FFI. So re-running inlining
  // should not change anything.
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(false));

  // The caller now contains the inlined function callee1, which invokes ffi. So
  // that invoke is all that is expected to be left in the toplevel function.
  Function* f = FindFunction("caller", package.get());
  EXPECT_THAT(f->return_value(), m::Invoke(m::Literal(2), m::Literal(2)));

  // The invoke that was not inlined points to the ffi_callee
  EXPECT_EQ(f->return_value()->As<Invoke>()->to_apply()->name(), "ffi_callee");
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

TEST_F(InliningPassTest, SingleInline) {
  auto p = CreatePackage();
  FunctionBuilder i1(TestName() + "InvokeTarget1", p.get());
  i1.Add(i1.Param("x", p->GetBitsType(4)), i1.Param("y", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * i1_func, i1.Build());

  FunctionBuilder i2(TestName() + "InvokeTarget2", p.get());
  i2.UMul(i2.Param("x", p->GetBitsType(4)), i2.Param("y", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * i2_func, i2.Build());

  FunctionBuilder top(TestName() + "Top", p.get());
  BValue p1 = top.Param("p1", p->GetBitsType(4));
  BValue p2 = top.Param("p2", p->GetBitsType(4));
  BValue i1_res = top.Invoke({p1, p2}, i1_func);
  BValue i2_res = top.Invoke({p1, p2}, i2_func);
  top.Invoke({i1_res, i2_res}, i1_func);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, top.Build());

  XLS_ASSERT_OK(InliningPass::InlineOneInvoke(i1_res.node()->As<Invoke>()));

  ASSERT_THAT(f->return_value(),
              m::Invoke(m::Add(p1.node(), p2.node()), i2_res.node()));
  EXPECT_EQ(f->return_value()->As<Invoke>()->to_apply(), i1_func);
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

// Duplicated source covers are commonized into one OR'd cover; asserts keep
// their distinct prefixed labels.
TEST_F(InliningPassTest, CoversAndAssertsDeduplicated) {
  const std::string kProgram = R"(
package some_package

fn callee(the_token: token, x: bits[32]) -> (token, bits[32]) {
  literal.10: bits[32] = literal(value=666)
  eq.20: bits[1] = eq(x, literal.10)
  ne.30: bits[1] = ne(x, literal.10)
  cover.40: () = cover(ne.30, label="cover_label")
  assert.50: token = assert(the_token, eq.20, label="assert_label", message="derp")
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

  std::vector<const Cover*> covers;
  std::vector<const Assert*> asserts;
  for (const Node* node : f->nodes()) {
    if (node->Is<Cover>()) {
      covers.push_back(node->As<Cover>());
    } else if (node->Is<Assert>()) {
      asserts.push_back(node->As<Assert>());
    }
  }

  // All clones of the single source cover collapse into one cover whose
  // condition is the OR of the two `ne(_, 666)` conditions from the callee
  // (666 is the distinctive literal in the original cover's condition).
  ASSERT_EQ(covers.size(), 1);
  EXPECT_THAT(covers.front()->condition(),
              m::Or(m::Ne(::testing::_, m::Literal(666)),
                    m::Ne(::testing::_, m::Literal(666))));
  // Asserts are unchanged: two distinct, prefixed clones.
  ASSERT_EQ(asserts.size(), 2);
  EXPECT_THAT(asserts[0]->label(), AnyOf(Eq("caller_0_callee_assert_label"),
                                         Eq("caller_1_callee_assert_label")));
  EXPECT_THAT(asserts[1]->label(), AnyOf(Eq("caller_0_callee_assert_label"),
                                         Eq("caller_1_callee_assert_label")));
}

// Unrolling a body with a cover, then inlining, must collapse to a single cover
// whose condition is the OR over all iterations ("ever reached").
TEST_F(InliningPassTest, UnrolledLoopCoverCommonized) {
  const std::string kProgram = R"(
package some_package

fn body(i: bits[2], accum: bits[8]) -> bits[8] {
  zero_ext.1: bits[8] = zero_ext(i, new_bit_count=8)
  literal.2: bits[2] = literal(value=1)
  eq.3: bits[1] = eq(i, literal.2)
  cover.4: () = cover(eq.3, label="loop_cover")
  add.5: bits[8] = add(zero_ext.1, accum)
  literal.6: bits[8] = literal(value=1)
  ret add.7: bits[8] = add(add.5, literal.6)
}

fn main() -> bits[8] {
  literal.8: bits[8] = literal(value=0)
  ret counted_for.9: bits[8] = counted_for(literal.8, trip_count=4, stride=1, body=body)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kProgram));
  PassResults results;
  OptimizationContext context;
  // Unroll, then inline the invokes and commonize the duplicated covers.
  XLS_ASSERT_OK_AND_ASSIGN(
      bool unrolled, UnrollPass().Run(package.get(), OptimizationPassOptions(),
                                      &results, context));
  ASSERT_THAT(unrolled, true);
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("main"));
  std::vector<const Cover*> covers;
  for (const Node* node : f->nodes()) {
    if (node->Is<Cover>()) {
      covers.push_back(node->As<Cover>());
    }
  }
  // Exactly one cover remains, combining all four iterations; each OR'd
  // condition is the `eq(_, 1)` from the body's cover.
  ASSERT_EQ(covers.size(), 1);
  EXPECT_THAT(covers.front()->condition(),
              m::Or(m::Eq(::testing::_, m::Literal(1)),
                    m::Eq(::testing::_, m::Literal(1)),
                    m::Eq(::testing::_, m::Literal(1)),
                    m::Eq(::testing::_, m::Literal(1))));
}

// Two different functions sharing the same cover label must NOT be merged (the
// merge keys on the source cover node, not the label string).
TEST_F(InliningPassTest, SameLabelDifferentFunctionsNotMerged) {
  const std::string kProgram = R"(
package some_package

fn a(x: bits[8]) -> bits[8] {
  literal.1: bits[8] = literal(value=100)
  ult.2: bits[1] = ult(x, literal.1)
  cover.3: () = cover(ult.2, label="X")
  ret literal.4: bits[8] = literal(value=1)
}

fn b(y: bits[8]) -> bits[8] {
  literal.5: bits[8] = literal(value=200)
  ult.6: bits[1] = ult(y, literal.5)
  cover.7: () = cover(ult.6, label="X")
  ret literal.8: bits[8] = literal(value=2)
}

fn caller(x: bits[8], y: bits[8]) -> bits[8] {
  invoke.9: bits[8] = invoke(x, to_apply=a)
  invoke.10: bits[8] = invoke(y, to_apply=b)
  ret add.11: bits[8] = add(invoke.9, invoke.10)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kProgram));
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("caller"));
  std::vector<const Cover*> covers;
  for (const Node* node : f->nodes()) {
    if (node->Is<Cover>()) {
      covers.push_back(node->As<Cover>());
    }
  }
  // Two genuinely distinct covers (from different functions) stay separate even
  // though they share the same original label "X".
  ASSERT_EQ(covers.size(), 2);
  EXPECT_THAT(covers[0]->label(), Eq("caller_0_a_X"));
  EXPECT_THAT(covers[1]->label(), Eq("caller_1_b_X"));
}

// Covers survive multi-level inlining (a -> {b, c} -> d) and still collapse to
// a single cover in the final caller d, OR-ing the conditions from every path.
TEST_F(InliningPassTest, MultiLevelCoverCommonized) {
  const std::string kProgram = R"(
package some_package

fn a(x: bits[8]) -> bits[8] {
  literal.1: bits[8] = literal(value=42)
  ult.2: bits[1] = ult(x, literal.1)
  cover.3: () = cover(ult.2, label="a_cover")
  ret literal.4: bits[8] = literal(value=1)
}

fn b(x: bits[8]) -> bits[8] {
  invoke.5: bits[8] = invoke(x, to_apply=a)
  ret literal.6: bits[8] = literal(value=2)
}

fn c(x: bits[8]) -> bits[8] {
  invoke.7: bits[8] = invoke(x, to_apply=a)
  ret literal.8: bits[8] = literal(value=3)
}

fn d(x: bits[8]) -> bits[8] {
  invoke.9: bits[8] = invoke(x, to_apply=b)
  invoke.10: bits[8] = invoke(x, to_apply=c)
  ret add.11: bits[8] = add(invoke.9, invoke.10)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kProgram));
  ASSERT_THAT(Inline(package.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("d"));
  std::vector<const Cover*> covers;
  for (const Node* node : f->nodes()) {
    if (node->Is<Cover>()) {
      covers.push_back(node->As<Cover>());
    }
  }
  // The two clones of a's cover that reach d (via b and via c) collapse into
  // one cover whose condition ORs both paths' `ult(_, 42)` (42 is a's covering
  // literal).
  ASSERT_EQ(covers.size(), 1);
  EXPECT_THAT(covers.front()->condition(),
              m::Or(m::ULt(::testing::_, m::Literal(42)),
                    m::ULt(::testing::_, m::Literal(42))));
}

void IrFuzzInlining(FuzzPackageWithArgs fuzz_package_with_args) {
  InliningPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzInlining)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
