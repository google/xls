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

#include "xls/passes/loop_idiom_pass.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

// A zero-filled left shift of a bits[1][8] array written as a counted_for
// loop, matching the DSLX lowering of
//   for (i, result) in u32:0..u32:8 {
//     update(result, i, if i < shift { false } else { data[i - shift] })
//   }
constexpr const char* kShiftLoopProgram = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  zero_ext.3: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.4: bits[32] = sub(add.2, zero_ext.3)
  ult.5: bits[1] = ult(add.2, zero_ext.3)
  array_index.6: bits[1] = array_index(data, indices=[sub.4])
  literal.7: bits[1] = literal(value=0)
  sel.8: bits[1] = sel(ult.5, cases=[array_index.6, literal.7])
  ret array_update.9: bits[1][8] = array_update(result, sel.8, indices=[add.2])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.10: bits[1] = literal(value=0)
  array.11: bits[1][8] = array(literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10)
  ret counted_for.12: bits[1][8] = counted_for(array.11, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";

// Runs the pass on the function with the given name and returns the function.
absl::StatusOr<Function*> RunPassOn(Package* p, std::string_view fn_name) {
  XLS_ASSIGN_OR_RETURN(Function * f, p->GetFunction(fn_name));
  LoopIdiomPass pass;
  PassResults results;
  OptimizationContext context;
  XLS_RETURN_IF_ERROR(
      pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results, context)
          .status());
  return f;
}

// The rewritten function must be the packed shift shape:
//   array(bit_slice(shll(concat(data[7]..data[0]), shift), i, 1))
MATCHER_P(MatchPackedShift, shift_name, "") {
  const Node* array = arg;
  if (!array->Is<Array>()) {
    return false;
  }
  if (array->operand_count() != 8) {
    return false;
  }
  for (int64_t i = 0; i < 8; ++i) {
    Node* bit = array->operand(i);
    if (!bit->Is<BitSlice>() || bit->operand(0)->op() != Op::kShll ||
        bit->As<BitSlice>()->start() != i ||
        bit->As<BitSlice>()->width() != 1) {
      return false;
    }
  }
  Node* shll = array->operand(0)->operand(0);
  if (!shll->operand(1)->Is<Param>() ||
      shll->operand(1)->As<Param>()->GetName() != shift_name) {
    return false;
  }
  Node* packed = shll->operand(0);
  if (!packed->Is<Concat>() || packed->operand_count() != 8) {
    return false;
  }
  for (int64_t i = 0; i < 8; ++i) {
    Node* element = packed->operand(7 - i);
    if (!element->Is<ArrayIndex>() ||
        !element->As<ArrayIndex>()->indices()[0]->Is<Literal>() ||
        element->As<ArrayIndex>()
                ->indices()[0]
                ->As<Literal>()
                ->value()
                .bits()
                .ToUint64()
                .value() != i) {
      return false;
    }
  }
  return true;
}

TEST(LoopIdiomPassTest, RewritesShiftLoop) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kShiftLoopProgram));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), MatchPackedShift("shift"));
}

TEST(LoopIdiomPassTest, RewritesUgePolarityShiftLoop) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  zero_ext.3: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.4: bits[32] = sub(add.2, zero_ext.3)
  uge.5: bits[1] = uge(add.2, zero_ext.3)
  array_index.6: bits[1] = array_index(data, indices=[sub.4])
  literal.7: bits[1] = literal(value=0)
  sel.8: bits[1] = sel(uge.5, cases=[literal.7, array_index.6])
  ret array_update.9: bits[1][8] = array_update(result, sel.8, indices=[add.2])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.10: bits[1] = literal(value=0)
  array.11: bits[1][8] = array(literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10)
  ret counted_for.12: bits[1][8] = counted_for(array.11, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), MatchPackedShift("shift"));
}

TEST(LoopIdiomPassTest, RewritesDirectInductionVar) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[3], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  sub.4: bits[3] = sub(i, shift)
  ult.5: bits[1] = ult(i, shift)
  array_index.6: bits[1] = array_index(data, indices=[sub.4])
  literal.7: bits[1] = literal(value=0)
  sel.8: bits[1] = sel(ult.5, cases=[array_index.6, literal.7])
  ret array_update.9: bits[1][8] = array_update(result, sel.8, indices=[i])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.10: bits[1] = literal(value=0)
  array.11: bits[1][8] = array(literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10)
  ret counted_for.12: bits[1][8] = counted_for(array.11, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), MatchPackedShift("shift"));
}

// A matching loop after a non-matching loop must still be rewritten: the pass
// scans past candidates that do not match instead of stopping.
TEST(LoopIdiomPassTest, RewritesShiftLoopAfterNonMatchingLoop) {
  const std::string program = R"(
package loop_idiom_test

fn non_shift_body(i: bits[32], result: bits[1][8]) -> bits[1][8] {
  literal.1: bits[1] = literal(value=1)
  ret array_update.2: bits[1][8] = array_update(result, literal.1, indices=[i])
}

fn shift_body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.3: bits[32] = literal(value=0)
  add.4: bits[32] = add(i, literal.3)
  zero_ext.5: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.6: bits[32] = sub(add.4, zero_ext.5)
  ult.7: bits[1] = ult(add.4, zero_ext.5)
  array_index.8: bits[1] = array_index(data, indices=[sub.6])
  literal.9: bits[1] = literal(value=0)
  sel.10: bits[1] = sel(ult.7, cases=[array_index.8, literal.9])
  ret array_update.11: bits[1][8] = array_update(result, sel.10, indices=[add.4])
}

fn mixed(data: bits[1][8], shift: bits[3]) -> (bits[1][8], bits[1][8]) {
  literal.12: bits[1] = literal(value=0)
  array.13: bits[1][8] = array(literal.12, literal.12, literal.12, literal.12, literal.12, literal.12, literal.12, literal.12)
  array.14: bits[1][8] = array(literal.12, literal.12, literal.12, literal.12, literal.12, literal.12, literal.12, literal.12)
  counted_for.15: bits[1][8] = counted_for(array.13, trip_count=8, stride=1, body=non_shift_body, invariant_args=[])
  counted_for.16: bits[1][8] = counted_for(array.14, trip_count=8, stride=1, body=shift_body, invariant_args=[data, shift])
  ret tuple.17: (bits[1][8], bits[1][8]) = tuple(counted_for.15, counted_for.16)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "mixed"));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::CountedFor(), MatchPackedShift("shift")));
}

// The pass must not rewrite the loop if any part of the shape differs.
TEST(LoopIdiomPassTest, DoesNotRewriteNonShiftLoops) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  zero_ext.3: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.4: bits[32] = sub(add.2, zero_ext.3)
  ult.5: bits[1] = ult(add.2, zero_ext.3)
  array_index.6: bits[1] = array_index(data, indices=[sub.4])
  literal.7: bits[1] = literal(value=1)
  sel.8: bits[1] = sel(ult.5, cases=[array_index.6, literal.7])
  ret array_update.9: bits[1][8] = array_update(result, sel.8, indices=[add.2])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.10: bits[1] = literal(value=0)
  array.11: bits[1][8] = array(literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10)
  ret counted_for.12: bits[1][8] = counted_for(array.11, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), m::CountedFor());
}

TEST(LoopIdiomPassTest, DoesNotRewriteNonUnitStride) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  zero_ext.3: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.4: bits[32] = sub(add.2, zero_ext.3)
  ult.5: bits[1] = ult(add.2, zero_ext.3)
  array_index.6: bits[1] = array_index(data, indices=[sub.4])
  literal.7: bits[1] = literal(value=0)
  sel.8: bits[1] = sel(ult.5, cases=[array_index.6, literal.7])
  ret array_update.9: bits[1][8] = array_update(result, sel.8, indices=[add.2])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.10: bits[1] = literal(value=0)
  array.11: bits[1][8] = array(literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10)
  ret counted_for.12: bits[1][8] = counted_for(array.11, trip_count=8, stride=2, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), m::CountedFor());
}

TEST(LoopIdiomPassTest, DoesNotRewriteNonBitsElements) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[2][8], data: bits[2][8], shift: bits[3]) -> bits[2][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  zero_ext.3: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.4: bits[32] = sub(add.2, zero_ext.3)
  ult.5: bits[1] = ult(add.2, zero_ext.3)
  array_index.6: bits[2] = array_index(data, indices=[sub.4])
  literal.7: bits[2] = literal(value=0)
  sel.8: bits[2] = sel(ult.5, cases=[array_index.6, literal.7])
  ret array_update.9: bits[2][8] = array_update(result, sel.8, indices=[add.2])
}

fn shift_loop(data: bits[2][8], shift: bits[3]) -> bits[2][8] {
  literal.10: bits[2] = literal(value=0)
  array.11: bits[2][8] = array(literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10)
  ret counted_for.12: bits[2][8] = counted_for(array.11, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), m::CountedFor());
}

TEST(LoopIdiomPassTest, DoesNotRewriteSignExtendedShift) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  sign_ext.3: bits[32] = sign_ext(shift, new_bit_count=32)
  sub.4: bits[32] = sub(add.2, sign_ext.3)
  ult.5: bits[1] = ult(add.2, sign_ext.3)
  array_index.6: bits[1] = array_index(data, indices=[sub.4])
  literal.7: bits[1] = literal(value=0)
  sel.8: bits[1] = sel(ult.5, cases=[array_index.6, literal.7])
  ret array_update.9: bits[1][8] = array_update(result, sel.8, indices=[add.2])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.10: bits[1] = literal(value=0)
  array.11: bits[1][8] = array(literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10, literal.10)
  ret counted_for.12: bits[1][8] = counted_for(array.11, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), m::CountedFor());
}

TEST(LoopIdiomPassTest, DoesNotRewriteMismatchedShiftParams) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift_a: bits[3], shift_b: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  zero_ext.3: bits[32] = zero_ext(shift_a, new_bit_count=32)
  zero_ext.4: bits[32] = zero_ext(shift_b, new_bit_count=32)
  sub.5: bits[32] = sub(add.2, zero_ext.3)
  ult.6: bits[1] = ult(add.2, zero_ext.4)
  array_index.7: bits[1] = array_index(data, indices=[sub.5])
  literal.8: bits[1] = literal(value=0)
  sel.9: bits[1] = sel(ult.6, cases=[array_index.7, literal.8])
  ret array_update.10: bits[1][8] = array_update(result, sel.9, indices=[add.2])
}

fn shift_loop(data: bits[1][8], shift_a: bits[3], shift_b: bits[3]) -> bits[1][8] {
  literal.11: bits[1] = literal(value=0)
  array.12: bits[1][8] = array(literal.11, literal.11, literal.11, literal.11, literal.11, literal.11, literal.11, literal.11)
  ret counted_for.13: bits[1][8] = counted_for(array.12, trip_count=8, stride=1, body=body, invariant_args=[data, shift_a, shift_b])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), m::CountedFor());
}

TEST(LoopIdiomPassTest, DoesNotRewriteBodyWithSideEffects) {
  const std::string program = R"(
package loop_idiom_test

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  zero_ext.3: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.4: bits[32] = sub(add.2, zero_ext.3)
  ult.5: bits[1] = ult(add.2, zero_ext.3)
  array_index.6: bits[1] = array_index(data, indices=[sub.4])
  literal.7: bits[1] = literal(value=0)
  sel.8: bits[1] = sel(ult.5, cases=[array_index.6, literal.7])
  literal.10: token = literal(value=token)
  literal.11: bits[1] = literal(value=1)
  assert.12: token = assert(literal.10, literal.11, message="msg")
  ret array_update.9: bits[1][8] = array_update(result, sel.8, indices=[add.2])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.13: bits[1] = literal(value=0)
  array.14: bits[1][8] = array(literal.13, literal.13, literal.13, literal.13, literal.13, literal.13, literal.13, literal.13)
  ret counted_for.15: bits[1][8] = counted_for(array.14, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), m::CountedFor());
}

TEST(LoopIdiomPassTest, DoesNotRewriteBodyWithInvoke) {
  const std::string program = R"(
package loop_idiom_test

fn helper(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}

fn body(i: bits[32], result: bits[1][8], data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.1: bits[32] = literal(value=0)
  add.2: bits[32] = add(i, literal.1)
  invoke.3: bits[32] = invoke(add.2, to_apply=helper)
  zero_ext.4: bits[32] = zero_ext(shift, new_bit_count=32)
  sub.5: bits[32] = sub(invoke.3, zero_ext.4)
  ult.6: bits[1] = ult(invoke.3, zero_ext.4)
  array_index.7: bits[1] = array_index(data, indices=[sub.5])
  literal.8: bits[1] = literal(value=0)
  sel.9: bits[1] = sel(ult.6, cases=[array_index.7, literal.8])
  ret array_update.10: bits[1][8] = array_update(result, sel.9, indices=[invoke.3])
}

fn shift_loop(data: bits[1][8], shift: bits[3]) -> bits[1][8] {
  literal.11: bits[1] = literal(value=0)
  array.12: bits[1][8] = array(literal.11, literal.11, literal.11, literal.11, literal.11, literal.11, literal.11, literal.11)
  ret counted_for.13: bits[1][8] = counted_for(array.12, trip_count=8, stride=1, body=body, invariant_args=[data, shift])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, RunPassOn(p.get(), "shift_loop"));
  EXPECT_THAT(f->return_value(), m::CountedFor());
}

void IrFuzzLoopIdiom(FuzzPackageWithArgs fuzz_package_with_args) {
  LoopIdiomPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzLoopIdiom)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
