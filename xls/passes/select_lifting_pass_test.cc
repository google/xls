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

#include "xls/passes/select_lifting_pass.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

class SelectLiftingPassTest : public IrTestBase {
 protected:
  SelectLiftingPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;

    // Run the select lifting pass.
    XLS_ASSIGN_OR_RETURN(bool changed,
                         SelectLiftingPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results, context));

    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results, context)
            .status());

    // Return whether select lifting changed anything.
    return changed;
  }
};

TEST_F(SelectLiftingPassTest, LiftSingleSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue a = fb.Param("array", p->GetArrayType(16, u32_type));
  BValue c = fb.Param("condition", u32_type);
  BValue i = fb.Param("first_index", u32_type);
  BValue j = fb.Param("second_index", u32_type);

  // Create the body of the IR function
  BValue condition_constant = fb.Literal(UBits(10, 32));
  BValue selector = fb.AddCompareOp(Op::kUGt, c, condition_constant);
  BValue array_index_i = fb.ArrayIndex(a, {i});
  BValue array_index_j = fb.ArrayIndex(a, {j});
  BValue select_node = fb.Select(selector, array_index_i, array_index_j);

  // Build the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  // Set the expected outputs
  VLOG(3) << "Before the transformations: " << f->DumpIr();
  EXPECT_EQ(f->node_count(), 9);
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));
  VLOG(3) << "After the transformations:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 8);
  VLOG(3) << f->DumpIr();
}

TEST_F(SelectLiftingPassTest, LiftSingleSelectNotModifiedDueToDifferentInputs) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue a = fb.Param("array", p->GetArrayType(16, u32_type));
  BValue c = fb.Param("condition", u32_type);
  BValue i = fb.Param("first_index", u32_type);
  BValue j = fb.Param("second_index", u32_type);

  // Create the body of the IR function
  BValue condition_constant = fb.Literal(UBits(10, 32));
  BValue selector = fb.AddCompareOp(Op::kUGt, c, condition_constant);
  BValue array_index_i = fb.ArrayIndex(a, {i});
  BValue select_node = fb.Select(selector, array_index_i, j);

  // Build the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  // Set the expected outputs
  VLOG(3) << "Before the transformations: " << f->DumpIr();
  EXPECT_EQ(f->node_count(), 8);
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After the transformations: " << f->DumpIr();
  EXPECT_EQ(f->node_count(), 8);
}

TEST_F(SelectLiftingPassTest,
       LiftSingleSelectNotModifiedDueToDifferentInputTypes) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);
  Type* u16_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue a = fb.Param("array", p->GetArrayType(16, u32_type));
  BValue b = fb.Param("array2", p->GetArrayType(16, u16_type));
  BValue c = fb.Param("condition", u32_type);
  BValue i = fb.Param("first_index", u32_type);
  BValue j = fb.Param("second_index", u32_type);

  // Create the body of the IR function
  BValue condition_constant = fb.Literal(UBits(10, 32));
  BValue selector = fb.AddCompareOp(Op::kUGt, c, condition_constant);
  BValue array_index_i = fb.ArrayIndex(a, {i});
  BValue array_index_j = fb.ArrayIndex(b, {j});
  BValue select_node = fb.Select(selector, array_index_i, array_index_j);

  // Build the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  // Set the expected outputs
  VLOG(3) << "Before the transformations: " << f->DumpIr();
  EXPECT_EQ(f->node_count(), 10);
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After the transformations: " << f->DumpIr();
  EXPECT_EQ(f->node_count(), 10);
}

TEST_F(SelectLiftingPassTest, LiftSingleSelectWithMultiIndicesArray) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue a =
      fb.Param("array", p->GetArrayType(16, p->GetArrayType(8, u32_type)));
  BValue c = fb.Param("condition", u32_type);
  BValue i = fb.Param("first_index", u32_type);
  BValue j = fb.Param("second_index", u32_type);

  // Create the body of the IR function
  BValue condition_constant = fb.Literal(UBits(10, 32));
  BValue selector = fb.AddCompareOp(Op::kUGt, c, condition_constant);
  BValue array_index_i = fb.ArrayIndex(a, {i, j});
  BValue array_index_j = fb.ArrayIndex(a, {i, j});
  BValue select_node = fb.Select(selector, array_index_i, array_index_j);

  // Build the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  // Set the expected outputs
  EXPECT_EQ(f->node_count(), 9);
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 9);
}

TEST_F(SelectLiftingPassTest, LiftSingleSelectWithIndicesOfDifferentBitwidth) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);
  Type* u16_type = p->GetBitsType(16);

  // Create the parameters of the IR function
  BValue a = fb.Param("array", p->GetArrayType(16, u32_type));
  BValue c = fb.Param("condition", u32_type);
  BValue i = fb.Param("first_index", u32_type);
  BValue j = fb.Param("second_index", u16_type);

  // Create the body of the IR function
  BValue condition_constant = fb.Literal(UBits(10, 32));
  BValue selector = fb.AddCompareOp(Op::kUGt, c, condition_constant);
  BValue array_index_i = fb.ArrayIndex(a, {i});
  BValue array_index_j = fb.ArrayIndex(a, {j});
  BValue select_node = fb.Select(selector, array_index_i, array_index_j);

  // Build the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  // Set the expected outputs
  EXPECT_EQ(f->node_count(), 9);
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 9);
}

TEST_F(SelectLiftingPassTest, LiftSingleSelectWithNoCases) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue a = fb.Param("array", p->GetArrayType(16, u32_type));
  BValue c = fb.Param("condition", u32_type);
  BValue i = fb.Param("first_index", u32_type);

  // Create the body of the IR function
  BValue condition_constant = fb.Literal(UBits(10, 32));
  BValue selector = fb.AddCompareOp(Op::kUGt, c, condition_constant);
  BValue array_index_i = fb.ArrayIndex(a, {i});
  std::vector<BValue> cases;
  BValue select_node = fb.Select(selector, cases, array_index_i);

  // Build the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  // Set the expected outputs
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(SelectLiftingPassTest, LiftBinaryOperationSharedLHS) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue selector = fb.Param("selector", p->GetBitsType(1));
  BValue shared = fb.Param("shared", p->GetBitsType(32));
  BValue case_a = fb.Param("case_a", p->GetBitsType(32));
  BValue case_b = fb.Param("case_b", p->GetBitsType(32));
  BValue add_a = fb.Add(shared, case_a);
  BValue add_b = fb.Add(shared, case_b);
  BValue select_node = fb.Select(selector, {add_a, add_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  auto p_expect = CreatePackage();
  FunctionBuilder fb_expect(TestName(), p_expect.get());
  BValue final_selector = fb_expect.Param("selector", p_expect->GetBitsType(1));
  BValue final_shared = fb_expect.Param("shared", p_expect->GetBitsType(32));
  BValue final_case_a = fb_expect.Param("case_a", p_expect->GetBitsType(32));
  BValue final_case_b = fb_expect.Param("case_b", p_expect->GetBitsType(32));
  BValue final_select =
      fb_expect.Select(final_selector, {final_case_a, final_case_b});
  BValue final_add = fb_expect.Add(final_shared, final_select);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f_expect,
                           fb_expect.BuildWithReturnValue(final_add));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));
  VLOG(3) << "After the transformations:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 6);
  EXPECT_TRUE(f->IsDefinitelyEqualTo(f_expect));
}

TEST_F(SelectLiftingPassTest, LiftBinaryOperationSharedRHSWithDefault) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(2));
  BValue shared = fb.Param("shared", p->GetBitsType(32));
  BValue case_a = fb.Param("case_a", p->GetBitsType(32));
  BValue case_b = fb.Param("case_b", p->GetBitsType(32));
  BValue default_case = fb.Param("default_case", p->GetBitsType(32));

  BValue smul_a = fb.SMul(case_a, shared);
  BValue smul_b = fb.SMul(case_b, shared);
  BValue smul_default = fb.SMul(default_case, shared);
  BValue select_node = fb.Select(selector, {smul_a, smul_b}, smul_default);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  auto p_expect = CreatePackage();
  FunctionBuilder fb_expect(TestName(), p_expect.get());
  BValue final_selector = fb_expect.Param("selector", p_expect->GetBitsType(2));
  BValue final_shared = fb_expect.Param("shared", p_expect->GetBitsType(32));
  BValue final_case_a = fb_expect.Param("case_a", p_expect->GetBitsType(32));
  BValue final_case_b = fb_expect.Param("case_b", p_expect->GetBitsType(32));
  BValue final_default =
      fb_expect.Param("default_case", p_expect->GetBitsType(32));
  BValue final_select = fb_expect.Select(
      final_selector, {final_case_a, final_case_b}, final_default);
  BValue final_smul = fb_expect.SMul(final_select, final_shared);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f_expect,
                           fb_expect.BuildWithReturnValue(final_smul));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));
  VLOG(3) << "After the transformations:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 7);
  EXPECT_TRUE(f->IsDefinitelyEqualTo(f_expect));
}

TEST_F(SelectLiftingPassTest, DontLiftBinaryOpsSharingInconsistentOperandSide) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(1));
  BValue shared = fb.Param("shared", p->GetBitsType(32));
  BValue case_a = fb.Param("case_a", p->GetBitsType(32));
  BValue case_b = fb.Param("case_b", p->GetBitsType(32));

  BValue add_a = fb.Add(shared, case_a);
  BValue add_b = fb.Add(case_b, shared);
  BValue select_node = fb.Select(selector, {add_a, add_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After no transformation:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 7);
}

TEST_F(SelectLiftingPassTest, DontLiftBinaryOpsUsingDifferentOperationTypes) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(1));
  BValue shared = fb.Param("shared", p->GetBitsType(32));
  BValue case_a = fb.Param("case_a", p->GetBitsType(32));
  BValue case_b = fb.Param("case_b", p->GetBitsType(32));

  BValue add_a = fb.Add(shared, case_a);
  BValue sub_b = fb.Subtract(shared, case_b);
  BValue select_node = fb.Select(selector, {add_a, sub_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After no transformation:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 7);
}

TEST_F(SelectLiftingPassTest, DontLiftBinaryOpsUsingDifferentBitWidths) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(1));
  BValue shared = fb.Param("shared", p->GetBitsType(22));
  BValue case_a = fb.Param("case_a", p->GetBitsType(24));
  BValue case_b = fb.Param("case_b", p->GetBitsType(26));
  BValue shll_a = fb.Shll(shared, case_a);
  BValue shll_b = fb.Shll(shared, case_b);
  BValue select_differ_bitwidth = fb.Select(selector, {shll_a, shll_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(select_differ_bitwidth));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After no transformation:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 7);

  p = CreatePackage();
  FunctionBuilder fb2(TestName(), p.get());

  selector = fb2.Param("selector", p->GetBitsType(1));
  shared = fb2.Param("shared", p->GetBitsType(22));
  case_a = fb2.Param("case_a", p->GetBitsType(24));
  case_b = fb2.Param("case_b", p->GetBitsType(24));
  shll_a = fb2.Shll(shared, case_a);
  shll_b = fb2.Shll(shared, case_b);
  BValue select_same_bitwidth = fb2.Select(selector, {shll_a, shll_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2,
                           fb2.BuildWithReturnValue(select_same_bitwidth));

  EXPECT_THAT(Run(f2), absl_testing::IsOkAndHolds(true));
  VLOG(3) << "After the transformation:" << f2->DumpIr();
  EXPECT_EQ(f2->node_count(), 6);
}

TEST_F(SelectLiftingPassTest, DontLiftNonBinaryOperation) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(1));
  BValue shared = fb.Param("shared", p->GetBitsType(32));
  BValue case_a = fb.Param("case_a", p->GetBitsType(32));
  BValue case_b = fb.Param("case_b", p->GetBitsType(32));
  BValue case_c = fb.Param("case_c", p->GetBitsType(32));
  BValue and_a = fb.And(shared, case_a);
  BValue and_b = fb.And({shared, case_b, case_c});
  BValue select_node = fb.Select(selector, {and_a, and_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select_node));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After no transformation:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 8);
}

TEST_F(SelectLiftingPassTest, LiftBinaryOpIfDecreasesOutputBitwidth) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(2));
  BValue shared = fb.Param("shared", p->GetBitsType(32));
  BValue case_a = fb.Param("case_a", p->GetBitsType(32));
  BValue case_b = fb.Param("case_b", p->GetBitsType(32));
  BValue case_c = fb.Param("case_c", p->GetBitsType(32));

  BValue add_a = fb.Add(shared, case_a);
  BValue add_b = fb.Add(shared, case_b);
  BValue add_c = fb.Add(shared, case_c);
  BValue select = fb.Select(selector, {add_a, add_b, add_c, add_c});
  BValue final_add = fb.Add(add_a, select);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(final_add));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));
  VLOG(3) << "After the transformation:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 9);
}

TEST_F(SelectLiftingPassTest, DontLiftBinaryOpIfIncreasesOutputBitwidth) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(1));
  BValue shared = fb.Param("shared", p->GetBitsType(32));
  BValue case_a = fb.Param("case_a", p->GetBitsType(32));
  BValue case_b = fb.Param("case_b", p->GetBitsType(32));

  BValue add_a = fb.Add(shared, case_a);
  BValue add_b = fb.Add(shared, case_b);
  BValue select = fb.Select(selector, {add_a, add_b});
  BValue later_add = fb.Add(add_a, add_b);
  BValue final_add = fb.Add(later_add, select);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(final_add));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After no transformation:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 9);
}

TEST_F(SelectLiftingPassTest, DontLiftUnaryOpInDefaultCase) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue input = fb.Param("input", p->GetBitsType(1));

  BValue and_a = fb.And({input});
  BValue and_b = fb.And({and_a, and_a});
  BValue select = fb.PrioritySelect(and_a, {and_b}, and_a);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
  VLOG(3) << "After no transformation:" << f->DumpIr();
  EXPECT_EQ(f->node_count(), 4);
}

void IrFuzzSelectLifting(FuzzPackageWithArgs fuzz_package_with_args) {
  SelectLiftingPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzSelectLifting)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace

}  // namespace xls
