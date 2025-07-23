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

void IrFuzzSelectLifting(FuzzPackageWithArgs fuzz_package_with_args) {
  SelectLiftingPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzSelectLifting)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace

}  // namespace xls
