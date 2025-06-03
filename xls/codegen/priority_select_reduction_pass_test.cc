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

#include "xls/codegen/priority_select_reduction_pass.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/package.h"

namespace m = xls::op_matchers;

namespace xls::verilog {
namespace {
using ::absl_testing::IsOkAndHolds;

// Helper which builds a simple function containing a priority-select that the
// reduction pass can transform. Returns the newly-created package.
static absl::StatusOr<std::unique_ptr<Package>> BuildPkgWithPrioritySelect() {
  auto package = std::make_unique<Package>("test_pkg");

  FunctionBuilder fb("f", package.get());
  // Selector is constant one-hot 0b01 so the pass can prove properties.
  BValue selector = fb.Literal(xls::UBits(1, 2));
  BValue case0 = fb.Literal(xls::UBits(11, 32));
  BValue case1 = fb.Literal(xls::UBits(22, 32));
  BValue def = fb.Literal(xls::UBits(0, 32));
  BValue prio = fb.PrioritySelect(selector, {case0, case1}, def);
  XLS_RETURN_IF_ERROR(fb.BuildWithReturnValue(prio).status());
  XLS_RETURN_IF_ERROR(package->SetTopByName("f"));
  return package;
}

void RunPassAndCheck(bool enable_asserts, bool expect_assert) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           BuildPkgWithPrioritySelect());
  FunctionBase* top = package->GetTop().value();
  CodegenOptions cg_opts;  // Minimal options.

  XLS_ASSERT_OK_AND_ASSIGN(CodegenContext context,
                           FunctionBaseToCombinationalBlock(top, cg_opts));

  CodegenPassOptions pass_opts;
  pass_opts.codegen_options = cg_opts;
  pass_opts.add_invariant_assertions = enable_asserts;

  OptimizationContext opt_ctx;
  PassResults results;
  XLS_ASSERT_OK(CreateCodegenPassPipeline(opt_ctx)
                    ->Run(package.get(), pass_opts, &results, context)
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("f"));
  bool found_assert = false;
  for (Node* n : block->nodes()) {
    if (n->Is<Assert>()) {
      found_assert = true;
      break;
    }
  }
  EXPECT_EQ(found_assert, expect_assert);
}

TEST(PrioritySelectReductionPassTest, AssertsInsertedWhenEnabled) {
  RunPassAndCheck(/*enable_asserts=*/true, /*expect_assert=*/true);
}

TEST(PrioritySelectReductionPassTest, AssertsOmittedWhenDisabled) {
  RunPassAndCheck(/*enable_asserts=*/false, /*expect_assert=*/false);
}

}  // namespace
}  // namespace xls::verilog
