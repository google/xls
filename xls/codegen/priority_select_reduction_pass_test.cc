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


#include <memory>

#include "gtest/gtest.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

class PrioritySelectReductionPassTest : public ::testing::TestWithParam<bool> {
};

TEST_P(PrioritySelectReductionPassTest, AssertEmissionMatchesFlag) {
  const bool enable_asserts = GetParam();

  // Build a minimal package containing a priority_select operation
  // that the reduction pass can transform.
  auto package = std::make_unique<Package>("test_pkg");
  FunctionBuilder fb("f", package.get());
  BValue selector = fb.Literal(xls::UBits(1, 2));  // one-hot 0b01
  BValue case0 = fb.Literal(xls::UBits(11, 32));
  BValue case1 = fb.Literal(xls::UBits(22, 32));
  BValue def = fb.Literal(xls::UBits(0, 32));
  BValue prio = fb.PrioritySelect(selector, {case0, case1}, def);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(prio).status());
  XLS_ASSERT_OK(package->SetTopByName("f"));

  FunctionBase* top = package->GetTop().value();

  // Run the code-gen pipeline with/without invariant assertions enabled.
  const CodegenOptions cg_opts =
      CodegenOptions().add_invariant_assertions(enable_asserts);
  XLS_ASSERT_OK_AND_ASSIGN(CodegenContext context,
                           FunctionBaseToCombinationalBlock(top, cg_opts));

  const CodegenPassOptions pass_opts = {
      .codegen_options = cg_opts,
  };

  OptimizationContext opt_ctx;
  PassResults results;
  XLS_ASSERT_OK(CreateCodegenPassPipeline(opt_ctx)
                    ->Run(package.get(), pass_opts, &results, context)
                    .status());

  // Assert: Presence/absence of Assert nodes matches expectation.
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("f"));
  bool found_assert = false;
  for (Node* n : block->nodes()) {
    if (n->Is<Assert>()) {
      found_assert = true;
      break;
    }
  }
  EXPECT_EQ(found_assert, enable_asserts);
}

INSTANTIATE_TEST_SUITE_P(AllConfigurations, PrioritySelectReductionPassTest,
                         ::testing::Bool());

}  // namespace
}  // namespace xls::verilog
