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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/codegen/codegen_wrapper_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/solvers/z3_assert_testutils.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using solvers::z3::IsAssertClean;
using solvers::z3::ScopedVerifyBlockEquivalence;

class PrioritySelectReductionPassTest
    : public IrTestBase,
      public ::testing::WithParamInterface<bool> {};

// Found via IR fuzzer w/ RTL sim.
TEST_P(PrioritySelectReductionPassTest, ComplexPrioritySelToOhs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * f, Parser::ParseFunction(R"ir(
fn function_0(param: bits[17] id=1, param__1: (bits[41], bits[20]) id=2, param__2: bits[24] id=3, param__3: bits[16][13] id=4, param__4: (bits[62], bits[60]) id=5) -> (bits[50]) {
  dynamic_bit_slice.6: bits[12] = dynamic_bit_slice(param__2, param__2, width=12, id=6)
  not.7: bits[12] = not(dynamic_bit_slice.6, id=7)
  literal.8: bits[12] = literal(value=0, id=8)
  bit_slice.9: bits[1] = bit_slice(not.7, start=0, width=1, id=9)
  literal.10: bits[1] = literal(value=0, id=10)
  eq.11: bits[1] = eq(literal.8, not.7, id=11)
  concat.12: bits[4] = concat(bit_slice.9, eq.11, literal.10, eq.11, id=12)
  literal.13: (bits[50]) = literal(value=(718851624737306), id=13)
  literal.14: (bits[50]) = literal(value=(0), id=14)
  literal.15: (bits[50]) = literal(value=(750599937895082), id=15)
  ret priority_sel.16: (bits[50]) = priority_sel(concat.12, cases=[literal.14, literal.14, literal.13, literal.14], default=literal.15, id=16)
})ir",
                                                                   p.get()));

  const CodegenOptions cg_opts =
      CodegenOptions().add_invariant_assertions(GetParam());
  CodegenPassOptions pass_opts = {
      .codegen_options = cg_opts,
  };
  XLS_ASSERT_OK_AND_ASSIGN(auto sched, PipelineSchedule::SingleStage(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto context,
      FunctionBaseToPipelinedBlock(sched,
                                   CodegenOptions()
                                       .emit_as_pipeline(true)
                                       .module_name("foobar")
                                       .clock_name("clk")
                                       .reset("rst", false, false, false)
                                       .streaming_channel_data_suffix("_d")
                                       .streaming_channel_ready_suffix("_r")
                                       .streaming_channel_valid_suffix("_v"),
                                   f));
  ScopedRecordIr sri(p.get());
  // NB Combo block so no need to actually tick it.
  ScopedVerifyBlockEquivalence svbe(context.top_block(), /*tick_count=*/1);
  svbe.SetIgnoreAsserts();
  EXPECT_THAT(context.top_block(), IsAssertClean(/*activation_count=*/1));
  CodegenCompoundPass pass(TestName(), TestName());
  pass.Add<PrioritySelectReductionPass>();
  OptimizationContext ctx;
  pass.Add<CodegenWrapperPass>(std::make_unique<DeadCodeEliminationPass>(),
                               ctx);
  PassResults results;
  ASSERT_THAT(pass.Run(p.get(), pass_opts, &results, context),
              IsOkAndHolds(true));
  EXPECT_THAT(context.top_block(), IsAssertClean(/*activation_count=*/1));
}

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
