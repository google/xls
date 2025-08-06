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

#include "xls/codegen/passes_ng/block_pipeline_inserter_pass.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/passes_ng/passes_ng_test_fixtures.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_pass.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAreArray;

// Test the conversion to stage blocks by creating a multi-stage pipeline,
// sensitizing the block and testing the I/O behavior.
TEST_P(SweepTrivialPipelinedFunctionFixture, TestPipelineCreation) {
  XLS_ASSERT_OK(CreateStageProcInPackage());

  XLS_ASSERT_OK_AND_ASSIGN(ProcMetadata * top_metadata,
                           stage_conversion_metadata_.GetTopProcMetadata(
                               package_->GetTop().value()));

  PassResults results;
  CodegenContext context;
  context.stage_conversion_metadata() = std::move(stage_conversion_metadata_);
  context.block_conversion_metadata() = std::move(block_conversion_metadata_);

  CodegenPassOptions options;
  options.codegen_options = codegen_options();

  StageToBlockConversionPass pass;
  XLS_ASSERT_OK(pass.Run(package_.get(), options, &results, context));

  BlockPipelineInserterPass pipeline_pass;
  XLS_ASSERT_OK(pipeline_pass.Run(package_.get(), options, &results, context));

  XLS_VLOG_LINES(2, package_->DumpIr());

  // Simulate the pipeline
  // out = 2*x + y
  std::vector<uint64_t> x = {0x1, 0x10, 0x30};
  std::vector<uint64_t> y = {0x2, 0x20, 0x30};

  std::vector<uint64_t> out_expected(x.size());
  for (int64_t i = 0; i < out_expected.size(); ++i) {
    out_expected[i] = x[i] * 2 + y[i];
  }

  EXPECT_THAT(SimulateBlock(top_metadata->proc()->name(), absl::MakeSpan(x),
                            absl::MakeSpan(y), /*cycle_count=*/100),
              IsOkAndHolds(ElementsAreArray(out_expected)));
}

INSTANTIATE_TEST_SUITE_P(
    TestProcHierarchyCreationAndSimulation,
    SweepTrivialPipelinedFunctionFixture, testing::Values(1, 2, 3, 4, 5),
    SweepTrivialPipelinedFunctionFixture::PrintToStringParamName);

}  // namespace
}  // namespace xls::verilog
