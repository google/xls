// Copyright 2021 The XLS Authors
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

// Create a standard pipeline of passes. This pipeline should
// be used in the main driver as well as in testing.

#include "xls/codegen/codegen_pass_pipeline.h"

#include "xls/codegen/codegen_checker.h"
#include "xls/codegen/port_legalization_pass.h"
#include "xls/codegen/signature_generation_pass.h"

namespace xls::verilog {

std::unique_ptr<CodegenCompoundPass> CreateCodegenPassPipeline() {
  auto top = absl::make_unique<CodegenCompoundPass>(
      "codegen", "Top level codegen pass pipeline");
  top->AddInvariantChecker<CodegenChecker>();

  // Generate the signature from the initial proc and options prior to any
  // transformations. If necessary the signature can be mutated later if the
  // proc is transformed in a way which affects its externally visible
  // interface.
  top->Add<SignatureGenerationPass>();

  // TODO(meheff): 2021/04/29. Add the following passes:
  // * pass to optionally generate pipeline.
  // * pass to optionally flop inputs and outputs.

  // Remove zero-width input/output ports.
  // TODO(meheff): 2021/04/29 Also flatten ports with types here.
  top->Add<PortLegalizationPass>();
  return top;
}

absl::StatusOr<bool> RunCodegenPassPipeline(const CodegenPassOptions& options,
                                            Block* block) {
  std::unique_ptr<CodegenCompoundPass> pipeline = CreateCodegenPassPipeline();
  CodegenPassUnit unit(block->package(), block);
  PassResults results;
  return pipeline->Run(&unit, options, &results);
}

}  // namespace xls::verilog
