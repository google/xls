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

#ifndef XLS_CODEGEN_CODEGEN_PASS_PIPELINE_H_
#define XLS_CODEGEN_CODEGEN_PASS_PIPELINE_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"

namespace xls::verilog {

// Returns the the codegen pass pipeline which runs on a package and prepares
// the IR for lowering to Verilog. After the pipeline is complete a signature is
// generated and the Block may be passed to block_generator for generating
// Verilog.
std::unique_ptr<CodegenCompoundPass> CreateCodegenPassPipeline();

// Runs the codegen pass pipeline on the given package with the given options.
absl::Status RunCodegenPassPipeline(const CodegenPassOptions& options,
                                    Package* package);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_PASS_PIPELINE_H_
