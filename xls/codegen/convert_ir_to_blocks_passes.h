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

#ifndef XLS_CODEGEN_CONVERT_IR_TO_BLOCKS_PASSES_H_
#define XLS_CODEGEN_CONVERT_IR_TO_BLOCKS_PASSES_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"

namespace xls::verilog {

// Converts function bases that are funcs to combinational blocks.
class ConvertFuncsToCombinationalBlocksPass : public CodegenPass {
 public:
  ConvertFuncsToCombinationalBlocksPass()
      : CodegenPass("ir_funcs_to_blocks", "Func block conversion") {}
  ~ConvertFuncsToCombinationalBlocksPass() override = default;

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   CodegenPassResults* results) const override;
};

// Converts function bases that are procs to combinational blocks.
class ConvertProcsToCombinationalBlocksPass : public CodegenPass {
 public:
  ConvertProcsToCombinationalBlocksPass()
      : CodegenPass("ir_procs_to_blocks", "Func block conversion") {}
  ~ConvertProcsToCombinationalBlocksPass() override = default;

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   CodegenPassResults* results) const override;
};

// Converts function bases that are funcs to pipelined blocks.
class ConvertFuncsToPipelinedBlocksPass : public CodegenPass {
 public:
  ConvertFuncsToPipelinedBlocksPass()
      : CodegenPass("ir_funcs_to_blocks", "Func block conversion") {}
  ~ConvertFuncsToPipelinedBlocksPass() override = default;

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   CodegenPassResults* results) const override;
};

// Converts function bases that are procs to pipelined blocks.
class ConvertProcsToPipelinedBlocksPass : public CodegenPass {
 public:
  ConvertProcsToPipelinedBlocksPass()
      : CodegenPass("ir_procs_to_blocks", "Func block conversion") {}
  ~ConvertProcsToPipelinedBlocksPass() override = default;

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   CodegenPassResults* results) const override;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CONVERT_IR_TO_BLOCKS_PASS_H_
