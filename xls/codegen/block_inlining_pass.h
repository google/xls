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

#ifndef XLS_CODEGEN_BLOCK_INLINING_PASS_H_
#define XLS_CODEGEN_BLOCK_INLINING_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

// Inlines blocks together into one top-level container block.
class BlockInliningPass : public CodegenPass {
 public:
  BlockInliningPass()
      : CodegenPass("block_inlining",
                    "Inline a hierarchy of blocks into a single block.") {}
  ~BlockInliningPass() override = default;

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   PassResults* results) const final;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_BLOCK_INLINING_PASS_H_
