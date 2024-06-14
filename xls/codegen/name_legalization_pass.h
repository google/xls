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

#ifndef XLS_CODEGEN_NAME_LEGALIZATION_PASS_H_
#define XLS_CODEGEN_NAME_LEGALIZATION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"

namespace xls::verilog {

// Legalizes names in all blocks to avoid Verilog/SystemVerilog keywords.
// If a Block has a keyword for a name or port with a keyword for a name, this
// pass throws an error. Otherwise, this pass renames nodes to avoid colliding
// with keywords.
class NameLegalizationPass : public CodegenPass {
 public:
  NameLegalizationPass()
      : CodegenPass("name_legalization", "Legalize names to avoid keywords") {}
  ~NameLegalizationPass() override = default;

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   PassResults* results) const override;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_NAME_LEGALIZATION_PASS_H_
