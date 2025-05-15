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

#ifndef XLS_CODEGEN_PRIORITY_SELECT_REDUCTION_PASS_H_
#define XLS_CODEGEN_PRIORITY_SELECT_REDUCTION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

// Strength-reduces priority-select operations to selects that will convert to
// simpler Verilog, potentially with better PPA.
class PrioritySelectReductionPass : public CodegenPass {
 public:
  PrioritySelectReductionPass()
      : CodegenPass(
            "priority_select_reduction",
            "Strength-reduces priority-select operations to simpler selects.") {
  }
  ~PrioritySelectReductionPass() override = default;

  absl::StatusOr<bool> RunInternal(Package* package,
                                   const CodegenPassOptions& options,
                                   PassResults* results,
                                   CodegenContext& context) const final;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PRIORITY_SELECT_REDUCTION_PASS_H_
