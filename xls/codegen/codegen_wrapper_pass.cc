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

#include "xls/codegen/codegen_wrapper_pass.h"

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"

namespace xls::verilog {

absl::StatusOr<bool> CodegenWrapperPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  XLS_ASSIGN_OR_RETURN(
      bool res,
      wrapped_pass_->Run(unit->package(), OptimizationPassOptions(options),
                         results, context_));
  if (res) {
    unit->GcMetadata();
  }
  return res;
}

}  // namespace xls::verilog
