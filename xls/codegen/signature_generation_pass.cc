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

#include "xls/codegen/signature_generation_pass.h"

#include "absl/strings/str_format.h"
#include "xls/codegen/signature_generator.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_util.h"

namespace xls::verilog {

absl::StatusOr<bool> SignatureGenerationPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  if (unit->signature.has_value()) {
    return absl::InvalidArgumentError("Signature already generated.");
  }
  XLS_ASSIGN_OR_RETURN(unit->signature,
                       GenerateSignature(options.codegen_options, unit->block,
                                         options.schedule));
  return true;
}

}  // namespace xls::verilog
