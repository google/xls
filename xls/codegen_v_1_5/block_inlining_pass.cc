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

#include "xls/codegen_v_1_5/block_inlining_pass.h"

#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

absl::StatusOr<bool> BlockInliningPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  return true;
}

}  // namespace xls::codegen
