// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/block_conversion_checker.h"

#include "absl/status/status.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

absl::Status BlockConversionChecker::Run(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results, BlockConversionContext& context) const {
  return VerifyPackage(package, VerifyOptions{.incomplete_lowering = true});
}

}  // namespace xls::codegen
