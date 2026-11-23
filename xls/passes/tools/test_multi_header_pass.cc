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

#include "xls/passes/tools/test_multi_header_pass.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/tools/test_multi_header_helper.h"

namespace xls {

int64_t TestMultiHeaderHelperValue() { return 42; }

absl::StatusOr<bool> TestMultiHeaderPass::RunInternal(
    Package* ir, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  return TestMultiHeaderHelperValue() < 0;
}

}  // namespace xls
