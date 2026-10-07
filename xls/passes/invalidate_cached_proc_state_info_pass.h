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

#ifndef XLS_PASSES_INVALIDATE_CACHED_PROC_STATE_INFO_PASS_H_
#define XLS_PASSES_INVALIDATE_CACHED_PROC_STATE_INFO_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Helper fake class that invalidates cached proc-state information in the
// query engines.
//
// This invalidates the givens in PartialInfoQueryEngine and BddQueryEngine.
class InvalidateCachedProcStateInfoPass : public OptimizationProcPass {
 public:
  static constexpr std::string_view kName = "invalidate_cached_proc_state_info";
  InvalidateCachedProcStateInfoPass()
      : OptimizationProcPass(
            kName, "Invalidate proc state cached range information.") {}
  ~InvalidateCachedProcStateInfoPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnProcInternal(
      Proc* proc, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_INVALIDATE_CACHED_PROC_STATE_INFO_PASS_H_
