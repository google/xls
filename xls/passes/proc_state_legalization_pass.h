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

#ifndef XLS_PASSES_STATE_LEGALIZATION_PASS_H_
#define XLS_PASSES_STATE_LEGALIZATION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which legalizes state parameters by introducing a default `next_value`
// node that fires if no other `next_value` node will fire.
//
// Attempts to pattern-match scenarios where no default is needed, but may
// produce dead `next_value` nodes if it can't recognize that one of the
// explicit `next_value` nodes is guaranteed to fire; these can be cleaned up in
// later optimization passes.
class ProcStateLegalizationPass : public OptimizationProcPass {
 public:
  static constexpr std::string_view kName = "proc_state_legal";
  ProcStateLegalizationPass()
      : OptimizationProcPass(kName, "Proc State Legalization") {}
  ~ProcStateLegalizationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnProcInternal(Proc* proc,
                                         const OptimizationPassOptions& options,
                                         PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_STATE_LEGALIZATION_PASS_H_
