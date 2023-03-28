// Copyright 2022 The XLS Authors
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

#ifndef XLS_PASSES_PROC_STATE_OPTIMIZATION_PASS_H_
#define XLS_PASSES_PROC_STATE_OPTIMIZATION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/passes.h"

namespace xls {

// Pass which tries to minimize the size and total number of elements of the
// proc state.  The optimizations include removal of dead state elements and
// zero-width elements.
class ProcStateOptimizationPass : public ProcPass {
 public:
  ProcStateOptimizationPass()
      : ProcPass("proc_state_opt", "Proc State Optimization") {}
  ~ProcStateOptimizationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnProcInternal(Proc* proc, const PassOptions& options,
                                         PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_PROC_STATE_OPTIMIZATION_PASS_H_
