// Copyright 2020 The XLS Authors
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

#ifndef XLS_CODEGEN_STATE_REMOVAL_PASS_H_
#define XLS_CODEGEN_STATE_REMOVAL_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/passes.h"

namespace xls {

// Replaces a proc's state with a channel which communicates the recurrent state
// from the end of the proc back up to the beginning. The proc state is replaced
// with a nil type (empty tuple). This pass is part of the lowering of a proc
// for code generation.
class StateRemovalPass : public ProcPass {
 public:
  StateRemovalPass() : ProcPass("state_removal", "State removal") {}
  ~StateRemovalPass() override {}

  absl::StatusOr<bool> RunOnProc(Proc* proc, const PassOptions& options,
                                 PassResults* results) const override;

  // Name of the channel created to communicate the recurrent state.
  static constexpr char kStateChannelName[] = "proc_state";
};

}  // namespace xls

#endif  // XLS_CODEGEN_STATE_REMOVAL_PASS_H_
