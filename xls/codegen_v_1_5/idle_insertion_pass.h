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

#ifndef XLS_CODEGEN_V_1_5_IDLE_INSERTION_PASS_H_
#define XLS_CODEGEN_V_1_5_IDLE_INSERTION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

// Adds an idle signal output to each scheduled block. The goal is that if the
// block is signalling idle, then it is safe to clock-gate the block until an
// external signal changes (including incoming data or incoming "ready" signals
// on outgoing channels). Note that this does not guarantee that power-gating
// the block would be safe, since the pipeline registers may not be empty.
//
// To meet this, we define that a block is idle if *every* stage is effectively
// idle. A stage is idle if it is:
//   1. Empty (nothing coming down the pipeline),
//   2. Blocked on active inputs, or
//   3. All active outputs have resolved, but we're stalled due to internal
//      backpressure (the next stage isn't ready).
//
// Additionally, if the `flop_inputs`/`flop_outputs` codegen options are
// enabled, then the block is not idle if any I/O flops are actively resolving a
// transaction (i.e., both `valid` and `ready` are high).
//
// WARNING: This pass must be run after I/O port lowering is finished for both
//          function and proc I/O.
class IdleInsertionPass : public BlockConversionPass {
 public:
  IdleInsertionPass()
      : BlockConversionPass("idle_insertion", "Idle insertion pass") {}

 protected:
  absl::StatusOr<bool> RunInternal(Package* package,
                                   const BlockConversionPassOptions& options,
                                   PassResults* results) const override;
};

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_IDLE_INSERTION_PASS_H_
