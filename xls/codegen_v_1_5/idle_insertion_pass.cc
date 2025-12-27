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

#include "xls/codegen_v_1_5/idle_insertion_pass.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

absl::StatusOr<bool> IdleInsertionPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  if (!options.codegen_options.add_idle_output()) {
    return false;
  }

  bool changed = false;
  for (std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block = down_cast<ScheduledBlock*>(block.get());
    if (scheduled_block->stages().empty()) {
      continue;
    }

    std::vector<Node*> stage_idle_signals;
    stage_idle_signals.reserve(scheduled_block->stages().size());

    // The block is idle if *every* stage is effectively idle.
    // A stage is idle if it is:
    //   1. Empty (nothing coming down the pipeline),
    //   2. Blocked on active inputs, or
    //   3. All active outputs have resolved, but we're stalled due to internal
    //      backpressure (the next stage isn't ready).
    for (int64_t i = 0; i < scheduled_block->stages().size(); ++i) {
      const Stage& stage = scheduled_block->stages()[i];
      XLS_RET_CHECK(stage.IsControlled());

      // NOTE: The stage is idle iff `empty || blocked || stalled`.
      //       That's equivalent to `NAND(!empty, !blocked, !stalled)`.
      //       Since `!empty` and `!blocked` correspond to existing flow-control
      //       signals, this requires fewer operation nodes.

      // The stage is not empty (pipeline inputs are present) iff `inputs_valid`
      // is 1.
      Node* stage_not_empty = stage.inputs_valid();

      // The stage is not blocked on active inputs (all active inputs are
      // present or disabled) iff `active_inputs_valid` is 1.
      Node* stage_not_blocked_on_active_inputs = stage.active_inputs_valid();

      // The stage is stalled due to internal backpressure iff `outputs_valid`
      // is 1 and `outputs_ready` is 0.
      // NOTE: `!stalled` is `!(outputs_valid && !outputs_ready)`. That's
      //       equivalent to `(!outputs_valid || outputs_ready)`, which requires
      //       fewer operations.
      XLS_ASSIGN_OR_RETURN(Node * outputs_not_finished,
                           scheduled_block->MakeNode<UnOp>(
                               SourceInfo(), stage.outputs_valid(), Op::kNot));
      XLS_ASSIGN_OR_RETURN(Node * stage_not_stalled,
                           scheduled_block->MakeNode<NaryOp>(
                               SourceInfo(),
                               absl::MakeConstSpan({outputs_not_finished,
                                                    stage.outputs_ready()}),
                               Op::kOr));

      XLS_ASSIGN_OR_RETURN(
          Node * stage_idle,
          scheduled_block->MakeNodeWithName<NaryOp>(
              SourceInfo(),
              absl::MakeConstSpan({stage_not_empty,
                                   stage_not_blocked_on_active_inputs,
                                   stage_not_stalled}),
              Op::kNand, absl::StrFormat("stage_%d_idle", i)));
      stage_idle_signals.push_back(stage_idle);
    }

    XLS_ASSIGN_OR_RETURN(Node * idle_signal,
                         NaryAndIfNeeded(scheduled_block, stage_idle_signals,
                                         /*name=*/"pipeline_idle"));

    XLS_RETURN_IF_ERROR(
        scheduled_block->AddOutputPort("idle", idle_signal).status());
    changed = true;
  }

  return changed;
}

}  // namespace xls::codegen
