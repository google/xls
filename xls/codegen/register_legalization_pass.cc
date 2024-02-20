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

#include "xls/codegen/register_legalization_pass.h"

#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

absl::StatusOr<bool> RegisterLegalizationPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  Block* block = unit->block;

  std::vector<Register*> registers(block->GetRegisters().begin(),
                                   block->GetRegisters().end());
  absl::flat_hash_set<void*> removed_regs;
  for (Register* reg : registers) {
    if (reg->type()->GetFlatBitCount() == 0) {
      // Replace the uses of RegisterRead of a zero-width register with a
      // zero-valued literal and delete the register, RegisterRead, and
      // RegisterWrite.
      XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read,
                           block->GetRegisterRead(reg));
      XLS_ASSIGN_OR_RETURN(RegisterWrite * reg_write,
                           block->GetRegisterWrite(reg));
      XLS_RETURN_IF_ERROR(
          reg_read->ReplaceUsesWithNew<xls::Literal>(ZeroOfType(reg->type()))
              .status());
      removed_regs.insert(reg);
      XLS_RETURN_IF_ERROR(block->RemoveNode(reg_read));
      XLS_RETURN_IF_ERROR(block->RemoveNode(reg_write));
      XLS_RETURN_IF_ERROR(block->RemoveRegister(reg));
      changed = true;
    }
  }

  if (changed) {
    unit->GcMetadata();
    // Pull the registers out of pipeline-register & state list if they are
    // there.
    for (std::optional<StateRegister>& reg :
         unit->streaming_io_and_pipeline.state_registers) {
      if (reg && removed_regs.contains(reg->reg)) {
        reg.reset();
      }
    }
    for (auto& stage : unit->streaming_io_and_pipeline.pipeline_registers) {
      PipelineStageRegisters new_regs;
      new_regs.reserve(stage.size());
      absl::c_copy_if(stage, std::back_inserter(new_regs),
                      [&](const PipelineRegister& p) {
                        return !removed_regs.contains(p.reg);
                      });
      stage = std::move(new_regs);
    }
  }

  return changed;
}

}  // namespace xls::verilog
