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

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

// For registers with multiple RegisterWrites, combine into a single
// RegisterWrite.
absl::StatusOr<bool> ReplaceMultiwriteRegisters(Package* package,
                                                CodegenContext& context) {
  bool changed = false;
  absl::flat_hash_map<Register*, RegisterWrite*> new_reg_writes;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    for (Register* reg : block->GetRegisters()) {
      XLS_ASSIGN_OR_RETURN(absl::Span<RegisterWrite* const> reg_writes_span,
                           block->GetRegisterWrites(reg));
      XLS_RET_CHECK(!reg_writes_span.empty());
      if (reg_writes_span.size() == 1) {
        continue;
      }
      VLOG(3) << "Compbining multiple write register " << reg->name();
      std::vector<RegisterWrite*> reg_writes(reg_writes_span.begin(),
                                             reg_writes_span.end());
      std::vector<Node*> data;
      std::vector<Node*> load_enables;
      for (RegisterWrite* reg_write : reg_writes) {
        data.push_back(reg_write->data());
        if (!reg_write->load_enable().has_value()) {
          return absl::InternalError(
              absl::StrFormat("Register `%s` has multiple writes but write "
                              "`%s` is not predicated",
                              reg->name(), reg_write->GetName()));
        }
        load_enables.push_back(*reg_write->load_enable());
      }
      const SourceInfo& loc = reg_writes.front()->loc();

      XLS_ASSIGN_OR_RETURN(Node * new_load_enable,
                           block->MakeNode<NaryOp>(loc, load_enables, Op::kOr));
      XLS_ASSIGN_OR_RETURN(Node * selector,
                           block->MakeNode<xls::Concat>(loc, load_enables));

      // Reverse the order of the values, so they match up to the selector.
      std::reverse(data.begin(), data.end());
      XLS_ASSIGN_OR_RETURN(Node * new_data,
                           block->MakeNode<OneHotSelect>(loc, selector, data));
      XLS_ASSIGN_OR_RETURN(
          new_reg_writes[reg],
          block->MakeNode<RegisterWrite>(loc, new_data, new_load_enable,
                                         reg_writes.front()->reset(), reg));

      // Remove the old register_writes.
      for (RegisterWrite* reg_write : reg_writes) {
        XLS_RETURN_IF_ERROR(block->RemoveNode(reg_write));
      }

      changed = true;
    }
  }

  if (changed) {
    // Patch up the metadata. State registers may have multiple writes so
    // replace with the new single write.
    for (const std::unique_ptr<Block>& block : package->blocks()) {
      StreamingIOPipeline& streaming_io =
          context.GetMetadataForBlock(block.get()).streaming_io_and_pipeline;
      for (std::optional<StateRegister>& state_reg :
           streaming_io.state_registers) {
        if (!state_reg.has_value()) {
          continue;
        }
        if (new_reg_writes.contains(state_reg->reg)) {
          state_reg->reg_writes = {new_reg_writes.at(state_reg->reg)};
        }
        if (state_reg->reg_full.has_value() &&
            new_reg_writes.contains(state_reg->reg_full->reg)) {
          state_reg->reg_full->sets = {
              new_reg_writes.at(state_reg->reg_full->reg)};
        }
      }
    }

    context.GcMetadata();
  }
  return changed;
}

absl::StatusOr<bool> RemoveZeroWidthRegisters(Package* package,
                                              CodegenContext& context) {
  bool changed = false;

  // Build vector of (Block, Register) because removing registers invalidates
  // block->GetRegisters(). Removing the registers later requires a pointer to
  // the block that contains the register.
  std::vector<std::pair<Block*, Register*>> to_remove;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    for (Register* reg : block->GetRegisters()) {
      if (reg->type()->GetFlatBitCount() == 0) {
        to_remove.push_back(std::make_pair(block.get(), reg));
      }
    }
  }

  // Now, remove the list of registers we've built. Make a set because later we
  // clean up dangling pointers.
  absl::flat_hash_set<Register*> removed_regs;
  removed_regs.reserve(to_remove.size());
  for (auto [block, reg] : to_remove) {
    // Replace the uses of RegisterRead of a zero-width register with a
    // zero-valued literal and delete the register, RegisterRead, and
    // RegisterWrite.
    XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read, block->GetRegisterRead(reg));
    XLS_ASSIGN_OR_RETURN(RegisterWrite * reg_write,
                         block->GetUniqueRegisterWrite(reg));
    XLS_RETURN_IF_ERROR(
        reg_read->ReplaceUsesWithNew<xls::Literal>(ZeroOfType(reg->type()))
            .status());
    removed_regs.insert(reg);
    VLOG(3) << "Removing zero-width register " << reg->name();
    XLS_RETURN_IF_ERROR(block->RemoveNode(reg_read));
    XLS_RETURN_IF_ERROR(block->RemoveNode(reg_write));
    XLS_RETURN_IF_ERROR(block->RemoveRegister(reg));
    changed = true;
  }

  if (changed) {
    context.GcMetadata();
    // Pull the registers out of pipeline-register & state list if they are
    // there.
    for (auto& [block, metadata] : context.metadata()) {
      for (std::optional<StateRegister>& reg :
           metadata.streaming_io_and_pipeline.state_registers) {
        if (reg.has_value() && removed_regs.contains(reg->reg)) {
          reg.reset();
        }
      }

      for (PipelineStageRegisters& stage :
           metadata.streaming_io_and_pipeline.pipeline_registers) {
        stage.erase(std::remove_if(stage.begin(), stage.end(),
                                   [&](const PipelineRegister& p) {
                                     return removed_regs.contains(p.reg);
                                   }),
                    stage.end());
      }
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> RegisterLegalizationPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  XLS_ASSIGN_OR_RETURN(bool multi_write_changed,
                       ReplaceMultiwriteRegisters(package, context));
  XLS_ASSIGN_OR_RETURN(bool zero_width_changed,
                       RemoveZeroWidthRegisters(package, context));

  return multi_write_changed || zero_width_changed;
}

}  // namespace xls::verilog
