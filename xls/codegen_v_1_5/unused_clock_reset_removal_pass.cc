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

#include "xls/codegen_v_1_5/unused_clock_reset_removal_pass.h"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

Block* InstantiatedBlock(::xls::Instantiation* instantiation) {
  return absl::down_cast<BlockInstantiation*>(instantiation)
      ->instantiated_block();
}

// Returns true if the Verilog generated for `block` references the block's
// clock.
bool BlockUsesClock(Block* block) {
  if (!block->GetRegisters().empty()) {
    return true;
  }
  for (::xls::Instantiation* instantiation : block->GetInstantiations()) {
    switch (instantiation->kind()) {
      case InstantiationKind::kBlock:
        if (InstantiatedBlock(instantiation)->GetClockPort().has_value()) {
          return true;
        }
        break;
      case InstantiationKind::kFifo:
      case InstantiationKind::kDelayLine:
      case InstantiationKind::kExtern:
        // FIFOs and delay lines are clocked; extern code templates may
        // reference the clock. Conservatively keep the clock.
        return true;
    }
  }
  return false;
}

// Returns true if the block contains nodes whose emission depends on the
// presence of a clock or reset: asserts, covers and traces are emitted
// differently with and without a clock and asserts are disabled during reset
// without the reset appearing as an operand; invokes (FFI calls which are not
// lowered to extern instantiations at this point) may expand to code
// templates which reference the clock. Such blocks are left alone.
bool HasEmissionSensitiveNodes(Block* block) {
  for (Node* node : block->nodes()) {
    if (node->Is<xls::Assert>() || node->Is<xls::Cover>() ||
        node->Is<xls::Trace>() || node->Is<xls::Invoke>()) {
      return true;
    }
  }
  return false;
}

// Appends `block` and the blocks it (transitively) instantiates to `order`,
// children before parents.
void PostOrderBlocks(Block* block, absl::flat_hash_set<Block*>& visited,
                     std::vector<Block*>& order) {
  if (!visited.insert(block).second) {
    return;
  }
  for (::xls::Instantiation* instantiation : block->GetInstantiations()) {
    if (instantiation->kind() == InstantiationKind::kBlock) {
      PostOrderBlocks(InstantiatedBlock(instantiation), visited, order);
    }
  }
  order.push_back(block);
}

// Removes the InstantiationInput nodes which drive the port named `port_name`
// of instantiations of `block` in other blocks.
absl::Status DisconnectInstantiationInputs(Package* package, Block* block,
                                           std::string_view port_name) {
  for (const std::unique_ptr<Block>& parent : package->blocks()) {
    std::vector<InstantiationInput*> dead_inputs;
    for (::xls::Instantiation* instantiation : parent->GetInstantiations()) {
      if (instantiation->kind() != InstantiationKind::kBlock ||
          InstantiatedBlock(instantiation) != block) {
        continue;
      }
      for (InstantiationInput* input :
           parent->GetInstantiationInputs(instantiation)) {
        if (input->port_name() == port_name) {
          dead_inputs.push_back(input);
        }
      }
    }
    for (InstantiationInput* input : dead_inputs) {
      XLS_RETURN_IF_ERROR(parent->RemoveNode(input));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> UnusedClockResetRemovalPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results, BlockConversionContext& context) const {
  // Order blocks children-first so that removing a leaf's ports can make the
  // ports of the blocks above it removable as well.
  std::vector<Block*> order;
  absl::flat_hash_set<Block*> visited;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    PostOrderBlocks(block.get(), visited, order);
  }

  std::optional<FunctionBase*> top = package->GetTop();
  bool changed = false;
  for (Block* block : order) {
    if (top.has_value() && *top == block) {
      // Keep the top block's interface stable; unused top-level signals are
      // visible to the user and can be handled externally.
      continue;
    }
    if (block->GetSignature().has_value()) {
      // A signature already exists for this block; removing ports now would
      // make it stale, so leave the block's interface alone.
      continue;
    }
    if (HasEmissionSensitiveNodes(block)) {
      continue;
    }
    if (block->GetClockPort().has_value() && !BlockUsesClock(block)) {
      XLS_RETURN_IF_ERROR(block->RemoveClockPort());
      changed = true;
    }
    std::optional<InputPort*> reset_port = block->GetResetPort();
    if (reset_port.has_value() && (*reset_port)->users().empty()) {
      // Disconnect the reset input of every instantiation of this block, then
      // remove the port itself.
      XLS_RETURN_IF_ERROR(DisconnectInstantiationInputs(
          package, block, (*reset_port)->name()));
      XLS_RETURN_IF_ERROR(block->RemoveNode(*reset_port));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xls::codegen
