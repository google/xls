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

#include "xls/codegen_v_1_5/state_to_register_io_lowering_pass.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

absl::StatusOr<Node*> LowerStateRead(ScheduledBlock& block,
                                     const StateRead& state_read, int64_t stage,
                                     std::vector<Register*>& state_registers) {
  XLS_RET_CHECK(block.source() != nullptr);
  const Proc& source_proc = *block.source()->AsProcOrDie();
  XLS_ASSIGN_OR_RETURN(int64_t index, source_proc.GetStateElementIndex(
                                          state_read.state_element()));
  StateElement* state_element = state_read.state_element();
  Type* type = state_read.GetType();
  if (type->IsToken() || type->GetFlatBitCount() == 0) {
    // Parameter has no meaningful data contents; replace with a literal. (We
    // know the parameter has flat bit-count 0, so any literal will have the
    // same result.)
    return block.MakeNode<xls::Literal>(state_read.loc(), ZeroOfType(type));
  }

  // Create a temporary name as this register will later be removed and
  // updated. That register should be created with the state parameter's name.
  std::string name =
      block.UniquifyNodeName(absl::StrCat("__", state_element->name()));

  XLS_ASSIGN_OR_RETURN(Register * reg,
                       block.AddRegister(name, state_read.GetType(),
                                         state_element->initial_value()));
  state_registers[index] = reg;
  return block.MakeNodeWithNameInStage<RegisterRead>(stage, state_read.loc(),
                                                     reg,
                                                     /*name=*/reg->name());
}

absl::StatusOr<Node*> LowerLastNextValue(ScheduledBlock& block,
                                         const StateElement& state_element,
                                         const Next& next_value,
                                         Register* state_register,
                                         int64_t read_stage,
                                         int64_t write_stage, Node* new_value) {
  RegisterWrite* reg_write = nullptr;
  if (state_element.type()->GetFlatBitCount() > 0) {
    // Make a placeholder RegisterWrite; the real one requires access to all
    // the `next_value` nodes and the control flow logic.
    XLS_ASSIGN_OR_RETURN(reg_write,
                         block.MakeNodeInStage<RegisterWrite>(
                             write_stage, next_value.loc(), new_value,
                             /*load_enable=*/std::nullopt,
                             /*reset=*/block.GetResetPort(), state_register));
  } else if (!state_element.type()->IsToken() &&
             state_element.type() != block.package()->GetTupleType({})) {
    return absl::UnimplementedError(absl::StrFormat(
        "Proc has zero-width state element `%s`, but type is "
        "not token or empty tuple, instead got %s.",
        state_element.name(), next_value.GetType()->ToString()));
  }

  // If the next state can be determined in a later cycle than the state read,
  // we have a non-trivial backedge between initiations (II>1); use a "full"
  // bit to track whether the state is currently valid.
  //
  // TODO(epastor): Consider an optimization that merges the "full" bits for
  // all states with the same read stage & matching write stages/predicates...
  // or maybe a more general optimization that merges registers with identical
  // type, input, and load-enable values.
  if (write_stage > read_stage) {
    XLS_ASSIGN_OR_RETURN(
        Register * reg_full,
        block.AddRegister(absl::StrCat("__", state_element.name(), "_full"),
                          block.package()->GetBitsType(1), Value(UBits(1, 1))));
    XLS_RETURN_IF_ERROR(block
                            .MakeNodeWithNameInStage<RegisterRead>(
                                write_stage, next_value.loc(), reg_full,
                                /*name=*/reg_full->name())
                            .status());
    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        block.MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    XLS_RETURN_IF_ERROR(block
                            .MakeNodeInStage<RegisterWrite>(
                                read_stage, next_value.loc(), literal_1,
                                /*load_enable=*/std::nullopt,
                                /*reset=*/block.GetResetPort(), reg_full)
                            .status());
  }

  return reg_write;
}

absl::StatusOr<bool> LowerStateIoForBlock(ScheduledBlock& block) {
  XLS_RET_CHECK(block.source() != nullptr);
  Proc& source_proc = *block.source()->AsProcOrDie();
  if (source_proc.StateElements().empty()) {
    return false;
  }

  absl::flat_hash_map<StateElement*, Next*> last_next_values;
  absl::flat_hash_map<StateElement*, int> read_stage;
  absl::flat_hash_map<StateElement*, int> write_stage;
  absl::flat_hash_map<Node*, Node*> node_map;
  std::vector<StateRead*> reads;
  std::vector<Register*> state_registers;
  state_registers.resize(source_proc.GetStateElementCount());

  for (int stage = 0; stage < block.stages().length(); ++stage) {
    for (Node* node : block.stages()[stage]) {
      if (node->Is<StateRead>()) {
        reads.push_back(node->As<StateRead>());
        read_stage[node->As<StateRead>()->state_element()] = stage;
      } else if (node->Is<Next>()) {
        StateElement* state_element =
            node->As<Next>()->state_read()->As<StateRead>()->state_element();
        last_next_values[state_element] = node->As<Next>();
        write_stage[state_element] = stage;
      }
    }
  }

  for (StateRead* read : reads) {
    XLS_ASSIGN_OR_RETURN(int64_t stage, block.GetStageIndex(read));
    XLS_ASSIGN_OR_RETURN(
        Node * register_read,
        LowerStateRead(block, *read->As<StateRead>(), stage, state_registers));
    node_map[read] = register_read;
  }

  for (const auto& [state_element, next_value] : last_next_values) {
    XLS_ASSIGN_OR_RETURN(int64_t index,
                         source_proc.GetStateElementIndex(state_element));
    Register* state_register = state_registers.at(index);
    Node* new_value = next_value->value();
    if (const auto it = node_map.find(new_value); it != node_map.end()) {
      new_value = it->second;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * register_write,
        LowerLastNextValue(block, *state_element, *next_value, state_register,
                           read_stage.at(state_element),
                           write_stage.at(state_element), new_value));
    if (register_write != nullptr) {
      node_map[next_value] = register_write;
    }
  }

  for (const auto& [old_node, new_node] : node_map) {
    XLS_RETURN_IF_ERROR(
        old_node->ReplaceUsesWith(new_node, /*replace_implicit_uses=*/false));
    if (old_node->Is<StateRead>()) {
      XLS_RETURN_IF_ERROR(block.RemoveNodeFromStage(old_node).status());
    } else {
      XLS_RETURN_IF_ERROR(block.RemoveNode(old_node));
    }
  }

  XLS_RETURN_IF_ERROR(source_proc.RemoveAllStateElements());
  return !node_map.empty();
}

}  // namespace

absl::StatusOr<bool> StateToRegisterIoLoweringPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (FunctionBase* fb : package->GetFunctionBases()) {
    if (fb->IsBlock() && fb->IsScheduled()) {
      ScheduledBlock& sb = *down_cast<ScheduledBlock*>(fb);
      if (sb.source() != nullptr && sb.source()->IsProc()) {
        XLS_ASSIGN_OR_RETURN(
            bool block_changed,
            LowerStateIoForBlock(*down_cast<ScheduledBlock*>(fb)));
        changed |= block_changed;
      }
    }
  }
  return changed;
}

}  // namespace xls::codegen
