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

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
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

// Replaces `old_node` with `new_node` in the given block. Returns whether
// `old_node` was removed after the replacement.
absl::StatusOr<bool> ReplaceNode(Block* block, Node* old_node, Node* new_node,
                                 bool replace_implicit_uses = true) {
  XLS_RETURN_IF_ERROR(old_node->ReplaceUsesWith(
      new_node, [&](Node* n) { return n != new_node; }, replace_implicit_uses));

  if (old_node->Is<StateRead>()) {
    XLS_RETURN_IF_ERROR(block->RemoveNodeFromStage(old_node).status());
    return false;
  }
  if (old_node->IsDead()) {
    XLS_RETURN_IF_ERROR(block->RemoveNode(old_node));
    return true;
  }
  return false;
}

absl::StatusOr<Node*> NodeOrOne(FunctionBase* fb, std::optional<Node*> node) {
  if (node.has_value()) {
    return *node;
  }
  return fb->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1)));
}

absl::StatusOr<Register*> CreateFullRegisterAndRead(
    Block* block, const StateElement& state_element,
    std::optional<Node*> read_predicate, const Next& last_write,
    int read_stage_index, Stage& read_stage) {
  XLS_ASSIGN_OR_RETURN(
      Register * reg_full,
      block->AddRegister(absl::StrCat("__", state_element.name(), "_full"),
                         block->package()->GetBitsType(1), Value(UBits(1, 1))));

  XLS_ASSIGN_OR_RETURN(RegisterRead * reg_full_read,
                       block->MakeNodeWithNameInStage<RegisterRead>(
                           read_stage_index, last_write.loc(), reg_full,
                           /*name=*/reg_full->name()));

  Node* state_ready_or_not_needed = reg_full_read;
  if (read_predicate.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        Node * not_predicate,
        block->MakeNodeInStage<UnOp>(read_stage_index, SourceInfo{},
                                     *read_predicate, Op::kNot));

    XLS_ASSIGN_OR_RETURN(
        state_ready_or_not_needed,
        block->MakeNodeInStage<NaryOp>(
            read_stage_index, SourceInfo(),
            std::vector<Node*>{reg_full_read, not_predicate}, Op::kOr));
  }

  XLS_ASSIGN_OR_RETURN(
      Node * new_active_inputs_valid,
      NaryAndIfNeeded(block,
                      absl::MakeConstSpan({read_stage.active_inputs_valid(),
                                           state_ready_or_not_needed}),
                      /*name=*/"", SourceInfo{},
                      /*drop_literal_one_operands=*/true));
  XLS_RETURN_IF_ERROR(ReplaceNode(block, read_stage.active_inputs_valid(),
                                  new_active_inputs_valid)
                          .status());
  // Replacement is prone to dropping the read from the stage.
  XLS_RETURN_IF_ERROR(
      block->AddNodeToStage(read_stage_index, reg_full_read).status());
  return reg_full;
}

absl::Status AddFullRegisterWrite(Block* block, Register* reg_full,
                                  std::optional<Node*> read_predicate,
                                  const Stage& read_stage,
                                  const SourceInfo& last_write_loc,
                                  std::optional<Node*> write_load_enable) {
  std::vector<Node*> full_from_read_operands{read_stage.outputs_valid(),
                                             read_stage.outputs_ready()};
  if (read_predicate.has_value()) {
    full_from_read_operands.push_back(*read_predicate);
  }

  XLS_ASSIGN_OR_RETURN(Node * full_from_read,
                       NaryAndIfNeeded(block, full_from_read_operands));

  XLS_ASSIGN_OR_RETURN(Node * write_load_enable_or_1,
                       NodeOrOne(block, write_load_enable));
  XLS_ASSIGN_OR_RETURN(
      Node * full_load_enable,
      NaryOrIfNeeded(block, absl::MakeConstSpan(
                                {full_from_read, write_load_enable_or_1})));

  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<RegisterWrite>(last_write_loc, write_load_enable_or_1,
                                    full_load_enable,
                                    /*reset=*/block->GetResetPort(), reg_full)
          .status());
  return absl::OkStatus();
}

absl::Status LowerStateElement(ScheduledBlock* block,
                               const StateElement& state_element,
                               StateRead* read, absl::Span<Next*> writes) {
  // A token or empty tuple state element has no real functionality.
  Type* type = state_element.type();
  const bool has_data = type->GetFlatBitCount() > 0;

  if (!has_data && !state_element.type()->IsToken() &&
      state_element.type() != block->package()->GetTupleType({})) {
    return absl::UnimplementedError(
        absl::StrFormat("Proc has zero-width state element `%s`, but type is "
                        "not token or empty tuple, instead got %s.",
                        state_element.name(), type->ToString()));
  }

  // Lower the read of the state element.
  std::optional<Node*> read_predicate = read->predicate();
  std::string name =
      block->UniquifyNodeName(absl::StrCat("__", state_element.name()));

  XLS_ASSIGN_OR_RETURN(int read_stage_index, block->GetStageIndex(read));
  Stage& read_stage = block->stages()[read_stage_index];
  Node* reg_read_or_zero = nullptr;
  Register* state_register = nullptr;
  if (has_data) {
    XLS_ASSIGN_OR_RETURN(state_register,
                         block->AddRegister(name, state_element.type(),
                                            state_element.initial_value()));
    XLS_ASSIGN_OR_RETURN(reg_read_or_zero,
                         block->MakeNodeWithNameInStage<RegisterRead>(
                             read_stage_index, read->loc(), state_register,
                             /*name=*/state_register->name()));
  } else {
    XLS_ASSIGN_OR_RETURN(reg_read_or_zero, block->MakeNode<xls::Literal>(
                                               read->loc(), ZeroOfType(type)));
  }

  XLS_ASSIGN_OR_RETURN(bool read_removed,
                       ReplaceNode(block, read, reg_read_or_zero,
                                   /*replace_implicit_uses=*/false));
  XLS_RET_CHECK(!read_removed);

  Next* last_write = nullptr;
  Node* last_value = nullptr;
  SourceInfo last_write_loc;
  SourceInfo last_value_loc;
  int last_write_stage_index = -1;
  if (!writes.empty()) {
    last_write = writes[writes.size() - 1];
    last_write_loc = last_write->loc();
    last_value = last_write->value();
    last_value_loc = last_value->loc();
    XLS_ASSIGN_OR_RETURN(last_write_stage_index,
                         block->GetStageIndex(last_write));
  }

  // If the next state can be determined in a later cycle than the state read,
  // we have a non-trivial backedge between initiations (II>1); use a "full" bit
  // to track whether the state is currently valid.
  Register* reg_full = nullptr;
  if (last_write_stage_index > read_stage_index) {
    XLS_ASSIGN_OR_RETURN(
        reg_full,
        CreateFullRegisterAndRead(block, state_element, read_predicate,
                                  *last_write, read_stage_index, read_stage));
  }

  std::vector<Node*> gates;
  std::vector<Node*> write_conditions;
  std::vector<Node*> values;
  gates.reserve(writes.size());
  write_conditions.reserve(writes.size());
  values.reserve(writes.size());
  for (Next* write : writes) {
    XLS_ASSIGN_OR_RETURN(int stage_index, block->GetStageIndex(write));
    Stage& stage = block->stages()[stage_index];

    Node* value = write->value();
    std::optional<Node*> predicate = write->predicate();

    // If needed, add identity nodes to signal that the value and predicate both
    // need to be available at the write's stage. (This enables pipeline
    // register insertion later.)
    if (block->IsStaged(value) && *block->GetStageIndex(value) != stage_index) {
      XLS_ASSIGN_OR_RETURN(
          value, block->MakeNodeInStage<UnOp>(stage_index, write->loc(), value,
                                              Op::kIdentity));
    }
    if (predicate.has_value() && block->IsStaged(*predicate) &&
        *block->GetStageIndex(*predicate) != stage_index) {
      XLS_ASSIGN_OR_RETURN(
          predicate, block->MakeNodeInStage<UnOp>(stage_index, write->loc(),
                                                  *predicate, Op::kIdentity));
    }

    std::vector<Node*> gate_operands{stage.inputs_valid(),
                                     stage.active_inputs_valid()};
    std::vector<Node*> condition_operands{stage.outputs_valid(),
                                          stage.outputs_ready()};
    if (predicate.has_value()) {
      gate_operands.push_back(*predicate);
      condition_operands.push_back(*predicate);
    }
    XLS_ASSIGN_OR_RETURN(Node * gate, NaryAndIfNeeded(block, gate_operands));
    XLS_ASSIGN_OR_RETURN(Node * condition,
                         NaryAndIfNeeded(block, condition_operands));
    gates.push_back(gate);
    write_conditions.push_back(condition);
    values.push_back(value);
  }

  Node* value = nullptr;
  std::optional<Node*> write_load_enable;
  if (values.empty()) {
    // If we never change the state element, write the current value
    // unconditionally.
    value = reg_read_or_zero;
  } else if (values.size() == 1) {
    value = values[0];
    write_load_enable = write_conditions[0];
  } else {
    XLS_ASSIGN_OR_RETURN(
        write_load_enable,
        block->MakeNode<NaryOp>(last_write->loc(), write_conditions, Op::kOr));

    if (has_data) {
      XLS_ASSIGN_OR_RETURN(Node * selector, block->MakeNode<xls::Concat>(
                                                last_write->loc(), gates));

      // Reverse the order of the values, so they match up to the selector.
      std::reverse(values.begin(), values.end());
      XLS_ASSIGN_OR_RETURN(value, block->MakeNode<OneHotSelect>(
                                      last_write->loc(), selector, values));
    }
  }

  // Create the register write, if it's not a token or empty bits.
  if (has_data) {
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                last_value_loc, value,
                                /*load_enable=*/write_load_enable,
                                /*reset=*/block->GetResetPort(), state_register)
                            .status());
  }

  // Replace the Next nodes with empty tuples.
  XLS_ASSIGN_OR_RETURN(
      Node * empty_tuple,
      block->MakeNode<Tuple>(SourceInfo{}, absl::Span<Node*>{}));
  for (Next* write : writes) {
    XLS_ASSIGN_OR_RETURN(bool removed, ReplaceNode(block, write, empty_tuple));
    XLS_RET_CHECK(removed);
  }

  // Write the full bit if needed.
  if (reg_full != nullptr) {
    XLS_RETURN_IF_ERROR(AddFullRegisterWrite(block, reg_full, read_predicate,
                                             read_stage, last_write_loc,
                                             write_load_enable));
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> LowerStateIoForBlock(ScheduledBlock* block) {
  XLS_RET_CHECK(block->source() != nullptr);
  Proc* source_proc = block->source()->AsProcOrDie();
  if (source_proc->StateElements().empty()) {
    return false;
  }

  absl::flat_hash_map<StateElement*, std::vector<StateRead*>> state_reads;
  absl::flat_hash_map<StateElement*, std::vector<Next*>> next_values;

  for (int stage = 0; stage < block->stages().length(); ++stage) {
    for (Node* node : block->stages()[stage]) {
      if (node->Is<StateRead>()) {
        StateRead* state_read = node->As<StateRead>();
        state_reads[state_read->state_element()].push_back(state_read);
      } else if (node->Is<Next>()) {
        Next* next = node->As<Next>();
        next_values[next->state_read()->As<StateRead>()->state_element()]
            .push_back(next);
      }
    }
  }

  for (StateElement* state_element : source_proc->StateElements()) {
    std::vector<StateRead*> reads_for_element = state_reads.at(state_element);
    // Multiple reads are not yet supported.
    XLS_RET_CHECK_EQ(reads_for_element.size(), 1)
        << "Found multiple reads for state element: " << state_element->name();
    XLS_RETURN_IF_ERROR(
        LowerStateElement(block, *state_element, reads_for_element[0],
                          absl::MakeSpan(next_values[state_element])));
  }

  XLS_RETURN_IF_ERROR(source_proc->RemoveAllStateElements());
  return true;
}

}  // namespace

absl::StatusOr<bool> StateToRegisterIoLoweringPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (FunctionBase* fb : package->GetFunctionBases()) {
    if (fb->IsBlock() && fb->IsScheduled()) {
      ScheduledBlock* sb = down_cast<ScheduledBlock*>(fb);
      if (sb->source() != nullptr && sb->source()->IsProc()) {
        XLS_ASSIGN_OR_RETURN(
            bool block_changed,
            LowerStateIoForBlock(down_cast<ScheduledBlock*>(fb)));
        changed |= block_changed;
      }
    }
  }
  return changed;
}

}  // namespace xls::codegen
