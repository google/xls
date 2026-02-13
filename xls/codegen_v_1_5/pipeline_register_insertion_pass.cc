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

#include "xls/codegen_v_1_5/pipeline_register_insertion_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/enumerate.hpp"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"
#include "re2/re2.h"

namespace xls::codegen {

namespace {

// Returns true if tuple_type has a zero width element at the top level.
bool HasZeroWidthType(TupleType* tuple_type) {
  CHECK(tuple_type != nullptr);

  for (Type* element_type : tuple_type->element_types()) {
    if (element_type->GetFlatBitCount() == 0) {
      return true;
    }
  }

  return false;
}

std::string PipelineSignalName(std::string_view root, int64_t stage) {
  std::string base;
  // Strip any existing pipeline prefix from the name.
  static constexpr LazyRE2 kPipelinePrefix = {.pattern_ = R"(^p\d+_(.+))"};
  if (!RE2::PartialMatch(root, *kPipelinePrefix, &base)) {
    base = root;
  }
  return absl::StrFormat("p%d_%s", stage, base);
}

absl::StatusOr<RegisterRead*> CreatePipelineRegister(
    std::string_view name, Node* node, int64_t stage_index, Node* stage_done,
    ScheduledBlock* block, const BlockConversionPassOptions& options) {
  std::optional<Value> reset_value;
  std::optional<Node*> reset_signal;
  if (block->GetResetPort().has_value() &&
      options.codegen_options.reset().has_value() &&
      options.codegen_options.reset()->reset_data_path()) {
    reset_value = ZeroOfType(node->GetType());
    reset_signal = block->GetResetPort();
  }

  XLS_ASSIGN_OR_RETURN(Register * reg,
                       block->AddRegister(name, node->GetType(), reset_value));

  Node* load_enable = stage_done;
  if (block->GetResetPort().has_value() &&
      options.codegen_options.reset().has_value() &&
      !options.codegen_options.reset()->reset_data_path()) {
    // Note that if data registers are not reset, data path registers are
    // transparent during reset.
    Node* reset_active = *block->GetResetPort();
    if (options.codegen_options.reset()->active_low()) {
      XLS_ASSIGN_OR_RETURN(reset_active, block->MakeNodeWithName<UnOp>(
                                             SourceInfo(), reset_active,
                                             Op::kNot, "reset_active"));
    }
    XLS_ASSIGN_OR_RETURN(
        load_enable,
        block->MakeNode<NaryOp>(node->loc(),
                                absl::MakeConstSpan({stage_done, reset_active}),
                                Op::kOr));
  }
  // NOTE: The RegisterWrite is added to the block, but not to the stage. Its
  //       `load_enable` depends on `outputs_ready`, which comes from outside
  //       the stage (and in practice, often from the next stage).
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNodeWithName<RegisterWrite>(
              node->loc(), /*data=*/node, load_enable,
              /*reset=*/reset_signal, reg,
              block->UniquifyNodeName(absl::StrCat(name, "_write")))
          .status());
  XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read,
                       block->MakeNodeWithNameInStage<RegisterRead>(
                           stage_index + 1, node->loc(), reg,
                           /*name=*/reg->name()));
  return reg_read;
}

// Adds a pipeline register for `node` exiting `stage` (with index
// `stage_index`) of `block`. (As an optimization, may create multiple
// registers if splitting the node is believed beneficial.)
//
// Returns the (unstaged) node reporting the pipeline register's value.
// (If only one pipeline register is used, this will be the RegisterRead.)
absl::StatusOr<Node*> AddPipelineRegisterFor(
    Node* node, int64_t stage_index, Node* stage_done, ScheduledBlock* block,
    const BlockConversionPassOptions& options) {
  std::string base_name = PipelineSignalName(node->GetName(), stage_index);

  // As a special case, check if the node is a tuple
  // containing types that are of zero-width.  If so, separate them out so
  // that future optimization passes can remove them.
  //
  // Note that for nested tuples, only the first level will be split,
  // any nested tuple will remain as a tuple.
  Type* node_type = node->GetType();
  if (node_type->IsTuple()) {
    TupleType* tuple_type = node_type->AsTupleOrDie();

    if (HasZeroWidthType(tuple_type)) {
      std::vector<Node*> split_registers(tuple_type->size());

      // Create registers for each element.
      for (int64_t i = 0; i < split_registers.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Node * split_node,
                             block->MakeNodeInStage<TupleIndex>(
                                 stage_index, node->loc(), node, i));
        XLS_ASSIGN_OR_RETURN(
            Node * reg_read,
            CreatePipelineRegister(absl::StrFormat("%s_index%d", base_name, i),
                                   split_node, stage_index, stage_done, block,
                                   options));
        split_registers[i] = reg_read;
      }

      // Reconstruct tuple for the rest of the graph.
      XLS_ASSIGN_OR_RETURN(Node * merge_after_reg_read,
                           block->MakeNodeInStage<Tuple>(
                               stage_index + 1, node->loc(), split_registers));

      return merge_after_reg_read;
    }
  }

  // Create a single register to store the node, and return the register read.
  return CreatePipelineRegister(base_name, node, stage_index, stage_done, block,
                                options);
}

// Determine which stages are mutually exclusive with each other.
//
// Since each state element defines a mutual exclusive zone lasting from its
// first read to its first write we can walk through the stage list updating
// the mutual exclusion state.
absl::StatusOr<ConcurrentStageGroups> CalculateConcurrentStages(
    ScheduledBlock* block) {
  ConcurrentStageGroups result(block->stages().size());
  if (block->source() == nullptr || !block->source()->IsProc()) {
    return result;
  }

  // Find all the mutex regions created by unconditional state feedback.
  absl::flat_hash_map<StateElement*, int64_t> read_by_stage;
  absl::flat_hash_map<StateElement*, int64_t> first_write_stage;
  for (Node* node : block->nodes()) {
    if (!node->Is<Next>()) {
      continue;
    }
    Next* next = node->As<Next>();
    StateRead* state_read = node->As<Next>()->state_read()->As<StateRead>();
    StateElement* state_element = state_read->state_element();
    if (state_read->predicate().has_value()) {
      // If the state read is predicated, then it doesn't start a mutual
      // exclusion zone.
      continue;
    }
    XLS_ASSIGN_OR_RETURN(read_by_stage[state_element],
                         block->GetStageIndex(state_read));

    XLS_ASSIGN_OR_RETURN(int64_t write_stage, block->GetStageIndex(next));
    auto [it, inserted] =
        first_write_stage.try_emplace(state_element, write_stage);
    if (!inserted) {
      it->second = std::min(it->second, write_stage);
    }
  }
  for (const auto& [state_element, read_stage] : read_by_stage) {
    auto write_stage_it = first_write_stage.find(state_element);
    if (write_stage_it == first_write_stage.end()) {
      // This state element is never written, so it doesn't create a mutex
      // region. (It should be optimized away elsewhere.)
      continue;
    }
    int64_t write_stage = write_stage_it->second;
    VLOG(2) << "State element " << state_element->name()
            << " creates a mutex region from stage " << read_stage
            << " to stage " << write_stage;
    for (int64_t i = read_stage; i < write_stage; ++i) {
      // NB <= since end is inclusive.
      for (int64_t j = i + 1; j <= write_stage; ++j) {
        result.MarkMutuallyExclusive(i, j);
      }
    }
  }

  return result;
}

}  // namespace

absl::StatusOr<bool> PipelineRegisterInsertionPass::InsertPipelineRegisters(
    ScheduledBlock* block, const BlockConversionPassOptions& options) const {
  XLS_ASSIGN_OR_RETURN(ConcurrentStageGroups concurrent_stages,
                       CalculateConcurrentStages(block));

  // Map of StateElement -> earliest stage index containing a Next for it.
  // This is used to determine if a StateRead's lifetime can be extended.
  absl::flat_hash_map<StateElement*, int64_t> earliest_next_stage;
  for (Node* node : block->nodes()) {
    if (node->Is<Next>()) {
      Next* next = node->As<Next>();
      StateElement* state_element =
          next->state_read()->As<StateRead>()->state_element();
      int64_t stage = *block->GetStageIndex(next);
      auto [it, inserted] =
          earliest_next_stage.try_emplace(state_element, stage);
      if (!inserted) {
        it->second = std::min(it->second, stage);
      }
    }
  }

  // Compile all the references that could require pipeline registers.
  absl::flat_hash_map<Node*, int64_t> last_stage_referencing;
  absl::btree_map<Node*, absl::flat_hash_map<int64_t, std::vector<Node*>>,
                  Node::NodeIdLessThan>
      cross_stage_references;
  for (const auto& [stage_index, stage] : iter::enumerate(block->stages())) {
    for (Node* node : stage) {
      auto stage_it = last_stage_referencing.find(node);
      auto ref_it = cross_stage_references.find(node);

      for (Node* user : node->users()) {
        if (!block->IsStaged(user)) {
          // This user is outside the schedule, so its pipeline support should
          // be handled separately.
          continue;
        }
        if (user->Is<Next>() && user->As<Next>()->value() != node &&
            user->As<Next>()->predicate() != node) {
          // This user is just referencing `node` as its associated StateRead;
          // no actual data is being passed, so we don't need pipeline
          // registers.
          continue;
        }
        int64_t user_stage_index = *block->GetStageIndex(user);
        if (user_stage_index <= stage_index) {
          XLS_RET_CHECK_EQ(user_stage_index, stage_index)
              << "Found node " << user->GetName() << " in stage "
              << user_stage_index << ", with operand " << node->GetName()
              << " in earlier stage " << stage_index;
          // This user is in the same stage as `node`, so there's no need for a
          // pipeline register.
          continue;
        }
        CHECK_GT(user_stage_index, stage_index);

        if (stage_it == last_stage_referencing.end()) {
          bool inserted;
          std::tie(stage_it, inserted) =
              last_stage_referencing.emplace(node, user_stage_index);
          CHECK(inserted);
        } else if (stage_it->second < user_stage_index) {
          stage_it->second = user_stage_index;
        }

        if (ref_it == cross_stage_references.end()) {
          bool inserted;
          std::tie(ref_it, inserted) = cross_stage_references.emplace(
              node, absl::flat_hash_map<int64_t, std::vector<Node*>>{
                        {user_stage_index, std::vector<Node*>({user})}});
          CHECK(inserted);
        } else {
          ref_it->second[user_stage_index].push_back(user);
        }
      }
    }
  }

  if (cross_stage_references.empty()) {
    return false;
  }

  std::vector<Node*> stage_done(block->stages().size(), nullptr);
  for (const auto& [node, references] : cross_stage_references) {
    auto stage_it = last_stage_referencing.find(node);
    CHECK(stage_it != last_stage_referencing.end());
    XLS_ASSIGN_OR_RETURN(int64_t orig_stage_index, block->GetStageIndex(node));
    int64_t last_stage_index = stage_it->second;

    Node* live_node = node;
    int64_t live_node_update_stage = orig_stage_index;
    bool live_node_is_pipelined = false;
    for (int64_t stage_index = orig_stage_index;
         stage_index <= last_stage_index; ++stage_index) {
      const Stage& stage = block->stages()[stage_index];

      if (live_node != node) {
        // Update all references to `node` in the current stage to point to the
        // live node holding its value.
        auto stage_references_it = references.find(stage_index);
        if (stage_references_it != references.end()) {
          absl::Span<Node* const> stage_references =
              stage_references_it->second;
          for (Node* const user : stage_references) {
            // The only exception: if the user is a Next and `node` is its
            // corresponding StateRead, we need to avoid updating that specific
            // operand, since no data is being passed. For simplicity, we put it
            // back afterwards rather than actually avoiding the update.
            bool restore_state_read =
                user->Is<Next>() && user->As<Next>()->state_read() == node;
            user->ReplaceOperand(node, live_node);
            if (restore_state_read) {
              XLS_RETURN_IF_ERROR(user->As<Next>()->ReplaceOperandNumber(
                  Next::kStateReadOperand, node));
            }
          }
        }
      }

      if (stage_index == last_stage_index) {
        // No later stages reference this value, so we're finished.
        break;
      }

      if (options.codegen_options.register_merge_strategy() !=
          verilog::CodegenOptions::RegisterMergeStrategy::kDontMerge) {
        // Check if we can extend the lifetime of the `live_node` to cover
        // the next stage. If so, we can avoid adding a pipeline register here.
        //
        // NOTE: Only RegisterReads (and StateReads, which lower to
        //       RegisterReads) can be treated as having lifetimes beyond a
        //       single stage without creating multi-cycle paths.
        //
        // In general, we can extend the lifetime of the `live_node` to the next
        // stage if the next stage is mutually exclusive with everything back to
        // the earliest stage that can *change* the value in the register.
        // Currently, we only extend lifetimes for StateReads and pipeline
        // registers.
        //
        // If the live node is a pipeline register, it's safe to extend as long
        // as the next stage is mutually exclusive with everything up to the
        // register write (which is from the previous stage).
        //
        // If the live node is a StateRead, it's safe to extend its lifetime up
        // to its earliest Next operation. If the state could be updated in
        // stage K, its backing register will contain the *new* value in stage
        // K+1, so we cannot extend the StateRead's lifetime into K+1; we must
        // insert a pipeline register to capture the old value. (Mutual
        // exclusion is guaranteed by the way state feedback loops are
        // implemented.)
        //
        // TODO(epastor): Add support for extending lifetimes for more
        //                registers, (e.g.) flopped Receives. Might be simplest
        //                if each register records the earliest stage that can
        //                enable a write (where known).
        bool can_extend_lifetime = false;
        if (live_node_is_pipelined) {
          can_extend_lifetime = true;
          for (int64_t i = live_node_update_stage;
               can_extend_lifetime && i <= stage_index; ++i) {
            if (!concurrent_stages.IsMutuallyExclusive(i, stage_index + 1)) {
              VLOG(3) << "Can't extend lifetime of " << live_node->GetName()
                      << " through stage " << stage_index + 1
                      << " because of possible concurrency between stages " << i
                      << " and " << stage_index + 1;
              can_extend_lifetime = false;
            }
          }
        } else if (live_node->Is<StateRead>()) {
          can_extend_lifetime = true;
          StateElement* se = live_node->As<StateRead>()->state_element();
          if (auto it = earliest_next_stage.find(se);
              it != earliest_next_stage.end()) {
            if (stage_index + 1 > it->second) {
              VLOG(3) << "Can't extend lifetime of " << live_node->GetName()
                      << " through stage " << stage_index + 1
                      << " because state is updated in stage " << it->second;
              can_extend_lifetime = false;
            }
          }
        }
        if (can_extend_lifetime) {
          VLOG(3) << "Extending lifetime of " << live_node->GetName()
                  << " through stage " << stage_index + 1;
          continue;
        }
      }

      // Store the current value in the next pipeline register.
      if (stage_done[stage_index] == nullptr) {
        XLS_ASSIGN_OR_RETURN(stage_done[stage_index],
                             block->MakeNodeWithName<NaryOp>(
                                 SourceInfo(),
                                 absl::MakeConstSpan({stage.outputs_valid(),
                                                      stage.outputs_ready()}),
                                 Op::kAnd,
                                 block->UniquifyNodeName(PipelineSignalName(
                                     "stage_done", stage_index))));
      }
      XLS_ASSIGN_OR_RETURN(
          live_node,
          AddPipelineRegisterFor(live_node, stage_index,
                                 stage_done[stage_index], block, options));
      live_node_update_stage = stage_index;
      live_node_is_pipelined = true;
    }
  }

  return true;
}

absl::StatusOr<bool> PipelineRegisterInsertionPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (block->IsScheduled()) {
      ScheduledBlock* scheduled_block =
          absl::down_cast<ScheduledBlock*>(block.get());
      XLS_ASSIGN_OR_RETURN(bool changed_block,
                           InsertPipelineRegisters(scheduled_block, options));
      changed |= changed_block;
    }
  }
  return changed;
}

}  // namespace xls::codegen
