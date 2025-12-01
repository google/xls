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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/enumerate.hpp"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
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
    XLS_ASSIGN_OR_RETURN(
        Node * reset_asserted,
        block->MakeNode<UnOp>(SourceInfo(), *block->GetResetPort(),
                              options.codegen_options.reset()->active_low()
                                  ? Op::kNot
                                  : Op::kIdentity));
    XLS_ASSIGN_OR_RETURN(
        load_enable,
        block->MakeNode<NaryOp>(
            node->loc(), absl::MakeConstSpan({stage_done, reset_asserted}),
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

}  // namespace

absl::StatusOr<bool> PipelineRegisterInsertionPass::InsertPipelineRegisters(
    ScheduledBlock* block, const BlockConversionPassOptions& options) const {
  // Compile all the references that will require pipeline registers.
  absl::flat_hash_map<Node*, int64_t> last_stage_referencing;
  absl::flat_hash_map<Node*, absl::flat_hash_map<int64_t, std::vector<Node*>>>
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
    Node* last_stage_node = node;
    for (int64_t stage_index = orig_stage_index;
         stage_index <= last_stage_index; ++stage_index) {
      const Stage& stage = block->stages()[stage_index];
      auto stage_references_it = references.find(stage_index);
      if (stage_references_it != references.end()) {
        absl::Span<Node* const> stage_references = stage_references_it->second;
        for (Node* const user : stage_references) {
          // Replace `node` with the pipeline register holding its value for the
          // current stage.
          user->ReplaceOperand(node, last_stage_node);
        }
      }

      if (stage_index == last_stage_index) {
        continue;
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
          last_stage_node,
          AddPipelineRegisterFor(last_stage_node, stage_index,
                                 stage_done[stage_index], block, options));
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
      ScheduledBlock* scheduled_block = down_cast<ScheduledBlock*>(block.get());
      XLS_ASSIGN_OR_RETURN(bool changed_block,
                           InsertPipelineRegisters(scheduled_block, options));
      changed |= changed_block;
    }
  }
  return changed;
}

}  // namespace xls::codegen
