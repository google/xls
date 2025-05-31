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

#ifndef XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_H_
#define XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/passes_ng/block_channel_adapter.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::verilog {

// Groups together slots and adapters associated with a channel-ref in a block.
class BlockChannelMetadata {
 public:
  // Creates a new metadata object that can associate a channel interface
  // (in a stage proc) with slots and an adapter in a block.
  explicit BlockChannelMetadata(const ChannelInterface* ABSL_NONNULL channel_interface)
      : channel_interface_(channel_interface) {}

  // Returns the channel interface associated with this set of slots
  // and adapters.
  const ChannelInterface* channel_interface() { return channel_interface_; }

  // Associates a slot with this object.
  BlockChannelMetadata& AddSlot(BlockRDVSlot slot) {
    slots_.push_back(std::move(slot));
    return *this;
  }

  // Returns the associated slots.
  absl::Span<BlockRDVSlot> slots() { return absl::MakeSpan(slots_); }

  // Associates an adapter with this object.  Returns an error if an adapter
  // is already associated.
  absl::Status AddAdapter(RDVAdapter adapter) {
    if (adapter_.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("ChannelInterface %s already has an adapter.",
                          channel_interface_->name()));
    }

    adapter_ = std::move(adapter);

    return absl::OkStatus();
  }

  // Returns the associated adapter.
  std::optional<RDVAdapter>& adapter() { return adapter_; }

  // Returns the direction of the channel.
  ChannelDirection direction() const { return channel_interface_->direction(); }

  // Returns the channel associated with the channel interface.
  //
  // This is only valid for internal channels.
  const Channel* channel() const { return channel_; }

  // Returns true if the channel is an internal channel (and not an interface
  // channel).
  bool IsInternalChannel() const { return channel_ != nullptr; }

  // Record that this set of slots/adapters is associated with a specific
  // internal channel.
  BlockChannelMetadata& SetChannel(const Channel* channel) {
    channel_ = channel;
    return *this;
  }

 private:
  // The channel interface associated with this set of slots and adapters.
  const ChannelInterface* channel_interface_;

  // The slots/adapters that is metadata is for.
  std::vector<BlockRDVSlot> slots_;
  std::optional<RDVAdapter> adapter_;

  // For internal channels, this is the channel associated with the channel
  // reference.
  const Channel* channel_ = nullptr;
};

// Groups together the metadata needed by codegen passes to generate code
// for a block associated with a stage proc.
class BlockMetadata {
 public:
  // Creates a new block metadata associated with the metadata for a stage proc
  // and a block.
  BlockMetadata(ProcMetadata* ABSL_NONNULL stage_metadata,
                Block* ABSL_NONNULL block)
      : stage_metadata_(stage_metadata), block_(block) {}

  Block* block() { return block_; }
  const Block* block() const { return block_; }

  ProcMetadata* stage_metadata() { return stage_metadata_; }
  const ProcMetadata* stage_metadata() const { return stage_metadata_; }

  // Associates the metadata for a channel with this object.
  void AddChannelMetadata(BlockChannelMetadata metadata) {
    if (metadata.direction() == xls::ChannelDirection::kSend) {
      outputs_.push_back(
          std::make_unique<BlockChannelMetadata>(std::move(metadata)));

      channel_interface_to_metadata_map_[metadata.channel_interface()] =
          outputs_.back().get();
    } else {
      inputs_.push_back(
          std::make_unique<BlockChannelMetadata>(std::move(metadata)));

      channel_interface_to_metadata_map_[metadata.channel_interface()] =
          inputs_.back().get();
    }
  }

  // Returns the metadata objects for input channels.
  absl::Span<std::unique_ptr<BlockChannelMetadata>> inputs() {
    return absl::MakeSpan(inputs_);
  }

  // Returns the metadata objects for output channels.
  absl::Span<std::unique_ptr<BlockChannelMetadata>> outputs() {
    return absl::MakeSpan(outputs_);
  }

  // Returns the metadata object associated with stage proc's a channel
  // interface.
  absl::StatusOr<BlockChannelMetadata*> GetChannelMetadata(
      const ChannelInterface* channel_interface) {
    auto it = channel_interface_to_metadata_map_.find(channel_interface);
    if (it == channel_interface_to_metadata_map_.end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("No block channel metadata found for channel ref %s",
                          channel_interface->name()));
    }
    return it->second;
  }

 private:
  // List of BlockChannelMetadata objects for input channels.
  std::vector<std::unique_ptr<BlockChannelMetadata>> inputs_;

  // List of BlockChannelMetadata objects for output channels.
  std::vector<std::unique_ptr<BlockChannelMetadata>> outputs_;

  // Map from stage proc-ir channel interfaces to BlockChannelMetadata objects.
  absl::flat_hash_map<const ChannelInterface*, BlockChannelMetadata*>
      channel_interface_to_metadata_map_;

  // The metadata for the stage proc associated with this block.
  ProcMetadata* stage_metadata_;

  // The block associated with this metadata.
  Block* block_;
};

// Groups together the metadata needed by codegen passes to generate code
// for a block hierarchy associated with a stage proc hierarchy.
class BlockConversionMetadata {
 public:
  // Creates a new metadata object for the given stage block.
  //
  // The block is associated with a stage proc previously created with
  // Stage Conversion.
  BlockMetadata* AssociateWithNewBlock(ProcMetadata* ABSL_NONNULL stage,
                                       Block* ABSL_NONNULL block) {
    BlockMetadata* metadata =
        block_metadata_
            .emplace_back(std::make_unique<BlockMetadata>(stage, block))
            .get();
    proc_to_block_metadata_map_[stage] = metadata;
    block_to_block_metadata_map_[block] = metadata;
    return metadata;
  }

  // Returns the block metadata associated with the given stage proc metadata.
  absl::StatusOr<BlockMetadata*> GetBlockMetadata(
      ProcMetadata* ABSL_NONNULL stage) {
    auto it = proc_to_block_metadata_map_.find(stage);
    if (it == proc_to_block_metadata_map_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No block metadata found for stage %s", stage->proc()->name()));
    }
    return it->second;
  }

  // Returns the block metadata associated with the given block.
  absl::StatusOr<BlockMetadata*> GetBlockMetadata(Block* ABSL_NONNULL block) {
    auto it = block_to_block_metadata_map_.find(block);
    if (it == block_to_block_metadata_map_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No block metadata found for block %s", block->name()));
    }
    return it->second;
  }

 private:
  // Owned list of block metadata objects.
  std::vector<std::unique_ptr<BlockMetadata>> block_metadata_;

  // Map from stage proc metadata to block metadata.
  absl::flat_hash_map<ProcMetadata*, BlockMetadata*>
      proc_to_block_metadata_map_;

  // Map from blocks to block metadata.
  absl::flat_hash_map<Block*, BlockMetadata*> block_to_block_metadata_map_;
};

// Creates new blocks for each proc in the proc hierarchy under the given
// proc associated with top_metadata.  The proc hierarchy is assumed to be
// a proc hierarchy previously created with Stage Conversion.
absl::StatusOr<Block*> CreateBlocksForProcHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata);

// Depending on codegen options, add in reset and clock ports to the block
// hierarchy.
absl::StatusOr<Block*> AddResetAndClockPortsToBlockHierarchy(
    const CodegenOptions& options, ProcMetadata& top_metadata,
    StageConversionMetadata& stage_conversion_metadata,
    BlockConversionMetadata& block_conversion_metadata);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_STAGE_TO_BLOCK_CONVERSION_H_
