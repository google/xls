// Copyright 2024 The XLS Authors
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

#ifndef XLS_CODEGEN_PASSES_NG_STAGE_CONVERSION_H_
#define XLS_CODEGEN_PASSES_NG_STAGE_CONVERSION_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// TODO(tedhong): 2024-12-18 - Move metadata into the IR rather than
// using this bag-on-the-side mechanism.  For migration, metadata
// used in codegen is separated as we develop the right set of metadata
// needed for current and envisioned codegen passes.

// This metadata is attached to these channel and nodes so that passes can
// identify these nodes.
class SpecialUseMetadata {
 public:
  // Certain channels and nodes have special purposes like:
  //   - External IO - associated with i/o channels to the pipeline.
  //   - Datapath - associated channels between stages.
  //   - Internal loopback - associated with loopback channels between stages.
  //   - Read state - associated with ops that read state.
  //   - Next state - associated with next state ops.
  //
  // This enum encodes the different purposes as a bit-set.
  enum Purpose : uint32_t {
    kNone = 0x0,
    kExternalInput = 0x1,
    kExternalOutput = 0x2,
    kExternalIO = kExternalInput | kExternalOutput,
    kDatapathInput = 0x4,
    kDatapathOutput = 0x8,
    kDatapathIO = kDatapathInput | kDatapathOutput,
    kLoopback = 0x10,
    kReadState = 0x20,
    kNextState = 0x40,
  };

  // Associates a double-ended channel with the node.
  SpecialUseMetadata(ChannelWithInterfaces channel, Purpose purpose)
      : channel_(channel), purpose_(purpose), is_single_ended_(false) {
    QCHECK(channel_.channel != nullptr);
    QCHECK(channel_.send_interface != nullptr);
    QCHECK(channel_.receive_interface != nullptr);
  }

  // Associates a single-ended receive channel with the node.
  SpecialUseMetadata(ReceiveChannelInterface* chan_interface, Purpose purpose)
      : channel_(ChannelWithInterfaces{.channel = nullptr,
                                       .send_interface = nullptr,
                                       .receive_interface = chan_interface}),
        purpose_(purpose),
        is_single_ended_(true) {}

  // Associates a single-ended send channel with the node.
  SpecialUseMetadata(SendChannelInterface* chan_interface, Purpose purpose)
      : channel_(ChannelWithInterfaces{.channel = nullptr,
                                       .send_interface = chan_interface,
                                       .receive_interface = nullptr}),
        purpose_(purpose),
        is_single_ended_(true) {}

  void SetGroupId(int64_t group_id) { group_id_ = group_id; }

  int64_t group_id() const { return group_id_; }

  Purpose purpose() const { return purpose_; }

  bool is_single_ended() const { return is_single_ended_; }

  bool strictly_after() const { return strictly_after_; }

  Channel* GetChannel() const {
    QCHECK(channel_.channel != nullptr);
    return channel_.channel;
  }

  SendChannelInterface* GetSendChannelInterface() {
    QCHECK(channel_.send_interface != nullptr);
    return channel_.send_interface;
  }

  ReceiveChannelInterface* GetReceiveChannelInterface() {
    QCHECK(channel_.receive_interface != nullptr);
    return channel_.receive_interface;
  }

  void SetStrictlyAfterAll(bool strictly_after) {
    strictly_after_ = strictly_after;
  }

 private:
  // May contain both ends of a particular channel.
  // If a single-ended channel, only one ChannelInterface is used.
  const ChannelWithInterfaces channel_;

  // Associates a set of purposes with the channel.
  // The purpose is a bit-set of the Purpose enum.
  const Purpose purpose_;

  // True if the channel is a single-ended channel (i.e. only a single
  // Send or Receive reference).
  const bool is_single_ended_ = false;

  // Channels marked with strictly_after, do not fire until the channels
  // it is dependent on are all ready and valid.
  bool strictly_after_ = false;

  // Group ID for the channel.  All channels with the same non-negative
  // group id will share flow control (i.e. ready valid signals).
  int64_t group_id_ = -1;
};

// Metadata associated with a single proc after stage conversion.
//
// Each proc created by stage conversion has a corresponding ProcMetadata
// object.  The top-level proc has a parent of nullptr.
//
// The metadata object is used to track the following:
//   - Maps nodes from the original function base to nodes in this proc.
//     Note that a given node may be associated with multiple stages/proc.
//   - Identify nodes and channels with special purposes
//     (i.e. I/O, datapath channels, etc...)
class ProcMetadata {
 public:
  // Create a new ProcMetadata object.
  //
  // Parameters:
  //  input - proc:     Pointer to the proc this metadata is associated with.
  //  input - orig:     Pointer to the original function/proc this proc was
  //                    created from.
  //  input - parent:   Pointer to the parent metadata object.  nullptr for
  //                    the top-level proc.
  ProcMetadata(Proc* proc, FunctionBase* orig, ProcMetadata* parent)
      : proc_(proc), orig_(orig), parent_(parent) {}

  bool IsTop() const { return parent_ == nullptr; }

  Proc* proc() const { return proc_; }

  FunctionBase* orig() const { return orig_; }

  ProcMetadata* parent() const { return parent_; }

  // Associate with a receive channel interface with a node in the original
  // function/proc.  After stage conversion, a receive op is used
  // instead of the node in this stage.
  SpecialUseMetadata* AssociateReceiveChannelInterface(
      Node* node, ReceiveChannelInterface* channel,
      SpecialUseMetadata::Purpose purpose) {
    SpecialUseMetadata* metadata =
        special_use_metadata_
            .emplace_back(
                std::make_unique<SpecialUseMetadata>(channel, purpose))
            .get();

    orig_node_to_metadata_map_[node].push_back(metadata);
    channel_interface_to_metadata_map_[channel] = metadata;

    return metadata;
  }

  // Associate with a send channel interface with a node in the original
  // function/proc.  After stage conversion a send op is used to send the
  // node's value to subsequent stages and/or external blocks.
  SpecialUseMetadata* AssociateSendChannelInterface(
      Node* node, SendChannelInterface* channel,
      SpecialUseMetadata::Purpose purpose) {
    SpecialUseMetadata* metadata =
        special_use_metadata_
            .emplace_back(
                std::make_unique<SpecialUseMetadata>(channel, purpose))
            .get();

    orig_node_to_metadata_map_[node].push_back(metadata);
    channel_interface_to_metadata_map_[channel] = metadata;

    return metadata;
  }

  // Associated a channel with a node in the original function/proc.
  //
  // After stage conversion, the channel is used to pass
  // the node's value between stages.
  SpecialUseMetadata* AssociateChannelInterfaces(
      Node* node, ChannelWithInterfaces channel,
      SpecialUseMetadata::Purpose purpose) {
    SpecialUseMetadata* metadata =
        special_use_metadata_
            .emplace_back(
                std::make_unique<SpecialUseMetadata>(channel, purpose))
            .get();

    orig_node_to_metadata_map_[node].push_back(metadata);
    channel_interface_to_metadata_map_[channel.send_interface] = metadata;
    channel_interface_to_metadata_map_[channel.receive_interface] = metadata;

    return metadata;
  }

  // For the specific purposes (i.e I/O, datapath, etc...), get the
  // corresponding metadata objects.  More than one use of a node may exist.
  absl::StatusOr<std::vector<SpecialUseMetadata*>> FindSpecialUseMetadata(
      Node* orig_node, SpecialUseMetadata::Purpose purpose);

  // For the specific channel interface, get the metadata associated with it.
  absl::StatusOr<SpecialUseMetadata*> FindSpecialUseMetadataForChannel(
      ChannelInterface* chan_interface);

  // Given a node from the source Proc IR, returns true if
  // that node has a recorded mapping to this stage.
  bool HasFromOrigMapping(Node* orig_node) const {
    return from_orig_map_.contains(orig_node);
  }

  // Given a node from the source Proc IR, returns the corresponding node
  // in the stage-proc this metadata is associated with.
  //
  // Terminates the program if the corresponding node is not found.
  Node* GetFromOrigMapping(Node* orig_node) const {
    auto it = from_orig_map_.find(orig_node);

    if (it == from_orig_map_.end()) {
      LOG(FATAL) << absl::StreamFormat("Node %s, not found in source ir.",
                                       orig_node->ToString());
    }

    return it->second;
  }

  // Records a mapping for a given proc-ir node to the corresponding node
  // in the stage-proc this metadata is associated with.
  void RecordNodeMapping(Node* orig_node, Node* proc_node) {
    from_orig_map_[orig_node] = proc_node;
  }

 private:
  // Pointer to the proc this metadata is associated with..
  Proc* proc_;

  // Pointer to the original function/proc this proc was created from.
  FunctionBase* orig_ = nullptr;

  // Pointer to the metadata associated with the parent proc.
  // nullptr if this is associated with the top-level proc.
  ProcMetadata* parent_ = nullptr;

  // Maps nodes from the original function base to the new nodes in this proc.
  absl::flat_hash_map<Node*, Node*> from_orig_map_;

  // Associates additional metadata with special nodes in the original
  // function base.
  std::vector<std::unique_ptr<SpecialUseMetadata>> special_use_metadata_;
  absl::flat_hash_map<Node*, std::vector<SpecialUseMetadata*>>
      orig_node_to_metadata_map_;
  absl::flat_hash_map<ChannelInterface*, SpecialUseMetadata*>
      channel_interface_to_metadata_map_;
};

// Groups together metadata associated with procs created by stage conversion.
//
// Each proc created by stage conversion has two sets of procs
//   1. A pipeline proc with stitches together all stages and contain any
//      sub-procs that are instantiated due to multi-proc codegen.
//   2. A stage proc for each stage of the pipeline.
class StageConversionMetadata {
 public:
  // Creates a new metadata object for the given pipeline.
  //
  // A pipeline is associated with a single original function/proc and
  // may have multiple stages which will be associated with children of this
  // metadata object and pipeline proc.
  ProcMetadata* AssociateWithNewPipeline(Proc* pipeline, FunctionBase* orig,
                                         ProcMetadata* parent = nullptr) {
    ProcMetadata* metadata = proc_metadata_
                                 .emplace_back(std::make_unique<ProcMetadata>(
                                     pipeline, orig, parent))
                                 .get();
    orig_to_metadata_map_[orig].push_back(metadata);
    return metadata;
  }

  // Creates a new metadata object for the given stage.
  //
  // A stage is associated with a pipeline previously created with
  // AssociateWithNewPipeline().
  //
  // All stages of a pipeline share the same original function/proc.
  ProcMetadata* AssociateWithNewStage(Proc* stage, FunctionBase* orig,
                                      ProcMetadata* parent) {
    ProcMetadata* metadata =
        proc_metadata_
            .emplace_back(std::make_unique<ProcMetadata>(stage, orig, parent))
            .get();
    orig_to_metadata_map_[orig].push_back(metadata);
    return metadata;
  }

  // Returns the top-level proc metadata associated with the given function
  // base.
  absl::StatusOr<ProcMetadata*> GetTopProcMetadata(FunctionBase* orig);

  // Get associated children ProcMetdata objects.
  absl::StatusOr<std::vector<ProcMetadata*>> GetChildrenOf(
      ProcMetadata* proc_m);

 private:
  // Stores metadata associated with the procs that stage conversion
  // creates.
  std::vector<std::unique_ptr<ProcMetadata>> proc_metadata_;

  // Associates the source proc with each of the new procs that are
  // created by stage conversion.
  absl::flat_hash_map<FunctionBase*, std::vector<ProcMetadata*>>
      orig_to_metadata_map_;
};

// Converts a func/proc into a pipelined series of stages.
//
// Parameters:
//  input - top_name: Name to give the top proc of the pipeline.
//  input - schedule: Schedule and scheduled func/proc to use.
//  input - options:  Codegen options.
//  inout - unit:     Metadata for codegen passes.
//  inout - package:  Package to create the pipeline in.
absl::Status SingleFunctionBaseToPipelinedStages(
    std::string_view top_name, const PipelineSchedule& schedule,
    const CodegenOptions& options, StageConversionMetadata& metadata);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_STAGE_CONVERSION_H_
