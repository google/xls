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

#include "xls/codegen_v_1_5/merge_ports_pass.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "google/protobuf/message_static_reflection.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/module_signature.proto.static_reflection.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

namespace {

template <google::protobuf::FieldId field, typename Msg>
void MaybeRemap(Msg& msg,
                const absl::flat_hash_map<std::string, std::string>& remap) {
  auto field_info = google::protobuf::FieldInfo<Msg, field>();
  if (field_info.Has(msg)) {
    if (auto it = remap.find(field_info.Get(msg)); it != remap.end()) {
      field_info.Set(msg, it->second);
    }
  }
}

absl::StatusOr<bool> MergeOutputPorts(Block* block) {
  // Collect the channel ports, mapping them back to the associated channels.
  enum class PortRole { kData, kValid, kReady };
  absl::flat_hash_map<
      OutputPort*,
      std::vector<std::tuple<std::string, ChannelDirection, PortRole>>>
      channel_ports;
  for (const auto& [key, metadata] : block->GetAllChannelPortMetadata()) {
    const auto& [channel_name, direction] = key;
    if (direction == ChannelDirection::kSend) {
      if (metadata.data_port.has_value()) {
        XLS_ASSIGN_OR_RETURN(OutputPort * port,
                             block->GetOutputPort(*metadata.data_port));
        channel_ports[port].push_back(
            {channel_name, direction, PortRole::kData});
      }
      if (metadata.valid_port.has_value()) {
        XLS_ASSIGN_OR_RETURN(OutputPort * port,
                             block->GetOutputPort(*metadata.valid_port));
        channel_ports[port].push_back(
            {channel_name, direction, PortRole::kValid});
      }
    }
    if (direction == ChannelDirection::kReceive &&
        metadata.ready_port.has_value()) {
      XLS_ASSIGN_OR_RETURN(OutputPort * port,
                           block->GetOutputPort(*metadata.ready_port));
      channel_ports[port].push_back(
          {channel_name, direction, PortRole::kReady});
    }
  }

  bool changed = false;
  struct AssociatedPorts {
    OutputPort* authoritative_port = nullptr;
    bool authoritative_is_channel = false;
    std::vector<OutputPort*> equivalent_ports;
  };
  absl::flat_hash_map<Node*, AssociatedPorts> ports_by_source;

  // Group the ports by their output source; since not all non-channel ports are
  // tracked by metadata, we take the first in a group as authoritative, and
  // leave the others out to avoid attempting to merge them.
  for (OutputPort* output_port : block->GetOutputPorts()) {
    bool is_channel_port = channel_ports.contains(output_port);
    auto [it, inserted] = ports_by_source.try_emplace(
        output_port->output_source(),
        AssociatedPorts{.authoritative_port = output_port,
                        .authoritative_is_channel = is_channel_port});
    if (!inserted) {
      AssociatedPorts& entry = it->second;

      if (!is_channel_port && !it->second.authoritative_is_channel) {
        // There's already a non-channel port; to avoid merging two non-channel
        // ports, we leave this one out.
        continue;
      }
      if (is_channel_port) {
        // Since we already have an entry, it has an authoritative port already.
        entry.equivalent_ports.push_back(output_port);
      } else {
        // The first non-channel port is always promoted to the authoritative
        // port, as mentioned above.
        entry.equivalent_ports.push_back(it->second.authoritative_port);
        entry.authoritative_port = output_port;
        entry.authoritative_is_channel = false;
      }
    }
  }

  // If the signature exists, collect the authoritative ports for each port
  // we're removing; we'll use them to update the signature in a single pass at
  // the end.
  absl::flat_hash_map<std::string, std::string> remapped_port;
  if (block->GetSignature().has_value()) {
    for (const auto& [source, ports] : ports_by_source) {
      for (OutputPort* port : ports.equivalent_ports) {
        remapped_port[port->name()] = ports.authoritative_port->name();
      }
    }
  }

  for (const auto& [source, ports] : ports_by_source) {
    if (ports.equivalent_ports.empty()) {
      // No ports to merge.
      continue;
    }

    // All equivalent ports should be replaced by the authoritative port.
    OutputPort* authoritative_port = ports.authoritative_port;
    for (OutputPort* port : ports.equivalent_ports) {
      for (auto& [channel_name, channel_direction, port_role] :
           channel_ports[port]) {
        // Update the corresponding port on the channel metadata.
        XLS_ASSIGN_OR_RETURN(
            ChannelPortMetadata metadata,
            block->GetChannelPortMetadata(channel_name, channel_direction));
        switch (port_role) {
          case PortRole::kData:
            metadata.data_port = authoritative_port->name();
            break;
          case PortRole::kValid:
            metadata.valid_port = authoritative_port->name();
            break;
          case PortRole::kReady:
            metadata.ready_port = authoritative_port->name();
            break;
        }
        XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(std::move(metadata)));
      }
      XLS_RETURN_IF_ERROR(block->RemoveNode(port));
    }
    changed = true;
  }

  if (changed && block->GetSignature().has_value()) {
    // We need to update the signature, updating all port references & deleting
    // the old ports.
    verilog::ModuleSignatureProto signature = *block->GetSignature();

    for (verilog::ChannelInterfaceProto& channel_interface :
         *signature.mutable_channel_interfaces()) {
      if (channel_interface.has_streaming()) {
        verilog::StreamingChannelInterfaceProto* streaming =
            channel_interface.mutable_streaming();
        MaybeRemap<"data_port_name">(*streaming, remapped_port);
        MaybeRemap<"ready_port_name">(*streaming, remapped_port);
        MaybeRemap<"valid_port_name">(*streaming, remapped_port);
      }
      if (channel_interface.has_single_value()) {
        verilog::SingleValueChannelInterfaceProto* single_value =
            channel_interface.mutable_single_value();
        MaybeRemap<"data_port_name">(*single_value, remapped_port);
      }
    }

    // Now that all the references are updated, we can delete the old ports.
    for (auto it = signature.mutable_data_ports()->begin();
         it != signature.mutable_data_ports()->end();) {
      if (remapped_port.contains(it->name())) {
        it = signature.mutable_data_ports()->erase(it);
      } else {
        ++it;
      }
    }

    block->SetSignature(std::move(signature));
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> MergePortsPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results, BlockConversionContext& context) const {
  if (options.codegen_options.preserve_ports()) {
    return false;
  }

  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    XLS_ASSIGN_OR_RETURN(bool block_changed, MergeOutputPorts(block.get()));
    changed |= block_changed;
  }
  return changed;
}

}  // namespace xls::codegen
