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

// This file contains classes used to index different network objects.
//
// Indexing orders and assigns contiguous numeric indexes for selected
// network objects such as virtual channels and destination components.
// These indices are used by the simulator instead of actual names/ids of
// the objects to better simulate hardware behavior.

#ifndef XLS_NOC_SIMULATION_INDEXER_H_
#define XLS_NOC_SIMULATION_INDEXER_H_

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/parameters.h"

namespace xls {
namespace noc {

// Associates each port used by a set of network components with an
// ordered numeric index.  Separate output and input indices are
// provided and each are sequential in the range  [0 ... Input/OutputCount()).
class PortIndexMap {
 public:
  // Number of input ports associated with a network component.
  absl::StatusOr<int64_t> InputPortCount(NetworkComponentId nc_id) const;

  // Number of output ports associated with a network component.
  // If port is not found then -1 is returned.
  absl::StatusOr<int64_t> OutputPortCount(NetworkComponentId nc_id) const;

  // Retrieve an input/output port based off of their index.
  // Note - must be called after FinalizePortOrder();
  absl::StatusOr<PortId> GetPortByIndex(NetworkComponentId nc_id,
                                        PortDirection dir,
                                        int64_t port_index) const;

  // Returns corresponding index of a port.
  absl::StatusOr<int64_t> GetPortIndex(PortId port_id, PortDirection dir) const;

  // Add to the index an ordering of ports.
  // All ports should be from the same network component, with the stated
  // direction, and the ordering should be unique within [0, port_index.size()).
  absl::Status Add(NetworkComponentId nc_id, PortDirection dir,
                   absl::Span<const std::pair<PortId, int64_t>> port_index);

 private:
  // Stores ordering of a particular network component's ports.
  struct PortOrder {
    std::vector<PortId> ordered_input_ports;
    std::vector<PortId> ordered_output_ports;
  };

  absl::flat_hash_map<NetworkComponentId, PortOrder> nc_to_ports_;
};

// Used to construct a PortIndexMap object.
class PortIndexMapBuilder {
 public:
  // Associate a given input/output port and vc with an index.
  // Note - index must be in the range [0, Input/OutputPortCount).
  absl::Status SetPortIndex(PortId port_id, PortDirection dir, int64_t index);

  // Returns corresponding index of a port.
  absl::StatusOr<int64_t> GetPortIndex(PortId port_id, PortDirection dir) const;

  // Returns a PortIndexMap which enables directly accessing ports based
  // off their index.
  absl::StatusOr<PortIndexMap> BuildPortIndex();

 private:
  struct PortOrder {
    std::vector<std::pair<PortId, int64_t>> input_port_index;
    std::vector<std::pair<PortId, int64_t>> output_port_index;
  };

  absl::flat_hash_map<NetworkComponentId, PortOrder> nc_to_ports_;
};

// Associates each virtual channel used in a design with an ordered numeric
// index. The index goes from 0 ... VirtualChannelCount()-1.
//
// This is to enable optimization in which the order of the virtual channels
// in a config is not the same as the index for that virtual channel.  For,
// example, the config could have ordered VC0/VC1 for port A and VC1/VC0 for
// port B.  This object enables reordering those virtual channels within the
// simulator.
class VirtualChannelIndexMap {
 public:
  // Number of virtual channels associated with a port.
  // If port is not found then -1 is returned.
  absl::StatusOr<int64_t> VirtualChannelCount(PortId port_id) const;

  // Returns corresponding index of virtual channel.
  absl::StatusOr<int64_t> GetVirtualChannelIndex(PortId port_id,
                                                 VirtualChannelParam vc) const;

  // Returns corresponding index of virtual channel name.
  absl::StatusOr<int64_t> GetVirtualChannelIndexByName(
      PortId port_id, std::string_view vc_name) const;

  // Retrieve virtual channel based off of their index.
  // Note - must be called after FinalizeVirtualChannelOrder();
  absl::StatusOr<VirtualChannelParam> GetVirtualChannelByIndex(
      PortId port_id, int64_t vc_index) const;

  // Add to the index an ordering of virtual channels.
  // All vcs should be from the same port
  // and the ordering should be unique within [0, vc_index.size()).
  absl::Status Add(
      PortId port_id,
      absl::Span<const std::pair<VirtualChannelParam, int64_t>> vc_index);

 private:
  // Stores ordering of a particular port's virtual channels.
  using VirtualChannelOrder = std::vector<VirtualChannelParam>;

  absl::flat_hash_map<PortId, VirtualChannelOrder> port_to_vcs_;
};

class VirtualChannelIndexMapBuilder {
 public:
  // Number of virtual channels associated with a port.
  // If port is not found then -1 is returned.
  absl::StatusOr<int64_t> VirtualChannelCount(PortId port_id) const;

  // Associate a given port and vc with an index.
  // Note - index must be in the range [0, VirtualChannelCount).
  absl::Status SetVirtualChannelIndex(PortId port_id, PortParam port_param,
                                      int64_t orig_index, int64_t index);

  // Returns corresponding index of virtual channel.
  absl::StatusOr<int64_t> GetVirtualChannelIndex(PortId port_id,
                                                 VirtualChannelParam vc) const;

  // Returns a VirtualChannelIndexMap which enables directly accessing vcs based
  // off their index.
  absl::StatusOr<VirtualChannelIndexMap> BuildVirtualChannelIndex() const;

 private:
  // Stores ordering of a particular port's virtual channels.
  struct VirtualChannelOrder {
    std::vector<std::pair<VirtualChannelParam, int64_t>> param_vc_index;
  };

  absl::flat_hash_map<PortId, VirtualChannelOrder> port_to_vcs_;
};

// Associates network interface (destination) with a numeric index.
class NetworkComponentIndexMap {
 public:
  // Number of network components ordered by this object.
  int64_t NetworkComponentCount() const;

  // Returns corresponding index of network component.
  absl::StatusOr<int64_t> GetNetworkComponentIndex(NetworkComponentId id) const;

  // Retrieve network component based off of its index.
  absl::StatusOr<NetworkComponentId> GetNetworkComponentByIndex(
      int64_t index) const;

  // Add to the index an ordering of
  // All vcs should be from the same port
  // and the ordering should be unique within [0, vc_index.size()).
  absl::Status Add(
      const absl::flat_hash_map<NetworkComponentId, int64_t>& nc_index);

  // Retrieve list of all network components.
  absl::Span<const NetworkComponentId> GetNetworkComponents() const {
    return absl::MakeSpan(ordered_components_);
  }

 private:
  absl::flat_hash_map<NetworkComponentId, int64_t> component_index_;
  std::vector<NetworkComponentId> ordered_components_;
};

// Associates network interface (destination) with a numeric index.
class NetworkComponentIndexMapBuilder {
 public:
  // Number of network components ordered by this object.
  int64_t NetworkComponentCount() const;

  // Associate a given network component with an index.
  // Note - index must be positive, and should be contiguous starting from
  //        0 once FinalizeNetworkComponentOrder() is called.
  absl::Status SetNetworkComponentIndex(NetworkComponentId component_id,
                                        int64_t index);

  // Returns corresponding index of network component.
  absl::StatusOr<int64_t> GetNetworkComponentIndex(NetworkComponentId id) const;

  // Returns a NetworkComponentIndexMap which enables directly accessing
  // components based off their index.
  absl::StatusOr<NetworkComponentIndexMap> BuildNetworkComponentIndex() const;

 private:
  absl::flat_hash_map<NetworkComponentId, int64_t> component_index_;
};

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_INDEXER_H_
