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

// This file provides objects that track and manage the routing tables
// of all components within a network.  This enables more efficient simulation
// and opportunities for global optimization of routing.

#ifndef XLS_NOC_SIMULATION_GLOBAL_ROUTING_TABLE_H_
#define XLS_NOC_SIMULATION_GLOBAL_ROUTING_TABLE_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/indexer.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/parameters.h"

namespace xls {
namespace noc {

class NetworkManager;

// Used to denote a specific vc attached to a given port id.
struct PortAndVCIndex {
  PortId port_id_;
  int64_t vc_index_;
};

// Used to denote a specific vc attached to a given port index.
struct PortIndexAndVCIndex {
  int64_t port_index_;
  int64_t vc_index_;
};

// Generic class to store routing tables for an entire network.
class DistributedRoutingTable {
 public:
  // Stores a routing table for a single network component (ex. Router).
  //
  // For each port/vc pair, it stores the routing table as a list of
  // tuples <destination, output_port, vc_output_index>
  //
  // RouterRoutingTable::routes[port_index][vc_input_index]
  //   is a PortRoutingList corresponding to the given input port and vc.
  //
  // PortRoutingList
  //   is a list of tuples of <destination, PortAndVCIndex>
  //
  // To find the output port for a flit arriving at index and vc that has
  // a specific destination
  //   1. Retrieve the associated PortRoutingList
  //   2. Find the tuple that matched the given destination within the list.
  using PortRoutingList = std::vector<std::pair<int64_t, PortAndVCIndex>>;

  // See comment above.
  struct RouterRoutingTable {
    std::vector<std::vector<PortRoutingList>> routes;
  };

  // Returns route to destination from a particular source network interface
  // to a sink network interface.
  //
  // - Route includes the starting component.
  // - initial_vc_index is the starting vc_index used to exit source, if no vc
  //   is used the index is 0.
  //
  // - Error conditions include:
  //   - From/to are not network interfaces.
  //   - No route is found, then
  //   - The route exceeds max_hops (likely loop detected).
  absl::StatusOr<std::vector<NetworkComponentId>> ComputeRoute(
      NetworkComponentId source, NetworkComponentId sink,
      int64_t initial_vc_index = 0, int64_t max_hops = 999);

  // Given a port, a local virtual channel, and
  // final destination (sink), return the next port and vc the data
  // should traverse.
  absl::StatusOr<PortAndVCIndex> GetNextHopPort(PortAndVCIndex from,
                                                NetworkComponentId sink);

  // Given an input port (of a router), a local virtual channel, and
  // final destination (sink), return the output port and vc the data
  // should go out on.
  absl::StatusOr<PortAndVCIndex> GetRouterOutputPort(PortAndVCIndex from,
                                                     NetworkComponentId sink);

  // Given an input port (of a router), a local virtual channel, and
  // final destination index (sink), return the output port and vc the data
  // should go out on.
  absl::StatusOr<PortAndVCIndex> GetRouterOutputPortByIndex(
      PortAndVCIndex from, int64_t destination_index);

  // TODO(tedhong): 2020-01-25 Add indexer for input/output ports of a router
  //                          and support routing directly via indices.

  // Returns mapping of vc params to local indicies.
  const VirtualChannelIndexMap& GetVirtualChannelIndices() {
    return vc_indices_;
  }

  // Returns mapping of ports to local indicies.
  const PortIndexMap& GetPortIndices() { return port_indices_; }

  // Returns mapping of source network interfaces to indicies.
  const NetworkComponentIndexMap& GetSourceIndices() { return source_indices_; }

  // Returns mapping of sink network interfaces to indicies.
  const NetworkComponentIndexMap& GetSinkIndices() { return sink_indices_; }

  // Prints the routing table of the network. Used for debugging.
  absl::Status DumpRouterRoutingTable(NetworkId network_id) const;

 private:
  friend class DistributedRoutingTableBuilderBase;
  friend class DistributedRoutingTableBuilderForTrees;
  friend class DistributedRoutingTableBuilderForMultiplePaths;

  // Resize routing_tables to accommodate the number of networks and
  // number of components in a network.
  void AllocateTableForNetwork(NetworkId network_id, int64_t component_count);

  // Get (and create if necessary) routing table associated for a component.
  RouterRoutingTable& GetRoutingTable(NetworkComponentId nc_id) {
    return routing_tables_[nc_id.network()][nc_id.id()];
  }

  // Get possible routes associated with given port and vc.
  PortRoutingList& GetRoutingList(PortAndVCIndex port_and_vc) {
    NetworkComponentId nc_id = port_and_vc.port_id_.GetNetworkComponentId();

    return GetRoutingTable(nc_id)
        .routes.at(port_and_vc.port_id_.id())
        .at(port_and_vc.vc_index_);
  }

  PortIndexMap port_indices_;
  VirtualChannelIndexMap vc_indices_;
  NetworkComponentIndexMap source_indices_;
  NetworkComponentIndexMap sink_indices_;
  NetworkId network_;

  NetworkManager* network_manager_;
  NocParameters* network_parameters_;

  // Routing tables for all components.
  // This vector of vectors is indexed via network and component local id's.
  // ie. routing table for ComponentId id is
  //  routing_tables_[id.network()][id.id()]
  std::vector<std::vector<RouterRoutingTable>> routing_tables_;
};

// Abstract base class for distributed routing table builder.
class DistributedRoutingTableBuilderBase {
 public:
  virtual absl::StatusOr<DistributedRoutingTable> BuildNetworkRoutingTables(
      NetworkId network_id, NetworkManager& network_manager,
      NocParameters& network_parameters) = 0;

  virtual ~DistributedRoutingTableBuilderBase() = default;

 protected:
  // Setup source_indices_ and sink_indices_ for network.
  virtual absl::Status BuildNetworkInterfaceIndices(
      NetworkId network_id, DistributedRoutingTable* routing_table);

  // Setup port_indices_ and vc_indices_ for network.
  virtual absl::Status BuildPortAndVirtualChannelIndices(
      NetworkId network_id, DistributedRoutingTable* routing_table);
};

// Build a routing table given a network with a tree topology.
class DistributedRoutingTableBuilderForTrees
    : public DistributedRoutingTableBuilderBase {
 public:
  absl::StatusOr<DistributedRoutingTable> BuildNetworkRoutingTables(
      NetworkId network_id, NetworkManager& network_manager,
      NocParameters& network_parameters) override;

 private:
  // Trace and setup routing table for tree-based topologies.
  absl::Status BuildRoutingTable(NetworkId network_id,
                                 DistributedRoutingTable* routing_table);

  // Updates routing table of nc for routes that travel to destination via_port.
  absl::Status AddRoutes(
      int64_t destination_index, NetworkComponentId nc, PortId via_port,
      DistributedRoutingTable* routing_table,
      absl::flat_hash_set<NetworkComponentId>& visited_components);
};

// Build a routing table given a network where multiple paths exists between a
// source-sink pair (e.g. via multiple links interconnecting routers). Algorithm
// calculates the minimum hop to derive the routing table. The modulo is
// calculated for each input-output pairs.
class DistributedRoutingTableBuilderForMultiplePaths
    : public DistributedRoutingTableBuilderBase {
 public:
  absl::StatusOr<DistributedRoutingTable> BuildNetworkRoutingTables(
      NetworkId network_id, NetworkManager& network_manager,
      NocParameters& network_parameters) override;

 private:
  // Trace and setup routing table for tree-based topologies.
  absl::Status BuildRoutingTable(NetworkId network_id,
                                 DistributedRoutingTable* routing_table);

  // Returns routing information from nc to a source port. The first element
  // defines the network component of the route, the second element defines the
  // output ports for the network component to reach the nc.
  // network : The network.
  // nc : The starting network component of the route.
  // network_parameters : Network parameters.
  absl::StatusOr<absl::flat_hash_map<NetworkComponentId, std::vector<PortId>>>
  CalculateRoutes(Network& network, NetworkComponentId nc,
                  const NocParameters& network_parameters);

  // Updates routing table for network component using their
  // network-component-port map.
  // For router's with multiple paths, assign the input ports to a the output
  // port using their modulo value (similar to round robin). For example, for a
  // router with six input ports and three output ports routing to the
  // destination, the assignment is shown in the table that follows:
  // Output Port Index   | Input Port Index
  // ----------------------------------------
  //           0         |     0, 3
  //           1         |     1, 4
  //           2         |     2, 5
  absl::Status AddRoutes(
      int64_t destination_index, const Network& network,
      const absl::flat_hash_map<NetworkComponentId, std::vector<PortId>>&
          nc_ports_map,
      DistributedRoutingTable* routing_table);
};

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_GLOBAL_ROUTING_TABLE_H_
