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

#include "xls/noc/simulation/global_routing_table.h"

#include <cstdint>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/parameters.h"

namespace xls {
namespace noc {

absl::StatusOr<std::vector<NetworkComponentId>>
DistributedRoutingTable::ComputeRoute(NetworkComponentId source,
                                      NetworkComponentId sink,
                                      int64_t initial_vc_index,
                                      int64_t max_hops) {
  NetworkComponent& source_nc = network_manager_->GetNetworkComponent(source);
  if (source_nc.kind() != NetworkComponentKind::kNISrc) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Source network component %x is of kind %d, expected kNISrc",
        source_nc.id().AsUInt64(), source_nc.kind()));
  }

  NetworkComponent& sink_nc = network_manager_->GetNetworkComponent(sink);
  if (sink_nc.kind() != NetworkComponentKind::kNISink) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Sink network component %x is of kind %d, expected kNISink",
        sink_nc.id().AsUInt64(), sink_nc.kind()));
  }

  // Network interfaces should only have one port.
  PortId final_port = sink_nc.GetPortIdByIndex(0);

  int64_t current_vc_index = initial_vc_index;
  PortId current_port = source_nc.GetPortIdByIndex(0);
  NetworkComponentId current_network_component = source;

  std::vector<NetworkComponentId> route;
  route.push_back(current_network_component);

  while (current_port != final_port) {
    absl::StatusOr<PortAndVCIndex> next =
        GetNextHopPort(PortAndVCIndex{current_port, current_vc_index}, sink);

    if (!next.ok()) {
      return absl::InternalError(
          absl::StrFormat("Route from source %x to sink %x - "
                          "could not get next hop from port %x vc index %d",
                          source_nc.id().AsUInt64(), sink_nc.id().AsUInt64(),
                          current_port.AsUInt64(), current_vc_index));
    }

    current_port = next->port_id_;
    current_vc_index = next->vc_index_;

    if (current_network_component != next->port_id_.GetNetworkComponentId()) {
      current_network_component = next->port_id_.GetNetworkComponentId();
      route.push_back(current_network_component);
    }

    if (route.size() > max_hops) {
      std::stringstream route_error_msg;
      route_error_msg << "Network components in the route thus far are:"
                      << '\n';
      for (const NetworkComponentId nc_id : route) {
        XLS_ASSIGN_OR_RETURN(
            NetworkComponentParam nc_param,
            network_parameters_->GetNetworkComponentParam(nc_id));
        std::string name = "NOT A ROUTER OR LINK.";
        if (std::holds_alternative<RouterParam>(nc_param)) {
          name = std::get<RouterParam>(nc_param).GetName();
        } else if (std::holds_alternative<LinkParam>(nc_param)) {
          name = std::get<LinkParam>(nc_param).GetName();
        }
        route_error_msg << name << '\n';
      }
      return absl::InternalError(absl::StrFormat(
          "Route from source %x to sink %x exceeded max hops %d. %s",
          source_nc.id().AsUInt64(), sink_nc.id().AsUInt64(), max_hops,
          route_error_msg.str()));
    }
  }

  return route;
}

absl::StatusOr<PortAndVCIndex> DistributedRoutingTable::GetNextHopPort(
    PortAndVCIndex from, NetworkComponentId sink) {
  Port& from_port = network_manager_->GetPort(from.port_id_);

  if (from_port.direction() == PortDirection::kOutput) {
    // If port is an output, the next port is the corresponding input port
    // on the other side of the connection.

    if (!from_port.connection().IsValid()) {
      XLS_ASSIGN_OR_RETURN(PortParam port_param,
                           network_parameters_->GetPortParam(from.port_id_));
      XLS_ASSIGN_OR_RETURN(NetworkComponentParam sink_param,
                           network_parameters_->GetNetworkComponentParam(sink));

      return absl::NotFoundError(absl::StrFormat(
          "Unable to find hop from port %s (%x) to sink %s (%x)"
          " - port is dandling.",
          port_param.GetName(), from.port_id_.AsUInt64(),
          std::get<NetworkInterfaceSinkParam>(sink_param).GetName(),
          sink.AsUInt64()));
    }
    PortId next_port =
        network_manager_->GetConnection(from_port.connection()).sink();
    return PortAndVCIndex{next_port, from.vc_index_};

  } else {
    NetworkComponentId from_nc_id = from.port_id_.GetNetworkComponentId();
    NetworkComponent& from_nc =
        network_manager_->GetNetworkComponent(from_nc_id);
    NetworkComponentKind from_nc_kind = from_nc.kind();

    if (from_nc_kind == NetworkComponentKind::kRouter) {
      return GetRouterOutputPort(from, sink);
    }
    if (from_nc_kind == NetworkComponentKind::kLink) {
      if (from_nc.GetOutputPortIds().size() == 1) {
        PortId next_port = from_nc.GetOutputPortIds()[0];
        return PortAndVCIndex{next_port, from.vc_index_};
      }
    }
  }

  XLS_ASSIGN_OR_RETURN(PortParam port_param,
                       network_parameters_->GetPortParam(from.port_id_));
  XLS_ASSIGN_OR_RETURN(NetworkComponentParam sink_param,
                       network_parameters_->GetNetworkComponentParam(sink));

  return absl::NotFoundError(
      absl::StrFormat("Unable to find hop from port %s (%x) to sink %s (%x)",
                      port_param.GetName(), from.port_id_.AsUInt64(),
                      std::get<NetworkInterfaceSinkParam>(sink_param).GetName(),
                      sink.AsUInt64()));
}

absl::StatusOr<PortAndVCIndex> DistributedRoutingTable::GetRouterOutputPort(
    PortAndVCIndex from, NetworkComponentId sink) {
  XLS_ASSIGN_OR_RETURN(int64_t destination_index,
                       sink_indices_.GetNetworkComponentIndex(sink));
  return GetRouterOutputPortByIndex(from, destination_index);
}

absl::StatusOr<PortAndVCIndex>
DistributedRoutingTable::GetRouterOutputPortByIndex(PortAndVCIndex from,
                                                    int64_t destination_index) {
  PortRoutingList& routes = GetRoutingList(from);

  for (std::pair<int64_t, PortAndVCIndex>& hop : routes) {
    if (hop.first == destination_index) {
      return hop.second;
    }
  }

  XLS_ASSIGN_OR_RETURN(
      NetworkComponentId sink,
      sink_indices_.GetNetworkComponentByIndex(destination_index));
  XLS_ASSIGN_OR_RETURN(PortParam port_param,
                       network_parameters_->GetPortParam(from.port_id_));
  XLS_ASSIGN_OR_RETURN(NetworkComponentParam sink_param,
                       network_parameters_->GetNetworkComponentParam(sink));

  return absl::NotFoundError(
      absl::StrFormat("Unable to find hop from port %s (%x) to sink %s (%x)",
                      port_param.GetName(), from.port_id_.AsUInt64(),
                      std::get<NetworkInterfaceSinkParam>(sink_param).GetName(),
                      sink.AsUInt64()));
}

absl::Status DistributedRoutingTable::DumpRouterRoutingTable(
    NetworkId network_id) const {
  const Network& network = network_manager_->GetNetwork(network_id);
  std::vector<RouterRoutingTable> nc_routing_tables =
      routing_tables_.at(network_id.id());
  XLS_ASSIGN_OR_RETURN(NetworkParam network_param,
                       network_parameters_->GetNetworkParam(network_id));
  LOG(INFO) << "Routing table for network: " << network_param.GetName();
  for (const NetworkComponentId& nc_id : network.GetNetworkComponentIds()) {
    const NetworkComponent& nc = network.GetNetworkComponent(nc_id);
    if (nc.kind() != NetworkComponentKind::kRouter) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(NetworkComponentParam nc_param,
                         network_parameters_->GetNetworkComponentParam(nc_id));
    LOG(INFO) << "Routing table for network component: "
              << std::get<RouterParam>(nc_param).GetName();
    LOG(INFO) << "Input Port Name | Input VC Name | Sink Name | Output "
                 "Port Name | Output VC Name";
    RouterRoutingTable& router_routing_table = nc_routing_tables.at(nc_id.id());
    for (PortId input_port_id : nc.GetInputPortIds()) {
      XLS_ASSIGN_OR_RETURN(PortParam input_port_param,
                           network_parameters_->GetPortParam(input_port_id));
      std::vector<VirtualChannelParam> input_vc_params =
          input_port_param.GetVirtualChannels();
      std::string_view input_port_name = input_port_param.GetName();
      // Default values setup for port with no VCs.
      std::string_view vc_name = "Default";
      int64_t vc_count = 1;
      bool get_vc_param = false;
      if (input_port_param.VirtualChannelCount() != 0) {
        vc_count = input_port_param.VirtualChannelCount();
        get_vc_param = true;
      }
      for (int64_t vc_index = 0; vc_index < vc_count; ++vc_index) {
        if (get_vc_param) {
          vc_name = input_vc_params.at(vc_index).GetName();
        }
        const PortRoutingList& routes =
            router_routing_table.routes.at(input_port_id.id()).at(vc_index);
        for (auto& [destination_index, port_id_vc_index] : routes) {
          XLS_ASSIGN_OR_RETURN(
              NetworkComponentId sink_id,
              sink_indices_.GetNetworkComponentByIndex(destination_index));
          XLS_ASSIGN_OR_RETURN(
              NetworkComponentParam sink_param,
              network_parameters_->GetNetworkComponentParam(sink_id));
          XLS_ASSIGN_OR_RETURN(
              PortParam output_port_param,
              network_parameters_->GetPortParam(port_id_vc_index.port_id_));
          std::string_view output_port_name = output_port_param.GetName();
          std::vector<VirtualChannelParam> output_vc_params =
              output_port_param.GetVirtualChannels();
          LOG(INFO)
              << input_port_name << "   " << vc_name << "   "
              << std::get<NetworkInterfaceSinkParam>(sink_param).GetName()
              << "   " << output_port_name << "   "
              << output_vc_params.at(port_id_vc_index.vc_index_).GetName();
        }
      }
    }
  }
  return absl::OkStatus();
}

void DistributedRoutingTable::AllocateTableForNetwork(NetworkId network_id,
                                                      int64_t component_count) {
  int64_t network_index = network_id.id();

  if (routing_tables_.size() <= network_index) {
    routing_tables_.resize(network_index + 1);
  }

  routing_tables_[network_index].resize(component_count);
}

absl::Status DistributedRoutingTableBuilderBase::BuildNetworkInterfaceIndices(
    NetworkId network_id, DistributedRoutingTable* routing_table) {
  NetworkComponentIndexMapBuilder source_index_builder;
  NetworkComponentIndexMapBuilder sink_index_builder;

  // This function assigns network interface indices in-order.
  NetworkManager* network_manager = routing_table->network_manager_;
  int64_t next_src_index = 0;
  int64_t next_sink_index = 0;

  for (NetworkComponent& nc :
       network_manager->GetNetwork(network_id).GetNetworkComponents()) {
    if (nc.kind() == NetworkComponentKind::kNISrc) {
      XLS_RET_CHECK_OK(source_index_builder.SetNetworkComponentIndex(
          nc.id(), next_src_index));
      ++next_src_index;
    } else if (nc.kind() == NetworkComponentKind::kNISink) {
      XLS_RET_CHECK_OK(sink_index_builder.SetNetworkComponentIndex(
          nc.id(), next_sink_index));
      ++next_sink_index;
    }
  }

  XLS_ASSIGN_OR_RETURN(routing_table->source_indices_,
                       source_index_builder.BuildNetworkComponentIndex());
  XLS_ASSIGN_OR_RETURN(routing_table->sink_indices_,
                       sink_index_builder.BuildNetworkComponentIndex());

  return absl::OkStatus();
}

absl::Status
DistributedRoutingTableBuilderBase::BuildPortAndVirtualChannelIndices(
    NetworkId network_id, DistributedRoutingTable* routing_table) {
  PortIndexMapBuilder port_index_builder;
  VirtualChannelIndexMapBuilder vc_index_builder;

  NetworkManager* network_manager = routing_table->network_manager_;
  NocParameters* network_parameters = routing_table->network_parameters_;

  for (NetworkComponent& nc :
       network_manager->GetNetwork(network_id).GetNetworkComponents()) {
    int64_t input_port_index = 0;
    int64_t output_port_index = 0;

    for (Port& ports : nc.GetPorts()) {
      // This function assigns port indices in-order.
      if (ports.direction() == PortDirection::kInput) {
        XLS_RET_CHECK_OK(port_index_builder.SetPortIndex(
            ports.id(), ports.direction(), input_port_index));
        ++input_port_index;
      } else {
        XLS_RET_CHECK_OK(port_index_builder.SetPortIndex(
            ports.id(), ports.direction(), output_port_index));
        ++output_port_index;
      }

      // This function assigns virtual channel indices in-order, per port.
      XLS_ASSIGN_OR_RETURN(PortParam port_param,
                           network_parameters->GetPortParam(ports.id()));

      int64_t vc_count = port_param.VirtualChannelCount();

      if (vc_count > 0) {
        for (int64_t i = 0; i < vc_count; ++i) {
          XLS_RET_CHECK_OK(vc_index_builder.SetVirtualChannelIndex(
              ports.id(), port_param, i, i));
        }
      }
    }
  }

  XLS_ASSIGN_OR_RETURN(routing_table->port_indices_,
                       port_index_builder.BuildPortIndex());
  XLS_ASSIGN_OR_RETURN(routing_table->vc_indices_,
                       vc_index_builder.BuildVirtualChannelIndex());

  return absl::OkStatus();
}

absl::StatusOr<DistributedRoutingTable>
DistributedRoutingTableBuilderForTrees::BuildNetworkRoutingTables(
    NetworkId network_id, NetworkManager& network_manager,
    NocParameters& network_parameters) {
  DistributedRoutingTable routing_table;

  routing_table.network_manager_ = &network_manager;
  routing_table.network_parameters_ = &network_parameters;

  XLS_RET_CHECK_OK(BuildNetworkInterfaceIndices(network_id, &routing_table));
  XLS_RET_CHECK_OK(
      BuildPortAndVirtualChannelIndices(network_id, &routing_table));
  XLS_RET_CHECK_OK(BuildRoutingTable(network_id, &routing_table));

  return routing_table;
}

absl::Status DistributedRoutingTableBuilderForTrees::BuildRoutingTable(
    NetworkId network_id, DistributedRoutingTable* routing_table) {
  NetworkManager* network_manager = routing_table->network_manager_;

  int64_t component_count =
      network_manager->GetNetwork(network_id).GetNetworkComponentCount();

  routing_table->AllocateTableForNetwork(network_id, component_count);

  const NetworkComponentIndexMap& sink_indices =
      routing_table->GetSinkIndices();

  // Algorithm:
  //  For each sink
  //   Perform DFS to srcs
  //     Record hop needed to reach destination
  for (int64_t i = 0; i < sink_indices.NetworkComponentCount(); ++i) {
    XLS_ASSIGN_OR_RETURN(NetworkComponentId sink_id,
                         sink_indices.GetNetworkComponentByIndex(i));
    absl::flat_hash_set<NetworkComponentId> visited_components;
    XLS_RET_CHECK_OK(AddRoutes(i, sink_id, PortId::kInvalid, routing_table,
                               visited_components));
  }

  return absl::OkStatus();
}

absl::Status DistributedRoutingTableBuilderForTrees::AddRoutes(
    int64_t destination_index, NetworkComponentId nc_id, PortId via_port,
    DistributedRoutingTable* routing_table,
    absl::flat_hash_set<NetworkComponentId>& visited_components) {
  NetworkManager* network_manager = routing_table->network_manager_;
  NocParameters* network_parameters = routing_table->network_parameters_;

  NetworkComponent& nc = network_manager->GetNetworkComponent(nc_id);

  if (!visited_components.insert(nc_id).second) {
    return absl::OkStatus();
  }

  if (nc.kind() == NetworkComponentKind::kRouter) {
    // Size routing table appropriately.
    DistributedRoutingTable::RouterRoutingTable& table =
        routing_table->GetRoutingTable(nc_id);
    table.routes.resize(nc.GetPortCount());

    // Update routing table.
    for (Port& port : nc.GetPorts()) {
      if (port.direction() == PortDirection::kInput) {
        PortId from_port = port.id();
        XLS_ASSIGN_OR_RETURN(PortParam from_port_param,
                             network_parameters->GetPortParam(from_port));
        XLS_ASSIGN_OR_RETURN(PortParam via_port_param,
                             network_parameters->GetPortParam(via_port));

        // TODO(tedhong): 2020-01-15 Support other VC mapping strategies.
        int64_t from_port_vc_count = from_port_param.VirtualChannelCount();
        int64_t via_port_vc_count = via_port_param.VirtualChannelCount();

        if (from_port_vc_count != via_port_vc_count) {
          return absl::UnimplementedError(absl::StrFormat(
              "VC route inference is unimplemented "
              " when vc count changes on path between"
              " port %s and port %s",
              from_port_param.GetName(), via_port_param.GetName()));
        }

        if (from_port_param.VirtualChannelCount() == 0 &&
            via_port_param.VirtualChannelCount() == 0) {
          int64_t default_vc = 0;
          table.routes[from_port.id()].resize(1);
          table.routes[from_port.id()][default_vc].emplace_back(
              destination_index, PortAndVCIndex{via_port, default_vc});
        } else {
          // VCs are mapped in-order,
          // ie traffic is rounded from the vc at index 0 to the
          //    vc at index 0 of the next port.
          // TODO(tedhong): 2020-01-15 Update this to use global virtual
          //                           channels to allow for more flexibility.
          table.routes[from_port.id()].resize(from_port_vc_count);
          for (int64_t i = 0; i < from_port_vc_count; ++i) {
            table.routes[from_port.id()][i].emplace_back(
                destination_index, PortAndVCIndex{via_port, i});
          }
        }
      }
    }
  }

  // Recurse to all input ports of the component if this component isn't
  // already a source.
  if (nc.kind() != NetworkComponentKind::kNISrc) {
    for (Port& port : nc.GetPorts()) {
      if (port.direction() == PortDirection::kInput &&
          port.connection().IsValid()) {
        PortId prior_port =
            network_manager->GetConnection(port.connection()).src();
        NetworkComponentId prior_component = prior_port.GetNetworkComponentId();
        // std::cout << absl::StrFormat("Traversing to port %x component %x",
        //  prior_port.AsUInt64(), prior_component.AsUInt64()) << std::endl;

        XLS_RET_CHECK_OK(AddRoutes(destination_index, prior_component,
                                   prior_port, routing_table,
                                   visited_components));
      }
    }
  }

  return absl::OkStatus();
}

// DistributedRoutingTableBuilderForMultiplePaths
absl::StatusOr<DistributedRoutingTable>
DistributedRoutingTableBuilderForMultiplePaths::BuildNetworkRoutingTables(
    NetworkId network_id, NetworkManager& network_manager,
    NocParameters& network_parameters) {
  DistributedRoutingTable routing_table;

  routing_table.network_manager_ = &network_manager;
  routing_table.network_parameters_ = &network_parameters;

  XLS_RET_CHECK_OK(BuildNetworkInterfaceIndices(network_id, &routing_table));
  XLS_RET_CHECK_OK(
      BuildPortAndVirtualChannelIndices(network_id, &routing_table));
  XLS_RET_CHECK_OK(BuildRoutingTable(network_id, &routing_table));

  return routing_table;
}

absl::StatusOr<absl::flat_hash_map<NetworkComponentId, std::vector<PortId>>>
DistributedRoutingTableBuilderForMultiplePaths::CalculateRoutes(
    Network& network, NetworkComponentId nc_id,
    const NocParameters& network_parameters) {
  absl::flat_hash_map<NetworkComponentId, std::vector<PortId>> nc_ports_map;
  std::queue<PortId> BFS_queue;
  absl::flat_hash_set<PortId> visited_output_ports;
  // Setup BFS state with initial output port
  const NetworkComponent& network_component =
      network.GetNetworkComponent(nc_id);
  for (PortId input_port_id : network_component.GetInputPortIds()) {
    // traverse to next component and add it to BFS queue.
    Port output_port = network.GetPort(input_port_id);
    ConnectionId conn_id = output_port.connection();
    Connection connection = network.GetConnection(conn_id);
    PortId output_port_id = connection.src();
    if (output_port_id == PortId::kInvalid) {
      XLS_ASSIGN_OR_RETURN(NetworkComponentParam nc_param,
                           network_parameters.GetNetworkComponentParam(nc_id));
      return absl::FailedPreconditionError(absl::StrFormat(
          "%s has an invalid port.",
          absl::visit([](const auto& nc) { return nc.GetName(); }, nc_param)));
    }
    BFS_queue.push(output_port_id);
  }
  BFS_queue.push(PortId::kInvalid);
  int64_t BFS_level_index = 1;
  absl::flat_hash_map<NetworkComponentId, int64_t> nc_level_map;
  while (!BFS_queue.empty()) {
    PortId current_port_id = BFS_queue.front();
    BFS_queue.pop();
    // A new level marker reached.
    if (current_port_id == PortId::kInvalid) {
      // Increase the level value.
      BFS_level_index++;
      // Add level marker to queue if elements are present in the queue.
      if (!BFS_queue.empty()) {
        BFS_queue.push(PortId::kInvalid);
      }
      continue;
    }
    // Skip if port has already been visited.
    if (!visited_output_ports.insert(current_port_id).second) {
      continue;
    }
    NetworkComponentId current_nc_id = current_port_id.GetNetworkComponentId();
    auto current_nc_id_iter = nc_level_map.find(current_nc_id);
    if (current_nc_id_iter == nc_level_map.end()) {
      // The network component was not visited prior, add new entry for
      // network component.
      nc_level_map[current_nc_id] = BFS_level_index;
    } else if (current_nc_id_iter->second < BFS_level_index) {
      // The network component was visited prior at a lower level.
      continue;
    }
    // Add output port to list.
    nc_ports_map[current_nc_id].emplace_back(current_port_id);
    const NetworkComponent& nc = network.GetNetworkComponent(current_nc_id);
    for (PortId input_port_id : nc.GetInputPortIds()) {
      // traverse to next component and add it to BFS queue.
      Port input_port = network.GetPort(input_port_id);
      ConnectionId conn_id = input_port.connection();
      Connection connection = network.GetConnection(conn_id);
      PortId output_port_id = connection.src();
      if (output_port_id == PortId::kInvalid) {
        XLS_ASSIGN_OR_RETURN(
            NetworkComponentParam nc_param,
            network_parameters.GetNetworkComponentParam(current_nc_id));
        return absl::FailedPreconditionError(absl::StrFormat(
            "%s has an invalid port.",
            absl::visit([](const auto& nc) { return nc.GetName(); },
                        nc_param)));
      }
      BFS_queue.push(output_port_id);
    }
  }
  return nc_ports_map;
}

absl::Status DistributedRoutingTableBuilderForMultiplePaths::BuildRoutingTable(
    NetworkId network_id, DistributedRoutingTable* routing_table) {
  NetworkManager* network_manager = routing_table->network_manager_;

  int64_t component_count =
      network_manager->GetNetwork(network_id).GetNetworkComponentCount();

  routing_table->AllocateTableForNetwork(network_id, component_count);

  Network& network = network_manager->GetNetwork(network_id);

  const NetworkComponentIndexMap& sink_indices =
      routing_table->GetSinkIndices();

  // Algorithm:
  //  For each sink
  //   Perform BFS to srcs
  for (int64_t i = 0; i < sink_indices.NetworkComponentCount(); ++i) {
    XLS_ASSIGN_OR_RETURN(NetworkComponentId sink_id,
                         sink_indices.GetNetworkComponentByIndex(i));
    absl::flat_hash_map<NetworkComponentId, std::vector<PortId>> nc_ports_map;
    XLS_ASSIGN_OR_RETURN(
        nc_ports_map,
        CalculateRoutes(network, sink_id, *routing_table->network_parameters_));
    XLS_RET_CHECK_OK(AddRoutes(i, network, nc_ports_map, routing_table));
  }

  return absl::OkStatus();
}

absl::Status DistributedRoutingTableBuilderForMultiplePaths::AddRoutes(
    int64_t destination_index, const Network& network,
    const absl::flat_hash_map<NetworkComponentId, std::vector<PortId>>&
        nc_ports_map,
    DistributedRoutingTable* routing_table) {
  NocParameters* network_parameters = routing_table->network_parameters_;
  for (auto& [current_nc_id, output_port_ids] : nc_ports_map) {
    const NetworkComponent& nc = network.GetNetworkComponent(current_nc_id);
    // Skip non router types.
    if (nc.kind() != NetworkComponentKind::kRouter) {
      continue;
    }
    // A route from router must be defined.
    if (output_port_ids.empty()) {
      XLS_ASSIGN_OR_RETURN(
          NetworkComponentParam nc_param,
          network_parameters->GetNetworkComponentParam(current_nc_id));

      return absl::InternalError(
          absl::StrFormat("There are no ports for network component: %s.",
                          std::get<RouterParam>(nc_param).GetName()));
    }
    DistributedRoutingTable::RouterRoutingTable& table =
        routing_table->GetRoutingTable(current_nc_id);
    table.routes.resize(nc.GetPortCount());
    const int64_t output_port_count = output_port_ids.size();
    int64_t output_port_index = 0;
    // for each input port
    for (PortId input_port_id : nc.GetInputPortIds()) {
      if (output_port_index == output_port_count) {
        output_port_index = 0;
      }
      PortId output_port_id = output_port_ids[output_port_index];
      XLS_ASSIGN_OR_RETURN(PortParam input_port_param,
                           network_parameters->GetPortParam(input_port_id));
      XLS_ASSIGN_OR_RETURN(PortParam output_port_param,
                           network_parameters->GetPortParam(output_port_id));
      // TODO(vmirian): 2021-10-11 Support other VC mapping strategies.
      int64_t input_port_vc_count = input_port_param.VirtualChannelCount();
      int64_t output_port_vc_count = output_port_param.VirtualChannelCount();

      if (input_port_vc_count != output_port_vc_count) {
        return absl::UnimplementedError(absl::StrFormat(
            "VC route inference is unimplemented "
            " when vc count changes on path between"
            " port %s and port %s",
            input_port_param.GetName(), output_port_param.GetName()));
      }

      if (input_port_param.VirtualChannelCount() == 0 &&
          output_port_param.VirtualChannelCount() == 0) {
        int64_t default_vc = 0;
        table.routes[input_port_id.id()].resize(1);
        table.routes[input_port_id.id()][default_vc].emplace_back(
            destination_index, PortAndVCIndex{output_port_id, default_vc});
      } else {
        // VCs are mapped in-order,
        // ie traffic is rounded from the vc at index 0 to the
        //    vc at index 0 of the next port.
        table.routes[input_port_id.id()].resize(input_port_vc_count);
        for (int64_t i = 0; i < input_port_vc_count; ++i) {
          table.routes[input_port_id.id()][i].emplace_back(
              destination_index, PortAndVCIndex{output_port_id, i});
        }
      }
      output_port_index++;
    }
  }
  return absl::OkStatus();
}

}  // namespace noc
}  // namespace xls
