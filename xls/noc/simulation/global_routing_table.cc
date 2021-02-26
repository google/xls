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

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/integral_types.h"
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
                                      int64 initial_vc_index, int64 max_hops) {
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

  int64 current_vc_index = initial_vc_index;
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
      return absl::InternalError(absl::StrFormat(
          "Route from source %x to sink %x exceeded max hops %d",
          source_nc.id().AsUInt64(), sink_nc.id().AsUInt64(), max_hops));
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
          absl::get<NetworkInterfaceSinkParam>(sink_param).GetName(),
          sink.AsUInt64()));
    } else {
      PortId next_port =
          network_manager_->GetConnection(from_port.connection()).sink();
      return PortAndVCIndex{next_port, from.vc_index_};
    }
  } else {
    NetworkComponentId from_nc_id = from.port_id_.GetNetworkComponentId();
    NetworkComponent& from_nc =
        network_manager_->GetNetworkComponent(from_nc_id);
    NetworkComponentKind from_nc_kind = from_nc.kind();

    if (from_nc_kind == NetworkComponentKind::kRouter) {
      return GetRouterOutputPort(from, sink);
    } else if (from_nc_kind == NetworkComponentKind::kLink) {
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

  return absl::NotFoundError(absl::StrFormat(
      "Unable to find hop from port %s (%x) to sink %s (%x)",
      port_param.GetName(), from.port_id_.AsUInt64(),
      absl::get<NetworkInterfaceSinkParam>(sink_param).GetName(),
      sink.AsUInt64()));
}

absl::StatusOr<PortAndVCIndex> DistributedRoutingTable::GetRouterOutputPort(
    PortAndVCIndex from, NetworkComponentId sink) {
  XLS_ASSIGN_OR_RETURN(int64 destination_index,
                       sink_indices_.GetNetworkComponentIndex(sink));
  return GetRouterOutputPortByIndex(from, destination_index);
}

absl::StatusOr<PortAndVCIndex>
DistributedRoutingTable::GetRouterOutputPortByIndex(PortAndVCIndex from,
                                                    int64 destination_index) {
  PortRoutingList& routes = GetRoutingList(from);

  for (std::pair<int64, PortAndVCIndex>& hop : routes) {
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

  return absl::NotFoundError(absl::StrFormat(
      "Unable to find hop from port %s (%x) to sink %s (%x)",
      port_param.GetName(), from.port_id_.AsUInt64(),
      absl::get<NetworkInterfaceSinkParam>(sink_param).GetName(),
      sink.AsUInt64()));
}

void DistributedRoutingTable::AllocateTableForNetwork(NetworkId network_id,
                                                      int64 component_count) {
  int64 network_index = network_id.id();

  if (routing_tables_.size() <= network_index) {
    routing_tables_.resize(network_index + 1);
  }

  routing_tables_[network_index].resize(component_count);
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

absl::Status
DistributedRoutingTableBuilderForTrees::BuildNetworkInterfaceIndices(
    NetworkId network_id, DistributedRoutingTable* routing_table) {
  NetworkComponentIndexMapBuilder source_index_builder;
  NetworkComponentIndexMapBuilder sink_index_builder;

  // This function assigns network interface indices in-order.
  NetworkManager* network_manager = routing_table->network_manager_;
  int64 next_src_index = 0;
  int64 next_sink_index = 0;

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
DistributedRoutingTableBuilderForTrees::BuildPortAndVirtualChannelIndices(
    NetworkId network_id, DistributedRoutingTable* routing_table) {
  PortIndexMapBuilder port_index_builder;
  VirtualChannelIndexMapBuilder vc_index_builder;

  NetworkManager* network_manager = routing_table->network_manager_;
  NocParameters* network_parameters = routing_table->network_parameters_;

  for (NetworkComponent& nc :
       network_manager->GetNetwork(network_id).GetNetworkComponents()) {
    int64 input_port_index = 0;
    int64 output_port_index = 0;

    for (Port& ports : nc.GetPorts()) {
      // This function assigns port indicies in-order.
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

      int64 vc_count = port_param.VirtualChannelCount();

      if (vc_count > 0) {
        for (int64 i = 0; i < vc_count; ++i) {
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

absl::Status DistributedRoutingTableBuilderForTrees::BuildRoutingTable(
    NetworkId network_id, DistributedRoutingTable* routing_table) {
  NetworkManager* network_manager = routing_table->network_manager_;

  int64 component_count =
      network_manager->GetNetwork(network_id).GetNetworkComponentCount();

  routing_table->AllocateTableForNetwork(network_id, component_count);

  const NetworkComponentIndexMap& sink_indices =
      routing_table->GetSinkIndices();

  // Algorithm:
  //  For each sink
  //   Perform DFS to srcs
  //     Record hop needed to reach destination
  for (int64 i = 0; i < sink_indices.NetworkComponentCount(); ++i) {
    XLS_ASSIGN_OR_RETURN(NetworkComponentId sink_id,
                         sink_indices.GetNetworkComponentByIndex(i));

    XLS_RET_CHECK_OK(AddRoutes(i, sink_id, PortId::kInvalid, routing_table));
  }

  return absl::OkStatus();
}

absl::Status DistributedRoutingTableBuilderForTrees::AddRoutes(
    int64 destination_index, NetworkComponentId nc_id, PortId via_port,
    DistributedRoutingTable* routing_table) {
  NetworkManager* network_manager = routing_table->network_manager_;
  NocParameters* network_parameters = routing_table->network_parameters_;

  NetworkComponent& nc = network_manager->GetNetworkComponent(nc_id);

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
        int64 from_port_vc_count = from_port_param.VirtualChannelCount();
        int64 via_port_vc_count = via_port_param.VirtualChannelCount();

        if (from_port_vc_count != via_port_vc_count) {
          return absl::UnimplementedError(absl::StrFormat(
              "VC route inference is unimplemented "
              " when vc count changes on path between"
              " port %s and port %s",
              from_port_param.GetName(), via_port_param.GetName()));
        }

        if (from_port_param.VirtualChannelCount() == 0 &&
            via_port_param.VirtualChannelCount() == 0) {
          int64 default_vc = 0;
          table.routes[from_port.id()].resize(1);
          table.routes[from_port.id()][default_vc].emplace_back(
              destination_index, PortAndVCIndex{via_port, default_vc});
        } else {
          // VCs are mapped in-order,
          // ie traffic is rouded from the vc at index 0 to the
          //    vc at index 0 of the next port.
          // TODO(tedhong): 2020-01-15 Update this to use global virtual
          //                           channels to allow for more flexiblity.
          table.routes[from_port.id()].resize(from_port_vc_count);
          for (int64 i = 0; i < from_port_vc_count; ++i) {
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
                                   prior_port, routing_table));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace noc
}  // namespace xls
