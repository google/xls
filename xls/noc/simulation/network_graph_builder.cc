// Copyright 2020 The XLS Authors
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

#include "xls/noc/simulation/network_graph_builder.h"

#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/parameters.h"

namespace xls {
namespace noc {

namespace {

// Helper class to group related functions used to help build
// a network graph from its proto description.
class NetworkGraphBuilderImpl {
 public:
  // Initializes the builder.
  // The constructor is expected to be passed
  // NetworkManager and NocParameters
  NetworkGraphBuilderImpl(NetworkManager* mgr, NocParameters* params)
      : mgr_(mgr), params_(params) {}

  // Adds a NetworkGraph to the NetworkManager given a proto description.
  absl::Status BuildNetworkGraph(const NetworkConfigProto& proto) {
    // Delegate to BuildNetwork for now.
    return BuildNetwork(proto);
  }

 private:
  // Implementation of to build a NetworkGraph given a proto description.
  absl::Status BuildNetwork(const NetworkConfigProto& network);

  // Builds a specific router given proto.
  absl::Status BuildRouter(NetworkId network_id,
                           const NetworkConfigProto& network,
                           const RouterConfigProto& router);

  // Builds a specific port of a router given proto.
  absl::Status BuildRouterPort(NetworkComponentId component_id,
                               const NetworkConfigProto& network,
                               const PortConfigProto& port);

  // Builds a link and associated connections to ports given proto.
  //   Should be called after ports link is assoicated with are built.
  absl::Status BuildLink(NetworkId network_id,
                         const NetworkConfigProto& network,
                         const LinkConfigProto& link);

  // Builds a network interface given proto.
  absl::Status BuildNetworkInterface(NetworkId network_id,
                                     const NetworkConfigProto& network,
                                     const PortConfigProto& port);

  // Memoizes mapping from proto naming of ports to PortIds.
  void MemoizePortNameToId(const std::string& name, PortId id);

  // Retreives PortId given the proto naming of a port.
  absl::StatusOr<PortId> GetPortIdFromName(const std::string& name);

  // External reference to NetworkGraph that will be built.
  NetworkManager* mgr_;

  // External reference to object mapping graph objects to proto objects.
  NocParameters* params_;

  // Map from proto port names to PortIds used in the NetworkGraph.
  absl::flat_hash_map<std::string, PortId> network_ports_;
};

absl::StatusOr<PortConfigProto::Direction> GetPortConfigProtoDirection(
    const PortConfigProto& port, std::string_view msg) {
  PortConfigProto::Direction dir = port.direction();

  if (dir == PortConfigProto::INPUT || dir == PortConfigProto::OUTPUT) {
    return dir;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("%s PortConfigProto %s "
                      "must have a direction field set.",
                      msg, port.name()));
}

absl::Status NetworkGraphBuilderImpl::BuildNetwork(
    const NetworkConfigProto& network) {
  NetworkParam network_param(network);
  XLS_ASSIGN_OR_RETURN(NetworkId network_id, mgr_->CreateNetwork());
  params_->SetNetworkParam(network_id, network_param);

  for (const PortConfigProto& port : network.ports()) {
    XLS_RETURN_IF_ERROR(BuildNetworkInterface(network_id, network, port));
  }

  for (const RouterConfigProto& router : network.routers()) {
    XLS_RETURN_IF_ERROR(BuildRouter(network_id, network, router));
  }

  // Add links
  for (const LinkConfigProto& link : network.links()) {
    XLS_RETURN_IF_ERROR(BuildLink(network_id, network, link));
  }

  return absl::OkStatus();
}

absl::Status NetworkGraphBuilderImpl::BuildRouter(
    NetworkId network_id, const NetworkConfigProto& network,
    const RouterConfigProto& router) {
  XLS_ASSIGN_OR_RETURN(
      NetworkComponentId component_id,
      mgr_->CreateNetworkComponent(network_id, NetworkComponentKind::kRouter));
  params_->SetNetworkComponentParam(component_id, RouterParam(network, router));

  for (const PortConfigProto& port : router.ports()) {
    XLS_RETURN_IF_ERROR(BuildRouterPort(component_id, network, port));
  }

  return absl::OkStatus();
}

absl::Status NetworkGraphBuilderImpl::BuildRouterPort(
    NetworkComponentId component_id, const NetworkConfigProto& network,
    const PortConfigProto& port) {
  XLS_ASSIGN_OR_RETURN(PortConfigProto::Direction proto_direction,
                       GetPortConfigProtoDirection(port, "Router Port"));
  PortDirection port_direction = (proto_direction == PortConfigProto::INPUT)
                                     ? PortDirection::kInput
                                     : PortDirection::kOutput;
  XLS_ASSIGN_OR_RETURN(PortId port_id,
                       mgr_->CreatePort(component_id, port_direction));
  params_->SetPortParam(port_id, PortParam(network, port));
  MemoizePortNameToId(port.name(), port_id);

  return absl::OkStatus();
}

absl::Status NetworkGraphBuilderImpl::BuildLink(
    NetworkId network_id, const NetworkConfigProto& network,
    const LinkConfigProto& link) {
  XLS_ASSIGN_OR_RETURN(
      NetworkComponentId component_id,
      mgr_->CreateNetworkComponent(network_id, NetworkComponentKind::kLink));
  params_->SetNetworkComponentParam(component_id, LinkParam(network, link));

  // Create ports and create connections to the other side of the link.
  ConnectionId conn_id;

  // The source port in the proto is the other side of the link's input
  //   src_port_id => link_input_port
  XLS_ASSIGN_OR_RETURN(PortId src_port_id,
                       GetPortIdFromName(link.source_port_name()));
  XLS_ASSIGN_OR_RETURN(PortId link_input_id,
                       mgr_->CreatePort(component_id, PortDirection::kInput));
  XLS_ASSIGN_OR_RETURN(
      conn_id, mgr_->CreateConnection(network_id, src_port_id, link_input_id));
  // Also associate the link's port with the same proto as the router.
  XLS_ASSIGN_OR_RETURN(PortParam port_param,
                       params_->GetPortParam(src_port_id));
  params_->SetPortParam(link_input_id,
                        PortParam(network, port_param.GetPortProto()));

  // The sink port in the proto is the other side of the link's output
  //   link_output_port => sink_port_id
  XLS_ASSIGN_OR_RETURN(PortId link_output_id,
                       mgr_->CreatePort(component_id, PortDirection::kOutput));
  XLS_ASSIGN_OR_RETURN(PortId sink_port_id,
                       GetPortIdFromName(link.sink_port_name()));
  XLS_ASSIGN_OR_RETURN(conn_id, mgr_->CreateConnection(
                                    network_id, link_output_id, sink_port_id));
  // Also associate the link's port with the same proto as the router.
  XLS_ASSIGN_OR_RETURN(port_param, params_->GetPortParam(sink_port_id));
  params_->SetPortParam(link_output_id,
                        PortParam(network, port_param.GetPortProto()));

  return absl::OkStatus();
}

absl::Status NetworkGraphBuilderImpl::BuildNetworkInterface(
    NetworkId network_id, const NetworkConfigProto& network,
    const PortConfigProto& port) {
  XLS_ASSIGN_OR_RETURN(PortConfigProto::Direction proto_direction,
                       GetPortConfigProtoDirection(port, "Network top-level"));

  NetworkComponentKind nc_kind;
  PortDirection port_direction;
  NetworkComponentParam network_component_param =
      NetworkInterfaceSrcParam(network, port);

  if (proto_direction == PortConfigProto::INPUT) {
    nc_kind = NetworkComponentKind::kNISrc;
    port_direction = PortDirection::kOutput;
    network_component_param = NetworkInterfaceSrcParam(network, port);
  } else {
    nc_kind = NetworkComponentKind::kNISink;
    port_direction = PortDirection::kInput;
    network_component_param = NetworkInterfaceSinkParam(network, port);
  }

  // Create an implicit network interface.
  XLS_ASSIGN_OR_RETURN(NetworkComponentId component_id,
                       mgr_->CreateNetworkComponent(network_id, nc_kind));
  params_->SetNetworkComponentParam(component_id, network_component_param);

  // Network interfaces will also have a port
  XLS_ASSIGN_OR_RETURN(PortId port_id,
                       mgr_->CreatePort(component_id, port_direction));
  PortParam port_param(network, port);
  params_->SetPortParam(port_id, port_param);
  MemoizePortNameToId(port.name(), port_id);

  return absl::OkStatus();
}

void NetworkGraphBuilderImpl::MemoizePortNameToId(const std::string& name,
                                                  PortId id) {
  network_ports_[name] = id;
}

absl::StatusOr<PortId> NetworkGraphBuilderImpl::GetPortIdFromName(
    const std::string& name) {
  if (network_ports_.contains(name)) {
    return network_ports_[name];
  }

  return absl::InternalError(
      absl::StrFormat("Expected to find %s in network port mapping.", name));
}

}  // namespace

absl::Status BuildNetworkGraphFromProto(const NetworkConfigProto& proto,
                                        NetworkManager* network_manager,
                                        NocParameters* parameters) {
  if (network_manager == nullptr) {
    return absl::InternalError("Expected non-null network_manager.");
  }
  if (parameters == nullptr) {
    return absl::InternalError("Expected non-null parameters.");
  }

  NetworkGraphBuilderImpl builder(network_manager, parameters);
  return builder.BuildNetworkGraph(proto);
}

absl::StatusOr<NetworkComponentId> FindNetworkComponentByName(
    std::string_view name, NetworkManager& network_mgr,
    NocParameters& noc_parameters) {
  for (Network& network : network_mgr.GetNetworks()) {
    for (NetworkComponentId nc : network.GetNetworkComponentIds()) {
      absl::StatusOr<NetworkComponentParam> param_status =
          noc_parameters.GetNetworkComponentParam(nc);

      if (param_status.ok()) {
        std::string_view nc_name =
            absl::visit([](auto nc) { return nc.GetName(); }, *param_status);
        if (nc_name == name) {
          return nc;
        }
      }
    }
  }

  return absl::NotFoundError(absl::StrFormat(
      "Unable to find network component with param name %s", name));
}

absl::StatusOr<PortId> FindPortByName(std::string_view name,
                                      NetworkManager& network_mgr,
                                      NocParameters& noc_parameters) {
  for (Network& network : network_mgr.GetNetworks()) {
    for (NetworkComponent& nc : network.GetNetworkComponents()) {
      for (PortId port : nc.GetPortIds()) {
        absl::StatusOr<PortParam> param_status =
            noc_parameters.GetPortParam(port);

        if (param_status.ok() && param_status->GetName() == name) {
          return port;
        }
      }
    }
  }

  return absl::NotFoundError(
      absl::StrFormat("Unable to find port with param name %s", name));
}

}  // namespace noc
}  // namespace xls
