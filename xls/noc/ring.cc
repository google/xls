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

// Contains the ring network configuration builder.

#include "google/protobuf/text_format.h"
#include "absl/strings/str_format.h"
#include "xls/common/math_util.h"
#include "xls/noc/network_config_builder.h"
#include "xls/noc/network_config_builder_factory.h"
#include "xls/noc/network_config_builder_options.pb.h"
#include "xls/noc/network_config_builder_util.h"

namespace xls::noc {

// TODO(vmirian): make kMaxNumEndpointsPerRouter a Ring Config Option
constexpr int64 kMaxNumEndpointsPerRouter = 1;
constexpr int64 kUpstreamPortIdx = 0;
constexpr int64 kDownstreamPortIdx = 1;
constexpr int64 kNumRouterNeighbours = 2;

REGISTER_IN_NETWORK_CONFIG_BUILDER_FACTORY(ring, Ring)

absl::Status Ring::ValidateArguments(
    const NetworkConfigBuilderOptions& options) const {
  if (options.options().num_send_ports() !=
      options.options().num_recv_ports()) {
    return absl::InvalidArgumentError(
        "For ring network builder, "
        "the number of send ports must equal "
        "to the number of receive ports.");
  }
  if (options.options().num_send_ports() < 3) {
    return absl::InvalidArgumentError(
        "For ring network builder, "
        "the number of send ports must be "
        "greater than two.");
  }
  if (options.CustomNetworkConfigBuilderOptions_case() != options.kRing) {
    return absl::InvalidArgumentError(
        "For ring network builder, "
        "the custom options must use the "
        "RingNetworkOptions message format.");
  }
  return absl::OkStatus();
}

// Generate routers and their ports for the network using the endpoint router
// connection metadata.
static NetworkConfigProto& GenerateRouters(
    absl::Span<const int64> num_endpoints_from_router_idx,
    NetworkConfigProto& network) {
  const int64 total_routers = num_endpoints_from_router_idx.size();
  for (int64 router_idx = 0; router_idx < total_routers; router_idx++) {
    RouterConfigProto* router = network.add_routers();
    router->set_name(absl::StrFormat("Router.%d", router_idx));
    // Add the input and output ports to each router. The total number of ports
    // is equivalent to the number of endpoints for the router plus the number
    // of neighbours for the router.
    // Note that the number of ports for neighbouring routers is defined by
    // kNumRouterNeighbours.
    for (int64 port_count = 0;
         port_count <
         num_endpoints_from_router_idx[router_idx] + kNumRouterNeighbours;
         port_count++) {
      PortConfigProto* input_port = router->add_input_ports();
      PortConfigProto* output_port = router->add_output_ports();
      input_port->set_name(
          absl::StrFormat("%s.input.port.%d", router->name(), port_count));
      input_port->set_id(port_count);
      output_port->set_name(
          absl::StrFormat("%s.output.port.%d", router->name(), port_count));
      output_port->set_id(port_count);
    }
  }
  return network;
}

// Generate the connections for the endpoint ports of the network.
static NetworkConfigProto& GenerateConnectionsRouterNeighbours(
    NetworkConfigProto& network) {
  const int64 total_routers = network.routers_size();
  ConnectionConfigProto* connection;
  // connect router ports to their respective neighbor routers
  for (int64 router_idx = 0; router_idx < total_routers; router_idx++) {
    // An upstream connection is a connection to the 'left' of the router.
    // The router's upstream output port is connected to the input port of the
    // 'left' router.
    connection = network.add_connections();
    connection->set_source_port_name(
        network.routers(router_idx).output_ports(kUpstreamPortIdx).name());
    connection->set_sink_port_name(
        network.routers((router_idx + total_routers - 1) % total_routers)
            .input_ports(kDownstreamPortIdx)
            .name());
    // A downstream connection is a connection to the 'right' of the router.
    // The router's downstream output port is connected to the input port of the
    // 'right' router.
    connection = network.add_connections();
    connection->set_source_port_name(
        network.routers(router_idx).output_ports(kDownstreamPortIdx).name());
    connection->set_sink_port_name(
        network.routers((router_idx + total_routers + 1) % total_routers)
            .input_ports(kUpstreamPortIdx)
            .name());
  }
  return network;
}

// Generate the connections for the router ports to/from their neighbour
// in the network.
static NetworkConfigProto& GenerateConnectionsRouterEndpoints(
    absl::Span<const EndpointConnection> endpoint_connection_from_endpoint_idx,
    NetworkConfigProto& network) {
  const int64 total_endpoints = endpoint_connection_from_endpoint_idx.size();
  ConnectionConfigProto* connection;
  // connect router ports to their respective endpoint ports
  for (int64 endpoint_idx = 0; endpoint_idx < total_endpoints; endpoint_idx++) {
    const EndpointConnection& endpoint_connection =
        endpoint_connection_from_endpoint_idx[endpoint_idx];
    // The endpoint send port is connected to an input port of the router.
    connection = network.add_connections();
    connection->set_source_port_name(network.send_ports(endpoint_idx).name());
    connection->set_sink_port_name(
        network.routers(endpoint_connection.router_idx)
            .input_ports(endpoint_connection.port_idx)
            .name());
    // An output port of the router is connected to the endpoint recv port.
    connection = network.add_connections();
    connection->set_source_port_name(
        network.routers(endpoint_connection.router_idx)
            .output_ports(endpoint_connection.port_idx)
            .name());
    connection->set_sink_port_name(network.recv_ports(endpoint_idx).name());
  }
  return network;
}
// Generate routing table for routers in network
static NetworkConfigProto& GenerateRoutingTable(
    absl::Span<const EndpointConnection> endpoint_connection_from_endpoint_idx,
    NetworkConfigProto& network) {
  const int64 total_routers = network.routers_size();
  const int64 total_endpoints = endpoint_connection_from_endpoint_idx.size();
  const int64 num_routers_for_upstream = xls::CeilOfRatio(
      static_cast<int64>(total_routers - 1), static_cast<int64>(2));
  ConnectionConfigProto* connection;
  for (int64 router_idx = 0; router_idx < total_routers; router_idx++) {
    RouterConfigProto* router = network.mutable_routers(router_idx);
    const std::string& upstream_port_name =
        router->output_ports(kUpstreamPortIdx).name();
    const std::string& downstream_port_name =
        router->output_ports(kDownstreamPortIdx).name();
    for (int64 recv_port_idx = 0; recv_port_idx < total_endpoints;
         recv_port_idx++) {
      // Create a routing table entry for each recv port.
      connection = router->add_routing_table();
      const EndpointConnection& endpoint_connection =
          endpoint_connection_from_endpoint_idx[recv_port_idx];
      connection->set_source_port_name(
          network.recv_ports(recv_port_idx).name());
      // The number of routers between the destination router and the given
      // router.
      int64 diff = endpoint_connection.router_idx - router_idx;
      if (diff == 0) {
        // The destination port for an endpoint at the given router is
        // the output port connected to the receive port.
        connection->set_sink_port_name(
            router->output_ports(endpoint_connection.port_idx).name());
      } else {
        if (diff < 0) {
          diff += total_routers;
        }
        if (diff <= num_routers_for_upstream) {
          // downstream
          connection->set_sink_port_name(downstream_port_name);
        } else {
          // upstream
          connection->set_sink_port_name(upstream_port_name);
        }
      }
    }
  }
  return network;
}

absl::StatusOr<NetworkConfigProto> Ring::GenerateNetworkConfig(
    const NetworkConfigBuilderOptions& options) const {
  NetworkConfigProto network;
  int64 total_send_ports = options.options().num_send_ports();
  int64 total_recv_ports = options.options().num_recv_ports();
  XLS_ASSIGN_OR_RETURN(std::vector<int64> num_endpoints_from_router_idx,
                       UniformlyDistributeEndpointsOverRouters(
                           total_send_ports, kMaxNumEndpointsPerRouter));
  std::vector<int64> num_neighbours_from_router_idx;
  num_neighbours_from_router_idx.resize(num_endpoints_from_router_idx.size());
  std::fill(num_neighbours_from_router_idx.begin(),
            num_neighbours_from_router_idx.end(), kNumRouterNeighbours);
  XLS_ASSIGN_OR_RETURN(
      std::vector<EndpointConnection> endpoint_connection_from_endpoint_idx,
      GetEndpointConnection(num_neighbours_from_router_idx,
                            num_endpoints_from_router_idx));
  XLS_ASSIGN_OR_RETURN(
      network,
      GenerateEndpointPorts(total_send_ports, total_recv_ports, network));
  network = GenerateRouters(num_endpoints_from_router_idx, network);
  network = GenerateConnectionsRouterNeighbours(network);
  network = GenerateConnectionsRouterEndpoints(
      endpoint_connection_from_endpoint_idx, network);
  network =
      GenerateRoutingTable(endpoint_connection_from_endpoint_idx, network);
  return network;
}

std::string Ring::GetUsage() const {
  std::string usage;
  RingNetworkConfigOptions proto;
  google::protobuf::TextFormat::PrintToString(proto, &usage);
  return usage;
}

}  // namespace xls::noc
