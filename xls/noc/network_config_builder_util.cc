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

#include "xls/noc/network_config_builder_util.h"

#include "absl/strings/str_format.h"
#include "xls/common/math_util.h"

namespace xls::noc {

absl::StatusOr<std::vector<int64>> UniformlyDistributeEndpointsOverRouters(
    int64 total_endpoints, int64 max_num_endpoints_per_router) {
  int64 total_routers =
      xls::CeilOfRatio(total_endpoints, max_num_endpoints_per_router);
  // Prerequisite check
  if (total_endpoints < 1) {
    return absl::InvalidArgumentError(
        "Total Endpoints must be greater than zero.");
  }
  if (max_num_endpoints_per_router < 1) {
    return absl::InvalidArgumentError(
        "Maximum number of endpoints per router must be greater than zero.");
  }

  std::vector<int64> num_endpoints_from_router_idx;
  num_endpoints_from_router_idx.reserve(total_routers);
  // Uniformly distribute the endpoints amongst the routers.
  // Assign a remaining router using the number of remaining endpoints and
  // the number of remaining routers.
  while (total_routers > 0) {
    int64 endpoints_for_current_router =
        xls::CeilOfRatio(total_endpoints, total_routers);
    num_endpoints_from_router_idx.emplace_back(endpoints_for_current_router);
    total_routers--;
    total_endpoints -= endpoints_for_current_router;
  }
  return num_endpoints_from_router_idx;
}

absl::StatusOr<std::vector<EndpointConnection>> GetEndpointConnection(
    absl::Span<const int64> num_neighbours_from_router_idx,
    absl::Span<const int64> num_endpoints_from_router_idx) {
  int64 total_routers = num_neighbours_from_router_idx.size();
  // Prerequisite check
  if (total_routers != num_endpoints_from_router_idx.size()) {
    return absl::FailedPreconditionError(
        "The number of routers differ between two lists: 1) the number of "
        "neighbours per router, and, 2) the number of endpoints per router.");
  }
  int64 total_endpoints = 0;
  for (int64 router_idx = 0; router_idx < total_routers; router_idx++) {
    if (num_neighbours_from_router_idx[router_idx] < 1) {
      return absl::FailedPreconditionError(
          "The number of neighbours per router must be greater than zero.");
    }
    if (num_endpoints_from_router_idx[router_idx] < 1) {
      return absl::FailedPreconditionError(
          "The number of endpoints per router must be greater than zero.");
    }
    total_endpoints += num_endpoints_from_router_idx[router_idx];
  }
  std::vector<EndpointConnection> endpoint_connections;
  endpoint_connections.reserve(total_endpoints);
  for (int64 router_idx = 0; router_idx < num_endpoints_from_router_idx.size();
       router_idx++) {
    int64 num_neighbours = num_neighbours_from_router_idx[router_idx];
    for (int64 port_idx = num_neighbours;
         port_idx < num_endpoints_from_router_idx[router_idx] + num_neighbours;
         port_idx++) {
      endpoint_connections.emplace_back(
          EndpointConnection{router_idx, port_idx});
    }
  }
  return endpoint_connections;
}

absl::StatusOr<NetworkConfigProto> GenerateEndpointPorts(
    int64 total_send_ports, int64 total_receive_ports,
    NetworkConfigProto& network) {
  // Prerequisite check
  if (total_send_ports < 1) {
    return absl::InvalidArgumentError(
        "Total send ports must be greater than zero.");
  }
  if (total_receive_ports < 1) {
    return absl::InvalidArgumentError(
        "Total send ports must be greater than zero.");
  }
  // send ports
  for (int64 count = 0; count < total_send_ports; count++) {
    PortConfigProto* port = network.add_send_ports();
    port->set_name(absl::StrFormat("Send.Endpoint.%d", count));
    port->set_id(count);
  }
  // receive ports
  for (int64 count = 0; count < total_receive_ports; count++) {
    PortConfigProto* port = network.add_recv_ports();
    port->set_name(absl::StrFormat("Recv.Endpoint.%d", count));
    port->set_id(count);
  }
  return network;
}

}  // namespace xls::noc
