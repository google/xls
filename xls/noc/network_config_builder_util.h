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

// Contains helper functions for network configuration description.

#ifndef XLS_NOC_NETWORK_CONFIG_BUILDER_UTIL_H_
#define XLS_NOC_NETWORK_CONFIG_BUILDER_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"
#include "xls/common/integral_types.h"
#include "xls/noc/network_config.pb.h"
#include "xls/noc/network_config_builder_options.pb.h"

namespace xls::noc {

// Endpoint connection defined by a router index and a port index.
struct EndpointConnection {
  int64 router_idx;
  int64 port_idx;
};

// Endpoint and router connection metadata
struct EndpointRouterConnection {
  // The index of the vector corresponds to the index of the router. The value
  // at the index represents the number of endpoints at the given router.
  std::vector<int64> num_endpoints_from_router_idx;
  // The index of the vector corresponds to the index of the endpoint. The value
  // at the index represents the endpoint connection.
  std::vector<EndpointConnection> endpoint_connection_from_endpoint_idx;
};

// Returns a vector with the number of endpoints per router. The index of the
// vector represents a router. The value of each element is the number of
// endpoints for the given router. The size of the vector is equivalent to the
// number of routers. Prerequisite: 1) Total Endpoints must be greater than
// zero. 2) Maximum number of endpoints per router must be greater than zero.
absl::StatusOr<std::vector<int64>> UniformlyDistributeEndpointsOverRouters(
    int64 total_endpoints, int64 max_num_endpoints_per_router = 1);

// Get the endpoint connection for each endpoint.
// Returns a vector with the endpoint connection for each endpoint. The index of
// the vector represents a endpoint. The value of each element represents the
// endpoint connection for the given endpoint. The size of the vector is
// equivalent to the number of endpoints. Prerequisite: 1) The number of
// neighbours for each router must be greater than zero. 2) The number of
// endpoints for each router must be greater than zero.
absl::StatusOr<std::vector<EndpointConnection>> GetEndpointConnection(
    absl::Span<const int64> num_neighbours_from_router_idx,
    absl::Span<const int64> num_endpoints_from_router_idx);

// Generate the send ports and receive ports for the network.
// Returns the network with the generated send ports and receive ports.
// The the ports in each set (send ports and receive ports) will have a unique
// name and unique ids. Prerequisite: 1) Total send ports must be greater than
// zero. 2) Total receive ports must be greater than zero.
absl::StatusOr<NetworkConfigProto> GenerateEndpointPorts(
    int64 total_send_ports, int64 total_receive_ports,
    NetworkConfigProto& network);

}  // namespace xls::noc

#endif  // XLS_NOC_NETWORK_CONFIG_BUILDER_UTIL_H_
