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

#ifndef XLS_NOC_SIMULATION_NETWORK_GRAPH_BUILDER_H_
#define XLS_NOC_SIMULATION_NETWORK_GRAPH_BUILDER_H_

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/parameters.h"

namespace xls {
namespace noc {

// Builds a simulation NetworkGraph from proto.
//   proto - Input - Validated protos describing a single network.
//   manager - Output - Manager of the network graphs.
//                    - Upon return, a new network will be added
//                      to the manager.
//   parameters - Output - Mapping from network graph objects to param objects.
//                       - Upon return, parameters will be updated with
//                         additional Param objects mapping the newly
//                         created network to proto.
//
// The expected usage of this function is to create basic objects
// by the simulator to describe the connectivity of the network
// (see xls::noc::NetworkManager in network_graph.h) and
// associate those objects with their configuration protos
// (see xls::noc::NocParameters in parameters.h).
absl::Status BuildNetworkGraphFromProto(const NetworkConfigProto& proto,
                                        NetworkManager* manager,
                                        NocParameters* parameters);

// Retrieves NetworkComponentId, given name as specified in proto.
absl::StatusOr<NetworkComponentId> FindNetworkComponentByName(
    std::string_view name, NetworkManager& network_mgr,
    NocParameters& noc_parameters);

// Retrieves PortId, given name as specified in proto.
absl::StatusOr<PortId> FindPortByName(std::string_view name,
                                      NetworkManager& network_mgr,
                                      NocParameters& noc_parameters);

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_NETWORK_GRAPH_BUILDER_H_
