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

#ifndef XLS_NOC_SIMULATION_SAMPLE_NETWORK_GRAPHS_H_
#define XLS_NOC_SIMULATION_SAMPLE_NETWORK_GRAPHS_H_

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/network_graph_builder.h"
#include "xls/noc/simulation/parameters.h"

// This file contains functions used to build sample network graphs.
// These graphs are used within the simulator's unit tests.

namespace xls::noc {

// Builds Sample Linear Network 000
//
//   SendPort0
//       |
//       | L=2
//     Ain0
//  [ RouterA ]
//     Aout0
//       |
//       | L=2
//   RecvPort0
//
absl::Status BuildNetworkGraphLinear000(NetworkConfigProto* nc_proto,
                                        NetworkManager* graph,
                                        NocParameters* params);

// Builds Sample Tree Network 000.
//
//   SendPort0     SendPort1    SendPort2
//     \           /                |
//      \         /                 |
//     Ain0     Ain1                |
//      [ RouterA ]                 |
//     Aout0    Aout1               |
//       |        \                 |
//       |         \----------|     |
//       |                  Bin0   Bin1
//       |                    [    RouterB   ]
//       |                  Bout0  Bout1  Bout2
//       |                   /      |         \
//       |                  /L=2    |          \
//   RecvPort0          RecvPort1   RecvPort2  RecvPort3
absl::Status BuildNetworkGraphTree000(NetworkConfigProto* nc_proto,
                                      NetworkManager* graph,
                                      NocParameters* params);

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_SAMPLE_NETWORK_GRAPHS_H_
