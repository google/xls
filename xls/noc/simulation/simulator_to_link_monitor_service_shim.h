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

#ifndef XLS_NOC_SIMULATION_SIMULATOR_TO_LINK_MONITOR_SERVICE_SHIM_H_
#define XLS_NOC_SIMULATION_SIMULATOR_TO_LINK_MONITOR_SERVICE_SHIM_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/sim_objects.h"

namespace xls::noc {

struct FlitDestination {
  int64_t sink_index;
  int64_t vc;

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, FlitDestination flit_destination) {
    return H::combine(std::move(h), flit_destination.sink_index,
                      flit_destination.vc);
  }
};

inline bool operator<(const FlitDestination& lhs, const FlitDestination& rhs) {
  return (lhs.sink_index < rhs.sink_index) ||
         (lhs.sink_index == rhs.sink_index && lhs.vc < rhs.vc);
}

inline bool operator==(const FlitDestination& lhs, const FlitDestination& rhs) {
  return lhs.sink_index == rhs.sink_index && lhs.vc == rhs.vc;
}

using DestinationToPacketCount = absl::flat_hash_map<FlitDestination, int64_t>;

// Shim to collect information from the links of the simulator.
class NocSimulatorToLinkMonitorServiceShim : public NocSimulatorServiceShim {
 public:
  explicit NocSimulatorToLinkMonitorServiceShim(NocSimulator& simulator);
  absl::Status RunCycle() override;

  const absl::flat_hash_map<NetworkComponentId, DestinationToPacketCount>&
  GetLinkToPacketCountMap() const;

 private:
  // Contains the packet count for each destination/vc pair at each link.
  absl::flat_hash_map<NetworkComponentId, DestinationToPacketCount>
      link_to_packet_count_map_;
  // TODO(vmirian) 11-8-21 should be const, but other API require change. So it
  // leave for now.
  NocSimulator& simulator_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_SIMULATOR_TO_LINK_MONITOR_SERVICE_SHIM_H_
