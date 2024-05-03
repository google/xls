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

#include "xls/noc/simulation/simulator_to_link_monitor_service_shim.h"

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/noc/simulation/common.h"

namespace xls::noc {

NocSimulatorToLinkMonitorServiceShim::NocSimulatorToLinkMonitorServiceShim(
    NocSimulator& simulator)
    : simulator_(simulator) {}

absl::Status NocSimulatorToLinkMonitorServiceShim::RunCycle() {
  // Count the number of packet passing through the link. A tail flit indicates
  // the end of a packet.
  for (const SimLink& link : simulator_.GetLinks()) {
    SimConnectionState& src =
        simulator_.GetSimConnectionByIndex(link.GetSourceConnectionIndex());
    const DataFlit& flit = src.forward_channels.flit;
    if (flit.type == FlitType::kTail) {
      DestinationToPacketCount& destination_to_pkt_count_map =
          link_to_packet_count_map_[link.GetId()];
      destination_to_pkt_count_map[FlitDestination{flit.destination_index,
                                                   flit.vc}]++;
    }
  }
  return absl::OkStatus();
}

const absl::flat_hash_map<NetworkComponentId, DestinationToPacketCount>&
NocSimulatorToLinkMonitorServiceShim::GetLinkToPacketCountMap() const {
  return link_to_packet_count_map_;
}

}  // namespace xls::noc
