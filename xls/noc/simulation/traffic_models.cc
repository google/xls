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

#include "xls/noc/simulation/traffic_models.h"

namespace xls::noc {

std::vector<DataPacket> GeneralizedGeometricTrafficModel::GetNewCyclePackets(
    int64_t cycle) {
  if (next_packet_cycle_ > cycle) {
    // No packets to be sent until next_packet_cycle_.
    return std::vector<DataPacket>();
  }

  std::vector<DataPacket> packets;

  if (next_packet_cycle_ == -1) {
    // Packet sent on cycle 0 will only be due to a burst.
    next_packet_cycle_ = 0;
  } else {
    // Except for the initial packet.
    // We expect GetNewCyclePackets to be called without skipping
    // a cycle in which a packet would be sent.
    XLS_CHECK_EQ(cycle, next_packet_cycle_);
    packets.push_back(next_packet_);
  }

  // See if we have a burst of packets.
  int64_t next_packet_delta =
      random_interface_->GeneralizedGeometric(lambda_, burst_prob_);
  while (next_packet_delta == 0) {
    // New burst packet.  Rest of the packet fields other than size
    // will be filled later.
    absl::StatusOr<DataPacket> burst_packet =
        DataPacketBuilder().Valid(true).ZeroedData(packet_size_bits_).Build();
    XLS_CHECK(burst_packet.ok());
    packets.push_back(burst_packet.value());

    next_packet_delta =
        random_interface_->GeneralizedGeometric(lambda_, burst_prob_);
  }

  absl::StatusOr<DataPacket> future_packet =
      DataPacketBuilder().Valid(true).ZeroedData(packet_size_bits_).Build();
  XLS_CHECK(future_packet.ok());
  next_packet_ = future_packet.value();
  next_packet_cycle_ += next_packet_delta;

  return packets;
}

}  // namespace xls::noc
