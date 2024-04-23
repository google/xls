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

#include "xls/noc/simulation/packetizer.h"

#include <cstdint>
#include <vector>

namespace xls::noc {

absl::StatusOr<std::vector<Bits>*>
Packetizer::AllocateOrRetrievePartialPacketStore(int64_t source_index) {
  auto source_index_and_store_index_iter =
      partial_packet_index_map_.find(source_index);

  if (source_index_and_store_index_iter != partial_packet_index_map_.end()) {
    return &partial_packets_.at(source_index_and_store_index_iter->second);
  }

  int64_t index = -1;
  for (int64_t i = 0; i < partial_packets_.size(); ++i) {
    if (partial_packets_[i].empty()) {
      index = i;
      break;
    }
  }

  if (index == -1) {
    return absl::InternalError(
        "Depacketizer unable to "
        "allocate buffer for next packet");
  }

  partial_packet_index_map_[source_index] = index;
  return &partial_packets_.at(index);
}

void Packetizer::DeallocatePartialPacketStore(int64_t source_index) {
  int64_t partial_packet_index = partial_packet_index_map_.at(source_index);
  partial_packets_.at(partial_packet_index).clear();
  partial_packet_index_map_.erase(partial_packet_index);
}

absl::Status Packetizer::AcceptNewFlit(DataFlit flit) {
  XLS_ASSIGN_OR_RETURN(std::vector<Bits> * partial_packet_store,
                       AllocateOrRetrievePartialPacketStore(flit.source_index));

  partial_packet_store->push_back(flit.data);

  if (flit.type == FlitType::kTail) {
    // Concat all received flits together.
    int64_t packet_bit_count = 0;
    for (Bits& b : *partial_packet_store) {
      packet_bit_count += b.bit_count();
    }

    BitsRope received_data(packet_bit_count);
    for (Bits& b : *partial_packet_store) {
      VLOG(1) << absl::StreamFormat(
          "... packetizer concat %s",
          BitsToString(b, FormatPreference::kBinary, true));
      received_data.push_back(b);
    }

    // We've received the tail, so the packet is complete and
    // can be received.
    DeallocatePartialPacketStore(flit.source_index);

    XLS_ASSIGN_OR_RETURN(DataPacket received_packet,
                         DataPacketBuilder()
                             .Data(received_data.Build())
                             .VirtualChannel(flit.vc)
                             .SourceIndex(flit.source_index)
                             .DestinationIndex(flit.destination_index)
                             .Valid(true)
                             .Build());

    received_packets_.push_back(received_packet);
  }

  return absl::OkStatus();
}

}  // namespace xls::noc
