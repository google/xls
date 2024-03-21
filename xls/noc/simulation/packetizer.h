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

#ifndef XLS_NOC_SIMULATION_PACKETIZER_H_
#define XLS_NOC_SIMULATION_PACKETIZER_H_

#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/sim_objects.h"

// This file contains classes used to represent and build a packet, along
// with auxilary functions to convert packets to flits (depacketize) and
// flits back to packets (packetize).
//
// These classes assume that all flits and phits are of equivalent size.

namespace xls::noc {

// Represents a packet being sent from a source to a sink (forward).
struct DataPacket {
  bool valid;
  int16_t source_index;
  int16_t destination_index;
  int16_t vc;  // Virtual channel.
  Bits data;

  std::string ToString() const {
    return absl::StrFormat(
        "{valid: %d, source_index: %d, dest_index: %d, "
        "vc: %d, data: %v}",
        valid, source_index, destination_index, vc, data);
  }

  // String converter to support absl::StrFormat() and related functions.
  friend absl::FormatConvertResult<absl::FormatConversionCharSet::kString>
  AbslFormatConvert(const DataPacket& packet,
                    const absl::FormatConversionSpec& spec,
                    absl::FormatSink* s) {
    s->Append(packet.ToString());
    return {true};
  }
};

// Fluent builder for a DataPacket.
class DataPacketBuilder {
 public:
  DataPacketBuilder& Valid(bool valid) {
    packet_.valid = valid;
    return *this;
  }

  DataPacketBuilder& DestinationIndex(int16_t dest_index) {
    packet_.destination_index = dest_index;
    return *this;
  }

  DataPacketBuilder& SourceIndex(int16_t source_index) {
    packet_.source_index = source_index;
    return *this;
  }

  DataPacketBuilder& VirtualChannel(int16_t vc) {
    packet_.vc = vc;
    return *this;
  }

  DataPacketBuilder& Data(Bits bits) {
    packet_.data = std::move(bits);
    return *this;
  }

  DataPacketBuilder& ZeroedData(int64_t bit_count) {
    packet_.data = Bits(bit_count);
    return *this;
  }

  absl::StatusOr<DataPacket> Build() { return packet_; }

 private:
  DataPacket packet_ = {
      .valid = false, .source_index = 0, .destination_index = 0, .vc = 0};
};

// Represents a depacketizer that converts a packet into a sequence of flits.
//
// The max_packet_bit_count should be both
//   1) The maximum packet payload accpeted by this DePacketizer
//   2) The maximum packet payload accepted by any destination this
//      source will sent to.
//
// This way, if max_packet_bit_count_  <= flit_payload_bit_count, then
// both the source (this DePacketizer) and destination (a destination
// Packetizer) can assume all packets are single-flits.  And
// all of the flit payload can be used to send data.
//
// Otherwise, we reserve a few bits in the flit payload to store the
// source index so that the receiving Packetizer can associated
// flits coming from the same source together.
class DePacketizer {
 public:
  DePacketizer(int64_t flit_payload_bit_count, int64_t source_index_bit_count,
               int64_t max_packet_bit_count)
      : bits_left_to_send_(0),
        flit_payload_bit_count_(flit_payload_bit_count),
        source_index_bit_count_(source_index_bit_count),
        max_packet_bit_count_(max_packet_bit_count) {}

  // Provides depacketizer the next packet to depacketize.
  //   Returns error if a packet is already being depacketized.
  absl::Status AcceptNewPacket(DataPacket packet) {
    if (!IsIdle()) {
      return absl::InternalError("Packetize already handling a packet");
    }

    packet_ = packet;
    bits_left_to_send_ = packet_.data.bit_count();

    return absl::OkStatus();
  }

  // Returns true if the depacketizer is idle and is not in the midst of
  // handling a packet.
  bool IsIdle() { return bits_left_to_send_ == 0; }

  // Returns next flit to send.
  //
  // If a packet is too large to fit within a single flit, the packet
  // is broken up into multiple flits, with the LSB of the packet's data
  // sent first.
  absl::StatusOr<DataFlit> ComputeNextFlit() {
    if (IsIdle()) {
      return DataFlitBuilder().Invalid().BuildFlit();
    }

    DataFlitBuilder next_flit =
        DataFlitBuilder()
            .VirtualChannel(packet_.vc)
            .SourceIndex(packet_.source_index)
            .DestinationIndex(packet_.destination_index);

    // Reserve a number of bits to send a header.
    // For the simulator, the header is actually not encoded, but only
    // the number of bits able to be sent as data is reduced.
    int64_t reserved_bits = GetReservedFlitPayloadBitCount();
    int64_t max_bits_to_send = flit_payload_bit_count_ - reserved_bits;
    int64_t bits_already_sent = packet_.data.bit_count() - bits_left_to_send_;

    VLOG(1) << absl::StreamFormat(
        "... packetizer reserved %d max_bits_to_send %d already_sent %d left "
        "%d",
        reserved_bits, max_bits_to_send, bits_already_sent, bits_left_to_send_);

    Bits data_to_send(max_bits_to_send);

    if (bits_left_to_send_ <= max_bits_to_send) {
      next_flit.Type(FlitType::kTail);
      data_to_send = packet_.data.Slice(bits_already_sent, bits_left_to_send_);
      bits_left_to_send_ = 0;
    } else {
      next_flit.Type(bits_already_sent == 0 ? FlitType::kHead
                                            : FlitType::kBody);
      data_to_send = packet_.data.Slice(bits_already_sent, max_bits_to_send);
      bits_left_to_send_ -= max_bits_to_send;
    }

    next_flit.Data(data_to_send);

    return next_flit.BuildFlit();
  }

  int64_t GetFlitPayloadBitCount() const { return flit_payload_bit_count_; }

  int64_t GetSourceIndexBitCount() const { return source_index_bit_count_; }

  int64_t GetMaxPacketBitCount() const { return max_packet_bit_count_; }

 private:
  // Returns the amount of flit payload used for packetization.
  //
  // Given a maximum flit payload and the max packet data size, there are
  // two possibilities.
  //  1. If the packet data can all fit within a flit then the
  //     format is a single flit.  No additional bits are needed to reserve
  //     within a flit.
  //
  //  2. Otherwise -- part of the payload is reserved for the packet source id
  //     (for the flit) or the used bit count (for the tail flit).
  int64_t GetReservedFlitPayloadBitCount() {
    return (flit_payload_bit_count_ >= max_packet_bit_count_)
               ? 0
               : source_index_bit_count_;
  }

  // Current packet that is being sent into the network.
  DataPacket packet_;

  // Number of data bits left to send.
  // Once this reaches 0, the packetizer can accept a new packet.
  int64_t bits_left_to_send_;

  // Represents the number of bits that can be sent with a flit.
  // Usage of this is split between
  //   1. data
  //   2. identification of which packet the flit is for (ex.
  //      source node id, squence number, etc ...)
  int64_t flit_payload_bit_count_;

  // Number of bits needed to represent the source.
  int64_t source_index_bit_count_;

  // Maximum size of the packet.
  int64_t max_packet_bit_count_;
};

// Represents a packetizer that converts a sequence of flits into a packet.
//
// See DePacketizer for a description of how source_index_bit_count,
// and max_packet_bit_count are used to determine whether the Packetizer
// can asusme that all flits received are single-flit packets.
//
// max_outstanding_packets is used to determine how many partial packets
// can be handled at any given point.  It should be sized to guarantee
// that any flit can be received can be accepted by the packetizer.
//
// TODO(tedhong): 2020-04-06 Add support for packet-level credit-based
// flow control.
class Packetizer {
 public:
  Packetizer(int64_t flit_payload_bit_count, int64_t source_index_bit_count,
             int64_t max_packet_bit_count, int64_t max_outstanding_packets)
      : flit_payload_bit_count_(flit_payload_bit_count),
        source_index_bit_count_(source_index_bit_count),
        max_packet_bit_count_(max_packet_bit_count) {
    partial_packets_.resize(max_outstanding_packets);
  }

  // Returns count of packets this packetizer has only partially received.
  int64_t PartialPacketCount() {
    int64_t count = 0;

    for (int64_t i = 0; i < partial_packets_.size(); ++i) {
      count += !partial_packets_[i].empty();
    }

    return count;
  }

  // Provides packetizer the next flit to packetize.
  //   Returns error if a packet is already being packetized.
  absl::Status AcceptNewFlit(DataFlit flit);

  // Retreives list of received packets;
  absl::Span<const DataPacket> GetPackets() {
    return absl::MakeSpan(received_packets_);
  }

 private:
  // Allocates or retrieves a new vector to store flits received from a source.
  absl::StatusOr<std::vector<Bits>*> AllocateOrRetrievePartialPacketStore(
      int64_t source_index);

  // Clears storage initialized from AllocateOrRetreivePartialPacketStore.
  void DeallocatePartialPacketStore(int64_t source_index);

  // Returns the amount of flit payload used for packetization.
  //
  // Given a maximum flit payload and the max packet data size, there are
  // two possibilities.
  //  1. If the packet data can all fit within a flit then the
  //     format is a single flit.  No additional bits are needed to reserve
  //     within a flit.
  //
  //  2. Otherwise -- part of the payload is reserved for the packet source id
  //     and/or sequence number (if used).
  int64_t GetReservedFlitPayloadBitCount() {
    return (flit_payload_bit_count_ >= max_packet_bit_count_)
               ? 0
               : source_index_bit_count_;
  }

  // Represents the number of bits that can be sent with a flit.
  // Usage of this is split between
  //   1. data
  //   2. identification of which packet the flit is for (ex.
  //      source node id, squence number, etc ...)
  int64_t flit_payload_bit_count_;

  // Number of bits needed to represent the source.
  int64_t source_index_bit_count_;

  // Maximum size of the packet.
  int64_t max_packet_bit_count_;

  // Flit buffer for packets that are in the midst of being received.
  std::vector<std::vector<Bits>> partial_packets_;

  // The packetizer uses partial_packet_index_ to index into partial_packets_.
  //
  // Each flit from a single source arrives in order, so to a depacketizer,
  // there can only be one outstanding packet per source.  For this,
  // partial_packet_index_map_[ <source_index> ] = Y, where Y is used
  // to index into partial_packets_ to obtain the flits for the packet
  // being received.
  absl::flat_hash_map<int64_t, int64_t> partial_packet_index_map_;

  // List of packets in the order of being received.
  std::vector<DataPacket> received_packets_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_PACKETIZER_H_
