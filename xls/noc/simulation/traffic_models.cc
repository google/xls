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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"

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
    CHECK_EQ(cycle, next_packet_cycle_);
    packets.push_back(next_packet_);
  }

  // See if we have a burst of packets.
  int64_t next_packet_delta =
      random_interface_->GeneralizedGeometric(lambda_, burst_prob_);
  while (next_packet_delta == 0) {
    // New burst packet.  Rest of the packet fields other than size
    // will be filled later.
    absl::StatusOr<DataPacket> burst_packet =
        DataPacketBuilder()
            .Valid(true)
            .ZeroedData(packet_size_bits_)
            .VirtualChannel(vc_)
            .SourceIndex(source_index_)
            .DestinationIndex(destination_index_)
            .Build();
    CHECK(burst_packet.ok());
    packets.push_back(burst_packet.value());

    next_packet_delta =
        random_interface_->GeneralizedGeometric(lambda_, burst_prob_);
  }

  absl::StatusOr<DataPacket> future_packet =
      DataPacketBuilder()
          .Valid(true)
          .ZeroedData(packet_size_bits_)
          .VirtualChannel(vc_)
          .SourceIndex(source_index_)
          .DestinationIndex(destination_index_)
          .Build();
  CHECK(future_packet.ok());
  next_packet_ = future_packet.value();
  next_packet_cycle_ += next_packet_delta;

  return packets;
}

GeneralizedGeometricTrafficModelBuilder::
    GeneralizedGeometricTrafficModelBuilder(double lambda, double burst_prob,
                                            int64_t packet_size_bits,
                                            RandomNumberInterface& rnd)
    : lambda_(lambda), burst_prob_(burst_prob), random_interface_(&rnd) {
  SetPacketSizeBits(packet_size_bits);
}

absl::StatusOr<std::unique_ptr<GeneralizedGeometricTrafficModel>>
GeneralizedGeometricTrafficModelBuilder::Build() const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<GeneralizedGeometricTrafficModel> model,
                       TrafficModelBuilder::Build());
  model->SetLambda(lambda_);
  model->SetBurstProb(burst_prob_);
  model->SetRandomNumberInterface(*random_interface_);
  return model;
}

ReplayTrafficModelBuilder::ReplayTrafficModelBuilder(
    int64_t packet_size_bits, absl::Span<const int64_t> clock_cycles) {
  SetPacketSizeBits(packet_size_bits);
  clock_cycles_.insert(clock_cycles_.end(), clock_cycles.begin(),
                       clock_cycles.end());
}

absl::StatusOr<std::unique_ptr<ReplayTrafficModel>>
ReplayTrafficModelBuilder::Build() const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ReplayTrafficModel> model,
                       TrafficModelBuilder::Build());
  model->SetClockCycles(clock_cycles_);
  return model;
}

ReplayTrafficModel::ReplayTrafficModel(int64_t packet_size_bits,
                                       absl::Span<const int64_t> clock_cycles)
    : TrafficModel(packet_size_bits) {
  SetClockCycles(clock_cycles);
  clock_cycle_iter_ = clock_cycles_.cbegin();
}

std::vector<DataPacket> ReplayTrafficModel::GetNewCyclePackets(int64_t cycle) {
  if (cycle > cycle_count_) {
    cycle_count_ = cycle;
  }

  if ((clock_cycle_iter_ == clock_cycles_.end()) ||
      (*clock_cycle_iter_ != cycle)) {
    // No packets to be sent until next_packet_cycle_.
    return std::vector<DataPacket>();
  }

  std::vector<DataPacket> packets;

  CHECK_EQ(cycle, *clock_cycle_iter_);

  absl::StatusOr<DataPacket> packet = DataPacketBuilder()
                                          .Valid(true)
                                          .ZeroedData(packet_size_bits_)
                                          .VirtualChannel(vc_)
                                          .SourceIndex(source_index_)
                                          .DestinationIndex(destination_index_)
                                          .Build();
  CHECK(packet.ok());
  packets.push_back(packet.value());
  clock_cycle_iter_++;
  return packets;
}

double ReplayTrafficModel::ExpectedTrafficRateInMiBps(
    int64_t cycle_time_ps) const {
  double total_sec = static_cast<double>(cycle_count_ + 1) *
                     static_cast<double>(cycle_time_ps) * 1.0e-12;
  int64_t num_packets = std::distance(clock_cycles_.begin(), clock_cycle_iter_);
  double bits_per_sec = static_cast<double>(packet_size_bits_) * num_packets;
  bits_per_sec = bits_per_sec / 1024.0 / 1024.0 / 8.0;
  return bits_per_sec / total_sec;
}

void ReplayTrafficModel::SetClockCycles(
    absl::Span<const int64_t> clock_cycles) {
  clock_cycles_ =
      std::vector<int64_t>(clock_cycles.begin(), clock_cycles.end());
  std::sort(clock_cycles_.begin(), clock_cycles_.end());
  clock_cycle_iter_ = clock_cycles_.cbegin();
}

absl::Span<const int64_t> ReplayTrafficModel::GetClockCycles() const {
  return clock_cycles_;
}

}  // namespace xls::noc
