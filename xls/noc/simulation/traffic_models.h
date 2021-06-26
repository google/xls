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

#ifndef XLS_NOC_SIMULATION_TRAFFIC_MODELS_H_
#define XLS_NOC_SIMULATION_TRAFFIC_MODELS_H_

#include <queue>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/packetizer.h"
#include "xls/noc/simulation/random_number_interface.h"
#include "xls/noc/simulation/units.h"

// This file contains classes used to model traffic of a NOC.

namespace xls::noc {

// Models the traffic injected into a single source according to
// a Generalized Geometric Distribution
// (https://ieeexplore.ieee.org/document/9153030).
//
class GeneralizedGeometricTrafficModel {
 public:
  GeneralizedGeometricTrafficModel(double lambda, double burst_prob,
                                   int64_t packet_size_bits,
                                   RandomNumberInterface& rnd)
      : next_packet_cycle_(-1),
        packet_size_bits_(packet_size_bits),
        lambda_(lambda),
        burst_prob_(burst_prob),
        random_interface_(&rnd) {}

  // Retrieves packets sent in the next cycle.
  //
  // Note: A call to this function advances the model's internal state, so
  //       a call to GetNewCyclePackets(N) should not be called multiple times.
  // Note: The simulator will successivly call GetNewCyclePackets(0),
  //       GetNewCyclePackets(1), GetNewCyclePackets(2), ...
  //
  // TODO(tedhong): 2021-06-27 Add an interface to support fast-forwarding.
  std::vector<DataPacket> GetNewCyclePackets(int64_t cycle);

  // Returns expected rate of traffic injected in MebiBytes Per Sec.
  //
  // Note - actual traffic rate will be close, but not exact due to
  //      - statistical variation.
  double ExpectedTrafficRateInMiBps(int64_t cycle_time_ps) {
    double num_cycles = 1.0e12 / static_cast<double>(cycle_time_ps);
    double num_packets = lambda_ * num_cycles;
    double bits_per_sec = static_cast<double>(packet_size_bits_) * num_packets;
    return bits_per_sec / 1024.0 / 1024.0;
  }

 private:
  DataPacket next_packet_;
  int64_t next_packet_cycle_;

  int64_t packet_size_bits_;  // All packets are sent with uniform size

  double lambda_;      // Lambda of the distribution (unit 1/cycle)
  double burst_prob_;  // Probability of a burst

  RandomNumberInterface* random_interface_;
};

// Measures the traffic injected and computes aggregate statistics.
class TrafficModelMonitor {
 public:
  // Measures new packets that were sent on cycle.
  void AcceptNewPackets(absl::Span<DataPacket> packets, int64_t cycle) {
    if (cycle > max_cycle_) {
      max_cycle_ = cycle;
    }

    for (DataPacket& p : packets) {
      num_bits_sent_ += p.data.bit_count();
    }
  }

  // Returns observed rate of traffic in MebiBytes Per Second seen in all
  // previous calls to AcceptNewPackets().
  double MeasuredTrafficRateInMiBps(int64_t cycle_time_ps) {
    double total_sec = static_cast<double>(max_cycle_ + 1) *
                       static_cast<double>(cycle_time_ps) * 1.0e-12;
    double bits_per_sec = static_cast<double>(num_bits_sent_) / total_sec;
    return bits_per_sec / 1024.0 / 1024.0;
  }

 private:
  int64_t max_cycle_ = 0;
  int64_t num_bits_sent_ = 0;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_TRAFFIC_MODELS_H_
