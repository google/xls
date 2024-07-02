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

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/noc/simulation/packetizer.h"
#include "xls/noc/simulation/random_number_interface.h"
#include "xls/noc/simulation/units.h"

// This file contains classes used to model traffic of a NOC.

namespace xls::noc {

// Traffic Model Base Class
class TrafficModel {
 public:
  explicit TrafficModel(int64_t packet_size_bits)
      : vc_(0), packet_size_bits_(packet_size_bits) {}

  // Retrieves packets sent in the next cycle.
  //
  // Note: A call to this function advances the model's internal state, so
  //       a call to GetNewCyclePackets(N) should not be called multiple times.
  // Note: The simulator will successively call GetNewCyclePackets(0),
  //       GetNewCyclePackets(1), GetNewCyclePackets(2), ...
  //
  // TODO(tedhong): 2021-06-27 Add an interface to support fast-forwarding.
  virtual std::vector<DataPacket> GetNewCyclePackets(int64_t cycle) = 0;

  // Returns expected rate of traffic injected in MebiBytes Per Sec.
  virtual double ExpectedTrafficRateInMiBps(int64_t cycle_time_ps) const = 0;

  void SetVCIndex(int64_t vc) { vc_ = vc; }
  int64_t GetVCIndex() const { return vc_; }

  void SetSourceIndex(int64_t src) { source_index_ = src; }
  int64_t GetSourceIndex() const { return source_index_; }

  void SetDestinationIndex(int64_t dest) { destination_index_ = dest; }

  int64_t GetDestinationIndex() const { return destination_index_; }

  int64_t GetPacketSizeInBits() const { return packet_size_bits_; }

  virtual ~TrafficModel() = default;

 protected:
  int64_t vc_;                 // VC index to send packets on.
  int64_t source_index_;       // Source index to send packets on.
  int64_t destination_index_;  // Destination index that packets will arrive on.

  int64_t packet_size_bits_;  // All packets are sent with uniform size
};

// Traffic Model Builder Base Class
template <class TrafficModelBuilderType, class TrafficModelType,
          class =
              std::enable_if<std::is_base_of_v<TrafficModel, TrafficModelType>>>
class TrafficModelBuilder {
 public:
  TrafficModelBuilder()
      : vc_(-1),
        source_index_(-1),
        destination_index_(-1),
        packet_size_bits_(-1) {}

  TrafficModelBuilderType& SetVCIndex(int64_t vc) {
    vc_ = vc;
    return static_cast<TrafficModelBuilderType&>(*this);
  }

  TrafficModelBuilderType& SetSourceIndex(int64_t src) {
    source_index_ = src;
    return static_cast<TrafficModelBuilderType&>(*this);
  }

  TrafficModelBuilderType& SetDestinationIndex(int64_t dest) {
    destination_index_ = dest;
    return static_cast<TrafficModelBuilderType&>(*this);
  }

  TrafficModelBuilderType& SetPacketSizeBits(int64_t packet_size_bits) {
    packet_size_bits_ = packet_size_bits;
    return static_cast<TrafficModelBuilderType&>(*this);
  }

  absl::StatusOr<std::unique_ptr<TrafficModelType>> Build() const {
    auto model =
        std::make_unique<TrafficModelType>(TrafficModelType(packet_size_bits_));
    model->SetVCIndex(vc_);
    model->SetSourceIndex(source_index_);
    model->SetDestinationIndex(destination_index_);
    return model;
  }

  virtual ~TrafficModelBuilder() = default;

 protected:
  int64_t vc_;                 // VC index to send packets on.
  int64_t source_index_;       // Source index to send packets on.
  int64_t destination_index_;  // Destination index that packets will arrive on.

  int64_t packet_size_bits_;  // All packets are sent with uniform size
};

// Models the traffic injected into a single source according to
// a Generalized Geometric Distribution
// (https://ieeexplore.ieee.org/document/9153030).
//
class GeneralizedGeometricTrafficModel : public TrafficModel {
 public:
  explicit GeneralizedGeometricTrafficModel(int64_t packet_size_bits)
      : TrafficModel(packet_size_bits), next_packet_cycle_(-1) {}
  GeneralizedGeometricTrafficModel(double lambda, double burst_prob,
                                   int64_t packet_size_bits,
                                   RandomNumberInterface& rnd)
      : TrafficModel(packet_size_bits),
        lambda_(lambda),
        burst_prob_(burst_prob),
        next_packet_cycle_(-1),
        random_interface_(&rnd) {}

  std::vector<DataPacket> GetNewCyclePackets(int64_t cycle) override;

  double ExpectedTrafficRateInMiBps(int64_t cycle_time_ps) const override {
    double num_cycles = 1.0e12 / static_cast<double>(cycle_time_ps);
    double num_packets = lambda_ * num_cycles;
    double bits_per_sec = static_cast<double>(packet_size_bits_) * num_packets;
    return bits_per_sec / 1024.0 / 1024.0 / 8.0;
  }

  double GetLambda() const { return lambda_; }
  void SetLambda(double lambda) { lambda_ = lambda; }

  double GetBurstProb() const { return burst_prob_; }
  void SetBurstProb(double burst_prob) { burst_prob_ = burst_prob; }

  RandomNumberInterface* GetRandomNumberInterface() const {
    return random_interface_;
  }
  void SetRandomNumberInterface(RandomNumberInterface& rnd) {
    random_interface_ = &rnd;
  }

 private:
  DataPacket next_packet_;

  double lambda_;      // Lambda of the distribution (unit 1/cycle)
  double burst_prob_;  // Probability of a burst

  int64_t next_packet_cycle_;

  RandomNumberInterface* random_interface_;
};

class GeneralizedGeometricTrafficModelBuilder
    : public TrafficModelBuilder<GeneralizedGeometricTrafficModelBuilder,
                                 GeneralizedGeometricTrafficModel> {
 public:
  GeneralizedGeometricTrafficModelBuilder(double lambda, double burst_prob,
                                          int64_t packet_size_bits,
                                          RandomNumberInterface& rnd);

  absl::StatusOr<std::unique_ptr<GeneralizedGeometricTrafficModel>> Build()
      const;

 private:
  double lambda_;      // Lambda of the distribution (unit 1/cycle)
  double burst_prob_;  // Probability of a burst

  RandomNumberInterface* random_interface_;
};

// Models the traffic injected into a single source according at specified
// cycle times.
class ReplayTrafficModel : public TrafficModel {
 public:
  explicit ReplayTrafficModel(int64_t packet_size_bits)
      : TrafficModel(packet_size_bits),
        clock_cycle_iter_{clock_cycles_.cend()} {}
  // Assumes element in clock_cycles are valid.
  ReplayTrafficModel(int64_t packet_size_bits,
                     absl::Span<const int64_t> clock_cycles);

  std::vector<DataPacket> GetNewCyclePackets(int64_t cycle) override;

  double ExpectedTrafficRateInMiBps(int64_t cycle_time_ps) const override;

  // Sets clock cycles to list and sorts the complete list of clock cycle.
  void SetClockCycles(absl::Span<const int64_t> clock_cycles);
  absl::Span<const int64_t> GetClockCycles() const;

 private:
  // cycle count
  int64_t cycle_count_ = 0;
  // clock cycles to inject a packet.
  std::vector<int64_t> clock_cycles_;
  // Iterator to next clock cycle.
  std::vector<int64_t>::const_iterator clock_cycle_iter_;
};

class ReplayTrafficModelBuilder
    : public TrafficModelBuilder<ReplayTrafficModelBuilder,
                                 ReplayTrafficModel> {
 public:
  ReplayTrafficModelBuilder(int64_t packet_size_bits,
                            absl::Span<const int64_t> clock_cycles);

  absl::StatusOr<std::unique_ptr<ReplayTrafficModel>> Build() const;

 private:
  std::vector<int64_t> clock_cycles_;
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
      ++packets_sent_count_;
    }
  }

  // Returns observed count of packets sent in all previous calls to
  // AcceptNewPackets().
  int64_t MeasuredPacketCount() const { return packets_sent_count_; }

  // Returns observed rate of traffic in MebiBytes Per Second seen in all
  // previous calls to AcceptNewPackets().
  double MeasuredTrafficRateInMiBps(int64_t cycle_time_ps) const {
    double total_sec = static_cast<double>(max_cycle_ + 1) *
                       static_cast<double>(cycle_time_ps) * 1.0e-12;
    double bits_per_sec = static_cast<double>(num_bits_sent_) / total_sec;
    return bits_per_sec / 1024.0 / 1024.0 / 8.0;
  }

  int64_t MeasuredBitsSent() const { return num_bits_sent_; }

 private:
  int64_t max_cycle_ = 0;
  int64_t num_bits_sent_ = 0;
  int64_t packets_sent_count_ = 0;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_TRAFFIC_MODELS_H_
