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

#ifndef XLS_NOC_SIMULATION_TRAFFIC_DESCRIPTION_H_
#define XLS_NOC_SIMULATION_TRAFFIC_DESCRIPTION_H_

#include <queue>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/units.h"

// This file contains classes used specify the traffic patterns from
// and to the endpoints of a NOC.
//
// Individual traffic flows from a single source to a single description can be
// specified as TrafficFlows.  TrafficFlows are then grouped into different
// modes corresponding to different modes of the NOC operation.
//
// This traffic description is used to drive
//  1. The configuration of the NOC to satisfy the stated constraints.
//  2. The instantiation of NOC simulator components to inject traffic.

namespace xls::noc {

// Describes a single traffic flow from a single source to a single destination
// (unicast).
//
//  - There may be more than one flow from a single source to a single
//    destination.
//  - A traffic flow provides both a model of the flow, along with
//    specific constraints.
class TrafficFlow {
 public:
  // Construct a TrafficFlow object.
  explicit TrafficFlow(TrafficFlowId id) : id_(id) {}

  // Return id of this flow.
  TrafficFlowId id() const { return id_; }

  // Get name of the flow.
  absl::string_view GetName() const { return name_; }

  // String representation of the ingress port of the this flow.
  absl::string_view GetSource() const { return source_; }

  // String representation of the egress port of the this flow.
  absl::string_view GetDestination() const { return destination_; }

  // Average bit count moved by this flow in N ps.
  double GetTrafficPerNumPsInBits(int64_t num) const {
    return (static_cast<double>(bandwidth_bits_) * static_cast<double>(num)) /
           static_cast<double>(bandwidth_per_n_ps_);
  }

  // Traffic in bytes / sec.
  double GetTrafficRateInBytesPerSec() const {
    return ConvertDataRate(bandwidth_bits_, bandwidth_per_n_ps_, kUnitPsBits,
                           kUnitSecBytes);
  }

  // Packet size in bits.
  double GetPacketSizeInBits() const { return packet_size_bits_; }

  // Packet size in bytes.
  double GetPacketSizeInBytes() const {
    return ConvertDataVolume(packet_size_bits_, kUnitPsBits, kUnitSecBytes);
  }

  TrafficFlow& SetName(absl::string_view s) {
    name_ = s;
    return *this;
  }

  TrafficFlow& SetSource(absl::string_view s) {
    source_ = s;
    return *this;
  }

  TrafficFlow& SetDestination(absl::string_view d) {
    destination_ = d;
    return *this;
  }

  TrafficFlow& SetTrafficRateInBitsPerPS(int64_t bits, int64_t ps) {
    SetBandwidthBits(bits);
    SetBandwidthPs(ps);
    return *this;
  }

  TrafficFlow& SetTrafficRateInMiBps(int64_t mebibytes_per_sec) {
    return SetTrafficRateInBitsPerPS(mebibytes_per_sec * 1024 * 1024 * 8,
                                     int64_t{1'000'000'000'000});
  }

  TrafficFlow& SetPacketSizeInBits(int64_t bits) {
    packet_size_bits_ = bits;
    return *this;
  }

  // Internally the traffic bandwidth is represented as a rational number:
  //  bandwidth_bits_ / bandwidth_per_n_ps_ -- bits / ps
  // Below are accesors for direct access to those member variables.

  // Get numerator of bandwidth rate (bits / ps) in bits.
  int64_t GetBandwidthBits() const { return bandwidth_bits_; }

  // Get denominator of bandwidth rate (bits / ps ) in ps.
  int64_t GetBandwidthPerNumPs() const { return bandwidth_per_n_ps_; }

  TrafficFlow& SetBandwidthBits(int64_t bits) {
    bandwidth_bits_ = bits;
    return *this;
  }

  TrafficFlow& SetBandwidthPs(int64_t per_n_ps) {
    bandwidth_per_n_ps_ = per_n_ps;
    return *this;
  }

 private:
  TrafficFlowId id_;

  std::string name_;
  std::string source_;
  std::string destination_;

  int64_t bandwidth_bits_ = 0;
  int64_t bandwidth_per_n_ps_ = 1;
  int64_t packet_size_bits_ = 1;
};

class NocTrafficManager;

// Represents a set of flows that may be active together within a mode.
// Modes represent different operating conditions of a network.
class TrafficMode {
 public:
  // Construct a TrafficMode object.
  TrafficMode(NocTrafficManager& mgr, TrafficModeId id)
      : id_(id), traffic_manager_(&mgr) {}

  // Return id of this flow.
  TrafficModeId id() const { return id_; }

  // Get the NocTrafficManager that contains this object.
  NocTrafficManager& GetNocTrafficManager() { return *traffic_manager_; }

  // Return all traffic flows associated with this mode.
  absl::Span<const TrafficFlowId> GetTrafficFlows() const {
    return absl::MakeSpan(flows_);
  }

  // Returns True if TrafficFlow is associated with this mode.
  bool HasTrafficFlow(TrafficFlowId id) const {
    return std::find(flows_.begin(), flows_.end(), id) != flows_.end();
  }

  // Name of this mode.
  absl::string_view GetName() const { return name_; }

  // Set name.
  TrafficMode& SetName(absl::string_view s) {
    name_ = s;
    return *this;
  }

  // Associate a traffic flow with this mode.
  TrafficMode& RegisterTrafficFlow(TrafficFlowId id) {
    if (!HasTrafficFlow(id)) {
      flows_.push_back(id);
    }

    return *this;
  }

 private:
  TrafficModeId id_;
  NocTrafficManager* traffic_manager_;

  std::string name_;
  std::vector<TrafficFlowId> flows_;
};

// Manages and groups a set of traffic flows for a given NOC.
class NocTrafficManager {
 public:
  // Allocates and returns id of a new TrafficFlow.
  absl::StatusOr<TrafficFlowId> CreateTrafficFlow() {
    int64_t n_flows = traffic_flows_.size();

    XLS_ASSIGN_OR_RETURN(TrafficFlowId next_id,
                         TrafficFlowId::ValidateAndReturnId(n_flows));
    traffic_flows_.emplace_back(next_id);
    return next_id;
  }

  // Retreives reference to a TrafficFlow object given an id.
  TrafficFlow& GetTrafficFlow(TrafficFlowId id) {
    return traffic_flows_.at(id.id());
  }

  // Allocates and returns id of a new TrafficMode.
  absl::StatusOr<TrafficModeId> CreateTrafficMode() {
    int64_t n_flows = traffic_modes_.size();

    XLS_ASSIGN_OR_RETURN(TrafficModeId next_id,
                         TrafficModeId::ValidateAndReturnId(n_flows));
    traffic_modes_.emplace_back(*this, next_id);
    return next_id;
  }

  // Retreives reference to a TrafficMode object given an id.
  TrafficMode& GetTrafficMode(TrafficModeId id) {
    return traffic_modes_.at(id.id());
  }

 private:
  std::vector<TrafficFlow> traffic_flows_;
  std::vector<TrafficMode> traffic_modes_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_TRAFFIC_DESCRIPTION_H_
