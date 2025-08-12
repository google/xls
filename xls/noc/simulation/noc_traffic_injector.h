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

#ifndef XLS_NOC_SIMULATION_NOC_TRAFFIC_INJECTOR_H_
#define XLS_NOC_SIMULATION_NOC_TRAFFIC_INJECTOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/packetizer.h"
#include "xls/noc/simulation/parameters.h"
#include "xls/noc/simulation/random_number_interface.h"
#include "xls/noc/simulation/sim_objects.h"
#include "xls/noc/simulation/simulator_shims.h"
#include "xls/noc/simulation/traffic_description.h"
#include "xls/noc/simulation/traffic_models.h"

// This file contains classes used to model traffic of a NOC.

namespace xls::noc {

class NocTrafficInjectorBuilder;

// Directs and injects packets into the NOC simulator based on
// predefined Traffic Models.
class NocTrafficInjector {
 public:
  // Run a single cycle, using the simulator shim to inject flits to be sent
  // on the current_cycle.
  absl::Status RunCycle();

  // Provides the interface between this object and the NOC simulator.
  void SetSimulatorShim(NocSimulatorTrafficServiceShim& simulator) {
    simulator_ = &simulator;
  }

  // Number of source network interfaces in the network graph.
  // Note: Depending on the flows, some of these interfaces may not
  //       have packets injected.
  int64_t SourceNetworkInterfaceCount() const {
    return source_network_interfaces_.size();
  }

  // Number of flows modeled by this injector.
  int64_t FlowCount() const { return traffic_models_.size(); }

  // List of all source network interface ids.
  absl::Span<const NetworkComponentId> GetSourceNetworkInterfaces() const {
    return source_network_interfaces_;
  }

  // Provides mapping from flows to network sources.
  absl::Span<const int64_t> GetFlowsIndexToSourcesIndexMap() const {
    return flows_index_to_sources_index_map_;
  }

  // List of all depacketizers responsible for converting modeled
  // packets to flits for injection.
  absl::Span<const DePacketizer> GetDePacketizers() const {
    return depacketizers_;
  }

  // List of all traffic models.
  absl::Span<const std::unique_ptr<TrafficModel>> GetTrafficModels() const {
    return traffic_models_;
  }

  // List of all traffic flows.
  absl::Span<const TrafficFlowId> GetTrafficFlows() const {
    return traffic_flows_;
  }

  // Get measured traffic rate injected during simulation for a single flow.
  //
  // Note: This is before any contention in the network so the rate should be
  //       close to the flow's specified rate.
  double MeasuredTrafficRateInMiBps(int64_t cycle_time_in_ps,
                                    int64_t flow_index) const {
    return traffic_model_monitor_[flow_index].MeasuredTrafficRateInMiBps(
        cycle_time_in_ps);
  }

  // Get measured bits injected during simulation for a single flow.
  int64_t MeasuredBitsSent(int64_t flow_index) const {
    return traffic_model_monitor_[flow_index].MeasuredBitsSent();
  }

 private:
  friend NocTrafficInjectorBuilder;

  // Interface to simulator for injecting flits.
  NocSimulatorTrafficServiceShim* simulator_ = nullptr;

  // Cycle that the simulator has simulated up to.
  int64_t cycle_ = -1;

  // Below vectors are all sized identically to a size equal
  // to that of the number of source network interfaces.

  // Ids of all network interfaces used.
  std::vector<NetworkComponentId> source_network_interfaces_;

  // Converts packets to a stream of flits to be injected.
  std::vector<DePacketizer> depacketizers_;

  // Below vectors are all sized identically to a size equal
  // to that of the number of network flows used.

  // Ids of all network flows used.
  std::vector<TrafficFlowId> traffic_flows_;

  // Associate each flow with a specific NetworkComponentId in network_sources.
  //
  // More than one flow may be mapped to the same network interface.
  std::vector<int64_t> flows_index_to_sources_index_map_;

  // Associate each flow with a specific destination index.
  std::vector<int64_t> flows_index_to_sinks_index_map_;

  // Associate each flow with a specific vc.
  std::vector<int64_t> flows_index_to_vc_index_map_;

  // TrafficModel, one per flow, responsible for defining when
  // each packet is injected and how big each packet is.
  std::vector<std::unique_ptr<TrafficModel>> traffic_models_;

  // Measure injected traffic rate.
  std::vector<TrafficModelMonitor> traffic_model_monitor_;
};

// Builder for constructing a NocTrafficInjector.
class NocTrafficInjectorBuilder {
 public:
  // Analyzes the network, parameters, and traffic specification of the NOC to
  // build a NocTrafficInjector.
  absl::StatusOr<NocTrafficInjector> Build(
      int64_t cycle_time_ps, TrafficModeId traffic_mode,
      absl::Span<const NetworkComponentId> network_sources,
      absl::Span<const NetworkComponentId> network_sinks,
      absl::Span<const VirtualChannelParam> network_vcs,
      const NocTrafficManager& traffic_manager,
      const NetworkManager& network_manager,
      const NocParameters& noc_parameters,
      RandomNumberInterface& random_number_interface) {
    absl::Span<const TrafficFlowId> traffic_flows =
        traffic_manager.GetTrafficMode(traffic_mode).GetTrafficFlows();

    NocTrafficInjector traffic_injector;

    XLS_ASSIGN_OR_RETURN(std::vector<int64_t> max_packet_size_per_source,
                         CalculateMaxPacketSizePerSource(
                             network_sources, traffic_manager, noc_parameters));
    XLS_RET_CHECK_OK(AssociateFlowsToNetworkSources(
        traffic_flows, network_sources, traffic_manager, noc_parameters,
        traffic_injector));
    XLS_RET_CHECK_OK(AssociateFlowsToNetworkSinks(
        traffic_flows, network_sinks, traffic_manager, noc_parameters,
        traffic_injector));
    XLS_RET_CHECK_OK(AssociateFlowsToVCs(traffic_flows, network_vcs,
                                         traffic_manager, noc_parameters,
                                         traffic_injector));
    XLS_RET_CHECK_OK(
        BuildPerFlowTrafficModels(cycle_time_ps, traffic_flows, traffic_manager,
                                  random_number_interface, traffic_injector));
    XLS_RET_CHECK_OK(BuildPerInterfaceDepacketizer(
        network_sources, absl::MakeSpan(max_packet_size_per_source),
        network_manager, noc_parameters, traffic_injector));

    return traffic_injector;
  }

 private:
  // Compute the max packet size for each source among all the possible flows.
  // Note: this is independent of the actual mode being simulated.
  absl::StatusOr<std::vector<int64_t>> CalculateMaxPacketSizePerSource(
      absl::Span<const NetworkComponentId> network_sources,
      const NocTrafficManager& traffic_manager,
      const NocParameters& noc_parameters);

  // Setup flows_index_to_sources_index_map_ which maps flows in traffic_flows_
  // to network source in network_sources_ by matching the flow's source
  // and the port name that the network source connects to.
  //
  // If X = traffic_flows[i], and Y = network_sources[j] are associated, then
  //   injector.flows_index_to_sources_index_map[i] = j
  absl::Status AssociateFlowsToNetworkSources(
      absl::Span<const TrafficFlowId> traffic_flows,
      absl::Span<const NetworkComponentId> network_sources,
      const NocTrafficManager& traffic_manager,
      const NocParameters& noc_parameters, NocTrafficInjector& injector);

  // Setup flows_index_to_sinks_index_map_ which maps flows in traffic_flows_
  // to network sinks in network_sinks by matching the flow's destination
  // and the port name that the network sink connects to.
  //
  // If X = traffic_flows[i], and Y = network_sinks[j] are associated, then
  //   injector.flows_index_to_sinks_index_map[i] = j
  absl::Status AssociateFlowsToNetworkSinks(
      absl::Span<const TrafficFlowId> traffic_flows,
      absl::Span<const NetworkComponentId> network_sinks,
      const NocTrafficManager& traffic_manager,
      const NocParameters& noc_parameters, NocTrafficInjector& injector);

  // Setup flows_index_to_vc_index_map_ which maps flows in traffic_flows_
  // to vcs in network_vcs by matching the flow's vc name to the
  // network vc with the same name.
  //
  // If X = traffic_flows[i], and Y = network_vcs[j] are associated, then
  //   injector.flows_index_to_vc_index_map[i] = j
  absl::Status AssociateFlowsToVCs(
      absl::Span<const TrafficFlowId> traffic_flows,
      absl::Span<const VirtualChannelParam> network_vcs,
      const NocTrafficManager& traffic_manager,
      const NocParameters& noc_parameters, NocTrafficInjector& injector);

  // Setup traffic models for each flow.
  absl::Status BuildPerFlowTrafficModels(
      int64_t cycle_time_ps, absl::Span<const TrafficFlowId> traffic_flows,
      const NocTrafficManager& traffic_manager,
      RandomNumberInterface& random_number_interface,
      NocTrafficInjector& injector);

  // Setup depacketizer for each network source used.
  absl::Status BuildPerInterfaceDepacketizer(
      absl::Span<const NetworkComponentId> network_sources,
      absl::Span<int64_t> max_packet_size_per_source,
      const NetworkManager& network_manager,
      const NocParameters& noc_parameters, NocTrafficInjector& injector);
};

// Shim to call the NocTrafficInjector from a simulator.
class NocTrafficInjectorService : public NocSimulatorServiceShim {
 public:
  NocTrafficInjectorService(NocTrafficInjector& injector, NocSimulator)
      : injector_(&injector) {}

  absl::Status RunCycle() final { return injector_->RunCycle(); }

 private:
  NocTrafficInjector* injector_;
};

};  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_NOC_TRAFFIC_INJECTOR_H_
