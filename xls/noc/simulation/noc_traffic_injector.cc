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

#include "xls/noc/simulation/noc_traffic_injector.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/simulation/flit.h"

namespace xls::noc {

absl::Status NocTrafficInjector::RunCycle() {
  if (simulator_ == nullptr) {
    return absl::InternalError(
        "Unable to run traffic injection without calling SetSimulatorShim().");
  }

  ++cycle_;

  for (int64_t i = 0; i < traffic_models_.size(); ++i) {
    // Retrieve packets.
    std::vector<DataPacket> packets =
        traffic_models_[i]->GetNewCyclePackets(cycle_);

    this->traffic_model_monitor_[i].AcceptNewPackets(absl::MakeSpan(packets),
                                                     cycle_);

    // Convert to flits, and sequence them for injection.
    int64_t source_index = flows_index_to_sources_index_map_.at(i);
    NetworkComponentId source = source_network_interfaces_.at(source_index);

    // All packets are depacketized and converted to flits on a single cycle.
    // Those flits are injected into the simulator.  The simulator is
    // expected to have an infinite queue and will send a single flit per
    // cycle in the order in which they are received..
    //
    // TODO(tedhong): 2021-06-29 - Model a fixed queue between the depacketizer
    //                            and the network interface.
    //
    // TODO(tedhong): 2021-06-29 - Model priority between different flows.
    //                             so packets are not handled and sent in-order.

    DePacketizer& depacketizer = depacketizers_[source_index];
    for (DataPacket& p : packets) {
      XLS_RET_CHECK_OK(depacketizer.AcceptNewPacket(p));

      while (!depacketizer.IsIdle()) {
        XLS_ASSIGN_OR_RETURN(DataFlit flit, depacketizer.ComputeNextFlit());
        // Add information defining the cycle iteration the flit is injected
        // into the network.
        TimedDataFlitInfo info{cycle_};
        TimedDataFlit timed_data_flit{cycle_, flit, info};
        XLS_RET_CHECK_OK(simulator_->SendFlitAtTime(timed_data_flit, source));
      }
    }
  }

  return absl::OkStatus();
}

namespace {

// Function that calls run_action(i, j) for each flow and network_component
// such that flow[i] corresponds to a flow that carries traffic
// associated with network_object[j].
//   - Each i in [0, flows.size()) will be called in in-order
//   - It is an error for a flow to not have a corresponding network_object
//
// A flow corresponds to the network component if
// the string returned from flow_component_func(flow) matches the network
// components name.
template <typename ParamType, typename ActionFunction,
          typename FlowComponentFunc>
absl::Status MatchFlowToSourceAndRunAction(
    absl::Span<const TrafficFlowId> flows,
    absl::Span<const NetworkComponentId> network_components,
    const NocTrafficManager& traffic_manager,
    const NocParameters& noc_parameters, ActionFunction&& run_action,
    FlowComponentFunc&& flow_component_func) {
  int64_t flow_count = flows.size();

  for (int64_t i = 0; i < flow_count; ++i) {
    TrafficFlowId flow_id = flows[i];
    XLS_RET_CHECK(flow_id.IsValid());
    const TrafficFlow& flow = traffic_manager.GetTrafficFlow(flow_id);

    std::string_view flow_component = flow_component_func(flow);

    bool matched_flow = false;
    for (int64_t j = 0; j < network_components.size(); ++j) {
      XLS_ASSIGN_OR_RETURN(
          NetworkComponentParam param,
          noc_parameters.GetNetworkComponentParam(network_components[j]));

      if (!std::holds_alternative<ParamType>(param)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected all network components to be of type %s",
                            typeid(ParamType).name()));
      }

      std::string_view port_name = std::get<ParamType>(param).GetName();

      if (port_name == flow_component) {
        matched_flow = true;

        run_action(i, j);

        VLOG(1) << absl::StrFormat("Mapped flow %s %s to nc %x index %d\n",
                                   flow.GetName(), port_name,
                                   network_components[j].AsUInt64(), j);
        break;
      }
    }

    if (!matched_flow) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unable to find matching port %s for flow %s",
                          flow.GetName(), flow.GetSource()));
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<int64_t>>
NocTrafficInjectorBuilder::CalculateMaxPacketSizePerSource(
    absl::Span<const NetworkComponentId> network_sources,
    const NocTrafficManager& traffic_manager,
    const NocParameters& noc_parameters) {
  std::vector<int64_t> packet_size_per_source(network_sources.size(), 0);
  std::vector<TrafficFlowId> flows = traffic_manager.GetTrafficFlowIds();

  XLS_RET_CHECK_OK(MatchFlowToSourceAndRunAction<NetworkInterfaceSrcParam>(
      absl::MakeSpan(flows), network_sources, traffic_manager, noc_parameters,
      [&flows, &packet_size_per_source, &traffic_manager](
          int64_t flow_index, int64_t source_index) -> void {
        const TrafficFlow& flow =
            traffic_manager.GetTrafficFlow(flows[flow_index]);

        if (packet_size_per_source[source_index] < flow.GetPacketSizeInBits()) {
          packet_size_per_source[source_index] = flow.GetPacketSizeInBits();
        }
      },
      [](const TrafficFlow& flow) -> std::string_view {
        return flow.GetSource();
      }));

  return packet_size_per_source;
}

absl::Status NocTrafficInjectorBuilder::AssociateFlowsToNetworkSources(
    absl::Span<const TrafficFlowId> traffic_flows,
    absl::Span<const NetworkComponentId> network_sources,
    const NocTrafficManager& traffic_manager,
    const NocParameters& noc_parameters, NocTrafficInjector& injector) {
  injector.source_network_interfaces_ =
      std::vector(network_sources.begin(), network_sources.end());

  int64_t flow_count = traffic_flows.size();

  injector.flows_index_to_sources_index_map_.resize(flow_count);
  injector.traffic_model_monitor_.resize(flow_count);

  XLS_RET_CHECK_OK(MatchFlowToSourceAndRunAction<NetworkInterfaceSrcParam>(
      traffic_flows, network_sources, traffic_manager, noc_parameters,
      [&injector](int64_t flow_index, int64_t source_index) -> void {
        injector.flows_index_to_sources_index_map_[flow_index] = source_index;
      },
      [](const TrafficFlow& flow) -> std::string_view {
        return flow.GetSource();
      }));

  return absl::OkStatus();
}

absl::Status NocTrafficInjectorBuilder::AssociateFlowsToNetworkSinks(
    absl::Span<const TrafficFlowId> traffic_flows,
    absl::Span<const NetworkComponentId> network_sinks,
    const NocTrafficManager& traffic_manager,
    const NocParameters& noc_parameters, NocTrafficInjector& injector) {
  int64_t flow_count = traffic_flows.size();
  injector.flows_index_to_sinks_index_map_.resize(flow_count);

  XLS_RET_CHECK_OK(MatchFlowToSourceAndRunAction<NetworkInterfaceSinkParam>(
      traffic_flows, network_sinks, traffic_manager, noc_parameters,
      [&injector](int64_t flow_index, int64_t sink_index) -> void {
        injector.flows_index_to_sinks_index_map_[flow_index] = sink_index;
      },
      [](const TrafficFlow& flow) -> std::string_view {
        return flow.GetDestination();
      }));

  return absl::OkStatus();
}

absl::Status NocTrafficInjectorBuilder::AssociateFlowsToVCs(
    absl::Span<const TrafficFlowId> traffic_flows,
    absl::Span<const VirtualChannelParam> network_vcs,
    const NocTrafficManager& traffic_manager,
    const NocParameters& noc_parameters, NocTrafficInjector& injector) {
  // TODO(tedhong): 2021-07-15.  Factor out and combine this functionality
  // with similar functionality in the indexer.h classes.

  int64_t flow_count = traffic_flows.size();
  injector.flows_index_to_vc_index_map_.resize(flow_count);

  for (int64_t i = 0; i < flow_count; ++i) {
    TrafficFlowId flow_id = traffic_flows[i];
    XLS_RET_CHECK(flow_id.IsValid());

    const TrafficFlow& flow = traffic_manager.GetTrafficFlow(flow_id);
    std::string_view flow_vc = flow.GetVC();

    // If there are no VCs in use, always map to 0.
    int64_t vc_index = 0;

    if (!network_vcs.empty()) {
      absl::Span<const VirtualChannelParam>::iterator iter =
          std::find_if(network_vcs.begin(), network_vcs.end(),
                       [&flow_vc](const VirtualChannelParam& param) -> bool {
                         return param.GetName() == flow_vc;
                       });

      if (iter == network_vcs.end()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unable to find matching vc %s for flow %s",
                            flow.GetVC(), flow.GetName()));
      }
      vc_index = iter - network_vcs.begin();
    }

    injector.flows_index_to_vc_index_map_[i] = vc_index;
    VLOG(1) << absl::StrFormat("Mapped flow %s vc %s to vc index %d\n",
                               flow.GetName(), flow_vc, vc_index);
  }

  return absl::OkStatus();
}

// Setup traffic models for each flow.
absl::Status NocTrafficInjectorBuilder::BuildPerFlowTrafficModels(
    int64_t cycle_time_ps, absl::Span<const TrafficFlowId> traffic_flows,
    const NocTrafficManager& traffic_manager,
    RandomNumberInterface& random_number_interface,
    NocTrafficInjector& injector) {
  for (int64_t i = 0; i < traffic_flows.size(); ++i) {
    TrafficFlowId flow_id = traffic_flows[i];
    const TrafficFlow& flow = traffic_manager.GetTrafficFlow(flow_id);

    double bits_per_packet = flow.GetPacketSizeInBits();
    int64_t source_index = injector.flows_index_to_sources_index_map_.at(i);
    int64_t sink_index = injector.flows_index_to_sinks_index_map_.at(i);
    int64_t vc_index = injector.flows_index_to_vc_index_map_.at(i);

    if (flow.IsReplay()) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ReplayTrafficModel> model,
          ReplayTrafficModelBuilder(bits_per_packet, flow.GetClockCycleTimes())
              .SetVCIndex(vc_index)
              .SetSourceIndex(source_index)
              .SetDestinationIndex(sink_index)
              .Build());
      injector.traffic_models_.push_back(std::move(model));
    } else {
      double burst_prob = flow.GetBurstProb();
      double bits_per_cycle = flow.GetTrafficPerNumPsInBits(cycle_time_ps);
      double lambda = bits_per_cycle / bits_per_packet;

      if (lambda > 1.0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unable to build traffic model for flow %s "
                            "rate %g MiBps "
                            "at cycle time %d ps %d bits per packet "
                            "requires sending %g > 1 packet per cycle.",
                            flow.GetName(), flow.GetTrafficRateInMiBps(),
                            cycle_time_ps, flow.GetPacketSizeInBits(), lambda));
      }

      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<GeneralizedGeometricTrafficModel> model,
          GeneralizedGeometricTrafficModelBuilder(
              lambda, burst_prob, bits_per_packet, random_number_interface)
              .SetVCIndex(vc_index)
              .SetSourceIndex(source_index)
              .SetDestinationIndex(sink_index)
              .Build());
      injector.traffic_models_.push_back(std::move(model));
    }
    injector.traffic_flows_.push_back(flow_id);
  }

  return absl::OkStatus();
}

absl::Status NocTrafficInjectorBuilder::BuildPerInterfaceDepacketizer(
    absl::Span<const NetworkComponentId> network_sources,
    absl::Span<int64_t> max_packet_size_per_source,
    const NetworkManager& network_manager, const NocParameters& noc_parameters,
    NocTrafficInjector& injector) {
  for (int64_t i = 0; i < network_sources.size(); ++i) {
    NetworkComponentId source_id = network_sources[i];
    // Get the one and only link attached to this source
    std::vector<PortId> source_port_ids =
        network_manager.GetNetworkComponent(source_id).GetPortIds();
    XLS_RET_CHECK_EQ(source_port_ids.size(), 1);
    ConnectionId connection_id =
        network_manager.GetPort(source_port_ids.at(0)).connection();
    XLS_RET_CHECK(connection_id.IsValid());
    PortId link_port_id = network_manager.GetConnection(connection_id).sink();
    XLS_RET_CHECK(link_port_id.IsValid());
    NetworkComponentId link_id = link_port_id.GetNetworkComponentId();

    // Retrieve parameter and associated bit width.
    XLS_ASSIGN_OR_RETURN(NetworkComponentParam param,
                         noc_parameters.GetNetworkComponentParam(link_id));

    if (!std::holds_alternative<LinkParam>(param)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected Link Parameter for network component %x",
                          link_id.AsUInt64()));
    }

    int64_t phit_width_in_bits =
        std::get<LinkParam>(param).GetPhitDataBitWidth();
    int64_t source_index_bit_count =
        Bits::MinBitCountUnsigned(network_sources.size() - 1);
    int64_t max_packet_bit_count = max_packet_size_per_source[i];

    // Test packet to flit behavior.
    injector.depacketizers_.push_back(DePacketizer(
        phit_width_in_bits, source_index_bit_count, max_packet_bit_count));
  }
  return absl::OkStatus();
}

}  // namespace xls::noc
