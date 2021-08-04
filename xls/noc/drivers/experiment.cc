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

#include "xls/noc/drivers/experiment.h"

#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/network_graph_builder.h"
#include "xls/noc/simulation/noc_traffic_injector.h"
#include "xls/noc/simulation/parameters.h"
#include "xls/noc/simulation/sim_objects.h"
#include "xls/noc/simulation/simulator_to_traffic_injector_shim.h"
#include "xls/noc/simulation/traffic_description.h"

namespace xls::noc {

absl::Status ExperimentMetrics::DebugDump() const {
  XLS_LOG(INFO) << "Dumping Metrics ...";

  for (auto& [name, val] : float_metrics_) {
    XLS_LOG(INFO) << absl::StreamFormat("%s : %g", name, val);
  }

  for (auto& [name, val] : integer_metrics_) {
    XLS_LOG(INFO) << absl::StreamFormat("%s : %g", name, val);
  }

  return absl::OkStatus();
}

absl::StatusOr<ExperimentMetrics> ExperimentRunner::RunExperiment(
    const ExperimentConfig& experiment_config) const {
  // Build and assign simulation objects.
  NetworkManager graph;
  NocParameters params;

  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(
      experiment_config.GetNetworkConfig(), &graph, &params));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSIGN_OR_RETURN(DistributedRoutingTable routing_table,
                       route_builder.BuildNetworkRoutingTables(
                           graph.GetNetworkIds()[0], graph, params));

  // Build traffic model.
  RandomNumberInterface rnd;
  rnd.SetSeed(seed_);

  const NocTrafficManager& traffic_manager =
      experiment_config.GetTrafficConfig();
  XLS_ASSIGN_OR_RETURN(TrafficModeId mode_id,
                       traffic_manager.GetTrafficModeIdByName(mode_name_));
  XLS_ASSIGN_OR_RETURN(
      NocTrafficInjector traffic_injector,
      NocTrafficInjectorBuilder().Build(
          cycle_time_in_ps_, mode_id,
          routing_table.GetSourceIndices().GetNetworkComponents(),
          routing_table.GetSinkIndices().GetNetworkComponents(),
          params.GetNetworkParam(graph.GetNetworkIds()[0])
              ->GetVirtualChannels(),
          traffic_manager, graph, params, rnd));

  // Build simulator objects.
  NocSimulator simulator;
  XLS_RET_CHECK_OK(simulator.Initialize(graph, params, routing_table,
                                        graph.GetNetworkIds()[0]));
  simulator.Dump();

  // Hook traffic injector and simulator together.
  NocSimulatorToNocTrafficInjectorShim injector_shim(simulator,
                                                     traffic_injector);
  traffic_injector.SetSimulatorShim(injector_shim);
  simulator.RegisterPreCycleService(injector_shim);

  // Run simulation.
  for (int64_t i = 0; i < total_simulation_cycle_count_; ++i) {
    XLS_RET_CHECK_OK(simulator.RunCycle());
  }

  // Obtain metrics.  For now, the runner will measure traffic rate
  // for each flow, and sink.
  //
  // TODO(tedhong): 2021-07-13 Factor this out to make it possible for
  //                each experiment to define the set of metrics needed.
  ExperimentMetrics metrics;

  for (int64_t i = 0; i < traffic_injector.FlowCount(); ++i) {
    TrafficFlowId flow_id = traffic_injector.GetTrafficFlows().at(i);

    std::string metric_name =
        absl::StrFormat("Flow:%s:TrafficRateInMiBps",
                        traffic_manager.GetTrafficFlow(flow_id).GetName());

    double traffic_rate =
        traffic_injector.MeasuredTrafficRateInMiBps(cycle_time_in_ps_, i);

    metrics.SetFloatMetric(metric_name, traffic_rate);
  }

  for (NetworkComponentId sink_id :
       routing_table.GetSinkIndices().GetNetworkComponents()) {
    XLS_ASSIGN_OR_RETURN(NetworkComponentParam nc_param,
                         params.GetNetworkComponentParam(sink_id));
    XLS_CHECK(absl::holds_alternative<NetworkInterfaceSinkParam>(nc_param));
    std::string nc_name =
        std::string(absl::get<NetworkInterfaceSinkParam>(nc_param).GetName());

    std::string metric_name =
        absl::StrFormat("Sink:%s:TrafficRateInMiBps", nc_name);

    XLS_ASSIGN_OR_RETURN(SimNetworkInterfaceSink * sink,
                         simulator.GetSimNetworkInterfaceSink(sink_id));
    double traffic_rate = sink->MeasuredTrafficRateInMiBps(cycle_time_in_ps_);
    metrics.SetFloatMetric(metric_name, traffic_rate);

    metric_name = absl::StrFormat("Sink:%s:FlitCount", nc_name);
    metrics.SetIntegerMetric(metric_name, sink->GetReceivedTraffic().size());

    // Per VC Metrics
    int64_t vc_count =
        params.GetNetworkParam(graph.GetNetworkIds()[0])->VirtualChannelCount();
    for (int64_t vc = 0; vc < vc_count; ++vc) {
      metric_name =
          absl::StrFormat("Sink:%s:VC:%d:TrafficRateInMiBps", nc_name, vc);
      traffic_rate = sink->MeasuredTrafficRateInMiBps(cycle_time_in_ps_, vc);
      metrics.SetFloatMetric(metric_name, traffic_rate);
    }
  }

  return metrics;
}

}  // namespace xls::noc
