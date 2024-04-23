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

#include "xls/noc/drivers/samples/aggregate_tree_experiment.h"

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"
#include "xls/noc/simulation/common.h"

namespace xls::noc {

namespace {

// Simple VC experiment:
//  Sweeps the vc count of the network from no-vcs (1 effective) to having
//  vc's (2 effective) and measures the traffic sent and received across
//  two flows.

class SimpleVCExperimentBuilder : public ExperimentBuilderBase {
 protected:
  absl::StatusOr<ExperimentConfig> BuildExperimentConfig() override;
  absl::StatusOr<ExperimentSweeps> BuildExperimentSweeps() override;
  absl::StatusOr<ExperimentRunner> BuildExperimentRunner() override;
};

absl::StatusOr<ExperimentConfig>
SimpleVCExperimentBuilder::BuildExperimentConfig() {
  ExperimentConfig config;

  //  This test tests that a router can aggregate multiple smaller flows
  //  together to achive close to the maximum bandwidth of the network.
  //
  //   SendPort0     SendPortN
  //       |            |
  //       | L=2 ...    | L=2
  //     Ain0          AinN
  //  [ RouterA ]
  //     Aout0
  //       |
  //       | L=2
  //   RecvPort0
  //
  NetworkConfigProtoBuilder builder("AggregateTreeExperiment");
  builder.WithVirtualChannel("VC0").WithDepth(10);

  int64_t input_port_count = 16;

  for (int64_t i = 0; i < input_port_count; ++i) {
    std::string input_port_name = absl::StrFormat("SendPort%d", i);
    std::string router_port_name = absl::StrFormat("Ain%d", i);
    std::string link_name = absl::StrFormat("Link%dA", i);

    builder.WithPort(input_port_name)
        .AsInputDirection()
        .WithVirtualChannel("VC0");

    builder.WithLink(link_name)
        .WithSourcePort(input_port_name)
        .WithSinkPort(router_port_name)
        .WithPhitBitWidth(128)
        .WithSourceSinkPipelineStage(2);
  }

  builder.WithPort("RecvPort0").AsOutputDirection().WithVirtualChannel("VC0");

  builder.WithLink("LinkA0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2);

  auto routera = builder.WithRouter("RouterA");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0");

  for (int64_t i = 0; i < input_port_count; ++i) {
    std::string router_port_name = absl::StrFormat("Ain%d", i);

    routera.WithInputPort(router_port_name).WithVirtualChannel("VC0");
  }

  XLS_ASSIGN_OR_RETURN(NetworkConfigProto proto, builder.Build());
  config.SetNetworkConfig(proto);

  // Setup traffic flow per input port.
  NocTrafficManager& traffic_mgr = config.GetTrafficConfig();

  XLS_ASSIGN_OR_RETURN(TrafficModeId mode0_id, traffic_mgr.CreateTrafficMode());
  TrafficMode& mode = traffic_mgr.GetTrafficMode(mode0_id);
  mode.SetName("Main");

  int64_t individual_rate_mibps = 1 * 1024;
  for (int64_t i = 0; i < input_port_count; ++i) {
    std::string flow_name = absl::StrFormat("flow_%d", i);
    std::string input_port_name = absl::StrFormat("SendPort%d", i);

    XLS_ASSIGN_OR_RETURN(TrafficFlowId flow_id,
                         traffic_mgr.CreateTrafficFlow());

    TrafficFlow& flow = traffic_mgr.GetTrafficFlow(flow_id);
    flow.SetName(flow_name)
        .SetSource(input_port_name)
        .SetDestination("RecvPort0")
        .SetTrafficRateInMiBps(individual_rate_mibps)
        .SetPacketSizeInBits(128)
        .SetBurstProbInMils(7)
        .SetVC("VC0");

    mode.RegisterTrafficFlow(flow_id);
  }
  return config;
}

absl::StatusOr<ExperimentSweeps>
SimpleVCExperimentBuilder::BuildExperimentSweeps() {
  ExperimentSweeps sweep;

  // Add steps that changes the phit bit width from 128 downto 16.
  // Note: The base case starts with 128, so the steps start with 128-16.
  for (int64_t phit_width = 128 - 16; phit_width >= 16; phit_width -= 16) {
    sweep.AddNewStep([phit_width](ExperimentConfig& config) -> absl::Status {
      for (LinkConfigProto& link : *config.GetNetworkConfig().mutable_links()) {
        link.set_phit_bit_width(phit_width);
      }
      return absl::OkStatus();
    });
  }

  return sweep;
}

absl::StatusOr<ExperimentRunner>
SimpleVCExperimentBuilder::BuildExperimentRunner() {
  ExperimentRunner runner;
  runner.SetSimulationCycleCount(100'000)
      .SetCycleTimeInPs(500)
      .SetTrafficMode("Main")
      .SetSimulationSeed(100);
  return runner;
}

}  // namespace

// Adds the sample experiments to the factory.
absl::Status RegisterAggregateTreeExperiment(ExperimentFactory& factory) {
  return factory
      .RegisterNewBuilder<SimpleVCExperimentBuilder>("AggregateTreeExperiment")
      .status();
}

}  // namespace xls::noc
