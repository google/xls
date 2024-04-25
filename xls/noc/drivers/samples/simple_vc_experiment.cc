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

#include "xls/noc/drivers/samples/simple_vc_experiment.h"

#include "absl/status/status.h"
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

  //   SendPort0
  //       |
  //       | L=2
  //     Ain0
  //  [ RouterA ]
  //     Aout0
  //       |
  //       | L=2
  //   RecvPort0
  //
  NetworkConfigProtoBuilder builder("SimpleVCExperiment");

  builder.WithVirtualChannel("VC0").WithDepth(10);
  builder.WithVirtualChannel("VC1").WithDepth(10);

  builder.WithPort("SendPort0")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort0")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");

  builder.WithLink("Link0A")
      .WithSourcePort("SendPort0")
      .WithSinkPort("Ain0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2);
  builder.WithLink("LinkA0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2);

  XLS_ASSIGN_OR_RETURN(NetworkConfigProto proto, builder.Build());
  config.SetNetworkConfig(proto);

  NocTrafficManager& traffic_mgr = config.GetTrafficConfig();

  XLS_ASSIGN_OR_RETURN(TrafficFlowId flow_0_id,
                       traffic_mgr.CreateTrafficFlow());
  TrafficFlow& flow_0 = traffic_mgr.GetTrafficFlow(flow_0_id);
  flow_0.SetName("flow_0")
      .SetSource("SendPort0")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(3 * 1024)
      .SetPacketSizeInBits(128)
      .SetBurstProbInMils(7)
      .SetVC("VC0");

  XLS_ASSIGN_OR_RETURN(TrafficFlowId flow_1_id,
                       traffic_mgr.CreateTrafficFlow());
  TrafficFlow& flow_1 = traffic_mgr.GetTrafficFlow(flow_1_id);
  flow_1.SetName("flow_1")
      .SetSource("SendPort0")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(2 * 1024)
      .SetPacketSizeInBits(128)
      .SetBurstProbInMils(7)
      .SetVC("VC0");

  XLS_ASSIGN_OR_RETURN(TrafficModeId mode0_id, traffic_mgr.CreateTrafficMode());
  TrafficMode& mode0 = traffic_mgr.GetTrafficMode(mode0_id);
  mode0.SetName("Main").RegisterTrafficFlow(flow_0_id).RegisterTrafficFlow(
      flow_1_id);

  return config;
}

absl::StatusOr<ExperimentSweeps>
SimpleVCExperimentBuilder::BuildExperimentSweeps() {
  ExperimentSweeps sweep;

  // Add a step that changes flow_1's VC from VC0 to VC1.
  sweep.AddNewStep([](ExperimentConfig& config) -> absl::Status {
    XLS_ASSIGN_OR_RETURN(
        TrafficFlowId id,
        config.GetTrafficConfig().GetTrafficFlowIdByName("flow_1"));
    config.GetTrafficConfig().GetTrafficFlow(id).SetVC("VC1");
    return absl::OkStatus();
  });

  // Add a step that changes the phit bit width from 128 to 16.
  sweep.AddNewStep([](ExperimentConfig& config) -> absl::Status {
    config.GetNetworkConfig().mutable_links(0)->set_phit_bit_width(16);
    config.GetNetworkConfig().mutable_links(1)->set_phit_bit_width(16);
    return absl::OkStatus();
  });

  // Add a step that combines both of the mutations above.
  sweep.AddNewStep([](ExperimentConfig& config) -> absl::Status {
    XLS_ASSIGN_OR_RETURN(
        TrafficFlowId id,
        config.GetTrafficConfig().GetTrafficFlowIdByName("flow_1"));
    config.GetTrafficConfig().GetTrafficFlow(id).SetVC("VC1");

    config.GetNetworkConfig().mutable_links(0)->set_phit_bit_width(16);
    config.GetNetworkConfig().mutable_links(1)->set_phit_bit_width(16);

    return absl::OkStatus();
  });

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
absl::Status RegisterSimpleVCExperiment(ExperimentFactory& factory) {
  return factory
      .RegisterNewBuilder<SimpleVCExperimentBuilder>("SimpleVCExperiment")
      .status();
}

}  // namespace xls::noc
