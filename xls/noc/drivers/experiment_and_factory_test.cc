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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"
#include "xls/noc/drivers/experiment.h"
#include "xls/noc/drivers/experiment_factory.h"

namespace xls::noc {
namespace {

// Dummy ExperimentBuilder used in this unit test.
class TestExperimentBuilder : public ExperimentBuilderBase {
 protected:
  virtual absl::StatusOr<ExperimentConfig> BuildExperimentConfig() override;
  virtual absl::StatusOr<ExperimentSweeps> BuildExperimentSweeps() override;
  virtual absl::StatusOr<ExperimentRunner> BuildExperimentRunner() override;
};

absl::StatusOr<ExperimentConfig>
TestExperimentBuilder::BuildExperimentConfig() {
  ExperimentConfig config;
  return config;
}

absl::StatusOr<ExperimentSweeps>
TestExperimentBuilder::BuildExperimentSweeps() {
  ExperimentSweeps sweep;
  return sweep;
}

absl::StatusOr<ExperimentRunner>
TestExperimentBuilder::BuildExperimentRunner() {
  ExperimentRunner runner;
  return runner;
}

TEST(ExperimentsTest, ExperimentFactory) {
  ExperimentFactory experiment_factory;

  EXPECT_EQ(experiment_factory.ListExperimentTags().size(), 0);

  XLS_ASSERT_OK_AND_ASSIGN(
      TestExperimentBuilder * builder1,
      experiment_factory.RegisterNewBuilder<TestExperimentBuilder>("Builder1"));
  EXPECT_EQ(experiment_factory.ListExperimentTags().size(), 1);
  EXPECT_EQ(experiment_factory.ListExperimentTags().at(0), "Builder1");

  XLS_ASSERT_OK_AND_ASSIGN(
      TestExperimentBuilder * builder2,
      experiment_factory.RegisterNewBuilder<TestExperimentBuilder>("Builder2"));
  EXPECT_EQ(experiment_factory.ListExperimentTags().size(), 2);
  EXPECT_EQ(experiment_factory.ListExperimentTags().at(0), "Builder1");
  EXPECT_EQ(experiment_factory.ListExperimentTags().at(1), "Builder2");

  EXPECT_NE(builder1, builder2);

  XLS_ASSERT_OK_AND_ASSIGN(Experiment experiment1,
                           experiment_factory.BuildExperiment("Builder1"));
}

TEST(ExperimentsTest, ExperimentConfig) {
  ExperimentConfig config;

  // Test set and retrieval of network config.
  NetworkConfigProtoBuilder builder("SimpleVCExperiment");
  builder.WithVirtualChannel("VCA").WithDepth(3);
  builder.WithVirtualChannel("VCB").WithDepth(5);
  builder.WithVirtualChannel("VCC").WithDepth(6);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto proto, builder.Build());
  config.SetNetworkConfig(proto);

  ASSERT_EQ(config.GetNetworkConfig().virtual_channels().size(), 3);
  EXPECT_EQ(config.GetNetworkConfig().virtual_channels(0).name(), "VCA");
  EXPECT_EQ(config.GetNetworkConfig().virtual_channels(1).name(), "VCB");
  EXPECT_EQ(config.GetNetworkConfig().virtual_channels(2).name(), "VCC");
  EXPECT_EQ(config.GetNetworkConfig().virtual_channels(0).depth(), 3);
  EXPECT_EQ(config.GetNetworkConfig().virtual_channels(1).depth(), 5);
  EXPECT_EQ(config.GetNetworkConfig().virtual_channels(2).depth(), 6);

  // Test set and retrieval of traffic config.
  NocTrafficManager traffic_mgr;
  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow_id,
                           traffic_mgr.CreateTrafficFlow());
  TrafficFlow& flow_0 = traffic_mgr.GetTrafficFlow(flow_id);
  flow_0.SetName("flow").SetVC("VCA");

  config.SetTrafficConfig(traffic_mgr);
  EXPECT_EQ(config.GetTrafficConfig().GetTrafficFlowIds().size(), 1);
  EXPECT_EQ(config.GetTrafficConfig().GetTrafficFlowIds().at(0), flow_id);
  EXPECT_EQ(config.GetTrafficConfig().GetTrafficFlow(flow_id).GetName(),
            "flow");
  EXPECT_EQ(config.GetTrafficConfig().GetTrafficFlow(flow_id).GetVC(), "VCA");
}

TEST(ExperimentsTest, ExperimentSweeps) {
  ExperimentConfig config;
  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow_id,
                           config.GetTrafficConfig().CreateTrafficFlow());
  config.GetTrafficConfig()
      .GetTrafficFlow(flow_id)
      .SetName("flow")
      .SetPacketSizeInBits(10);

  ExperimentSweeps sweep;
  EXPECT_EQ(sweep.GetStepCount(), 1);

  XLS_EXPECT_OK(sweep.ApplyMutationStep(0, config));
  EXPECT_EQ(
      config.GetTrafficConfig().GetTrafficFlow(flow_id).GetPacketSizeInBits(),
      10);

  for (int64_t i = 1; i <= 64; ++i) {
    sweep.AddNewStep([flow_id, i](ExperimentConfig& config) -> absl::Status {
      config.GetTrafficConfig().GetTrafficFlow(flow_id).SetPacketSizeInBits(
          100 + i);
      return absl::OkStatus();
    });
  }

  EXPECT_EQ(sweep.GetStepCount(), 65);

  for (int64_t i = 0; i <= 64; ++i) {
    XLS_EXPECT_OK(sweep.ApplyMutationStep(i, config));
    if (i == 0) {
      EXPECT_EQ(config.GetTrafficConfig()
                    .GetTrafficFlow(flow_id)
                    .GetPacketSizeInBits(),
                10);

    } else {
      EXPECT_EQ(config.GetTrafficConfig()
                    .GetTrafficFlow(flow_id)
                    .GetPacketSizeInBits(),
                100 + i);
    }
  }
}

TEST(ExperimentsTest, ExperimentMetrics) {
  ExperimentMetrics metrics;
  XLS_EXPECT_OK(metrics.DebugDump());

  metrics.SetIntegerMetric("metric 10", 10);
  metrics.SetIntegerMetric("metric -10", -10);
  metrics.SetIntegerMetric("metric 101", 101);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t m10, metrics.GetIntegerMetric("metric 10"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t mm10,
                           metrics.GetIntegerMetric("metric -10"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t m101,
                           metrics.GetIntegerMetric("metric 101"));

  EXPECT_EQ(m10, 10);
  EXPECT_EQ(mm10, -10);
  EXPECT_EQ(m101, 101);
  XLS_EXPECT_OK(metrics.DebugDump());

  metrics.SetFloatMetric("metric 0.10", 0.10);
  metrics.SetFloatMetric("metric -7.1", -7.1);

  XLS_ASSERT_OK_AND_ASSIGN(double m0p10, metrics.GetFloatMetric("metric 0.10"));
  XLS_ASSERT_OK_AND_ASSIGN(double mm7p1, metrics.GetFloatMetric("metric -7.1"));

  EXPECT_DOUBLE_EQ(m0p10, 0.10);
  EXPECT_DOUBLE_EQ(mm7p1, -7.1);

  XLS_EXPECT_OK(metrics.DebugDump());
}

}  // namespace
}  // namespace xls::noc
