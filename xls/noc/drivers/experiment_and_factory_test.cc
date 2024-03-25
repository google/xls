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

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"
#include "xls/noc/drivers/experiment.h"
#include "xls/noc/drivers/experiment_factory.h"
#include "xls/noc/simulation/common.h"

namespace xls::noc {
namespace {

using ::xls::status_testing::StatusIs;

// Dummy ExperimentBuilder used in this unit test.
class TestExperimentBuilder : public ExperimentBuilderBase {
 protected:
  absl::StatusOr<ExperimentConfig> BuildExperimentConfig() override;
  absl::StatusOr<ExperimentSweeps> BuildExperimentSweeps() override;
  absl::StatusOr<ExperimentRunner> BuildExperimentRunner() override;
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

  metrics.SetIntegerIntegerMapMetric("metric int_int_map_rvalue",
                                     {{1, 1}, {2, 2}, {3, 3}});

  absl::flat_hash_map<int64_t, int64_t> int_int_map_rvalue;
  XLS_ASSERT_OK_AND_ASSIGN(
      int_int_map_rvalue,
      metrics.GetIntegerIntegerMapMetric("metric int_int_map_rvalue"));

  const absl::flat_hash_map<int64_t, int64_t> int_int_map_expected = {
      {1, 1}, {2, 2}, {3, 3}};
  EXPECT_THAT(int_int_map_rvalue, ::testing::UnorderedPointwise(
                                      ::testing::Eq(), int_int_map_expected));

  const absl::flat_hash_map<int64_t, int64_t> int_int_map_lvalue = {
      {1, 1}, {2, 2}, {3, 3}};
  metrics.SetIntegerIntegerMapMetric("metric int_int_map_lvalue",
                                     int_int_map_lvalue);
  EXPECT_THAT(int_int_map_lvalue, ::testing::UnorderedPointwise(
                                      ::testing::Eq(), int_int_map_expected));

  XLS_EXPECT_OK(metrics.DebugDump());
}

TEST(ExperimentsTest, ExperimentInfo) {
  ExperimentInfo experiment_info;
  std::vector<TimedRouteInfo> timed_route_info_a_expected;
  TimedRouteInfo timed_route_info_a_0;
  timed_route_info_a_0.route.emplace_back(
      TimedRouteItem{NetworkComponentId(0, 0), 0});
  timed_route_info_a_0.route.emplace_back(
      TimedRouteItem{NetworkComponentId(0, 1), 1});
  timed_route_info_a_0.route.emplace_back(
      TimedRouteItem{NetworkComponentId(0, 2), 2});
  TimedRouteInfo timed_route_info_a_1;
  timed_route_info_a_0.route.emplace_back(
      TimedRouteItem{NetworkComponentId(0, 3), 3});
  timed_route_info_a_expected.emplace_back(timed_route_info_a_0);
  timed_route_info_a_expected.emplace_back(timed_route_info_a_1);
  // Add two entries for "timed_route_info_a".
  experiment_info.AppendTimedRouteInfo("timed_route_info_a",
                                       std::move(timed_route_info_a_0));
  experiment_info.AppendTimedRouteInfo("timed_route_info_a",
                                       std::move(timed_route_info_a_1));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<TimedRouteInfo> timed_route_info_a,
      experiment_info.GetTimedRouteInfo("timed_route_info_a"));
  EXPECT_THAT(timed_route_info_a,
              ::testing::ContainerEq(timed_route_info_a_expected));
  // Fail as "timed_route_info_b" is not present.
  EXPECT_THAT(
      experiment_info.GetTimedRouteInfo("timed_route_info_b"),
      StatusIs(absl::StatusCode::kNotFound, "timed_route_info_b not found."));
  std::vector<TimedRouteInfo> timed_route_info_b_expected;
  TimedRouteInfo timed_route_info_b_0;
  timed_route_info_b_0.route.emplace_back(
      TimedRouteItem{NetworkComponentId(1, 42), 42});
  timed_route_info_b_expected.emplace_back(timed_route_info_b_0);
  // Add an entry for "timed_route_info_b".
  experiment_info.AppendTimedRouteInfo("timed_route_info_b",
                                       std::move(timed_route_info_b_0));
  // "timed_route_info_b" is now present.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<TimedRouteInfo> timed_route_info_b,
      experiment_info.GetTimedRouteInfo("timed_route_info_b"));
  EXPECT_THAT(timed_route_info_b,
              ::testing::ContainerEq(timed_route_info_b_expected));
}

}  // namespace

namespace internal {
namespace {

// Returns the TimedDataFlits of two packets.
std::vector<TimedDataFlit> GetPacketsForTest() {
  return std::vector<TimedDataFlit>{
      // packet 0
      {.cycle = 10,
       .flit = DataFlit{.type = FlitType::kHead, .vc = 0},
       .metadata = {.injection_cycle_time = 0}},
      {.cycle = 11,
       .flit = DataFlit{.type = FlitType::kTail, .vc = 0},
       .metadata = {.injection_cycle_time = 1}},
      // packet 1
      {.cycle = 12,
       .flit = DataFlit{.type = FlitType::kTail, .vc = 0},
       .metadata = {.injection_cycle_time = 2}},
      // invalid packet
      {.cycle = 13,
       .flit = DataFlit{.type = FlitType::kInvalid, .vc = 0},
       .metadata = {.injection_cycle_time = 3}},
  };
}

TEST(ExperimentsTest, GetPacketInfoTest) {
  std::vector<PacketInfo> packet_info =
      GetPacketInfo(GetPacketsForTest(), /*vc_index=*/0);
  // only two packets
  EXPECT_EQ(packet_info.size(), 2);
  EXPECT_EQ(packet_info[0].injection_clock_cycle_time, 0);
  EXPECT_EQ(packet_info[0].arrival_clock_cycle_time, 11);
  EXPECT_EQ(packet_info[1].injection_clock_cycle_time, 2);
  EXPECT_EQ(packet_info[1].arrival_clock_cycle_time, 12);
}

TEST(ExperimentsTest, GetStatsTest) {
  std::vector<TimedDataFlit> packets = GetPacketsForTest();
  std::vector<PacketInfo> packet_info = GetPacketInfo(packets, /*vc_index=*/0);
  Stats result = GetStats(packet_info);
  // packet 0 sent at cycle 0
  EXPECT_EQ(result.min_injection_cycle_time, 0);
  // packet 1 sent at cycle 2
  EXPECT_EQ(result.max_injection_cycle_time, 2);
  // packet 0 arrives at cycle 11
  EXPECT_EQ(result.min_arrival_cycle_time, 11);
  // packet 1 arrives at cycle 12
  EXPECT_EQ(result.max_arrival_cycle_time, 12);
  // packet 1 took 10 cycles
  EXPECT_EQ(result.min_latency, 10);
  // packet 0 took 11 cycles
  EXPECT_EQ(result.max_latency, 11);
  // latency histogram
  EXPECT_EQ(result.latency_histogram[10], 1);
  EXPECT_EQ(result.latency_histogram[11], 1);
  // average latency
  EXPECT_DOUBLE_EQ(result.average_latency, 10.5);
}

}  // namespace
}  // namespace internal

}  // namespace xls::noc
