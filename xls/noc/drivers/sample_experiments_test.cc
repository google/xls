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

#include "xls/noc/drivers/sample_experiments.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/drivers/experiment_factory.h"

namespace xls::noc {
namespace {

TEST(SampleExperimentsTest, SimpleVCExperiment) {
  ExperimentFactory experiment_factory;
  XLS_ASSERT_OK(RegisterSampleExperiments(experiment_factory));

  EXPECT_EQ(experiment_factory.ListExperimentTags().size(), 2);
  EXPECT_EQ(experiment_factory.ListExperimentTags().at(1),
            "SimpleVCExperiment");

  XLS_ASSERT_OK_AND_ASSIGN(
      Experiment experiment,
      experiment_factory.BuildExperiment("SimpleVCExperiment"));

  EXPECT_EQ(
      experiment.GetBaseConfig().GetTrafficConfig().GetTrafficFlowIds().size(),
      2);
  XLS_EXPECT_OK(
      experiment.GetBaseConfig().GetTrafficConfig().GetTrafficFlowIdByName(
          "flow_0"));
  XLS_EXPECT_OK(
      experiment.GetBaseConfig().GetTrafficConfig().GetTrafficFlowIdByName(
          "flow_1"));
  XLS_EXPECT_OK(
      experiment.GetBaseConfig().GetTrafficConfig().GetTrafficModeIdByName(
          "Main"));

  EXPECT_EQ(experiment.GetSweeps().GetStepCount(), 4);

  EXPECT_EQ(experiment.GetRunner().GetTrafficMode(), "Main");
  EXPECT_EQ(experiment.GetRunner().GetSimulationCycleCount(), 100'000);
  EXPECT_EQ(experiment.GetRunner().GetSeed(), 100);
  EXPECT_EQ(experiment.GetRunner().GetCycleTimeInPs(), 500);

  std::vector<ExperimentData> experiment_data(4);
  for (int64_t i = 0; i < experiment.GetSweeps().GetStepCount(); ++i) {
    LOG(INFO) << absl::StreamFormat("Experiment Step %d", i);
    XLS_ASSERT_OK_AND_ASSIGN(experiment_data.at(i), experiment.RunStep(i));
    XLS_EXPECT_OK(experiment_data.at(i).metrics.DebugDump());
  }

  // Flow 0 @ 3 GBps and Flow 1 @ 2 GBps

  // Experiment 0 - Both on VC0
  XLS_ASSERT_OK_AND_ASSIGN(double ex0_flow0_traffic_rate,
                           experiment_data.at(0).metrics.GetFloatMetric(
                               "Flow:flow_0:TrafficRateInMiBps"));
  XLS_ASSERT_OK_AND_ASSIGN(double ex0_flow1_traffic_rate,
                           experiment_data.at(0).metrics.GetFloatMetric(
                               "Flow:flow_1:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(ex0_flow0_traffic_rate) / 100, 30);
  EXPECT_EQ(static_cast<int64_t>(ex0_flow1_traffic_rate) / 100, 20);

  XLS_ASSERT_OK_AND_ASSIGN(double ex0_traffic_rate,
                           experiment_data.at(0).metrics.GetFloatMetric(
                               "Sink:RecvPort0:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(ex0_traffic_rate) / 100, 51);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t ex0_flits,
                           experiment_data.at(0).metrics.GetIntegerMetric(
                               "Sink:RecvPort0:FlitCount"));
  EXPECT_EQ(ex0_flits / 1000, 16);

  XLS_ASSERT_OK_AND_ASSIGN(double ex0_vc0_traffic_rate,
                           experiment_data.at(0).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:0:TrafficRateInMiBps"));
  XLS_ASSERT_OK_AND_ASSIGN(double ex0_vc1_traffic_rate,
                           experiment_data.at(0).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:1:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(ex0_vc0_traffic_rate) / 100, 51);
  EXPECT_DOUBLE_EQ(ex0_vc1_traffic_rate, 0.0);

  // Experiment 1 - Flow 0 on VC0, Flow 1 on VC1
  XLS_ASSERT_OK_AND_ASSIGN(double ex1_vc0_traffic_rate,
                           experiment_data.at(1).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:0:TrafficRateInMiBps"));
  XLS_ASSERT_OK_AND_ASSIGN(double ex1_vc1_traffic_rate,
                           experiment_data.at(1).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:1:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(ex1_vc0_traffic_rate) / 100, 30);
  EXPECT_EQ(static_cast<int64_t>(ex1_vc1_traffic_rate) / 100, 20);

  // Experiment 2 - Flow 0 on VC0, Flow 1 on VC1, but network can only
  // Support support 16 bits / cycle @ 500ps cycle == 3814 MiBps
  XLS_ASSERT_OK_AND_ASSIGN(double ex2_vc0_traffic_rate,
                           experiment_data.at(2).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:0:TrafficRateInMiBps"));
  XLS_ASSERT_OK_AND_ASSIGN(double ex2_vc1_traffic_rate,
                           experiment_data.at(2).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:1:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(ex2_vc0_traffic_rate) / 100, 38);
  EXPECT_DOUBLE_EQ(ex2_vc1_traffic_rate, 0.0);

  // Experiment 3 - Flow 0 on VC0, Flow 1 on VC1, and VC0 gets priority
  XLS_ASSERT_OK_AND_ASSIGN(double ex3_vc0_traffic_rate,
                           experiment_data.at(3).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:0:TrafficRateInMiBps"));
  XLS_ASSERT_OK_AND_ASSIGN(double ex3_vc1_traffic_rate,
                           experiment_data.at(3).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:1:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(ex3_vc0_traffic_rate) / 100, 30);
  EXPECT_EQ(static_cast<int64_t>(ex3_vc1_traffic_rate) / 100, 7);

  // Get route info for VC 1 of RecvPort0 for the four experiments.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<TimedRouteInfo> timed_route_info_0,
                           experiment_data.at(0).info.GetTimedRouteInfo(
                               "Sink:RecvPort0:VC:1:TimedRouteInfo"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<TimedRouteInfo> timed_route_info_1,
                           experiment_data.at(1).info.GetTimedRouteInfo(
                               "Sink:RecvPort0:VC:1:TimedRouteInfo"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<TimedRouteInfo> timed_route_info_2,
                           experiment_data.at(2).info.GetTimedRouteInfo(
                               "Sink:RecvPort0:VC:1:TimedRouteInfo"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<TimedRouteInfo> timed_route_info_3,
                           experiment_data.at(3).info.GetTimedRouteInfo(
                               "Sink:RecvPort0:VC:1:TimedRouteInfo"));
  EXPECT_THAT(timed_route_info_0.size(), ::testing::Gt(16800));
  EXPECT_THAT(timed_route_info_2.size(), ::testing::Gt(99900));
  EXPECT_EQ(timed_route_info_0.size(), timed_route_info_1.size());
  EXPECT_EQ(timed_route_info_2.size(), timed_route_info_3.size());

  // Get link info for the first experiments.
  // The links contains the same result.
  const absl::btree_map<std::string, SinkVcPairPacketCount>&
      link_to_packet_count_map =
          experiment_data.at(0).info.GetLinkToPacketCountMap();
  EXPECT_EQ(link_to_packet_count_map.size(), 2);
  EXPECT_EQ(link_to_packet_count_map.at("LinkA0").size(), 1);
  EXPECT_EQ(link_to_packet_count_map.at("Link0A").size(), 1);
  EXPECT_EQ(link_to_packet_count_map.at("LinkA0").begin()->second,
            link_to_packet_count_map.at("Link0A").begin()->second);
  EXPECT_EQ(
      link_to_packet_count_map.at("LinkA0").at(SinkVcPair{"RecvPort0", "VC0"}),
      link_to_packet_count_map.at("Link0A").begin()->second);
}

TEST(SampleExperimentsTest, AggregateTreeTest) {
  ExperimentFactory experiment_factory;
  XLS_ASSERT_OK(RegisterSampleExperiments(experiment_factory));

  EXPECT_EQ(experiment_factory.ListExperimentTags().size(), 2);
  EXPECT_EQ(experiment_factory.ListExperimentTags().at(0),
            "AggregateTreeExperiment");

  XLS_ASSERT_OK_AND_ASSIGN(
      Experiment experiment,
      experiment_factory.BuildExperiment("AggregateTreeExperiment"));

  TrafficFlowId flow0_id =
      experiment.GetBaseConfig().GetTrafficConfig().GetTrafficFlowIds().at(0);
  EXPECT_EQ(experiment.GetBaseConfig()
                .GetTrafficConfig()
                .GetTrafficFlow(flow0_id)
                .GetBandwidthBits(),
            8l * 1024 * 1024 * 1024);

  int64_t step_count = experiment.GetSweeps().GetStepCount();

  std::vector<ExperimentData> experiment_data(step_count);
  for (int64_t i = 0; i < step_count; ++i) {
    LOG(INFO) << absl::StreamFormat("Experiment Step %d", i);
    XLS_ASSERT_OK_AND_ASSIGN(experiment_data.at(i), experiment.RunStep(i));
    XLS_EXPECT_OK(experiment_data.at(i).metrics.DebugDump());
  }

  // Max rate used is 16 flows each at 1GBps.
  XLS_ASSERT_OK_AND_ASSIGN(double step0_traffic_rate,
                           experiment_data.at(0).metrics.GetFloatMetric(
                               "Sink:RecvPort0:VC:0:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(step0_traffic_rate) / 1000, 16);

  XLS_ASSERT_OK_AND_ASSIGN(double step0_flow0_traffic_rate,
                           experiment_data.at(0).metrics.GetFloatMetric(
                               "Flow:flow_0:TrafficRateInMiBps"));
  EXPECT_EQ(static_cast<int64_t>(step0_flow0_traffic_rate) / 100, 10);

  // As we go in the steps, phit width decreases so traffic rate
  // will either stay the same or decrease
  int64_t prior_traffic_rate = static_cast<int64_t>(step0_traffic_rate) / 100;
  for (int64_t i = 1; i < step_count; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(double traffic_rate,
                             experiment_data.at(i).metrics.GetFloatMetric(
                                 "Sink:RecvPort0:VC:0:TrafficRateInMiBps"));
    int64_t next_traffic_rate = static_cast<int64_t>(traffic_rate) / 100;

    EXPECT_GE(prior_traffic_rate, next_traffic_rate);
    prior_traffic_rate = next_traffic_rate;
  }
}

}  // namespace
}  // namespace xls::noc
