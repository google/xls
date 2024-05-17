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

#include "xls/noc/simulation/traffic_description.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/simulation/common.h"

namespace xls::noc {
namespace {

TEST(TrafficDescriptionTest, TrafficFlow) {
  NocTrafficManager traffic_mgr;

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id,
                           traffic_mgr.CreateTrafficFlow());
  TrafficFlow& flow0 = traffic_mgr.GetTrafficFlow(flow0_id);

  EXPECT_EQ(flow0.id().AsUInt64(), 0);
  EXPECT_EQ(flow0.GetSource(), "");
  EXPECT_EQ(flow0.GetDestination(), "");
  EXPECT_EQ(flow0.GetVC(), "");
  EXPECT_EQ(flow0.GetName(), "");
  EXPECT_DOUBLE_EQ(flow0.GetTrafficRateInBytesPerSec(), 0);
  EXPECT_DOUBLE_EQ(flow0.GetTrafficPerNumPsInBits(100), 0);
  EXPECT_DOUBLE_EQ(flow0.GetBurstPercent(), 0);
  EXPECT_DOUBLE_EQ(flow0.GetBurstProb(), 0);
  EXPECT_EQ(flow0.GetBandwidthBits(), 0);
  EXPECT_EQ(flow0.GetBandwidthPerNumPs(), 1);
  EXPECT_EQ(flow0.GetPacketSizeInBits(), 1);
  EXPECT_EQ(flow0.GetClockCycleTimes().size(), 0);
  EXPECT_FALSE(flow0.IsReplay());

  flow0.SetName("Flow 0");

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow1_id,
                           traffic_mgr.CreateTrafficFlow());
  TrafficFlow& flow1 = traffic_mgr.GetTrafficFlow(flow1_id)
                           .SetSource("source")
                           .SetDestination("destination")
                           .SetVC("VC0")
                           .SetName("Flow 1")
                           .SetTrafficRateInBitsPerPS(64, 500)
                           .SetPacketSizeInBits(100)
                           .SetBurstPercentInMils(100);

  EXPECT_EQ(flow1.id().AsUInt64(), 1);
  EXPECT_EQ(flow1.GetSource(), "source");
  EXPECT_EQ(flow1.GetDestination(), "destination");
  EXPECT_EQ(flow1.GetVC(), "VC0");
  EXPECT_DOUBLE_EQ(flow1.GetTrafficRateInBytesPerSec(), 8.0 / 500e-12);
  EXPECT_DOUBLE_EQ(flow1.GetTrafficPerNumPsInBits(100), 64.0 / 5.0);
  EXPECT_DOUBLE_EQ(flow1.GetBurstPercent(), 0.1);
  EXPECT_DOUBLE_EQ(flow1.GetBurstProb(), flow1.GetBurstPercent() / 100.0);
  EXPECT_EQ(flow1.GetBandwidthBits(), 64);
  EXPECT_EQ(flow1.GetBandwidthPerNumPs(), 500);
  EXPECT_EQ(flow1.GetPacketSizeInBits(), 100);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id_by_name,
                           traffic_mgr.GetTrafficFlowIdByName("Flow 0"));
  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow1_id_by_name,
                           traffic_mgr.GetTrafficFlowIdByName("Flow 1"));
  EXPECT_EQ(flow0_id_by_name, flow0_id);
  EXPECT_EQ(flow1_id_by_name, flow1_id);

  flow1.SetTrafficRateInMiBps(300);
  EXPECT_DOUBLE_EQ(flow1.GetTrafficRateInBytesPerSec(), 300.0 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(flow1.GetTrafficPerNumPsInBits(500),
                   300.0 * 1024.0 * 1024.0 * 8.0 / (1.0e12) * 500.0);
  EXPECT_DOUBLE_EQ(flow1.GetTrafficRateInMiBps(), 300.0);

  flow1.SetBurstProbInMils(3);
  EXPECT_DOUBLE_EQ(flow1.GetBurstProb(), 0.003);
  EXPECT_DOUBLE_EQ(flow1.GetBurstPercent(), 0.3);

  const NocTrafficManager& const_traffic_mgr = traffic_mgr;
  EXPECT_EQ(const_traffic_mgr.GetTrafficFlowIds().size(), 2);
  EXPECT_EQ(const_traffic_mgr.GetTrafficFlowIds().at(0), flow0_id);
  EXPECT_EQ(const_traffic_mgr.GetTrafficFlowIds().at(1), flow1_id);
  EXPECT_EQ(&const_traffic_mgr.GetTrafficFlow(flow0_id),
            &traffic_mgr.GetTrafficFlow(flow0_id));
  EXPECT_EQ(&const_traffic_mgr.GetTrafficFlow(flow1_id),
            &traffic_mgr.GetTrafficFlow(flow1_id));
}

TEST(TrafficDescriptionTest, ReplayType) {
  NocTrafficManager traffic_mgr;

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id,
                           traffic_mgr.CreateTrafficFlow());

  TrafficFlow& flow0 = traffic_mgr.GetTrafficFlow(flow0_id).SetClockCycleTimes(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  EXPECT_EQ(flow0.GetClockCycleTimes(),
            std::vector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_TRUE(flow0.IsReplay());
}

TEST(TrafficDescriptionTest, TrafficMode) {
  NocTrafficManager traffic_mgr;

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id,
                           traffic_mgr.CreateTrafficFlow());
  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow1_id,
                           traffic_mgr.CreateTrafficFlow());
  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow2_id,
                           traffic_mgr.CreateTrafficFlow());

  traffic_mgr.GetTrafficFlow(flow0_id).SetName("Flow 0");
  traffic_mgr.GetTrafficFlow(flow1_id).SetName("Flow 0");
  traffic_mgr.GetTrafficFlow(flow2_id).SetName("Flow 0");

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode0_id,
                           traffic_mgr.CreateTrafficMode());
  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode1_id,
                           traffic_mgr.CreateTrafficMode());

  TrafficMode& mode0 = traffic_mgr.GetTrafficMode(mode0_id);
  TrafficMode& mode1 = traffic_mgr.GetTrafficMode(mode1_id);

  mode0.SetName("Mode 0").RegisterTrafficFlow(flow0_id).RegisterTrafficFlow(
      flow2_id);
  mode1.SetName("Mode 1").RegisterTrafficFlow(flow2_id).RegisterTrafficFlow(
      flow1_id);

  EXPECT_TRUE(mode0.HasTrafficFlow(flow0_id));
  EXPECT_FALSE(mode0.HasTrafficFlow(flow1_id));
  EXPECT_TRUE(mode0.HasTrafficFlow(flow2_id));

  EXPECT_FALSE(mode1.HasTrafficFlow(flow0_id));
  EXPECT_TRUE(mode1.HasTrafficFlow(flow1_id));
  EXPECT_TRUE(mode1.HasTrafficFlow(flow2_id));

  EXPECT_EQ(mode0.GetTrafficFlows().at(1), flow2_id);
  EXPECT_EQ(mode1.GetTrafficFlows().at(0), flow2_id);

  EXPECT_EQ(mode0.GetName(), "Mode 0");
  EXPECT_EQ(mode1.GetName(), "Mode 1");

  EXPECT_EQ(&mode0.GetNocTrafficManager(), &traffic_mgr);
  EXPECT_EQ(&mode1.GetNocTrafficManager(), &traffic_mgr);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode0_id_by_name,
                           traffic_mgr.GetTrafficModeIdByName("Mode 0"));
  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode1_id_by_name,
                           traffic_mgr.GetTrafficModeIdByName("Mode 1"));
  EXPECT_EQ(mode0_id_by_name, mode0_id);
  EXPECT_EQ(mode1_id_by_name, mode1_id);

  const NocTrafficManager& const_traffic_mgr = traffic_mgr;
  EXPECT_EQ(&const_traffic_mgr.GetTrafficMode(mode0_id),
            &traffic_mgr.GetTrafficMode(mode0_id));
  EXPECT_EQ(&const_traffic_mgr.GetTrafficMode(mode1_id),
            &traffic_mgr.GetTrafficMode(mode1_id));
}

}  // namespace
}  // namespace xls::noc
