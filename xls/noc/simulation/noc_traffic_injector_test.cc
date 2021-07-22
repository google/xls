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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/simulation/packetizer.h"
#include "xls/noc/simulation/sample_network_graphs.h"
#include "xls/noc/simulation/sim_objects.h"
#include "xls/noc/simulation/traffic_description.h"

namespace xls::noc {
namespace {

TEST(NocTrafficInjectorTest, SingleSource) {
  // Construct traffic flows
  NocTrafficManager traffic_mgr;

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id,
                           traffic_mgr.CreateTrafficFlow());
  TrafficFlow& flow0 = traffic_mgr.GetTrafficFlow(flow0_id);
  flow0.SetName("flow0")
      .SetSource("SendPort0")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(4 * 1024)
      .SetPacketSizeInBits(128)
      .SetBurstProbInMils(7);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode0_id,
                           traffic_mgr.CreateTrafficMode());
  TrafficMode& mode0 = traffic_mgr.GetTrafficMode(mode0_id);
  mode0.SetName("Mode 0").RegisterTrafficFlow(flow0_id);

  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphLinear000(&proto, &graph, &params));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  // Build input traffic model
  RandomNumberInterface rnd;
  int64_t cycle_time_in_ps = 400;
  rnd.SetSeed(1000);
  XLS_ASSERT_OK_AND_ASSIGN(
      NocTrafficInjector traffic_injector,
      NocTrafficInjectorBuilder().Build(
          cycle_time_in_ps, mode0_id,
          routing_table.GetSourceIndices().GetNetworkComponents(), traffic_mgr,
          graph, params, rnd));

  EXPECT_EQ(traffic_injector.FlowCount(), 1);
  EXPECT_EQ(traffic_injector.SourceNetworkInterfaceCount(), 1);

  ASSERT_EQ(traffic_injector.GetDePacketizers().size(), 1);
  ASSERT_EQ(traffic_injector.GetSourceNetworkInterfaces().size(), 1);
  ASSERT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().size(), 1);
  ASSERT_EQ(traffic_injector.GetTrafficModels().size(), 1);

  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetSourceIndexBitCount(),
            0);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetFlitPayloadBitCount(),
            64);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetMaxPacketBitCount(),
            128);

  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_0,
      FindNetworkComponentByName("SendPort0", graph, params));
  EXPECT_EQ(traffic_injector.GetSourceNetworkInterfaces().at(0), send_port_0);
  EXPECT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().at(0), 0);

  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(0).GetPacketSizeInBits(), 128);
  EXPECT_DOUBLE_EQ(traffic_injector.GetTrafficModels().at(0).GetBurstProb(),
                   0.007);
  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(0).ExpectedTrafficRateInMiBps(
          cycle_time_in_ps),
      4.0 * 1024.0);
}

TEST(NocTrafficInjectorTest, SingleSourceTwoFlows) {
  // Construct traffic flows
  NocTrafficManager traffic_mgr;

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow0_id)
      .SetName("flow0")
      .SetSource("SendPort0")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(4 * 1024)
      .SetPacketSizeInBits(128)
      .SetBurstProbInMils(7);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow1_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow1_id)
      .SetName("flow1")
      .SetSource("SendPort0")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(18 * 1024)
      .SetPacketSizeInBits(256)
      .SetBurstProbInMils(0);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode0_id,
                           traffic_mgr.CreateTrafficMode());
  traffic_mgr.GetTrafficMode(mode0_id)
      .SetName("Mode 0")
      .RegisterTrafficFlow(flow0_id)
      .RegisterTrafficFlow(flow1_id);

  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphLinear000(&proto, &graph, &params));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  // Build input traffic model
  RandomNumberInterface rnd;
  int64_t cycle_time_in_ps = 400;
  rnd.SetSeed(1000);
  XLS_ASSERT_OK_AND_ASSIGN(
      NocTrafficInjector traffic_injector,
      NocTrafficInjectorBuilder().Build(
          cycle_time_in_ps, mode0_id,
          routing_table.GetSourceIndices().GetNetworkComponents(), traffic_mgr,
          graph, params, rnd));

  EXPECT_EQ(traffic_injector.FlowCount(), 2);
  EXPECT_EQ(traffic_injector.SourceNetworkInterfaceCount(), 1);

  ASSERT_EQ(traffic_injector.GetDePacketizers().size(), 1);
  ASSERT_EQ(traffic_injector.GetSourceNetworkInterfaces().size(), 1);
  ASSERT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().size(), 2);
  ASSERT_EQ(traffic_injector.GetTrafficModels().size(), 2);

  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetSourceIndexBitCount(),
            0);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetFlitPayloadBitCount(),
            64);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetMaxPacketBitCount(),
            256);

  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_0,
      FindNetworkComponentByName("SendPort0", graph, params));
  EXPECT_EQ(traffic_injector.GetSourceNetworkInterfaces().at(0), send_port_0);
  EXPECT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().at(0), 0);
  EXPECT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().at(1), 0);

  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(0).GetPacketSizeInBits(), 128);
  EXPECT_DOUBLE_EQ(traffic_injector.GetTrafficModels().at(0).GetBurstProb(),
                   0.007);
  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(0).ExpectedTrafficRateInMiBps(
          cycle_time_in_ps),
      4.0 * 1024.0);

  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(1).GetPacketSizeInBits(), 256);
  EXPECT_DOUBLE_EQ(traffic_injector.GetTrafficModels().at(1).GetBurstProb(),
                   0.000);
  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(1).ExpectedTrafficRateInMiBps(
          cycle_time_in_ps),
      18.0 * 1024.0);
}

TEST(NocTrafficInjectorTest, TwoSourcesThreeFlows) {
  // Construct traffic flows
  NocTrafficManager traffic_mgr;

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow0_id)
      .SetName("flow0")
      .SetSource("SendPort0")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(4 * 1024)
      .SetPacketSizeInBits(128)
      .SetBurstProbInMils(7);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow1_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow1_id)
      .SetName("flow1")
      .SetSource("SendPort1")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(18 * 1024)
      .SetPacketSizeInBits(256)
      .SetBurstProbInMils(1);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow2_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow2_id)
      .SetName("flow2")
      .SetSource("SendPort0")
      .SetDestination("RecvPort1")
      .SetTrafficRateInMiBps(2 * 1024)
      .SetPacketSizeInBits(64)
      .SetBurstProbInMils(2);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode0_id,
                           traffic_mgr.CreateTrafficMode());
  traffic_mgr.GetTrafficMode(mode0_id)
      .SetName("Mode 0")
      .RegisterTrafficFlow(flow0_id)
      .RegisterTrafficFlow(flow1_id)
      .RegisterTrafficFlow(flow2_id);

  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphTree001(&proto, &graph, &params));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  // Build input traffic model
  RandomNumberInterface rnd;
  int64_t cycle_time_in_ps = 1000;
  rnd.SetSeed(1000);
  XLS_ASSERT_OK_AND_ASSIGN(
      NocTrafficInjector traffic_injector,
      NocTrafficInjectorBuilder().Build(
          cycle_time_in_ps, mode0_id,
          routing_table.GetSourceIndices().GetNetworkComponents(), traffic_mgr,
          graph, params, rnd));

  EXPECT_EQ(traffic_injector.FlowCount(), 3);
  EXPECT_EQ(traffic_injector.SourceNetworkInterfaceCount(), 2);

  ASSERT_EQ(traffic_injector.GetDePacketizers().size(), 2);
  ASSERT_EQ(traffic_injector.GetSourceNetworkInterfaces().size(), 2);
  ASSERT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().size(), 3);
  ASSERT_EQ(traffic_injector.GetTrafficModels().size(), 3);

  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetSourceIndexBitCount(),
            1);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetFlitPayloadBitCount(),
            128);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(0).GetMaxPacketBitCount(),
            128);

  EXPECT_EQ(traffic_injector.GetDePacketizers().at(1).GetSourceIndexBitCount(),
            1);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(1).GetFlitPayloadBitCount(),
            128);
  EXPECT_EQ(traffic_injector.GetDePacketizers().at(1).GetMaxPacketBitCount(),
            256);

  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_1,
      FindNetworkComponentByName("SendPort1", graph, params));

  EXPECT_EQ(traffic_injector.GetSourceNetworkInterfaces().at(0), send_port_0);
  EXPECT_EQ(traffic_injector.GetSourceNetworkInterfaces().at(1), send_port_1);
  EXPECT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().at(0), 0);
  EXPECT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().at(1), 1);
  EXPECT_EQ(traffic_injector.GetFlowsIndexToSourcesIndexMap().at(2), 0);

  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(0).GetPacketSizeInBits(), 128);
  EXPECT_DOUBLE_EQ(traffic_injector.GetTrafficModels().at(0).GetBurstProb(),
                   0.007);
  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(0).ExpectedTrafficRateInMiBps(
          cycle_time_in_ps),
      4.0 * 1024.0);

  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(1).GetPacketSizeInBits(), 256);
  EXPECT_DOUBLE_EQ(traffic_injector.GetTrafficModels().at(1).GetBurstProb(),
                   0.001);
  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(1).ExpectedTrafficRateInMiBps(
          cycle_time_in_ps),
      18.0 * 1024.0);

  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(2).GetPacketSizeInBits(), 64);
  EXPECT_DOUBLE_EQ(traffic_injector.GetTrafficModels().at(2).GetBurstProb(),
                   0.002);
  EXPECT_DOUBLE_EQ(
      traffic_injector.GetTrafficModels().at(2).ExpectedTrafficRateInMiBps(
          cycle_time_in_ps),
      2.0 * 1024.0);
}

TEST(NocTrafficInjectorTest, MeasureTrafficInjectionRate) {
  // Construct traffic flows
  NocTrafficManager traffic_mgr;

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow0_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow0_id)
      .SetName("flow0")
      .SetSource("SendPort0")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(8 * 1024)
      .SetPacketSizeInBits(128)
      .SetBurstProbInMils(7);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow1_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow1_id)
      .SetName("flow1")
      .SetSource("SendPort1")
      .SetDestination("RecvPort0")
      .SetTrafficRateInMiBps(18 * 1024)
      .SetPacketSizeInBits(256)
      .SetBurstProbInMils(1);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId flow2_id,
                           traffic_mgr.CreateTrafficFlow());
  traffic_mgr.GetTrafficFlow(flow2_id)
      .SetName("flow2")
      .SetSource("SendPort0")
      .SetDestination("RecvPort1")
      .SetTrafficRateInMiBps(4 * 1024)
      .SetPacketSizeInBits(64)
      .SetBurstProbInMils(2);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId mode0_id,
                           traffic_mgr.CreateTrafficMode());
  traffic_mgr.GetTrafficMode(mode0_id)
      .SetName("Mode 0")
      .RegisterTrafficFlow(flow0_id)
      .RegisterTrafficFlow(flow1_id)
      .RegisterTrafficFlow(flow2_id);

  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphTree001(&proto, &graph, &params));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  // Build input traffic model
  RandomNumberInterface rnd;
  int64_t cycle_time_in_ps = 1000;
  rnd.SetSeed(1000);
  XLS_ASSERT_OK_AND_ASSIGN(
      NocTrafficInjector traffic_injector,
      NocTrafficInjectorBuilder().Build(
          cycle_time_in_ps, mode0_id,
          routing_table.GetSourceIndices().GetNetworkComponents(), traffic_mgr,
          graph, params, rnd));

  // Accepts flits and measures traffic from each source.
  class NocTrafficInjectorSink : public NocSimulatorTrafficServiceShim {
   public:
    // Not used in this test.
    absl::Status RunCycle() override { return absl::OkStatus(); }

    // Measures traffic sent from source.
    absl::Status SendFlitAtTime(TimedDataFlit flit,
                                NetworkComponentId source) override {
      if (flit.cycle > max_cycle_) {
        max_cycle_ = flit.cycle;
      }

      bits_sent_[source] += flit.flit.data_bit_count;

      XLS_VLOG(1) << absl::StrFormat(
          "  - Source %x Measured %d bits Now %d\n", source.AsUInt64(),
          flit.flit.data_bit_count, bits_sent_[source]);
      return absl::OkStatus();
    }

    // Returns observed rate of traffic in MebiBytes Per Second seen in all
    // previous calls to SendFlitAtTime().
    double MeasuredTrafficRateInMiBps(int64_t cycle_time_ps,
                                      NetworkComponentId source) {
      int64_t num_bits_sent = bits_sent_[source];

      double total_sec = static_cast<double>(max_cycle_ + 1) *
                         static_cast<double>(cycle_time_ps) * 1.0e-12;
      double bits_per_sec = static_cast<double>(num_bits_sent) / total_sec;
      return bits_per_sec / 1024.0 / 1024.0 / 8.0;
    }

   private:
    absl::flat_hash_map<NetworkComponentId, int64_t> bits_sent_;
    int64_t max_cycle_ = 0;
  };

  // Run NocTrafficInjector.
  NocTrafficInjectorSink sink;
  traffic_injector.SetSimulatorShim(sink);

  int64_t cycle = 0;
  for (cycle = 0; cycle < 1'000'000; ++cycle) {
    XLS_ASSERT_OK(traffic_injector.RunCycle());
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_1,
      FindNetworkComponentByName("SendPort1", graph, params));

  EXPECT_EQ(static_cast<int64_t>(sink.MeasuredTrafficRateInMiBps(
                cycle_time_in_ps, send_port_0)) /
                100,
            (8 + 4) * 1024 / 100);
  EXPECT_EQ(static_cast<int64_t>(sink.MeasuredTrafficRateInMiBps(
                cycle_time_in_ps, send_port_1)) /
                100,
            18 * 1024 / 100);

  EXPECT_DOUBLE_EQ(
      traffic_injector.MeasuredTraficRateInMiBps(cycle_time_in_ps, 0) +
          traffic_injector.MeasuredTraficRateInMiBps(cycle_time_in_ps, 2),
      sink.MeasuredTrafficRateInMiBps(cycle_time_in_ps, send_port_0));
  EXPECT_DOUBLE_EQ(
      traffic_injector.MeasuredTraficRateInMiBps(cycle_time_in_ps, 1),
      sink.MeasuredTrafficRateInMiBps(cycle_time_in_ps, send_port_1));
}

}  // namespace
}  // namespace xls::noc
