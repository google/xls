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

#include "xls/noc/simulation/sim_objects.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/simulation/flit.h"
#include "xls/noc/simulation/network_graph_builder.h"
#include "xls/noc/simulation/sample_network_graphs.h"

namespace xls {
namespace noc {
namespace {

TEST(SimObjectsTest, BackToBackNetwork0) {
  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphLinear000(&proto, &graph, &params));

  // Validate the network.
  ASSERT_EQ(graph.GetNetworkIds().size(), 1);
  EXPECT_EQ(params.GetNetworkParam(graph.GetNetworkIds()[0])->GetName(),
            "Test");
  LOG(INFO) << "Network Checks Complete";

  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId routera_id,
      FindNetworkComponentByName("RouterA", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(PortId ain0, FindPortByName("Ain0", graph, params));
  EXPECT_EQ(ain0, graph.GetNetworkComponent(routera_id).GetPortIdByIndex(0));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  EXPECT_EQ(routing_table.GetSourceIndices().NetworkComponentCount(), 1);
  EXPECT_EQ(routing_table.GetSinkIndices().NetworkComponentCount(), 1);

  // Build simulator objects.
  NocSimulator simulator;
  XLS_ASSERT_OK(simulator.Initialize(graph, params, routing_table,
                                     graph.GetNetworkIds()[0]));
  simulator.Dump();

  // Retrieve src and sink objects
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recv_port_0,
      FindNetworkComponentByName("RecvPort0", graph, params));

  // Retrieve routers
  EXPECT_EQ(simulator.GetRouters().size(), 1);
  EXPECT_EQ(simulator.GetRouters()[0].GetUtilizationCycleCount(), 0);

  // phit0 traverses a link of latency 4 so will arrive on cycle 5.
  XLS_ASSERT_OK_AND_ASSIGN(
      int64_t dest_index_0,
      simulator.GetRoutingTable()->GetSinkIndices().GetNetworkComponentIndex(
          recv_port_0));

  XLS_ASSERT_OK_AND_ASSIGN(TimedDataFlit flit0,
                           DataFlitBuilder()
                               .Cycle(1)
                               .Type(FlitType::kTail)
                               .VirtualChannel(0)
                               .SourceIndex(0)
                               .DestinationIndex(dest_index_0)
                               .Data(UBits(707, 64))
                               .Cycle(1)
                               .BuildTimedFlit());

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSrc * sim_send_port_0,
                           simulator.GetSimNetworkInterfaceSrc(send_port_0));
  XLS_ASSERT_OK(sim_send_port_0->SendFlitAtTime(flit0));

  // Queue up a couple of more flits, each should arrive in subsquent cycles.
  for (int64_t i = 0; i < 4; ++i) {
    XLS_ASSERT_OK(sim_send_port_0->SendFlitAtTime(flit0));
  }

  for (int64_t i = 0; i < 15; ++i) {
    XLS_ASSERT_OK(simulator.RunCycle());
  }

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSink * sim_recv_port_0,
                           simulator.GetSimNetworkInterfaceSink(recv_port_0));

  absl::Span<const TimedDataFlit> traffic_recv_port_0 =
      sim_recv_port_0->GetReceivedTraffic();

  EXPECT_EQ(traffic_recv_port_0.size(), 5);
  EXPECT_EQ(traffic_recv_port_0[0].cycle, 5);
  EXPECT_EQ(traffic_recv_port_0[0].flit.data, UBits(707, 64));
  EXPECT_EQ(traffic_recv_port_0[4].cycle, 9);
  EXPECT_EQ(traffic_recv_port_0[4].flit.data, UBits(707, 64));
}

TEST(SimObjectsTest, TreeNetwork0) {
  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphTree000(&proto, &graph, &params));

  // Validate the network.
  ASSERT_EQ(graph.GetNetworkIds().size(), 1);
  EXPECT_EQ(params.GetNetworkParam(graph.GetNetworkIds()[0])->GetName(),
            "Test");
  LOG(INFO) << "Network Graph Complete ...";

  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId routera_id,
      FindNetworkComponentByName("RouterA", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId routerb_id,
      FindNetworkComponentByName("RouterB", graph, params));

  XLS_ASSERT_OK_AND_ASSIGN(PortId ain0, FindPortByName("Ain0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(PortId bin1, FindPortByName("Bin1", graph, params));

  EXPECT_EQ(ain0, graph.GetNetworkComponent(routera_id).GetPortIdByIndex(0));
  EXPECT_EQ(bin1, graph.GetNetworkComponent(routerb_id).GetPortIdByIndex(1));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  EXPECT_EQ(routing_table.GetSourceIndices().NetworkComponentCount(), 3);
  EXPECT_EQ(routing_table.GetSinkIndices().NetworkComponentCount(), 4);

  // Build simulator objects.
  NocSimulator simulator;
  XLS_ASSERT_OK(simulator.Initialize(graph, params, routing_table,
                                     graph.GetNetworkIds()[0]));
  simulator.Dump();

  // Retrieve src and sink objects
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId send_port_1,
      FindNetworkComponentByName("SendPort1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recv_port_0,
      FindNetworkComponentByName("RecvPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recv_port_1,
      FindNetworkComponentByName("RecvPort1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recv_port_2,
      FindNetworkComponentByName("RecvPort2", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recv_port_3,
      FindNetworkComponentByName("RecvPort3", graph, params));

  XLS_ASSERT_OK_AND_ASSIGN(
      int64_t dest_index_1,
      simulator.GetRoutingTable()->GetSinkIndices().GetNetworkComponentIndex(
          recv_port_1));
  XLS_ASSERT_OK_AND_ASSIGN(
      int64_t dest_index_3,
      simulator.GetRoutingTable()->GetSinkIndices().GetNetworkComponentIndex(
          recv_port_3));

  // Retrieve routers
  EXPECT_EQ(simulator.GetRouters().size(), 2);
  EXPECT_EQ(simulator.GetRouters()[0].GetUtilizationCycleCount(), 0);
  EXPECT_EQ(simulator.GetRouters()[1].GetUtilizationCycleCount(), 0);

  // phit0 traverses a link of latency 2 so will arrive on cycle 3.
  XLS_ASSERT_OK_AND_ASSIGN(TimedDataFlit flit0,
                           DataFlitBuilder()
                               .Cycle(1)
                               .Type(FlitType::kTail)
                               .VirtualChannel(0)
                               .SourceIndex(0)
                               .DestinationIndex(dest_index_1)
                               .Data(UBits(707, 64))
                               .Cycle(1)
                               .BuildTimedFlit());

  // phit1 will be blocked by phit0 so it will actually arrive on cycle 2.
  XLS_ASSERT_OK_AND_ASSIGN(TimedDataFlit flit1,
                           DataFlitBuilder()
                               .Cycle(1)
                               .Type(FlitType::kTail)
                               .VirtualChannel(1)
                               .SourceIndex(0)
                               .DestinationIndex(dest_index_3)
                               .Data(UBits(1001, 64))
                               .BuildTimedFlit());

  // phit2 goes though all router bypass paths so will arrive on cycle 3.
  XLS_ASSERT_OK_AND_ASSIGN(TimedDataFlit flit2,
                           DataFlitBuilder()
                               .Cycle(3)
                               .Type(FlitType::kTail)
                               .VirtualChannel(1)
                               .SourceIndex(0)
                               .DestinationIndex(dest_index_3)
                               .Data(UBits(2002, 64))
                               .BuildTimedFlit());

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSrc * sim_send_port_0,
                           simulator.GetSimNetworkInterfaceSrc(send_port_0));
  XLS_ASSERT_OK(sim_send_port_0->SendFlitAtTime(flit0));

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSrc * sim_send_port_1,
                           simulator.GetSimNetworkInterfaceSrc(send_port_1));
  XLS_ASSERT_OK(sim_send_port_1->SendFlitAtTime(flit1));
  XLS_ASSERT_OK(sim_send_port_1->SendFlitAtTime(flit2));

  for (int64_t i = 0; i < 10; ++i) {
    XLS_ASSERT_OK(simulator.RunCycle());
  }

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSink * sim_recv_port_0,
                           simulator.GetSimNetworkInterfaceSink(recv_port_0));
  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSink * sim_recv_port_1,
                           simulator.GetSimNetworkInterfaceSink(recv_port_1));
  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSink * sim_recv_port_2,
                           simulator.GetSimNetworkInterfaceSink(recv_port_2));
  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSink * sim_recv_port_3,
                           simulator.GetSimNetworkInterfaceSink(recv_port_3));

  absl::Span<const TimedDataFlit> traffic_recv_port_0 =
      sim_recv_port_0->GetReceivedTraffic();
  absl::Span<const TimedDataFlit> traffic_recv_port_1 =
      sim_recv_port_1->GetReceivedTraffic();
  absl::Span<const TimedDataFlit> traffic_recv_port_2 =
      sim_recv_port_2->GetReceivedTraffic();
  absl::Span<const TimedDataFlit> traffic_recv_port_3 =
      sim_recv_port_3->GetReceivedTraffic();

  EXPECT_EQ(traffic_recv_port_0.size(), 0);
  EXPECT_EQ(traffic_recv_port_1.size(), 1);
  EXPECT_EQ(traffic_recv_port_2.size(), 0);
  EXPECT_EQ(traffic_recv_port_3.size(), 2);

  EXPECT_EQ(traffic_recv_port_1[0].cycle, 3);
  EXPECT_EQ(traffic_recv_port_1[0].flit.data, UBits(707, 64));
  EXPECT_EQ(traffic_recv_port_3[0].cycle, 2);
  EXPECT_EQ(traffic_recv_port_3[0].flit.data, UBits(1001, 64));
  EXPECT_EQ(traffic_recv_port_3[1].cycle, 3);
  EXPECT_EQ(traffic_recv_port_3[1].flit.data, UBits(2002, 64));

  // Measure traffic received on all VCs.
  EXPECT_DOUBLE_EQ(sim_recv_port_0->MeasuredTrafficRateInMiBps(100), 0.0);
  EXPECT_DOUBLE_EQ(sim_recv_port_2->MeasuredTrafficRateInMiBps(100), 0.0);
  EXPECT_EQ(
      static_cast<int64_t>(sim_recv_port_1->MeasuredTrafficRateInMiBps(100)),
      19073);
  EXPECT_EQ(
      static_cast<int64_t>(sim_recv_port_3->MeasuredTrafficRateInMiBps(100)),
      38146);

  // Measure traffic received on VC 0.
  EXPECT_DOUBLE_EQ(sim_recv_port_0->MeasuredTrafficRateInMiBps(100, 0), 0.0);
  EXPECT_DOUBLE_EQ(sim_recv_port_2->MeasuredTrafficRateInMiBps(100, 0), 0.0);
  EXPECT_EQ(
      static_cast<int64_t>(sim_recv_port_1->MeasuredTrafficRateInMiBps(100, 0)),
      19073);
  EXPECT_DOUBLE_EQ(sim_recv_port_3->MeasuredTrafficRateInMiBps(100, 0), 0.0);

  // Measure traffic received on VC 1.
  EXPECT_DOUBLE_EQ(sim_recv_port_0->MeasuredTrafficRateInMiBps(100, 1), 0.0);
  EXPECT_DOUBLE_EQ(sim_recv_port_2->MeasuredTrafficRateInMiBps(100, 1), 0.0);
  EXPECT_DOUBLE_EQ(sim_recv_port_1->MeasuredTrafficRateInMiBps(100, 1), 0.0);
  EXPECT_EQ(
      static_cast<int64_t>(sim_recv_port_3->MeasuredTrafficRateInMiBps(100, 1)),
      38146);
}

}  // namespace
}  // namespace noc
}  // namespace xls
