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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"
#include "xls/noc/simulation/network_graph_builder.h"

namespace xls {
namespace noc {
namespace {

TEST(SimObjectsTest, BackToBackNetwork0) {
  XLS_LOG(INFO) << "Setting up network ...";
  NetworkConfigProtoBuilder builder("Test");

  // Network:
  //   SendPort0
  //       |
  //       |
  //     Ain0
  //  [ RouterA ]
  //     Aout0
  //       |
  //       |
  //   RecvPort0
  builder.WithVirtualChannel("VC0").WithFlitBitWidth(100).WithDepth(3);

  builder.WithPort("SendPort0").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort0").AsOutputDirection().WithVirtualChannel("VC0");

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0");

  builder.WithLink("Link0A")
      .WithSourcePort("SendPort0")
      .WithSinkPort("Ain0")
      .WithSourceSinkPipelineStage(2);
  builder.WithLink("LinkA0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithSourceSinkPipelineStage(2);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());
  XLS_LOG(INFO) << nc_proto.DebugString();
  XLS_LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  NetworkManager graph;
  NocParameters params;

  XLS_ASSERT_OK(BuildNetworkGraphFromProto(nc_proto, &graph, &params));
  graph.Dump();

  // Sanity check network
  ASSERT_EQ(graph.GetNetworkIds().size(), 1);
  EXPECT_EQ(params.GetNetworkParam(graph.GetNetworkIds()[0])->GetName(),
            "Test");
  XLS_LOG(INFO) << "Network Graph Complete ...";

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

  // phit0 traverses a link of latency 4 so will arrive on cycle 5.
  XLS_ASSERT_OK_AND_ASSIGN(
      int64 dest_index_0,
      simulator.GetRoutingTable()->GetSinkIndices().GetNetworkComponentIndex(
          recv_port_0));

  TimedDataPhit phit0;
  phit0.phit.valid = true;
  phit0.phit.vc = 0;
  phit0.phit.data = 707;
  phit0.phit.destination_index = dest_index_0;
  phit0.cycle = 1;

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSrc * sim_send_port_0,
                           simulator.GetSimNetworkInterfaceSrc(send_port_0));
  XLS_ASSERT_OK(sim_send_port_0->SendPhitAtTime(phit0));

  for (int64 i = 0; i < 10; ++i) {
    XLS_ASSERT_OK(simulator.RunCycle());
  }

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSink * sim_recv_port_0,
                           simulator.GetSimNetworkInterfaceSink(recv_port_0));

  absl::Span<const TimedDataPhit> traffic_recv_port_0 =
      sim_recv_port_0->GetReceivedTraffic();

  EXPECT_EQ(traffic_recv_port_0.size(), 1);
  EXPECT_EQ(traffic_recv_port_0[0].cycle, 5);
  EXPECT_EQ(traffic_recv_port_0[0].phit.data, 707);
}

TEST(SimObjectsTest, TreeNework0) {
  XLS_LOG(INFO) << "Setting up network ...";
  NetworkConfigProtoBuilder builder("Test");

  // Network:
  //
  //   SendPort0     SendPort1    SendPort2
  //     \           /                |
  //      \         /                 |
  //     Ain0     Ain1                |
  //      [ RouterA ]                 |
  //     Aout0    Aout1               |
  //       |        \                 |
  //       |         \----------|     |
  //       |                  Bin0   Bin1
  //       |                    [    RouterB   ]
  //       |                  Bout0  Bout1  Bout2
  //       |                   /      |         \
  //       |                  /L=2    |          \
  //   RecvPort0          RecvPort1   RecvPort2  RecvPort3
  builder.WithVirtualChannel("VC0").WithFlitBitWidth(100).WithDepth(3);
  builder.WithVirtualChannel("VC1").WithFlitBitWidth(200).WithDepth(3);

  builder.WithPort("SendPort0")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("SendPort1")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("SendPort2")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort0")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort1")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort2")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort3")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithInputPort("Ain1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithOutputPort("Aout1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");

  auto routerb = builder.WithRouter("RouterB");
  routerb.WithInputPort("Bin0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithInputPort("Bin1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithOutputPort("Bout0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithOutputPort("Bout1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithOutputPort("Bout2").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");

  builder.WithLink("Link0A").WithSourcePort("SendPort0").WithSinkPort("Ain0");
  builder.WithLink("Link1A").WithSourcePort("SendPort1").WithSinkPort("Ain1");
  builder.WithLink("LinkAB").WithSourcePort("Aout1").WithSinkPort("Bin0");
  builder.WithLink("Link2A").WithSourcePort("SendPort2").WithSinkPort("Bin1");

  builder.WithLink("LinkA0").WithSourcePort("Aout0").WithSinkPort("RecvPort0");
  builder.WithLink("LinkB1")
      .WithSourcePort("Bout0")
      .WithSinkPort("RecvPort1")
      .WithSourceSinkPipelineStage(2);
  builder.WithLink("LinkB2").WithSourcePort("Bout1").WithSinkPort("RecvPort2");
  builder.WithLink("LinkB3").WithSourcePort("Bout2").WithSinkPort("RecvPort3");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());
  XLS_LOG(INFO) << nc_proto.DebugString();
  XLS_LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  NetworkManager graph;
  NocParameters params;

  XLS_ASSERT_OK(BuildNetworkGraphFromProto(nc_proto, &graph, &params));
  graph.Dump();

  // Sanity check network
  ASSERT_EQ(graph.GetNetworkIds().size(), 1);
  EXPECT_EQ(params.GetNetworkParam(graph.GetNetworkIds()[0])->GetName(),
            "Test");
  XLS_LOG(INFO) << "Network Graph Complete ...";

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
      int64 dest_index_1,
      simulator.GetRoutingTable()->GetSinkIndices().GetNetworkComponentIndex(
          recv_port_1));
  XLS_ASSERT_OK_AND_ASSIGN(
      int64 dest_index_3,
      simulator.GetRoutingTable()->GetSinkIndices().GetNetworkComponentIndex(
          recv_port_3));

  // phit0 traverses a link of latency 2 so will arrive on cycle 3.
  TimedDataPhit phit0;
  phit0.phit.valid = true;
  phit0.phit.vc = 0;
  phit0.phit.data = 707;
  phit0.phit.destination_index = dest_index_1;
  phit0.cycle = 1;

  // phit1 will be blocked by phit0 so it will actually arrive on cycle 2.
  TimedDataPhit phit1;
  phit1.phit.valid = true;
  phit1.phit.vc = 1;
  phit1.phit.data = 1001;
  phit1.phit.destination_index = dest_index_3;
  phit1.cycle = 1;

  // phit2 goes though all router bypass paths so will arrive on cycle 3.
  TimedDataPhit phit2;
  phit2.phit.valid = true;
  phit2.phit.vc = 1;
  phit2.phit.data = 2002;
  phit2.phit.destination_index = dest_index_3;
  phit2.cycle = 3;

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSrc * sim_send_port_0,
                           simulator.GetSimNetworkInterfaceSrc(send_port_0));
  XLS_ASSERT_OK(sim_send_port_0->SendPhitAtTime(phit0));

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSrc * sim_send_port_1,
                           simulator.GetSimNetworkInterfaceSrc(send_port_1));
  XLS_ASSERT_OK(sim_send_port_1->SendPhitAtTime(phit1));
  XLS_ASSERT_OK(sim_send_port_1->SendPhitAtTime(phit2));

  for (int64 i = 0; i < 10; ++i) {
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

  absl::Span<const TimedDataPhit> traffic_recv_port_0 =
      sim_recv_port_0->GetReceivedTraffic();
  absl::Span<const TimedDataPhit> traffic_recv_port_1 =
      sim_recv_port_1->GetReceivedTraffic();
  absl::Span<const TimedDataPhit> traffic_recv_port_2 =
      sim_recv_port_2->GetReceivedTraffic();
  absl::Span<const TimedDataPhit> traffic_recv_port_3 =
      sim_recv_port_3->GetReceivedTraffic();

  EXPECT_EQ(traffic_recv_port_0.size(), 0);
  EXPECT_EQ(traffic_recv_port_1.size(), 1);
  EXPECT_EQ(traffic_recv_port_2.size(), 0);
  EXPECT_EQ(traffic_recv_port_3.size(), 2);

  EXPECT_EQ(traffic_recv_port_1[0].cycle, 3);
  EXPECT_EQ(traffic_recv_port_1[0].phit.data, 707);
  EXPECT_EQ(traffic_recv_port_3[0].cycle, 2);
  EXPECT_EQ(traffic_recv_port_3[0].phit.data, 1001);
  EXPECT_EQ(traffic_recv_port_3[1].cycle, 3);
  EXPECT_EQ(traffic_recv_port_3[1].phit.data, 2002);
}

}  // namespace
}  // namespace noc
}  // namespace xls
