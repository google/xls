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

#include "xls/noc/simulation/global_routing_table.h"

#include <cstdint>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/network_graph_builder.h"
#include "xls/noc/simulation/sample_network_graphs.h"

namespace xls {
namespace noc {
namespace {

void LogRoute(std::string_view route_name,
              absl::Span<const NetworkComponentId> route,
              const NocParameters& params) {
  LOG(INFO) << route_name << " is ...";
  for (int64_t i = 0; i < route.size(); ++i) {
    LOG(INFO) << absl::StrFormat(
        "%d : %s %x", i,
        absl::visit([](auto nc) { return nc.GetName(); },
                    *params.GetNetworkComponentParam(route[i])),
        route[i].AsUInt64());
  }
}

TEST(GlobalRoutingTableTest, Index) {
  LOG(INFO) << "Setting up network ...";
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
  //       |                  /       |          \
  //   RecvPort0          RecvPort1   RecvPort2  RecvPort3

  builder.WithPort("SendPort0").AsInputDirection();
  builder.WithPort("SendPort1").AsInputDirection();
  builder.WithPort("SendPort2").AsInputDirection();
  builder.WithPort("RecvPort0").AsOutputDirection();
  builder.WithPort("RecvPort1").AsOutputDirection();
  builder.WithPort("RecvPort2").AsOutputDirection();
  builder.WithPort("RecvPort3").AsOutputDirection();

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(100).WithDepth(33);
  builder.WithVirtualChannel("VC1").WithFlitBitWidth(200).WithDepth(34);

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
  builder.WithLink("LinkB1").WithSourcePort("Bout0").WithSinkPort("RecvPort1");
  builder.WithLink("LinkB2").WithSourcePort("Bout1").WithSinkPort("RecvPort2");
  builder.WithLink("LinkB3").WithSourcePort("Bout2").WithSinkPort("RecvPort3");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());
  LOG(INFO) << nc_proto;
  LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  NetworkManager graph;
  NocParameters params;

  XLS_ASSERT_OK(BuildNetworkGraphFromProto(nc_proto, &graph, &params));
  graph.Dump();

  // Validate network
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

  NetworkComponentId sendport0 =
      *routing_table.GetSourceIndices().GetNetworkComponentByIndex(0);
  NetworkComponentId recvport3 =
      *routing_table.GetSinkIndices().GetNetworkComponentByIndex(3);

  EXPECT_EQ(std::get<NetworkInterfaceSrcParam>(
                *params.GetNetworkComponentParam(sendport0))
                .GetName(),
            "SendPort0");
  EXPECT_EQ(std::get<NetworkInterfaceSinkParam>(
                *params.GetNetworkComponentParam(recvport3))
                .GetName(),
            "RecvPort3");

  // Test hops.
  XLS_ASSERT_OK_AND_ASSIGN(PortId aout0,
                           FindPortByName("Aout0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(PortId aout1,
                           FindPortByName("Aout1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport0,
      FindNetworkComponentByName("RecvPort0", graph, params));

  XLS_ASSERT_OK_AND_ASSIGN(
      PortAndVCIndex hop_aout0,
      routing_table.GetNextHopPort(PortAndVCIndex{ain0, 0}, recvport0));
  EXPECT_EQ(hop_aout0.port_id_, aout0);
  EXPECT_EQ(hop_aout0.vc_index_, 0);

  XLS_ASSERT_OK_AND_ASSIGN(
      PortAndVCIndex hop_aout1,
      routing_table.GetNextHopPort(PortAndVCIndex{ain0, 1}, recvport3));
  EXPECT_EQ(hop_aout1.port_id_, aout1);
  EXPECT_EQ(hop_aout1.vc_index_, 1);

  EXPECT_THAT(
      routing_table.GetNextHopPort(PortAndVCIndex{bin1, 1}, recvport0).status(),
      status_testing::StatusIs(absl::StatusCode::kNotFound,
                               testing::HasSubstr("Unable to find")));

  // Test route.
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId routera_nc,
      FindNetworkComponentByName("RouterA", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId routerb_nc,
      FindNetworkComponentByName("RouterB", graph, params));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route03,
                           routing_table.ComputeRoute(sendport0, recvport3));
  LogRoute("Route 03", route03, params);

  EXPECT_EQ(route03.size(), 7);
  EXPECT_EQ(route03[0], sendport0);
  EXPECT_EQ(route03[2], routera_nc);
  EXPECT_EQ(route03[4], routerb_nc);
  EXPECT_EQ(route03[6], recvport3);

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route00,
                           routing_table.ComputeRoute(sendport0, recvport0));
  LogRoute("Route 00", route00, params);

  EXPECT_EQ(route00.size(), 5);
  EXPECT_EQ(route00[0], sendport0);
  EXPECT_EQ(route00[2], routera_nc);
  EXPECT_EQ(route00[4], recvport0);
}

TEST(GlobalRoutingTableTest, RouterLoop) {
  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphLoop000(&proto, &graph, &params));
  graph.Dump();

  // Validate network
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
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport0,
      FindNetworkComponentByName("RecvPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport1,
      FindNetworkComponentByName("RecvPort1", graph, params));

  XLS_ASSERT_OK_AND_ASSIGN(PortId ain0, FindPortByName("Ain0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(PortId bin1, FindPortByName("Bin1", graph, params));

  EXPECT_EQ(ain0, graph.GetNetworkComponent(routera_id).GetPortIdByIndex(0));
  EXPECT_EQ(bin1, graph.GetNetworkComponent(routerb_id).GetPortIdByIndex(1));

  // Create global routing table.
  DistributedRoutingTableBuilderForTrees route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  EXPECT_EQ(routing_table.GetSourceIndices().NetworkComponentCount(), 2);
  EXPECT_EQ(routing_table.GetSinkIndices().NetworkComponentCount(), 2);

  // Test route.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route00,
                           routing_table.ComputeRoute(sendport0, recvport0));
  LogRoute("Route 00", route00, params);

  EXPECT_EQ(route00.size(), 5);
  EXPECT_EQ(route00[0], sendport0);
  EXPECT_EQ(route00[2], routera_id);
  EXPECT_EQ(route00[4], recvport0);

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route01,
                           routing_table.ComputeRoute(sendport0, recvport1));
  LogRoute("Route 01", route01, params);

  EXPECT_EQ(route01.size(), 7);
  EXPECT_EQ(route01[0], sendport0);
  EXPECT_EQ(route01[2], routera_id);
  EXPECT_EQ(route01[4], routerb_id);
  EXPECT_EQ(route01[6], recvport1);
}

TEST(GlobalRoutingTableTest, MultiplePathsBetweenRouters) {
  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphLinear001(&proto, &graph, &params));
  graph.Dump();

  // Validate network
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
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkai0_id,
      FindNetworkComponentByName("LinkAI0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkai1_id,
      FindNetworkComponentByName("LinkAI1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkao0_id,
      FindNetworkComponentByName("LinkAO0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkao1_id,
      FindNetworkComponentByName("LinkAO1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbo0_id,
      FindNetworkComponentByName("LinkBO0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbo1_id,
      FindNetworkComponentByName("LinkBO1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport1,
      FindNetworkComponentByName("SendPort1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport0,
      FindNetworkComponentByName("RecvPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport1,
      FindNetworkComponentByName("RecvPort1", graph, params));

  // Create global routing table.
  DistributedRoutingTableBuilderForMultiplePaths route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  EXPECT_EQ(routing_table.GetSourceIndices().NetworkComponentCount(), 2);
  EXPECT_EQ(routing_table.GetSinkIndices().NetworkComponentCount(), 2);

  // Test route.
  // For routes from SendPort0 use link LinkAO0 between RouterA and RouterB.
  // For routes from SendPort1 use link LinkAO1 between RouterA and RouterB.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route00,
                           routing_table.ComputeRoute(sendport0, recvport0));
  LogRoute("Route 00", route00, params);

  EXPECT_EQ(route00.size(), 7);
  EXPECT_THAT(route00, ::testing::ElementsAre(sendport0, linkai0_id, routera_id,
                                              linkao0_id, routerb_id,
                                              linkbo0_id, recvport0));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route01,
                           routing_table.ComputeRoute(sendport0, recvport1));
  LogRoute("Route 01", route01, params);

  EXPECT_EQ(route01.size(), 7);
  EXPECT_THAT(route01, ::testing::ElementsAre(sendport0, linkai0_id, routera_id,
                                              linkao0_id, routerb_id,
                                              linkbo1_id, recvport1));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route10,
                           routing_table.ComputeRoute(sendport1, recvport0));
  LogRoute("Route 10", route10, params);

  EXPECT_EQ(route10.size(), 7);
  EXPECT_THAT(route10, ::testing::ElementsAre(sendport1, linkai1_id, routera_id,
                                              linkao1_id, routerb_id,
                                              linkbo0_id, recvport0));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route11,
                           routing_table.ComputeRoute(sendport1, recvport1));
  LogRoute("Route 11", route11, params);

  EXPECT_EQ(route11.size(), 7);
  EXPECT_THAT(route11, ::testing::ElementsAre(sendport1, linkai1_id, routera_id,
                                              linkao1_id, routerb_id,
                                              linkbo1_id, recvport1));
}

TEST(GlobalRoutingTableTest, MultiplePathsBetweenRoutersWithLoop0) {
  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphLoop000(&proto, &graph, &params));
  graph.Dump();

  // Validate network
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
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkai0_id,
      FindNetworkComponentByName("LinkAI0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkao0_id,
      FindNetworkComponentByName("LinkAO0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkab1_id,
      FindNetworkComponentByName("LinkAB1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkab2_id,
      FindNetworkComponentByName("LinkAB2", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbi0_id,
      FindNetworkComponentByName("LinkBI0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbo0_id,
      FindNetworkComponentByName("LinkBO0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport1,
      FindNetworkComponentByName("SendPort1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport0,
      FindNetworkComponentByName("RecvPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport1,
      FindNetworkComponentByName("RecvPort1", graph, params));

  // Create global routing table.
  DistributedRoutingTableBuilderForMultiplePaths route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  XLS_ASSERT_OK(routing_table.DumpRouterRoutingTable(graph.GetNetworkIds()[0]));

  EXPECT_EQ(routing_table.GetSourceIndices().NetworkComponentCount(), 2);
  EXPECT_EQ(routing_table.GetSinkIndices().NetworkComponentCount(), 2);

  // Test route.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route00,
                           routing_table.ComputeRoute(sendport0, recvport0));
  LogRoute("Route 00", route00, params);

  EXPECT_EQ(route00.size(), 5);
  EXPECT_THAT(route00, ::testing::ElementsAre(sendport0, linkai0_id, routera_id,
                                              linkao0_id, recvport0));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route01,
                           routing_table.ComputeRoute(sendport0, recvport1));
  LogRoute("Route 01", route01, params);

  EXPECT_EQ(route01.size(), 7);
  EXPECT_THAT(route01, ::testing::ElementsAre(sendport0, linkai0_id, routera_id,
                                              linkab1_id, routerb_id,
                                              linkbo0_id, recvport1));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route10,
                           routing_table.ComputeRoute(sendport1, recvport0));
  LogRoute("Route 10", route10, params);

  EXPECT_EQ(route10.size(), 7);
  EXPECT_THAT(route10, ::testing::ElementsAre(sendport1, linkbi0_id, routerb_id,
                                              linkab2_id, routera_id,
                                              linkao0_id, recvport0));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route11,
                           routing_table.ComputeRoute(sendport1, recvport1));
  LogRoute("Route 11", route11, params);

  EXPECT_EQ(route11.size(), 5);
  EXPECT_THAT(route11, ::testing::ElementsAre(sendport1, linkbi0_id, routerb_id,
                                              linkbo0_id, recvport1));
}

TEST(GlobalRoutingTableTest, MultiplePathsBetweenRoutersWithLoop1) {
  // Build and assign simulation objects
  NetworkConfigProto proto;
  NetworkManager graph;
  NocParameters params;
  XLS_ASSERT_OK(BuildNetworkGraphLoop001(&proto, &graph, &params));
  graph.Dump();

  // Validate network
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
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkai0_id,
      FindNetworkComponentByName("LinkAI0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkai1_id,
      FindNetworkComponentByName("LinkAI1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkao0_id,
      FindNetworkComponentByName("LinkAO0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkao1_id,
      FindNetworkComponentByName("LinkAO1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkba0_id,
      FindNetworkComponentByName("LinkBA0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkba1_id,
      FindNetworkComponentByName("LinkBA1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkab0_id,
      FindNetworkComponentByName("LinkAB0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkab1_id,
      FindNetworkComponentByName("LinkAB1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbi0_id,
      FindNetworkComponentByName("LinkBI0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbi1_id,
      FindNetworkComponentByName("LinkBI1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbo0_id,
      FindNetworkComponentByName("LinkBO0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId linkbo1_id,
      FindNetworkComponentByName("LinkBO1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport0,
      FindNetworkComponentByName("SendPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport1,
      FindNetworkComponentByName("SendPort1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport2,
      FindNetworkComponentByName("SendPort2", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId sendport3,
      FindNetworkComponentByName("SendPort3", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport0,
      FindNetworkComponentByName("RecvPort0", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport1,
      FindNetworkComponentByName("RecvPort1", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport2,
      FindNetworkComponentByName("RecvPort2", graph, params));
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId recvport3,
      FindNetworkComponentByName("RecvPort3", graph, params));

  // Create global routing table.
  DistributedRoutingTableBuilderForMultiplePaths route_builder;
  XLS_ASSERT_OK_AND_ASSIGN(DistributedRoutingTable routing_table,
                           route_builder.BuildNetworkRoutingTables(
                               graph.GetNetworkIds()[0], graph, params));

  XLS_ASSERT_OK(routing_table.DumpRouterRoutingTable(graph.GetNetworkIds()[0]));

  EXPECT_EQ(routing_table.GetSourceIndices().NetworkComponentCount(), 4);
  EXPECT_EQ(routing_table.GetSinkIndices().NetworkComponentCount(), 4);

  // Test route.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route00,
                           routing_table.ComputeRoute(sendport0, recvport0));
  LogRoute("Route 00", route00, params);

  EXPECT_EQ(route00.size(), 5);
  EXPECT_THAT(route00, ::testing::ElementsAre(sendport0, linkai0_id, routera_id,
                                              linkao0_id, recvport0));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route12,
                           routing_table.ComputeRoute(sendport1, recvport2));
  LogRoute("Route 12", route12, params);

  EXPECT_EQ(route12.size(), 7);
  EXPECT_THAT(route12, ::testing::ElementsAre(sendport1, linkai1_id, routera_id,
                                              linkab1_id, routerb_id,
                                              linkbo0_id, recvport2));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route23,
                           routing_table.ComputeRoute(sendport2, recvport3));
  LogRoute("Route 23", route23, params);

  EXPECT_EQ(route23.size(), 5);
  EXPECT_THAT(route23, ::testing::ElementsAre(sendport2, linkbi0_id, routerb_id,
                                              linkbo1_id, recvport3));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route31,
                           routing_table.ComputeRoute(sendport3, recvport1));
  LogRoute("Route 31", route31, params);

  EXPECT_EQ(route31.size(), 7);
  EXPECT_THAT(route31, ::testing::ElementsAre(sendport3, linkbi1_id, routerb_id,
                                              linkba1_id, routera_id,
                                              linkao1_id, recvport1));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route20,
                           routing_table.ComputeRoute(sendport2, recvport0));
  LogRoute("Route 20", route20, params);

  EXPECT_EQ(route20.size(), 7);
  EXPECT_THAT(route20, ::testing::ElementsAre(sendport2, linkbi0_id, routerb_id,
                                              linkba0_id, routera_id,
                                              linkao0_id, recvport0));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route03,
                           routing_table.ComputeRoute(sendport0, recvport3));
  LogRoute("Route 03", route03, params);

  EXPECT_EQ(route03.size(), 7);
  EXPECT_THAT(route03, ::testing::ElementsAre(sendport0, linkai0_id, routera_id,
                                              linkab0_id, routerb_id,
                                              linkbo1_id, recvport3));
}

}  // namespace
}  // namespace noc
}  // namespace xls
