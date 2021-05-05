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

TEST(GlobalRoutingTableTest, Index) {
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
  XLS_LOG(INFO) << nc_proto.DebugString();
  XLS_LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  NetworkManager graph;
  NocParameters params;

  XLS_ASSERT_OK(BuildNetworkGraphFromProto(nc_proto, &graph, &params));
  graph.Dump();

  // Validate network
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

  NetworkComponentId sendport0 =
      *routing_table.GetSourceIndices().GetNetworkComponentByIndex(0);
  NetworkComponentId recvport3 =
      *routing_table.GetSinkIndices().GetNetworkComponentByIndex(3);

  EXPECT_EQ(absl::get<NetworkInterfaceSrcParam>(
                *params.GetNetworkComponentParam(sendport0))
                .GetName(),
            "SendPort0");
  EXPECT_EQ(absl::get<NetworkInterfaceSinkParam>(
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

  XLS_LOG(INFO) << "Route 03 is ...";
  for (int64_t i = 0; i < route03.size(); ++i) {
    XLS_LOG(INFO) << absl::StrFormat(
        "%d : %s %x", i,
        absl::visit([](auto nc) { return nc.GetName(); },
                    *params.GetNetworkComponentParam(route03[i])),
        route03[i].AsUInt64());
  }

  EXPECT_EQ(route03.size(), 7);
  EXPECT_EQ(route03[0], sendport0);
  EXPECT_EQ(route03[2], routera_nc);
  EXPECT_EQ(route03[4], routerb_nc);
  EXPECT_EQ(route03[6], recvport3);

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<NetworkComponentId> route00,
                           routing_table.ComputeRoute(sendport0, recvport0));

  XLS_LOG(INFO) << "Route 00 is ...";
  for (int64_t i = 0; i < route00.size(); ++i) {
    XLS_LOG(INFO) << absl::StrFormat(
        "%d : %s %x", i,
        absl::visit([](auto nc) { return nc.GetName(); },
                    *params.GetNetworkComponentParam(route00[i])),
        route00[i].AsUInt64());
  }

  EXPECT_EQ(route00.size(), 5);
  EXPECT_EQ(route00[0], sendport0);
  EXPECT_EQ(route00[2], routera_nc);
  EXPECT_EQ(route00[4], recvport0);
}

}  // namespace
}  // namespace noc
}  // namespace xls
