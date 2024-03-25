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
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/simulation/packetizer.h"
#include "xls/noc/simulation/sample_network_graphs.h"
#include "xls/noc/simulation/sim_objects.h"

namespace xls::noc {
namespace {

TEST(SimObjectsTest, BackToBackNetwork0) {
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
  XLS_ASSERT_OK_AND_ASSIGN(
      NetworkComponentId router,
      FindNetworkComponentByName("RouterA", graph, params));

  XLS_ASSERT_OK_AND_ASSIGN(
      int64_t src_index_0,
      simulator.GetRoutingTable()->GetSourceIndices().GetNetworkComponentIndex(
          send_port_0));

  XLS_ASSERT_OK_AND_ASSIGN(
      int64_t dest_index_0,
      simulator.GetRoutingTable()->GetSinkIndices().GetNetworkComponentIndex(
          recv_port_0));

  // Send a packet and expect that the flits it turns into
  // arrive at the other end in-order.
  XLS_ASSERT_OK_AND_ASSIGN(DataPacket packet,
                           DataPacketBuilder()
                               .Valid(true)
                               .SourceIndex(src_index_0)
                               .DestinationIndex(dest_index_0)
                               .VirtualChannel(0)
                               .Data(UBits(0b1010'0110'1100'0011, 16))
                               .Build());

  DePacketizer depacketizer(16, 3, 128);

  XLS_ASSERT_OK(depacketizer.AcceptNewPacket(packet));
  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSrc * sim_send_port_0,
                           simulator.GetSimNetworkInterfaceSrc(send_port_0));

  int64_t cycle_to_send = 0;
  while (!depacketizer.IsIdle()) {
    XLS_ASSERT_OK_AND_ASSIGN(DataFlit flit, depacketizer.ComputeNextFlit());
    TimedDataFlit timed_flit{cycle_to_send, flit};

    XLS_ASSERT_OK(sim_send_port_0->SendFlitAtTime(timed_flit));
    LOG(INFO) << ++cycle_to_send;
  }

  for (int64_t i = 0; i < 10; ++i) {
    XLS_ASSERT_OK(simulator.RunCycle());
  }

  XLS_ASSERT_OK_AND_ASSIGN(SimNetworkInterfaceSink * sim_recv_port_0,
                           simulator.GetSimNetworkInterfaceSink(recv_port_0));

  absl::Span<const TimedDataFlit> traffic_recv_port_0 =
      sim_recv_port_0->GetReceivedTraffic();

  EXPECT_EQ(traffic_recv_port_0.size(), 2);

  TimedDataFlit flit0 = traffic_recv_port_0[0];
  TimedDataFlit flit1 = traffic_recv_port_0[1];

  EXPECT_EQ(flit0.cycle, 5);
  EXPECT_EQ(flit0.flit.source_index, src_index_0);
  EXPECT_EQ(flit0.flit.destination_index, dest_index_0);
  EXPECT_EQ(flit0.flit.type, FlitType::kHead);
  EXPECT_EQ(flit0.flit.vc, 0);
  EXPECT_EQ(flit0.flit.data.Slice(0, flit0.flit.data_bit_count),
            UBits(0b0'0110'1100'0011, 13));
  EXPECT_THAT(flit0.metadata.timed_route_info.route,
              ::testing::ElementsAre(TimedRouteItem{send_port_0, 1},
                                     TimedRouteItem{router, 3},
                                     TimedRouteItem{recv_port_0, 5}));

  EXPECT_EQ(flit1.cycle, 6);
  EXPECT_EQ(flit1.flit.source_index, src_index_0);
  EXPECT_EQ(flit1.flit.destination_index, dest_index_0);
  EXPECT_EQ(flit1.flit.type, FlitType::kTail);
  EXPECT_EQ(flit1.flit.vc, 0);
  EXPECT_EQ(flit1.flit.data.Slice(0, flit1.flit.data_bit_count),
            UBits(0b101, 3));
  EXPECT_THAT(flit1.metadata.timed_route_info.route,
              ::testing::ElementsAre(TimedRouteItem{send_port_0, 2},
                                     TimedRouteItem{router, 4},
                                     TimedRouteItem{recv_port_0, 6}));
}

}  // namespace
}  // namespace xls::noc
