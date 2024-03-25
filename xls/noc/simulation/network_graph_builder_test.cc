// Copyright 2020 The XLS Authors
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

#include "xls/noc/simulation/network_graph_builder.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"

namespace xls {
namespace noc {
namespace {

TEST(SimNetworkGraphBuilderTest, UnitTest) {
  LOG(INFO) << "Setting up network ...";

  // Use builder to build noc
  NetworkConfigProtoBuilder builder("Test");
  builder.WithPort("SendPort").AsInputDirection();
  builder.WithPort("RecvPort").AsOutputDirection();
  auto router = builder.WithRouter("Router0");
  router.WithInputPort("in0");
  router.WithOutputPort("out0");

  builder.WithLink("Link0").WithSourcePort("SendPort").WithSinkPort("in0");
  builder.WithLink("Link1").WithSourcePort("out0").WithSinkPort("RecvPort");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  LOG(INFO) << network;
  LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  NetworkManager graph;
  NocParameters params;

  XLS_ASSERT_OK(BuildNetworkGraphFromProto(network, &graph, &params));
  graph.Dump();

  // Verify network
  ASSERT_EQ(graph.GetNetworkIds().size(), 1);
  EXPECT_EQ(params.GetNetworkParam(graph.GetNetworkIds()[0])->GetName(),
            "Test");

  // 5 components - 2 links, 2 network interfaces, 1 router
  ASSERT_EQ(graph.GetNetworkByIndex(0).GetNetworkComponentCount(), 5);
  for (NetworkComponentId nid :
       graph.GetNetworkByIndex(0).GetNetworkComponentIds()) {
    NetworkComponent& network_component = graph.GetNetworkComponent(nid);
    network_component.Dump();
    XLS_ASSERT_OK_AND_ASSIGN(NetworkComponentParam nc_param,
                             params.GetNetworkComponentParam(nid));
    ConnectionId connection_id;
    NetworkComponentId link_id;
    switch (network_component.kind()) {
      case NetworkComponentKind::kNISrc:
        EXPECT_EQ(std::get<NetworkInterfaceSrcParam>(nc_param).GetName(),
                  "SendPort");
        EXPECT_EQ(
            params.GetPortParam(network_component.GetPortIds()[0])->GetName(),
            "SendPort");

        // Retrieve and check link attached with this network component
        connection_id = network_component.GetPortByIndex(0).connection();
        link_id =
            graph.GetConnection(connection_id).sink().GetNetworkComponentId();
        EXPECT_EQ(std::get<LinkParam>(*params.GetNetworkComponentParam(link_id))
                      .GetName(),
                  "Link0");

        // Link ports are associated with the protos for the network components.
        EXPECT_EQ(
            params
                .GetPortParam(
                    graph.GetNetworkComponent(link_id).GetInputPortIds()[0])
                ->GetName(),
            "SendPort");
        EXPECT_EQ(
            params
                .GetPortParam(
                    graph.GetNetworkComponent(link_id).GetOutputPortIds()[0])
                ->GetName(),
            "in0");
        break;
      case NetworkComponentKind::kNISink:
        EXPECT_EQ(std::get<NetworkInterfaceSinkParam>(nc_param).GetName(),
                  "RecvPort");
        EXPECT_EQ(
            params.GetPortParam(network_component.GetPortIds()[0])->GetName(),
            "RecvPort");

        // Retrieve and check link attached with this network component
        connection_id = network_component.GetPortByIndex(0).connection();
        link_id =
            graph.GetConnection(connection_id).src().GetNetworkComponentId();
        EXPECT_EQ(std::get<LinkParam>(*params.GetNetworkComponentParam(link_id))
                      .GetName(),
                  "Link1");

        // Link ports are associated with the protos for the network components.
        EXPECT_EQ(
            params
                .GetPortParam(
                    graph.GetNetworkComponent(link_id).GetInputPortIds()[0])
                ->GetName(),
            "out0");
        EXPECT_EQ(
            params
                .GetPortParam(
                    graph.GetNetworkComponent(link_id).GetOutputPortIds()[0])
                ->GetName(),
            "RecvPort");
        break;
      case NetworkComponentKind::kRouter:
        EXPECT_EQ(std::get<RouterParam>(nc_param).GetName(), "Router0");
        EXPECT_EQ(params.GetPortParam(network_component.GetInputPortIds()[0])
                      ->GetName(),
                  "in0");
        EXPECT_EQ(params.GetPortParam(network_component.GetOutputPortIds()[0])
                      ->GetName(),
                  "out0");

        // Check that we can also retrieve by name.
        {
          XLS_ASSERT_OK_AND_ASSIGN(
              NetworkComponentId router0,
              FindNetworkComponentByName("Router0", graph, params));
          XLS_ASSERT_OK_AND_ASSIGN(PortId in0,
                                   FindPortByName("in0", graph, params));
          EXPECT_EQ(router0, nid);
          EXPECT_EQ(in0, network_component.GetInputPortIds()[0]);
        }

        // Retrieve and check link attached to input port
        connection_id =
            graph.GetPort(network_component.GetInputPortIds()[0]).connection();
        link_id =
            graph.GetConnection(connection_id).src().GetNetworkComponentId();
        EXPECT_EQ(std::get<LinkParam>(*params.GetNetworkComponentParam(link_id))
                      .GetName(),
                  "Link0");

        // Retrieve and check link attached to output port
        connection_id =
            graph.GetPort(network_component.GetOutputPortIds()[0]).connection();
        link_id =
            graph.GetConnection(connection_id).sink().GetNetworkComponentId();
        EXPECT_EQ(std::get<LinkParam>(*params.GetNetworkComponentParam(link_id))
                      .GetName(),
                  "Link1");
        break;
      default:
        break;
    }
  }
}

}  // namespace
}  // namespace noc
}  // namespace xls
