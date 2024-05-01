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

#include "xls/noc/simulation/network_graph.h"

#include "gtest/gtest.h"
#include "xls/noc/simulation/common.h"

namespace xls {
namespace noc {
namespace {

TEST(SimNetworkGraph, NetworkCreateTest) {
  NetworkManager mgr;

  NetworkId network0 = *mgr.CreateNetwork();
  NetworkId network1 = *mgr.CreateNetwork();

  EXPECT_TRUE(network0.IsValid());
  EXPECT_TRUE(network1.IsValid());

  EXPECT_EQ(mgr.GetNetworkIdByIndex(0), network0);
  EXPECT_EQ(mgr.GetNetworkIdByIndex(1), network1);
  EXPECT_EQ(mgr.GetNetworkIds()[0], network0);
  EXPECT_EQ(mgr.GetNetworkIds()[1], network1);

  EXPECT_EQ(mgr.GetNetworkByIndex(0).id(), network0);
  EXPECT_EQ(mgr.GetNetworkByIndex(1).id(), network1);
  EXPECT_EQ(mgr.GetNetworks()[0].id(), network0);
  EXPECT_EQ(mgr.GetNetworks()[1].id(), network1);

  EXPECT_EQ(mgr.GetNetworkCount(), 2);
  EXPECT_EQ(mgr.GetNetworks().size(), 2);

  EXPECT_EQ(&mgr.GetNetworks()[0], &mgr.GetNetwork(network0));
  EXPECT_EQ(&mgr.GetNetworks()[1], &mgr.GetNetwork(network1));
}

TEST(SimNetworkGraph, NetworkComponentCreateTest) {
  NetworkManager mgr;

  NetworkId network = *mgr.CreateNetwork();
  EXPECT_TRUE(network.IsValid());
  Network& network_obj = mgr.GetNetwork(network);

  NetworkComponentId link0 =
      *mgr.CreateNetworkComponent(network, NetworkComponentKind::kLink);
  EXPECT_EQ(network_obj.GetNetworkComponentCount(), 1);

  NetworkComponentId link1 =
      *mgr.CreateNetworkComponent(network, NetworkComponentKind::kLink);
  EXPECT_EQ(network_obj.GetNetworkComponentCount(), 2);

  EXPECT_EQ(network_obj.GetNetworkComponentIdByIndex(0), link0);
  EXPECT_EQ(network_obj.GetNetworkComponentIdByIndex(1), link1);
  EXPECT_EQ(network_obj.GetNetworkComponentByIndex(0).id(), link0);
  EXPECT_EQ(network_obj.GetNetworkComponentByIndex(1).id(), link1);

  EXPECT_EQ(&network_obj.GetNetworkComponents()[0],
            &mgr.GetNetworkComponent(link0));

  EXPECT_EQ(&network_obj.GetNetworkComponents()[1],
            &mgr.GetNetworkComponent(link1));

  EXPECT_EQ(link0.GetNetworkId(), network);
  EXPECT_EQ(link1.GetNetworkId(), network);

  EXPECT_EQ(mgr.GetNetworkComponent(link0).GetPortCount(), 0);
  PortId link0_p0 = *mgr.CreatePort(link0, PortDirection::kInput);
  EXPECT_EQ(mgr.GetNetworkComponent(link0).GetPortCount(), 1);
  PortId link0_p1 = *mgr.CreatePort(link0, PortDirection::kOutput);
  EXPECT_EQ(mgr.GetNetworkComponent(link0).GetPortCount(), 2);

  EXPECT_NE(link0_p0, link0_p1);
  EXPECT_EQ(link0_p0.id(), 0);
  EXPECT_EQ(link0_p1.id(), 1);

  EXPECT_EQ(&mgr.GetNetworkComponent(link0).GetPorts()[0],
            &mgr.GetPort(link0_p0));
  EXPECT_EQ(&mgr.GetNetworkComponent(link0).GetPorts()[1],
            &mgr.GetPort(link0_p1));
}

TEST(SimNetworkGraph, CreateUnitTest) {
  NetworkManager mgr;

  NetworkId network = *mgr.CreateNetwork();
  EXPECT_TRUE(network.IsValid());

  NetworkComponentId src =
      *mgr.CreateNetworkComponent(network, NetworkComponentKind::kNISrc);
  NetworkComponentId link =
      *mgr.CreateNetworkComponent(network, NetworkComponentKind::kLink);
  NetworkComponentId dest =
      *mgr.CreateNetworkComponent(network, NetworkComponentKind::kNISink);
  EXPECT_TRUE(src.IsValid());
  EXPECT_TRUE(link.IsValid());
  EXPECT_TRUE(dest.IsValid());

  PortId src_p0 = *mgr.CreatePort(src, PortDirection::kOutput);
  PortId link_p0 = *mgr.CreatePort(link, PortDirection::kInput);
  PortId link_p1 = *mgr.CreatePort(link, PortDirection::kOutput);
  PortId dest_p0 = *mgr.CreatePort(dest, PortDirection::kInput);
  EXPECT_TRUE(src_p0.IsValid());
  EXPECT_TRUE(link_p0.IsValid());
  EXPECT_TRUE(link_p1.IsValid());
  EXPECT_TRUE(dest_p0.IsValid());

  ConnectionId conn_0 = *mgr.CreateConnection(network);
  ConnectionId conn_1 = *mgr.CreateConnection(network, link_p1, dest_p0);

  mgr.Attach(conn_0, src_p0);
  mgr.Attach(conn_0, link_p0);

  // Test Traversals
  mgr.Dump();

  EXPECT_EQ(conn_0.GetNetworkId(), network);
  EXPECT_EQ(conn_1.GetNetworkId(), network);

  EXPECT_EQ(mgr.GetNetworkComponent(src).GetPortCount(), 1);
  EXPECT_EQ(mgr.GetNetworkComponent(src).GetPortIdByIndex(0), src_p0);

  EXPECT_EQ(src_p0.GetNetworkComponentId(), src);
  EXPECT_EQ(mgr.GetPort(src_p0).connection(), conn_0);

  EXPECT_EQ(mgr.GetConnection(conn_0).src(), src_p0);
  EXPECT_EQ(mgr.GetConnection(conn_0).sink(), link_p0);

  EXPECT_EQ(link_p0.GetNetworkComponentId(), link);
  EXPECT_EQ(link_p1.GetNetworkComponentId(), link);

  EXPECT_EQ(mgr.GetNetworkComponent(link).GetPortIdByIndex(0), link_p0);
  EXPECT_EQ(mgr.GetNetworkComponent(link).GetPortIdByIndex(1), link_p1);
  EXPECT_EQ(mgr.GetNetworkComponent(link).GetPortIds()[0], link_p0);
  EXPECT_EQ(mgr.GetNetworkComponent(link).GetPortIds()[1], link_p1);
  EXPECT_EQ(mgr.GetNetworkComponent(link).GetInputPortIds()[0], link_p0);
  EXPECT_EQ(mgr.GetNetworkComponent(link).GetOutputPortIds()[0], link_p1);

  EXPECT_EQ(mgr.GetConnection(conn_1).src(), link_p1);
  EXPECT_EQ(mgr.GetConnection(conn_1).sink(), dest_p0);

  EXPECT_EQ(mgr.GetNetworkComponent(dest).GetPortCount(), 1);
  EXPECT_EQ(mgr.GetNetworkComponent(dest).GetPortIdByIndex(0), dest_p0);
}

}  // namespace
}  // namespace noc
}  // namespace xls
