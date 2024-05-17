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

#include "xls/noc/config_ng/network_topology_component.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/coordinate.h"
#include "xls/noc/config_ng/network_topology_view.h"

namespace xls::noc {
namespace {

// Test member functions for a channel topology component.
TEST(ChannelTopologyComponentTest, MemberFunction) {
  NetworkTopologyView topology;
  topology.AddChannel();
  EXPECT_EQ(topology.GetChannelCount(), 1);
}

// Test member functions for a receive port topology component.
TEST(ReceivePortTopologyComponentTest, MemberFunction) {
  NetworkTopologyView topology;
  topology.AddReceivePort();
  EXPECT_EQ(topology.GetReceivePortCount(), 1);
}

// Test member functions for a router topology component.
TEST(RouterTopologyComponentTest, MemberFunction) {
  NetworkTopologyView topology;
  RouterTopologyComponent& component = topology.AddRouter();
  EXPECT_EQ(topology.GetRouterCount(), 1);
  component.SetCoordinate({4, 3, 5});
  const Coordinate& coordinate = component.GetCoordinate();
  EXPECT_EQ(coordinate.GetDimensionCount(), 3);
  EXPECT_EQ(coordinate.GetCoordinate(0), 4);
  EXPECT_EQ(coordinate.GetCoordinate(1), 3);
  EXPECT_EQ(coordinate.GetCoordinate(2), 5);
}

// Test member functions for a send port topology component.
TEST(SendPortTopologyComponentTest, MemberFunction) {
  NetworkTopologyView topology;
  topology.AddSendPort();
  EXPECT_EQ(topology.GetSendPortCount(), 1);
}

}  // namespace
}  // namespace xls::noc
