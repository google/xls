// Copyright 2021 The XLS AuthorsNetworkComponent
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

#include "xls/noc/config_ng/network_topology_view.h"

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config_ng/network_component.h"

namespace xls::noc {
namespace {

// Test AddSendPort function for network topology view.
TEST(NetworkTopologyViewTest, AddSendPortTopologyComponent) {
  NetworkTopologyView view;
  view.AddSendPort();
  EXPECT_EQ(view.GetComponentCount(), 1);
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test AddReceivePort function for network topology view.
TEST(NetworkTopologyViewTest, AddReceivePortTopologyComponent) {
  NetworkTopologyView view;
  view.AddReceivePort();
  EXPECT_EQ(view.GetComponentCount(), 1);
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test AddRouter function for network topology view.
TEST(NetworkTopologyViewTest, AddRouterTopologyComponent) {
  NetworkTopologyView view;
  view.AddRouter();
  EXPECT_EQ(view.GetComponentCount(), 1);
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test AddChannel function for network topology view.
TEST(NetworkTopologyViewTest, AddChannelTopologyComponent) {
  NetworkTopologyView view;
  view.AddChannel();
  EXPECT_EQ(view.GetComponentCount(), 1);
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test GetSendPortCount function for network topology view.
TEST(NetworkTopologyViewTest, GetSendPortTopologyComponentCount) {
  NetworkTopologyView view;
  view.AddSendPort();
  EXPECT_EQ(view.GetSendPortCount(), view.GetComponentCount());
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test GetReceivePortCount function for network topology view.
TEST(NetworkTopologyViewTest, GetReceivePortTopologyComponentCount) {
  NetworkTopologyView view;
  view.AddReceivePort();
  EXPECT_EQ(view.GetReceivePortCount(), view.GetComponentCount());
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test GetRouterCount function for network topology view.
TEST(NetworkTopologyViewTest, GetRouterTopologyComponentCount) {
  NetworkTopologyView view;
  view.AddRouter();
  EXPECT_EQ(view.GetRouterCount(), view.GetComponentCount());
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test GetChannelCount function for network topology view.
TEST(NetworkTopologyViewTest, GetChannelTopologyComponentCount) {
  NetworkTopologyView view;
  view.AddChannel();
  EXPECT_EQ(view.GetChannelCount(), view.GetComponentCount());
  EXPECT_EQ(view.GetConnectionCount(), 0);
}

// Test ConnectThroughChannel function for network topology view.
TEST(NetworkTopologyViewTest, ConnectThroughChannel) {
  NetworkTopologyView view;
  XLS_EXPECT_OK(view.ConnectThroughChannel(view.AddRouter(), view.AddRouter()));
  EXPECT_EQ(view.GetComponentCount(), 3);
  EXPECT_EQ(view.GetConnectionCount(), 2);
}

// Test AddRouter function for network topology view.
TEST(NetworkTopologyViewTest, AddRouterTopologyComponentSendPortReceivePort) {
  NetworkTopologyView view;
  XLS_EXPECT_OK(view.AddRouter(4, 5));
  EXPECT_EQ(view.GetComponentCount(), 19);
  EXPECT_EQ(view.GetConnectionCount(), 18);
}

// Test ConnectSendPortsToComponent function for network topology view.
TEST(NetworkTopologyViewTest, ConnectSendPortsToComponent) {
  NetworkTopologyView view;
  NetworkComponent& component = view.AddRouter();
  XLS_EXPECT_OK(view.ConnectSendPortsToComponent(4, component));
  EXPECT_EQ(view.GetComponentCount(), 9);
  EXPECT_EQ(view.GetConnectionCount(), 8);
}

// Test ConnectComponentToReceivePort function for network topology view.
TEST(NetworkTopologyViewTest, ConnectComponentToReceivePort) {
  NetworkTopologyView view;
  NetworkComponent& component = view.AddRouter();
  XLS_EXPECT_OK(view.ConnectComponentToReceivePort(component, 3));
  EXPECT_EQ(view.GetComponentCount(), 7);
  EXPECT_EQ(view.GetConnectionCount(), 6);
}

}  // namespace
}  // namespace xls::noc
