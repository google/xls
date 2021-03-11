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

#include "xls/noc/config_ng/network_component_port.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/fake_network_component.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {
namespace {

// Test member functions for a network component port.
TEST(NetworkComponentPortTest, MemberFunctionDataOutput) {
  NetworkView view;
  FakeNetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& port =
      component.AddPort(PortType::kData, PortDirection::kOutput);
  EXPECT_EQ(&port.GetComponent(), &component);
  EXPECT_TRUE(port.IsOutput());
  EXPECT_TRUE(port.IsData());
  EXPECT_EQ(port.GetConnections().size(), 0);
  NetworkConnection& connection =
      view.AddConnection().ConnectToSourcePort(&port);
  ASSERT_EQ(port.GetConnections().size(), 1);
  EXPECT_EQ(*port.GetConnections().begin(), &connection);
}

// Test member functions for a network component port.
TEST(NetworkComponentPortTest, MemberFunctionControlInput) {
  NetworkView view;
  FakeNetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& port =
      component.AddPort(PortType::kControl, PortDirection::kInput);
  EXPECT_TRUE(port.IsInput());
  EXPECT_TRUE(port.IsControl());
  EXPECT_EQ(port.GetConnections().size(), 0);
  NetworkConnection& connection = view.AddConnection().ConnectToSinkPort(&port);
  ASSERT_EQ(port.GetConnections().size(), 1);
  EXPECT_EQ(*port.GetConnections().begin(), &connection);
}

}  // namespace
}  // namespace xls::noc
