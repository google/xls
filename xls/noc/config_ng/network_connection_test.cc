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

#include "xls/noc/config_ng/network_connection.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/fake_network_component.h"
#include "xls/noc/config_ng/network_component_port.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {
namespace {

// Test member functions for network connection.
TEST(NetworkConnectionTest, MemberFunction) {
  NetworkView view;
  FakeNetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& source_port =
      component.AddPort(PortType::kData, PortDirection::kOutput);
  NetworkComponentPort& sink_port =
      component.AddPort(PortType::kData, PortDirection::kInput);
  NetworkConnection& connection = view.AddConnection();
  EXPECT_EQ(&connection.GetNetworkView(), &view);
  connection.ConnectToSourcePort(&source_port);
  EXPECT_EQ(connection.GetSourcePort(), &source_port);
  connection.ConnectToSinkPort(&sink_port);
  EXPECT_EQ(connection.GetSinkPort(), &sink_port);
  EXPECT_EQ(source_port.GetConnections().size(), 1);
  EXPECT_EQ(sink_port.GetConnections().size(), 1);
}

}  // namespace
}  // namespace xls::noc
