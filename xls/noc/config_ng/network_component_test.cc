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

#include "gtest/gtest.h"
#include "xls/noc/config_ng/fake_network_component.h"
#include "xls/noc/config_ng/network_component_port.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {

namespace {

// Test basic member functions for network component.
TEST(NetworkComponentTest, MemberFunctionBasics) {
  NetworkView network_view;
  FakeNetworkComponent& component =
      network_view.AddComponent<FakeNetworkComponent>();
  EXPECT_EQ(&component.GetNetworkView(), &network_view);
  component.SetName("Test");
  EXPECT_EQ(component.GetName(), "Test");
}

// Test member functions for network component with an input data port.
TEST(NetworkComponentTest, MemberFunctionPortInputData) {
  NetworkView network_view;
  FakeNetworkComponent& component =
      network_view.AddComponent<FakeNetworkComponent>();
  component.AddPort(PortType::kData, PortDirection::kInput);
  EXPECT_EQ(component.GetPortCount(), 1);
  EXPECT_TRUE(component.ports().begin()->IsInput());
  EXPECT_TRUE(component.ports().begin()->IsData());
}

// Test member functions for network component with an output control port.
TEST(NetworkComponentTest, MemberFunctionPortOutputControl) {
  NetworkView network_view;
  FakeNetworkComponent& component =
      network_view.AddComponent<FakeNetworkComponent>();
  component.AddPort(PortType::kControl, PortDirection::kOutput);
  EXPECT_EQ(component.GetPortCount(), 1);
  EXPECT_TRUE(component.ports().begin()->IsOutput());
  EXPECT_TRUE(component.ports().begin()->IsControl());
}

}  // namespace
}  // namespace xls::noc
