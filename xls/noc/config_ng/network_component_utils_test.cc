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

#include "xls/noc/config_ng/network_component_utils.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config_ng/fake_network_component.h"
#include "xls/noc/config_ng/network_component_port.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {
namespace {

using ::testing::StrEq;

// Validate a component with one port.
TEST(NetworkComponentUtilsTest, ValidateNetworkComponentOfValidComponent) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  XLS_EXPECT_OK(ValidateNetworkComponent(component));
  component.AddPort(PortType::kData, PortDirection::kInput);
  XLS_EXPECT_OK(ValidateNetworkComponent(component));
}

// GetComponentsConnectedToOutputPortsFrom a component with one output port that
// is connected to a single component.
TEST(NetworkComponentUtilsTest, GetComponentsConnectedToOutputPortsFrom) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  NetworkComponent& componentA = view.AddComponent<FakeNetworkComponent>();
  view.AddConnection()
      .ConnectToSourcePort(
          &component.AddPort(PortType::kData, PortDirection::kOutput))
      .ConnectToSinkPort(
          &componentA.AddPort(PortType::kData, PortDirection::kInput));
  std::vector<const NetworkComponent*> components =
      GetComponentsConnectedToOutputPortsFrom(component);
  ASSERT_EQ(components.size(), 1);
  EXPECT_EQ(components[0], &componentA);
}

// Populate two components with names using AddNameToComponent.
TEST(NetworkComponentTest, AddNameToComponent) {
  absl::flat_hash_map<std::string, NetworkComponent*> names;
  NetworkView network_view;
  NetworkComponent& test0 = network_view.AddComponent<FakeNetworkComponent>();
  test0.SetName("Test");
  NetworkComponent& test1 = network_view.AddComponent<FakeNetworkComponent>();
  names.insert({"Test0", &test0});
  names.insert({"Test1", &test1});
  XLS_EXPECT_OK(AddNameToComponent(names));
  EXPECT_THAT(test0.GetName(), StrEq("Test0"));
  EXPECT_THAT(test1.GetName(), StrEq("Test1"));
}

// GetInputPortsFrom a component with one input port and two output ports.
TEST(NetworkComponentTest, GetInputPortsFrom) {
  NetworkView network_view;
  NetworkComponent& component =
      network_view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& port =
      component.AddPort(PortType::kData, PortDirection::kInput);
  component.AddPort(PortType::kData, PortDirection::kOutput);
  component.AddPort(PortType::kControl, PortDirection::kOutput);
  ASSERT_EQ(component.GetPortCount(), 3);
  EXPECT_EQ(GetInputPortsFrom(component).size(), 1);
  EXPECT_EQ(GetInputPortsFrom(component)[0], &port);
}

// GetOutputPortsFrom a component with one output port and two input ports.
TEST(NetworkComponentTest, GetOutputPortsFrom) {
  NetworkView network_view;
  NetworkComponent& component =
      network_view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& port =
      component.AddPort(PortType::kData, PortDirection::kOutput);
  component.AddPort(PortType::kData, PortDirection::kInput);
  component.AddPort(PortType::kControl, PortDirection::kInput);
  ASSERT_EQ(component.GetPortCount(), 3);
  EXPECT_EQ(GetOutputPortsFrom(component).size(), 1);
  EXPECT_EQ(GetOutputPortsFrom(component)[0], &port);
}

// GetDataPortsFrom a component with one data port and two control ports.
TEST(NetworkComponentTest, GetDataPortsFrom) {
  NetworkView network_view;
  NetworkComponent& component =
      network_view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& port =
      component.AddPort(PortType::kData, PortDirection::kOutput);
  component.AddPort(PortType::kControl, PortDirection::kInput);
  component.AddPort(PortType::kControl, PortDirection::kOutput);
  ASSERT_EQ(component.GetPortCount(), 3);
  EXPECT_EQ(GetDataPortsFrom(component).size(), 1);
  EXPECT_EQ(GetOutputPortsFrom(component)[0], &port);
}

// GetControlPortsFrom a component with one control port and two data ports.
TEST(NetworkComponentTest, GetControlPortsFrom) {
  NetworkView network_view;
  NetworkComponent& component =
      network_view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& port =
      component.AddPort(PortType::kControl, PortDirection::kOutput);
  component.AddPort(PortType::kData, PortDirection::kInput);
  component.AddPort(PortType::kData, PortDirection::kOutput);
  ASSERT_EQ(component.GetPortCount(), 3);
  EXPECT_EQ(GetControlPortsFrom(component).size(), 1);
  EXPECT_EQ(GetControlPortsFrom(component)[0], &port);
}

}  // namespace
}  // namespace xls::noc
