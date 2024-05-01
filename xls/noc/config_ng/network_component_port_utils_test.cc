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

#include "xls/noc/config_ng/network_component_port_utils.h"

#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config_ng/fake_network_component.h"
#include "xls/noc/config_ng/network_component_port.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {
namespace {

// GetInputPortsFrom a port list with one input port and two output ports.
TEST(NetworkComponentPortUtilsTest, GetInputPortsFrom) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  std::vector<NetworkComponentPort*> ports;
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kInput));
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kOutput));
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kOutput));
  ASSERT_EQ(ports.size(), 3);
  EXPECT_EQ(GetInputPortsFrom(ports).size(), 1);
}

// GetOutputPortsFrom a port list with one output port and two input ports.
TEST(NetworkComponentPortUtilsTest, GetOutputPortsFrom) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  std::vector<NetworkComponentPort*> ports;
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kOutput));
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kInput));
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kInput));
  ASSERT_EQ(ports.size(), 3);
  EXPECT_EQ(GetOutputPortsFrom(ports).size(), 1);
}

// GetDataPortsFrom a port list with one data port and two control ports.
TEST(NetworkComponentPortUtilsTest, GetDataPortsFrom) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  std::vector<NetworkComponentPort*> ports;
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kOutput));
  ports.emplace_back(
      &component.AddPort(PortType::kControl, PortDirection::kOutput));
  ports.emplace_back(
      &component.AddPort(PortType::kControl, PortDirection::kOutput));
  ASSERT_EQ(ports.size(), 3);
  EXPECT_EQ(GetDataPortsFrom(ports).size(), 1);
}

// GetControlPortsFrom a port list with one control port and two data ports.
TEST(NetworkComponentPortUtilsTest, GetControlPortsFrom) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  std::vector<NetworkComponentPort*> ports;
  ports.emplace_back(
      &component.AddPort(PortType::kControl, PortDirection::kOutput));
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kOutput));
  ports.emplace_back(
      &component.AddPort(PortType::kData, PortDirection::kOutput));
  ASSERT_EQ(ports.size(), 3);
  EXPECT_EQ(GetControlPortsFrom(ports).size(), 1);
}

// Validate a data input port.
TEST(NetworkComponentPortUtilsTest, ValidateNetworkComponentPort) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& port =
      component.AddPort(PortType::kData, PortDirection::kInput);
  XLS_EXPECT_OK(ValidateNetworkComponentPort(port));
}

// GetComponentsConnectedTo a port that is connected to a single component.
TEST(NetworkComponentPortUtilsTest, GetComponentsConnectedTo) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  NetworkComponent& componentA = view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& output_port =
      component.AddPort(PortType::kData, PortDirection::kOutput);
  view.AddConnection()
      .ConnectToSourcePort(&output_port)
      .ConnectToSinkPort(
          &componentA.AddPort(PortType::kData, PortDirection::kInput));
  std::vector<const NetworkComponent*> components =
      GetComponentsConnectedTo(output_port);
  ASSERT_EQ(components.size(), 1);
  EXPECT_EQ(components[0], &componentA);
}

}  // namespace
}  // namespace xls::noc
