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

#include "xls/noc/config_ng/network_connection_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config_ng/fake_network_component.h"
#include "xls/noc/config_ng/network_component.h"
#include "xls/noc/config_ng/network_component_port.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {
namespace {

using ::absl_testing::IsOk;
using ::testing::Not;

// Validate a connection with a data output port as a source and a data input
// port as a sink.
TEST(NetworkConnectionUtilsTest, ValidateNetworkConnectionOfValidConnection) {
  NetworkView view;
  NetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  NetworkComponentPort& source =
      component.AddPort(PortType::kData, PortDirection::kOutput);
  NetworkComponentPort& sink =
      component.AddPort(PortType::kData, PortDirection::kInput);
  NetworkConnection& connection = view.AddConnection();
  EXPECT_THAT(ValidateNetworkConnection(connection), Not(IsOk()));
  connection.ConnectToSourcePort(&source);
  EXPECT_THAT(ValidateNetworkConnection(connection), Not(IsOk()));
  connection.ConnectToSinkPort(&sink);
  XLS_EXPECT_OK(ValidateNetworkConnection(connection));
}

}  // namespace
}  // namespace xls::noc
