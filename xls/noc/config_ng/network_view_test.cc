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

#include "xls/noc/config_ng/network_view.h"

#include "gtest/gtest.h"
#include "xls/noc/config_ng/fake_network_component.h"
#include "xls/noc/config_ng/network_connection.h"

namespace xls::noc {
namespace {

// Test member functions for network view.
TEST(NetworkViewTest, MemberFunction) {
  NetworkView view;
  EXPECT_EQ(view.GetComponentCount(), 0);
  EXPECT_EQ(view.GetConnectionCount(), 0);
  FakeNetworkComponent& component = view.AddComponent<FakeNetworkComponent>();
  EXPECT_EQ(view.GetComponentCount(), 1);
  EXPECT_EQ(view.GetCount<FakeNetworkComponent>(), 1);
  EXPECT_EQ(*view.components().begin(), &component);
  NetworkConnection& connection = view.AddConnection();
  EXPECT_EQ(view.GetConnectionCount(), 1);
  EXPECT_EQ(*view.connections().begin(), &connection);
}

}  // namespace
}  // namespace xls::noc
