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

#include "xls/noc/simulation/common.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace noc {
namespace {

TEST(SimCommonTest, NetworkId) {
  NetworkId id(0);
  EXPECT_EQ(id.id(), 0);
  EXPECT_TRUE(id.IsValid());

  id = NetworkId::kInvalid;
  // TODO(tedhong) : 2020-12-01 - consider parameterizing these EQ constants.
  EXPECT_EQ(id.id(), 0xffff);
  EXPECT_EQ(id.AsUInt64(), 0xffff000000000000);
  EXPECT_FALSE(id.IsValid());

  EXPECT_EQ(NetworkId::MaxIndex(), (1 << 16) - 2);
}

TEST(SimCommonTest, NetworkComponentId) {
  NetworkComponentId id(1, 2);
  EXPECT_EQ(id.network(), 1);
  EXPECT_EQ(id.id(), 2);
  EXPECT_TRUE(id.IsValid());

  id = NetworkComponentId::kInvalid;
  EXPECT_EQ(id.network(), 0xffff);
  EXPECT_EQ(id.id(), 0xffffffff);
  EXPECT_EQ(id.AsUInt64(), 0xffffffffffff0000);
  EXPECT_FALSE(id.IsValid());

  EXPECT_EQ(NetworkComponentId::MaxIndex(), (1ul << 32) - 2);
}

TEST(SimCommonTest, ConnectionId) {
  ConnectionId id{1, 2};
  EXPECT_EQ(id.network(), 1);
  EXPECT_EQ(id.id(), 2);
  EXPECT_TRUE(id.IsValid());

  id = ConnectionId::kInvalid;
  EXPECT_EQ(id.network(), 0xffff);
  EXPECT_EQ(id.id(), 0xffffffff);
  EXPECT_EQ(id.AsUInt64(), 0xffffffffffff0000);
  EXPECT_FALSE(id.IsValid());

  EXPECT_EQ(ConnectionId::MaxIndex(), (1ul << 32) - 2);
}
TEST(SimCommonTest, PortId) {
  PortId id{1, 2, 3};
  EXPECT_EQ(id.network(), 1);
  EXPECT_EQ(id.component(), 2);
  EXPECT_EQ(id.id(), 3);
  EXPECT_TRUE(id.IsValid());

  id = PortId::kInvalid;
  EXPECT_EQ(id.network(), 0xffff);
  EXPECT_EQ(id.component(), 0xffffffff);
  EXPECT_EQ(id.id(), 0xffff);
  EXPECT_EQ(id.AsUInt64(), 0xffffffffffffffff);
  EXPECT_FALSE(id.IsValid());

  EXPECT_EQ(PortId::MaxIndex(), (1ul << 16) - 2);
}

}  // namespace
}  // namespace noc
}  // namespace xls
