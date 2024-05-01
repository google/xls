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

#include <cstdint>

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

TEST(SimCommonTest, TrafficFlowId) {
  TrafficFlowId id(0);
  EXPECT_EQ(id.id(), 0);
  EXPECT_TRUE(id.IsValid());

  id = TrafficFlowId::kInvalid;
  EXPECT_EQ(id.id(), static_cast<int32_t>(kNullIdValue));
  EXPECT_EQ(id.AsUInt64(), static_cast<int64_t>(id.id()));
  EXPECT_FALSE(id.IsValid());

  EXPECT_EQ(TrafficFlowId::MaxIndex(), (1l << 32) - 2);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId id0a,
                           TrafficFlowId::ValidateAndReturnId(0));
  EXPECT_EQ(id0a.id(), 0);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId id0b,
                           TrafficFlowId::ValidateAndReturnId(0));
  EXPECT_EQ(id0b.id(), 0);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficFlowId id1,
                           TrafficFlowId::ValidateAndReturnId(1));
  EXPECT_EQ(id1.id(), 1);

  EXPECT_EQ(id0a, id0b);
  EXPECT_NE(id0a, id1);
}

TEST(SimCommonTest, TrafficModeId) {
  TrafficModeId id(0);
  EXPECT_EQ(id.id(), 0);
  EXPECT_TRUE(id.IsValid());

  id = TrafficModeId::kInvalid;
  EXPECT_EQ(id.id(), static_cast<int32_t>(kNullIdValue));
  EXPECT_EQ(id.AsUInt64(), static_cast<int64_t>(id.id()));
  EXPECT_FALSE(id.IsValid());

  EXPECT_EQ(TrafficModeId::MaxIndex(), (1l << 32) - 2);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId id0a,
                           TrafficModeId::ValidateAndReturnId(0));
  EXPECT_EQ(id0a.id(), 0);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId id0b,
                           TrafficModeId::ValidateAndReturnId(0));
  EXPECT_EQ(id0b.id(), 0);

  XLS_ASSERT_OK_AND_ASSIGN(TrafficModeId id1,
                           TrafficModeId::ValidateAndReturnId(1));
  EXPECT_EQ(id1.id(), 1);

  EXPECT_EQ(id0a, id0b);
  EXPECT_NE(id0a, id1);
}

}  // namespace
}  // namespace noc
}  // namespace xls
