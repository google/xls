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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/noc/config/common_network_config_builder_options_proto_builder.h"

namespace xls::noc {

// Test field values for a virtual channel option.
TEST(VirtualChannelOptionsProtoBuilderTest, FieldValues) {
  const int64_t kDepth0 = 42;
  const int64_t kDepth1 = 4;
  const int64_t kDepth2 = 2;
  VirtualChannelOptionsProto proto;
  VirtualChannelOptionsProtoBuilder virtual_channel_options_proto_builder(
      &proto);
  virtual_channel_options_proto_builder.WithVirtualChannelDepth(kDepth0);
  virtual_channel_options_proto_builder.WithVirtualChannelDepth(kDepth1);
  virtual_channel_options_proto_builder.WithVirtualChannelDepth(kDepth2);

  EXPECT_EQ(proto.virtual_channel_depth_size(), 3);
  EXPECT_EQ(proto.virtual_channel_depth(0), kDepth0);
  EXPECT_EQ(proto.virtual_channel_depth(1), kDepth1);
  EXPECT_EQ(proto.virtual_channel_depth(2), kDepth2);
}

}  // namespace xls::noc
