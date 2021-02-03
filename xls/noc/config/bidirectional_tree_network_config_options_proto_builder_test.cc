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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/noc/config/custom_network_config_builder_options_proto_builder.h"

namespace xls::noc {
namespace {

// Test field values for a bidirectional tree network configuration option.
TEST(BidirectionalTreeOptionsProtoBuilderTest, FieldValues) {
  const int64 kRadix = 42;
  const int64 kNumSendPortsAtRoot = 4;
  const int64 kNumRecvPortsAtRoot = 2;
  BidirectionalTreeNetworkConfigOptionsProto proto;
  BidirectionalTreeNetworkConfigOptionsProtoBuilder builder(&proto);
  builder.WithRadix(kRadix);
  builder.WithNumSendPortsAtRoot(kNumSendPortsAtRoot);
  builder.WithNumRecvPortsAtRoot(kNumRecvPortsAtRoot);

  EXPECT_TRUE(proto.has_radix());
  EXPECT_TRUE(proto.has_num_send_ports_at_root());
  EXPECT_TRUE(proto.has_num_recv_ports_at_root());
  EXPECT_EQ(proto.radix(), kRadix);
  EXPECT_EQ(proto.num_send_ports_at_root(), kNumSendPortsAtRoot);
  EXPECT_EQ(proto.num_recv_ports_at_root(), kNumRecvPortsAtRoot);
}
}  // namespace
}  // namespace xls::noc
