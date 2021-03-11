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

// Test field values for a endpoint options.
TEST(EndpointOptionsProtoBuilderTest, FieldValues) {
  const int64_t kNumSendPorts = 42;
  const int64_t kNumRecvPorts = 1337;
  EndpointOptionsProto proto;
  EndpointOptionsProtoBuilder builder(&proto);
  builder.WithNumSendPorts(kNumSendPorts);
  builder.WithNumRecvPorts(kNumRecvPorts);

  EXPECT_TRUE(proto.has_num_send_ports());
  EXPECT_TRUE(proto.has_num_recv_ports());
  EXPECT_EQ(proto.num_send_ports(), kNumSendPorts);
  EXPECT_EQ(proto.num_recv_ports(), kNumRecvPorts);
}

}  // namespace xls::noc
