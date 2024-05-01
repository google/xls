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

#include "xls/noc/config/arbiter_scheme_config_proto_builder.h"

#include <vector>

#include "gtest/gtest.h"

namespace xls::noc {

// Test field values of the priority arbiter scheme.
TEST(ArbiterSchemeConfigBuilderTest, PriorityArbiterSchemeFieldValues) {
  const char* kPortName0 = "Port0";
  const char* kVirtualChannelName0 = "VC0";
  const char* kPortName1 = "Port1";
  const char* kVirtualChannelName1 = "VC1";
  const char* kOutputPortName = "OutputPort";
  RouterConfigProto::ArbiterSchemeConfigProto proto;
  ArbiterSchemeConfigProtoBuilder builder(&proto);
  std::vector<PortVirtualChannelTuple> priority_list;
  priority_list.push_back({kPortName0, kVirtualChannelName0});
  priority_list.push_back({kPortName1, kVirtualChannelName1});
  builder.WithPriorityEntry(kOutputPortName, priority_list);

  EXPECT_TRUE(proto.has_priority());
  EXPECT_EQ(proto.priority().entries_size(), 1);
  const RouterConfigProto::ArbiterPriorityEntryConfig&
      routing_table_entry_config = proto.priority().entries(0);
  EXPECT_TRUE(routing_table_entry_config.has_output_port_name());
  EXPECT_EQ(routing_table_entry_config.input_port_tuples_size(), 2);
  EXPECT_EQ(routing_table_entry_config.input_port_tuples(0).port_name(),
            kPortName0);
  EXPECT_EQ(
      routing_table_entry_config.input_port_tuples(0).virtual_channel_name(),
      kVirtualChannelName0);
  EXPECT_EQ(routing_table_entry_config.input_port_tuples(1).port_name(),
            kPortName1);
  EXPECT_EQ(
      routing_table_entry_config.input_port_tuples(1).virtual_channel_name(),
      kVirtualChannelName1);
}

}  // namespace xls::noc
