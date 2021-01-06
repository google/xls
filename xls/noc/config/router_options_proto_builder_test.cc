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

// Test field values for a router option.
TEST(RouterOptionsProtoBuilderTest, FieldValues) {
  RouterOptionsProto proto;
  RouterOptionsProtoBuilder router_options_proto_builder(&proto);
  EXPECT_FALSE(proto.has_routing_scheme());
  EXPECT_FALSE(proto.has_arbiter_scheme());
  router_options_proto_builder.GetRoutingSchemeOptionsProtoBuilder();
  router_options_proto_builder.GetArbiterSchemeOptionsProtoBuilder();
  EXPECT_TRUE(proto.has_routing_scheme());
  EXPECT_TRUE(proto.has_arbiter_scheme());
}

}  // namespace xls::noc
