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

// Test field values for a unidirectional tree network configuration option.
TEST(UnidirectionalTreeOptionsProtoBuilderTest, FieldValues) {
  const int64_t kRadix = 4;
  UnidirectionalTreeNetworkConfigOptionsProto proto;
  UnidirectionalTreeNetworkConfigOptionsProtoBuilder builder(&proto);
  builder.WithRadix(kRadix);

  EXPECT_TRUE(proto.has_radix());
  EXPECT_EQ(proto.radix(), kRadix);
}

// Test distribution type for a unidirectional tree network configuration
// option.
TEST(UnidirectionalTreeOptionsProtoBuilderTest, DistributionType) {
  UnidirectionalTreeNetworkConfigOptionsProto proto;
  UnidirectionalTreeNetworkConfigOptionsProtoBuilder builder(&proto);
  builder.AsDistributionTree();

  EXPECT_EQ(proto.type(),
            UnidirectionalTreeNetworkConfigOptionsProto::DISTRIBUTION);
}

// Test aggregation type for a unidirectional tree network configuration option.
TEST(UnidirectionalTreeOptionsProtoBuilderTest, AggregationType) {
  UnidirectionalTreeNetworkConfigOptionsProto proto;
  UnidirectionalTreeNetworkConfigOptionsProtoBuilder builder(&proto);
  builder.AsAggregationTree();

  EXPECT_EQ(proto.type(),
            UnidirectionalTreeNetworkConfigOptionsProto::AGGREGATION);
}

}  // namespace
}  // namespace xls::noc
