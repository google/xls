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

#include "xls/noc/config/custom_network_config_builder_options_proto_builder.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace xls::noc {

namespace {
// Test field values for a grid network configuration option.
TEST(GridOptionsProtoBuilderTest, FieldValues) {
  const int64_t kNumRow = 4;
  const int64_t kNumColumn = 2;
  const bool kRowLoopback = true;
  const bool kColumnLoopback = false;
  GridNetworkConfigOptionsProto proto;
  GridNetworkConfigOptionsProtoBuilder builder(&proto);
  builder.WithNumRows(kNumRow);
  builder.WithNumColumns(kNumColumn);
  builder.WithRowLoopback(kRowLoopback);
  builder.WithColumnLoopback(kColumnLoopback);

  EXPECT_TRUE(proto.has_num_rows());
  EXPECT_TRUE(proto.has_num_columns());
  EXPECT_TRUE(proto.has_row_loopback());
  EXPECT_TRUE(proto.has_column_loopback());
  EXPECT_EQ(proto.num_rows(), kNumRow);
  EXPECT_EQ(proto.num_columns(), kNumColumn);
  EXPECT_EQ(proto.row_loopback(), kRowLoopback);
  EXPECT_EQ(proto.column_loopback(), kColumnLoopback);
}

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

// Test field values for a bidirectional tree network configuration option.
TEST(BidirectionalTreeOptionsProtoBuilderTest, FieldValues) {
  const int64_t kRadix = 42;
  const int64_t kNumSendPortsAtRoot = 4;
  const int64_t kNumRecvPortsAtRoot = 2;
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
