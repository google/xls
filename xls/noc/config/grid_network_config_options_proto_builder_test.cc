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

}  // namespace
}  // namespace xls::noc
