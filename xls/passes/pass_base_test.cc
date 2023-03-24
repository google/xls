// Copyright 2023 The XLS Authors
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

#include "xls/passes/pass_base.h"

#include <optional>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ram_rewrite.pb.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::Contains;

TEST(RamDatastructuresTest, AddrWidthCorrect) {
  RamConfig config{.kind = RamKind::kAbstract, .depth = 2};
  EXPECT_EQ(config.addr_width(), 1);
  config.depth = 3;
  EXPECT_EQ(config.addr_width(), 2);
  config.depth = 4;
  EXPECT_EQ(config.addr_width(), 2);
  config.depth = 1023;
  EXPECT_EQ(config.addr_width(), 10);
  config.depth = 1024;
  EXPECT_EQ(config.addr_width(), 10);
  config.depth = 1025;
  EXPECT_EQ(config.addr_width(), 11);
}

TEST(RamDatastructuresTest, MaskWidthCorrect) {
  int64_t data_width = 32;
  RamConfig config{.kind = RamKind::kAbstract,
                   .depth = 2,
                   .word_partition_size = std::nullopt};
  EXPECT_EQ(config.mask_width(data_width), std::nullopt);
  config.word_partition_size = 1;
  EXPECT_EQ(config.mask_width(data_width), 32);
  config.word_partition_size = 2;
  EXPECT_EQ(config.mask_width(data_width), 16);
  config.word_partition_size = 32;
  EXPECT_EQ(config.mask_width(data_width), 1);

  data_width = 7;
  config.word_partition_size = std::nullopt;
  EXPECT_EQ(config.mask_width(data_width), std::nullopt);
  config.word_partition_size = 1;
  EXPECT_EQ(config.mask_width(data_width), 7);
  config.word_partition_size = 2;
  EXPECT_EQ(config.mask_width(data_width), 4);
  config.word_partition_size = 3;
  EXPECT_EQ(config.mask_width(data_width), 3);
  config.word_partition_size = 4;
  EXPECT_EQ(config.mask_width(data_width), 2);
  config.word_partition_size = 5;
  EXPECT_EQ(config.mask_width(data_width), 2);
  config.word_partition_size = 6;
  EXPECT_EQ(config.mask_width(data_width), 2);
  config.word_partition_size = 7;
  EXPECT_EQ(config.mask_width(data_width), 1);
}

TEST(RamDatastructuresTest, RamKindProtoTest) {
  EXPECT_THAT(RamKindFromProto(RamKindProto::RAM_ABSTRACT),
              IsOkAndHolds(RamKind::kAbstract));
  EXPECT_THAT(RamKindFromProto(RamKindProto::RAM_1RW),
              IsOkAndHolds(RamKind::k1RW));
  EXPECT_THAT(RamKindFromProto(RamKindProto::RAM_INVALID),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RamDatastructuresTest, RamConfigProtoTest) {
  RamConfigProto proto;
  proto.set_kind(RamKindProto::RAM_ABSTRACT);
  proto.set_depth(1024);
  XLS_EXPECT_OK(RamConfig::FromProto(proto));
  EXPECT_EQ(RamConfig::FromProto(proto)->kind, RamKind::kAbstract);
  EXPECT_EQ(RamConfig::FromProto(proto)->depth, 1024);
  EXPECT_EQ(RamConfig::FromProto(proto)->word_partition_size, std::nullopt);
  EXPECT_EQ(RamConfig::FromProto(proto)->initial_value, std::nullopt);

  proto.set_word_partition_size(1);
  XLS_EXPECT_OK(RamConfig::FromProto(proto));
  EXPECT_EQ(RamConfig::FromProto(proto)->kind, RamKind::kAbstract);
  EXPECT_EQ(RamConfig::FromProto(proto)->depth, 1024);
  EXPECT_EQ(RamConfig::FromProto(proto)->word_partition_size, 1);
  EXPECT_EQ(RamConfig::FromProto(proto)->initial_value, std::nullopt);
}

TEST(RamDatastructuresTest, RamRewriteProtoTest) {
  RamRewriteProto proto;
  proto.mutable_from_config()->set_kind(RamKindProto::RAM_ABSTRACT);
  proto.mutable_from_config()->set_depth(1024);
  proto.mutable_to_config()->set_kind(RamKindProto::RAM_1RW);
  proto.mutable_to_config()->set_depth(1024);
  proto.mutable_from_channels_logical_to_physical()->insert(
      {"read_req", "ram_read_req"});
  proto.set_to_name_prefix("ram");

  XLS_EXPECT_OK(RamRewrite::FromProto(proto));
  EXPECT_EQ(RamRewrite::FromProto(proto)->from_config.kind, RamKind::kAbstract);
  EXPECT_EQ(RamRewrite::FromProto(proto)->from_config.depth, 1024);
  EXPECT_EQ(RamRewrite::FromProto(proto)->to_config.kind, RamKind::k1RW);
  EXPECT_EQ(RamRewrite::FromProto(proto)->to_config.depth, 1024);
  EXPECT_EQ(
      RamRewrite::FromProto(proto)->from_channels_logical_to_physical.size(),
      1);
  EXPECT_THAT(RamRewrite::FromProto(proto)->from_channels_logical_to_physical,
              Contains(std::make_pair("read_req", "ram_read_req")));
  EXPECT_EQ(RamRewrite::FromProto(proto)->to_name_prefix, "ram");
}

TEST(RamDatastructuresTest, RamRewritesProtoTest) {
  RamRewritesProto proto;
  RamRewriteProto rewrite_proto;
  rewrite_proto.mutable_from_config()->set_kind(RamKindProto::RAM_ABSTRACT);
  rewrite_proto.mutable_from_config()->set_depth(1024);
  rewrite_proto.mutable_to_config()->set_kind(RamKindProto::RAM_1RW);
  rewrite_proto.mutable_to_config()->set_depth(1024);
  rewrite_proto.mutable_from_channels_logical_to_physical()->insert(
      {"read_req", "ram_read_req"});
  rewrite_proto.set_to_name_prefix("ram");
  proto.mutable_rewrites()->Add(std::move(rewrite_proto));

  XLS_EXPECT_OK(RamRewritesFromProto(proto));
  EXPECT_EQ(RamRewritesFromProto(proto)->size(), 1);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).from_config.kind,
            RamKind::kAbstract);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).from_config.depth, 1024);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).to_config.kind, RamKind::k1RW);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).to_config.depth, 1024);
  EXPECT_EQ(RamRewritesFromProto(proto)
                ->at(0)
                .from_channels_logical_to_physical.size(),
            1);
  EXPECT_THAT(
      RamRewritesFromProto(proto)->at(0).from_channels_logical_to_physical,
      Contains(std::make_pair("read_req", "ram_read_req")));
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).to_name_prefix, "ram");
}
}  // namespace
}  // namespace xls
