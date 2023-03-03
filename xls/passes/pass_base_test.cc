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

#include "gtest/gtest.h"

namespace xls {
namespace {
TEST(RamDatastructuresTest, AddrWidthCorrect) {
  RamConfig config{.kind = RamKind::kAbstract, .width = 32, .depth = 2};
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
  RamConfig config{.kind = RamKind::kAbstract,
                   .width = 32,
                   .depth = 2,
                   .word_partition_size = std::nullopt};
  EXPECT_EQ(config.mask_width(), 0);
  config.word_partition_size = 1;
  EXPECT_EQ(config.mask_width(), 32);
  config.word_partition_size = 2;
  EXPECT_EQ(config.mask_width(), 16);
  config.word_partition_size = 32;
  EXPECT_EQ(config.mask_width(), 1);

  config.width = 7;
  config.word_partition_size = std::nullopt;
  EXPECT_EQ(config.mask_width(), 0);
  config.word_partition_size = 1;
  EXPECT_EQ(config.mask_width(), 7);
  config.word_partition_size = 2;
  EXPECT_EQ(config.mask_width(), 4);
  config.word_partition_size = 3;
  EXPECT_EQ(config.mask_width(), 3);
  config.word_partition_size = 4;
  EXPECT_EQ(config.mask_width(), 2);
  config.word_partition_size = 5;
  EXPECT_EQ(config.mask_width(), 2);
  config.word_partition_size = 6;
  EXPECT_EQ(config.mask_width(), 2);
  config.word_partition_size = 7;
  EXPECT_EQ(config.mask_width(), 1);
}

}  // namespace
}  // namespace xls
