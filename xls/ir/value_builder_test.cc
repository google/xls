// Copyright 2024 The XLS Authors
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

#include "xls/ir/value_builder.h"

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

TEST(ValueBuilderTest, Array) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto value, ValueBuilder::Array({
                                          ValueBuilder::Array({
                                              ValueBuilder::Bits(UBits(1, 3)),
                                              ValueBuilder::Bits(UBits(2, 3)),
                                          }),
                                          ValueBuilder::Array({
                                              ValueBuilder::Bits(UBits(3, 3)),
                                              ValueBuilder::Bits(UBits(4, 3)),
                                          }),
                                      })
                      .Build());
  EXPECT_TRUE(value.IsArray());
  EXPECT_EQ(value.size(), 2);
  EXPECT_TRUE(value.element(0).IsArray());
  EXPECT_EQ(value.element(0).size(), 2);
  EXPECT_EQ(value.element(0).element(0), Value(UBits(1, 3)));
  EXPECT_EQ(value.element(0).element(1), Value(UBits(2, 3)));
  EXPECT_TRUE(value.element(1).IsArray());
  EXPECT_EQ(value.element(1).size(), 2);
  EXPECT_EQ(value.element(1).element(0), Value(UBits(3, 3)));
  EXPECT_EQ(value.element(1).element(1), Value(UBits(4, 3)));
}

TEST(ValueBuilderTest, ArrayFails) {
  EXPECT_THAT(ValueBuilder::Array({
                                      ValueBuilder::Array({
                                          ValueBuilder::Bits(UBits(1, 3)),
                                          ValueBuilder::Bits(UBits(2, 3)),
                                      }),
                                      ValueBuilder::Array({
                                          ValueBuilder::Bits(UBits(3, 3)),
                                          ValueBuilder::Bits(UBits(4, 3)),
                                          ValueBuilder::Bits(UBits(5, 3)),
                                      }),
                                  })
                  .Build(),
              status_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST(ValueBuilderTest, Tuple) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto value, ValueBuilder::Tuple({
                                          ValueBuilder::Tuple({
                                              ValueBuilder::Bits(UBits(1, 1)),
                                              ValueBuilder::Bits(UBits(2, 2)),
                                          }),
                                          ValueBuilder::Tuple({
                                              ValueBuilder::Bits(UBits(3, 3)),
                                              ValueBuilder::Bits(UBits(4, 4)),
                                          }),
                                      })
                      .Build());
  EXPECT_TRUE(value.IsTuple());
  EXPECT_EQ(value.size(), 2);
  EXPECT_TRUE(value.element(0).IsTuple());
  EXPECT_EQ(value.element(0).size(), 2);
  EXPECT_EQ(value.element(0).element(0), Value(UBits(1, 1)));
  EXPECT_EQ(value.element(0).element(1), Value(UBits(2, 2)));
  EXPECT_TRUE(value.element(1).IsTuple());
  EXPECT_EQ(value.element(1).size(), 2);
  EXPECT_EQ(value.element(1).element(0), Value(UBits(3, 3)));
  EXPECT_EQ(value.element(1).element(1), Value(UBits(4, 4)));
}

}  // namespace
}  // namespace xls
