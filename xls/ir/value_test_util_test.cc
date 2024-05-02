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

#include "xls/ir/value_test_util.h"

#include "gtest/gtest.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

TEST(ValueTestUtilTest, ValuesEqual) {
  EXPECT_TRUE(ValuesEqual(Value(UBits(1, 1)), Value(UBits(1, 1))));
  EXPECT_FALSE(ValuesEqual(Value(UBits(1, 1)), Value(UBits(0, 1))));

  EXPECT_FALSE(ValuesEqual(Value(UBits(1, 1234)), Value(UBits(1, 10))));
}

}  // namespace
}  // namespace xls
