// Copyright 2020 Google LLC
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

// Unit tests for the JIT wrapper using fpadd_2x32 as a basis.

#include "xls/modules/fpadd_2x32_jit_wrapper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

TEST(Fpadd2x32JitWrapperTest, CanAdd) {
  XLS_ASSERT_OK_AND_ASSIGN(Fpadd2x32 adder, Fpadd2x32::Create());
  Value one = F32ToTuple(1.0f);
  Value two = F32ToTuple(2.0f);

  XLS_ASSERT_OK_AND_ASSIGN(Value expected, adder.Run(one, two));
  XLS_ASSERT_OK_AND_ASSIGN(float result, TupleToF32(expected));
  EXPECT_EQ(result, 3.0f);
}

}  // namespace
}  // namespace xls
