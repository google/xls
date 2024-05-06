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

#include <bit>
#include <cstdint>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/float32_upcast_jit_wrapper.h"
#include "xls/dslx/stdlib/float32_mul_jit_wrapper.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"
#include "xls/ir/value_view.h"
#include "xls/jit/compound_type_jit_wrapper.h"

namespace xls {
namespace {

using something::cool::CompoundJitWrapper;

TEST(JitWrapperTest, BasicFunctionCall) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto complex_value,
      ValueBuilder::Tuple(
          {Value(UBits(1, 1)), Value(UBits(33, 223)),
           ValueBuilder::Array(
               {ValueBuilder::Tuple(
                    {Value(UBits(3, 3)),
                     ValueBuilder::Array({ValueBuilder::Tuple(
                         {Value(UBits(32, 32)), Value(UBits(2, 2))})})}),
                ValueBuilder::Tuple(
                    {Value(UBits(5, 3)),
                     ValueBuilder::Array({ValueBuilder::Tuple(
                         {Value(UBits(3333, 32)), Value(UBits(0, 2))})})})})})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, CompoundJitWrapper::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      jit->Run(Value(UBits(0, 32)), Value::Tuple({}), complex_value));
  EXPECT_EQ(result, Value::Tuple({Value::Tuple({}), Value(UBits(1, 32)),
                                  complex_value}));
}

TEST(JitWrapperTest, SpecializedFunctionCall) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::F32ToF64::Create());
  XLS_ASSERT_OK_AND_ASSIGN(double dv, jit->Run(3.14f));
  EXPECT_EQ(dv, static_cast<double>(3.14f));
}

TEST(JitWrapperTest, SpecializedFunctionCall2) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::Float32Mul::Create());
  XLS_ASSERT_OK_AND_ASSIGN(float res, jit->Run(3.14f, 1.2345f));
  EXPECT_EQ(res, 3.14f * 1.2345f);
}

TEST(JitWrapperTest, PackedFunctionCall) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::F32ToF64::Create());
  double dv = -1.23;
  float fv = 3.14;
  PackedFloat pfv(std::bit_cast<uint8_t*>(&fv), 0);
  PackedDouble pdv(std::bit_cast<uint8_t*>(&dv), 0);
  XLS_ASSERT_OK(jit->Run(pfv, pdv));
  EXPECT_EQ(dv, static_cast<double>(3.14f));
  EXPECT_EQ(fv, 3.14f);
}

TEST(JitWrapperTest, PackedFunctionCall2) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::Float32Mul::Create());
  float lv = 3.14f;
  float rv = 1.2345f;
  float result;
  PackedFloat plv(std::bit_cast<uint8_t*>(&lv), 0);
  PackedFloat prv(std::bit_cast<uint8_t*>(&rv), 0);
  PackedFloat pres(std::bit_cast<uint8_t*>(&result), 0);
  XLS_ASSERT_OK(jit->Run(plv, prv, pres));
  EXPECT_EQ(result, 3.14f * 1.2345f);
  EXPECT_EQ(lv, 3.14f);
  EXPECT_EQ(rv, 1.2345f);
}

}  // namespace
}  // namespace xls
