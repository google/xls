// Copyright 2022 The XLS Authors
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

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/float32_add_cc.h"
#include "xls/dslx/stdlib/float32_fma_cc.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/jit/compound_type_cc.h"
#include "xls/jit/multi_function_one_cc.h"
#include "xls/jit/multi_function_two_cc.h"
#include "xls/jit/null_function_cc.h"

// Rather than do a pattern-matching unit test of aot_compile.cc's output, this
// "unit test" focuses on using libraries compiled from such.

namespace xls {
namespace {

Value F32Value(bool sign, uint8_t exp, uint32_t frac) {
  return Value::Tuple({Value(UBits(static_cast<uint64_t>(sign), 1)),
                       Value(UBits(exp, 8)), Value(UBits(frac, 23))});
}

// Tests straightforward usage of a straightforward library/AOT-compiled module.
TEST(AotCompileTest, BasicUsage) {
  Value f32_one = F32Value(false, 0x7f, 0);
  Value f32_two = F32Value(false, 0x80, 0);
  Value f32_three = F32Value(false, 0x80, 0x400000);
  XLS_ASSERT_OK_AND_ASSIGN(Value result, xls::fp::add(f32_one, f32_two));
  EXPECT_EQ(result, f32_three);
}

// Another basic test, just for some more mileage.
TEST(AotCompileTest, AnotherBasicUsage) {
  Value f32_one = F32Value(false, 0x7f, 0);
  Value f32_two = F32Value(false, 0x80, 0);
  Value f32_three = F32Value(false, 0x80, 0x400000);
  Value f32_five = F32Value(false, 0x81, 0x200000);
  XLS_ASSERT_OK_AND_ASSIGN(Value result,
                           xls::fp::fma(f32_one, f32_two, f32_three));
  EXPECT_EQ(result, f32_five);
}

TEST(AotCompileTest, NullFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(Value result, xls::foo::bar::null_function());
  EXPECT_EQ(result, Value::Tuple({}));
}

TEST(AotCompileTest, CompoundType) {
  Value a = Parser::ParseTypedValue("bits[32]:42").value();
  Value b = Parser::ParseTypedValue("()").value();
  Value c = Parser::ParseTypedValue(
                "(bits[1]:1, bits[223]:0xdeadbeef_a1b2c3d4_01234567, "
                "[(bits[3]:2, [(bits[32]:123, bits[2]:1)]), "
                " (bits[3]:0, [(bits[32]:333, bits[2]:0)])])")
                .value();
  XLS_ASSERT_OK_AND_ASSIGN(Value result, xls::fun_test_function(a, b, c));
  EXPECT_EQ(result, Value::Tuple({b, Value(UBits(43, 32)), c}));
}

#ifndef NDEBUG
// In non-opt mode, argument values are type-checked using DCHECK.
TEST(AotCompileTest, InvalidTypes) {
  Value a = Value::Tuple({});
  EXPECT_DEATH((void)xls::fp::add(a, a),
               testing::HasSubstr(
                   "Value `()` is not of type `(bits[1], bits[8], bits[23])`"));
}
#endif

TEST(AotCompileTest, TopFunction) {
  Value a = Value(UBits(2, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Value v, xls::foo::bar::multi_function_one(a));
  EXPECT_EQ(v.bits().ToInt64().value(), 10);
}

TEST(AotCompileTest, NonTopFunction) {
  Value a = Value(UBits(2, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Value v, xls::foo::bar::multi_function_two(a));
  EXPECT_EQ(v.bits().ToInt64().value(), 4);
}

}  // namespace
}  // namespace xls
