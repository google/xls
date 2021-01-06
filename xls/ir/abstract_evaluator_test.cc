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

#include "xls/ir/abstract_evaluator.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/ir/bits.h"

namespace xls {
namespace {

// Simple wrapper to avoid std::vector<bool> specialization.
struct BoxedBool {
  bool value;
  bool operator!=(const BoxedBool& other) const { return value != other.value; }
  bool operator==(const BoxedBool& other) const { return value == other.value; }
};

std::vector<BoxedBool> ToBoxedVector(const Bits& input) {
  std::vector<BoxedBool> output;
  auto bits = input.ToBitVector();
  output.reserve(bits.size());
  for (bool bit : bits) {
    output.push_back({bit});
  }
  return output;
}

Bits FromBoxedVector(const std::vector<BoxedBool>& input) {
  BitsRope rope(input.size());
  for (BoxedBool bit : input) {
    rope.push_back(bit.value);
  }
  return rope.Build();
}

class TestAbstractEvaluator
    : public AbstractEvaluator<BoxedBool, TestAbstractEvaluator> {
 public:
  BoxedBool One() const { return {true}; }
  BoxedBool Zero() const { return {false}; }
  BoxedBool Not(const BoxedBool& input) const { return {!input.value}; }
  BoxedBool And(const BoxedBool& a, const BoxedBool& b) const {
    return {static_cast<bool>(a.value & b.value)};
  }
  BoxedBool Or(const BoxedBool& a, const BoxedBool& b) const {
    return {static_cast<bool>(a.value | b.value)};
  }
};

TEST(AbstractEvaluatorTest, Add) {
  TestAbstractEvaluator eval;
  Bits a = UBits(2, 32);
  Bits b = UBits(4, 32);
  Bits c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 6);

  a = UBits(1024, 32);
  b = UBits(1023, 32);
  c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 2047);

  a = UBits(1024768, 32);
  b = UBits(5893798, 32);
  c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 6918566);

  a = SBits(-1024, 32);
  b = SBits(1023, 32);
  c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), -1);
}

TEST(AbstractEvaluatorTest, Neg) {
  TestAbstractEvaluator eval;
  Bits a = SBits(4, 32);
  Bits b = FromBoxedVector(eval.Neg(ToBoxedVector(a)));
  EXPECT_EQ(b.ToInt64().value(), -4);

  a = SBits(1023, 32);
  b = FromBoxedVector(eval.Neg(ToBoxedVector(a)));
  EXPECT_EQ(b.ToInt64().value(), -1023);

  a = SBits(-1024, 32);
  b = FromBoxedVector(eval.Neg(ToBoxedVector(a)));
  EXPECT_EQ(b.ToInt64().value(), 1024);

  a = SBits(5893798, 32);
  b = FromBoxedVector(eval.Neg(ToBoxedVector(a)));
  EXPECT_EQ(b.ToInt64().value(), -5893798);

  a = SBits(0, 32);
  b = FromBoxedVector(eval.Neg(ToBoxedVector(a)));
  EXPECT_EQ(b.ToInt64().value(), 0);
}

TEST(AbstractEvaluatorTest, UMul) {
  TestAbstractEvaluator eval;
  Bits a = UBits(3, 8);
  Bits b = UBits(3, 8);
  Bits c = FromBoxedVector(eval.UMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 9);

  a = UBits(127, 10);
  b = UBits(64, 7);
  c = FromBoxedVector(eval.UMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 8128);
}

TEST(AbstractEvaluatorTest, SMul) {
  TestAbstractEvaluator eval;
  Bits a = SBits(3, 8);
  Bits b = SBits(5, 8);
  Bits c = FromBoxedVector(eval.SMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 15);

  a = SBits(-7, 4);
  b = SBits(-1, 4);
  c = FromBoxedVector(eval.SMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 7);

  a = SBits(127, 15);
  b = SBits(-64, 9);
  c = FromBoxedVector(eval.SMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), -8128);

  a = SBits(-127, 15);
  b = SBits(64, 9);
  c = FromBoxedVector(eval.SMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), -8128);

  a = SBits(-127, 64);
  b = SBits(-64, 64);
  c = FromBoxedVector(eval.SMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 8128);
}

TEST(AbstractEvaluatorTest, SLessThan) {
  TestAbstractEvaluator eval;
  for (int a = -4; a <= 3; ++a) {
    for (int b = -4; b <= 3; ++b) {
      EXPECT_EQ(
          eval.SLessThan(ToBoxedVector(SBits(a, 3)), ToBoxedVector(SBits(b, 3)))
              .value,
          a < b);
    }
  }
  Bits a = SBits(2, 32);
  Bits b = SBits(4, 32);
  EXPECT_EQ(eval.SLessThan(ToBoxedVector(a), ToBoxedVector(b)).value, 1);

  a = SBits(2, 32);
  b = SBits(-4, 32);
  EXPECT_EQ(eval.SLessThan(ToBoxedVector(a), ToBoxedVector(b)).value, 0);

  a = SBits(-2, 32);
  b = SBits(-4, 32);
  EXPECT_EQ(eval.SLessThan(ToBoxedVector(a), ToBoxedVector(b)).value, 0);

  a = SBits(-2, 32);
  b = SBits(4, 32);
  EXPECT_EQ(eval.SLessThan(ToBoxedVector(a), ToBoxedVector(b)).value, 1);

  a = SBits(0, 32);
  b = SBits(0, 32);
  EXPECT_EQ(eval.SLessThan(ToBoxedVector(a), ToBoxedVector(b)).value, 0);

  a = SBits(0, 32);
  b = SBits(16, 32);
  EXPECT_EQ(eval.SLessThan(ToBoxedVector(a), ToBoxedVector(b)).value, 1);

  a = SBits(0, 32);
  b = SBits(-16, 32);
  EXPECT_EQ(eval.SLessThan(ToBoxedVector(a), ToBoxedVector(b)).value, 0);
}

}  // namespace
}  // namespace xls
