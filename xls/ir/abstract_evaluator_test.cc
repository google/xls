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

#include <cstdint>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/ir/big_int.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/bits_test_utils.h"

namespace xls {
namespace {

// How many bytes we will let fuzz inputs to mul/div operations be. This is
// chosen to avoid timeout with inordinately long fuzz test cases.
static constexpr int64_t kMaxMulBytes = 100;

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

Bits FromBoxedVector(absl::Span<BoxedBool const> input) {
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
    return {static_cast<bool>(static_cast<int>(a.value) &
                              static_cast<int>(b.value))};
  }
  BoxedBool Or(const BoxedBool& a, const BoxedBool& b) const {
    return {static_cast<bool>(static_cast<int>(a.value) |
                              static_cast<int>(b.value))};
  }
  BoxedBool If(const BoxedBool& a, const BoxedBool& b,
               const BoxedBool& c) const {
    if (a.value) {
      return b;
    }
    return c;
  }
};

TEST(AbstractEvaluatorTest, Add) {
  TestAbstractEvaluator eval;
  Bits a = UBits(2, 32);
  Bits b = UBits(4, 32);
  Bits c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 6);

  a = UBits(1024, 32);
  b = UBits(1023, 32);
  c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 2047);

  a = UBits(1024768, 32);
  b = UBits(5893798, 32);
  c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 6918566);

  a = SBits(-1024, 32);
  b = SBits(1023, 32);
  c = FromBoxedVector(eval.Add(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), -1);
}

TEST(AbstractEvaluatorTest, AddWithCarry) {
  TestAbstractEvaluator eval;
  {
    Bits a = UBits(2, 32);
    Bits b = UBits(4, 32);
    auto c = eval.AddWithCarry(ToBoxedVector(a), ToBoxedVector(b));
    bool carry = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToUint64().value(), 6);
    EXPECT_FALSE(carry);
  }

  {
    Bits a = UBits(0xff, 8);
    Bits b = UBits(1, 8);
    auto c = eval.AddWithCarry(ToBoxedVector(a), ToBoxedVector(b));
    bool carry = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToUint64().value(), 0x00);
    EXPECT_TRUE(carry);
  }

  {
    Bits a = SBits(-1024, 32);
    Bits b = SBits(1023, 32);
    auto c = eval.AddWithCarry(ToBoxedVector(a), ToBoxedVector(b));
    bool carry = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -1);
    EXPECT_FALSE(carry);
  }
}

void AddWithCarryFuzz(uint8_t lhs, uint8_t rhs) {
  TestAbstractEvaluator eval;
  Bits a = UBits(lhs, 8);
  Bits b = UBits(rhs, 8);
  uint64_t l_big = lhs;
  uint64_t r_big = rhs;
  auto c = eval.AddWithCarry(ToBoxedVector(a), ToBoxedVector(b));
  uint64_t c_big = l_big + r_big;
  if (c.overflow.value) {
    // Overflow happened.
    EXPECT_GT(Bits::MinBitCountUnsigned(c_big), 8);
  } else {
    EXPECT_LE(Bits::MinBitCountUnsigned(c_big), 8);
  }
  EXPECT_EQ(FromBoxedVector(c.result), UBits(c_big, 64).Slice(0, 8));
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, AddWithCarryFuzz)
    .WithDomains(fuzztest::Arbitrary<uint8_t>(),
                 fuzztest::Arbitrary<uint8_t>());

TEST(AbstractEvaluatorTest, AddWithSignedOverflow) {
  TestAbstractEvaluator eval;
  {
    Bits a = UBits(2, 32);
    Bits b = UBits(4, 32);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToUint64().value(), 6);
    EXPECT_FALSE(overflow);
  }

  {
    Bits a = SBits(-2, 32);
    Bits b = SBits(-4, 32);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -6);
    EXPECT_FALSE(overflow);
  }
  {
    Bits a = SBits(-2, 32);
    Bits b = SBits(4, 32);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), 2);
    EXPECT_FALSE(overflow);
  }
  {
    Bits a = SBits(2, 32);
    Bits b = SBits(-4, 32);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -2);
    EXPECT_FALSE(overflow);
  }
  {
    Bits b = SBits(-4, 32);
    Bits a = SBits(2, 32);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -2);
    EXPECT_FALSE(overflow);
  }

  {
    Bits a = SBits(-120, 8);
    Bits b = SBits(-30, 8);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), 106);
    EXPECT_TRUE(overflow);
  }
  {
    Bits a = SBits(120, 8);
    Bits b = SBits(30, 8);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -106);
    EXPECT_TRUE(overflow);
  }
  {
    Bits a = Bits::MinSigned(8);
    Bits b = Bits::MaxSigned(8);
    auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
    bool overflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -1);
    EXPECT_FALSE(overflow);
  }
}

void AddWithOverflowFuzz(int8_t lhs, int8_t rhs) {
  TestAbstractEvaluator eval;
  Bits a = SBits(lhs, 8);
  Bits b = SBits(rhs, 8);
  int64_t l_big = lhs;
  int64_t r_big = rhs;
  auto c = eval.AddWithSignedOverflow(ToBoxedVector(a), ToBoxedVector(b));
  uint64_t c_big = l_big + r_big;
  if (c.overflow.value) {
    // Overflow happened.
    EXPECT_GT(Bits::MinBitCountSigned(c_big), 8);
  } else {
    EXPECT_LE(Bits::MinBitCountSigned(c_big), 8);
  }
  EXPECT_EQ(FromBoxedVector(c.result), SBits(c_big, 64).Slice(0, 8));
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, AddWithOverflowFuzz)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

TEST(AbstractEvaluatorTest, Sub) {
  TestAbstractEvaluator eval;
  {
    Bits a = UBits(2, 32);
    Bits b = UBits(4, 32);
    Bits c = FromBoxedVector(eval.Sub(ToBoxedVector(a), ToBoxedVector(b)));
    EXPECT_EQ(c.ToInt64().value(), -2);
  }
  {
    Bits a = UBits(4, 32);
    Bits b = UBits(2, 32);
    Bits c = FromBoxedVector(eval.Sub(ToBoxedVector(a), ToBoxedVector(b)));
    EXPECT_EQ(c.ToUint64().value(), 2);
  }
  {
    Bits a = SBits(12, 32);
    Bits b = SBits(-128, 32);
    Bits c = FromBoxedVector(eval.Sub(ToBoxedVector(a), ToBoxedVector(b)));
    EXPECT_EQ(c.ToUint64().value(), 140);
  }
}

void SubFuzz(uint8_t lhs, uint8_t rhs) {
  TestAbstractEvaluator eval;
  Bits a = UBits(lhs, 8);
  Bits b = UBits(rhs, 8);
  uint64_t l_big = lhs;
  uint64_t r_big = rhs;
  auto c = eval.Sub(ToBoxedVector(a), ToBoxedVector(b));
  uint64_t c_big = l_big - r_big;
  EXPECT_EQ(FromBoxedVector(c), SBits(c_big, 64).Slice(0, 8));
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, SubFuzz)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

TEST(AbstractEvaluatorTest, SubWithUnsignedUnderflow) {
  TestAbstractEvaluator eval;
  {
    Bits a = UBits(2, 32);
    Bits b = UBits(4, 32);
    auto c = eval.SubWithUnsignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -2);
    EXPECT_TRUE(underflow);
  }
  {
    Bits a = UBits(4, 32);
    Bits b = UBits(2, 32);
    auto c = eval.SubWithUnsignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToUint64().value(), 2);
    EXPECT_FALSE(underflow);
  }
  {
    Bits a = UBits(255, 8);
    Bits b = UBits(255, 8);
    auto c = eval.SubWithUnsignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToUint64().value(), 0);
    EXPECT_FALSE(underflow);
  }
  {
    Bits a = UBits(0, 8);
    Bits b = UBits(255, 8);
    auto c = eval.SubWithUnsignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToUint64().value(), 1);
    EXPECT_TRUE(underflow);
  }
  {
    Bits a = UBits(0, 8);
    Bits b = UBits(0x80, 8);
    auto c = eval.SubWithUnsignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToUint64().value(), 0x80);
    EXPECT_TRUE(underflow);
  }
}

void SubWithUnsignedUnderflowFuzz(uint8_t lhs, uint8_t rhs) {
  TestAbstractEvaluator eval;
  Bits a = UBits(lhs, 8);
  Bits b = UBits(rhs, 8);
  uint64_t l_big = lhs;
  uint64_t r_big = rhs;
  auto c = eval.SubWithUnsignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
  uint64_t c_big = l_big - r_big;
  if (c.overflow.value) {
    // Underflow happened.
    EXPECT_NE(bits_ops::ZeroExtend(FromBoxedVector(c.result), 64),
              UBits(c_big, 64));
  } else {
    EXPECT_EQ(bits_ops::ZeroExtend(FromBoxedVector(c.result), 64),
              UBits(c_big, 64));
  }
  EXPECT_EQ(FromBoxedVector(c.result), SBits(c_big, 64).Slice(0, 8));
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, SubWithUnsignedUnderflowFuzz)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

TEST(AbstractEvaluatorTest, SubWithSignedUnderflow) {
  TestAbstractEvaluator eval;
  {
    Bits a = SBits(2, 32);
    Bits b = SBits(4, 32);
    auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -2);
    EXPECT_FALSE(underflow);
  }
  {
    Bits a = SBits(4, 32);
    Bits b = SBits(2, 32);
    auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), 2);
    EXPECT_FALSE(underflow);
  }
  {
    Bits a = SBits(0, 8);
    Bits b = SBits(-128, 8);
    auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -128);
    EXPECT_TRUE(underflow);
  }
  {
    Bits a = SBits(-1, 8);
    Bits b = SBits(-128, 8);
    auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), 127);
    EXPECT_FALSE(underflow);
  }
  {
    Bits a = SBits(1, 8);
    Bits b = SBits(127, 8);
    auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -126);
    EXPECT_FALSE(underflow);
  }
  {
    Bits a = SBits(-23, 8);
    Bits b = SBits(120, 8);
    auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), 113);
    EXPECT_TRUE(underflow);
  }
  {
    Bits a = SBits(23, 8);
    Bits b = SBits(-120, 8);
    auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
    bool underflow = c.overflow.value;
    Bits result = FromBoxedVector(c.result);
    EXPECT_EQ(result.ToInt64().value(), -113);
    EXPECT_TRUE(underflow);
  }
}

void SubWithSignedUnderflowFuzz(int8_t lhs, int8_t rhs) {
  TestAbstractEvaluator eval;
  Bits a = SBits(lhs, 8);
  Bits b = SBits(rhs, 8);
  int64_t l_big = lhs;
  int64_t r_big = rhs;
  auto c = eval.SubWithSignedUnderflow(ToBoxedVector(a), ToBoxedVector(b));
  int64_t c_big = l_big - r_big;
  if (c.overflow.value) {
    // Underflow happened.
    EXPECT_NE(bits_ops::SignExtend(FromBoxedVector(c.result), 64),
              UBits(c_big, 64));
  } else {
    EXPECT_EQ(bits_ops::SignExtend(FromBoxedVector(c.result), 64),
              UBits(c_big, 64));
  }
  EXPECT_EQ(FromBoxedVector(c.result), SBits(c_big, 64).Slice(0, 8));
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, SubWithSignedUnderflowFuzz)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

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
  EXPECT_EQ(c.ToUint64().value(), 9);

  a = UBits(127, 10);
  b = UBits(64, 7);
  c = FromBoxedVector(eval.UMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 8128);
}

void EvaluatorMatchesReferenceUMul(const Bits& lhs, const Bits& rhs) {
  TestAbstractEvaluator eval;
  Bits got = FromBoxedVector(eval.UMul(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::UMul(lhs, rhs);

  EXPECT_EQ(got, want) << "unsigned: " << BigInt::MakeUnsigned(lhs) << " * "
                       << BigInt::MakeUnsigned(rhs) << " = "
                       << BigInt::MakeUnsigned(got)
                       << ", should be: " << BigInt::MakeUnsigned(want);
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUMul)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

TEST(AbstractEvaluatorTest, UMulWithOverflow) {
  TestAbstractEvaluator eval;
  Bits a = UBits(3, 8);
  Bits b = UBits(3, 8);
  auto c = eval.UMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
  EXPECT_EQ(FromBoxedVector(c.result).ToUint64().value(), 9);
  EXPECT_FALSE(c.overflow.value);

  a = UBits(127, 10);
  b = UBits(64, 7);
  c = eval.UMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
  EXPECT_EQ(FromBoxedVector(c.result).ToUint64().value(), 192);
  EXPECT_TRUE(c.overflow.value);
}

TEST(AbstractEvaluatorTest, SMulWithOverflow) {
  TestAbstractEvaluator eval;
  {
    Bits a = SBits(3, 8);
    Bits b = SBits(5, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), 15);
    EXPECT_FALSE(c.overflow.value);
  }
  {
    Bits a = SBits(-3, 8);
    Bits b = SBits(-5, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), 15);
    EXPECT_FALSE(c.overflow.value);
  }
  {
    Bits a = SBits(3, 8);
    Bits b = SBits(-5, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), -15);
    EXPECT_FALSE(c.overflow.value);
  }
  {
    Bits a = SBits(-3, 8);
    Bits b = SBits(5, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), -15);
    EXPECT_FALSE(c.overflow.value);
  }
  {
    Bits a = SBits(120, 8);
    Bits b = SBits(3, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), 104);
    EXPECT_TRUE(c.overflow.value);
  }
  {
    Bits a = SBits(-120, 8);
    Bits b = SBits(3, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), -104);
    EXPECT_TRUE(c.overflow.value);
  }
  {
    Bits a = SBits(120, 8);
    Bits b = SBits(-3, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), -104);
    EXPECT_TRUE(c.overflow.value);
  }
  {
    Bits a = SBits(-120, 8);
    Bits b = SBits(-3, 8);
    auto c = eval.SMulWithOverflow(ToBoxedVector(a), ToBoxedVector(b), 8);
    EXPECT_EQ(FromBoxedVector(c.result).ToInt64().value(), 104);
    EXPECT_TRUE(c.overflow.value);
  }
}

void EvaluatorMatchesReferenceSMul(const Bits& lhs, const Bits& rhs) {
  TestAbstractEvaluator eval;
  Bits got = FromBoxedVector(eval.SMul(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::SMul(lhs, rhs);
  EXPECT_EQ(got, want) << "signed: " << BigInt::MakeSigned(lhs) << " * "
                       << BigInt::MakeSigned(rhs) << " = "
                       << BigInt::MakeSigned(got)
                       << ", should be: " << BigInt::MakeSigned(want);
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSMul)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

TEST(AbstractEvaluatorTest, UDiv) {
  TestAbstractEvaluator eval;
  Bits a = UBits(4, 8);
  Bits b = UBits(1, 8);
  Bits c = FromBoxedVector(eval.UDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 4);

  a = UBits(1, 8);
  b = UBits(4, 8);
  c = FromBoxedVector(eval.UDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 0);

  a = UBits(4, 3);
  b = UBits(1, 3);
  c = FromBoxedVector(eval.UDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 4);
}

TEST(AbstractEvaluatorTest, Gate) {
  TestAbstractEvaluator eval;
  Bits b = UBits(4, 8);
  Bits c = FromBoxedVector(eval.Gate(BoxedBool{true}, ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 4);

  b = UBits(4, 8);
  c = FromBoxedVector(eval.Gate(BoxedBool{false}, ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 0);
}

TEST(AbstractEvaluatorTest, SDiv) {
  TestAbstractEvaluator eval;
  Bits a = SBits(4, 8);
  Bits b = SBits(1, 8);
  Bits c = FromBoxedVector(eval.SDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 4);

  a = SBits(-4, 8);
  b = SBits(1, 8);
  c = FromBoxedVector(eval.SDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), -4);

  a = SBits(4, 8);
  b = SBits(-1, 8);
  c = FromBoxedVector(eval.SDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), -4);

  a = SBits(-4, 8);
  b = SBits(-1, 8);
  c = FromBoxedVector(eval.SDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 4);

  a = SBits(1, 8);
  b = SBits(4, 8);
  c = FromBoxedVector(eval.SDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), 0);

  // Note: an unsigned 3-bit 4 is, when interpreted as signed, -4.
  a = UBits(4, 3);
  b = UBits(1, 3);
  c = FromBoxedVector(eval.SDiv(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToInt64().value(), -4);
}

void EvaluatorMatchesReferenceUDiv(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator eval;
  Bits got = FromBoxedVector(eval.UDiv(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::UDiv(lhs, rhs);
  EXPECT_EQ(got, want) << "unsigned: " << BigInt::MakeUnsigned(lhs) << " / "
                       << BigInt::MakeUnsigned(rhs) << " = "
                       << BigInt::MakeUnsigned(got)
                       << ", should be: " << BigInt::MakeUnsigned(want);
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUDiv)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

void EvaluatorMatchesReferenceSDiv(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator eval;
  Bits got = FromBoxedVector(eval.SDiv(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::SDiv(lhs, rhs);
  EXPECT_EQ(got, want) << "signed: " << BigInt::MakeSigned(lhs) << " / "
                       << BigInt::MakeSigned(rhs) << " = "
                       << BigInt::MakeSigned(got)
                       << ", should be: " << BigInt::MakeSigned(want);
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSDiv)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

void EvaluatorMatchesReferenceUMod(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator eval;
  Bits got = FromBoxedVector(eval.UMod(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::UMod(lhs, rhs);
  EXPECT_EQ(got, want) << "unsigned: " << BigInt::MakeUnsigned(lhs) << " % "
                       << BigInt::MakeUnsigned(rhs) << " = "
                       << BigInt::MakeUnsigned(got)
                       << ", should be: " << BigInt::MakeUnsigned(want);
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUMod)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

void EvaluatorMatchesReferenceSMod(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator eval;
  Bits got = FromBoxedVector(eval.SMod(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::SMod(lhs, rhs);
  EXPECT_EQ(got, want) << "signed: " << BigInt::MakeSigned(lhs) << " % "
                       << BigInt::MakeSigned(rhs) << " = "
                       << BigInt::MakeSigned(got)
                       << ", should be: " << BigInt::MakeSigned(want);
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSMod)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

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

TEST(AbstractEvaluatorTest, Select) {
  TestAbstractEvaluator eval;
  auto eval_select = [&](const Bits& selector, absl::Span<const Bits> cases,
                         std::optional<Bits> default_value = std::nullopt) {
    std::vector<std::vector<BoxedBool>> boxed_cases;
    for (auto const& i : cases) {
      boxed_cases.push_back(ToBoxedVector(i));
    }
    std::optional<std::vector<BoxedBool>> boxed_default_value;
    if (default_value.has_value()) {
      boxed_default_value = ToBoxedVector(*default_value);
    }
    return FromBoxedVector(eval.Select(
        ToBoxedVector(selector), eval.SpanOfVectorsToVectorOfSpans(boxed_cases),
        boxed_default_value));
  };

  EXPECT_EQ(
      eval_select(UBits(0, 2),
                  {UBits(0x00FF, 16), UBits(0x0FF0, 16), UBits(0xFF00, 16)},
                  UBits(0xF00F, 16)),
      UBits(0x00FF, 16));
  EXPECT_EQ(
      eval_select(UBits(1, 2),
                  {UBits(0x00FF, 16), UBits(0x0FF0, 16), UBits(0xFF00, 16)},
                  UBits(0xF00F, 16)),
      UBits(0x0FF0, 16));
  EXPECT_EQ(
      eval_select(UBits(2, 2),
                  {UBits(0x00FF, 16), UBits(0x0FF0, 16), UBits(0xFF00, 16)},
                  UBits(0xF00F, 16)),
      UBits(0xFF00, 16));
  EXPECT_EQ(
      eval_select(UBits(3, 2),
                  {UBits(0x00FF, 16), UBits(0x0FF0, 16), UBits(0xFF00, 16)},
                  UBits(0xF00F, 16)),
      UBits(0xF00F, 16));

  EXPECT_EQ(eval_select(UBits(0, 2), {UBits(0x00EE, 16)}, UBits(0xE00E, 16)),
            UBits(0x00EE, 16));
  EXPECT_EQ(eval_select(UBits(1, 2), {UBits(0x00EE, 16)}, UBits(0xE00E, 16)),
            UBits(0xE00E, 16));
  EXPECT_EQ(eval_select(UBits(2, 2), {UBits(0x00EE, 16)}, UBits(0xE00E, 16)),
            UBits(0xE00E, 16));
  EXPECT_EQ(eval_select(UBits(3, 2), {UBits(0x00EE, 16)}, UBits(0xE00E, 16)),
            UBits(0xE00E, 16));

  EXPECT_EQ(eval_select(UBits(0, 2), {}, UBits(0xD00D, 16)), UBits(0xD00D, 16));
  EXPECT_EQ(eval_select(UBits(1, 2), {}, UBits(0xD00D, 16)), UBits(0xD00D, 16));
  EXPECT_EQ(eval_select(UBits(2, 2), {}, UBits(0xD00D, 16)), UBits(0xD00D, 16));
  EXPECT_EQ(eval_select(UBits(3, 2), {}, UBits(0xD00D, 16)), UBits(0xD00D, 16));

  EXPECT_EQ(eval_select(UBits(0, 1), {UBits(0x00CC, 16), UBits(0xCC00, 16)}),
            UBits(0x00CC, 16));
  EXPECT_EQ(eval_select(UBits(1, 1), {UBits(0x00CC, 16), UBits(0xCC00, 16)}),
            UBits(0xCC00, 16));

  EXPECT_EQ(eval_select(UBits(0, 0), {}, UBits(0x0BB0, 16)), UBits(0x0BB0, 16));
}

TEST(AbstractEvaluatorTest, PrioritySelect) {
  TestAbstractEvaluator eval;
  auto test_eq = [&](int64_t expected, const Bits& selector,
                     absl::Span<const Bits> cases, bool selector_can_be_zero,
                     const Bits& default_value) {
    std::vector<std::vector<BoxedBool>> boxed_cases;
    for (auto const& i : cases) {
      boxed_cases.push_back(ToBoxedVector(i));
    }
    std::vector<BoxedBool> boxed_default_value = ToBoxedVector(default_value);
    EXPECT_EQ(UBits(expected, default_value.bit_count()),
              FromBoxedVector(eval.PrioritySelect(
                  ToBoxedVector(selector),
                  eval.SpanOfVectorsToVectorOfSpans(boxed_cases),
                  selector_can_be_zero, boxed_default_value)));
  };

  test_eq(0x00FF, UBits(1, 2), {UBits(0x00FF, 16), UBits(0xFF00, 16)}, false,
          UBits(0x0FF0, 16));
  test_eq(0xFF00, UBits(2, 2), {UBits(0x00FF, 16), UBits(0xFF00, 16)}, false,
          UBits(0x0FF0, 16));
  test_eq(0x00FF, UBits(3, 2), {UBits(0x00FF, 16), UBits(0xFF00, 16)}, false,
          UBits(0x0FF0, 16));
  test_eq(0x00FF, UBits(1, 2), {UBits(0x00FF, 16), UBits(0xFF00, 16)}, true,
          UBits(0x0FF0, 16));
  test_eq(0xFF00, UBits(2, 2), {UBits(0x00FF, 16), UBits(0xFF00, 16)}, true,
          UBits(0x0FF0, 16));
  test_eq(0x00FF, UBits(3, 2), {UBits(0x00FF, 16), UBits(0xFF00, 16)}, true,
          UBits(0x0FF0, 16));
  test_eq(0x0FF0, UBits(0, 2), {UBits(0x00FF, 16), UBits(0xFF00, 16)}, true,
          UBits(0x0FF0, 16));
  test_eq(0x0FF0, UBits(0, 0), {}, true, UBits(0x0FF0, 16));
}

TEST(AbstractEvaluatorTest, BitSliceUpdate) {
  TestAbstractEvaluator eval;
  auto test_eq = [&](int64_t expected, const Bits& a, const Bits& start,
                     const Bits& value) {
    EXPECT_EQ(
        UBits(expected, a.bit_count()),
        FromBoxedVector(eval.BitSliceUpdate(
            ToBoxedVector(a), ToBoxedVector(start), ToBoxedVector(value))));
  };

  test_eq(0x123f, UBits(0x1234, 16), UBits(0, 32), UBits(0xf, 4));
  test_eq(0x12f4, UBits(0x1234, 16), UBits(4, 32), UBits(0xf, 4));
  test_eq(0xf234, UBits(0x1234, 16), UBits(12, 32), UBits(0xf, 4));
  test_eq(0x1234, UBits(0x1234, 16), UBits(16, 32), UBits(0xf, 4));
  test_eq(0x1234, UBits(0x1234, 16), UBits(100000, 32), UBits(0xf, 4));

  test_eq(0xcd, UBits(0x12, 8), UBits(0, 32), UBits(0xabcd, 16));
  test_eq(0xd2, UBits(0x12, 8), UBits(4, 32), UBits(0xabcd, 16));
  test_eq(0x12, UBits(0x12, 8), UBits(8, 32), UBits(0xabcd, 16));
}

TEST(AbstractEvaluatorTest, BitSliceUpdateConsts) {
  TestAbstractEvaluator eval;
  auto test_eq = [&](int64_t expected, const Bits& a, const int64_t& start,
                     const Bits& value) {
    EXPECT_EQ(UBits(expected, a.bit_count()),
              FromBoxedVector(eval.BitSliceUpdate(ToBoxedVector(a), start,
                                                  ToBoxedVector(value))));
  };

  test_eq(0x123f, UBits(0x1234, 16), 0, UBits(0xf, 4));
  test_eq(0x12f4, UBits(0x1234, 16), 4, UBits(0xf, 4));
  test_eq(0xf234, UBits(0x1234, 16), 12, UBits(0xf, 4));
  test_eq(0x1234, UBits(0x1234, 16), 16, UBits(0xf, 4));
  test_eq(0x1234, UBits(0x1234, 16), 100000, UBits(0xf, 4));

  test_eq(0xcd, UBits(0x12, 8), 0, UBits(0xabcd, 16));
  test_eq(0xd2, UBits(0x12, 8), 4, UBits(0xabcd, 16));
  test_eq(0x12, UBits(0x12, 8), 8, UBits(0xabcd, 16));
}

void UMulMatches32BitMultiplication(uint32_t a, uint32_t b) {
  TestAbstractEvaluator eval;
  Bits a_bits = UBits(a, 32);
  Bits b_bits = UBits(b, 32);
  Bits c =
      FromBoxedVector(eval.UMul(ToBoxedVector(a_bits), ToBoxedVector(b_bits)));
  EXPECT_EQ(static_cast<uint32_t>(c.ToUint64().value()), a * b);
}
FUZZ_TEST(AbstractEvaluatorFuzzTest, UMulMatches32BitMultiplication);

}  // namespace
}  // namespace xls
