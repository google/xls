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
#include "xls/common/status/matchers.h"
#include "xls/ir/big_int.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/bits_test_utils.h"

namespace xls {
namespace {

template <typename EvaluatorT>
class AbstractEvaluatorTest : public ::testing::Test {
 public:
  using Evaluator = EvaluatorT;
};
using testing::Types;

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

template <bool kIsITEFundamentalArg>
class TestAbstractEvaluator;

template <bool kIsITEFundamentalArg>
struct AbstractEvaluatorOptionsForTest
    : public AbstractEvaluatorOptions<
          TestAbstractEvaluator<kIsITEFundamentalArg>> {
  static constexpr bool kIsITEFundamental = kIsITEFundamentalArg;
};

template <bool kIsITEFundamental>
class TestAbstractEvaluator
    : public AbstractEvaluator<
          BoxedBool, TestAbstractEvaluator<kIsITEFundamental>,
          AbstractEvaluatorOptionsForTest<kIsITEFundamental>> {
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

using Implementations =
    Types<TestAbstractEvaluator<false>, TestAbstractEvaluator<true>>;

TYPED_TEST_SUITE(AbstractEvaluatorTest, Implementations);

TYPED_TEST(AbstractEvaluatorTest, Add) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, AddWithCarry) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void AddWithCarryFuzz(uint8_t lhs, uint8_t rhs) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
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

void AddWithCarryFuzzNormal(uint8_t lhs, uint8_t rhs) {
  AddWithCarryFuzz<false>(lhs, rhs);
}

void AddWithCarryFuzzITE(uint8_t lhs, uint8_t rhs) {
  AddWithCarryFuzz<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, AddWithCarryFuzzNormal)
    .WithDomains(fuzztest::Arbitrary<uint8_t>(),
                 fuzztest::Arbitrary<uint8_t>());

FUZZ_TEST(AbstractEvaluatorFuzzTest, AddWithCarryFuzzITE)
    .WithDomains(fuzztest::Arbitrary<uint8_t>(),
                 fuzztest::Arbitrary<uint8_t>());

TYPED_TEST(AbstractEvaluatorTest, AddWithSignedOverflow) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void AddWithOverflowFuzz(int8_t lhs, int8_t rhs) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
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

void AddWithOverflowFuzzNormal(int8_t lhs, int8_t rhs) {
  AddWithOverflowFuzz<false>(lhs, rhs);
}

void AddWithOverflowFuzzITE(int8_t lhs, int8_t rhs) {
  AddWithOverflowFuzz<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, AddWithOverflowFuzzNormal)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

FUZZ_TEST(AbstractEvaluatorFuzzTest, AddWithOverflowFuzzITE)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

TYPED_TEST(AbstractEvaluatorTest, Sub) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void SubFuzz(uint8_t lhs, uint8_t rhs) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits a = UBits(lhs, 8);
  Bits b = UBits(rhs, 8);
  uint64_t l_big = lhs;
  uint64_t r_big = rhs;
  auto c = eval.Sub(ToBoxedVector(a), ToBoxedVector(b));
  uint64_t c_big = l_big - r_big;
  EXPECT_EQ(FromBoxedVector(c), SBits(c_big, 64).Slice(0, 8));
}

void SubFuzzNormal(uint8_t lhs, uint8_t rhs) { SubFuzz<false>(lhs, rhs); }

void SubFuzzITE(uint8_t lhs, uint8_t rhs) { SubFuzz<true>(lhs, rhs); }

FUZZ_TEST(AbstractEvaluatorFuzzTest, SubFuzzNormal)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

FUZZ_TEST(AbstractEvaluatorFuzzTest, SubFuzzITE)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

TYPED_TEST(AbstractEvaluatorTest, SubWithUnsignedUnderflow) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void SubWithUnsignedUnderflowFuzz(uint8_t lhs, uint8_t rhs) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
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

void SubWithUnsignedUnderflowFuzzNormal(uint8_t lhs, uint8_t rhs) {
  SubWithUnsignedUnderflowFuzz<false>(lhs, rhs);
}

void SubWithUnsignedUnderflowFuzzITE(uint8_t lhs, uint8_t rhs) {
  SubWithUnsignedUnderflowFuzz<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, SubWithUnsignedUnderflowFuzzNormal)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

FUZZ_TEST(AbstractEvaluatorFuzzTest, SubWithUnsignedUnderflowFuzzITE)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

TYPED_TEST(AbstractEvaluatorTest, SubWithSignedUnderflow) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void SubWithSignedUnderflowFuzz(int8_t lhs, int8_t rhs) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
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

void SubWithSignedUnderflowFuzzNormal(int8_t lhs, int8_t rhs) {
  SubWithSignedUnderflowFuzz<false>(lhs, rhs);
}

void SubWithSignedUnderflowFuzzITE(int8_t lhs, int8_t rhs) {
  SubWithSignedUnderflowFuzz<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, SubWithSignedUnderflowFuzzNormal)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

FUZZ_TEST(AbstractEvaluatorFuzzTest, SubWithSignedUnderflowFuzzITE)
    .WithDomains(fuzztest::Arbitrary<int8_t>(), fuzztest::Arbitrary<int8_t>());

TYPED_TEST(AbstractEvaluatorTest, Neg) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, UMul) {
  typename TestFixture::Evaluator eval;
  Bits a = UBits(3, 8);
  Bits b = UBits(3, 8);
  Bits c = FromBoxedVector(eval.UMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 9);

  a = UBits(127, 10);
  b = UBits(64, 7);
  c = FromBoxedVector(eval.UMul(ToBoxedVector(a), ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 8128);
}

template <bool kIsITEFundamentalArg>
void EvaluatorMatchesReferenceUMul(const Bits& lhs, const Bits& rhs) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits got = FromBoxedVector(eval.UMul(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::UMul(lhs, rhs);

  EXPECT_EQ(got, want) << "unsigned: " << BigInt::MakeUnsigned(lhs) << " * "
                       << BigInt::MakeUnsigned(rhs) << " = "
                       << BigInt::MakeUnsigned(got)
                       << ", should be: " << BigInt::MakeUnsigned(want);
}

void EvaluatorMatchesReferenceUMulNormal(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceUMul<false>(lhs, rhs);
}

void EvaluatorMatchesReferenceUMulITE(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceUMul<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUMulNormal)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUMulITE)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

TYPED_TEST(AbstractEvaluatorTest, UMulWithOverflow) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, SMulWithOverflow) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void EvaluatorMatchesReferenceSMul(const Bits& lhs, const Bits& rhs) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits got = FromBoxedVector(eval.SMul(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::SMul(lhs, rhs);
  EXPECT_EQ(got, want) << "signed: " << BigInt::MakeSigned(lhs) << " * "
                       << BigInt::MakeSigned(rhs) << " = "
                       << BigInt::MakeSigned(got)
                       << ", should be: " << BigInt::MakeSigned(want);
}

void EvaluatorMatchesReferenceSMulNormal(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceSMul<false>(lhs, rhs);
}

void EvaluatorMatchesReferenceSMulITE(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceSMul<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSMulNormal)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSMulITE)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

TYPED_TEST(AbstractEvaluatorTest, UDiv) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, Gate) {
  typename TestFixture::Evaluator eval;
  Bits b = UBits(4, 8);
  Bits c = FromBoxedVector(eval.Gate(BoxedBool{true}, ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 4);

  b = UBits(4, 8);
  c = FromBoxedVector(eval.Gate(BoxedBool{false}, ToBoxedVector(b)));
  EXPECT_EQ(c.ToUint64().value(), 0);
}

TYPED_TEST(AbstractEvaluatorTest, SDiv) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void EvaluatorMatchesReferenceUDiv(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits got = FromBoxedVector(eval.UDiv(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::UDiv(lhs, rhs);
  EXPECT_EQ(got, want) << "unsigned: " << BigInt::MakeUnsigned(lhs) << " / "
                       << BigInt::MakeUnsigned(rhs) << " = "
                       << BigInt::MakeUnsigned(got)
                       << ", should be: " << BigInt::MakeUnsigned(want);
}

void EvaluatorMatchesReferenceUDivNormal(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceUDiv<false>(lhs, rhs);
}

void EvaluatorMatchesReferenceUDivITE(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceUDiv<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUDivNormal)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUDivITE)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

template <bool kIsITEFundamentalArg>
void EvaluatorMatchesReferenceSDiv(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits got = FromBoxedVector(eval.SDiv(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::SDiv(lhs, rhs);
  EXPECT_EQ(got, want) << "signed: " << BigInt::MakeSigned(lhs) << " / "
                       << BigInt::MakeSigned(rhs) << " = "
                       << BigInt::MakeSigned(got)
                       << ", should be: " << BigInt::MakeSigned(want);
}

void EvaluatorMatchesReferenceSDivNormal(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceSDiv<false>(lhs, rhs);
}

void EvaluatorMatchesReferenceSDivITE(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceSDiv<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSDivNormal)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSDivITE)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

template <bool kIsITEFundamentalArg>
void EvaluatorMatchesReferenceUMod(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits got = FromBoxedVector(eval.UMod(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::UMod(lhs, rhs);
  EXPECT_EQ(got, want) << "unsigned: " << BigInt::MakeUnsigned(lhs) << " % "
                       << BigInt::MakeUnsigned(rhs) << " = "
                       << BigInt::MakeUnsigned(got)
                       << ", should be: " << BigInt::MakeUnsigned(want);
}

void EvaluatorMatchesReferenceUModNormal(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceUMod<false>(lhs, rhs);
}

void EvaluatorMatchesReferenceUModITE(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceUMod<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUModNormal)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceUModITE)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

template <bool kIsITEFundamentalArg>
void EvaluatorMatchesReferenceSMod(const Bits& lhs, const Bits& rhs) {
  if (rhs.IsZero()) {
    return;
  }
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits got = FromBoxedVector(eval.SMod(ToBoxedVector(lhs), ToBoxedVector(rhs)));
  Bits want = bits_ops::SMod(lhs, rhs);
  EXPECT_EQ(got, want) << "signed: " << BigInt::MakeSigned(lhs) << " % "
                       << BigInt::MakeSigned(rhs) << " = "
                       << BigInt::MakeSigned(got)
                       << ", should be: " << BigInt::MakeSigned(want);
}

void EvaluatorMatchesReferenceSModNormal(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceSMod<false>(lhs, rhs);
}

void EvaluatorMatchesReferenceSModITE(const Bits& lhs, const Bits& rhs) {
  EvaluatorMatchesReferenceSMod<true>(lhs, rhs);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSModNormal)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

FUZZ_TEST(AbstractEvaluatorFuzzTest, EvaluatorMatchesReferenceSModITE)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 NonemptyBits(/*max_byte_count=*/kMaxMulBytes));

TYPED_TEST(AbstractEvaluatorTest, SMul) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, SLessThan) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, PrioritySelect) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, Select) {
  typename TestFixture::Evaluator eval;
  auto test_it = [&](Bits selector, absl::Span<uint32_t const> cases,
                     std::optional<uint32_t> default_value = std::nullopt) {
    std::vector<std::vector<BoxedBool>> cases_vec;
    for (const auto& case_val : cases) {
      cases_vec.push_back(ToBoxedVector(UBits(case_val, 32)));
    }
    std::optional<std::vector<BoxedBool>> default_val = std::nullopt;
    if (default_value.has_value()) {
      default_val = ToBoxedVector(UBits(*default_value, 32));
    }
    int64_t expected = selector.ToUint64().value() < cases.size()
                           ? cases[selector.ToUint64().value()]
                           : default_value.value_or(0);
    EXPECT_EQ(UBits(expected, 32),
              FromBoxedVector(eval.Select(
                  ToBoxedVector(selector),
                  eval.SpanOfVectorsToVectorOfSpans(cases_vec), default_val)));
  };
  test_it(UBits(0, 1), {0, 1});
  test_it(UBits(1, 1), {0, 1});
  test_it(UBits(4, 8), {0, 1}, 2);
  test_it(UBits(4, 8), {0, 1, 2, 3, 4, 5, 6}, 7);
  test_it(UBits(4, 3), {0, 1, 2, 3, 4, 5, 6, 7});
  test_it(UBits(2, 4), {0, 1, 2, 3, 4, 5, 6, 7}, 8);
}

TYPED_TEST(AbstractEvaluatorTest, Shift) {
  typename TestFixture::Evaluator eval;
  auto test_eq = [&](int64_t expected, const Bits& input, const Bits& amount) {
    EXPECT_EQ(UBits(expected, input.bit_count()),
              FromBoxedVector(eval.ShiftRightArith(ToBoxedVector(input),
                                                   ToBoxedVector(amount))))
        << expected << " != " << input << " >> " << amount;
  };
  test_eq(0, UBits(0, 0), UBits(12, 12));
  test_eq(0xff, UBits(0x80, 8), UBits(7, 12));
  test_eq(0x01, UBits(0x40, 8), UBits(6, 12));
}

TYPED_TEST(AbstractEvaluatorTest, BitSliceUpdate) {
  typename TestFixture::Evaluator eval;
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

TYPED_TEST(AbstractEvaluatorTest, BitSliceUpdateConsts) {
  typename TestFixture::Evaluator eval;
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

template <bool kIsITEFundamentalArg>
void UMulMatches32BitMultiplication(uint32_t a, uint32_t b) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits a_bits = UBits(a, 32);
  Bits b_bits = UBits(b, 32);
  Bits c =
      FromBoxedVector(eval.UMul(ToBoxedVector(a_bits), ToBoxedVector(b_bits)));
  EXPECT_EQ(static_cast<uint32_t>(c.ToUint64().value()), a * b);
}

void UMulMatches32BitMultiplicationNormal(uint32_t a, uint32_t b) {
  UMulMatches32BitMultiplication<false>(a, b);
}

void UMulMatches32BitMultiplicationITE(uint32_t a, uint32_t b) {
  UMulMatches32BitMultiplication<true>(a, b);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, UMulMatches32BitMultiplicationNormal);
FUZZ_TEST(AbstractEvaluatorFuzzTest, UMulMatches32BitMultiplicationITE);

TYPED_TEST(AbstractEvaluatorTest, Decode) {
  typename TestFixture::Evaluator eval;
  EXPECT_EQ(UBits(0b00001000, 8),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(3, 4)), 8)));
  EXPECT_EQ(UBits(0b00001000, 8),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(3, 400)), 8)));
  EXPECT_EQ(UBits(0b00000000, 8),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(300, 400)), 8)));
  EXPECT_EQ(UBits(0b00000001, 8),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(0, 33)), 8)));

  // Test result_width = 0
  EXPECT_EQ(UBits(0, 0),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(0, 1)), 0)));
  EXPECT_EQ(UBits(0, 0),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(5, 4)), 0)));

  // Test result_width = 1
  EXPECT_EQ(UBits(0b1, 1),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(0, 1)), 1)));
  EXPECT_EQ(UBits(0b0, 1),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(1, 1)), 1)));
  EXPECT_EQ(UBits(0b0, 1),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(5, 4)), 1)));

  // Recursive cases on result_width = 8 (used_bits = 5)
  // Value 8 is out of range [0, 7], so should be 0
  EXPECT_EQ(UBits(0, 8),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(8, 6)), 8)));
  // Value 7 is the last in-range value, should be 0b10000000
  EXPECT_EQ(UBits(0b10000000, 8),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(7, 6)), 8)));
  // Value 32 (high bit set on 6-bit input, greater than used_bits = 5) yields 0
  EXPECT_EQ(UBits(0, 8),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(32, 6)), 8)));

  // ZeroExtend optimization case: input.size() < 64 && (1 << input.size()) <
  // result_width result_width = 10, input.size() = 3. (2^3 = 8 < 10) Value 3
  // decoded in 8-wide is 0b00001000, zero-extended to 10-wide is 0b0000001000
  EXPECT_EQ(UBits(0b0000001000, 10),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(3, 3)), 10)));
  EXPECT_EQ(UBits(0b0000001000, 10000),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(3, 3)), 10000)));
  // Value 7 decoded in 8-wide is 0b10000000, zero-extended to 10-wide is
  // 0b0010000000
  EXPECT_EQ(UBits(0b0010000000, 10),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(7, 3)), 10)));

  // Base case with no optimizations: result_width = 10, input.size() = 4
  EXPECT_EQ(UBits(0b1000000000, 10),
            FromBoxedVector(eval.Decode(ToBoxedVector(UBits(9, 4)), 10)));
}

TYPED_TEST(AbstractEvaluatorTest, DynamicBitSlice) {
  typename TestFixture::Evaluator eval;
  auto test_eq = [&](int64_t expected, const Bits& a, const Bits& start,
                     const int64_t& width) {
    EXPECT_EQ(UBits(expected, width),
              FromBoxedVector(eval.DynamicBitSlice(
                  ToBoxedVector(a), ToBoxedVector(start), width)))
        << "expected: " << expected << ", a: " << a.ToDebugString() << " (" << a
        << "), start: " << start.ToDebugString() << " (" << start
        << "), width: " << width;
  };

  // NB 0x1234 == 4660
  test_eq(0x4, UBits(0x1234, 16), UBits(0, 32), 4);
  test_eq(0x4, UBits(0x1234, 16), UBits(0, 1), 4);
  test_eq(0x3, UBits(0x1234, 16), UBits(4, 32), 4);
  test_eq(0x2, UBits(0x1234, 16), UBits(8, 32), 4);
  test_eq(0x2, UBits(0x1234, 16), UBits(8, 32), 4);
  test_eq(0x2, UBits(0x1234, 16), UBits(8, 4), 4);
  test_eq(0x1, UBits(0x1234, 16), UBits(12, 32), 4);
  test_eq(0x0, UBits(0x1234, 16), UBits(16, 32), 4);
  test_eq(0x0, UBits(0x1234, 16), UBits(20, 32), 4);

  test_eq(0b1011, UBits(0b1000110011101111, 16), UBits(2, 32), 4);
  test_eq(0b1011, UBits(0b1000110011101111, 16), UBits(2, 2), 4);
  test_eq(0b1101, UBits(0b1000110011101111, 16), UBits(3, 2), 4);
}

template <bool kIsITEFundamentalArg>
void DecodeMatchesReference(const Bits& a, int32_t result_width) {
  TestAbstractEvaluator<kIsITEFundamentalArg> eval;
  Bits got = FromBoxedVector(eval.Decode(ToBoxedVector(a), result_width));
  Bits want;
  if (a.FitsInNBitsUnsigned(63)) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t shift, a.ToUint64());
    want = bits_ops::ShiftLeftLogical(UBits(1, result_width), shift);
  } else {
    want = Bits(result_width);
  }
  EXPECT_EQ(got, want) << "Decode(" << a << ", " << result_width
                       << ") = " << got << ", should be: " << want;
}

void DecodeMatchesReferenceNormal(const Bits& a, int32_t result_width) {
  DecodeMatchesReference<false>(a, result_width);
}

void DecodeMatchesReferenceITE(const Bits& a, int32_t result_width) {
  DecodeMatchesReference<true>(a, result_width);
}

FUZZ_TEST(AbstractEvaluatorFuzzTest, DecodeMatchesReferenceNormal)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 fuzztest::Positive<int32_t>());

FUZZ_TEST(AbstractEvaluatorFuzzTest, DecodeMatchesReferenceITE)
    .WithDomains(NonemptyBits(/*max_byte_count=*/kMaxMulBytes),
                 fuzztest::Positive<int32_t>());

TEST(AbstractEvaluatorFuzzTest, DecodeMatchesReferenceRegression) {
  DecodeMatchesReferenceNormal(UBits(1, 1), 1054074905);
  DecodeMatchesReferenceITE(UBits(1, 1), 1054074905);
}

}  // namespace
}  // namespace xls
