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

#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/big_int.h"
#include "xls/ir/bits.h"
#include "xls/tools/testbench_builder.h"

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

// Performs random UMul and SMul implementations as constructed by the
// TestAbstractEvaluator and compares them to the bits_ops reference
// implementation (up to 16 bit values on either side).
void DoMulRandoms(int seed, int64_t iterations) {
  TestAbstractEvaluator eval;

  std::mt19937 rng(seed);

  std::uniform_int_distribution<int64_t> bit_count_dist(1, 16);

  for (int64_t i = 0; i < iterations; ++i) {
    int64_t lhs_bits = bit_count_dist(rng);
    int64_t rhs_bits = bit_count_dist(rng);
    uint64_t lhs_value =
        std::uniform_int_distribution<uint64_t>(0, (1 << lhs_bits) - 1)(rng);
    uint64_t rhs_value =
        std::uniform_int_distribution<uint64_t>(0, (1 << rhs_bits) - 1)(rng);
    Bits lhs = UBits(lhs_value, /*bit_count=*/lhs_bits);
    Bits rhs = UBits(rhs_value, /*bit_count=*/rhs_bits);
    Bits umul_got =
        FromBoxedVector(eval.UMul(ToBoxedVector(lhs), ToBoxedVector(rhs)));
    Bits umul_want = bits_ops::UMul(lhs, rhs);
    EXPECT_EQ(umul_got, umul_want) << "lhs: " << lhs << " rhs: " << rhs;

    Bits smul_got =
        FromBoxedVector(eval.SMul(ToBoxedVector(lhs), ToBoxedVector(rhs)));
    Bits smul_want = bits_ops::SMul(lhs, rhs);
    EXPECT_EQ(smul_got, smul_want) << "lhs: " << lhs << " rhs: " << rhs;
  }
}

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

// Performs random UDiv, SDiv, UMod, and SMod implementations as constructed by
// the TestAbstractEvaluator and compares them to the bits_ops reference
// implementation (up to 16 bit values on either side).
void DoDivRandoms(int seed, int64_t iterations) {
  TestAbstractEvaluator eval;

  std::mt19937 rng(seed);

  std::uniform_int_distribution<int64_t> bit_count_dist(1, 16);

  for (int64_t i = 0; i < iterations; ++i) {
    int64_t lhs_bits = bit_count_dist(rng);
    int64_t rhs_bits = bit_count_dist(rng);
    uint64_t lhs_value =
        std::uniform_int_distribution<uint64_t>(0, (1 << lhs_bits) - 1)(rng);
    uint64_t rhs_value =
        std::uniform_int_distribution<uint64_t>(0, (1 << rhs_bits) - 1)(rng);
    Bits lhs = UBits(lhs_value, /*bit_count=*/lhs_bits);
    Bits rhs = UBits(rhs_value, /*bit_count=*/rhs_bits);

    Bits udiv_got =
        FromBoxedVector(eval.UDiv(ToBoxedVector(lhs), ToBoxedVector(rhs)));
    Bits udiv_want = bits_ops::UDiv(lhs, rhs);
    EXPECT_EQ(udiv_got, udiv_want) << "lhs: " << lhs << " rhs: " << rhs;

    Bits umod_got =
        FromBoxedVector(eval.UMod(ToBoxedVector(lhs), ToBoxedVector(rhs)));
    Bits umod_want = bits_ops::UMod(lhs, rhs);
    EXPECT_EQ(umod_got, umod_want) << "lhs: " << lhs << " rhs: " << rhs;

    Bits sdiv_got =
        FromBoxedVector(eval.SDiv(ToBoxedVector(lhs), ToBoxedVector(rhs)));
    Bits sdiv_want = bits_ops::SDiv(lhs, rhs);
    EXPECT_EQ(sdiv_got, sdiv_want)
        << "lhs: " << BigInt::MakeSigned(lhs)
        << " rhs: " << BigInt::MakeSigned(rhs)
        << " got: " << BigInt::MakeSigned(sdiv_got)
        << " want: " << BigInt::MakeSigned(sdiv_want);

    Bits smod_got =
        FromBoxedVector(eval.SMod(ToBoxedVector(lhs), ToBoxedVector(rhs)));
    Bits smod_want = bits_ops::SMod(lhs, rhs);
    EXPECT_EQ(smod_got, smod_want)
        << "lhs: " << BigInt::MakeSigned(lhs)
        << " rhs: " << BigInt::MakeSigned(rhs)
        << " got: " << BigInt::MakeSigned(smod_got)
        << " want: " << BigInt::MakeSigned(smod_want);
  }
}

// Note: a bit of manual sharding is easy and still gets us more coverage, since
// BitGen is constructed with a nondeterministic seed on each construction.

constexpr int64_t kRandomsPerShard = 4 * 1024;
TEST(AbstractEvaluatorTest, MulRandomsShard0) {
  DoMulRandoms(0, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, MulRandomsShard1) {
  DoMulRandoms(1, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, MulRandomsShard2) {
  DoMulRandoms(2, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, MulRandomsShard3) {
  DoMulRandoms(3, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, MulRandomsShard4) {
  DoMulRandoms(4, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, MulRandomsShard5) {
  DoMulRandoms(5, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, MulRandomsShard6) {
  DoMulRandoms(6, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, MulRandomsShard7) {
  DoMulRandoms(7, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard0) {
  DoDivRandoms(0, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard1) {
  DoDivRandoms(1, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard2) {
  DoDivRandoms(2, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard3) {
  DoDivRandoms(3, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard4) {
  DoDivRandoms(4, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard5) {
  DoDivRandoms(5, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard6) {
  DoDivRandoms(6, kRandomsPerShard);
}
TEST(AbstractEvaluatorTest, DivRandomsShard7) {
  DoDivRandoms(7, kRandomsPerShard);
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

// This test is temporary - the goal is to replace it (and the other random UMul
// testing above) with a RapidCheck-like macro. Currently this test demonstrates
// a non-JIT Testbench use case to try and simplify.
TEST(AbstractEvaluatorTest, SpeedyCheckUMul) {
  using InputT = std::tuple<uint32_t, uint32_t>;
  using ResultT = uint32_t;

  auto compute_expected = [](InputT input) {
    return std::get<0>(input) * std::get<1>(input);
  };
  auto compute_actual = [](InputT input) {
    TestAbstractEvaluator eval;
    uint32_t a = std::get<0>(input);
    uint32_t b = std::get<1>(input);
    Bits a_bits = UBits(a, 32);
    Bits b_bits = UBits(b, 32);
    Bits c = FromBoxedVector(
        eval.UMul(ToBoxedVector(a_bits), ToBoxedVector(b_bits)));
    return static_cast<uint32_t>(c.ToUint64().value());
  };
  TestbenchBuilder<InputT, ResultT> builder(compute_expected, compute_actual);
  XLS_ASSERT_OK(builder.Build().Run());
}

}  // namespace
}  // namespace xls
