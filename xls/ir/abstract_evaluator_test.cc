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
    .WithDomains(NonemptyBits(), NonemptyBits());

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
    .WithDomains(NonemptyBits(), NonemptyBits());

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
    .WithDomains(NonemptyBits(), NonemptyBits());

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
    .WithDomains(NonemptyBits(), NonemptyBits());

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
    .WithDomains(NonemptyBits(), NonemptyBits());

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
    .WithDomains(NonemptyBits(), NonemptyBits());

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
    EXPECT_EQ(UBits(expected, cases.front().bit_count()),
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
