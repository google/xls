// Copyright 2023 The XLS Authors
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

#include "xls/ir/interval_ops.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fuzztest/fuzztest.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/interval_test_utils.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/ternary.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace xls::interval_ops {

namespace {
using solvers::z3::IsProvenTrue;

IntervalSet SetOf(absl::Span<const Interval> intervals) {
  IntervalSet is(intervals.front().BitCount());
  absl::c_for_each(intervals, [&](auto v) { is.AddInterval(v); });
  is.Normalize();
  return is;
}

TEST(IntervalOpsTest, BitsPrecise) {
  IntervalSet is = SetOf({Interval::Precise(UBits(21, 8))});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, Bits::AllOnes(8));
  EXPECT_EQ(known.known_bit_values, UBits(21, 8));
}

TEST(IntervalOpsTest, BitsMaximal) {
  IntervalSet is = SetOf({Interval::Maximal(8)});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, Bits(8));
  EXPECT_EQ(known.known_bit_values, Bits(8));
}

TEST(IntervalOpsTest, BitsHalfFull) {
  IntervalSet is = SetOf({Interval::Maximal(4).ZeroExtend(8)});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, UBits(0xf0, 8));
  EXPECT_EQ(known.known_bit_values, Bits(8));
}

TEST(IntervalOpsTest, MiddleOut) {
  IntervalSet is = SetOf({Interval(UBits(0, 8), UBits(0x4, 8)),
                          Interval(UBits(0x10, 8), UBits(0x14, 8))});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, UBits(0xe8, 8));
  EXPECT_EQ(known.known_bit_values, Bits(8));
}

TEST(IntervalOpsTest, MiddleOutHigh) {
  IntervalSet is = SetOf({Interval(UBits(0xe0, 8), UBits(0xe4, 8)),
                          Interval(UBits(0xf0, 8), UBits(0xf4, 8))});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, UBits(0xe8, 8));
  EXPECT_EQ(known.known_bit_values, UBits(0xe0, 8));
}

TEST(IntervalOpsTest, MiddleOutTernary) {
  IntervalSet is = SetOf({Interval(UBits(0, 8), UBits(0x4, 8)),
                          Interval(UBits(0x10, 8), UBits(0x14, 8))});
  auto known = ExtractTernaryVector(is);
  TernaryVector expected{
      TernaryValue::kUnknown,   TernaryValue::kUnknown,
      TernaryValue::kUnknown,   TernaryValue::kKnownZero,
      TernaryValue::kUnknown,   TernaryValue::kKnownZero,
      TernaryValue::kKnownZero, TernaryValue::kKnownZero,
  };
  EXPECT_EQ(known, expected);
}

IntervalSet FromRanges(absl::Span<std::pair<int64_t, int64_t> const> ranges,
                       int64_t bits) {
  IntervalSet res(bits);
  for (const auto& [l, h] : ranges) {
    res.AddInterval(Interval::Closed(UBits(l, bits), UBits(h, bits)));
  }
  res.Normalize();
  return res;
}

IntervalSet FromTernaryString(std::string_view sv,
                              int64_t max_unknown_bits = 4) {
  auto tern_status = StringToTernaryVector(sv);
  if (!tern_status.ok()) {
    ADD_FAILURE() << "Unable to parse ternary string " << sv << "\n"
                  << tern_status.status();
    return IntervalSet(1);
  }
  return interval_ops::FromTernary(tern_status.value(), max_unknown_bits);
}

TEST(IntervalOpsTest, FromTernaryExact) {
  EXPECT_EQ(FromTernaryString("0b111000"),
            IntervalSet::Precise(UBits(0b111000, 6)));
}

TEST(IntervalOpsTest, FromTernaryAllUnknown) {
  EXPECT_EQ(FromTernaryString("0bXXXXXX"), IntervalSet::Maximal(6));
}

TEST(IntervalOpsTest, FromTernaryUnknownTrailing) {
  EXPECT_EQ(FromTernaryString("0b1010XXX"),
            FromRanges({{0b1010000, 0b1010111}}, 7));
}

TEST(IntervalOpsTest, FromTernarySegments) {
  EXPECT_EQ(FromTernaryString("0bXX1010XXX"),
            FromRanges({{0b001010000, 0b001010111},
                        {0b011010000, 0b011010111},
                        {0b101010000, 0b101010111},
                        {0b111010000, 0b111010111}},
                       9));
  EXPECT_EQ(
      FromTernaryString("0b0X1010XXX"),
      FromRanges({{0b001010000, 0b001010111}, {0b011010000, 0b011010111}}, 9));
  EXPECT_EQ(FromTernaryString("0b1X0X1010XXX"),
            FromRanges({{0b10001010000, 0b10001010111},
                        {0b10011010000, 0b10011010111},
                        {0b11001010000, 0b11001010111},
                        {0b11011010000, 0b11011010111}},
                       11));
}
TEST(IntervalOpsTest, FromTernaryPreciseSegments) {
  XLS_ASSERT_OK_AND_ASSIGN(auto tern, StringToTernaryVector("0b1X0X1X0X1"));
  IntervalSet expected(tern.size());
  for (const Bits& v : ternary_ops::AllBitsValues(tern)) {
    expected.AddInterval(Interval::Precise(v));
  }
  expected.Normalize();
  EXPECT_EQ(interval_ops::FromTernary(tern, /*max_interval_bits=*/4), expected);
}
TEST(IntervalOpsTest, FromTernaryPreciseSegmentsBig) {
  XLS_ASSERT_OK_AND_ASSIGN(auto tern,
                           StringToTernaryVector("0b1X0X1XXXXXXXXX0X1"));
  IntervalSet expected(tern.size());
  for (const Bits& v : ternary_ops::AllBitsValues(tern)) {
    expected.AddInterval(Interval::Precise(v));
  }
  expected.Normalize();
  EXPECT_EQ(interval_ops::FromTernary(tern, /*max_interval_bits=*/12),
            expected);
}

TEST(IntervalOpsTest, FromTernarySegmentsExtended) {
  // Only allow 4 segments so first 5 bits are all considered unknown
  EXPECT_EQ(FromTernaryString("0bXX_1X_0X0X", /*max_unknown_bits=*/2),
            FromRanges({{0b00100000, 0b00111111},
                        {0b01100000, 0b01111111},
                        {0b10100000, 0b10111111},
                        {0b11100000, 0b11111111}},
                       8));
  // Only allow 2 segments
  EXPECT_EQ(
      FromTernaryString("0b1X1_X1X0X", /*max_unknown_bits=*/1),
      FromRanges({{0b10100000, 0b10111111}, {0b11100000, 0b11111111}}, 8));
  // Only allow 1 segment
  EXPECT_EQ(FromTernaryString("0b1X1_X1X0X", /*max_unknown_bits=*/0),
            FromRanges({{0b10000000, 0b11111111}}, 8));
}

void OpFuzz(
    std::string_view name,
    const std::function<BValue(FunctionBuilder&, absl::Span<BValue const>)>&
        ir_op,
    const std::function<IntervalSet(absl::Span<IntervalSet const>)>& op,
    absl::Span<absl::Span<std::pair<int64_t, int64_t> const> const> args,
    int64_t bits = 16) {
  std::vector<IntervalSet> is_args;
  is_args.reserve(args.size());
  for (const absl::Span<std::pair<int64_t, int64_t> const>& arg : args) {
    is_args.push_back(FromRanges(arg, bits));
  }
  IntervalSet res = op(is_args);
  VerifiedPackage p(name);
  FunctionBuilder fb(absl::StrCat(name, "_test_func"), &p);
  std::vector<BValue> params;
  params.reserve(args.size());
  for (int64_t i = 0; i < args.size(); ++i) {
    params.push_back(
        fb.Param(absl::StrFormat("param_%d", i), p.GetBitsType(bits)));
  }
  auto is_in_intervals = [&](const IntervalSet& set, BValue inp) -> BValue {
    std::vector<BValue> components;
    components.reserve(set.NumberOfIntervals());
    if (set.BitCount() > inp.BitCountOrDie()) {
      inp = fb.ZeroExtend(inp, set.BitCount());
    }
    for (const Interval& i : set.Intervals()) {
      components.push_back(fb.And(fb.UGe(inp, fb.Literal(i.LowerBound())),
                                  fb.ULe(inp, fb.Literal(i.UpperBound()))));
    }
    return fb.Or(components);
  };
  BValue result = ir_op(fb, params);
  std::vector<BValue> inputs_implication;
  inputs_implication.reserve(args.size());
  for (int64_t i = 0; i < args.size(); ++i) {
    inputs_implication.push_back(
        fb.Not(is_in_intervals(is_args[i], params[i])));
  }
  inputs_implication.push_back(is_in_intervals(res, result));
  // left_is_in_range && right_is_in_range \implies result_is_in_range
  BValue implication = fb.Or(inputs_implication);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedMaybeRecord smr_in("input_ranges", is_args);
  ScopedMaybeRecord smr_out("output_ranges", res);
  ScopedMaybeRecord smr_ir("ir", p.DumpIr());
  EXPECT_THAT(solvers::z3::TryProve(f, implication.node(),
                                    solvers::z3::Predicate::NotEqualToZero(),
                                    absl::InfiniteDuration()),
              status_testing::IsOkAndHolds(IsProvenTrue()));
}

void BinaryOpFuzz(
    std::string_view name,
    std::function<BValue(FunctionBuilder&, BValue, BValue)> ir_op,
    std::function<IntervalSet(const IntervalSet&, const IntervalSet&)> op,
    absl::Span<std::pair<int64_t, int64_t> const> lhs,
    absl::Span<std::pair<int64_t, int64_t> const> rhs, int64_t bits = 16) {
  OpFuzz(
      name, [&](auto& fb, auto args) { return ir_op(fb, args[0], args[1]); },
      [&](auto args) { return op(args[0], args[1]); }, {lhs, rhs}, bits);
}
void UnaryOpFuzz(std::string_view name,
                 std::function<BValue(FunctionBuilder&, BValue)> ir_op,
                 std::function<IntervalSet(const IntervalSet&)> op,
                 absl::Span<std::pair<int64_t, int64_t> const> lhs,
                 int64_t bits = 16) {
  OpFuzz(
      name, [&](auto& fb, auto args) { return ir_op(fb, args[0]); },
      [&](auto args) { return op(args[0]); }, {lhs}, bits);
}

auto BitsRange(int64_t bits) {
  return fuzztest::InRange(int64_t{0}, int64_t{1 << bits} - 1);
}
auto IntervalDomain(int64_t bits) {
  return fuzztest::UniqueElementsVectorOf(
             fuzztest::PairOf(BitsRange(bits), BitsRange(bits)))
      .WithMaxSize(8)
      .WithMinSize(1);
}

TEST(IntervalOpsTest, Add) {
  {
    IntervalSet lhs = FromRanges({{0, 0}}, 64);
    IntervalSet rhs = FromRanges({{1, 10}, {21, 30}}, 64);
    EXPECT_EQ(Add(lhs, rhs), rhs);
    EXPECT_EQ(Add(rhs, lhs), rhs);
  }
  {
    IntervalSet lhs = FromRanges({{5, 10}}, 64);
    IntervalSet rhs = FromRanges({{0, 10}, {20, 30}}, 64);
    EXPECT_EQ(Add(lhs, rhs), FromRanges({{5, 20}, {25, 40}}, 64));
  }
}

void AddZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
               absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "add",
      [](FunctionBuilder& fb, BValue l, BValue r) { return fb.Add(l, r); }, Add,
      lhs, rhs, /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, AddZ3Fuzz)
    .WithDomains(IntervalDomain(16), IntervalDomain(16));

TEST(IntervalOpsTest, Sub) {
  {
    IntervalSet lhs = FromRanges({{1, 10}, {21, 30}}, 64);
    IntervalSet rhs = FromRanges({{0, 0}}, 64);
    EXPECT_EQ(Sub(lhs, rhs), lhs);
  }
  {
    IntervalSet lhs = FromRanges({{5, 10}, {20, 30}}, 64);
    IntervalSet rhs = FromRanges({{1, 4}}, 64);
    EXPECT_EQ(Sub(lhs, rhs), FromRanges({{1, 9}, {16, 29}}, 64));
  }
  {
    IntervalSet lhs = FromRanges({{5, 10}}, 8);
    IntervalSet rhs = FromRanges({{1, 6}}, 8);
    EXPECT_EQ(Sub(lhs, rhs), FromRanges({{255, 255}, {0, 9}}, 8));
  }
}

void SubZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
               absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "sub",
      [](FunctionBuilder& fb, BValue l, BValue r) { return fb.Subtract(l, r); },
      Sub, lhs, rhs, /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, SubZ3Fuzz)
    .WithDomains(IntervalDomain(16), IntervalDomain(16));

// Test the transform used to perform signed comparisons (with some loss of
// precision).
void AddSubIntMaxIsIdentity(absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  auto l = IntervalSet::Precise(Bits::MinSigned(16));
  auto r = FromRanges(rhs, 16);
  if (r.NumberOfIntervals() > 5 || Add(l, r).NumberOfIntervals() > 5) {
    // Precision loss will happen before doing sub breaking identity.
    return;
  }
  EXPECT_EQ(Sub(Add(l, r), l), r)
      << "add is add(" << l << ", " << r << ") = " << Add(l, r);
}
FUZZ_TEST(IntervalOpsTest, AddSubIntMaxIsIdentity)
    .WithDomains(IntervalDomain(16));

void AddSubConstantIntIsIdentity(
    int64_t constant, absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  auto l = IntervalSet::Precise(UBits(constant, 8));
  auto r = FromRanges(rhs, 8);
  if (r.NumberOfIntervals() > 5 || Add(l, r).NumberOfIntervals() > 5) {
    // Precision loss will happen before doing sub breaking identity.
    return;
  }
  EXPECT_EQ(Sub(Add(l, r), l), r)
      << "add is add(" << l << ", " << r << ") = " << Add(l, r);
}
FUZZ_TEST(IntervalOpsTest, AddSubConstantIntIsIdentity)
    .WithDomains(
        fuzztest::InRange<int64_t>(0, std::numeric_limits<uint8_t>::max()),
        IntervalDomain(8));

MATCHER_P(IntervalSupersetOf, initial,
          absl::StrFormat("Is %sa superset of %s", negation ? "not " : "",
                          initial.ToString())) {
  const IntervalSet& target = initial;
  auto comp = IntervalSet::Complement(arg);
  return testing::ExplainMatchResult(
      testing::IsEmpty(), IntervalSet::Intersect(comp, target).Intervals(),
      result_listener);
}

void AddSubIsGeneralSuperset(
    absl::Span<std::pair<int64_t, int64_t> const> lhs,
    absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  auto l = FromRanges(lhs, 16);
  auto r = FromRanges(rhs, 16);
  EXPECT_THAT(Add(Sub(l, r), r), IntervalSupersetOf(l));
  EXPECT_THAT(Sub(Add(l, r), r), IntervalSupersetOf(l));
}
FUZZ_TEST(IntervalOpsTest, AddSubIsGeneralSuperset)
    .WithDomains(IntervalDomain(16), IntervalDomain(16));

TEST(IntervalOpsTest, Neg) {
  {
    IntervalSet v = FromRanges({{0, 3}, {9, 90}}, 64);
    EXPECT_EQ(Neg(v), FromRanges({{0, 0}, {-3, -1}, {-90, -9}}, 64));
  }
}

void NegZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> vals) {
  UnaryOpFuzz(
      "neg", [](FunctionBuilder& fb, BValue v) { return fb.Negate(v); }, Neg,
      vals);
}
FUZZ_TEST(IntervalOpsTest, NegZ3Fuzz).WithDomains(IntervalDomain(16));

TEST(IntervalOpsTest, UMul) {
  {
    IntervalSet lhs = FromRanges({{1, 10}, {21, 30}}, 64);
    IntervalSet rhs = FromRanges({{0, 0}}, 64);
    EXPECT_EQ(UMul(lhs, rhs, 64), rhs);
    EXPECT_EQ(UMul(rhs, lhs, 64), rhs);
  }
  {
    IntervalSet lhs = FromRanges({{5, 10}, {200, 300}}, 64);
    IntervalSet rhs = FromRanges({{2, 4}}, 64);
    EXPECT_EQ(UMul(lhs, rhs, 64), FromRanges({{10, 40}, {400, 1200}}, 64));
    EXPECT_EQ(UMul(rhs, lhs, 64), FromRanges({{10, 40}, {400, 1200}}, 64));
  }
  {
    IntervalSet lhs = FromRanges({{2, 3}}, 8);
    IntervalSet rhs = FromRanges({{100, 100}}, 8);
    EXPECT_EQ(UMul(lhs, rhs, 8), FromRanges({{200, 255}, {0, 44}}, 8));
  }
}

void UMulZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
                absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "umul",
      [](FunctionBuilder& fb, BValue l, BValue r) { return fb.UMul(l, r, 10); },
      [](const auto& l, const auto& r) { return UMul(l, r, 10); }, lhs, rhs,
      /*bits=*/8);
}
FUZZ_TEST(IntervalOpsTest, UMulZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));

TEST(IntervalOpsTest, UDiv) {
  {
    IntervalSet lhs = FromRanges({{2, 12}, {100, 200}}, 16);
    IntervalSet rhs = FromRanges({{2, 2}}, 16);
    EXPECT_EQ(UDiv(lhs, rhs), FromRanges({{1, 6}, {50, 100}}, 16));
  }
  {
    IntervalSet lhs = FromRanges({{2, 12}, {100, 200}}, 16);
    IntervalSet rhs = FromRanges({{0, 2}}, 16);
    EXPECT_EQ(UDiv(lhs, rhs),
              FromRanges({{1, 12}, {50, 200}, {0xffff, 0xffff}}, 16));
  }
  {
    IntervalSet lhs = FromRanges({{2, 12}, {100, 200}}, 16);
    IntervalSet rhs = FromRanges({{1, 2}}, 16);
    EXPECT_EQ(UDiv(lhs, rhs), FromRanges({{1, 12}, {50, 200}}, 16));
  }
  {
    IntervalSet lhs = FromRanges({{0, 12}, {100, 200}}, 16);
    IntervalSet rhs = FromRanges({{0, 0}}, 16);
    EXPECT_EQ(UDiv(lhs, rhs), FromRanges({{0xffff, 0xffff}}, 16));
  }
}
void UDivZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
                absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "udiv",
      [](FunctionBuilder& fb, BValue l, BValue r) { return fb.UDiv(l, r); },
      [](const auto& l, const auto& r) { return UDiv(l, r); }, lhs, rhs,
      /*bits=*/8);
}
FUZZ_TEST(IntervalOpsTest, UDivZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));

TEST(IntervalOpsTest, Concat) {
  {
    IntervalSet lhs = FromRanges({{2, 2}}, 2);
    IntervalSet rhs = FromRanges({{0x2, 0xb}, {0xba, 0xfa}}, 8);
    EXPECT_EQ(Concat({lhs, rhs}),
              FromRanges({{0x202, 0x20b}, {0x2ba, 0x2fa}}, 10));
  }
}

void ConcatZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
                  absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "concat",
      [](FunctionBuilder& fb, BValue l, BValue r) { return fb.Concat({l, r}); },
      [](const auto& l, const auto& r) { return Concat({l, r}); }, lhs, rhs,
      /*bits=*/8);
}
FUZZ_TEST(IntervalOpsTest, ConcatZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));

TEST(IntervalOpsTest, SignExtend) {
  {
    IntervalSet lhs = FromRanges({{2, 12}}, 8);
    EXPECT_EQ(SignExtend(lhs, 32), FromRanges({{2, 12}}, 32));
  }
  {
    IntervalSet lhs = FromRanges({{12, 0xfc /* -4 */}}, 8);
    EXPECT_EQ(SignExtend(lhs, 32), FromRanges({{12, 0xfffffffc /* -4 */}}, 32));
  }
}

void SignExtendZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
                      int8_t bits) {
  UnaryOpFuzz(
      "sign_extend",
      [&](FunctionBuilder& fb, BValue l) { return fb.SignExtend(l, bits); },
      [&](const auto& l) { return SignExtend(l, bits); }, lhs,
      /*bits=*/8);
}
FUZZ_TEST(IntervalOpsTest, SignExtendZ3Fuzz)
    .WithDomains(IntervalDomain(8), fuzztest::InRange<int8_t>(9, 16));

TEST(IntervalOpsTest, ZeroExtend) {
  {
    IntervalSet lhs = FromRanges({{2, 12}}, 8);
    EXPECT_EQ(ZeroExtend(lhs, 32), FromRanges({{2, 12}}, 32));
  }
  {
    IntervalSet lhs = FromRanges({{12, 0xfc /* -4 */}}, 8);
    EXPECT_EQ(ZeroExtend(lhs, 32), FromRanges({{12, 0xfc}}, 32));
  }
}

void ZeroExtendZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
                      int8_t bits) {
  UnaryOpFuzz(
      "zero_extend",
      [&](FunctionBuilder& fb, BValue l) { return fb.ZeroExtend(l, bits); },
      [&](const auto& l) { return ZeroExtend(l, bits); }, lhs,
      /*bits=*/8);
}
FUZZ_TEST(IntervalOpsTest, ZeroExtendZ3Fuzz)
    .WithDomains(IntervalDomain(8), fuzztest::InRange<int8_t>(9, 16));

TEST(IntervalOpsTest, Truncate) {
  {
    IntervalSet lhs = FromRanges({{0xaff, 0xfff}}, 12);
    EXPECT_EQ(Truncate(lhs, 8), FromRanges({{0, 0xff}}, 8));
  }
  {
    IntervalSet lhs = FromRanges({{0xa30, 0xa40}}, 12);
    EXPECT_EQ(Truncate(lhs, 8), FromRanges({{0x30, 0x40}}, 8));
  }
  {
    IntervalSet lhs = FromRanges({{0xa40, 0xb20}}, 12);
    EXPECT_EQ(Truncate(lhs, 8), FromRanges({{0, 0x20}, {0x40, 0xff}}, 8));
  }
}

void TruncateZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
                    int8_t bits) {
  UnaryOpFuzz(
      "truncate",
      [&](FunctionBuilder& fb, BValue l) { return fb.BitSlice(l, 0, bits); },
      [&](const auto& l) { return Truncate(l, bits); }, lhs,
      /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, TruncateZ3Fuzz)
    .WithDomains(IntervalDomain(8), fuzztest::InRange<int8_t>(1, 15));

void EqZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
              absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "eq",
      [&](FunctionBuilder& fb, BValue l, BValue r) { return fb.Eq(l, r); }, Eq,
      lhs, rhs,
      /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, EqZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));

void NeZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
              absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "ne",
      [&](FunctionBuilder& fb, BValue l, BValue r) { return fb.Ne(l, r); }, Ne,
      lhs, rhs,
      /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, NeZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));
void SLtZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
               absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "slt",
      [&](FunctionBuilder& fb, BValue l, BValue r) { return fb.SLt(l, r); },
      SLt, lhs, rhs,
      /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, SLtZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));
void SGtZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
               absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "sgt",
      [&](FunctionBuilder& fb, BValue l, BValue r) { return fb.SGt(l, r); },
      SGt, lhs, rhs,
      /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, SGtZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));

void ULtZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
               absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "ult",
      [&](FunctionBuilder& fb, BValue l, BValue r) { return fb.ULt(l, r); },
      ULt, lhs, rhs,
      /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, ULtZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));

void UGtZ3Fuzz(absl::Span<std::pair<int64_t, int64_t> const> lhs,
               absl::Span<std::pair<int64_t, int64_t> const> rhs) {
  BinaryOpFuzz(
      "ugt",
      [&](FunctionBuilder& fb, BValue l, BValue r) { return fb.UGt(l, r); },
      UGt, lhs, rhs,
      /*bits=*/16);
}
FUZZ_TEST(IntervalOpsTest, UGtZ3Fuzz)
    .WithDomains(IntervalDomain(8), IntervalDomain(8));

TEST(MinimizeIntervalsTest, PrefersEarlyIntervals) {
  // All 32 6-bit [0, 63] even numbers.
  IntervalSet even_numbers =
      FromTernaryString("0bXXXXX0", /*max_unknown_bits=*/5);

  EXPECT_EQ(MinimizeIntervals(even_numbers, 1), FromRanges({{0, 62}}, 6));

  EXPECT_EQ(MinimizeIntervals(even_numbers, 2),
            FromRanges(
                {
                    // earlier entries are prefered.
                    {62, 62},
                    {0, 60},
                },
                6));

  EXPECT_EQ(MinimizeIntervals(even_numbers, 4),
            FromRanges(
                {
                    // earlier entries are prefered.
                    {62, 62},
                    {60, 60},
                    {58, 58},
                    {0, 56},
                },
                6));

  // More than number of intervals
  EXPECT_EQ(MinimizeIntervals(even_numbers, 40), even_numbers);

  // exactly the number of intervals
  EXPECT_EQ(MinimizeIntervals(even_numbers, 32), even_numbers);
}

TEST(MinimizeIntervalsTest, PrefersSmallerGaps) {
  IntervalSet source_intervals =
      // 0 - 255 range. 8 segments
      FromRanges(
          {
              // 2 to the end.
              {253, 253},
              // 103 gap
              {150, 150},
              // 20 gap
              {130, 130},
              // 10 gap
              {120, 120},
              // 5 gap
              {115, 115},
              // 10 gap
              {105, 105},
              // 20 gap
              {85, 85},
              // 82 gap
              {2, 2},
              // 2 gap to 0
          },
          8);

  ASSERT_EQ(source_intervals.NumberOfIntervals(), 8);

  EXPECT_EQ(MinimizeIntervals(source_intervals, 7),
            FromRanges(
                {
                    // 2 to the end.
                    {253, 253},
                    // 103 gap
                    {150, 150},
                    // 20 gap
                    {130, 130},
                    // 10 gap
                    {115, 120},
                    // 5 gap
                    // {115, 115}, -- merged with above.
                    // 10 gap
                    {105, 105},
                    // 20 gap
                    {85, 85},
                    // 82 gap
                    {2, 2},
                    // 2 gap to 0
                },
                8));

  EXPECT_EQ(MinimizeIntervals(source_intervals, 6),
            FromRanges(
                {
                    // 2 to the end.
                    {253, 253},
                    // 103 gap
                    {150, 150},
                    // 20 gap
                    {130, 130},
                    // 10 gap
                    {105, 120},
                    // 5 gap
                    // {115, 115}, -- merged with above.
                    // 10 gap
                    // {105, 105}, -- merged with above.
                    // 20 gap
                    {85, 85},
                    // 82 gap
                    {2, 2},
                    // 2 gap to 0
                },
                8));

  EXPECT_EQ(MinimizeIntervals(source_intervals, 5),
            FromRanges(
                {
                    // 2 to the end.
                    {253, 253},
                    // 103 gap
                    {150, 150},
                    // 20 gap
                    {105, 130},
                    // 10 gap
                    // {120, 120}, -- merged with above
                    // 5 gap
                    // {115, 115}, -- merged with above.
                    // 10 gap
                    // {105, 105}, -- merged with above.
                    // 20 gap
                    {85, 85},
                    // 82 gap
                    {2, 2},
                    // 2 gap to 0
                },
                8));
}

TEST(MinimizeIntervalsTest, MergeMultipleGroups) {
  IntervalSet source_intervals = FromRanges(
      {
          {130, 138},
          // 1 gap
          {120, 128},
          // 21 gap
          {90, 98},
          // 1 gap
          {80, 88},
          // 21 gap
          {50, 58},
          // 1 gap
          {40, 48},
          // 21 gap
          {20, 28},
          // 1 gap
          {10, 18},
      },
      8);

  ASSERT_EQ(source_intervals.NumberOfIntervals(), 8);

  EXPECT_EQ(MinimizeIntervals(source_intervals, 7),
            FromRanges(
                {
                    {130, 138},
                    // 1 gap
                    {120, 128},
                    // 21 gap
                    {90, 98},
                    // 1 gap
                    {80, 88},
                    // 21 gap
                    {50, 58},
                    // 1 gap
                    {40, 48},
                    // 21 gap
                    {10, 28},
                    // 1 gap
                    // {10, 18}, -- merge with above.
                },
                8));

  EXPECT_EQ(MinimizeIntervals(source_intervals, 6),
            FromRanges(
                {
                    {130, 138},
                    // 1 gap
                    {120, 128},
                    // 21 gap
                    {90, 98},
                    // 1 gap
                    {80, 88},
                    // 21 gap
                    {40, 58},
                    // 1 gap
                    // {40, 48}, -- merge with above.
                    // 21 gap
                    {10, 28},
                    // 1 gap
                    // {10, 18}, -- merge with above.
                },
                8));

  EXPECT_EQ(MinimizeIntervals(source_intervals, 5),
            FromRanges(
                {
                    {130, 138},
                    // 1 gap
                    {120, 128},
                    // 21 gap
                    {80, 98},
                    // 1 gap
                    // {80, 88}, -- merge with above.
                    // 21 gap
                    {40, 58},
                    // 1 gap
                    // {40, 48}, -- merge with above.
                    // 21 gap
                    {10, 28},
                    // 1 gap
                    // {10, 18}, -- merge with above.
                },
                8));

  EXPECT_EQ(MinimizeIntervals(source_intervals, 4),
            FromRanges(
                {
                    {120, 138},
                    // 1 gap
                    // {120, 128}, -- merge with above
                    // 21 gap
                    {80, 98},
                    // 1 gap
                    // {80, 88}, -- merge with above.
                    // 21 gap
                    {40, 58},
                    // 1 gap
                    // {40, 48}, -- merge with above.
                    // 21 gap
                    {10, 28},
                    // 1 gap
                    // {10, 18}, -- merge with above.
                },
                8));
}

void MinimizeIntervalsGeneratesSuperset(
    const std::vector<std::pair<int64_t, int64_t>>& ranges,
    int64_t requested_size) {
  IntervalSet source = FromRanges(ranges, 64);
  IntervalSet minimized = MinimizeIntervals(source, requested_size);

  ASSERT_LE(minimized.NumberOfIntervals(), requested_size);
  ASSERT_EQ(IntervalSet::Intersect(source, minimized), source);
}
FUZZ_TEST(MinimizeIntervalsTest, MinimizeIntervalsGeneratesSuperset)
    .WithDomains(fuzztest::VectorOf<>(
                     fuzztest::PairOf(fuzztest::NonNegative<int64_t>(),
                                      fuzztest::NonNegative<int64_t>())),
                 fuzztest::InRange<int64_t>(1, 256));

void CoversTernaryWorksForIntervals(const Interval& interval,
                                    TernarySpan ternary) {
  EXPECT_EQ(interval_ops::CoversTernary(interval, ternary),
            interval.ForEachElement([&](const Bits& element) {
              return ternary ==
                     ternary_ops::Intersection(
                         ternary_ops::BitsToTernary(element), ternary);
            }))
      << "interval: "
      << absl::StrFormat("[%s, %s]", interval.LowerBound().ToDebugString(),
                         interval.UpperBound().ToDebugString())
      << ", ternary: " << ToString(ternary);
}
FUZZ_TEST(IntervalOpsFuzzTest, CoversTernaryWorksForIntervals)
    .WithDomains(ArbitraryInterval(8),
                 fuzztest::VectorOf(fuzztest::ElementOf({
                                        TernaryValue::kKnownZero,
                                        TernaryValue::kKnownOne,
                                        TernaryValue::kUnknown,
                                    }))
                     .WithSize(8));

}  // namespace
}  // namespace xls::interval_ops
