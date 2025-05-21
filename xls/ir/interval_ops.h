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

#ifndef XLS_IR_INTERVAL_OPS_H_
#define XLS_IR_INTERVAL_OPS_H_

#include <cstdint>
#include <optional>

#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"

namespace xls::interval_ops {

// Convert a ternary into the corresponding (normalized) interval-set.
//
// To prevent creating a huge number of intervals in the set the
// max_interval_bits controls how many unknown, non-trailing bits we consider.
// The resulting interval set will have up to 1 << max_interval_bits intervals.
// A perfect interval set can be obtained by setting this to ternary.size() - 1
// but this can cost significant memory.
//
// By default up to 16 intervals (meaning 4 non-trailing unknown bits) will be
// created.
//
// Intervals will be expanded until the interval-set will remain under the
// requested size.
IntervalSet FromTernary(TernarySpan ternary, int64_t max_interval_bits = 4);

// Extract the ternary vector embedded in the interval-sets.
//
// Note this is linear in the number of intervals that make up the interval-set.
// If performance is an issue using the convex-hull might be worth considering.
TernaryVector ExtractTernaryVector(const IntervalSet& intervals,
                                   std::optional<Node*> source = std::nullopt);

// Determine whether the given `intervals` include any element that matches the
// given `ternary` span.
bool CoversTernary(const Interval& interval, TernarySpan ternary);
bool CoversTernary(const IntervalSet& intervals, TernarySpan ternary);

struct KnownBits {
  Bits known_bits;
  Bits known_bit_values;
};

// Extract the known-bits embedded in the interval-sets.
//
// Note this is linear in the number of intervals that make up the interval-set.
// If performance is an issue using the convex-hull might be worth considering.
KnownBits ExtractKnownBits(const IntervalSet& intervals,
                           std::optional<Node*> source = std::nullopt);

// Minimize interval set to 'size' by merging some intervals together. Intervals
// are chosen with a greedy algorithm that minimizes the number of additional
// values the overall interval set contains. That is first it will add the
// smallest components posible. In cases where multiple gaps are the same size
// it will prioritize earlier gaps over later ones.
// TODO(allight): Prioritizing smaller gaps seems correct but it should be
// relatively straightforward to make the tie-breaker more intelligent than just
// earlier first. Making it prioritize smaller total area in the intervals as a
// secondary objective by merging smaller intervals when their is a tie in how
// much distance is between 2 intervals might provide better results.
IntervalSet MinimizeIntervals(IntervalSet interval_set, int64_t size);

// How many bits are needed to cover all values in the interval.
int64_t MinimumBitCount(const IntervalSet& a);

// How many bits are needed to cover all *signed* integer values in the range.
int64_t MinimumSignedBitCount(const IntervalSet& a);

// Arithmetic
IntervalSet Add(const IntervalSet& a, const IntervalSet& b);
IntervalSet Sub(const IntervalSet& a, const IntervalSet& b);
IntervalSet Neg(const IntervalSet& a);
IntervalSet UMul(const IntervalSet& a, const IntervalSet& b,
                 int64_t output_bitwidth);
IntervalSet UDiv(const IntervalSet& a, const IntervalSet& b);
IntervalSet UMod(const IntervalSet& a, const IntervalSet& b);
IntervalSet SMul(const IntervalSet& a, const IntervalSet& b,
                 int64_t output_bitwidth);
IntervalSet SDiv(const IntervalSet& a, const IntervalSet& b);
IntervalSet SMod(const IntervalSet& a, const IntervalSet& b);

// Shift
IntervalSet Shll(const IntervalSet& a, const IntervalSet& b);
IntervalSet Shrl(const IntervalSet& a, const IntervalSet& b);
IntervalSet Shra(const IntervalSet& a, const IntervalSet& b);

// Encode/decode
IntervalSet Decode(const IntervalSet& a, int64_t width);

// Bit ops.
IntervalSet Not(const IntervalSet& a);
IntervalSet And(const IntervalSet& a, const IntervalSet& b);
IntervalSet Or(const IntervalSet& a, const IntervalSet& b);
IntervalSet Xor(const IntervalSet& a, const IntervalSet& b);
IntervalSet AndReduce(const IntervalSet& a);
IntervalSet OrReduce(const IntervalSet& a);
IntervalSet XorReduce(const IntervalSet& a);
IntervalSet Concat(absl::Span<IntervalSet const> sets);
IntervalSet SignExtend(const IntervalSet& a, int64_t width);
IntervalSet ZeroExtend(const IntervalSet& a, int64_t width);
IntervalSet Truncate(const IntervalSet& a, int64_t width);
IntervalSet BitSlice(const IntervalSet& a, int64_t start, int64_t width);

// Cmp
IntervalSet Eq(const IntervalSet& a, const IntervalSet& b);
IntervalSet Ne(const IntervalSet& a, const IntervalSet& b);

IntervalSet ULt(const IntervalSet& a, const IntervalSet& b);
inline IntervalSet UGt(const IntervalSet& a, const IntervalSet& b) {
  return ULt(b, a);
}
inline IntervalSet ULe(const IntervalSet& a, const IntervalSet& b) {
  return Not(ULt(b, a));
}
inline IntervalSet UGe(const IntervalSet& a, const IntervalSet& b) {
  return Not(ULt(a, b));
}

IntervalSet SLt(const IntervalSet& a, const IntervalSet& b);
inline IntervalSet SGt(const IntervalSet& a, const IntervalSet& b) {
  return SLt(b, a);
}
inline IntervalSet SLe(const IntervalSet& a, const IntervalSet& b) {
  return Not(SLt(b, a));
}
inline IntervalSet SGe(const IntervalSet& a, const IntervalSet& b) {
  return Not(SLt(a, b));
}

// Misc
IntervalSet Gate(const IntervalSet& cond, const IntervalSet& val);
IntervalSet OneHot(const IntervalSet& val, LsbOrMsb lsb_or_msb,
                   int64_t max_intervals = 16);

}  // namespace xls::interval_ops

#endif  // XLS_IR_INTERVAL_OPS_H_
