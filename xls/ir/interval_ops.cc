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
#include <optional>

#include "absl/log/check.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"

namespace xls::interval_ops {

TernaryVector ExtractTernaryVector(const IntervalSet& intervals,
                                   std::optional<Node*> source) {
  KnownBits bits = ExtractKnownBits(intervals, source);
  return ternary_ops::FromKnownBits(bits.known_bits, bits.known_bit_values);
}

KnownBits ExtractKnownBits(const IntervalSet& intervals,
                           std::optional<Node*> source) {
  CHECK(intervals.IsNormalized())
      << (source.has_value() ? source.value()->ToString() : "");
  CHECK(!intervals.Intervals().empty())
      << (source.has_value() ? source.value()->ToString() : "");
  Bits lcp = bits_ops::LongestCommonPrefixMSB(
      {intervals.Intervals().front().LowerBound(),
       intervals.Intervals().back().UpperBound()});
  int64_t size = intervals.BitCount();
  Bits remainder = Bits(size - lcp.bit_count());
  return KnownBits{
      .known_bits =
          bits_ops::Concat({Bits::AllOnes(lcp.bit_count()), remainder}),
      .known_bit_values = bits_ops::Concat({lcp, remainder}),
  };
}
}  // namespace xls::interval_ops
