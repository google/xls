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

#include "xls/ir/bits.h"
#include "xls/ir/interval_set.h"
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
// TODO(allight): Currently this only searches for the longest common MSB
// prefix. More complex analysis is possible though of questionable usefulness
// given they can be extracted by other analyses.
TernaryVector ExtractTernaryVector(const IntervalSet& intervals,
                                   std::optional<Node*> source = std::nullopt);

struct KnownBits {
  Bits known_bits;
  Bits known_bit_values;
};

KnownBits ExtractKnownBits(const IntervalSet& intervals,
                           std::optional<Node*> source = std::nullopt);

}  // namespace xls::interval_ops

#endif  // XLS_IR_INTERVAL_OPS_H_
