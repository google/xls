// Copyright 2025 The XLS Authors
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

#include "xls/ir/interval_set_test_utils.h"

#include <cstdint>
#include <utility>

#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"

namespace xls {

IntervalSet FromRanges(absl::Span<std::pair<int64_t, int64_t> const> ranges,
                       int64_t bits) {
  IntervalSet res(bits);
  for (const auto& [l, h] : ranges) {
    res.AddInterval(Interval::Closed(UBits(l, bits), UBits(h, bits)));
  }
  res.Normalize();
  return res;
}

}  // namespace xls
