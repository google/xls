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
#ifndef XLS_DSLX_EXHAUSTIVENESS_INTERP_VALUE_INTERVAL_H_
#define XLS_DSLX_EXHAUSTIVENESS_INTERP_VALUE_INTERVAL_H_

#include <cstdint>
#include <string>

#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// Represents a single contiguous interval of values; e.g. for a run of bits
// values with the same bits type.
//
// This is an inclusive interval, i.e. `[min, max]`, because there's not a lot
// of value in representing a "volumeless" interval via an exclusive limit, we
// can discard those early.
//
// Generally we can think of this as "the concrete representation of a range
// expression" a la `0..3` or similar.
class InterpValueInterval {
 public:
  static InterpValueInterval MakeFull(bool is_signed, int64_t bit_count);

  InterpValueInterval(InterpValue min, InterpValue max);

  // Returns true if this interval contains the given value.
  bool Contains(InterpValue value) const;

  // Returns true if this interval intersects with the given interval.
  bool Intersects(const InterpValueInterval& other) const {
    return Contains(other.min_) || Contains(other.max_) ||
           other.Contains(min_) || other.Contains(max_);
  }

  bool Covers(const InterpValueInterval& other) const {
    return Contains(other.min_) && Contains(other.max_);
  }

  bool operator==(const InterpValueInterval& other) const {
    return min_ == other.min_ && max_ == other.max_;
  }

  bool operator<(const InterpValueInterval& other) const {
    return min_ < other.min_ || (min_ == other.min_ && max_ < other.max_);
  }

  const InterpValue& min() const { return min_; }
  const InterpValue& max() const { return max_; }

  std::string ToString(bool show_types) const;

 private:
  bool IsSigned() const;
  int64_t GetBitCount() const;
  InterpValue min_;
  InterpValue max_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_EXHAUSTIVENESS_INTERP_VALUE_INTERVAL_H_
