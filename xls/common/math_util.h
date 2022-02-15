// Copyright 2022 The XLS Authors
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

#ifndef XLS_COMMON_MATH_UTIL_H_
#define XLS_COMMON_MATH_UTIL_H_

#include <limits>

#include "absl/types/span.h"
#include "xls/common/logging/logging.h"

namespace xls {

// ---- CeilOrFloorOfRatio ----
// This is a branching-free, cast-to-double-free implementation.
//
// Casting to double is in general incorrect because of loss of precision
// when casting an int64_t into a double.
//
// There's a bunch of 'recipes' to compute a integer ceil (or floor) on the web,
// and most of them are incorrect.
template <typename IntegralType, bool ceil>
constexpr IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  static_assert(std::numeric_limits<IntegralType>::is_integer,
                "CeilOfRatio is only defined for integral types");
  XLS_DCHECK_NE(0, denominator) << "Division by zero is not supported.";
  XLS_DCHECK(!std::numeric_limits<IntegralType>::is_signed ||
             numerator != std::numeric_limits<IntegralType>::min() ||
             denominator != -1)
      << "Dividing " << numerator << " by -1 is not supported: it would SIGFPE";

  const IntegralType rounded_toward_zero = numerator / denominator;
  const bool needs_round = (numerator % denominator) != 0;
  // It is important to use >= here, even for the denominator, to ensure that
  // this value is a compile-time constant for unsigned types.
  const bool same_sign = (numerator >= 0) == (denominator >= 0);

  if (ceil) {  // Compile-time condition: not an actual branching
    return rounded_toward_zero +
           static_cast<IntegralType>(same_sign && needs_round);
  } else {
    return rounded_toward_zero -
           static_cast<IntegralType>(!same_sign && needs_round);
  }
}

// ----------------------------------------------------------------------
// CeilOfRatio<IntegralType>
// FloorOfRatio<IntegralType>
//   Returns the ceil (resp. floor) of the ratio of two integers.
//
//  * IntegralType: any integral type, whether signed or not.
//  * numerator: any integer: positive, negative, or zero.
//  * denominator: a non-zero integer, positive or negative.
//
// This implementation is correct, meaning there is never any precision loss,
// and there is never an overflow. However, if the type is signed, having
// numerator == MathLimits<IntegralType>::kMin and denominator == -1 is not a
// valid input, because kMin has a greater absolute value than kMax.
//
// Input validity is DCHECKed. When not in debug mode, invalid inputs raise
// SIGFPE.
//
// This method has been designed and tested so that it should always be
// preferred to alternatives. Indeed, there exist popular recipes to compute
// the result, such as casting to double, but they are in general incorrect.
// In cases where an alternative technique is correct, performance measurement
// showed the provided implementation is faster.
// ----------------------------------------------------------------------
template <typename IntegralType>
static constexpr IntegralType CeilOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  return CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
}

template <typename IntegralType>
static constexpr IntegralType FloorOfRatio(IntegralType numerator,
                                           IntegralType denominator) {
  return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
}

template <typename T>
T RoundUpToNearest(T value, T divisor) {
  return CeilOfRatio(value, divisor) * divisor;
}

template <typename T>
T RoundDownToNearest(T value, T divisor) {
  return value / divisor * divisor;
}

template <typename T>
inline bool IsPowerOfTwo(T x) {
  static_assert(!std::numeric_limits<T>::is_signed, "unsigned types only");
  return x != 0 && (x & (x - 1)) == 0;
}

// Returns ceil(log2(value)). Returns zero for the value zero.
int64_t CeilOfLog2(uint64_t value);

// Returns floor(log2(value)). Returns zero for the value zero.
int64_t FloorOfLog2(uint64_t value);

// Returns true if the given floating-point value is 0 or subnormal.
template <typename T>
bool ZeroOrSubnormal(T value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// Returns +/-0 if the given floating-point value is subnormal (sign is
// preserved) or the original value otherwise.
template <typename T>
T FlushSubnormal(T value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    if (value < 0) {
      return -0.0;
    } else {
      return 0.0;
    }
  }

  return value;
}

// Calls the given function on every possible mixed-radix number of a given
// length (the length of the radix vector). Each element `i` of a
// "number vector" is thought of as a digit with radix equal to `radix[i]`.
// If the given function returns `true`, the iteration ends early and we return
// `true`. Otherwise, `false` is returned.
//
// For example, if the `radix` is `[6, 4, 8, 2]` and the most recent time the
// callback was called was on `[5, 3, 0, 0]` then the next call will be
// with `[0, 0, 1, 0]` (note that the convention is little-endian).
bool MixedRadixIterate(absl::Span<const int64_t> radix,
                       std::function<bool(const std::vector<int64_t>&)> f);

}  // namespace xls

#endif  // XLS_COMMON_MATH_UTIL_H_
