// Copyright 2021 The XLS Authors
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

#ifndef XLS_DSLX_STDLIB_TESTS_FLOAT32_TEST_UTILS_H_
#define XLS_DSLX_STDLIB_TESTS_FLOAT32_TEST_UTILS_H_

#include <cmath>
#include <cstdint>

#include "absl/base/casts.h"

namespace xls {

inline float FlushSubnormal(float value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    return std::copysign(/*magnitude=*/0.0f, /*sign=*/value);
  }

  return value;
}

inline bool ZeroOrSubnormal(float value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// Compares expected vs. actual results, taking into account two special cases.
inline bool CompareResults(float a, float b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just flush subnormals here as well.
  // Also, compare the exact bits, otherwise (0.0f == -0.0f) == true.
  // NaN is implementation defined what the mantissa bits mean, so we can
  // not assume that host-bits would be the same as dslx bits.
  uint32_t a_bits = absl::bit_cast<uint32_t>(FlushSubnormal(a));
  uint32_t b_bits = absl::bit_cast<uint32_t>(FlushSubnormal(b));
  return a_bits == b_bits || (std::isnan(a) && std::isnan(b));
}

// Compares expected vs. actual results within a 1% relative error margin,
// flushing subnormals to zero to match DSLX's FTZ semantics and allowing
// implementation-defined host NaN encodings.
inline bool CompareResultsWith1PercentMargin(float expected, float actual) {
  if (std::isnan(expected) || std::isnan(actual)) {
    return std::isnan(expected) && std::isnan(actual);
  }
  if (ZeroOrSubnormal(expected) && ZeroOrSubnormal(actual)) {
    return true;
  }
  if (expected == actual) {
    return true;
  }
  // Avoid divide by zero (and inf/inf -> NaN). From the checks above,
  // `expected` and `actual` are unequal, and if `expected` is zero/subnormal
  // then `actual` is not.
  if (ZeroOrSubnormal(expected) || std::isinf(expected) || std::isinf(actual)) {
    return false;
  }
  // Compute relative error in double so `expected - actual` cannot overflow
  // near FLT_MAX and small values near FLT_MIN do not lose precision.
  double expected_d = static_cast<double>(expected);
  double actual_d = static_cast<double>(actual);
  double relative_error =
      std::abs(expected_d - actual_d) / std::abs(expected_d);
  return relative_error < 0.01;
}

}  // namespace xls

#endif  // XLS_DSLX_STDLIB_TESTS_FLOAT32_TEST_UTILS_H_
