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

// Compares expected vs. actual results, taking into account two special cases.
inline bool CompareResultsWith1PercentMargin(float a, float b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  // We only check that results are approximately equal. Percent error
  // is used rather than simple difference because the input may vary
  // by many orders of magnitude.
  if ((std::isnan(a) && std::isnan(b)) ||
      (ZeroOrSubnormal(a) && ZeroOrSubnormal(b))) {
    return true;
  }
  // Avoid divide by zero. Necessarily from the condition above b is not
  // zero/subnormal in this case.
  if (ZeroOrSubnormal(a)) {
    return false;
  }
  float percent_error = (a - b) / a;
  return a == b || percent_error < 0.01;
}

}  // namespace xls

#endif  // XLS_DSLX_STDLIB_TESTS_FLOAT32_TEST_UTILS_H_
