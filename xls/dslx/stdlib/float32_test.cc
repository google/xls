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

#include <cstdint>

// Tests for the float32 specialization of the APFloat library.
#include <cmath>
#include <ios>
#include <limits>
#include <memory>

#include "gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/float32_from_int32_wrapper.h"
#include "xls/dslx/stdlib/float32_to_int32_wrapper.h"

ABSL_FLAG(bool, exhaustive, false,
          "Run exhaustively over the [32-bit] input space.");

namespace xls::dslx {
namespace {

// Tests the to_int32 float32 routine.
// This has been tested exhaustively, but in the interest of everyone's
// presubmits, only a random sample is tested here.
// 1024*1024 takes ~2 seconds to test; exhaustive takes ~1 minute (single-core).
TEST(Float32Test, ToInt32) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, Float32ToInt32::Create());
  absl::BitGen bitgen;
  bool exhaustive = absl::GetFlag(FLAGS_exhaustive);
  uint64_t num_iters =
      exhaustive ? std::numeric_limits<uint32_t>::max() : 1024 * 1024;
  for (uint64_t i = 0; i < num_iters; i++) {
    float input = absl::bit_cast<float>(static_cast<uint32_t>(i));
    if (!exhaustive) {
      input = absl::Uniform<float>(absl::IntervalClosedClosed, bitgen,
                                   std::numeric_limits<int32_t>::min(),
                                   std::numeric_limits<int32_t>::max());
    }

    // Frustratingly, float-to-int casts are undefined behavior if the source
    // float is outside the range of an int32_t, and since we don't have the
    // knobs to tell UBSan to ignore this, we just have to limit our test range.
    // We need >= and <= to handle precision loss when converting max\min into
    // float.
    if (input >= std::numeric_limits<int32_t>::max() ||
        input <= std::numeric_limits<int32_t>::min() ||
        std::isnan(input) || std::isinf(input)) {
      continue;
    }

    int32_t expected = static_cast<int32_t>(input);
    XLS_ASSERT_OK_AND_ASSIGN(int32_t actual, jit->Run(input));
    ASSERT_EQ(expected, actual)
        << std::hex << "i: " << i << ": "
        << "expected: " << expected << " vs. " << actual;
  }
}

// Tests the from_int32 float32 routine.
// This has been tested exhaustively, but in the interest of everyone's
// presubmits, only a random sample is tested here.
// 1024*1024 takes ~2 seconds to test; exhaustive takes
// ~3 minutes single-core. Significantly longer than the above - maybe due to
// the rounding step?
TEST(Float32Test, FromInt32) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, Float32FromInt32::Create());
  absl::BitGen bitgen;
  bool exhaustive = absl::GetFlag(FLAGS_exhaustive);
  uint64_t num_iters =
      exhaustive ? std::numeric_limits<uint32_t>::max() : 1024 * 1024;
  for (int i = 0; i < num_iters; i++) {
    int32_t input = i;
    if (!exhaustive) {
      input = absl::Uniform<int32_t>(absl::IntervalClosedClosed, bitgen,
                                     std::numeric_limits<int32_t>::min(),
                                     std::numeric_limits<int32_t>::max());
    }
    float expected = static_cast<float>(input);
    XLS_ASSERT_OK_AND_ASSIGN(float actual, jit->Run(input));
    ASSERT_EQ(expected, actual)
        << std::hex << "i: " << i << ": "
        << "expected: " << expected << " vs. " << actual;
  }
}

}  // namespace
}  // namespace xls::dslx
