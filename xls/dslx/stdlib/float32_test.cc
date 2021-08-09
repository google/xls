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

// Tests for the float32 specialization of the APFloat library.
#include <cmath>
#include <limits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/float32_to_int32_wrapper.h"

namespace xls::dslx {
namespace {

// Exhaustively tests the ToInt float32 routine.
TEST(Float32Test, ToInt) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, Float32ToInt32::Create());
  for (uint64_t i = 0; i < std::numeric_limits<uint32_t>::max(); i++) {
    float input = absl::bit_cast<float>(static_cast<uint32_t>(i));
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

}  // namespace
}  // namespace xls::dslx
