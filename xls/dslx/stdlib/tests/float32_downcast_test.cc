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

#include <fenv.h>  // NOLINT - Allow fenv header.

#include <cmath>
#include <cstdint>
#include <ios>
#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/base/casts.h"
#include "xls/dslx/stdlib/tests/float32_downcast_jit_wrapper.h"

namespace xls {
namespace {

static_assert(sizeof(double) == 8, "8 byte double required");
static_assert(sizeof(float) == 4, "4 byte float required");

class F64ToF32 {
 public:
  F64ToF32() { jit_ = std::move(fp::F64ToF32::Create()).value(); }
  void F64ToF32Test(uint64_t v) {
    if (fegetround() != FE_TONEAREST) {
      GTEST_SKIP() << "Unexpected rounding mode";
    }
    double d = absl::bit_cast<double>(v);
    float f = (float)d;
    float j = jit_->Run(d).value();
    if (std::isnan(f)) {
      ASSERT_THAT(j, testing::IsNan());
    } else {
      ASSERT_EQ(f, j) << std::boolalpha << "is subnormal: "
                      << (std::fpclassify(f) == FP_SUBNORMAL)
                      << " inp: " << std::hex << "0x" << v << " "
                      << std::hexfloat << d << " f=" << f << " j=" << j;
    }
  }

 private:
  std::unique_ptr<fp::F64ToF32> jit_;
};
FUZZ_TEST_F(F64ToF32, F64ToF32Test).WithDomains(fuzztest::Arbitrary<int64_t>());

}  // namespace
}  // namespace xls
