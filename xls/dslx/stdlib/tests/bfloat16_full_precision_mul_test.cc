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

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"  // NOLINT(misc-include-cleaner) exports some symbols
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/tests/bfloat16_full_precision_mul_jit_wrapper.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

// Flushes subnormal bfloat16 values to zero, preserving the sign.
uint16_t FlushSubnormal(uint16_t u) {
  Eigen::bfloat16 bf = Eigen::numext::bit_cast<Eigen::bfloat16>(u);
  if (std::fpclassify(static_cast<float>(bf)) == FP_SUBNORMAL) {
    if (std::signbit(static_cast<float>(bf))) {
      return Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(-0.0f));
    }
    return Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(0.0f));
  }
  return u;
}

float FlushSubnormal(float f) {
  if (std::fpclassify(f) == FP_SUBNORMAL) {
    if (std::signbit(f)) {
      return -0.0f;
    }
    return 0.0f;
  }
  return f;
}

// Helper to convert a u16 into a bfloat16 tuple value.
Value U16ToBfloat16Tuple(uint16_t val) {
  bool sign = (val >> 15) & 1;
  uint8_t exp = (val >> 7) & 0xff;
  uint8_t frac = val & 0x7f;
  return Value::Tuple(
      {Value(UBits(sign, 1)), Value(UBits(exp, 8)), Value(UBits(frac, 7))});
}

absl::StatusOr<float> F24ToFloat(const Value& val) {
  CHECK(val.IsTuple());
  CHECK_EQ(val.elements().size(), 3);
  CHECK_EQ(val.elements()[0].bits().bit_count(), 1);
  CHECK_EQ(val.elements()[1].bits().bit_count(), 8);
  CHECK_EQ(val.elements()[2].bits().bit_count(), 15);
  Bits f32 =
      bits_ops::Concat({val.elements()[0].bits(), val.elements()[1].bits(),
                        val.elements()[2].bits(), UBits(0, 8)});
  return absl::bit_cast<float>(static_cast<uint32_t>(*f32.ToUint64()));
}

std::vector<uint16_t> GetInterestingBfloat16Values() {
  return {
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(0.0f)),
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(-0.0f)),
      Eigen::numext::bit_cast<uint16_t>(
          std::numeric_limits<Eigen::bfloat16>::min()),
      Eigen::numext::bit_cast<uint16_t>(
          -std::numeric_limits<Eigen::bfloat16>::min()),
      Eigen::numext::bit_cast<uint16_t>(
          std::numeric_limits<Eigen::bfloat16>::max()),
      Eigen::numext::bit_cast<uint16_t>(
          std::numeric_limits<Eigen::bfloat16>::lowest()),
      Eigen::numext::bit_cast<uint16_t>(
          std::numeric_limits<Eigen::bfloat16>::infinity()),
      Eigen::numext::bit_cast<uint16_t>(
          -std::numeric_limits<Eigen::bfloat16>::infinity()),
      Eigen::numext::bit_cast<uint16_t>(
          std::numeric_limits<Eigen::bfloat16>::quiet_NaN()),
      Eigen::numext::bit_cast<uint16_t>(
          -std::numeric_limits<Eigen::bfloat16>::quiet_NaN()),
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(1.0f)),
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(-1.0f)),
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(2.0f)),
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(-2.0f)),
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(0.5f)),
      Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(-0.5f)),
  };
}

TEST(Bfloat16FullPrecisionMulTest, ExhaustiveVsInterestingValues) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bfloat16FullPrecisionMul> jit,
                           Bfloat16FullPrecisionMul::Create());

  auto do_test = [&](uint16_t u16_a, uint16_t u16_b) {
    // The DSLX bfloat16 operations flush subnormals to zero, so we must also do
    // so in our reference implementation.
    Eigen::bfloat16 bf_a =
        Eigen::numext::bit_cast<Eigen::bfloat16>(FlushSubnormal(u16_a));
    Eigen::bfloat16 bf_b =
        Eigen::numext::bit_cast<Eigen::bfloat16>(FlushSubnormal(u16_b));

    XLS_ASSERT_OK_AND_ASSIGN(
        Value xls_value,
        jit->Run(U16ToBfloat16Tuple(u16_a), U16ToBfloat16Tuple(u16_b)));
    XLS_ASSERT_OK_AND_ASSIGN(float xls_result, F24ToFloat(xls_value));

    float eigen_result =
        FlushSubnormal(static_cast<float>(bf_a) * static_cast<float>(bf_b));
    if (std::isnan(eigen_result) || std::isnan(xls_result)) {
      EXPECT_EQ(std::isnan(xls_result), std::isnan(eigen_result))
          << u16_a << " * " << u16_b << " (" << static_cast<float>(bf_a)
          << " vs " << static_cast<float>(bf_b) << ")";
    } else {
      EXPECT_EQ(xls_result, eigen_result)
          << u16_a << " * " << u16_b << " (" << static_cast<float>(bf_a)
          << " vs " << static_cast<float>(bf_b) << ")";
    }
  };

  const std::vector<uint16_t> interesting_values =
      GetInterestingBfloat16Values();
  for (uint32_t i = 0; i <= std::numeric_limits<uint16_t>::max(); ++i) {
    uint16_t u16_a = i;
    // Test against interesting values.
    for (uint16_t u16_b : interesting_values) {
      do_test(u16_a, u16_b);
      do_test(u16_b, u16_a);
    }

    // Test against self.
    do_test(u16_a, u16_a);
  }
}

TEST(Bfloat16FullPrecisionMulTest, Random) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bfloat16FullPrecisionMul> jit,
                           Bfloat16FullPrecisionMul::Create());

  absl::BitGen bitgen;
  constexpr int kNumSamples = 1024 * 1024;
  for (int i = 0; i < kNumSamples; ++i) {
    uint16_t u16_a = absl::Uniform<uint16_t>(bitgen);
    uint16_t u16_b = absl::Uniform<uint16_t>(bitgen);
    // The DSLX bfloat16 operations flush subnormals to zero, so we must also do
    // so in our reference implementation.
    Eigen::bfloat16 bf_a =
        Eigen::numext::bit_cast<Eigen::bfloat16>(FlushSubnormal(u16_a));
    Eigen::bfloat16 bf_b =
        Eigen::numext::bit_cast<Eigen::bfloat16>(FlushSubnormal(u16_b));

    XLS_ASSERT_OK_AND_ASSIGN(
        Value xls_value,
        jit->Run(U16ToBfloat16Tuple(u16_a), U16ToBfloat16Tuple(u16_b)));
    XLS_ASSERT_OK_AND_ASSIGN(float xls_result, F24ToFloat(xls_value));

    float eigen_result =
        FlushSubnormal(static_cast<float>(bf_a) * static_cast<float>(bf_b));
    if (std::isnan(eigen_result) || std::isnan(xls_result)) {
      EXPECT_EQ(std::isnan(xls_result), std::isnan(eigen_result))
          << u16_a << " * " << u16_b << " (" << static_cast<float>(bf_a)
          << " vs " << static_cast<float>(bf_b) << ")";
    } else {
      EXPECT_EQ(xls_result, eigen_result)
          << u16_a << " * " << u16_b << " (" << static_cast<float>(bf_a)
          << " vs " << static_cast<float>(bf_b) << ")";
    }
  }
}

}  // namespace
}  // namespace xls::dslx
