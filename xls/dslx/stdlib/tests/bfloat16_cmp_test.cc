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

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <cmath>

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/tests/bfloat16_eq_2_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/bfloat16_gt_2_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/bfloat16_gte_2_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/bfloat16_lt_2_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/bfloat16_lte_2_jit_wrapper.h"
#include "xls/ir/bits.h"
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

// Helper to convert a u16 into a bfloat16 tuple value.
Value U16ToBfloat16Tuple(uint16_t val) {
  bool sign = (val >> 15) & 1;
  uint8_t exp = (val >> 7) & 0xff;
  uint8_t frac = val & 0x7f;
  return Value::Tuple({Value(UBits(sign, 1)), Value(UBits(exp, 8)),
                       Value(UBits(frac, 7))});
}

enum class CmpOp { kEq, kGt, kGte, kLt, kLte };

std::string CmpOpToString(CmpOp op) {
  switch (op) {
    case CmpOp::kEq:
      return "==";
    case CmpOp::kGt:
      return ">";
    case CmpOp::kGte:
      return ">=";
    case CmpOp::kLt:
      return "<";
    case CmpOp::kLte:
      return "<=";
  }
  LOG(FATAL) << "Invalid CmpOp";
}

using JitVariant =
    std::variant<std::unique_ptr<Bfloat16Eq2>, std::unique_ptr<Bfloat16Gt2>,
                 std::unique_ptr<Bfloat16Gte2>, std::unique_ptr<Bfloat16Lt2>,
                 std::unique_ptr<Bfloat16Lte2>>;

absl::StatusOr<JitVariant> CreateJit(CmpOp op) {
  switch (op) {
    case CmpOp::kEq:
      return Bfloat16Eq2::Create();
    case CmpOp::kGt:
      return Bfloat16Gt2::Create();
    case CmpOp::kGte:
      return Bfloat16Gte2::Create();
    case CmpOp::kLt:
      return Bfloat16Lt2::Create();
    case CmpOp::kLte:
      return Bfloat16Lte2::Create();
  }
  LOG(FATAL) << "Invalid CmpOp";
}

bool EvalOp(CmpOp op, float a, float b) {
  switch (op) {
    case CmpOp::kEq:
      return a == b;
    case CmpOp::kGt:
      return a > b;
    case CmpOp::kGte:
      return a >= b;
    case CmpOp::kLt:
      return a < b;
    case CmpOp::kLte:
      return a <= b;
  }
  LOG(FATAL) << "Invalid CmpOp";
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

class Bfloat16CmpTest : public ::testing::TestWithParam<CmpOp> {};

TEST_P(Bfloat16CmpTest, ExhaustiveVsInterestingValues) {
  CmpOp op = GetParam();
  XLS_ASSERT_OK_AND_ASSIGN(JitVariant jit_variant, CreateJit(op));

  auto do_test = [&](uint16_t u16_a, uint16_t u16_b) {
    // The DSLX bfloat16 operations flush subnormals to zero, so we must also do
    // so in our reference implementation.
    Eigen::bfloat16 bf_a =
        Eigen::numext::bit_cast<Eigen::bfloat16>(FlushSubnormal(u16_a));
    Eigen::bfloat16 bf_b =
        Eigen::numext::bit_cast<Eigen::bfloat16>(FlushSubnormal(u16_b));

    XLS_ASSERT_OK_AND_ASSIGN(
        Value xls_value,
        std::visit(
            [&](auto& jit) {
              return jit->Run(U16ToBfloat16Tuple(u16_a),
                              U16ToBfloat16Tuple(u16_b));
            },
            jit_variant));
    XLS_ASSERT_OK_AND_ASSIGN(bool xls_result, xls_value.bits().ToBool());

    bool eigen_result =
        EvalOp(op, static_cast<float>(bf_a), static_cast<float>(bf_b));
    EXPECT_EQ(xls_result, eigen_result)
        << u16_a << " " << CmpOpToString(op) << " " << u16_b << " ("
        << static_cast<float>(bf_a) << " vs " << static_cast<float>(bf_b)
        << ")";
  };

  const std::vector<uint16_t> interesting_values = GetInterestingBfloat16Values();
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

TEST_P(Bfloat16CmpTest, Random) {
  CmpOp op = GetParam();
  XLS_ASSERT_OK_AND_ASSIGN(JitVariant jit_variant, CreateJit(op));

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
        std::visit([&](auto& jit) {
          return jit->Run(U16ToBfloat16Tuple(u16_a), U16ToBfloat16Tuple(u16_b));
        }, jit_variant));
    XLS_ASSERT_OK_AND_ASSIGN(bool xls_result, xls_value.bits().ToBool());

    bool eigen_result =
        EvalOp(op, static_cast<float>(bf_a), static_cast<float>(bf_b));
    EXPECT_EQ(xls_result, eigen_result)
        << u16_a << " " << CmpOpToString(op) << " " << u16_b << " ("
        << static_cast<float>(bf_a) << " vs " << static_cast<float>(bf_b)
        << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(
    Bfloat16CmpTests, Bfloat16CmpTest,
    ::testing::Values(CmpOp::kEq, CmpOp::kGt, CmpOp::kGte, CmpOp::kLt,
                      CmpOp::kLte),
    [](const ::testing::TestParamInfo<CmpOp>& info) {
      switch (info.param) {
        case CmpOp::kEq:
          return "Eq";
        case CmpOp::kGt:
          return "Gt";
        case CmpOp::kGte:
          return "Gte";
        case CmpOp::kLt:
          return "Lt";
        case CmpOp::kLte:
          return "Lte";
      }
      LOG(FATAL) << "Unknown CmpOp";
    });

}  // namespace
}  // namespace xls::dslx
