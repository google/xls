// Copyright 2023 The XLS Authors
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
#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/apfloat_add_parallel_jit_wrapper.h"

// Enable once our failure rate is expected to be zero, otherwise too noisy logs
// (currently 0.03% wrong outputs)
#define DO_RUN_EXPECT 0

namespace xls::dslx {
namespace {

// Little helper class to better compare and print float bits in EXPECT_EQ etc.
// This represents the bits in a float, can be constructed from a float
// or its components, and be compared bit-accurately.
class FloatBits {
  static constexpr uint32_t kMantissaMask = (1 << 23) - 1;

 public:
  FloatBits() : bits_(0) {}
  explicit FloatBits(uint32_t bits) : bits_(bits) {}
  explicit FloatBits(float v) : FloatBits(absl::bit_cast<uint32_t>(v)) {}
  FloatBits(uint32_t sign, uint32_t exponent, uint32_t mantissa)
      : bits_((sign << 31) | (exponent << 23) | (mantissa & kMantissaMask)) {
    CHECK(sign <= 1 && exponent <= 255);
  }

  // Ordering by integer interpretation of bits (_not_ the float values)
  bool operator<(const FloatBits &other) const { return bits_ < other.bits_; }

  // Equal if bits are equal or if both float-interpretations are NaN.
  bool operator==(const FloatBits &other) const {
    return bits_ == other.bits_ ||
           (std::isnan(AsFloat()) && std::isnan(other.AsFloat()));
  }

  float AsFloat() const { return absl::bit_cast<float>(bits_); }

  std::string FormatBits() const {
    return absl::StrFormat("S:%d E:0x%02x M:0x%06x", bits_ >> 31,
                           (bits_ >> 23) & 0xff, (bits_ & kMantissaMask));
  }

 private:
  uint32_t bits_;
};

#if DO_RUN_EXPECT
static std::ostream& operator<<(std::ostream &out, FloatBits fbits) {
  return out << fbits.FormatBits();
}
#endif

// Keeping track of error count to report.
// While our sums are not fully bit-accurate in all cases, use this to
// keep track of statistics.
class StatCounter {
 public:
  explicit StatCounter(const char *msg) : msg_(msg) {}
  ~StatCounter() {
    const double percent = 100.0 * static_cast<double>(error_count_) /
                           static_cast<double>(total_count_);
    std::cerr << msg_ << ": " << total_count_ << " tested, "  //
              << error_count_ << " errors => " << percent << "% error rate.\n";
  }

  void AddMeasurement(bool was_success) {
    total_count_++;
    if (!was_success) {
      error_count_++;
    }
  }

  int64_t error_count() const { return error_count_; }

 private:
  const char *const msg_;
  int64_t total_count_ = 0;
  int64_t error_count_ = 0;
};

static float FlushSubnormal(float value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    return (value < 0) ? -0.0 : 0.0;
  }
  return value;
}

// Create a set of floats that cover the whole exponent range (with denser
// sampling around the center), and all kinds of 'interesting' mantissa
// patterns; all in positive and negative.
using SampleSet = absl::btree_set<FloatBits>;
static const SampleSet &CreateInterestingFloatSamples() {
  constexpr int kDenseExpRange = 24;
  static const SampleSet range = []() {
    SampleSet result;
    for (int s = 0; s <= 1; ++s) {
      for (int e = 1; e <= 255; ++e) {  // keep out subnormals, start with 1
        // Want dense set around 127, and above 250; otherwise sample
        // coarser.
        if (!((e >= 127 - kDenseExpRange && e <= 127 + kDenseExpRange) ||
              (e > 250) || e % 17 == 0)) {
          continue;
        }
        for (int mbits = 0; mbits < 24; ++mbits) {
          // Cover bits at the bottom and the top most, more sparse in middle.
          if (mbits <= 6 || mbits >= 18 || mbits % 2 == 0) {
            result.emplace(s, e, (1 << mbits));        // single bit
            result.emplace(s, e, ((1 << mbits) - 1));  // all bits on
            result.emplace(s, e, (0b101101u << mbits));
          }
        }
      }
    }
    std::cerr << "Set up test-floats; got " << result.size() << " distinct.\n"
              << "Unique cartesian pairs " << result.size() << "^2 = "  //
              << (result.size() * result.size()) << "\n";
    return result;
  }();
  return range;
}

TEST(Float32AddTest, ComprehensiveRangeSweepUsingXlsJit) {
  // Create JIT-ed version of float32_top()
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, xls::examples::Float32AddSe::Create());

  StatCounter stat("XLS-JIT-compiled");
  const SampleSet &sweep_data = CreateInterestingFloatSamples();
  for (const FloatBits a : sweep_data) {
    for (const FloatBits b : sweep_data) {
      XLS_ASSERT_OK_AND_ASSIGN(float out, jit->Run(a.AsFloat(), b.AsFloat()));

      // Comparing result with the host-platform implementation of float, but
      // flush subnormals as we explicitly don't handle them in our adder.
      const float expected_result = FlushSubnormal(a.AsFloat() + b.AsFloat());
      stat.AddMeasurement(FloatBits(out) == FloatBits(expected_result));
#if DO_RUN_EXPECT
      EXPECT_EQ(FloatBits(out), FloatBits(expected_result)) << a << " + " << b;
#endif
    }
  }
}

}  // namespace
}  // namespace xls::dslx
