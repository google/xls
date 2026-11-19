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

// Random-sampling test for the 64-bit FMA (fused multiply-add) module.
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/stdlib/float64_fma_jit_wrapper.h"
#include "xls/tests/testbench.h"
#include "xls/tests/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all available.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {
namespace {

using Float3x64 = std::tuple<double, double, double>;

uint64_t fp_sign(uint64_t f) { return f >> 63; }

uint64_t fp_exp(uint64_t f) { return (f >> 52) & 0x7ff; }

uint64_t fp_fraction(uint64_t f) { return f & 0xfffffffffffffull; }

constexpr std::array<uint64_t, 12> kSpecials = {
    0x0000000000000000ull,  // +0.0
    0x8000000000000000ull,  // -0.0
    0x0000000000000001ull,  // +min_subnormal
    0x800fffffffffffffull,  // -max_subnormal
    0x0010000000000000ull,  // +min_normal
    0x8010000000000000ull,  // -min_normal
    0x3ff0000000000000ull,  // +1.0
    0xbff0000000000000ull,  // -1.0
    0x7fefffffffffffffull,  // +max_normal
    0x7ff0000000000000ull,  // +inf
    0xfff0000000000000ull,  // -inf
    0x7ff8000000000000ull,  // +qNaN
};

Float3x64 SpecialIndexToInput(int64_t index) {
  constexpr uint64_t kNumSpecials = kSpecials.size();
  uint64_t a = kSpecials[index % kNumSpecials];
  uint64_t b = kSpecials[(index / kNumSpecials) % kNumSpecials];
  uint64_t c =
      kSpecials[(index / (kNumSpecials * kNumSpecials)) % kNumSpecials];
  return Float3x64(absl::bit_cast<double>(a), absl::bit_cast<double>(b),
                   absl::bit_cast<double>(c));
}

double ComputeExpected(fp::Float64Fma* jit_wrapper, Float3x64 input) {
  return fma(FlushSubnormal(std::get<0>(input)),
             FlushSubnormal(std::get<1>(input)),
             FlushSubnormal(std::get<2>(input)));
}

double ComputeActual(fp::Float64Fma* jit_wrapper, Float3x64 input) {
  double result =
      jit_wrapper
          ->Run(std::get<0>(input), std::get<1>(input), std::get<2>(input))
          .value();
  return result;
}

bool CompareResults(double expected, double actual) {
  // IEEE-754 leaves host NaN sign and payload bits implementation-defined
  // (e.g., x86 produces -qNaN for 0 * inf and propagates NaN input payloads),
  // whereas DSLX apfloat always returns canonical +qNaN.
  if (std::isnan(expected)) {
    constexpr uint64_t kCanonicalQNaN = 0x7ff8000000000000ull;
    return absl::bit_cast<uint64_t>(actual) == kCanonicalQNaN;
  }
  uint64_t expected_bits = absl::bit_cast<uint64_t>(FlushSubnormal(expected));
  uint64_t actual_bits = absl::bit_cast<uint64_t>(FlushSubnormal(actual));
  return expected_bits == actual_bits;
}

std::string PrintDouble(double a) {
  uint64_t a_int = absl::bit_cast<uint64_t>(a);
  return absl::StrFormat("0x%016x (0x%01x, 0x%03x, 0x%013x)", a_int,
                         fp_sign(a_int), fp_exp(a_int), fp_fraction(a_int));
}

std::string PrintInput(const Float3x64& input) {
  return absl::StrFormat(
      "  A       : %s\n"
      "  B       : %s\n"
      "  C       : %s",
      PrintDouble(std::get<0>(input)), PrintDouble(std::get<1>(input)),
      PrintDouble(std::get<2>(input)));
}

absl::Status RealMain(int64_t num_samples, int num_threads) {
  constexpr int64_t kNumSpecialTriples = static_cast<int64_t>(
      kSpecials.size() * kSpecials.size() * kSpecials.size());
  TestbenchBuilder<Float3x64, double, fp::Float64Fma> special_builder(
      ComputeExpected, ComputeActual,
      []() { return fp::Float64Fma::Create().value(); });
  special_builder.SetCompareResultsFn(CompareResults)
      .SetIndexToInputFn(SpecialIndexToInput)
      .SetPrintInputFn(PrintInput)
      .SetPrintResultFn(PrintDouble)
      .SetNumSamples(kNumSpecialTriples);
  if (num_threads != 0) {
    special_builder.SetNumThreads(num_threads);
  }
  XLS_RETURN_IF_ERROR(special_builder.Build().Run());

  // Rely on TestbenchBuilder's default random generator for uniform bit-pattern
  // coverage across all 2^64 double values (including zeros and subnormals).
  TestbenchBuilder<Float3x64, double, fp::Float64Fma> random_builder(
      ComputeExpected, ComputeActual,
      []() { return fp::Float64Fma::Create().value(); });
  random_builder.SetCompareResultsFn(CompareResults)
      .SetPrintInputFn(PrintInput)
      .SetPrintResultFn(PrintDouble)
      .SetNumSamples(num_samples);
  if (num_threads != 0) {
    random_builder.SetNumThreads(num_threads);
  }
  return random_builder.Build().Run();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::RealMain(absl::GetFlag(FLAGS_num_samples),
                                       absl::GetFlag(FLAGS_num_threads)));
}
