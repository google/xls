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

// Random-sampling test for the 32-bit FMA (fused multiply-add) module.
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
#include "xls/common/status/status_macros.h"
#include "xls/dslx/stdlib/float32_fma_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/float32_test_utils.h"
#include "xls/tests/testbench.h"
#include "xls/tests/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all available.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {
namespace {

using Float3x32 = std::tuple<float, float, float>;

uint32_t fp_sign(uint32_t f) { return f >> 31; }

uint32_t fp_exp(uint32_t f) { return (f >> 23) & 0xff; }

uint32_t fp_fraction(uint32_t f) { return f & 0x7fffff; }

constexpr std::array<uint32_t, 12> kSpecials = {
    0x00000000u,  // +0.0
    0x80000000u,  // -0.0
    0x00000001u,  // +min_subnormal
    0x807fffffu,  // -max_subnormal
    0x00800000u,  // +min_normal
    0x80800000u,  // -min_normal
    0x3f800000u,  // +1.0
    0xbf800000u,  // -1.0
    0x7f7fffffu,  // +max_normal
    0x7f800000u,  // +inf
    0xff800000u,  // -inf
    0x7fc00000u,  // +qNaN
};

Float3x32 SpecialIndexToInput(int64_t index) {
  constexpr uint64_t kNumSpecials = kSpecials.size();
  uint32_t a = kSpecials[index % kNumSpecials];
  uint32_t b = kSpecials[(index / kNumSpecials) % kNumSpecials];
  uint32_t c =
      kSpecials[(index / (kNumSpecials * kNumSpecials)) % kNumSpecials];
  return Float3x32(absl::bit_cast<float>(a), absl::bit_cast<float>(b),
                   absl::bit_cast<float>(c));
}

float ComputeExpected(fp::Float32Fma* jit_wrapper, Float3x32 input) {
  return fmaf(FlushSubnormal(std::get<0>(input)),
              FlushSubnormal(std::get<1>(input)),
              FlushSubnormal(std::get<2>(input)));
}

float ComputeActual(fp::Float32Fma* jit_wrapper, Float3x32 input) {
  float result =
      jit_wrapper
          ->Run(std::get<0>(input), std::get<1>(input), std::get<2>(input))
          .value();
  return result;
}

std::string PrintFloat(float a) {
  uint32_t a_int = absl::bit_cast<uint32_t>(a);
  return absl::StrFormat("0x%08x (0x%01x, 0x%02x, 0x%06x)", a_int,
                         fp_sign(a_int), fp_exp(a_int), fp_fraction(a_int));
}

std::string PrintInput(const Float3x32& input) {
  return absl::StrFormat(
      "  A       : %s\n"
      "  B       : %s\n"
      "  C       : %s",
      PrintFloat(std::get<0>(input)), PrintFloat(std::get<1>(input)),
      PrintFloat(std::get<2>(input)));
}

absl::Status RealMain(int64_t num_samples, int num_threads) {
  constexpr int64_t kNumSpecialTriples = static_cast<int64_t>(
      kSpecials.size() * kSpecials.size() * kSpecials.size());
  TestbenchBuilder<Float3x32, float, fp::Float32Fma> special_builder(
      ComputeExpected, ComputeActual,
      []() { return fp::Float32Fma::Create().value(); });
  special_builder.SetCompareResultsFn(CompareResults)
      .SetIndexToInputFn(SpecialIndexToInput)
      .SetPrintInputFn(PrintInput)
      .SetPrintResultFn(PrintFloat)
      .SetNumSamples(kNumSpecialTriples);
  if (num_threads != 0) {
    special_builder.SetNumThreads(num_threads);
  }
  XLS_RETURN_IF_ERROR(special_builder.Build().Run());

  // Rely on TestbenchBuilder's default random generator for uniform bit-pattern
  // coverage across all 2^32 float values (including zeros and subnormals).
  TestbenchBuilder<Float3x32, float, fp::Float32Fma> random_builder(
      ComputeExpected, ComputeActual,
      []() { return fp::Float32Fma::Create().value(); });
  random_builder.SetCompareResultsFn(CompareResults)
      .SetPrintInputFn(PrintInput)
      .SetPrintResultFn(PrintFloat)
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
