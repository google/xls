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
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/math_util.h"
#include "xls/dslx/stdlib/float32_fma_jit_wrapper.h"
#include "xls/dslx/stdlib/float32_test_utils.h"
#include "xls/tools/testbench.h"
#include "xls/tools/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all available.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

using Float3x32 = std::tuple<float, float, float>;

static uint32_t fp_sign(uint32_t f) { return f >> 31; }

static uint32_t fp_exp(uint32_t f) { return (f >> 23) & 0xff; }

static uint32_t fp_fraction(uint32_t f) { return f & 0x7fffff; }

static Float3x32 IndexToInput(uint64_t index) {
  // Skip subnormal inputs - we currently don't handle them.
  thread_local absl::BitGen bitgen;
  auto generate_non_subnormal = []() {
    uint32_t value = 0;
    while (fp_exp(value) == 0) {
      value = absl::Uniform(bitgen, 0u, std::numeric_limits<uint32_t>::max());
    }
    return value;
  };

  uint32_t a = generate_non_subnormal();
  uint32_t b = generate_non_subnormal();
  uint32_t c = generate_non_subnormal();
  return Float3x32(absl::bit_cast<float>(a), absl::bit_cast<float>(b),
                   absl::bit_cast<float>(c));
}

static float ComputeExpected(fp::Float32Fma* jit_wrapper, Float3x32 input) {
  return fmaf(std::get<0>(input), std::get<1>(input), std::get<2>(input));
}

static float ComputeActual(fp::Float32Fma* jit_wrapper, Float3x32 input) {
  float result =
      jit_wrapper
          ->Run(std::get<0>(input), std::get<1>(input), std::get<2>(input))
          .value();
  return result;
}

static std::string PrintFloat(float a) {
  uint32_t a_int = absl::bit_cast<uint32_t>(a);
  return absl::StrFormat("0x%016x (0x%01x, 0x%03x, 0x%013x)", a_int,
                         fp_sign(a_int), fp_exp(a_int), fp_fraction(a_int));
}

static std::string PrintInput(const Float3x32& input) {
  return absl::StrFormat(
      "  A       : %s\n"
      "  B       : %s\n"
      "  C       : %s",
      PrintFloat(std::get<0>(input)), PrintFloat(std::get<1>(input)),
      PrintFloat(std::get<2>(input)));
}

static absl::Status RealMain(int64_t num_samples, int num_threads) {
  TestbenchBuilder<Float3x32, float, fp::Float32Fma> builder(
      ComputeExpected, ComputeActual,
      []() { return fp::Float32Fma::Create().value(); });
  builder.SetCompareResultsFn(CompareResults)
      .SetIndexToInputFn(IndexToInput)
      .SetPrintInputFn(PrintInput)
      .SetPrintResultFn(PrintFloat)
      .SetNumSamples(num_samples);
  if (num_threads != 0) {
    builder.SetNumThreads(num_threads);
  }
  return builder.Build().Run();
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::RealMain(absl::GetFlag(FLAGS_num_samples),
                                       absl::GetFlag(FLAGS_num_threads)));
}
