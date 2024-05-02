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
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/math_util.h"
#include "xls/dslx/stdlib/float64_fma_jit_wrapper.h"
#include "xls/tools/testbench.h"
#include "xls/tools/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all available.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

using Float3x64 = std::tuple<double, double, double>;

static uint64_t fp_sign(uint64_t f) { return f >> 63; }

static uint64_t fp_exp(uint64_t f) { return (f >> 52) & 0x7ff; }

static uint64_t fp_fraction(uint64_t f) { return f & 0xfffffffffffffull; }

static Float3x64 IndexToInput(uint64_t index) {
  // Skip subnormal inputs - we currently don't handle them.
  thread_local absl::BitGen bitgen;
  auto generate_non_subnormal = []() {
    uint64_t value = 0;
    while (fp_exp(value) == 0) {
      value = absl::Uniform(bitgen, 0u, std::numeric_limits<uint64_t>::max());
    }
    return value;
  };

  uint64_t a = generate_non_subnormal();
  uint64_t b = generate_non_subnormal();
  uint64_t c = generate_non_subnormal();
  return Float3x64(absl::bit_cast<double>(a), absl::bit_cast<double>(b),
                   absl::bit_cast<double>(c));
}

static double ComputeExpected(fp::Float64Fma* jit_wrapper, Float3x64 input) {
  return fma(std::get<0>(input), std::get<1>(input), std::get<2>(input));
}

static double ComputeActual(fp::Float64Fma* jit_wrapper, Float3x64 input) {
  double result =
      jit_wrapper
          ->Run(std::get<0>(input), std::get<1>(input), std::get<2>(input))
          .value();
  return result;
}

static bool CompareResults(double a, double b) {
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
}

static std::string PrintDouble(double a) {
  uint64_t a_int = absl::bit_cast<uint64_t>(a);
  return absl::StrFormat("0x%016x (0x%01x, 0x%03x, 0x%013x)", a_int,
                         fp_sign(a_int), fp_exp(a_int), fp_fraction(a_int));
}

static std::string PrintInput(const Float3x64& input) {
  return absl::StrFormat(
      "  A       : %s\n"
      "  B       : %s\n"
      "  C       : %s",
      PrintDouble(std::get<0>(input)), PrintDouble(std::get<1>(input)),
      PrintDouble(std::get<2>(input)));
}

static absl::Status RealMain(int64_t num_samples, int num_threads) {
  TestbenchBuilder<Float3x64, double, fp::Float64Fma> builder(
      ComputeExpected, ComputeActual,
      []() { return fp::Float64Fma::Create().value(); });
  builder.SetCompareResultsFn(CompareResults)
      .SetIndexToInputFn(IndexToInput)
      .SetPrintInputFn(PrintInput)
      .SetPrintResultFn(PrintDouble)
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
