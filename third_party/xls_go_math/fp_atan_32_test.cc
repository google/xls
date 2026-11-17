// Copyright 2026 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Random-sampling test for the DSLX 32-bit floating-point atan.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/init_xls.h"
#include "xls/common/math_util.h"
#include "xls/tests/testbench_builder.h"
#include "third_party/xls_go_math/fp_atan_32_jit_wrapper.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");
ABSL_FLAG(int64_t, max_ulp_error, 2,
          "Maximum allowed error, in units in the last place, relative to the "
          "correctly rounded result.");

namespace xls {

// Alternates between floats with uniformly random bit patterns, which cover
// the whole exponent range (including special values), and floats uniformly
// distributed in [-4, 4], which exercise the different range reductions.
float IndexToInput(int64_t index) {
  thread_local absl::BitGen bitgen;
  if (index % 2 == 0) {
    return absl::bit_cast<float>(absl::Uniform<uint32_t>(bitgen));
  }
  return absl::Uniform(absl::IntervalClosed, bitgen, -4.0f, 4.0f);
}

// The DSLX implementation flushes input subnormals to 0, so we do that here
// as well. The double-precision result is rounded to the nearest float.
float ComputeExpected(fp::FpAtan32* jit_wrapper, float input) {
  return static_cast<float>(
      std::atan(static_cast<double>(FlushSubnormal(input))));
}

// Computes FP atan via DSLX & the JIT.
float ComputeActual(fp::FpAtan32* jit_wrapper, float input) {
  return jit_wrapper->Run(input).value();
}

// Compares expected vs. actual results, allowing for a small error and for
// the flushing of subnormal outputs.
bool CompareResults(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) {
    return std::isnan(a) && std::isnan(b);
  }
  if (ZeroOrSubnormal(a) && ZeroOrSubnormal(b)) {
    return std::signbit(a) == std::signbit(b);
  }
  if (std::signbit(a) != std::signbit(b)) {
    return false;
  }
  int64_t ulp_error = std::llabs(int64_t{absl::bit_cast<uint32_t>(a)} -
                                 int64_t{absl::bit_cast<uint32_t>(b)});
  return ulp_error <= absl::GetFlag(FLAGS_max_ulp_error);
}

std::string PrintFloat(const float& value) {
  return absl::StrFormat("%a (0x%08x)", value, absl::bit_cast<uint32_t>(value));
}

absl::Status RealMain(int64_t num_samples, int num_threads) {
  TestbenchBuilder<float, float, fp::FpAtan32> builder(
      ComputeExpected, ComputeActual,
      []() { return fp::FpAtan32::Create().value(); });
  builder.SetIndexToInputFn(IndexToInput)
      .SetCompareResultsFn(CompareResults)
      .SetPrintInputFn(PrintFloat)
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
  QCHECK_OK(xls::RealMain(absl::GetFlag(FLAGS_num_samples),
                          absl::GetFlag(FLAGS_num_threads)));
  return 0;
}
