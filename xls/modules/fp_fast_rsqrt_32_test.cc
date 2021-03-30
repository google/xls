// Copyright 2021 The XLS Authors
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

// Random-sampling test for DSLX 32-bit floating-point fast, approximate
// inverse sqrt.
#include <cmath>
#include <limits>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value_helpers.h"
#include "xls/ir/value_view_helpers.h"
#include "xls/modules/fp_fast_rsqrt_32_jit_wrapper.h"
#include "xls/tools/testbench.h"

ABSL_FLAG(bool, use_opt_ir, true, "Use optimized IR.");
ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

constexpr const char kOptIrPath[] = "xls/modules/fp_fast_rsqrt_32.opt.ir";
constexpr const char kIrPath[] = "xls/modules/fp_fast_rsqrt_32.ir";

// Generates a float with reasonably unformly random bit patterns.
float IndexToInput(uint64_t index) {
  thread_local absl::BitGen bitgen;
  uint32_t a = absl::Uniform(bitgen, 0u, std::numeric_limits<uint32_t>::max());
  return absl::bit_cast<float>(a);
}

// The DSLX implementation uses the "round to nearest (half to even)"
// rounding mode, which is the default on most systems, hence we don't need
// to call fesetround().
// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
float ComputeExpected(float input) {
  float x = FlushSubnormal(input);
  return 1.0 / sqrtf(x);
}

// Computes FP sqrt via DSLX & the JIT.
float ComputeActual(FpFastRsqrt32* jit_wrapper, float input) {
  return jit_wrapper->Run(input).value();
}

// Compares expected vs. actual results, taking into account two special cases.
bool CompareResults(float a, float b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  // We only check that results are approximately equal. Percent error
  // is used rather than simple difference because the input may vary
  // by many orders of magnitude.
  float percent_error = (a - b) / a;
  return a == b || percent_error < 0.01 || (std::isnan(a) && std::isnan(b)) ||
         (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
}

void LogMismatch(uint64_t index, float input, float expected, float actual) {
  XLS_LOG(ERROR) << absl::StrFormat(
      "Value mismatch at index %d, input %f:\n"
      "  Input:  0x%x\n"
      "  Expected (approx.): 0x%x\n"
      "  Actual  : 0x%x",
      index, input, absl::bit_cast<uint32_t>(input),
      absl::bit_cast<uint32_t>(expected), absl::bit_cast<uint32_t>(actual));
}

absl::Status RealMain(bool use_opt_ir, uint64_t num_samples, int num_threads) {
  Testbench<FpFastRsqrt32, float, float> testbench(
      0, num_samples,
      /*max_failures=*/1, IndexToInput, ComputeExpected, ComputeActual,
      CompareResults, LogMismatch);
  if (num_threads != 0) {
    XLS_RETURN_IF_ERROR(testbench.SetNumThreads(num_threads));
  }
  return testbench.Run();
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  XLS_QCHECK_OK(xls::RealMain(absl::GetFlag(FLAGS_use_opt_ir),
                              absl::GetFlag(FLAGS_num_samples),
                              absl::GetFlag(FLAGS_num_threads)));
  return 0;
}
