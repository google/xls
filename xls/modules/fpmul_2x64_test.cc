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

// Random-sampling test for the DSLX 2x64 floating-point multiplier.
#include <cmath>
#include <tuple>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value_helpers.h"
#include "xls/ir/value_view_helpers.h"
#include "xls/modules/fpmul_2x64_jit_wrapper.h"
#include "xls/tools/testbench.h"

ABSL_FLAG(bool, use_opt_ir, true, "Use optimized IR.");
ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64, num_samples, 1024 * 1024, "Number of random samples to test.");

namespace xls {

using Float2x64 = std::tuple<double, double>;

double FlushSubnormals(double value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    return 0;
  }

  return value;
}

bool ZeroOrSubnormal(double value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// Generates two doubles with reasonably unformly random bit patterns.
Float2x64 IndexToInput(uint64 index) {
  thread_local absl::BitGen bitgen;
  uint64 a = absl::Uniform(bitgen, 0u, std::numeric_limits<uint64_t>::max());
  uint64 b = absl::Uniform(bitgen, 0u, std::numeric_limits<uint64_t>::max());
  return Float2x64(absl::bit_cast<double>(a), absl::bit_cast<double>(b));
}

// The DSLX implementation uses the "round to nearest (half to even)"
// rounding mode, which is the default on most systems, hence we don't need
// to call fesetround().
// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
double ComputeExpected(Float2x64 input) {
  double x = FlushSubnormals(std::get<0>(input));
  double y = FlushSubnormals(std::get<1>(input));
  return x * y;
}

// Computes FP addition via DSLX & the JIT.
double ComputeActual(Fpmul2x64* jit_wrapper, Float2x64 input) {
  return jit_wrapper->Run(std::get<0>(input), std::get<1>(input)).value();
}

// Compares expected vs. actual results, taking into account two special cases.
bool CompareResults(double a, double b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
}

void LogMismatch(uint64 index, Float2x64 input, double actual,
                 double expected) {
  XLS_LOG(ERROR) << absl::StrFormat(
      "Value mismatch at index %d, input (0x%x, 0x%x):\n"
      "  Expected: 0x%x\n"
      "  Actual  : 0x%x",
      index, absl::bit_cast<uint64>(std::get<0>(input)),
      absl::bit_cast<uint64>(std::get<1>(input)),
      absl::bit_cast<uint64>(expected), absl::bit_cast<uint64>(actual));
}

absl::Status RealMain(bool use_opt_ir, uint64 num_samples, int num_threads) {
  Testbench<Fpmul2x64, Float2x64, double> testbench(
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
