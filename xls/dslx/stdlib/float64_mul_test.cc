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

#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/stdlib/float64_mul_jit_wrapper.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/value_view_utils.h"
#include "xls/tools/testbench.h"
#include "xls/tools/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

using Float2x64 = std::tuple<double, double>;

static double FlushSubnormals(double value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    return 0;
  }

  return value;
}

static bool ZeroOrSubnormal(double value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// The DSLX implementation uses the "round to nearest (half to even)"
// rounding mode, which is the default on most systems, hence we don't need
// to call fesetround().
// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
static double ComputeExpected(fp::Float64Mul* jit_wrapper, Float2x64 input) {
  double x = FlushSubnormals(std::get<0>(input));
  double y = FlushSubnormals(std::get<1>(input));
  return x * y;
}

// Computes FP addition via DSLX & the JIT.
static double ComputeActual(fp::Float64Mul* jit_wrapper, Float2x64 input) {
  return jit_wrapper->Run(std::get<0>(input), std::get<1>(input)).value();
}

// Compares expected vs. actual results, taking into account two special cases.
static bool CompareResults(double a, double b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
}

static absl::Status RealMain(uint64_t num_samples, int num_threads) {
  TestbenchBuilder<Float2x64, double, fp::Float64Mul> builder(
      ComputeActual, ComputeExpected,
      []() { return fp::Float64Mul::Create().value(); });
  builder.SetCompareResultsFn(CompareResults).SetNumSamples(num_samples);
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
  return 0;
}
