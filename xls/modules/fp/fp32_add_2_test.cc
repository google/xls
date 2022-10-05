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

// Random-sampling test for the DSLX 2x32 floating-point adder.
#include <cmath>
#include <limits>

#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value_helpers.h"
#include "xls/ir/value_view_helpers.h"
#include "xls/modules/fp/fp32_add_2_jit_wrapper.h"
#include "xls/tools/testbench.h"
#include "xls/tools/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

using Float2x32 = std::tuple<float, float>;

// The DSLX implementation uses the "round to nearest (half to even)"
// rounding mode, which is the default on most systems, hence we don't need
// to call fesetround().
// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
float ComputeExpected(fp::Fp32Add2* jit_wrapper, Float2x32 input) {
  float x = FlushSubnormal(std::get<0>(input));
  float y = FlushSubnormal(std::get<1>(input));
  return x + y;
}

// Computes FP addition via DSLX & the JIT.
float ComputeActual(fp::Fp32Add2* jit_wrapper, Float2x32 input) {
  return jit_wrapper->Run(std::get<0>(input), std::get<1>(input)).value();
}

// Compares expected vs. actual results, taking into account two special cases.
bool CompareResults(float a, float b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
}

std::unique_ptr<fp::Fp32Add2> CreateJit() {
  return fp::Fp32Add2::Create().value();
}

absl::Status RealMain(uint64_t num_samples, int num_threads) {
  TestbenchBuilder<Float2x32, float, fp::Fp32Add2> builder(
      ComputeExpected, ComputeActual, CreateJit);
  builder.SetCompareResultsFn(CompareResults).SetNumSamples(num_samples);
  if (num_threads != 0) {
    builder.SetNumThreads(num_threads);
  }
  return builder.Build().Run();
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  XLS_QCHECK_OK(xls::RealMain(absl::GetFlag(FLAGS_num_samples),
                              absl::GetFlag(FLAGS_num_threads)));
  return 0;
}
