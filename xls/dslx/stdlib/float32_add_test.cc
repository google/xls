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
#include <cstdint>
#include <memory>
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/dslx/stdlib/float32_add_jit_wrapper.h"
#include "xls/dslx/stdlib/float32_test_utils.h"
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
static float ComputeExpected(fp::Float32Add* jit_wrapper, Float2x32 input) {
  float x = FlushSubnormal(std::get<0>(input));
  float y = FlushSubnormal(std::get<1>(input));
  return x + y;
}

// Computes FP addition via DSLX & the JIT.
static float ComputeActual(fp::Float32Add* jit_wrapper, Float2x32 input) {
  return jit_wrapper->Run(std::get<0>(input), std::get<1>(input)).value();
}

static std::unique_ptr<fp::Float32Add> CreateJit() {
  return fp::Float32Add::Create().value();
}

static absl::Status RealMain(uint64_t num_samples, int num_threads) {
  TestbenchBuilder<Float2x32, float, fp::Float32Add> builder(
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
  return xls::ExitStatus(xls::RealMain(absl::GetFlag(FLAGS_num_samples),
                                       absl::GetFlag(FLAGS_num_threads)));
}
