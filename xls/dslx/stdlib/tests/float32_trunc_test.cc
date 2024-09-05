// Copyright 2024 The XLS Authors
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

// Random-sampling test for the DSLX 32 bit floating-point flooring.
#include <cmath>
#include <cstdint>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/dslx/stdlib/float32_trunc_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/float32_test_utils.h"
#include "xls/tools/testbench.h"
#include "xls/tools/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

static float ComputeExpected(fp::F32Trunc* jit_wrapper, float input) {
  return truncf(input);
}

static float ComputeActual(fp::F32Trunc* jit_wrapper, float input) {
  return jit_wrapper->Run(input).value();
}

static absl::Status RealMain(uint64_t num_samples, int num_threads) {
  TestbenchBuilder<float, float, fp::F32Trunc> builder(
      ComputeActual, ComputeExpected,
      []() { return fp::F32Trunc::Create().value(); });
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
