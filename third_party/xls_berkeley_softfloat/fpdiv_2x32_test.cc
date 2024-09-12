// =================================================================
// Copyright 2021 The XLS Authors
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions, and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions, and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//  3. Neither the name of the University nor the names of its contributors may
//     be used to endorse or promote products derived from this software without
//     specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
// DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// =================================================================

// Random-sampling test for the DSLX 2x32 floating-point divider.
#include <cmath>
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/value_view_utils.h"
#include "xls/tests/testbench_builder.h"
#include "third_party/xls_berkeley_softfloat/fpdiv_2x32_jit_wrapper.h"

ABSL_FLAG(bool, use_opt_ir, true, "Use optimized IR.");
ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

using Float2x32 = std::tuple<float, float>;

float FlushSubnormals(float value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    // Note: When the divider flushes subnormals,
    // only the signficand is affected -
    // the sign is preserved.
    if (value < 0) {
      return -0.0;
    } else {
      return 0.0;
    }
  }

  return value;
}

bool ZeroOrSubnormal(float value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// Generates two floats with reasonably unformly random bit patterns.
Float2x32 IndexToInput(uint64_t index) {
  thread_local absl::BitGen bitgen;
  uint32_t a = absl::Uniform(bitgen, 0u, std::numeric_limits<uint32_t>::max());
  uint32_t b = absl::Uniform(bitgen, 0u, std::numeric_limits<uint32_t>::max());
  return Float2x32(absl::bit_cast<float>(a), absl::bit_cast<float>(b));
}

// The DSLX implementation uses the "round to nearest (half to even)"
// rounding mode, which is the default on most systems, hence we don't need
// to call fesetround().
// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
float ComputeExpected(fp::Fpdiv2x32* jit_wrapper, Float2x32 input) {
  float x = FlushSubnormals(std::get<0>(input));
  float y = FlushSubnormals(std::get<1>(input));
  return x / y;
}

// Computes FP addition via DSLX & the JIT.
float ComputeActual(fp::Fpdiv2x32* jit_wrapper, Float2x32 input) {
  return jit_wrapper->Run(std::get<0>(input), std::get<1>(input)).value();
}

// Compares expected vs. actual results, taking into account two special cases.
bool CompareResults(float a, float b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
}

std::unique_ptr<fp::Fpdiv2x32> CreateJit() {
  return fp::Fpdiv2x32::Create().value();
}

absl::Status RealMain(bool use_opt_ir, uint64_t num_samples, int num_threads) {
  TestbenchBuilder<Float2x32, float, fp::Fpdiv2x32> builder(
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
  QCHECK_OK(xls::RealMain(absl::GetFlag(FLAGS_use_opt_ir),
                          absl::GetFlag(FLAGS_num_samples),
                          absl::GetFlag(FLAGS_num_threads)));
  return 0;
}
