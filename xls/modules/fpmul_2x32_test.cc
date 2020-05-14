// Copyright 2020 Google LLC
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

// Random-sampling test for the DSLX 2x32 floating-point multiplier.
#include <cmath>
#include <tuple>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/llvm_ir_jit.h"
#include "xls/ir/value_helpers.h"
#include "xls/ir/value_view_helpers.h"
#include "xls/tools/testbench.h"

ABSL_FLAG(bool, use_opt_ir, true, "Use optimized IR.");
ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64, num_samples, 1024 * 1024, "Number of random samples to test.");

namespace xls {

constexpr const char kOptIrPath[] = "xls/modules/fpmul_2x32.opt.ir";
constexpr const char kIrPath[] = "xls/modules/fpmul_2x32.ir";

using Float2x32 = std::tuple<float, float>;

float FlushSubnormals(float value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    return 0;
  }

  return value;
}

bool ZeroOrSubnormal(float value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// Generates two floats with reasonably unformly random bit patterns.
Float2x32 IndexToInput(uint64 index) {
  thread_local absl::BitGen bitgen;
  uint32 a = absl::Uniform(bitgen, 0u, std::numeric_limits<uint32_t>::max());
  uint32 b = absl::Uniform(bitgen, 0u, std::numeric_limits<uint32_t>::max());
  return Float2x32(absl::bit_cast<float>(a), absl::bit_cast<float>(b));
}

// The DSLX implementation uses the "round to nearest (half to even)"
// rounding mode, which is the default on most systems, hence we don't need
// to call fesetround().
// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
float ComputeExpected(Float2x32 input) {
  float x = FlushSubnormals(std::get<0>(input));
  float y = FlushSubnormals(std::get<1>(input));
  return x * y;
}

// Computes FP addition via DSLX & the JIT.
float ComputeActual(LlvmIrJit* jit, absl::Span<uint8> result_buffer,
                    Float2x32 input) {
  auto x = F32ToTuple(std::get<0>(input));
  auto y = F32ToTuple(std::get<1>(input));
  memset(result_buffer.data(), 0, result_buffer.size());
  XLS_CHECK_OK(jit->RunToBuffer({x, y}, result_buffer));

  F32TupleView result(result_buffer.data());
  return F32TupleViewToFloat(result);
}

// Compares expected vs. actual results, taking into account two special cases.
bool CompareResults(float a, float b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  return a == b || (std::isnan(a) && std::isnan(b)) ||
         (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
}

absl::Status RealMain(bool use_opt_ir, uint64 num_samples, int num_threads) {
  Testbench<Float2x32, float> testbench(
      GetXlsRunfilePath(use_opt_ir ? kOptIrPath : kIrPath),
      /*entry_function=*/"__fpmul_2x32__fpmul_2x32", 0, num_samples,
      /*max_failures=*/1, IndexToInput, ComputeExpected, ComputeActual,
      CompareResults);
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
