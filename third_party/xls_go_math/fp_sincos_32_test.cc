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

// Random-sampling test for the DSLX 32-bit sin/cos unit.
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
#include "third_party/xls_go_math/fp_sincos_32_jit_wrapper.h"
#include "xls/tools/testbench.h"

ABSL_FLAG(bool, use_opt_ir, true, "Use optimized IR.");
ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int32_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

// Returned bits of fp_sincos_32
typedef struct {
  // Fields are reversed relative to xls
  // because earlier fields in an xls tuple
  // occupy higher bits.
  float cos;
  float sin;
} ResultT;

using PackedFloat32 = PackedTupleView<PackedBitsView<1>, PackedBitsView<8>, PackedBitsView<23>>;
using PackedFloat2x32 = PackedTupleView<PackedFloat32, PackedFloat32>;

float FlushSubnormals(float value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    return 0;
  }

  return value;
}

bool ZeroOrSubnormal(float value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// Generates two floats with reasonably uniformly random bit patterns.
float IndexToInput(uint32_t index) {
  thread_local absl::BitGen bitgen;
  // Loss of accuracy is expected as numbers grow larger than 2^30.
  uint32_t bexp = absl::Uniform(absl::IntervalClosed, bitgen, uint32_t(0), uint32_t(127 + 30));
  uint32_t sfd = absl::Uniform(absl::IntervalClosed, bitgen, uint32_t(0), (uint32_t(1) << 23) - uint32_t(1));
  uint32_t sign = absl::Uniform(absl::IntervalClosed, bitgen, uint32_t(0), uint32_t(1));
  uint32_t val = (sign << 31) | (bexp << 23) | sfd;
  return absl::bit_cast<float>(val);
}

// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
ResultT ComputeExpected(float input) {
  ResultT result;
  result.sin = sin(input);
  result.cos = cos(input);
  return result;
}

// Computes FP addition via DSLX & the JIT.
ResultT ComputeActual(FpSincos32* jit_wrapper, float input) {
  PackedFloat32 packed_input(reinterpret_cast<uint8_t*>(&input), 0);
  ResultT result;
  PackedFloat2x32 packed_result(reinterpret_cast<uint8_t*>(&result),0);
  jit_wrapper->Run(packed_input, packed_result);
  return result;
}

// Compares expected vs. actual results, taking into account two special cases.
bool CompareResults(ResultT a, ResultT b) {
  // DSLX flushes subnormal outputs, while regular FP addition does not, so
  // just check for that here.
  auto results_match = [](float a, float b) {
    float diff = a - b;
    return a == b || (abs(diff) < 0.0001) || (std::isnan(a) && std::isnan(b)) ||
           (ZeroOrSubnormal(a) && ZeroOrSubnormal(b));
  };

  return results_match(a.sin, b.sin) && results_match(a.cos, b.cos);
}

void LogMismatch(uint32_t index, float input, ResultT expected,
                 ResultT actual) {
  XLS_LOG(ERROR) << absl::StrFormat(
      "Value mismatch at index %d, input (%x == %lf):\n"
      "  Expected sin:\t%#08x == %lf\n"
      "  Actual sin:\t%#08x == %lf\n"
      "  Expected cos:\t%#08x == %lf\n"
      "  Actual cos:\t%#08x == %lf\n",
      index, absl::bit_cast<uint32_t>(input), input,
      absl::bit_cast<uint32_t>(expected.sin), expected.sin,
      absl::bit_cast<uint32_t>(actual.sin), actual.sin, 
      absl::bit_cast<uint32_t>(expected.cos), expected.cos, 
      absl::bit_cast<uint32_t>(actual.cos), actual.cos);
}

absl::Status RealMain(bool use_opt_ir, uint32_t num_samples, int num_threads) {
  Testbench<FpSincos32, float, ResultT> testbench(
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
