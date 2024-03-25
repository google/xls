// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2021 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Random-sampling test for the DSLX 32-bit sin/cos unit.
#include <cmath>
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/value_view_utils.h"
#include "xls/tools/testbench_builder.h"
#include "third_party/xls_go_math/fp_sincos_32_jit_wrapper.h"

ABSL_FLAG(int, num_threads, 4,
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

using PackedFloat32 =
    PackedTupleView<PackedBitsView<1>, PackedBitsView<8>, PackedBitsView<23>>;
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
  uint32_t bexp = absl::Uniform(absl::IntervalClosed, bitgen, uint32_t(0),
                                uint32_t(127 + 30));
  uint32_t fraction = absl::Uniform(absl::IntervalClosed, bitgen, uint32_t(0),
                               (uint32_t(1) << 23) - uint32_t(1));
  uint32_t sign =
      absl::Uniform(absl::IntervalClosed, bitgen, uint32_t(0), uint32_t(1));
  uint32_t val = (sign << 31) | (bexp << 23) | fraction;
  return absl::bit_cast<float>(val);
}

// The DSLX implementation also flushes input subnormals to 0, so we do that
// here as well.
ResultT ComputeExpected(fp::FpSincos32* jit_wrapper, float input) {
  ResultT result;
  result.sin = sin(input);
  result.cos = cos(input);
  return result;
}

// Computes FP addition via DSLX & the JIT.
ResultT ComputeActual(fp::FpSincos32* jit_wrapper, float input) {
  PackedFloat32 packed_input(reinterpret_cast<uint8_t*>(&input), 0);
  ResultT result;
  PackedFloat2x32 packed_result(reinterpret_cast<uint8_t*>(&result), 0);
  CHECK_OK(jit_wrapper->Run(packed_input, packed_result));
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

std::string PrintResult(const ResultT& result) {
  return absl::StrCat("(", result.sin, ", ", result.cos, ")");
}

absl::Status RealMain(uint32_t num_samples, int num_threads) {
  TestbenchBuilder<float, ResultT, fp::FpSincos32> builder(
      ComputeExpected, ComputeActual,
      []() { return fp::FpSincos32::Create().value(); });
  builder.SetIndexToInputFn(IndexToInput)
      .SetCompareResultsFn(CompareResults)
      .SetPrintResultFn(PrintResult);
  if (num_threads != 0) {
    builder.SetNumThreads(num_threads);
  }
  if (num_samples != 0) {
    builder.SetNumSamples(num_samples);
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
