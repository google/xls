// Copyright 2025 The XLS Authors
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

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/dslx/stdlib/bfloat16_mul_jit_wrapper.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/tests/testbench.h"
#include "xls/tests/testbench_builder.h"

ABSL_FLAG(int, num_threads, 0,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {
namespace dslx {
namespace {

using BF16Input = std::tuple<uint16_t, uint16_t>;

Eigen::bfloat16 FlushSubnormal(Eigen::bfloat16 bf) {
  float f = static_cast<float>(bf);
  if (std::fpclassify(f) == FP_SUBNORMAL) {
    return Eigen::bfloat16(std::copysign(0.0f, f));
  }
  return bf;
}

// Bijectively enumerates all 2^32 (a, b) bit pairs for exhaustive runs.
BF16Input IndexToInputExhaustive(int64_t index) {
  uint32_t bits = static_cast<uint32_t>(index);
  return BF16Input(static_cast<uint16_t>(bits),
                   static_cast<uint16_t>(bits >> 16));
}

std::string PrintBF16(const Eigen::bfloat16& bf) {
  uint16_t bits = Eigen::numext::bit_cast<uint16_t>(bf);
  return absl::StrFormat("0x%04x (%g)", bits, static_cast<float>(bf));
}

std::string PrintInput(const BF16Input& input) {
  return absl::StrFormat(
      "(%s, %s)",
      PrintBF16(Eigen::numext::bit_cast<Eigen::bfloat16>(std::get<0>(input))),
      PrintBF16(Eigen::numext::bit_cast<Eigen::bfloat16>(std::get<1>(input))));
}

bool CompareResults(Eigen::bfloat16 expected, Eigen::bfloat16 actual) {
  // IEEE-754 leaves host NaN sign and payload bits implementation-defined
  // (e.g., x86 produces -qNaN for 0 * inf and propagates NaN input payloads),
  // whereas DSLX apfloat always returns canonical +qNaN.
  if (std::isnan(static_cast<float>(expected))) {
    constexpr uint16_t kCanonicalQNaN = 0x7fc0u;
    return Eigen::numext::bit_cast<uint16_t>(actual) == kCanonicalQNaN;
  }
  return Eigen::numext::bit_cast<uint16_t>(FlushSubnormal(expected)) ==
         Eigen::numext::bit_cast<uint16_t>(FlushSubnormal(actual));
}

Eigen::bfloat16 ComputeExpected(Bfloat16Mul* jit_wrapper, BF16Input input) {
  Eigen::bfloat16 a =
      Eigen::numext::bit_cast<Eigen::bfloat16>(std::get<0>(input));
  Eigen::bfloat16 b =
      Eigen::numext::bit_cast<Eigen::bfloat16>(std::get<1>(input));
  float x = static_cast<float>(FlushSubnormal(a));
  float y = static_cast<float>(FlushSubnormal(b));
  return Eigen::bfloat16(x * y);
}

Eigen::bfloat16 ComputeActual(Bfloat16Mul* jit_wrapper, BF16Input input) {
  auto to_value = [](uint16_t u) {
    bool sign = (u >> 15) & 1;
    uint8_t exp = (u >> 7) & 0xff;
    uint8_t frac = u & 0x7f;
    return Value::Tuple(
        {Value(UBits(sign, 1)), Value(UBits(exp, 8)), Value(UBits(frac, 7))});
  };

  Value result_val =
      jit_wrapper
          ->Run(to_value(std::get<0>(input)), to_value(std::get<1>(input)))
          .value();

  absl::Span<const Value> elements = result_val.elements();
  bool sign = elements[0].bits().ToUint64().value();
  uint8_t exp = elements[1].bits().ToUint64().value();
  uint8_t frac = elements[2].bits().ToUint64().value();

  uint16_t result_u16 = (static_cast<uint16_t>(sign & 1) << 15) |
                        (static_cast<uint16_t>(exp) << 7) |
                        (static_cast<uint16_t>(frac & 0x7f));
  return Eigen::numext::bit_cast<Eigen::bfloat16>(result_u16);
}

absl::Status RealMain(uint64_t num_samples, int num_threads) {
  constexpr uint64_t kExhaustiveSamples = uint64_t{1} << 32;
  TestbenchBuilder<BF16Input, Eigen::bfloat16, Bfloat16Mul> builder(
      ComputeExpected, ComputeActual,
      []() { return Bfloat16Mul::Create().value(); });
  // Use the bijective enumerator for exhaustive sweeps; otherwise rely on
  // TestbenchBuilder's default random uint16_t tuple generator.
  if (num_samples >= kExhaustiveSamples) {
    builder.SetIndexToInputFn(IndexToInputExhaustive);
  }
  builder.SetCompareResultsFn(CompareResults)
      .SetPrintInputFn(PrintInput)
      .SetPrintResultFn(PrintBF16)
      .SetNumSamples(num_samples);
  if (num_threads != 0) {
    builder.SetNumThreads(num_threads);
  }
  return builder.Build().Run();
}

}  // namespace
}  // namespace dslx
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::dslx::RealMain(absl::GetFlag(FLAGS_num_samples),
                                             absl::GetFlag(FLAGS_num_threads)));
}
