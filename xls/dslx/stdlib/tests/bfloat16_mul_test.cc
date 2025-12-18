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
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
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

using BF16Tuple = std::tuple<uint8_t, uint8_t, uint8_t>;
using BF16Input = std::tuple<Eigen::bfloat16, Eigen::bfloat16>;

static Eigen::bfloat16 FlushSubnormal(Eigen::bfloat16 bf) {
  if (std::fpclassify(static_cast<float>(bf)) == FP_SUBNORMAL) {
    return Eigen::bfloat16(0.0f);
  }
  return bf;
}

static bool CompareResults(Eigen::bfloat16 a, Eigen::bfloat16 b) {
  if (std::isnan(static_cast<float>(a)) && std::isnan(static_cast<float>(b))) {
    return true;
  }
  return Eigen::numext::bit_cast<uint16_t>(FlushSubnormal(a)) ==
         Eigen::numext::bit_cast<uint16_t>(FlushSubnormal(b));
}

static Eigen::bfloat16 ComputeExpected(Bfloat16Mul* jit_wrapper,
                                       BF16Input input) {
  float x = static_cast<float>(FlushSubnormal(std::get<0>(input)));
  float y = static_cast<float>(FlushSubnormal(std::get<1>(input)));
  return Eigen::bfloat16(x * y);
}

static Eigen::bfloat16 ComputeActual(Bfloat16Mul* jit_wrapper,
                                     BF16Input input) {
  auto to_value = [](Eigen::bfloat16 bf) {
    uint16_t u = Eigen::numext::bit_cast<uint16_t>(bf);
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

static absl::Status RealMain(uint64_t num_samples, int num_threads) {
  TestbenchBuilder<BF16Input, Eigen::bfloat16, Bfloat16Mul> builder(
      ComputeActual, ComputeExpected,
      []() { return Bfloat16Mul::Create().value(); });
  builder.SetCompareResultsFn(CompareResults).SetNumSamples(num_samples);
  if (num_threads != 0) {
    builder.SetNumThreads(num_threads);
  }
  return builder.Build().Run();
}

}  // namespace dslx
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::dslx::RealMain(absl::GetFlag(FLAGS_num_samples),
                                             absl::GetFlag(FLAGS_num_threads)));
}
