// Copyright 2022 The XLS Authors
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

#include <vector>

#include "include/benchmark/benchmark.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_runtime.h"

namespace xls {
namespace {

// Measure the performance of conversion of a Value to/from the native XLS type
// layout used by the jit.
constexpr int kNumTypes = 10;
const char* kValueTypes[] = {
    "()",
    "bits[8]",
    "bits[64]",
    "bits[1024]",
    "bits[32][32]",
    "bits[64][32]",
    "bits[123][32]",
    "(bits[1], (bits[32], bits[64], bits[1][32])[5], bits[100])",
    "bits[1][1024]",
    "(bits[3], (), bits[5], bits[7])[3][10][42]",
};

static void BM_ValueToNativeLayout(benchmark::State& state) {
  Package package("BM");
  Type* type = Parser::ParseType(kValueTypes[state.range(0)], &package).value();
  std::minstd_rand bitgen;
  Value value = RandomValue(type, &bitgen);
  std::unique_ptr<JitRuntime> jit_runtime = JitRuntime::Create().value();
  std::vector<uint8_t> buffer(jit_runtime->GetTypeByteSize(type));
  for (auto _ : state) {
    jit_runtime->BlitValueToBuffer(value, type, absl::MakeSpan(buffer));
  }
}

static void BM_NativeLayoutToValue(benchmark::State& state) {
  Package package("BM");
  Type* type = Parser::ParseType(kValueTypes[state.range(0)], &package).value();
  std::unique_ptr<JitRuntime> jit_runtime = JitRuntime::Create().value();
  std::vector<uint8_t> buffer(jit_runtime->GetTypeByteSize(type), 0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(jit_runtime->UnpackBuffer(buffer.data(), type));
  }
}

BENCHMARK(BM_ValueToNativeLayout)->DenseRange(0, kNumTypes - 1);
BENCHMARK(BM_NativeLayoutToValue)->DenseRange(0, kNumTypes - 1);

BENCHMARK_MAIN();

}  // namespace
}  // namespace xls
