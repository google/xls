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

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "include/benchmark/benchmark.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/orc_jit.h"
#include "xls/jit/type_layout.h"

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

static TypeLayout CreateTypeLayout(Type* type) {
  std::unique_ptr<OrcJit> orc_jit = OrcJit::Create().value();
  LlvmTypeConverter type_converter(orc_jit->GetContext(),
                                   orc_jit->CreateDataLayout().value());
  return type_converter.CreateTypeLayout(type);
}

static void BM_ValueToNativeLayout(benchmark::State& state) {
  Package package("BM");
  Type* type = Parser::ParseType(kValueTypes[state.range(0)], &package).value();
  std::minstd_rand bitgen;
  Value value = RandomValue(type, bitgen);
  TypeLayout type_layout = CreateTypeLayout(type);
  std::vector<uint8_t> buffer(type_layout.size());
  for (auto _ : state) {
    type_layout.ValueToNativeLayout(value, buffer.data());
  }
}

static void BM_NativeLayoutToValue(benchmark::State& state) {
  Package package("BM");
  Type* type = Parser::ParseType(kValueTypes[state.range(0)], &package).value();
  TypeLayout type_layout = CreateTypeLayout(type);
  std::vector<uint8_t> buffer(type_layout.size(), 0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(type_layout.NativeLayoutToValue(buffer.data()));
  }
}

BENCHMARK(BM_ValueToNativeLayout)->DenseRange(0, kNumTypes - 1);
BENCHMARK(BM_NativeLayoutToValue)->DenseRange(0, kNumTypes - 1);

}  // namespace
}  // namespace xls

BENCHMARK_MAIN();
