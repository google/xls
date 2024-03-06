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

#include "xls/passes/bdd_function.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/examples/sample_packages.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_function_test.inc"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class BddFunctionTest : public IrTestBase {};

TEST_F(BddFunctionTest, SimpleOr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* t = p->GetBitsType(8);
  fb.Or(fb.Param("x", t), fb.Param("y", t));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BddFunction> bdd_function,
                           BddFunction::Run(f));
  EXPECT_THAT(bdd_function->Evaluate(
                  {Value(UBits(0b11110000, 8)), Value(UBits(0b10101010, 8))}),
              IsOkAndHolds(Value(UBits(0b11111010, 8))));
}

TEST_F(BddFunctionTest, JustAParam) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BddFunction> bdd_function,
                           BddFunction::Run(f));
  EXPECT_THAT(bdd_function->Evaluate({Value(UBits(0b11011011, 8))}),
              IsOkAndHolds(Value(UBits(0b11011011, 8))));
}

TEST_F(BddFunctionTest, AndNot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* t = p->GetBitsType(8);
  BValue x = fb.Param("x", t);
  fb.And(x, fb.Not(x));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BddFunction> bdd_function,
                           BddFunction::Run(f));
  // AND of a value and its inverse should be zero.
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_EQ(bdd_function->GetBddNode(f->return_value(), i),
              bdd_function->bdd().zero());
  }

  EXPECT_THAT(bdd_function->Evaluate({Value(UBits(0b11000011, 8))}),
              IsOkAndHolds(Value(UBits(0, 8))));
}

TEST_F(BddFunctionTest, Parity) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue parity = fb.Literal(UBits(0, 1));
  for (int64_t i = 0; i < 32; ++i) {
    parity = fb.Xor(parity, fb.BitSlice(x, i, 1));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  const int64_t kNumSamples = 100;
  std::minstd_rand engine;
  for (int64_t path_limit : {0, 10, 1000, 10000}) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BddFunction> bdd_function,
                             BddFunction::Run(f, path_limit));
    EXPECT_THAT(bdd_function->Evaluate({Value(UBits(0, 32))}),
                IsOkAndHolds(Value(UBits(0, 1))));
    EXPECT_THAT(bdd_function->Evaluate({Value(UBits(1, 32))}),
                IsOkAndHolds(Value(UBits(1, 1))));
    EXPECT_THAT(bdd_function->Evaluate({Value(UBits(0xffffffffLL, 32))}),
                IsOkAndHolds(Value(UBits(0, 1))));

    for (int64_t i = 0; i < kNumSamples; ++i) {
      std::vector<Value> inputs = RandomFunctionArguments(f, engine);
      XLS_ASSERT_OK_AND_ASSIGN(
          Value expected, DropInterpreterEvents(InterpretFunction(f, inputs)));
      XLS_ASSERT_OK_AND_ASSIGN(Value actual, bdd_function->Evaluate(inputs));
      EXPECT_EQ(expected, actual);
    }
  }
}

TEST_F(BddFunctionTest, BenchmarkTest) {
  // Run samples through various benchmarks and verify against the interpreter.
  //
  // TODO(leary): 2021-07-20 Temporary workaround for copybara rewrite -- want
  // to get this into a .inc file.
  // clang-format off
  std::vector<std::string> benchmarks = {
    "examples/crc32/crc32", "examples/sha256"}; // NOLINT
  // clang-format on
  for (std::string& benchmark : benchmarks) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Package> p,
        sample_packages::GetBenchmark(benchmark, /*optimized=*/true));
    XLS_ASSERT_OK_AND_ASSIGN(Function * entry, p->GetTopAsFunction());

    std::minstd_rand engine;
    const int64_t kSampleCount = 32;
    for (int64_t path_limit : {10, 100, 1000}) {
      XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BddFunction> bdd_function,
                               BddFunction::Run(entry, path_limit));
      for (int64_t i = 0; i < kSampleCount; ++i) {
        XLS_ASSERT_OK_AND_ASSIGN(
            std::vector<Value> inputs,
            GenerateFunctionArguments(entry, engine, benchmark));

        XLS_ASSERT_OK_AND_ASSIGN(
            Value expected,
            DropInterpreterEvents(InterpretFunction(entry, inputs)));
        XLS_ASSERT_OK_AND_ASSIGN(Value actual, bdd_function->Evaluate(inputs));
        EXPECT_EQ(expected, actual);
      }
    }
  }
}

}  // namespace
}  // namespace xls
