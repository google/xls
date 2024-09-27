// Copyright 2021 The XLS Authors
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

#include "xls/interpreter/block_interpreter.h"

#include <string>

#include "gtest/gtest.h"
#include "xls/interpreter/block_evaluator_test_base.h"

namespace xls {
namespace {

inline constexpr BlockEvaluatorTestParam kBlockInterpreterTestParam = {
    .evaluator = &kInterpreterBlockEvaluator,
    .supports_fifos = true,
    .supports_observer = true};

INSTANTIATE_TEST_SUITE_P(BlockInterpreterTest, BlockEvaluatorTest,
                         testing::Values(kBlockInterpreterTestParam),
                         [](const auto& v) -> std::string {
                           return std::string(v.param.evaluator->name());
                         });

INSTANTIATE_TEST_SUITE_P(
    BlockInterpreterFifoTest, FifoTest,
    testing::ValuesIn(GenerateFifoTestParams(kBlockInterpreterTestParam)),
    FifoTestName);

}  // namespace
}  // namespace xls
