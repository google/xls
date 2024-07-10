// Copyright 2023 The XLS Authors
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

#ifndef XLS_INTERPRETER_BLOCK_EVALUATOR_TEST_BASE_H_
#define XLS_INTERPRETER_BLOCK_EVALUATOR_TEST_BASE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_test_base.h"

namespace xls {

struct BlockEvaluatorTestParam {
  const BlockEvaluator* evaluator;
  bool supports_fifos;
};

class BlockEvaluatorTest
    : public IrTestBase,
      public testing::WithParamInterface<BlockEvaluatorTestParam> {
 public:
  const BlockEvaluator& evaluator() { return *GetParam().evaluator; }

  bool SupportsFifos() { return GetParam().supports_fifos; }
};

struct FifoTestParam {
  BlockEvaluatorTestParam block_evaluator_test_param;
  FifoConfig fifo_config;
};

class FifoTest : public IrTestBase,
                 public testing::WithParamInterface<FifoTestParam> {
 public:
  const BlockEvaluator& evaluator() {
    return *(GetParam().block_evaluator_test_param.evaluator);
  }

  bool SupportsFifos() {
    return GetParam().block_evaluator_test_param.supports_fifos;
  }

  FifoConfig fifo_config() { return GetParam().fifo_config; }
};

inline std::vector<FifoTestParam> GenerateFifoTestParams(
    const BlockEvaluatorTestParam& block_evaluator_test_param) {
  std::vector<FifoTestParam> params;
  for (int64_t depth : {0, 1, 2, 3, 4, 10, 128, 256}) {
    for (bool bypass : {true, false}) {
      for (bool register_push_outputs : {true, false}) {
        for (bool register_pop_outputs : {true, false}) {
          if (depth == 0 &&
              (!bypass || register_push_outputs || register_pop_outputs)) {
            // Unsupported configurations of depth=0 fifos.
            continue;
          }
          params.push_back(FifoTestParam{
              .block_evaluator_test_param = block_evaluator_test_param,
              .fifo_config = FifoConfig(depth, bypass, register_push_outputs,
                                        register_pop_outputs)});
        }
      }
    }
  }
  return params;
}

inline std::string FifoTestName(
    const ::testing::TestParamInfo<FifoTestParam>& info) {
  const auto& param = info.param;
  return absl::StrCat(
      param.block_evaluator_test_param.evaluator->name(), "Depth",
      param.fifo_config.depth(), "Bypass",
      static_cast<int>(param.fifo_config.bypass()), "RegisterPushOutputs",
      static_cast<int>(param.fifo_config.register_push_outputs()),
      "RegisterPopOutputs",
      static_cast<int>(param.fifo_config.register_pop_outputs()));
}

}  // namespace xls

#endif  // XLS_INTERPRETER_BLOCK_EVALUATOR_TEST_BASE_H_
