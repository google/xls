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

#include "gtest/gtest.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/ir_test_base.h"

namespace xls {

struct BlockEvaluatorTestParam {
  const BlockEvaluator* evaluator;
  bool supports_hierarhical_blocks;
};

class BlockEvaluatorTest
    : public IrTestBase,
      public testing::WithParamInterface<BlockEvaluatorTestParam> {
 public:
  const BlockEvaluator& evaluator() { return *GetParam().evaluator; }

  bool SupportsHierarchicalBlocks() {
    return GetParam().supports_hierarhical_blocks;
  }
};
}  // namespace xls

#endif  // XLS_INTERPRETER_BLOCK_EVALUATOR_TEST_BASE_H_
