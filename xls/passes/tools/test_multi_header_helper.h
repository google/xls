// Copyright 2026 The XLS Authors
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

#ifndef XLS_PASSES_TOOLS_TEST_MULTI_HEADER_HELPER_H_
#define XLS_PASSES_TOOLS_TEST_MULTI_HEADER_HELPER_H_

#include <cstdint>

#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

int64_t TestMultiHeaderHelperValue();

// Decoy base pass documentation in helper header.
class TestMultiHeaderBasePass : public OptimizationPass {
 public:
  using OptimizationPass::OptimizationPass;

  RedundancyGuard GetRedundancyGuard(
      const OptimizationPassOptions& options,
      OptimizationContext& context) const override {
    return RedundancyGuard::CanSkip();
  }
};

}  // namespace xls

#endif  // XLS_PASSES_TOOLS_TEST_MULTI_HEADER_HELPER_H_
