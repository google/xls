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

#ifndef XLS_PASSES_TOOLS_TEST_MULTI_HEADER_PASS_H_
#define XLS_PASSES_TOOLS_TEST_MULTI_HEADER_PASS_H_

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/tools/test_multi_header_helper.h"

namespace xls {

// Helper struct comment that must not be extracted as pass documentation.
struct TestMultiHeaderHelperData {
  int64_t count = 0;
};

// Primary multi-header pass documentation.
class TestMultiHeaderPass final : public TestMultiHeaderBasePass {
 public:
  static constexpr std::string_view kName = "test_multi_header";
  TestMultiHeaderPass()
      : TestMultiHeaderBasePass(kName, "Test multi-header pass") {}
  ~TestMultiHeaderPass() override = default;

 protected:
  absl::StatusOr<bool> RunInternal(Package* ir,
                                   const OptimizationPassOptions& options,
                                   PassResults* results,
                                   OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_TOOLS_TEST_MULTI_HEADER_PASS_H_
