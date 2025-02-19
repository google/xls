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

#ifndef XLS_PASSES_BASIC_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_BASIC_SIMPLIFICATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// This pass does simple pattern-matching optimizations which are ~always a good
// idea to do (replacing a node with a constant, removing operands of nodes,
// etc). They improve QoR, do not increase the number of nodes in the graph,
// preserve the same abstraction level, and do not impede later optimizations
// via obfuscation. These optimizations require no analyses beyond looking at
// the node and its operands. Examples include: not(not(x)) => x, x + 0 => x,
// etc.
class BasicSimplificationPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "basic_simp";
  explicit BasicSimplificationPass()
      : OptimizationFunctionBasePass(kName, "Basic Simplifications") {}
  ~BasicSimplificationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext* context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_BASIC_SIMPLIFICATION_PASS_H_
