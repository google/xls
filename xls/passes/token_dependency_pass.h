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

#ifndef XLS_PASSES_TOKEN_DEPENDENCY_PASS_H_
#define XLS_PASSES_TOKEN_DEPENDENCY_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which turns data dependencies between certain effectful operations into
// token dependencies. In particular, transitive data dependencies between
// receives and other effectful ops are turned into token dependencies whenever
// no such token dependency already exists.
class TokenDependencyPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "token_dependency";
  TokenDependencyPass()
      : OptimizationFunctionBasePass(kName,
                                     "Convert data dependencies between "
                                     "effectful operations into token "
                                     "dependencies") {}
  ~TokenDependencyPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_TOKEN_DEPENDENCY_PASS_H_
