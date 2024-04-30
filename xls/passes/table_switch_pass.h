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
#ifndef XLS_PASSES_TABLE_SWITCH_PASS_H_
#define XLS_PASSES_TABLE_SWITCH_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/passes/optimization_pass.h"

namespace xls {

// TableSwitchPass converts chains of Select nodes into ArrayIndex ops.
// These chains have the form:
// sel.(N)(eq.X, literal.A, literal.B)
// sel.(N+1)(eq.Y, sel.(N), literal.C)
// sel.(N+2)(eq.Z, sel.(N+1), literal.D)
// And so on. In these chains, eq.X, eq.Y, and eq.Z must all be comparisons of
// the same value against different literals.
//
// Current limitations:
//  - Either the start or end index in the chain must be 0.
//  - The increment between indices must be positive or negative 1.
//  - There can be no "gaps" between indices.
//  - The Select ops have to be binary (i.e., selecting between only two cases).
class TableSwitchPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "table_switch";
  TableSwitchPass()
      : OptimizationFunctionBasePass(kName, "Table switch conversion") {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_TABLE_SWITCH_PASS_H_
