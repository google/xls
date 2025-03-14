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

#ifndef XLS_SCHEDULING_MUTUAL_EXCLUSION_PASS_H_
#define XLS_SCHEDULING_MUTUAL_EXCLUSION_PASS_H_


#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {
// Pass which merges together nodes that are determined to be mutually exclusive
// via SMT solver analysis.
class MutualExclusionPass : public SchedulingOptimizationFunctionBasePass {
 public:
  MutualExclusionPass()
      : SchedulingOptimizationFunctionBasePass(
            "mutual_exclusion",
            "Merge mutually exclusively used nodes using SMT solver") {}
  ~MutualExclusionPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, SchedulingUnit* unit,
      const SchedulingPassOptions& options,
      SchedulingPassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_MUTUAL_EXCLUSION_PASS_H_
