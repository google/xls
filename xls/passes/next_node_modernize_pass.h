// Copyright 2024 The XLS Authors
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

#ifndef XLS_PASSES_NEXT_NODE_MODERNIZE_PASS_H_
#define XLS_PASSES_NEXT_NODE_MODERNIZE_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
namespace xls {

// Converts a proc from using old-style next() line to next nodes.
//
// Makes no attempt to make a clever next node and merely creates a single
// unconditional next node for each state element.
//
// TODO(allight): Remove this pass once transition to next-nodes is done
// everywhere.
class NextNodeModernizePass : public OptimizationProcPass {
 public:
  static constexpr std::string_view kName = "next_node_modernize";
  explicit NextNodeModernizePass()
      : OptimizationProcPass(kName, "Next node modernization conversion pass") {
  }
  ~NextNodeModernizePass() override = default;

 protected:
  absl::StatusOr<bool> RunOnProcInternal(Proc* proc,
                                         const OptimizationPassOptions& options,
                                         PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_NEXT_NODE_MODERNIZE_PASS_H_
