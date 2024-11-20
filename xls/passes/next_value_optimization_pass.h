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

#ifndef XLS_PASSES_NEXT_VALUE_OPTIMIZATION_PASS_H_
#define XLS_PASSES_NEXT_VALUE_OPTIMIZATION_PASS_H_

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which tries to optimize `next_value` nodes.
//
// Optimizations include:
// - removing literal predicates on `next_value` nodes (removing the
//   `next_value` node if dead),
// - splitting `next_value` nodes with `select`-based values (if small),
// - splitting `next_value` nodes with `priority_sel`-based values, and
// - splitting `next_value` nodes with `one_hot_sel`-based values (where safe).
//
// For best results, first modernizes old-style values on `next (...)` lines,
// converting them to `next_value` nodes.
class NextValueOptimizationPass : public OptimizationProcPass {
 public:
  static constexpr std::string_view kName = "next_value_opt";

  static constexpr int64_t kDefaultMaxSplitDepth = 10;
  explicit NextValueOptimizationPass(
      int64_t max_split_depth = kDefaultMaxSplitDepth)
      : OptimizationProcPass(kName, "Next Value Optimization"),
        max_split_depth_(max_split_depth) {}
  ~NextValueOptimizationPass() override = default;

 protected:
  const int64_t max_split_depth_;
  absl::StatusOr<bool> RunOnProcInternal(Proc* proc,
                                         const OptimizationPassOptions& options,
                                         PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_NEXT_VALUE_OPTIMIZATION_PASS_H_
