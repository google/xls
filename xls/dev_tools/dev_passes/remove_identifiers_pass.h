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

#ifndef XLS_DEV_TOOLS_DEV_PASSES_REMOVE_IDENTIFIERS_PASS_H_
#define XLS_DEV_TOOLS_DEV_PASSES_REMOVE_IDENTIFIERS_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dev_tools/remove_identifiers.h"
#include "xls/ir/node.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// A pass version of 'remove_identifers_main'. Running this pass will rewrite
// the package to remove the names of all functions, procs, nodes, channels,
// etc.
//
// This should only be used while investigating optimizer behavior using the
// ir-visualizer or other tools.
//
// WARNING: This pass does not preserve externally visible names. Once run on a
// design the various io-constraint and other configs which rely on naming
// pieces of the design will not work. This pass is only for dev/ investigatory
// purposes.
//
// NOTE: This pass **does not** preserve the id number of nodes.
class RemoveIdentifiersPass : public OptimizationPass {
 public:
  static constexpr std::string_view kName = "remove_identifiers_unsafe";
  explicit RemoveIdentifiersPass(
      StripOptions options = StripOptions{.new_package_name = "subrosa"})
      : OptimizationPass(kName, "UNSAFE: Remove all identifiers"),
        options_(options) {}
  ~RemoveIdentifiersPass() override = default;

  RedundancyGuard GetRedundancyGuard(
      const OptimizationPassOptions& options,
      OptimizationContext& context) const override {
    return RedundancyGuard::Never();
  }

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results,
                                   OptimizationContext& context) const override;

 private:
  StripOptions options_;
};

}  // namespace xls

#endif  // XLS_DEV_TOOLS_DEV_PASSES_REMOVE_IDENTIFIERS_PASS_H_
