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

#ifndef XLS_PASSES_COLLAPSE_SELECT_CHAINS_PASS_H_
#define XLS_PASSES_COLLAPSE_SELECT_CHAINS_PASS_H_

#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"

namespace xls {

struct SelectChainElement {
  Node* node;

  // Save the selector separately to minimize the need to convert between
  // different select types when building the final one-hot-select.
  Node* selector;
  std::vector<Node*> cases;
  std::vector<TreeBitLocation> bit_locations;
};

struct SelectNodeChain {
  std::vector<SelectChainElement> chain_nodes;

  // Indicates the presence of a default node that will be reached when all
  // selectors in the chain are false.
  std::optional<Node*> final_default;
};

// Obtains all select node chains in the given function, sorted by root node
// ID in ascending order. Chains may not necessarily be fully one-hot.
absl::StatusOr<std::vector<SelectNodeChain>> GetSelectNodeChains(
    FunctionBase* fb, OptimizationContext& context,
    const QueryEngine& query_engine);

// Returns the best subchain of the given chain to collapse based on heuristic
// profitability, or nullopt if no subchain is profitable to collapse.
// Currently this is determined to be the largest connected sequence of cases,
// breaking ties in favor of subchains closer to the root (first) node in the
// chain.
absl::StatusOr<std::optional<SelectNodeChain>> DetermineBestSelectSubchain(
    const SelectNodeChain& chain, const QueryEngine& query_engine);

// Performs the transform to collapse the given select node chain into a
// single one_hot select.
absl::Status CollapseSelectNodeChain(const SelectNodeChain& chain);

// Pass which collapses chains of selects with disjoint selectors into a single
// one-hot-select.
//
// Chains of binary `select` operations, particularly where one case of a
// `select` feeds into another `select` (e.g.,
// `sel(pred_a, {val_a, sel(pred_b, {val_b, default_val})})`), are transformed
// into a single `one_hot_select` if the selectors are provably disjoint (at
// most one can be true at any given time). This simplifies the IR structure and
// can lead to more efficient hardware implementations.
class CollapseSelectChainsPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "collapse_select_chains";
  explicit CollapseSelectChainsPass()
      : OptimizationFunctionBasePass(kName,
                                     "BDD-based Select Chain Collapsing") {}
  ~CollapseSelectChainsPass() override = default;

  RedundancyGuard GetRedundancyGuard(
      const OptimizationPassOptions& options,
      OptimizationContext& context) const override {
    return RedundancyGuard::CanSkip();
  }

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_COLLAPSE_SELECT_CHAINS_PASS_H_
