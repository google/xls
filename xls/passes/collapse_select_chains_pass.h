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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
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

  absl::StatusOr<SelectChainElement> Clone(
      absl::flat_hash_map<Node*, Node*>& original_node_to_clone) const {
    SelectChainElement cloned_element;

    auto orig_item = original_node_to_clone.find(node);
    if (orig_item == original_node_to_clone.end()) {
      return absl::NotFoundError(absl::StrFormat(
          "Node %s not found in original_node_to_clone", node->ToString()));
    }
    cloned_element.node = orig_item->second;

    auto orig_selector = original_node_to_clone.find(selector);
    if (orig_selector == original_node_to_clone.end()) {
      return absl::NotFoundError(
          absl::StrFormat("Selector %s not found in original_node_to_clone",
                          selector->ToString()));
    }
    cloned_element.selector = orig_selector->second;

    cloned_element.cases.reserve(cases.size());
    for (const Node* case_node : cases) {
      auto orig_case = original_node_to_clone.find(case_node);
      if (orig_case == original_node_to_clone.end()) {
        return absl::NotFoundError(
            absl::StrFormat("Case %s not found in original_node_to_clone",
                            case_node->ToString()));
      }
      cloned_element.cases.push_back(orig_case->second);
    }

    // todo joshuata: change to function maybe
    cloned_element.bit_locations = bit_locations;
    return cloned_element;
  }
};

struct SelectNodeChain {
  std::vector<SelectChainElement> chain_nodes;

  // Indicates the presence of a default node that will be reached when all
  // selectors in the chain are false.
  std::optional<Node*> final_default;

  absl::StatusOr<SelectNodeChain> Clone(
      absl::flat_hash_map<Node*, Node*>& original_node_to_clone) const {
    SelectNodeChain cloned_chain;
    cloned_chain.chain_nodes.reserve(chain_nodes.size());
    for (const SelectChainElement& element : chain_nodes) {
      XLS_ASSIGN_OR_RETURN(SelectChainElement cloned_element,
                           element.Clone(original_node_to_clone));
      cloned_chain.chain_nodes.push_back(cloned_element);
    }
    if (final_default.has_value()) {
      auto orig_default = original_node_to_clone.find(*final_default);
      if (orig_default == original_node_to_clone.end()) {
        return absl::NotFoundError(absl::StrFormat(
            "Default node %s not found in original_node_to_clone",
            (*final_default)->ToString()));
      }
      cloned_chain.final_default = orig_default->second;
    } else {
      cloned_chain.final_default = std::nullopt;
    }
    return cloned_chain;
  }
};

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

  // Obtains all select node chains in the given function. Chains may not
  // necessarily be fully one-hot.
  absl::StatusOr<std::vector<SelectNodeChain>> GetSelectNodeChains(
      FunctionBase* fb, OptimizationContext& context,
      const QueryEngine& query_engine) const;

  // Looks for the best subchain of the given chain to collapse based on
  // heuristic profitability.
  // Returns std::nullopt if the chain is not profitable to collapse.
  static absl::StatusOr<std::optional<SelectNodeChain>>
  DetermineBestSelectSubchain(const SelectNodeChain& chain,
                              const QueryEngine& query_engine);

  // Filters select chains to those that are profitable to collapse.
  static absl::StatusOr<std::vector<SelectNodeChain>>
  GetProfitableSelectNodeChains(std::vector<SelectNodeChain> chains,
                                const QueryEngine& query_engine);

  // Performs the transform to collapse the given select node chain into a
  // single one_hot select.
  absl::Status CollapseSelectNodeChain(const SelectNodeChain& chain) const;

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
