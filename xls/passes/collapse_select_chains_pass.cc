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

#include "xls/passes/collapse_select_chains_pass.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

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

  int64_t total_cases() const {
    int64_t total = 0;
    for (const SelectChainElement& element : chain_nodes) {
      total += element.cases.size();
    }
    return total;
  }
};

bool IsChainableSelect(Node* node) {
  if (!node->GetType()->IsBits()) {
    return false;
  }
  if (node->Is<Select>()) {
    Select* sel = node->As<Select>();
    if (sel->cases().size() == 2 && !sel->default_value().has_value()) {
      return true;
    }
    if (sel->cases().size() == 1 && sel->default_value().has_value() &&
        sel->selector()->BitCountOrDie() == 1) {
      return true;
    }
    return false;
  }
  if (node->Is<PrioritySelect>()) {
    return true;
  }
  if (node->Is<OneHotSelect>()) {
    return true;
  }
  return false;
}

absl::StatusOr<std::vector<SelectNodeChain>> GetSelectNodeChains(
    FunctionBase* fb, OptimizationContext& context,
    const QueryEngine& query_engine) {
  // A set containing the select instructions visited so far so we don't waste
  // time considering selects which have already been processed.
  absl::flat_hash_set<Node*> visited_selects;

  std::vector<SelectNodeChain> chains;

  // Walk the graph in reverse order looking for chains of selects (Select,
  // PrioritySelect, OneHotSelect) where the default of one select
  // is another select.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> reverse_topo_sort_nodes,
                       context.ReverseTopoSort(fb));
  for (Node* node : reverse_topo_sort_nodes) {
    if (!IsChainableSelect(node) || visited_selects.contains(node)) {
      continue;
    }

    std::vector<SelectChainElement> elements;
    std::vector<Node*> chain_nodes;
    std::optional<Node*> final_default = std::nullopt;

    Node* current = node;
    while (current != nullptr && IsChainableSelect(current) &&
           !visited_selects.contains(current)) {
      if (current->Is<Select>()) {
        Select* sel = current->As<Select>();
        chain_nodes.push_back(current);
        if (sel->cases().size() == 2 && !sel->default_value().has_value()) {
          elements.push_back(SelectChainElement{
              .node = sel,
              .selector = sel->selector(),
              .cases = {sel->get_case(1)},
              .bit_locations = {TreeBitLocation(sel->selector(), 0)},
          });
          current = sel->get_case(0);
        } else {
          CHECK_EQ(sel->cases().size(), 1);
          CHECK(sel->default_value().has_value());
          elements.push_back(SelectChainElement{
              .node = sel,
              .selector = sel->selector(),
              .cases = {*sel->default_value()},
              .bit_locations = {TreeBitLocation(sel->selector(), 0)},
          });
          current = sel->get_case(0);
        }
        final_default = current;
      } else if (current->Is<PrioritySelect>()) {
        PrioritySelect* ps = current->As<PrioritySelect>();
        chain_nodes.push_back(current);
        std::vector<Node*> chunk_cases;
        std::vector<TreeBitLocation> bit_locations;
        chunk_cases.reserve(ps->cases().size());
        bit_locations.reserve(ps->cases().size());
        for (int64_t i = 0; i < ps->cases().size(); ++i) {
          chunk_cases.push_back(ps->get_case(i));
          bit_locations.push_back(TreeBitLocation(ps->selector(), i));
        }
        elements.push_back(SelectChainElement{
            .node = ps,
            .selector = ps->selector(),
            .cases = std::move(chunk_cases),
            .bit_locations = std::move(bit_locations),
        });
        current = ps->default_value();
        final_default = current;
      } else if (current->Is<OneHotSelect>()) {
        OneHotSelect* ohs = current->As<OneHotSelect>();
        chain_nodes.push_back(current);
        std::vector<Node*> chunk_cases;
        std::vector<TreeBitLocation> bit_locations;
        chunk_cases.reserve(ohs->cases().size());
        bit_locations.reserve(ohs->cases().size());
        for (int64_t i = 0; i < ohs->cases().size(); ++i) {
          chunk_cases.push_back(ohs->get_case(i));
          bit_locations.push_back(TreeBitLocation(ohs->selector(), i));
        }
        elements.push_back(SelectChainElement{
            .node = ohs,
            .selector = ohs->selector(),
            .cases = std::move(chunk_cases),
            .bit_locations = std::move(bit_locations),
        });
        final_default = std::nullopt;
        current = nullptr;
      } else {
        break;
      }
    }

    if (chain_nodes.size() == 1 && node->Is<OneHotSelect>()) {
      VLOG(4) << "Skipping one-hot-select chain of size 1, already optimized.";
      continue;
    }

    chains.push_back(SelectNodeChain{
        .chain_nodes = std::move(elements),
        .final_default = final_default,
    });

    visited_selects.insert(chain_nodes.begin(), chain_nodes.end());
  }

  return chains;
}

// Looks for the best subchain of the given chain to collapse.
// Currently this is calculated as the longest chain of nodes starting from the
// earliest (last) node in the chain.
// TODO(joshuata): Expand to return multiple potential subchains.
absl::StatusOr<std::optional<SelectNodeChain>> DetermineBestSelectSubchain(
    const SelectNodeChain& chain, const QueryEngine& query_engine) {
  std::vector<SelectChainElement> best_subchain;
  best_subchain.reserve(chain.chain_nodes.size());

  std::vector<TreeBitLocation> bit_locations;

  // Walk backwards from the end of the chain to find the longest compatible
  // subchain.
  for (auto it = chain.chain_nodes.rbegin(); it != chain.chain_nodes.rend();
       ++it) {
    const SelectChainElement& el = *it;
    std::vector<TreeBitLocation> candidate_bit_locations = bit_locations;
    candidate_bit_locations.insert(candidate_bit_locations.end(),
                                   el.bit_locations.begin(),
                                   el.bit_locations.end());

    // We have reached a point where selectors are no longer disjoint
    if (!query_engine.AtMostOneTrue(candidate_bit_locations)) {
      break;
    }
    bit_locations = std::move(candidate_bit_locations);
    best_subchain.push_back(el);
  }

  // Reverse best_subchain so it is in root-to-leaf order.
  absl::c_reverse(best_subchain);

  int64_t total_cases = 0;
  for (const SelectChainElement& el : best_subchain) {
    total_cases += el.cases.size();
  }

  // Only transform if the select chain is sufficiently long to avoid
  // interfering with select optimizations as plain selects are generally
  // easier to analyse/transform.
  // TODO(meheff): 2021/12/23 Consider tuning this value.
  if (total_cases <= 4) {
    return std::nullopt;
  }

  if (best_subchain.size() == 1 &&
      best_subchain.front().node->Is<OneHotSelect>()) {
    return std::nullopt;
  }

  std::optional<Node*> final_default = chain.final_default;
  if (final_default.has_value() && query_engine.AtLeastOneTrue(bit_locations)) {
    final_default = std::nullopt;
  }

  return SelectNodeChain{.chain_nodes = std::move(best_subchain),
                         .final_default = final_default};
}

absl::StatusOr<std::vector<SelectNodeChain>> GetProfitableSelectNodeChains(
    std::vector<SelectNodeChain> chains, const QueryEngine& query_engine) {
  std::vector<SelectNodeChain> profitable_chains;
  profitable_chains.reserve(chains.size());

  for (const SelectNodeChain& chain : chains) {
    XLS_ASSIGN_OR_RETURN(std::optional<SelectNodeChain> subchain,
                         DetermineBestSelectSubchain(chain, query_engine));
    if (subchain.has_value()) {
      profitable_chains.push_back(*std::move(subchain));
    }
  }
  return profitable_chains;
}

absl::Status CollapseSelectNodeChain(const SelectNodeChain& chain) {
  Node* originating_node = chain.chain_nodes.front().node;

  VLOG(4) << absl::StreamFormat("Collapsing select chain rooted at %s:",
                                originating_node->ToString());
  if (VLOG_IS_ON(4)) {
    for (const SelectChainElement& s : chain.chain_nodes) {
      VLOG(4) << absl::StreamFormat("  %s", s.node->ToString());
    }
  }

  std::vector<Node*> cases;
  cases.reserve(chain.total_cases() + 1);

  Node* default_selector = nullptr;
  if (chain.final_default.has_value()) {
    // All the selectors may be simultaneously false, so we need to add a
    // "fall-through" case whose selector is true when all chunk selectors are
    // zero (AND of each chunk == 0).

    VLOG(4) << "All selectors may be false.";
    std::vector<Node*> selector_equals_zero_vec;
    selector_equals_zero_vec.reserve(chain.chain_nodes.size());
    for (const SelectChainElement& el : chain.chain_nodes) {
      XLS_ASSIGN_OR_RETURN(
          Node * zero,
          el.node->function_base()->MakeNode<Literal>(
              el.node->loc(), Value(UBits(0, el.selector->BitCountOrDie()))));
      XLS_ASSIGN_OR_RETURN(Node * is_zero,
                           el.node->function_base()->MakeNode<CompareOp>(
                               el.node->loc(), el.selector, zero, Op::kEq));
      selector_equals_zero_vec.push_back(is_zero);
    }
    if (selector_equals_zero_vec.size() == 1) {
      default_selector = selector_equals_zero_vec.front();
    } else {
      XLS_ASSIGN_OR_RETURN(
          default_selector,
          originating_node->function_base()->MakeNode<NaryOp>(
              originating_node->loc(), selector_equals_zero_vec, Op::kAnd));
    }
    cases.push_back(*chain.final_default);
  }

  // Add cases from leaf chunk to root chunk.
  for (auto it = chain.chain_nodes.rbegin(); it != chain.chain_nodes.rend();
       ++it) {
    cases.insert(cases.end(), it->cases.begin(), it->cases.end());
  }

  // Concat elements from root chunk (MSB) to leaf chunk to fallthrough (LSB).
  std::vector<Node*> concat_elements;
  concat_elements.reserve(chain.chain_nodes.size() + 1);
  for (const SelectChainElement& el : chain.chain_nodes) {
    concat_elements.push_back(el.selector);
  }
  if (default_selector != nullptr) {
    concat_elements.push_back(default_selector);
  }

  Node* ohs_selector;
  if (concat_elements.size() == 1) {
    ohs_selector = concat_elements.front();
  } else {
    XLS_ASSIGN_OR_RETURN(ohs_selector,
                         originating_node->function_base()->MakeNode<Concat>(
                             originating_node->loc(), concat_elements));
  }

  VLOG(4) << "Replacing select chain with one-hot-select.";
  XLS_RETURN_IF_ERROR(
      originating_node->ReplaceUsesWithNew<OneHotSelect>(ohs_selector, cases)
          .status());
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CollapseSelectChainsPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  auto query_engine =
      UnionQueryEngine::Of(StatelessQueryEngine(),
                           context.GetForwardingQueryEngine<BddQueryEngine>(f));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  // Get all applicable select node chains.
  XLS_ASSIGN_OR_RETURN(std::vector<SelectNodeChain> chains,
                       GetSelectNodeChains(f, context, query_engine));

  // Determine the best subchain to collapse for each chain and transform.
  XLS_ASSIGN_OR_RETURN(std::vector<SelectNodeChain> profitable_chains,
                       GetProfitableSelectNodeChains(chains, query_engine));
  bool modified = false;
  for (const SelectNodeChain& chain : profitable_chains) {
    XLS_RETURN_IF_ERROR(CollapseSelectNodeChain(chain));
    modified = true;
  }
  return modified;
}

}  // namespace xls
