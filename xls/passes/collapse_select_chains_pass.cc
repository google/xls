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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
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

// Minimum number of cases in a select chain to be eligible for collapsing.
// TODO(meheff): 2021/12/23 Consider tuning this value.
inline constexpr int64_t kMinChainCasesToCollapse = 4;

int64_t TotalCases(absl::Span<const SelectChainElement> chain_nodes) {
  int64_t total = 0;
  for (const SelectChainElement& element : chain_nodes) {
    total += element.cases.size();
  }
  return total;
}

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

}  // namespace

CollapseSelectSubspan::CollapseSelectSubspan(
    const SelectNodeChain* absl_nonnull chain, std::nullopt_t)
    : chain_(ABSL_DIE_IF_NULL(chain)),
      subchain_span_({}),
      final_default_(std::nullopt) {
  CHECK(!chain_->chain_nodes.empty());
}

CollapseSelectSubspan::CollapseSelectSubspan(
    const SelectNodeChain* absl_nonnull chain,
    absl::Span<const SelectChainElement> subchain_span,
    std::optional<Node*> final_default)
    : chain_(ABSL_DIE_IF_NULL(chain)),
      subchain_span_(subchain_span),
      final_default_(final_default) {
  CHECK(!chain_->chain_nodes.empty());
  CHECK(!subchain_span_.empty());
  const SelectChainElement* chain_begin = chain_->chain_nodes.data();
  const SelectChainElement* chain_end =
      chain_begin + chain_->chain_nodes.size();
  const SelectChainElement* span_begin = subchain_span_.data();
  CHECK(std::less_equal<const SelectChainElement*>{}(chain_begin, span_begin));
  CHECK(std::less<const SelectChainElement*>{}(span_begin, chain_end));
  CHECK_LE(static_cast<int64_t>(subchain_span_.size()), chain_end - span_begin);
}

absl::StatusOr<SelectNodeChain> CollapseSelectSubspan::GetSubchain() const {
  if (subchain_span_.empty()) {
    return absl::InvalidArgumentError(
        "GetSubchain called without a valid subchain span.");
  }

  std::vector<SelectChainElement> subchain_nodes(subchain_span_.begin(),
                                                 subchain_span_.end());

  return SelectNodeChain{.chain_nodes = std::move(subchain_nodes),
                         .final_default = final_default_};
}

bool operator==(const CollapseSelectSubspan& lhs,
                const CollapseSelectSubspan& rhs) {
  return lhs.final_default_ == rhs.final_default_ &&
         *lhs.chain_ == *rhs.chain_ && lhs.subchain_span_ == rhs.subchain_span_;
}

absl::StatusOr<std::vector<CollapseSelectSubspan>>
GetValidCollapseSelectSubspans(const SelectNodeChain* absl_nonnull chain
                                   ABSL_ATTRIBUTE_LIFETIME_BOUND,
                               const QueryEngine& query_engine) {
  XLS_RET_CHECK(!chain->chain_nodes.empty());

  const int64_t num_nodes = chain->chain_nodes.size();
  std::vector<CollapseSelectSubspan> subspans;

  // Max possible number of subspans is the triangle number sequence + 1 for
  // the empty subspan, e.g. 1, 3, 6, 10, 15... + 1
  subspans.reserve((num_nodes * (num_nodes + 1) / 2) + 1);

  // The no-transform case should be the first element.
  subspans.emplace_back(chain, std::nullopt);

  std::vector<int64_t> bit_offsets;
  bit_offsets.reserve(num_nodes + 1);
  bit_offsets.push_back(0);
  for (const SelectChainElement& el : chain->chain_nodes) {
    bit_offsets.push_back(bit_offsets.back() + el.bit_locations.size());
  }

  std::vector<TreeBitLocation> all_bit_locations;
  all_bit_locations.reserve(bit_offsets.back());
  for (const SelectChainElement& el : chain->chain_nodes) {
    all_bit_locations.insert(all_bit_locations.end(), el.bit_locations.begin(),
                             el.bit_locations.end());
  }
  absl::Span<const TreeBitLocation> all_bits_span =
      absl::MakeConstSpan(all_bit_locations);

  int64_t valid_end = 0;
  for (int64_t start = 0; start < num_nodes; ++start) {
    valid_end = std::max(valid_end, start);
    while (valid_end < num_nodes) {
      absl::Span<const TreeBitLocation> candidate_bits = all_bits_span.subspan(
          bit_offsets[start], bit_offsets[valid_end + 1] - bit_offsets[start]);
      // If we are no longer one-hot then we cannot create additional subspans
      // using this start node.
      if (!query_engine.AtMostOneTrue(candidate_bits)) {
        break;
      }
      ++valid_end;
    }

    bool at_least_one_true = false;
    for (int64_t end = start; end < valid_end; ++end) {
      // A single OneHotSelect is already collapsed.
      if (end == start && chain->chain_nodes[start].node->Is<OneHotSelect>()) {
        continue;
      }

      absl::Span<const TreeBitLocation> subspan_bits = all_bits_span.subspan(
          bit_offsets[start], bit_offsets[end + 1] - bit_offsets[start]);

      std::optional<Node*> final_default;
      if (end + 1 < num_nodes) {
        final_default = chain->chain_nodes[end + 1].node;
      } else {
        final_default = chain->final_default;
      }

      // If the all-zero case is impossible then we cannot fall through to the
      // default.
      if (final_default.has_value()) {
        if (!at_least_one_true && query_engine.AtLeastOneTrue(subspan_bits)) {
          at_least_one_true = true;
        }
        if (at_least_one_true) {
          final_default = std::nullopt;
        }
      }

      subspans.emplace_back(
          chain, absl::MakeSpan(&chain->chain_nodes[start], end - start + 1),
          final_default);
    }
  }
  return subspans;
}

// TODO(joshuata): Expand to return multiple potential subspans.
absl::StatusOr<CollapseSelectSubspan> DetermineBestSelectSubchain(
    const SelectNodeChain* absl_nonnull chain ABSL_ATTRIBUTE_LIFETIME_BOUND,
    const QueryEngine& query_engine) {
  if (!chain->chain_nodes.empty() &&
      TotalCases(chain->chain_nodes) <= kMinChainCasesToCollapse) {
    return CollapseSelectSubspan(chain, std::nullopt);
  }
  XLS_ASSIGN_OR_RETURN(std::vector<CollapseSelectSubspan> subspans,
                       GetValidCollapseSelectSubspans(chain, query_engine));

  const CollapseSelectSubspan* best_subspan = nullptr;
  int64_t best_total_cases = 0;

  // Search from the root of the chain toward the leaf to find the contiguous
  // compatible subchain with the most cases.
  for (const CollapseSelectSubspan& subspan : subspans) {
    if (subspan.IsNoopCase()) {
      continue;
    }
    int64_t current_cases = TotalCases(subspan.subchain_span());
    const bool same_start_longer_span =
        best_subspan != nullptr &&
        subspan.subchain_span().data() ==
            best_subspan->subchain_span().data() &&
        subspan.subchain_span().size() > best_subspan->subchain_span().size();
    if (current_cases > best_total_cases ||
        (current_cases == best_total_cases && same_start_longer_span)) {
      best_subspan = &subspan;
      best_total_cases = current_cases;
    }
  }

  // Only transform if the select chain is sufficiently long to avoid
  // interfering with select optimizations as plain selects are generally
  // easier to analyse/transform.
  if (best_total_cases <= kMinChainCasesToCollapse || best_subspan == nullptr) {
    return CollapseSelectSubspan(chain, std::nullopt);
  }

  return *best_subspan;
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

  absl::c_sort(chains, [](const SelectNodeChain& a, const SelectNodeChain& b) {
    return a.chain_nodes[0].node->id() < b.chain_nodes[0].node->id();
  });

  return chains;
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
  cases.reserve(TotalCases(chain.chain_nodes) + 1);

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

absl::StatusOr<bool> CollapseSelectChainsPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  UnionQueryEngine query_engine =
      UnionQueryEngine::Of(StatelessQueryEngine(),
                           context.GetForwardingQueryEngine<BddQueryEngine>(f));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  // Get all applicable select node chains.
  XLS_ASSIGN_OR_RETURN(std::vector<SelectNodeChain> chains,
                       GetSelectNodeChains(f, context, query_engine));

  // Determine the best subchain to collapse for each chain.
  std::vector<SelectNodeChain> profitable_chains;
  profitable_chains.reserve(chains.size());

  for (const SelectNodeChain& chain : chains) {
    XLS_ASSIGN_OR_RETURN(CollapseSelectSubspan subspan,
                         DetermineBestSelectSubchain(&chain, query_engine));
    if (!subspan.IsNoopCase()) {
      XLS_ASSIGN_OR_RETURN(SelectNodeChain profitable_subchain,
                           subspan.GetSubchain());
      profitable_chains.push_back(std::move(profitable_subchain));
    }
  }

  // Perform transforms.
  bool modified = false;
  for (const SelectNodeChain& chain : profitable_chains) {
    XLS_RETURN_IF_ERROR(CollapseSelectNodeChain(chain));
    modified = true;
  }
  return modified;
}

}  // namespace xls
