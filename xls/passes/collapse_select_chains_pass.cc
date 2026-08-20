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

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
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

// Collapse chain of selects with disjoint (one-hot or zero) selectors into a
// single one-hot-select.
absl::StatusOr<bool> CollapseSelectChains(FunctionBase* f,
                                          OptimizationContext& context,
                                          const QueryEngine& query_engine) {
  // A set containing the select instructions collapsed so far so we don't waste
  // time considering selects which have already been optimized.
  absl::flat_hash_set<Node*> collapsed_selects;
  bool modified = false;

  // Walk the graph in reverse order looking for chains of selects (Select,
  // PrioritySelect, OneHotSelect) where the fallthrough/default of one select
  // is another select.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> reverse_topo_sort_nodes,
                       context.ReverseTopoSort(f));
  for (Node* node : reverse_topo_sort_nodes) {
    if (!IsChainableSelect(node) || collapsed_selects.contains(node)) {
      continue;
    }
    struct SelectChunk {
      Node* selector;
      std::vector<Node*> cases;
    };

    std::vector<Node*> chain_nodes;
    std::vector<SelectChunk> chunks;
    std::vector<TreeBitLocation> bit_locations;
    std::optional<Node*> final_fallthrough = std::nullopt;

    Node* current = node;
    while (current != nullptr && IsChainableSelect(current) &&
           !collapsed_selects.contains(current)) {
      if (current->Is<Select>()) {
        Select* sel = current->As<Select>();
        chain_nodes.push_back(current);
        bit_locations.push_back(TreeBitLocation(sel->selector(), 0));
        if (sel->cases().size() == 2 && !sel->default_value().has_value()) {
          chunks.push_back(SelectChunk{
              .selector = sel->selector(),
              .cases = {sel->get_case(1)},
          });
          current = sel->get_case(0);
        } else {
          CHECK_EQ(sel->cases().size(), 1);
          CHECK(sel->default_value().has_value());
          chunks.push_back(SelectChunk{
              .selector = sel->selector(),
              .cases = {*sel->default_value()},
          });
          current = sel->get_case(0);
        }
        final_fallthrough = current;
      } else if (current->Is<PrioritySelect>()) {
        PrioritySelect* ps = current->As<PrioritySelect>();
        chain_nodes.push_back(current);
        std::vector<Node*> chunk_cases;
        chunk_cases.reserve(ps->cases().size());
        for (int64_t i = 0; i < ps->cases().size(); ++i) {
          chunk_cases.push_back(ps->get_case(i));
          bit_locations.push_back(TreeBitLocation(ps->selector(), i));
        }
        chunks.push_back(SelectChunk{
            .selector = ps->selector(),
            .cases = std::move(chunk_cases),
        });
        current = ps->default_value();
        final_fallthrough = current;
      } else if (current->Is<OneHotSelect>()) {
        OneHotSelect* ohs = current->As<OneHotSelect>();
        chain_nodes.push_back(current);
        std::vector<Node*> chunk_cases;
        chunk_cases.reserve(ohs->cases().size());
        for (int64_t i = 0; i < ohs->cases().size(); ++i) {
          chunk_cases.push_back(ohs->get_case(i));
          bit_locations.push_back(TreeBitLocation(ohs->selector(), i));
        }
        chunks.push_back(SelectChunk{
            .selector = ohs->selector(),
            .cases = std::move(chunk_cases),
        });
        final_fallthrough = std::nullopt;
        current = nullptr;
      } else {
        break;
      }
    }

    if (chain_nodes.size() == 1 && node->Is<OneHotSelect>()) {
      continue;
    }
    int64_t total_cases = 0;
    for (const auto& chunk : chunks) {
      total_cases += chunk.cases.size();
    }
    // Only transform if the select chain is sufficiently long to avoid
    // interfering with select optimizations as plain selects are generally
    // easier to analyse/transform.
    // TODO(meheff): 2021/12/23 Consider tuning this value.
    if (total_cases <= 4) {
      continue;
    }
    VLOG(4) << absl::StreamFormat("Considering select chain rooted at %s:",
                                  node->ToString());
    if (VLOG_IS_ON(4)) {
      for (Node* s : chain_nodes) {
        VLOG(4) << absl::StreamFormat("  %s", s->ToString());
      }
    }

    if (!query_engine.AtMostOneTrue(bit_locations)) {
      VLOG(4) << "Cannot collapse: more than one selector may be true.";
      continue;
    }

    std::vector<Node*> cases;
    cases.reserve(total_cases + 1);

    Node* fallthrough_selector = nullptr;
    if (final_fallthrough.has_value() &&
        !query_engine.AtLeastOneTrue(bit_locations)) {
      // All the selectors may be simultaneously false, so we need to add a
      // "fall-through" case whose selector is true when all chunk selectors are
      // zero (AND of each chunk == 0).
      VLOG(4) << "All selectors may be false.";
      std::vector<Node*> chunk_equals_zero;
      chunk_equals_zero.reserve(chunks.size());
      for (const auto& chunk : chunks) {
        XLS_ASSIGN_OR_RETURN(
            Node * zero,
            node->function_base()->MakeNode<Literal>(
                node->loc(), Value(UBits(0, chunk.selector->BitCountOrDie()))));
        XLS_ASSIGN_OR_RETURN(Node * is_zero,
                             node->function_base()->MakeNode<CompareOp>(
                                 node->loc(), chunk.selector, zero, Op::kEq));
        chunk_equals_zero.push_back(is_zero);
      }
      if (chunk_equals_zero.size() == 1) {
        fallthrough_selector = chunk_equals_zero.front();
      } else {
        XLS_ASSIGN_OR_RETURN(fallthrough_selector,
                             node->function_base()->MakeNode<NaryOp>(
                                 node->loc(), chunk_equals_zero, Op::kAnd));
      }
      cases.push_back(*final_fallthrough);
    }

    // Add cases from leaf chunk to root chunk.
    for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
      cases.insert(cases.end(), it->cases.begin(), it->cases.end());
    }

    // Concat elements from root chunk (MSB) to leaf chunk to fallthrough (LSB).
    std::vector<Node*> concat_elements;
    concat_elements.reserve(chunks.size() + 1);
    for (const auto& chunk : chunks) {
      concat_elements.push_back(chunk.selector);
    }
    if (fallthrough_selector != nullptr) {
      concat_elements.push_back(fallthrough_selector);
    }

    Node* ohs_selector;
    if (concat_elements.size() == 1) {
      ohs_selector = concat_elements.front();
    } else {
      XLS_ASSIGN_OR_RETURN(ohs_selector,
                           node->function_base()->MakeNode<Concat>(
                               node->loc(), concat_elements));
    }

    VLOG(4) << "Replacing select chain with one-hot-select.";
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<OneHotSelect>(ohs_selector, cases).status());
    collapsed_selects.insert(chain_nodes.begin(), chain_nodes.end());
    modified = true;
  }
  return modified;
}

}  // namespace

absl::StatusOr<bool> CollapseSelectChainsPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  auto query_engine =
      UnionQueryEngine::Of(StatelessQueryEngine(),
                           context.GetForwardingQueryEngine<BddQueryEngine>(f));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());
  return CollapseSelectChains(f, context, query_engine);
}

}  // namespace xls
