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

#include "xls/passes/merge_selector_pass.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

// Candidate node for merge select conversion.
struct MergeSelectorCandidate {
  Select* select;

  // The merge candidate's users which are themselves selects, use the candidate
  // as their selector, and are determined to profit from this transform.
  std::vector<Select*> child_selects;

  bool operator==(const MergeSelectorCandidate& other) const {
    return select == other.select && child_selects == other.child_selects;
  }
};

// Evaluates a given select node to determine if it should merge itself with
// its child selects. The candidate's cases must be fully known in order
// to consider merging.
absl::StatusOr<std::optional<MergeSelectorCandidate>> GetMergeSelectorCandidate(
    Select* candidate, const QueryEngine& query_engine) {
  for (Node* case_node : candidate->cases()) {
    if (!query_engine.KnownValue(case_node).has_value()) {
      VLOG(5) << "Skipping MergeSelectorCandidate " << candidate->GetName()
              << ", case " << case_node->GetName() << " is not known.";
      return std::nullopt;
    }
  }
  if (candidate->default_value().has_value() &&
      !query_engine.KnownValue(*candidate->default_value()).has_value()) {
    VLOG(5) << "Skipping MergeSelectorCandidate " << candidate->GetName()
            << ", default value is not known.";
    return std::nullopt;
  }

  std::vector<Select*> child_selects;
  child_selects.reserve(candidate->users().size());
  int64_t parent_selector_bit_count = candidate->selector()->BitCountOrDie();

  // Filter on select users of this candidate which are profitable to merge.
  for (Node* user : candidate->users()) {
    if (!user->Is<Select>()) {
      continue;
    }
    Select* child_select = user->As<Select>();
    if (child_select->selector() != candidate) {
      continue;
    }

    int64_t child_selector_bit_count =
        child_select->selector()->BitCountOrDie();

    // We can't narrow the controlled selects any further unless the selector is
    // actually constant... which should be handled by other passes.
    if (child_selector_bit_count <= 1) {
      VLOG(5) << "Skipping MergeSelectorCandidate user "
              << child_select->GetName() << ", selector is trivial.";
      continue;
    }

    // We are not narrowing this child select; do not merge.
    if (child_selector_bit_count < parent_selector_bit_count) {
      VLOG(5) << "Skipping MergeSelectorCandidate user "
              << child_select->GetName() << ", parent selector is wider.";
      continue;
    }
    child_selects.push_back(child_select);
  }
  if (child_selects.empty()) {
    VLOG(5) << "Skipping MergeSelectorCandidate " << candidate->GetName()
            << " has no applicable child selects.";
    return std::nullopt;
  }

  VLOG(3) << "MergeSelectorCandidate " << candidate->GetName() << " has "
          << child_selects.size() << " child select(s):";
  if (VLOG_IS_ON(3)) {
    for (Select* child_select : child_selects) {
      VLOG(3) << "- " << child_select->ToString();
    }
  }

  return MergeSelectorCandidate{.select = candidate,
                                .child_selects = std::move(child_selects)};
}

// Returns the select's case node corresponding to the given selector value.
Node* GetCase(Select* select, const Bits& selector) {
  if (bits_ops::UGreaterThanOrEqual(selector, select->cases().size())) {
    CHECK(select->default_value().has_value());
    return *select->default_value();
  }
  absl::StatusOr<uint64_t> selector_value = selector.ToUint64();
  CHECK_OK(selector_value.status());
  return select->get_case(static_cast<int64_t>(*selector_value));
}

absl::StatusOr<bool> MergeSelects(const MergeSelectorCandidate& candidate,
                                  const QueryEngine& query_engine) {
  if (candidate.select->IsDead()) {
    return false;
  }

  std::vector<Bits> new_case_sequence;
  new_case_sequence.reserve(candidate.select->cases().size());
  for (Node* case_node : candidate.select->cases()) {
    std::optional<Value> known_value = query_engine.KnownValue(case_node);
    CHECK(known_value.has_value());
    new_case_sequence.push_back(known_value->bits());
  }

  std::optional<Bits> default_value = std::nullopt;
  if (candidate.select->default_value().has_value()) {
    std::optional<Value> known_value =
        query_engine.KnownValue(*candidate.select->default_value());
    CHECK(known_value.has_value());
    default_value = known_value.value().bits();
  }

  for (Select* user : candidate.child_selects) {
    if (user->IsDead() || user->selector() != candidate.select) {
      continue;
    }
    absl::FixedArray<Node*> new_cases(new_case_sequence.size());
    for (int64_t i = 0; i < new_cases.size(); ++i) {
      new_cases[i] = GetCase(user, new_case_sequence[i]);
    }

    std::optional<Node*> default_case =
        default_value.has_value()
            ? std::make_optional(GetCase(user, default_value.value()))
            : std::nullopt;
    XLS_ASSIGN_OR_RETURN(Node * new_select, user->ReplaceUsesWithNew<Select>(
                                                candidate.select->selector(),
                                                new_cases, default_case));
    VLOG(3) << "Replaced " << user->GetName()
            << " with: " << new_select->ToString();
  }
  return true;
}

}  // namespace

RedundancyGuard MergeSelectorPass::GetRedundancyGuard(
    const OptimizationPassOptions& options,
    OptimizationContext& context) const {
  return RedundancyGuard::CanSkip(absl::StrFormat("O%d", options.opt_level));
}

absl::StatusOr<bool> MergeSelectorPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  if (!options.narrowing_enabled()) {
    return false;
  }

  auto query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      GetSharedQueryEngine<LazyTernaryQueryEngine>(context, func));
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  // By running in reverse topological order, the analyses will stay valid for
  // all nodes we're considering through the full pass.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> reverse_topo_sort_nodes,
                       context.ReverseTopoSort(func));

  std::vector<MergeSelectorCandidate> candidates;
  for (Node* node : reverse_topo_sort_nodes) {
    if (node->IsDead()) {
      continue;
    }
    if (!node->Is<Select>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        std::optional<MergeSelectorCandidate> candidate,
        GetMergeSelectorCandidate(node->As<Select>(), query_engine));
    if (candidate.has_value()) {
      candidates.push_back(std::move(candidate.value()));
    }
  }

  bool changed = false;

  for (const MergeSelectorCandidate& candidate : candidates) {
    XLS_ASSIGN_OR_RETURN(bool changed_at_node,
                         MergeSelects(candidate, query_engine));
    changed = changed || changed_at_node;
  }

  return changed;
}

}  // namespace xls
