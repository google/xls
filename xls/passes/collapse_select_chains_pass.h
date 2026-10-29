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

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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

  friend bool operator==(const SelectChainElement& lhs,
                         const SelectChainElement& rhs) = default;
};

struct SelectNodeChain {
  std::vector<SelectChainElement> chain_nodes;

  // Indicates the presence of a default node that will be reached when all
  // selectors in the chain are false.
  std::optional<Node*> final_default;

  friend bool operator==(const SelectNodeChain& lhs,
                         const SelectNodeChain& rhs) = default;
};

// A view into a single contiguous subchain of a SelectNodeChain.
// The no-transform case is represented by a nullopt subchain_span.
class CollapseSelectSubspan {
 public:
  // Constructs a no-op subspan for `chain`. Does not take ownership of
  // `chain`, which must refer to a non-empty `SelectNodeChain` that outlives
  // this object.
  explicit CollapseSelectSubspan(const SelectNodeChain* absl_nonnull chain
                                     ABSL_ATTRIBUTE_LIFETIME_BOUND,
                                 std::nullopt_t = std::nullopt);

  // Constructs a subspan viewing `subchain_span`, which must be a non-empty
  // contiguous slice of `chain->chain_nodes`. Does not take ownership of
  // `chain` or `subchain_span`'s underlying elements, which must outlive this
  // object.
  explicit CollapseSelectSubspan(
      const SelectNodeChain* absl_nonnull chain ABSL_ATTRIBUTE_LIFETIME_BOUND,
      absl::Span<const SelectChainElement> subchain_span
          ABSL_ATTRIBUTE_LIFETIME_BOUND,
      std::optional<Node*> final_default = std::nullopt);

  CollapseSelectSubspan(const CollapseSelectSubspan&) = default;
  CollapseSelectSubspan& operator=(const CollapseSelectSubspan&) = default;

  bool IsNoopCase() const { return subchain_span_.empty(); }

  const SelectNodeChain& chain() const { return *chain_; }

  absl::Span<const SelectChainElement> subchain_span() const {
    return subchain_span_;
  }

  std::optional<Node*> final_default() const { return final_default_; }

  // Returns the root (first) node of the parent SelectNodeChain,
  // regardless of where `subchain_span()` starts or whether `IsNoopCase()` is
  // true.
  Node* GetChainRootNode() const { return chain_->chain_nodes.front().node; }

  // Creates a new SelectNodeChain corresponding to the subspan.
  // Returns an error if the subchain span is not set (i.e. the no-transform
  // case).
  absl::StatusOr<SelectNodeChain> GetSubchain() const;

  // Two CollapseSelectSubspans are equal if they have equal parent chains,
  // subchain spans, and final defaults.
  friend bool operator==(const CollapseSelectSubspan& lhs,
                         const CollapseSelectSubspan& rhs);

 private:
  const SelectNodeChain* absl_nonnull chain_;
  absl::Span<const SelectChainElement> subchain_span_;
  std::optional<Node*> final_default_;
};

// Obtains all select node chains in the given function, sorted by root node
// ID in ascending order. Chains may not necessarily be fully one-hot.
absl::StatusOr<std::vector<SelectNodeChain>> GetSelectNodeChains(
    FunctionBase* fb, OptimizationContext& context,
    const QueryEngine& query_engine);

// Obtains all valid contiguous node subchains of the given select chain,
// with the no-transform case as the first element. Does not take ownership of
// `chain`, which must outlive the returned subspans.
absl::StatusOr<std::vector<CollapseSelectSubspan>>
GetValidCollapseSelectSubspans(const SelectNodeChain* absl_nonnull chain
                                   ABSL_ATTRIBUTE_LIFETIME_BOUND,
                               const QueryEngine& query_engine);

// Returns the best subchain of the given chain to collapse based on heuristic
// profitability, or a no-transform CollapseSelectSubspan if no subchain is
// profitable to collapse. Currently this is determined to be the largest
// connected sequence of cases, breaking ties in favor of subchains closer to
// the root (first) node in the chain. Does not take ownership of `chain`,
// which must outlive the returned subspan.
absl::StatusOr<CollapseSelectSubspan> DetermineBestSelectSubchain(
    const SelectNodeChain* absl_nonnull chain ABSL_ATTRIBUTE_LIFETIME_BOUND,
    const QueryEngine& query_engine);

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
//
// Note: What follows was generated by the Gemini LLM. Not human verified.
//
// The `CollapseSelectChainsPass` identifies chains of select operations whose
// selector bits are mutually exclusive and collapses them into a single
// `one_hot_sel` operation, reducing logic depth and mux delay.
//
// **How it Works**:
//
// 1.  **Query Engine Setup**: The pass uses a `UnionQueryEngine` combining a
//     `StatelessQueryEngine` and a `BddQueryEngine` to reason about Boolean
//     relationships (such as mutual exclusivity and exhaustiveness) among
//     selector bits.
//
// 2.  **Chain Discovery (`GetSelectNodeChains`)**: The pass traverses the
//     function in reverse topological order (`ReverseTopoSort`) to build chains
//     of `bits`-typed select operations (`IsChainableSelect`, sorted by root
//     node ID in ascending order) where the fall-through branch of one select
//     feeds into another:
//     *   **Binary `sel`**: Either a 2-case `sel` without a default (where
//         `case[1]` is the selected value when `selector == 1` and `case[0]` is
//         the fall-through) or a 1-case `sel` with a 1-bit selector and a
//         `default` (where `default` is selected when `selector == 1` and
//         `case[0]` is the fall-through).
//     *   **`priority_sel`**: Contributes all of its `cases` (selected by bits
//         `0..N-1` of its selector) and chains through its `default_value`.
//     *   **`one_hot_sel`**: Contributes all of its `cases` and terminates the
//         chain (with no `final_default`).
//
// 3.  **Subchain Selection and Profitability (`DetermineBestSelectSubchain`)**:
//     *   Searches from the root of each chain toward the leaf across
//         contiguous subchains whose selector bit locations (`TreeBitLocation`)
//         satisfy `query_engine.AtMostOneTrue(...)`, selecting the mutually
//         exclusive subchain with the **most total cases** (breaking ties in
//         favor of subchains closer to the root).
//     *   **Length Threshold**: To avoid interfering with other select
//         optimizations on small mux trees, a subchain is only collapsed if it
//         contains **more than 4 total active cases** (`total_cases > 4`, i.e.,
//         at least 5 cases) and has at least 2 select nodes (`subchain_length >
//         1`).
//     *   **Default Computation and Elimination**: If the selected subchain
//         ends before the leaf of the full chain, the next select node in the
//         chain becomes the subchain's `final_default`. If the subchain has a
//         `final_default` and `query_engine.AtLeastOneTrue(...)` proves that at
//         least one selector bit in the subchain is always active, the
//         unreachable `final_default` is dropped.
//
// 4.  **Collapsing to `one_hot_sel` (`CollapseSelectNodeChain`)**:
//     *   If `final_default` may still be reached (all selector bits can be
//         simultaneously zero), a default selector bit is constructed as the
//         conjunction (`and`) of `eq(selector, 0)` across all selects in the
//         subchain, and `final_default` is placed as `case[0]` (the LSB case).
//     *   Active cases are ordered from the leaf of the subchain to the root.
//     *   The new selector is formed by concatenating the subchain selectors
//         from root (MSB) to leaf, followed by the default selector bit (LSB)
//         if present, and the root select is replaced with `one_hot_sel`.
//
// Example (5 mutually exclusive predicates `p0`..`p4` where at most one is 1):
//
// ```
// s0: bits[8] = sel(p0, cases=[d, v0])
// s1: bits[8] = sel(p1, cases=[s0, v1])
// s2: bits[8] = sel(p2, cases=[s1, v2])
// s3: bits[8] = sel(p3, cases=[s2, v3])
// ret s4: bits[8] = sel(p4, cases=[s3, v4])
// ```
//
// becomes:
//
// ```
// z0: bits[1] = eq(p0, literal(0))
// z1: bits[1] = eq(p1, literal(0))
// z2: bits[1] = eq(p2, literal(0))
// z3: bits[1] = eq(p3, literal(0))
// z4: bits[1] = eq(p4, literal(0))
// all_zero: bits[1] = and(z4, z3, z2, z1, z0)
// sel_bits: bits[6] = concat(p4, p3, p2, p1, p0, all_zero)
// ret s4: bits[8] = one_hot_sel(sel_bits, cases=[d, v0, v1, v2, v3, v4])
// ```
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
