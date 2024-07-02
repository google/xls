// Copyright 2023 The XLS Authors
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

#ifndef XLS_PASSES_PREDICATE_DOMINATOR_ANALYSIS_H_
#define XLS_PASSES_PREDICATE_DOMINATOR_ANALYSIS_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/predicate_state.h"

namespace xls {

// An analysis which keeps track of which select arms a given instruction is
// fully guarded by (that is which, if any, select arms all demands for the
// value flow from). This analysis only considers standard 'Select' nodes as
// predicates.
class PredicateDominatorAnalysis {
 public:
  // Move and copy constructors.
  PredicateDominatorAnalysis(const PredicateDominatorAnalysis&) = default;
  PredicateDominatorAnalysis& operator=(const PredicateDominatorAnalysis&) =
      default;
  PredicateDominatorAnalysis(PredicateDominatorAnalysis&&) = default;
  PredicateDominatorAnalysis& operator=(PredicateDominatorAnalysis&&) = default;

  // Execute this analysis and return results.
  static PredicateDominatorAnalysis Run(FunctionBase* f);

  // Returns a single element of the common predicate dominators which is
  // closest to the node (ie the last predicate which gates the use of this
  // value).
  // NB Extending this to return the full predicate-set would be fairly
  // straightforward and could enable better optimization by allowing each
  // narrow to choose the best predicate to use for its pass.
  PredicateState GetSingleNearestPredicate(Node* node) const {
    // TODO(google/xls#1104): Due to not tracking replaced nodes correctly we
    // might ask for the predicate of a node which is a replacement. Once we
    // track these we can answer correctly but until then we need to just
    // fallback to the base.
    if (nearest_predicates_.contains(node)) {
      return nearest_predicates_.at(node);
    }
    return PredicateState();
  }

 private:
  explicit PredicateDominatorAnalysis(
      absl::flat_hash_map<Node*, PredicateState> nearest_predicates)
      : nearest_predicates_(std::move(nearest_predicates)) {}

  absl::flat_hash_map<Node*, PredicateState> nearest_predicates_;
};

}  // namespace xls

#endif  // XLS_PASSES_PREDICATE_DOMINATOR_ANALYSIS_H_
