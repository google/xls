// Copyright 2020 Google LLC
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

#ifndef XLS_PASSES_BDD_QUERY_ENGINE_H_
#define XLS_PASSES_BDD_QUERY_ENGINE_H_

#include <vector>

#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A query engine which uses binary decision diagrams (BDDs) to analyze an XLS
// function. BDDs provide sharp analysis of bits values and relationships
// between bit values in the function (relative to ternary abstract evaluation).
// The downside is that BDDs can be slow in general and exponentially slow in
// particular for some operations such as arithmetic and comparison
// operations. For this reason, these operations are generally excluded from the
// analysis.
class BddQueryEngine : public QueryEngine {
 public:
  // 'minterm_limit' is the maximum number of minterms to allow in a BDD
  // expression before truncating it. See BddFunction for details.
  static xabsl::StatusOr<std::unique_ptr<BddQueryEngine>> Run(
      Function* f, int64 minterm_limit = 0);

  bool IsTracked(Node* node) const override {
    return known_bits_.contains(node);
  }

  const Bits& GetKnownBits(Node* node) const override {
    return known_bits_.at(node);
  }
  const Bits& GetKnownBitsValues(Node* node) const override {
    return bits_values_.at(node);
  }

  bool AtMostOneTrue(absl::Span<BitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<BitLocation const> bits) const override;
  bool Implies(const BitLocation& a, const BitLocation& b) const override;
  absl::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<BitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  bool KnownEquals(const BitLocation& a, const BitLocation& b) const override;
  bool KnownNotEquals(const BitLocation& a,
                      const BitLocation& b) const override;

  // Returns the underlying BddFunction representing the XLS function.
  const BddFunction& bdd_function() const { return *bdd_function_; }

 private:
  explicit BddQueryEngine(int64 minterm_limit)
      : minterm_limit_(minterm_limit) {}

  // Returns the underlying BDD. This method is const, but queries on a BDD
  // generally mutate the object. We sneakily avoid conflicts with C++ const
  // because the BDD is only held indirectly via pointers.
  // TODO(meheff): Enable queries on a BDD with out mutating the BDD itself.
  BinaryDecisionDiagram& bdd() const { return bdd_function_->bdd(); }

  // Returns the BDD node associated with the given bit.
  BddNodeIndex GetBddNode(const BitLocation& location) const {
    return bdd_function_->GetBddNode(location.node, location.bit_index);
  }

  // A implies B  <=>  !(A && !B)
  bool Implies(const BddNodeIndex& a, const BddNodeIndex& b) const;

  // Returns true if the expression of the given BDD node exceeds the minterm
  // limit.
  // TODO(meheff): This should be part of the BDD itself where a query can be
  // performed and the BDD method returns a union of minterm limit exceeded or
  // the result of the query.
  bool ExceedsMintermLimit(BddNodeIndex node) const {
    return minterm_limit_ > 0 &&
           bdd().GetNode(node).minterm_count > minterm_limit_;
  }

  // The maximum number of minterms in expression in the BDD before truncating.
  int64 minterm_limit_;

  // Indicates the bits at the output of each node which have known values.
  absl::flat_hash_map<Node*, Bits> known_bits_;

  // Indicates the values of bits at the output of each node (if known)
  absl::flat_hash_map<Node*, Bits> bits_values_;

  std::unique_ptr<BddFunction> bdd_function_;
};

}  // namespace xls

#endif  // XLS_PASSES_BDD_QUERY_ENGINE_H_
