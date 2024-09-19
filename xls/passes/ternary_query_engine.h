// Copyright 2020 The XLS Authors
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

#ifndef XLS_PASSES_TERNARY_QUERY_ENGINE_H_
#define XLS_PASSES_TERNARY_QUERY_ENGINE_H_

#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls {

// A query engine which uses abstract evaluation of an XLS function using
// ternary logic (0, 1, and unknown value X). Ternary logic evaluation is fast
// and can expose statically known bit values in the function (known 0 or 1),
// but provides limited insight into relationships between bit values in the
// function (implications, equality, etc).
class TernaryQueryEngine : public QueryEngine {
 public:
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  bool IsTracked(Node* node) const override {
    return values_.contains(node) && values_.at(node).type() == node->GetType();
  }

  std::optional<LeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    return LeafTypeTree<TernaryVector>(node->GetType(),
                                       GetTernaryView(node).elements());
  }
  LeafTypeTreeView<TernaryVector> GetTernaryView(Node* node) const {
    CHECK(IsTracked(node)) << node;
    return values_.at(node).AsView();
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;

  // Ternary logic provides little information about bit implications.
  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return false;
  }

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }

  bool IsFullyKnown(Node* n) const override {
    if (!IsTracked(n) || TypeHasToken(n->GetType())) {
      return false;
    }
    return absl::c_all_of(values_.at(n).AsView().elements(),
                          [](const TernaryVector& tv) -> bool {
                            return ternary_ops::IsFullyKnown(tv);
                          });
  }

 private:
  // Holds which bits values are known for nodes in the function.
  absl::flat_hash_map<Node*, LeafTypeTree<TernaryEvaluator::Vector>> values_;
};

}  // namespace xls

#endif  // XLS_PASSES_TERNARY_QUERY_ENGINE_H_
