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

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A query engine which uses abstract evaluation of an XLS function using
// ternary logic (0, 1, and unknown value X). Ternary logic evaluation is fast
// and can expose statically known bit values in the function (known 0 or 1),
// but provides limited insight into relationships between bit values in the
// function (implications, equality, etc).
class TernaryQueryEngine : public QueryEngine {
 public:
  TernaryQueryEngine() = default;

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  bool IsTracked(Node* node) const override {
    return known_bits_.contains(node);
  }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    if (!node->GetType()->IsBits()) {
      return LeafTypeTree<TernaryVector>::CreateFromFunction(
                 node->GetType(),
                 [](Type* leaf_type, absl::Span<const int64_t> index) {
                   return TernaryVector(leaf_type->GetFlatBitCount(),
                                        TernaryValue::kUnknown);
                 })
          .value();
    }
    TernaryVector ternary =
        ternary_ops::FromKnownBits(known_bits_.at(node), bits_values_.at(node));
    LeafTypeTree<TernaryVector> result(node->GetType());
    result.Set({}, ternary);
    return result;
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

  bool IsFullyKnown(Node* n) const {
    return IsTracked(n) && known_bits_.at(n).IsAllOnes();
  }

 private:
  // Holds which bits values are known for nodes in the function. A one in a bit
  // position indications the respective bit value in the respective node is
  // statically known.
  absl::flat_hash_map<Node*, Bits> known_bits_;

  // Holds the values of statically known bits of nodes in the function.
  absl::flat_hash_map<Node*, Bits> bits_values_;
};

}  // namespace xls

#endif  // XLS_PASSES_TERNARY_QUERY_ENGINE_H_
