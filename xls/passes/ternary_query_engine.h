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

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/nodes.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A query engine which uses abstract evaluation of an XLS function using
// ternary logic (0, 1, and unknown value X). Ternary logic evaluation is fast
// and can expose statically known bit values in the function (known 0 or 1),
// but provides limited insight into relationships between bit values in the
// function (implications, equality, etc).
class TernaryQueryEngine : public QueryEngine {
 public:
  TernaryQueryEngine() {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  bool IsTracked(Node* node) const override {
    return known_bits_.contains(node);
  }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    if (!node->GetType()->IsBits()) {
      LeafTypeTree<absl::monostate> shape(node->GetType());
      return LeafTypeTree<Type*>(shape.type(), shape.leaf_types())
          .Map<TernaryVector>([](Type* type) -> TernaryVector {
            return TernaryVector(type->GetFlatBitCount(),
                                 TernaryValue::kUnknown);
          });
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
    return absl::nullopt;
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
