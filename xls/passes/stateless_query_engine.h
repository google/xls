// Copyright 2024 The XLS Authors
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

#ifndef XLS_PASSES_STATELESS_QUERY_ENGINE_H_
#define XLS_PASSES_STATELESS_QUERY_ENGINE_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A query engine which uses peephole (i.e., only immediate) context of an XLS
// node. This is simple and stateless; Populate() is always a no-op. This query
// engine can be used on any XLS node at any time with O(1) cost, though it's
// quite weak; it will most often report that the result is essentially unknown.
class StatelessQueryEngine : public QueryEngine {
 public:
  StatelessQueryEngine() = default;

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return ReachedFixpoint::Unchanged;
  }

  bool IsTracked(Node* node) const override { return true; }

  std::optional<Value> KnownValue(Node* node) const override;

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override;

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;

  // Returns true if at most/at least/exactly one of the bits in 'node' is true.
  // 'node' must be bits-typed.
  bool AtMostOneBitTrue(Node* node) const override;
  bool AtLeastOneBitTrue(Node* node) const override;
  bool ExactlyOneBitTrue(Node* node) const override;

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override;

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    if (!node->Is<Literal>()) {
      return std::nullopt;
    }
    Literal* literal = node->As<Literal>();
    if (!literal->value().IsBits()) {
      return std::nullopt;
    }
    return literal->value().bits();
  }

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    CHECK(node->GetType()->IsBits());
    return GetTernary(node).Get({});
  }

  bool IsFullyKnown(Node* n) const override;

 private:
  struct BitCounts {
    int64_t known_true = 0;
    int64_t known_false = 0;
  };
  BitCounts KnownBitCounts(absl::Span<TreeBitLocation const> bits) const;
};

}  // namespace xls

#endif  // XLS_PASSES_STATELESS_QUERY_ENGINE_H_
