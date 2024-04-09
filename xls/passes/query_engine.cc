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

#include "xls/passes/query_engine.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/predicate_state.h"

namespace xls {
namespace {

// Converts the bits of the given node into a vector of BitLocations.
std::vector<TreeBitLocation> ToTreeBitLocations(Node* node) {
  CHECK(node->GetType()->IsBits());
  std::vector<TreeBitLocation> locations;
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    locations.emplace_back(TreeBitLocation(node, i));
  }
  return locations;
}

// Converts the single-bit Nodes in preds into a vector of BitLocations. Each
// element in preds must be a single-bit bits-typed Node.
std::vector<TreeBitLocation> ToTreeBitLocations(absl::Span<Node* const> preds) {
  std::vector<TreeBitLocation> locations;
  for (Node* pred : preds) {
    CHECK(pred->GetType()->IsBits());
    CHECK_EQ(pred->BitCountOrDie(), 1);
    locations.emplace_back(TreeBitLocation(pred, 0));
  }
  return locations;
}

}  // namespace

LeafTypeTree<IntervalSet> QueryEngine::GetIntervals(Node* node) const {
  // How many non-trailing bits we want to consider when creating intervals from
  // a ternary. Each interval set will be made up of up to
  // `1 << kMaxTernaryIntervalBits` separate intervals.
  // "4" is arbitrary, but keeps the number of intervals from blowing up.
  constexpr int64_t kMaxTernaryIntervalBits = 4;
  LeafTypeTree<TernaryVector> tern = GetTernary(node);
  return leaf_type_tree::Map<IntervalSet, TernaryVector>(
      tern.AsView(), [](TernarySpan tv) -> IntervalSet {
        return interval_ops::FromTernary(
            tv, /*max_interval_bits=*/kMaxTernaryIntervalBits);
      });
}

bool QueryEngine::AtMostOneNodeTrue(absl::Span<Node* const> preds) const {
  return AtMostOneTrue(ToTreeBitLocations(preds));
}

bool QueryEngine::AtMostOneBitTrue(Node* node) const {
  return AtMostOneTrue(ToTreeBitLocations(node));
}

bool QueryEngine::AtLeastOneNodeTrue(absl::Span<Node* const> preds) const {
  return AtLeastOneTrue(ToTreeBitLocations(preds));
}

bool QueryEngine::AtLeastOneBitTrue(Node* node) const {
  return AtLeastOneTrue(ToTreeBitLocations(node));
}

bool QueryEngine::ExactlyOneBitTrue(Node* node) const {
  return AtLeastOneBitTrue(node) && AtMostOneBitTrue(node);
}

bool QueryEngine::IsKnown(const TreeBitLocation& bit) const {
  if (!IsTracked(bit.node())) {
    return false;
  }
  return GetTernary(bit.node()).Get(bit.tree_index())[bit.bit_index()] !=
         TernaryValue::kUnknown;
}

std::optional<bool> QueryEngine::KnownValue(const TreeBitLocation& bit) const {
  if (!IsTracked(bit.node())) {
    return std::nullopt;
  }

  switch (GetTernary(bit.node()).Get(bit.tree_index())[bit.bit_index()]) {
    case TernaryValue::kUnknown:
      return std::nullopt;
    case TernaryValue::kKnownZero:
      return false;
    case TernaryValue::kKnownOne:
      return true;
  }

  ABSL_UNREACHABLE();
  return std::nullopt;
}

std::optional<Value> QueryEngine::KnownValue(Node* node) const {
  if (!IsTracked(node)) {
    return std::nullopt;
  }

  bool fully_known = true;
  LeafTypeTree<TernaryVector> ternary = GetTernary(node);
  LeafTypeTree<Value> value = leaf_type_tree::Map<Value, TernaryVector>(
      ternary.AsView(), [&fully_known](const TernaryVector& v) {
        if (!ternary_ops::IsFullyKnown(v)) {
          fully_known = false;
        }
        return Value(ternary_ops::ToKnownBitsValues(v));
      });
  if (!fully_known) {
    return std::nullopt;
  }
  absl::StatusOr<Value> result = LeafTypeTreeToValue(value.AsView());
  CHECK_OK(result.status());
  return *result;
}

std::optional<Bits> QueryEngine::KnownValueAsBits(Node* node) const {
  CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return std::nullopt;
  }

  TernaryVector ternary = GetTernary(node).Get({});
  if (!ternary_ops::IsFullyKnown(ternary)) {
    return std::nullopt;
  }
  return ternary_ops::ToKnownBitsValues(ternary);
}

bool QueryEngine::IsMsbKnown(Node* node) const {
  CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  TernaryVector ternary = GetTernary(node).Get({});
  return ternary_ops::ToKnownBits(ternary).msb();
}

bool QueryEngine::IsOne(const TreeBitLocation& bit) const {
  std::optional<bool> known_value = KnownValue(bit);
  if (!known_value.has_value()) {
    return false;
  }
  return *known_value;
}

bool QueryEngine::IsZero(const TreeBitLocation& bit) const {
  std::optional<bool> known_value = KnownValue(bit);
  if (!known_value.has_value()) {
    return false;
  }
  return !*known_value;
}

bool QueryEngine::GetKnownMsb(Node* node) const {
  CHECK(node->GetType()->IsBits());
  CHECK(IsMsbKnown(node));
  TernaryVector ternary = GetTernary(node).Get({});
  return ternary_ops::ToKnownBitsValues(ternary).msb();
}

bool QueryEngine::IsAllZeros(Node* node) const {
  CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  TernaryVector ternary = GetTernary(node).Get({});
  for (TernaryValue t : ternary) {
    if (t != TernaryValue::kKnownZero) {
      return false;
    }
  }
  return true;
}

bool QueryEngine::IsAllOnes(Node* node) const {
  CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  TernaryVector ternary = GetTernary(node).Get({});
  for (TernaryValue t : ternary) {
    if (t != TernaryValue::kKnownOne) {
      return false;
    }
  }
  return true;
}

bool QueryEngine::IsFullyKnown(Node* node) const {
  if (!IsTracked(node)) {
    return false;
  }

  LeafTypeTree<TernaryVector> ternary = GetTernary(node);
  return absl::c_all_of(ternary.elements(), [](const TernaryVector& v) {
    return ternary_ops::IsFullyKnown(v);
  });
}

Bits QueryEngine::MaxUnsignedValue(Node* node) const {
  CHECK(node->GetType()->IsBits());
  absl::InlinedVector<bool, 1> bits(node->BitCountOrDie());
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    bits[i] = IsZero(TreeBitLocation(node, i)) ? false : true;
  }
  return Bits(bits);
}

Bits QueryEngine::MinUnsignedValue(Node* node) const {
  CHECK(node->GetType()->IsBits());
  absl::InlinedVector<bool, 16> bits(node->BitCountOrDie());
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    bits[i] = IsOne(TreeBitLocation(node, i));
  }
  return Bits(bits);
}

bool QueryEngine::NodesKnownUnsignedNotEquals(Node* a, Node* b) const {
  CHECK(a->GetType()->IsBits());
  CHECK(b->GetType()->IsBits());
  int64_t max_width = std::max(a->BitCountOrDie(), b->BitCountOrDie());
  auto get_known_bit = [this](Node* n, int64_t index) {
    if (index >= n->BitCountOrDie()) {
      return TernaryValue::kKnownZero;
    }
    TreeBitLocation location(n, index);
    if (IsZero(location)) {
      return TernaryValue::kKnownZero;
    }
    if (IsOne(location)) {
      return TernaryValue::kKnownOne;
    }
    return TernaryValue::kUnknown;
  };

  for (int64_t i = 0; i < max_width; ++i) {
    TernaryValue a_bit = get_known_bit(a, i);
    TernaryValue b_bit = get_known_bit(b, i);
    if (a_bit != b_bit && a_bit != TernaryValue::kUnknown &&
        b_bit != TernaryValue::kUnknown) {
      return true;
    }
  }
  return false;
}

bool QueryEngine::NodesKnownUnsignedEquals(Node* a, Node* b) const {
  CHECK(a->GetType()->IsBits());
  CHECK(b->GetType()->IsBits());
  TernaryVector a_ternary = GetTernary(a).Get({});
  TernaryVector b_ternary = GetTernary(b).Get({});
  return a == b ||
         (AllBitsKnown(a) && AllBitsKnown(b) &&
          bits_ops::UEqual(ternary_ops::ToKnownBitsValues(a_ternary),
                           ternary_ops::ToKnownBitsValues(b_ternary)));
}

std::string QueryEngine::ToString(Node* node) const {
  CHECK(IsTracked(node));
  if (node->GetType()->IsBits()) {
    return xls::ToString(GetTernary(node).Get({}));
  }
  return GetTernary(node).ToString(
      [](const TernaryVector& v) -> std::string { return xls::ToString(v); });
}

// A forwarder for query engine.
class ForwardingQueryEngine final : public QueryEngine {
 public:
  explicit ForwardingQueryEngine(const QueryEngine& real) : real_(real) {}
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return absl::UnimplementedError("Cannot populate forwarding engine!");
  }

  bool IsTracked(Node* node) const override { return real_.IsTracked(node); }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    return real_.GetTernary(node);
  };

  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const override {
    return real_.SpecializeGivenPredicate(state);
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return real_.GetIntervals(node);
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return real_.AtMostOneTrue(bits);
  }

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return real_.AtLeastOneTrue(bits);
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return real_.Implies(a, b);
  }

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return real_.ImpliedNodeValue(predicate_bit_values, node);
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    return real_.KnownEquals(a, b);
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return real_.KnownNotEquals(a, b);
  }

 private:
  const QueryEngine& real_;
};

std::unique_ptr<QueryEngine> QueryEngine::SpecializeGivenPredicate(
    const absl::flat_hash_set<PredicateState>& state) const {
  return std::make_unique<ForwardingQueryEngine>(*this);
}

}  // namespace xls
