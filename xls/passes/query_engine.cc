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
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/forwarding_query_engine.h"
#include "xls/passes/predicate_state.h"

namespace xls {
namespace {

// Converts the bits of the given node into a vector of BitLocations.
std::vector<TreeBitLocation> ToTreeBitLocations(Node* node) {
  CHECK(node->GetType()->IsBits());
  std::vector<TreeBitLocation> locations;
  locations.reserve(node->BitCountOrDie());
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
  std::optional<SharedLeafTypeTree<TernaryVector>> tern = GetTernary(node);
  if (!tern.has_value()) {
    return *LeafTypeTree<IntervalSet>::CreateFromFunction(
        node->GetType(), [](Type* leaf_type) -> absl::StatusOr<IntervalSet> {
          return IntervalSet::Maximal(leaf_type->GetFlatBitCount());
        });
  }
  return leaf_type_tree::Map<IntervalSet, TernaryVector>(
      tern->AsView(), [](TernarySpan tv) -> IntervalSet {
        return interval_ops::FromTernary(
            tv, /*max_interval_bits=*/kMaxTernaryIntervalBits);
      });
}

std::optional<TreeBitLocation> QueryEngine::ExactlyOneBitUnknown(
    Node* node) const {
  std::optional<TreeBitLocation> unknown;
  for (const TreeBitLocation& bit : ToTreeBitLocations(node)) {
    if (!IsKnown(bit)) {
      if (unknown.has_value()) {
        return std::nullopt;
      }
      unknown = bit;
    }
  }
  return unknown;
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
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
      GetTernary(bit.node());
  if (!ternary.has_value()) {
    return false;
  }
  return ternary->Get(bit.tree_index())[bit.bit_index()] !=
         TernaryValue::kUnknown;
}

std::optional<bool> QueryEngine::KnownValue(const TreeBitLocation& bit) const {
  if (!IsTracked(bit.node())) {
    return std::nullopt;
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
      GetTernary(bit.node());
  if (!ternary.has_value()) {
    return std::nullopt;
  }
  switch (ternary->Get(bit.tree_index())[bit.bit_index()]) {
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
  if (!IsTracked(node) || TypeHasToken(node->GetType())) {
    return std::nullopt;
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> ternary = GetTernary(node);
  if (!ternary.has_value() ||
      !absl::c_all_of(ternary->elements(), [](const TernaryVector& v) {
        return ternary_ops::IsFullyKnown(v);
      })) {
    return std::nullopt;
  }

  absl::StatusOr<LeafTypeTree<Value>> value =
      leaf_type_tree::MapIndex<Value, TernaryVector>(
          ternary->AsView(),
          [](Type* leaf_type, const TernaryVector& v,
             absl::Span<const int64_t>) -> absl::StatusOr<Value> {
            if (leaf_type->IsToken()) {
              return Value::Token();
            }
            CHECK(leaf_type->IsBits());
            return Value(ternary_ops::ToKnownBitsValues(v));
          });
  CHECK_OK(value.status());
  absl::StatusOr<Value> result = LeafTypeTreeToValue(value->AsView());
  CHECK_OK(result.status());
  return *result;
}

std::optional<Bits> QueryEngine::KnownValueAsBits(Node* node) const {
  CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return std::nullopt;
  }

  std::optional<Value> value = KnownValue(node);
  if (!value.has_value()) {
    return std::nullopt;
  }
  return value->bits();
}

bool QueryEngine::IsMsbKnown(Node* node) const {
  CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  if (node->BitCountOrDie() == 0) {
    // Zero-length is considered unknown.
    return false;
  }
  return IsKnown(TreeBitLocation(node, node->BitCountOrDie() - 1));
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
  return KnownValue(TreeBitLocation(node, node->BitCountOrDie() - 1)).value();
}

bool QueryEngine::IsAllZeros(Node* node) const {
  if (!IsTracked(node) || TypeHasToken(node->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(node);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsKnownZero(v);
         });
}

bool QueryEngine::IsAllOnes(Node* node) const {
  if (!IsTracked(node) || TypeHasToken(node->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(node);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsKnownOne(v);
         });
}

bool QueryEngine::IsFullyKnown(Node* node) const {
  if (!IsTracked(node) || TypeHasToken(node->GetType())) {
    return false;
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> ternary = GetTernary(node);
  return ternary.has_value() &&
         absl::c_all_of(ternary->elements(), [](const TernaryVector& v) {
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

std::optional<int64_t> QueryEngine::KnownLeadingZeros(Node* node) const {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary = GetTernary(node);
  if (!ternary) {
    return std::nullopt;
  }
  int64_t res = 0;
  for (TernaryValue v : iter::reversed(ternary->Get({}))) {
    if (v == TernaryValue::kKnownZero) {
      res++;
    } else {
      break;
    }
  }
  return res;
}

std::optional<int64_t> QueryEngine::KnownLeadingOnes(Node* node) const {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary = GetTernary(node);
  if (!ternary) {
    return std::nullopt;
  }
  int64_t res = 0;
  for (TernaryValue v : iter::reversed(ternary->Get({}))) {
    if (v == TernaryValue::kKnownOne) {
      res++;
    } else {
      break;
    }
  }
  return res;
}

std::optional<int64_t> QueryEngine::KnownLeadingSignBits(Node* node) const {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  if (node->GetType()->AsBitsOrDie()->bit_count() == 0) {
    // Zero bit values don't really have a sign bit in any meaningful sense.
    return std::nullopt;
  }
  int64_t res = 1;
  TreeBitLocation sign_bit_loc(node, node->BitCountOrDie() - 1);
  // First bit is exactly the sign bit so its always equal to itself.
  for (int64_t i = node->BitCountOrDie() - 2; i >= 0; --i) {
    if (KnownEquals(sign_bit_loc, TreeBitLocation(node, i))) {
      ++res;
    } else {
      break;
    }
  }
  return res;
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
  if (a == b) {
    return true;
  }
  std::optional<Bits> a_value = KnownValueAsBits(a);
  if (!a_value.has_value()) {
    return false;
  }
  std::optional<Bits> b_value = KnownValueAsBits(b);
  if (!b_value.has_value()) {
    return false;
  }
  return bits_ops::UEqual(*a_value, *b_value);
}

std::string QueryEngine::ToString(Node* node) const {
  CHECK(IsTracked(node));
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary = GetTernary(node);
  if (!ternary.has_value()) {
    ternary = LeafTypeTree<TernaryVector>::CreateFromFunction(
                  node->GetType(),
                  [](Type* leaf_type) -> absl::StatusOr<TernaryVector> {
                    return TernaryVector(leaf_type->GetFlatBitCount(),
                                         TernaryValue::kUnknown);
                  })
                  .value()
                  .AsShared();
  }
  if (node->GetType()->IsBits()) {
    return xls::ToString(ternary->Get({}));
  }
  return ternary->ToString(
      [](const TernaryVector& v) -> std::string { return xls::ToString(v); });
}

namespace {
class BaseForwardingQueryEngine final : public ForwardingQueryEngine {
 public:
  explicit BaseForwardingQueryEngine(const QueryEngine& qe) : qe_(qe) {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return absl::UnimplementedError(
        "Populate not supported on specialized query engines.");
  }

 protected:
  QueryEngine& real() final {
    LOG(FATAL) << "Mutable view not supported";
    ABSL_UNREACHABLE();
  }
  const QueryEngine& real() const final { return qe_; }

 private:
  const QueryEngine& qe_;
};

}  // namespace
std::unique_ptr<QueryEngine> QueryEngine::SpecializeGivenPredicate(
    const absl::btree_set<PredicateState>& state) const {
  return std::make_unique<BaseForwardingQueryEngine>(*this);
}

std::unique_ptr<QueryEngine> QueryEngine::SpecializeGiven(
    const absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan>& givens)
    const {
  return std::make_unique<BaseForwardingQueryEngine>(*this);
}

}  // namespace xls
