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

#include "xls/passes/stateless_query_engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/overload.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/query_engine.h"

namespace xls {

namespace {

// A Span - except that the underlying space might not have been allocated yet.
//
// Can be created from a Span, a non-null pointer to a single element, or a
// Span-returning initializer which will be called on demand. (If an
// initializer, the user must specify the size of the span that it will return.)
template <typename T>
class LazySpan {
 private:
  using SpanSource =
      std::variant<absl::Span<T>, absl::AnyInvocable<absl::Span<T>() &&>,
                   const LazySpan<T>*>;

 public:
  LazySpan(absl::Span<T> span)
      : pos_(0), len_(span.size()), span_source_(span) {}

  explicit LazySpan(T* absl_nonnull single_element)
      : LazySpan(absl::MakeSpan(single_element, 1)) {
    CHECK(single_element != nullptr);
  }

  LazySpan(absl::AnyInvocable<absl::Span<T>() &&> span_initializer, size_t size)
      : pos_(0),
        len_(size),
        span_source_(size > 0 ? SpanSource(std::move(span_initializer))
                              : SpanSource(absl::Span<T>())) {}

  LazySpan(const LazySpan<T>& other) : LazySpan(&other, 0, other.len_) {}
  LazySpan& operator=(const LazySpan<T>& other) {
    span_source_ = &other;
    pos_ = 0;
    len_ = other.len_;
    return *this;
  }

  LazySpan(LazySpan<T>&& other) = default;
  LazySpan& operator=(LazySpan<T>&& other) = default;

  // If `T` is not const-qualified, `LazySpan<T>` can be converted to
  // `LazySpan<const T>`, but not the other way around.
  operator LazySpan<const T>() const
    requires(!std::is_same_v<T, const T>)
  {
    return LazySpan<const T>(this, 0, len_);
  }

  absl::Span<T> Get() const {
    Populate();
    return std::get<absl::Span<T>>(span_source_);
  }

  operator absl::Span<T>() const { return Get(); }
  operator absl::Span<const T>() const
    requires(!std::is_same_v<T, const T>)
  {
    return Get();
  }

  bool empty() const { return len_ == 0; }
  int64_t size() const { return len_; }

  LazySpan<T> subspan(size_t pos, size_t len) const {
    CHECK_LT(pos, len_);
    CHECK_LE(pos + len, len_);
    return LazySpan<T>(this, pos, len);
  }
  LazySpan<T> subspan(size_t pos) const {
    return LazySpan<T>(this, pos, len_ - pos);
  }

 private:
  LazySpan(const LazySpan<T>* parent, size_t start, size_t size)
      : pos_(start), len_(size), span_source_(parent) {}

  void Populate() const {
    std::visit(
        absl::Overload(
            [](absl::Span<T> span) {},
            [this](absl::AnyInvocable<absl::Span<T>() &&>& span_initializer) {
              span_source_ = std::move(span_initializer)();
              CHECK_EQ(std::get<absl::Span<T>>(span_source_).size(), len_);
            },
            [this](const LazySpan<T>* parent) {
              span_source_ = parent->Get().subspan(pos_, len_);
              pos_ = 0;
            }),
        span_source_);
  }

  mutable size_t pos_;
  size_t len_;
  mutable SpanSource span_source_;
};

using LazyTernarySpan = LazySpan<TernaryValue>;

// Populates `out` with the ternary values for bits
// `[start_bit, start_bit + out.size())` of `node` (at `tree_index`).
// Returns true if any ternary information is known for `node` in that slice.
bool PopulateTernary(Node* node, absl::Span<const int64_t> tree_index,
                     int64_t start_bit, LazyTernarySpan out) {
  if (out.empty()) {
    // There's no information in an empty span.
    return false;
  }
  switch (node->op()) {
    case Op::kLiteral: {
      const Value* value = &node->As<Literal>()->value();
      for (int64_t idx : tree_index) {
        value = &value->element(idx);
      }
      CHECK(value->IsBits());
      const Bits& bits = value->bits();
      absl::Span<TernaryValue> span = out.Get();
      for (int64_t i = 0; i < span.size(); ++i) {
        span[i] = bits.Get(start_bit + i) ? TernaryValue::kKnownOne
                                          : TernaryValue::kKnownZero;
      }
      return true;
    }
    case Op::kConcat: {
      CHECK(tree_index.empty());
      int64_t end_bit = start_bit + out.size();
      int64_t op_lsb = 0;
      bool any_known = false;
      for (Node* operand : iter::reversed(node->operands())) {
        int64_t op_bits = operand->BitCountOrDie();
        int64_t op_msb = op_lsb + op_bits;
        int64_t slice_start = std::max(start_bit, op_lsb);
        int64_t slice_end = std::min(end_bit, op_msb);
        if (slice_start < slice_end) {
          any_known |= PopulateTernary(
              operand, /*tree_index=*/{}, slice_start - op_lsb,
              out.subspan(slice_start - start_bit, slice_end - slice_start));
        }
        op_lsb = op_msb;
      }
      return any_known;
    }
    case Op::kZeroExt: {
      CHECK(tree_index.empty());
      Node* operand = node->operand(0);
      int64_t op_bits = operand->BitCountOrDie();
      int64_t low_len = std::clamp<int64_t>(op_bits - start_bit, 0, out.size());
      bool any_known = false;
      if (low_len > 0) {
        any_known = PopulateTernary(operand, /*tree_index=*/{}, start_bit,
                                    out.subspan(0, low_len));
      }
      if (low_len < out.size()) {
        absl::Span<TernaryValue> span = out.Get().subspan(low_len);
        absl::c_fill(span, TernaryValue::kKnownZero);
        any_known = true;
      }
      return any_known;
    }
    case Op::kSignExt: {
      CHECK(tree_index.empty());
      Node* operand = node->operand(0);
      int64_t op_bits = operand->BitCountOrDie();
      if (op_bits == 0) {
        absl::Span<TernaryValue> span = out.Get();
        absl::c_fill(span, TernaryValue::kKnownZero);
        return true;
      }
      if (start_bit >= op_bits - 1) {
        TernaryValue sign = TernaryValue::kUnknown;
        if (!PopulateTernary(operand, /*tree_index=*/{}, op_bits - 1,
                             absl::MakeSpan(&sign, 1)) ||
            sign == TernaryValue::kUnknown) {
          return false;
        }
        absl::Span<TernaryValue> span = out.Get();
        absl::c_fill(span, sign);
        return true;
      }
      int64_t low_len = std::min<int64_t>(out.size(), op_bits - start_bit);
      if (!PopulateTernary(operand, /*tree_index=*/{}, start_bit,
                           out.subspan(0, low_len))) {
        return false;
      }
      if (out.size() > low_len) {
        absl::Span<TernaryValue> span = out.Get();
        TernaryValue sign = span[low_len - 1];
        absl::Span<TernaryValue> ext_span = span.subspan(low_len);
        absl::c_fill(ext_span, sign);
      }
      return true;
    }
    default:
      return false;
  }
}

}  // namespace

std::optional<bool> StatelessQueryEngine::KnownValue(
    const TreeBitLocation& bit) const {
  TernaryValue value = TernaryValue::kUnknown;
  if (!PopulateTernary(bit.node(), bit.tree_index(), bit.bit_index(),
                       LazyTernarySpan(&value))) {
    return std::nullopt;
  }
  switch (value) {
    case TernaryValue::kUnknown:
      return std::nullopt;
    case TernaryValue::kKnownZero:
      return false;
    case TernaryValue::kKnownOne:
      return true;
  }
  ABSL_UNREACHABLE();
}
std::optional<Value> StatelessQueryEngine::KnownValue(Node* node) const {
  if (!node->Is<Literal>()) {
    return QueryEngine::KnownValue(node);
  }
  return node->As<Literal>()->value();
}

bool StatelessQueryEngine::IsAllZeros(Node* node) const {
  if (node->Is<Literal>()) {
    return node->As<Literal>()->value().IsAllZeros();
  }
  return false;
}
bool StatelessQueryEngine::IsAllOnes(Node* node) const {
  if (node->Is<Literal>()) {
    return node->As<Literal>()->value().IsAllOnes();
  }
  return false;
}

std::optional<SharedTernaryTree> StatelessQueryEngine::GetTernary(
    Node* node) const {
  if (node->GetType()->IsBits()) {
    TernaryVector vec;
    if (!PopulateTernary(node, /*tree_index=*/{}, /*start_bit=*/0,
                         LazyTernarySpan(
                             [&vec, size = node->BitCountOrDie()]() {
                               vec.resize(size, TernaryValue::kUnknown);
                               return absl::MakeSpan(vec);
                             },
                             node->BitCountOrDie()))) {
      return std::nullopt;
    }
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(node->GetType(),
                                                                std::move(vec))
        .AsShared();
  }

  bool has_ternary_info = false;
  XLS_ASSIGN_OR_RETURN(
      auto ternary_tree,
      LeafTypeTree<TernaryVector>::CreateFromFunction(
          node->GetType(),
          [&](Type* leaf_type, absl::Span<const int64_t> tree_index)
              -> absl::StatusOr<TernaryVector> {
            TernaryVector vec;
            has_ternary_info |= PopulateTernary(
                node, tree_index, /*start_bit=*/0,
                LazyTernarySpan(
                    [&vec, bit_count = leaf_type->GetFlatBitCount()]() {
                      vec.resize(bit_count, TernaryValue::kUnknown);
                      return absl::MakeSpan(vec);
                    },
                    leaf_type->GetFlatBitCount()));
            return vec;
          }),
      /*error_expression=*/std::nullopt);
  if (!has_ternary_info) {
    return std::nullopt;
  }
  CHECK_OK(leaf_type_tree::ForEachIndex(
      ternary_tree.AsMutableView(),
      [](Type* leaf_type, TernaryVector& leaf_data, absl::Span<const int64_t>) {
        DCHECK(leaf_data.empty() ||
               leaf_data.size() == leaf_type->GetFlatBitCount());
        leaf_data.resize(leaf_type->GetFlatBitCount(), TernaryValue::kUnknown);
        return absl::OkStatus();
      }));
  return std::move(ternary_tree).AsShared();
}

StatelessQueryEngine::BitCounts StatelessQueryEngine::KnownBitCounts(
    absl::Span<TreeBitLocation const> bits) const {
  absl::flat_hash_map<Node*,
                      absl::flat_hash_map<TreeBitLocation, /*count=*/int64_t>>
      by_node;
  for (const TreeBitLocation& bit : bits) {
    by_node[bit.node()][bit]++;
  }

  BitCounts counts;
  for (const auto& [node, node_bits] : by_node) {
    if (node->Is<OneHot>()) {
      int64_t total_count = 0;
      int64_t min_count = std::numeric_limits<int64_t>::max();
      int64_t max_count = 0;
      for (const auto& [bit, count] : node_bits) {
        total_count += count;
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
      }
      // Exactly one output bit from this node is enabled, so all but one is
      // false; in the worst case, it's the one named the most times.
      counts.known_false += total_count - max_count;
      if (node_bits.size() == node->BitCountOrDie()) {
        // If every bit from this node is named, then one of the named bits is
        // true; in the worst case, it's the one named the fewest times.
        counts.known_true += min_count;
      }
      continue;
    }

    std::optional<SharedLeafTypeTree<TernaryVector>> ternary = GetTernary(node);
    if (!ternary.has_value()) {
      continue;
    }
    for (const auto& [bit, count] : node_bits) {
      switch (ternary->Get(bit.tree_index()).at(bit.bit_index())) {
        case TernaryValue::kKnownZero:
          counts.known_false += count;
          break;
        case TernaryValue::kKnownOne:
          counts.known_true += count;
          break;
        case TernaryValue::kUnknown:
          break;
      }
    }
  }
  return counts;
}

bool StatelessQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return bits.size() <= 1 ||
         KnownBitCounts(bits).known_false >= bits.size() - 1;
}

bool StatelessQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return !bits.empty() && KnownBitCounts(bits).known_true >= 1;
}

bool StatelessQueryEngine::KnownEquals(const TreeBitLocation& a,
                                       const TreeBitLocation& b) const {
  if (a == b) {
    return true;
  }

  if (a.node() == b.node() && a.node()->Is<OneHot>()) {
    // No two distinct bits from a OneHot node can be equal.
    return false;
  }

  std::optional<bool> a_value = KnownValue(a);
  if (!a_value.has_value()) {
    return false;
  }

  std::optional<bool> b_value = KnownValue(b);
  if (!b_value.has_value()) {
    return false;
  }

  return a_value == b_value;
}

bool StatelessQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                          const TreeBitLocation& b) const {
  if (a == b) {
    return false;
  }

  if (a.node() == b.node() && a.node()->Is<OneHot>()) {
    // No two distinct bits from a OneHot node can be equal.
    return true;
  }

  std::optional<bool> a_value = KnownValue(a);
  if (!a_value.has_value()) {
    return false;
  }

  std::optional<bool> b_value = KnownValue(b);
  if (!b_value.has_value()) {
    return false;
  }

  return a_value != b_value;
}

bool StatelessQueryEngine::AtMostOneBitTrue(Node* node) const {
  if (node->Is<OneHot>()) {
    return true;
  }
  // A <=1-bit value can never have more than one bit set.
  if (node->GetType()->IsBits() && node->BitCountOrDie() <= 1) {
    return true;
  }

  return QueryEngine::AtMostOneBitTrue(node);
}
bool StatelessQueryEngine::AtLeastOneBitTrue(Node* node) const {
  if (node->Is<OneHot>()) {
    return true;
  }

  return QueryEngine::AtLeastOneBitTrue(node);
}
bool StatelessQueryEngine::ExactlyOneBitTrue(Node* node) const {
  if (node->Is<OneHot>()) {
    return true;
  }

  // Fast pattern matches for selectors which are exactly-one-hot by
  // construction.
  //
  // Note: This is intentionally local/structural reasoning; stateless query
  // engine does not propagate facts through the graph.
  if (node->op() == Op::kConcat && node->operand_count() == 2) {
    Node* msb = node->operand(0);
    Node* lsb = node->operand(1);
    auto is_single_bit = [](Node* n) {
      return n->GetType()->IsBits() && n->BitCountOrDie() == 1;
    };
    auto is_not_of = [](Node* n, Node* x) {
      return n->op() == Op::kNot && n->operand_count() == 1 &&
             n->operand(0) == x;
    };

    // concat(not(x), x) or concat(x, not(x)) is exactly-one-hot when x is a
    // single-bit value.
    if (is_single_bit(lsb) && is_not_of(msb, lsb)) {
      return true;
    }
    if (is_single_bit(msb) && is_not_of(lsb, msb)) {
      return true;
    }

    // OneHot(x) may be rewritten into concat(eq(x, 0), x) (e.g. by select
    // simplification). This concat is exactly-one-hot when x is mutually
    // exclusive (at most one bit set).
    Node* maybe_eq = msb;
    Node* x = lsb;
    if (maybe_eq->op() == Op::kEq && maybe_eq->operand_count() == 2) {
      Node* eq_lhs = maybe_eq->operand(0);
      Node* eq_rhs = maybe_eq->operand(1);
      // We only handle the literal-zero comparison in stateless mode.
      if ((eq_lhs == x && IsAllZeros(eq_rhs)) ||
          (eq_rhs == x && IsAllZeros(eq_lhs))) {
        if (AtMostOneBitTrue(x)) {
          return true;
        }
      }
    }
  }

  return QueryEngine::ExactlyOneBitTrue(node);
}

bool StatelessQueryEngine::Implies(const TreeBitLocation& a,
                                   const TreeBitLocation& b) const {
  return a == b || IsZero(a) || IsOne(b);
}

bool StatelessQueryEngine::IsFullyKnown(Node* n) const {
  return n->Is<Literal>();
}

std::optional<int64_t> StatelessQueryEngine::KnownLeadingSignBits(
    Node* node) const {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  int64_t lead_zero = KnownLeadingZeros(node).value_or(0);
  int64_t lead_one = KnownLeadingOnes(node).value_or(0);
  int64_t lead_sign_ext =
      // NB The top bit of the operand is also equal to the sign bit.
      node->op() == Op::kSignExt
          ? 1 + node->BitCountOrDie() - node->operand(0)->BitCountOrDie()
          : 0;
  return std::max({lead_zero, lead_one, lead_sign_ext});
}

}  // namespace xls
