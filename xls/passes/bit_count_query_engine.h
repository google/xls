// Copyright 2025 The XLS Authors
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

#ifndef XLS_PASSES_BIT_COUNT_QUERY_ENGINE_H_
#define XLS_PASSES_BIT_COUNT_QUERY_ENGINE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/passes/lazy_query_engine.h"
#include "xls/passes/query_engine.h"
namespace xls {

namespace internal {
// Helper class holding information about the leading/most-significant bits of a
// node. Holds the value of the leading bits (if known) and the number of bits
// which are known to be exactly equal to the leading bit.
class LeadingBits {
 public:
  LeadingBits(TernaryValue value, int64_t count)
      : value_(value), count_(count) {}
  LeadingBits(const LeadingBits&) = default;
  LeadingBits(LeadingBits&&) = default;
  LeadingBits& operator=(const LeadingBits&) = default;
  LeadingBits& operator=(LeadingBits&&) = default;
  bool operator==(const LeadingBits&) const = default;
  bool operator!=(const LeadingBits&) const = default;

  static LeadingBits ZeroSize() {
    return LeadingBits(TernaryValue::kUnknown, 0);
  }
  static LeadingBits SignValues(int64_t cnt) {
    CHECK_GE(cnt, 1);
    return LeadingBits(TernaryValue::kUnknown, cnt);
  }
  static LeadingBits Unconstrained() { return SignValues(1); }
  static LeadingBits KnownZeros(int64_t cnt) {
    if (cnt == 0) {
      return Unconstrained();
    }
    CHECK_GE(cnt, 1);
    return LeadingBits(TernaryValue::kKnownZero, cnt);
  }
  static LeadingBits KnownOnes(int64_t cnt) {
    if (cnt == 0) {
      return Unconstrained();
    }
    CHECK_GE(cnt, 1);
    return LeadingBits(TernaryValue::kKnownOne, cnt);
  }

  LeadingBits ExtendBy(
      int64_t extend_by,
      int64_t limited_to = std::numeric_limits<int64_t>::max()) const {
    return LeadingBits(
        value_,
        std::min(limited_to,
                 count_ > (std::numeric_limits<int64_t>::max() - extend_by)
                     ? std::numeric_limits<int64_t>::max()
                     : count_ + extend_by));
  }
  LeadingBits LimitSizeTo(int64_t size) const {
    return LeadingBits(value_, std::min(count_, size));
  }
  LeadingBits ToSignBits() const {
    return LeadingBits(TernaryValue::kUnknown, count_);
  }
  LeadingBits ShortenBy(int64_t shorten_by) const {
    if (count_ == 0) {
      return *this;
    }
    int64_t new_cnt = count_ - shorten_by;
    if (new_cnt <= 0) {
      // Lose known value information.
      return LeadingBits::SignValues(1);
    }
    return LeadingBits(value_, new_cnt);
  }

  // The value the leading bits have.
  TernaryValue value() const { return value_; }
  // How many leading bits have the given value.
  int64_t count() const { return count_; }

  int64_t leading_zeros() const {
    if (value_ == TernaryValue::kKnownZero) {
      return count_;
    }
    return 0;
  }

  int64_t leading_ones() const {
    if (value_ == TernaryValue::kKnownOne) {
      return count_;
    }
    return 0;
  }

  int64_t leading_signs() const { return count_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const LeadingBits& l) {
    absl::Format(&sink, "LeadingBits{leading_value: %s, count: %d}",
                 ToString(l.value_), l.count_);
  }

 private:
  TernaryValue value_;
  int64_t count_;
};
}  // namespace internal

// A (lazy) query engine which keeps track of the number of leading zero, one
// and sign bits and nothing else.
class BitCountQueryEngine : public LazyQueryEngine<internal::LeadingBits> {
 public:
  // some information is present but its unlikely to be useful. If this is
  // required use a ternary engine.
  std::optional<SharedTernaryTree> GetTernary(Node* node) const override {
    return std::nullopt;
  }
  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return false;
  }
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return false;
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return KnownEquals(a, b);
  }

  // TODO(allight): We actually can implement this in some cases if there are
  // enough predicate bits. Not really worth it though.
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

  std::optional<bool> KnownValue(const TreeBitLocation& loc) const override;

  std::optional<int64_t> KnownLeadingOnes(Node* n) const override {
    auto info = GetInfo(n);
    if (!info || !n->GetType()->IsBits()) {
      return std::nullopt;
    }
    return info->Get({}).leading_ones();
  }

  std::optional<int64_t> KnownLeadingZeros(Node* n) const override {
    auto info = GetInfo(n);
    if (!info || !n->GetType()->IsBits()) {
      return std::nullopt;
    }
    return info->Get({}).leading_zeros();
  }

  std::optional<int64_t> KnownLeadingSignBits(Node* n) const override {
    auto info = GetInfo(n);
    if (!info || !n->GetType()->IsBits()) {
      return std::nullopt;
    }
    return info->Get({}).leading_signs();
  }

 protected:
  LeafTypeTree<internal::LeadingBits> ComputeInfo(
      Node* node, absl::Span<const LeafTypeTree<internal::LeadingBits>* const>
                      operand_infos) const override;
  absl::Status MergeWithGiven(
      internal::LeadingBits& info,
      const internal::LeadingBits& given) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_BIT_COUNT_QUERY_ENGINE_H_
