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

#ifndef XLS_IR_PARTIAL_INFORMATION_H_
#define XLS_IR_PARTIAL_INFORMATION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ternary.h"

namespace xls {

// Represents partial information about a bits-typed value, as a combination of
// ternary and interval information.
class PartialInformation {
 public:
  explicit PartialInformation(int64_t bit_count) : bit_count_(bit_count) {}
  explicit PartialInformation(TernarySpan ternary)
      : bit_count_(ternary.size()),
        ternary_(
            std::make_optional<TernaryVector>(ternary.begin(), ternary.end())) {
    Initialize();
  }
  explicit PartialInformation(TernaryVector ternary)
      : bit_count_(ternary.size()), ternary_(std::move(ternary)) {
    Initialize();
  }
  explicit PartialInformation(IntervalSet range)
      : bit_count_(range.BitCount()), range_(std::move(range)) {
    Initialize();
  }
  PartialInformation(TernarySpan ternary, IntervalSet range)
      : bit_count_(range.BitCount()),
        ternary_(
            std::make_optional<TernaryVector>(ternary.begin(), ternary.end())),
        range_(std::move(range)) {
    Initialize();
  }
  PartialInformation(std::optional<TernaryVector> ternary,
                     std::optional<IntervalSet> range)
      : bit_count_(range.has_value()
                       ? range->BitCount()
                       : (ternary.has_value() ? ternary->size() : -1)),
        ternary_(std::move(ternary)),
        range_(std::move(range)) {
    Initialize();
  }
  PartialInformation(int64_t bit_count, std::optional<TernaryVector> ternary,
                     std::optional<IntervalSet> range)
      : bit_count_(bit_count),
        ternary_(std::move(ternary)),
        range_(std::move(range)) {
    Initialize();
  }

  static PartialInformation Unconstrained(int64_t bit_count) {
    return PartialInformation(bit_count, std::nullopt, std::nullopt);
  }
  static PartialInformation Impossible(int64_t bit_count) {
    return PartialInformation(std::nullopt, IntervalSet(bit_count));
  }

  static PartialInformation Precise(Bits value) {
    return PartialInformation(value.bit_count(),
                              ternary_ops::BitsToTernary(value),
                              IntervalSet::Precise(value));
  }

  PartialInformation(const PartialInformation& other) = default;
  PartialInformation& operator=(const PartialInformation& other) = default;

  PartialInformation(PartialInformation&& other) = default;
  PartialInformation& operator=(PartialInformation&& other) = default;

  bool operator==(const PartialInformation& other) const = default;

  const std::optional<TernaryVector>& Ternary() const& { return ternary_; }
  std::optional<TernaryVector> Ternary() && { return std::move(ternary_); }

  const std::optional<IntervalSet>& Range() const& { return range_; }
  std::optional<IntervalSet> Range() && { return std::move(range_); }

  IntervalSet RangeOrMaximal() const {
    return range_.value_or(IntervalSet::Maximal(bit_count_));
  }

  int64_t BitCount() const { return bit_count_; }

  bool IsImpossible() const { return range_.has_value() && range_->IsEmpty(); }
  bool IsUnconstrained() const {
    return !ternary_.has_value() && !range_.has_value();
  }

  bool IsPrecise() const { return range_.has_value() && range_->IsPrecise(); }
  std::optional<Bits> GetPreciseValue() const {
    if (!range_.has_value()) {
      return std::nullopt;
    }
    return range_->GetPreciseValue();
  }

  // Returns true if this PartialInformation is compatible with the given value.
  bool IsCompatibleWith(Bits value) const;

  // Returns true if there are values satisfying both this and `other`.
  bool IsCompatibleWith(PartialInformation other) const;
  bool IsCompatibleWith(const IntervalSet& other) const;
  bool IsCompatibleWith(const Interval& other) const;
  bool IsCompatibleWith(TernarySpan other) const;

  // Returns true if this represents the exclusion of a single value.
  bool IsPunctured() const { return GetPuncturedValue().has_value(); }

  // Returns the single value that is excluded by this PartialInformation, if it
  // represents the exclusion of a single value.
  std::optional<Bits> GetPuncturedValue() const;

  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PartialInformation& info) {
    sink.Append(info.ToString());
  }

  PartialInformation& Not();
  PartialInformation& And(const PartialInformation& other);
  PartialInformation& Or(const PartialInformation& other);
  PartialInformation& Xor(const PartialInformation& other);
  PartialInformation& Nand(const PartialInformation& other) {
    return And(other).Not();
  }
  PartialInformation& Nor(const PartialInformation& other) {
    return Or(other).Not();
  }

  PartialInformation& Reverse();

  PartialInformation& Gate(const PartialInformation& control);

  PartialInformation& Neg();
  PartialInformation& Add(const PartialInformation& other);
  PartialInformation& Sub(const PartialInformation& other);

  PartialInformation& Shll(const PartialInformation& other);
  PartialInformation& Shrl(const PartialInformation& other);
  PartialInformation& Shra(const PartialInformation& other);

  // Updates this PartialInformation to include the information in `other`; if
  // the result would be impossible, marks this PartialInformation as
  // impossible. The result is compatible with any value that's compatible with
  // both this and `other`.
  PartialInformation& JoinWith(const PartialInformation& other);

  // Updates this PartialInformation to only the knowledge in common with
  // `other`. The result is compatible with anything that's compatible with
  // either `this` or `other`.
  PartialInformation& MeetWith(const PartialInformation& other);

  template <typename H>
  friend H AbslHashValue(H h, const PartialInformation& p) {
    return H::combine(std::move(h), p.bit_count_, p.ternary_, p.range_);
  }

 private:
  // Contains shared logic for the constructors, to be invoked after all data
  // members are initialized.
  void Initialize() {
    if (ternary_.has_value()) {
      CHECK_EQ(ternary_->size(), bit_count_);
    }
    if (range_.has_value()) {
      CHECK_EQ(range_->BitCount(), bit_count_);
      range_->Normalize();
    }
    ReconcileInformation();
  }

  void ReconcileInformation();

  void MarkImpossible() {
    ternary_ = std::nullopt;
    range_ = IntervalSet(bit_count_);
  }

  int64_t bit_count_;

  // The ternary information, if any; if not present, it means that the ternary
  // value is unconstrained.
  std::optional<TernaryVector> ternary_;

  // The range information, if any; if not present, it means that the range is
  // unconstrained.
  std::optional<IntervalSet> range_;
};

}  // namespace xls

#endif  // XLS_IR_PARTIAL_INFORMATION_H_
