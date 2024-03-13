// Copyright 2021 The XLS Authors
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

#include "xls/passes/range_query_engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/query_engine.h"

namespace xls {

enum class Tonicity { Monotone, Antitone, Unknown };

class RangeQueryVisitor : public DfsVisitor {
 public:
  explicit RangeQueryVisitor(RangeQueryEngine* engine,
                             RangeDataProvider& givens)
      : engine_(engine), givens_(givens), rf_(ReachedFixpoint::Unchanged) {}

  ReachedFixpoint GetReachedFixpoint() const { return rf_; }

 private:
  bool SetIfGiven(Node* node) {
    std::optional<RangeData> memoized_result = givens_.GetKnownIntervals(node);
    if (memoized_result.has_value()) {
      if (memoized_result->ternary.has_value()) {
        engine_->known_bits_[node] =
            ternary_ops::ToKnownBits(*memoized_result->ternary);
        engine_->known_bit_values_[node] =
            ternary_ops::ToKnownBitsValues(*memoized_result->ternary);
      }
      engine_->interval_sets_[node] = memoized_result->interval_set;
      return true;
    }
    return false;
  }

  // The maximum size of an interval set that can be resident in memory at any
  // one time. When accumulating a result interval set, if the set exceeds this
  // size, `MinimizeIntervals` will be called to reduce its size at the cost of
  // precision of the analysis.
  static constexpr int64_t kMaxResIntervalSetSize = 64;

  // The maximum number of points covered by an interval set that can be
  // iterated over in an analysis.
  static constexpr int64_t kMaxIterationSize = 1024;

  // Wrapper around GetIntervalSetTree for consistency with the
  // SetIntervalSetTree wrapper.
  IntervalSetTree GetIntervalSetTree(Node* node) const {
    return engine_->GetIntervalSetTree(node);
  }

  // Wrapper around engine_->SetIntervalSetTree that modifies rf_ if necessary.
  void SetIntervalSetTree(Node* node, const IntervalSetTree& interval_sets) {
    if (!engine_->interval_sets_.contains(node)) {
      engine_->SetIntervalSetTree(node, interval_sets);
      for (const IntervalSet& set : interval_sets.elements()) {
        if (!set.IsMaximal()) {
          rf_ = ReachedFixpoint::Changed;
        }
      }
      return;
    }

    // In the event of a hash collision, it is possible that this could report
    // having reached a fixed point when it hasn't actually. However, this is
    // fairly harmless and very unlikely.
    size_t hash_before =
        absl::Hash<IntervalSetTree>()(engine_->interval_sets_.at(node));
    engine_->SetIntervalSetTree(node, interval_sets);
    size_t hash_after =
        absl::Hash<IntervalSetTree>()(engine_->interval_sets_.at(node));
    if (!(hash_before == hash_after)) {
      rf_ = ReachedFixpoint::Changed;
    }
  }

  // Handles an operation that is variadic and has an implementation given by
  // the given function which has the given tonicities for each argument.
  //
  // The function may return `std::nullopt` to indicate that some
  // exception has occurred, like an integer overflow, which invalidates the
  // analysis. When that happens, this function will return without any side
  // effects having occurred.
  //
  // `Tonicity::Unknown` is not currently supported.
  //
  // CHECK fails if `op->operands().size()` is not equal to `tonicities.size()`.
  absl::Status HandleVariadicOp(
      std::function<std::optional<Bits>(absl::Span<const Bits>)> impl,
      absl::Span<const Tonicity> tonicities, Node* op);

  // Handles an operation that is monotone, unary (and whose argument is given
  // by `op->operand(0)`), and has an implementation given by the given
  // function. The function may return `std::nullopt` to indicate that some
  // exception has occurred, like an integer overflow, which invalidates the
  // analysis. When that happens, this function will return without any side
  // effects having occurred.
  //
  // An operation is monotone iff `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(x), impl(y))`.
  absl::Status HandleMonotoneUnaryOp(
      std::function<std::optional<Bits>(const Bits&)> impl, Node* op);

  // Handles an operation that is antitone, unary (and whose argument is given
  // by `op->operand(0)`), and has an implementation given by the given
  // function. The function may return `std::nullopt` to indicate that some
  // exception has occurred, like an integer overflow, which invalidates the
  // analysis. When that happens, this function will return without any side
  // effects having occurred.
  //
  // An operation is antitone iff `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(y), impl(x))`.
  absl::Status HandleAntitoneUnaryOp(
      std::function<std::optional<Bits>(const Bits&)> impl, Node* op);

  // Handles an operation that is binary (and whose arguments are given by
  // `op->operand(0)` and `op->operand(1)`), monotone in both arguments, and
  // has an implementation given by the given function. The function may return
  // `std::nullopt` to indicate that some exception has occurred, like an
  // integer overflow, which invalidates the analysis. When that happens, this
  // function will return without any side effects having occurred.
  //
  // A binary operation is monotone-monotone iff for every `k`,
  // `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(x, k), impl(y, k))`
  // and `bits_ops::ULessThanOrEqual(impl(k, x), impl(k, y))`.
  absl::Status HandleMonotoneMonotoneBinOp(
      std::function<std::optional<Bits>(const Bits&, const Bits&)> impl,
      Node* op);

  // Handles an operation that is binary (and whose arguments are given by
  // `op->operand(0)` and `op->operand(1)`), monotone in the first argument
  // and antitone in the second argument, and has an implementation given by
  // the given function. The function may return `std::nullopt` to indicate
  // that some exception has occurred, like an integer overflow, which
  // invalidates the analysis. When that happens, this function will return
  // without any side effects having occurred.
  //
  // A binary operation is monotone-antitone iff for every `k`,
  // `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(x, k), impl(y, k))`
  // and `bits_ops::ULessThanOrEqual(impl(k, y), impl(k, x))`.
  absl::Status HandleMonotoneAntitoneBinOp(
      std::function<std::optional<Bits>(const Bits&, const Bits&)> impl,
      Node* op);

  // Analyze whether elements of the two given interval sets must be equal, must
  // not be equal, or may be either.
  //
  // If the given interval sets are precise and identical, returns `true`.
  // If the given interval sets are disjoint, returns `false`.
  // In all other cases, returns `std::nullopt`.
  static std::optional<bool> AnalyzeEq(const IntervalSet& lhs,
                                       const IntervalSet& rhs);

  // Analyze whether elements of the two given interval sets must be less than,
  // must not be less than, or may be either.
  //
  // If `lhs` is disjoint from `rhs` and `lhs.ConvexHull() < rhs.ConvexHull()`,
  // returns `true`.
  // If `lhs` is disjoint from `rhs` and `rhs.ConvexHull() < lhs.ConvexHull()`,
  // returns `false`.
  // In all other cases, returns `std::nullopt`.
  static std::optional<bool> AnalyzeLt(const IntervalSet& lhs,
                                       const IntervalSet& rhs);

  // An interval set covering exactly the binary representation of `false`.
  static IntervalSet FalseIntervalSet();

  // An interval set covering exactly the binary representation of `true`.
  static IntervalSet TrueIntervalSet();

  // Returns an interval set tree which has empty ranges for all elements,
  static IntervalSetTree EmptyIntervalSetTree(Type* type);

  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleMinDelay(MinDelay* min_delay) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override;
  absl::Status HandleAssert(Assert* assert_op) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleCountedFor(CountedFor* counted_for) override;
  absl::Status HandleCover(Cover* cover) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleDynamicCountedFor(
      DynamicCountedFor* dynamic_counted_for) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleGate(Gate* gate) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleInstantiationInput(
      InstantiationInput* instantiation_input) override;
  absl::Status HandleInstantiationOutput(
      InstantiationOutput* instantiation_output) override;
  absl::Status HandleInputPort(InputPort* input_port) override;
  absl::Status HandleInvoke(Invoke* invoke) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleMap(Map* map) override;
  absl::Status HandleArrayIndex(ArrayIndex* index) override;
  absl::Status HandleArraySlice(ArraySlice* slice) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override;
  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryNand(NaryOp* nand_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* sel) override;
  absl::Status HandlePrioritySel(PrioritySelect* sel) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleOutputPort(OutputPort* output_port) override;
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleNext(Next* next) override;
  absl::Status HandleReceive(Receive* receive) override;
  absl::Status HandleRegisterRead(RegisterRead* reg_read) override;
  absl::Status HandleRegisterWrite(RegisterWrite* reg_write) override;
  absl::Status HandleReverse(UnOp* reverse) override;
  absl::Status HandleSDiv(BinOp* div) override;
  absl::Status HandleSGe(CompareOp* ge) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleSMod(BinOp* mod) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleSMulp(PartialProductOp* mul) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleShll(BinOp* shll) override;
  absl::Status HandleShra(BinOp* shra) override;
  absl::Status HandleShrl(BinOp* shrl) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleTrace(Trace* trace_op) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* index) override;
  absl::Status HandleUDiv(BinOp* div) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleUMod(BinOp* mod) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleUMulp(PartialProductOp* mul) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

  RangeQueryEngine* engine_;
  RangeDataProvider& givens_;
  ReachedFixpoint rf_;
};

struct MergeInterval {
  int64_t start;
  int64_t end;

  friend bool operator<(const MergeInterval& lhs, const MergeInterval& rhs) {
    std::pair<int64_t, int64_t> lhs_pair{lhs.start, lhs.end};
    std::pair<int64_t, int64_t> rhs_pair{rhs.start, rhs.end};
    return lhs_pair < rhs_pair;
  }
};

struct BitsWithIndex {
  Bits bits;
  int64_t index;

  friend bool operator<(const BitsWithIndex& lhs, const BitsWithIndex& rhs) {
    if (bits_ops::ULessThan(lhs.bits, rhs.bits)) {
      return true;
    }
    if (bits_ops::UEqual(lhs.bits, rhs.bits)) {
      return lhs.index < rhs.index;
    }
    return false;
  }
};

// Given a set of `Bits` (all the same bit-width) and a desired size for that
// set, reduce the number of elements in the set to the desired size by
// computing a way to merge together small elements of the set. Returns a list
// of ranges in the input vector that should be merged (i.e.: each range should
// be compacted down to a single point by whatever processes the output of
// this function).
static std::vector<MergeInterval> ReduceByMerging(
    absl::Span<Bits const> elements, int64_t desired_size) {
  if (elements.size() <= desired_size) {
    return {};
  }

  std::vector<BitsWithIndex> elements_with_index;

  {
    int64_t i = 0;
    for (const Bits& element : elements) {
      elements_with_index.push_back(BitsWithIndex{element, i});
      ++i;
    }
  }

  std::sort(elements_with_index.begin(), elements_with_index.end());

  std::vector<int64_t> indexes_to_merge;
  indexes_to_merge.reserve(elements.size() - desired_size);
  for (int64_t i = 0; i < elements.size() - desired_size; ++i) {
    indexes_to_merge.push_back(elements_with_index[i].index);
  }
  std::sort(indexes_to_merge.begin(), indexes_to_merge.end());

  // Merge contiguous runs of indices into intervals
  std::vector<MergeInterval> result;
  for (int64_t i = 0; i < indexes_to_merge.size(); ++i) {
    int64_t range_start = indexes_to_merge[i];
    while (((i + 1) < indexes_to_merge.size()) &&
           ((indexes_to_merge[i] + 1) == indexes_to_merge[i + 1])) {
      ++i;
    }
    int64_t range_end = indexes_to_merge[i];
    result.push_back({range_start, range_end});
  }

  return result;
}

IntervalSet MinimizeIntervals(IntervalSet intervals, int64_t size) {
  intervals.Normalize();

  if (intervals.NumberOfIntervals() <= 1) {
    return intervals;
  }

  std::vector<Bits> gap_vector;
  for (int64_t i = 0; i < intervals.NumberOfIntervals() - 1; ++i) {
    const Bits& x = intervals.Intervals()[i].UpperBound();
    const Bits& y = intervals.Intervals()[i + 1].LowerBound();
    gap_vector.push_back(bits_ops::Sub(y, x));
  }

  std::vector<MergeInterval> merges = ReduceByMerging(gap_vector, size - 1);

  IntervalSet result = intervals;

  for (const auto& m : merges) {
    Interval merged = intervals.Intervals()[m.start];
    for (int64_t i = m.start; i <= m.end + 1; ++i) {
      merged = Interval::ConvexHull(merged, intervals.Intervals()[i]);
    }
    result.AddInterval(merged);
  }

  result.Normalize();

  return result;
}

absl::StatusOr<ReachedFixpoint> RangeQueryEngine::PopulateWithGivens(
    RangeDataProvider& givens) {
  RangeQueryVisitor visitor(this, givens);
  XLS_RETURN_IF_ERROR(givens.IterateFunction(&visitor));
  return visitor.GetReachedFixpoint();
}

IntervalSetTree RangeQueryEngine::GetIntervalSetTree(Node* node) const {
  if (interval_sets_.contains(node)) {
    return interval_sets_.at(node);
  }

  if (node->GetType()->IsTuple() || node->GetType()->IsArray()) {
    IntervalSetTree result(node->GetType());
    int64_t i = 0;
    for (Type* type : result.leaf_types()) {
      int64_t size = type->GetFlatBitCount();
      result.elements()[i] = IntervalSet::Maximal(size);
      ++i;
    }
    return result;
  }

  int64_t size = node->GetType()->GetFlatBitCount();
  IntervalSetTree result(node->GetType());
  result.Set({}, IntervalSet::Maximal(size));
  return result;
}

void RangeQueryEngine::SetIntervalSetTree(
    Node* node, const IntervalSetTree& interval_sets) {
  IntervalSetTree ist = GetIntervalSetTree(node);
  leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
      ist.AsMutableView(), interval_sets.AsView(),
      [](IntervalSet& lhs, const IntervalSet& rhs) {
        lhs = IntervalSet::Intersect(lhs, rhs);
      });
  if (node->GetType()->IsBits()) {
    interval_ops::KnownBits bits =
        interval_ops::ExtractKnownBits(ist.Get({}), /*source=*/node);
    known_bits_[node] = bits.known_bits;
    known_bit_values_[node] = bits.known_bit_values;
  }
  interval_sets_[node] = ist;
}

void RangeQueryEngine::InitializeNode(Node* node) {
  if (!known_bits_.contains(node) || !known_bit_values_.contains(node)) {
    known_bits_[node] = Bits(node->GetType()->GetFlatBitCount());
    known_bit_values_[node] = Bits(node->GetType()->GetFlatBitCount());
  }
}

absl::Status RangeQueryVisitor::HandleVariadicOp(
    std::function<std::optional<Bits>(absl::Span<const Bits>)> impl,
    absl::Span<const Tonicity> tonicities, Node* op) {
  CHECK_EQ(op->operands().size(), tonicities.size());

  std::vector<IntervalSet> operands;
  operands.reserve(op->operands().size());

  {
    int64_t i = 0;
    for (Node* operand : op->operands()) {
      IntervalSet interval_set = GetIntervalSetTree(operand).Get({});

      // TODO(taktoa): we could choose the minimized interval sets more
      // carefully, since `MinimizeIntervals` is minimizing optimally for each
      // interval set without the knowledge that other interval sets exist.
      // For example, we could call `ConvexHull` greedily on the sets
      // that have the smallest difference between convex hull size and size.

      // Limit exponential growth after 12 parameters. 5^12 = 244 million
      interval_set = MinimizeIntervals(interval_set, (i < 12) ? 5 : 1);
      operands.push_back(interval_set);
      ++i;
    }
  }

  std::vector<int64_t> radix;
  radix.reserve(operands.size());
  for (const IntervalSet& interval_set : operands) {
    radix.push_back(interval_set.NumberOfIntervals());
  }

  IntervalSet result_intervals(op->BitCountOrDie());

  absl::Status early_status = absl::OkStatus();

  // Each iteration of this do-while loop explores a different choice of
  // intervals from each interval set associated with a parameter.
  bool returned_early = MixedRadixIterate(
      radix, [&](const std::vector<int64_t>& indexes) -> bool {
        std::vector<Bits> lower_bounds;
        lower_bounds.reserve(indexes.size());
        std::vector<Bits> upper_bounds;
        upper_bounds.reserve(indexes.size());
        for (int64_t i = 0; i < indexes.size(); ++i) {
          Interval interval = operands[i].Intervals()[indexes[i]];
          switch (tonicities[i]) {
            case Tonicity::Monotone: {
              // The essential property of a unary monotone function `f` is that
              // the codomain of `f` applied to `[x, y]` is `[f(x), f(y)]`.
              // For example, the cubing function applied to `[5, 8]` gives a
              // codomain of `[125, 512]`.
              lower_bounds.push_back(interval.LowerBound());
              upper_bounds.push_back(interval.UpperBound());
              break;
            }
            case Tonicity::Antitone: {
              // The essential property of a unary antitone function `f` is that
              // the codomain of `f` applied to `[x, y]` is `[f(y), f(x)]`.
              // For example, the negation function applied to `[10, 20]` gives
              // a codomain of `[-20, -10]`.
              lower_bounds.push_back(interval.UpperBound());
              upper_bounds.push_back(interval.LowerBound());
              break;
            }
            case Tonicity::Unknown: {
              early_status =
                  absl::InternalError("Tonicity::Unknown not yet supported");
              return true;
            }
          }
        }
        std::optional<Bits> lower = impl(lower_bounds);
        std::optional<Bits> upper = impl(upper_bounds);
        if (!lower.has_value() || !upper.has_value()) {
          return true;
        }
        result_intervals.AddInterval(Interval(lower.value(), upper.value()));
        return false;
      });

  if (returned_early) {
    return early_status;
  }

  result_intervals = MinimizeIntervals(result_intervals);

  LeafTypeTree<IntervalSet> result(op->GetType());
  result.Set({}, result_intervals);
  SetIntervalSetTree(op, result);

  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleMonotoneUnaryOp(
    std::function<std::optional<Bits>(const Bits&)> impl, Node* op) {
  return HandleVariadicOp(
      [impl](absl::Span<const Bits> bits) -> std::optional<Bits> {
        CHECK_EQ(bits.size(), 1);
        return impl(bits[0]);
      },
      std::vector<Tonicity>(1, Tonicity::Monotone), op);
}

absl::Status RangeQueryVisitor::HandleAntitoneUnaryOp(
    std::function<std::optional<Bits>(const Bits&)> impl, Node* op) {
  return HandleVariadicOp(
      [impl](absl::Span<const Bits> bits) -> std::optional<Bits> {
        CHECK_EQ(bits.size(), 1);
        return impl(bits[0]);
      },
      std::vector<Tonicity>(1, Tonicity::Antitone), op);
}

absl::Status RangeQueryVisitor::HandleMonotoneMonotoneBinOp(
    std::function<std::optional<Bits>(const Bits&, const Bits&)> impl,
    Node* op) {
  return HandleVariadicOp(
      [impl](absl::Span<const Bits> bits) -> std::optional<Bits> {
        CHECK_EQ(bits.size(), 2);
        return impl(bits[0], bits[1]);
      },
      {Tonicity::Monotone, Tonicity::Monotone}, op);
}

absl::Status RangeQueryVisitor::HandleMonotoneAntitoneBinOp(
    std::function<std::optional<Bits>(const Bits&, const Bits&)> impl,
    Node* op) {
  return HandleVariadicOp(
      [impl](absl::Span<const Bits> bits) -> std::optional<Bits> {
        CHECK_EQ(bits.size(), 2);
        return impl(bits[0], bits[1]);
      },
      {Tonicity::Monotone, Tonicity::Antitone}, op);
}

std::optional<bool> RangeQueryVisitor::AnalyzeEq(const IntervalSet& lhs,
                                                 const IntervalSet& rhs) {
  CHECK(lhs.IsNormalized());
  CHECK(rhs.IsNormalized());

  bool is_precise = lhs.IsPrecise() && rhs.IsPrecise();

  // TODO(taktoa): This way of checking disjointness is efficient but
  // unfortunately fails when there are abutting intervals; it shouldn't be too
  // hard to make a Disjoint static method on IntervalSet that doesn't have this
  // issue. Luckily this only results in a loss of precision, not incorrectness.

  IntervalSet combined = IntervalSet::Combine(lhs, rhs);
  bool is_disjoint = (lhs.Intervals().size() + rhs.Intervals().size()) ==
                     combined.Intervals().size();

  if (is_precise && (lhs.Intervals() == rhs.Intervals())) {
    return true;
  }
  if (is_disjoint) {
    return false;
  }
  return std::nullopt;
}

std::optional<bool> RangeQueryVisitor::AnalyzeLt(const IntervalSet& lhs,
                                                 const IntervalSet& rhs) {
  if (std::optional<Interval> lhs_hull = lhs.ConvexHull()) {
    if (std::optional<Interval> rhs_hull = rhs.ConvexHull()) {
      if (Interval::Disjoint(*lhs_hull, *rhs_hull)) {
        if (*lhs_hull < *rhs_hull) {
          return true;
        }
        if (*rhs_hull < *lhs_hull) {
          return false;
        }
      }
    }
  }
  return std::nullopt;
}

IntervalSet RangeQueryVisitor::FalseIntervalSet() {
  IntervalSet result(1);
  result.AddInterval(Interval(UBits(0, 1), UBits(0, 1)));
  result.Normalize();
  return result;
}

IntervalSet RangeQueryVisitor::TrueIntervalSet() {
  IntervalSet result(1);
  result.AddInterval(Interval(UBits(1, 1), UBits(1, 1)));
  result.Normalize();
  return result;
}

IntervalSetTree RangeQueryVisitor::EmptyIntervalSetTree(Type* type) {
  LeafTypeTree<IntervalSet> result(type);
  for (int64_t i = 0; i < result.elements().size(); ++i) {
    result.elements()[i] =
        IntervalSet(result.leaf_types()[i]->GetFlatBitCount());
  }
  return result;
}

// Helper to just pull result straight from givens if possible.
#define INITIALIZE_OR_SKIP(node)   \
  do {                             \
    if (SetIfGiven(node)) {        \
      return absl::OkStatus();     \
    }                              \
    engine_->InitializeNode(node); \
  } while (false)

absl::Status RangeQueryVisitor::HandleAdd(BinOp* add) {
  INITIALIZE_OR_SKIP(add);
  return HandleMonotoneMonotoneBinOp(
      [](const Bits& lhs, const Bits& rhs) -> std::optional<Bits> {
        int64_t padded_size = std::max(lhs.bit_count(), rhs.bit_count()) + 1;
        Bits padded_lhs = bits_ops::ZeroExtend(lhs, padded_size);
        Bits padded_rhs = bits_ops::ZeroExtend(rhs, padded_size);
        Bits padded_result = bits_ops::Add(padded_lhs, padded_rhs);
        // If the MSB is 1, then we overflowed.
        if (padded_result.msb()) {
          return std::nullopt;
        }
        return bits_ops::Add(lhs, rhs);
      },
      add);
}

absl::Status RangeQueryVisitor::HandleAfterAll(AfterAll* after_all) {
  INITIALIZE_OR_SKIP(after_all);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleMinDelay(MinDelay* min_delay) {
  INITIALIZE_OR_SKIP(min_delay);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleAndReduce(
    BitwiseReductionOp* and_reduce) {
  INITIALIZE_OR_SKIP(and_reduce);
  IntervalSet intervals = GetIntervalSetTree(and_reduce->operand(0)).Get({});

  LeafTypeTree<IntervalSet> result(and_reduce->GetType());
  bool result_valid = false;

  // Unless the intervals cover max, the and_reduce of the input must be 0.
  if (!intervals.CoversMax()) {
    result.Set({}, FalseIntervalSet());
    result_valid = true;
  }

  // If the intervals are known to only cover max, then the result must be 1.
  if (std::optional<Interval> hull = intervals.ConvexHull()) {
    Bits max = Bits::AllOnes(intervals.BitCount());
    if (hull.value() == Interval(max, max)) {
      result.Set({}, TrueIntervalSet());
      result_valid = true;
    }
  }

  if (result_valid) {
    SetIntervalSetTree(and_reduce, result);
  }

  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleArray(Array* array) {
  INITIALIZE_OR_SKIP(array);
  std::vector<LeafTypeTree<IntervalSet>> children;
  children.reserve(array->operand_count());
  for (Node* element : array->operands()) {
    children.push_back(GetIntervalSetTree(element));
  }
  // TODO(https://github.com/google/xls/issues/1334): Replace range query API to
  // take/return LeafTypeTree views rather than copying these objects all the
  // time.
  std::vector<LeafTypeTreeView<IntervalSet>> children_views;
  children_views.reserve(children.size());
  for (const LeafTypeTree<IntervalSet>& child : children) {
    children_views.push_back(child.AsView());
  }

  XLS_ASSIGN_OR_RETURN(LeafTypeTree<IntervalSet> result,
                       leaf_type_tree::CreateArray<IntervalSet>(
                           array->GetType()->AsArrayOrDie(), children_views));
  SetIntervalSetTree(array, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleArrayConcat(ArrayConcat* array_concat) {
  INITIALIZE_OR_SKIP(array_concat);
  std::vector<LeafTypeTree<IntervalSet>> children;
  for (Node* element : array_concat->operands()) {
    LeafTypeTree<IntervalSet> concatee = GetIntervalSetTree(element);
    const int64_t arr_size = element->GetType()->AsArrayOrDie()->size();
    for (int32_t i = 0; i < arr_size; ++i) {
      children.push_back(leaf_type_tree::Clone(concatee.AsView({i})));
    }
  }
  // TODO(https://github.com/google/xls/issues/1334): Replace range query API to
  // take/return LeafTypeTree views rather than copying these objects all the
  // time.
  std::vector<LeafTypeTreeView<IntervalSet>> children_views;
  children_views.reserve(children.size());
  for (const LeafTypeTree<IntervalSet>& child : children) {
    children_views.push_back(child.AsView());
  }

  XLS_ASSIGN_OR_RETURN(
      LeafTypeTree<IntervalSet> result,
      leaf_type_tree::CreateArray<IntervalSet>(
          array_concat->GetType()->AsArrayOrDie(), children_views));
  SetIntervalSetTree(array_concat, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleAssert(Assert* assert_op) {
  INITIALIZE_OR_SKIP(assert_op);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleBitSlice(BitSlice* bit_slice) {
  INITIALIZE_OR_SKIP(bit_slice);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleBitSliceUpdate(BitSliceUpdate* update) {
  INITIALIZE_OR_SKIP(update);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleConcat(Concat* concat) {
  INITIALIZE_OR_SKIP(concat);
  return HandleVariadicOp(
      bits_ops::Concat,
      std::vector<Tonicity>(concat->operands().size(), Tonicity::Monotone),
      concat);
}

absl::Status RangeQueryVisitor::HandleCountedFor(CountedFor* counted_for) {
  INITIALIZE_OR_SKIP(counted_for);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleCover(Cover* cover) {
  INITIALIZE_OR_SKIP(cover);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleDecode(Decode* decode) {
  INITIALIZE_OR_SKIP(decode);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  INITIALIZE_OR_SKIP(dynamic_bit_slice);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleDynamicCountedFor(
    DynamicCountedFor* dynamic_counted_for) {
  INITIALIZE_OR_SKIP(dynamic_counted_for);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleEncode(Encode* encode) {
  INITIALIZE_OR_SKIP(encode);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleEq(CompareOp* eq) {
  INITIALIZE_OR_SKIP(eq);

  if (!eq->operand(0)->GetType()->IsBits()) {
    // TODO(meheff): 2022/09/06 Add support for non-bits types.
    return absl::OkStatus();
  }

  IntervalSet lhs_intervals = GetIntervalSetTree(eq->operand(0)).Get({});
  IntervalSet rhs_intervals = GetIntervalSetTree(eq->operand(1)).Get({});

  std::optional<bool> analysis = AnalyzeEq(lhs_intervals, rhs_intervals);
  if (analysis.has_value()) {
    LeafTypeTree<IntervalSet> result(eq->GetType());
    if (analysis == std::optional<bool>(true)) {
      result.Set({}, TrueIntervalSet());
    } else if (analysis == std::optional<bool>(false)) {
      result.Set({}, FalseIntervalSet());
    }
    SetIntervalSetTree(eq, result);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleGate(Gate* gate) {
  INITIALIZE_OR_SKIP(gate);
  IntervalSet cond_intervals = GetIntervalSetTree(gate->operand(0)).Get({});

  // `cond` true passes through the data operand.
  LeafTypeTree<IntervalSet> result =
      cond_intervals.CoversOne()
          ? GetIntervalSetTree(gate->operand(1))  // data operand
          : EmptyIntervalSetTree(gate->GetType());

  if (cond_intervals.CoversZero()) {
    // `cond` false produces a zero value.
    for (int64_t i = 0; i < result.size(); ++i) {
      IntervalSet& set = result.elements()[i];
      Type* type = result.leaf_types()[i];
      XLS_RET_CHECK(type->IsBits());
      Bits zero = Bits(type->AsBitsOrDie()->bit_count());
      set.AddInterval(Interval(zero, zero));
      set.Normalize();
    }
  }
  SetIntervalSetTree(gate, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleIdentity(UnOp* identity) {
  INITIALIZE_OR_SKIP(identity);
  return HandleMonotoneUnaryOp([](const Bits& b) { return b; }, identity);
}

absl::Status RangeQueryVisitor::HandleInstantiationInput(
    InstantiationInput* instantiation_input) {
  INITIALIZE_OR_SKIP(instantiation_input);
  return absl::OkStatus();  // TODO(meheff): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleInstantiationOutput(
    InstantiationOutput* instantiation_output) {
  INITIALIZE_OR_SKIP(instantiation_output);
  return absl::OkStatus();  // TODO(meheff): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleInputPort(InputPort* input_port) {
  INITIALIZE_OR_SKIP(input_port);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleInvoke(Invoke* invoke) {
  INITIALIZE_OR_SKIP(invoke);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleLiteral(Literal* literal) {
  INITIALIZE_OR_SKIP(literal);
  XLS_ASSIGN_OR_RETURN(
      LeafTypeTree<Value> v_ltt,
      ValueToLeafTypeTree(literal->value(), literal->GetType()));
  SetIntervalSetTree(
      literal, leaf_type_tree::Map<IntervalSet, Value>(
                   v_ltt.AsView(), [](const Value& value) {
                     IntervalSet interval_set(value.GetFlatBitCount());
                     if (value.IsBits()) {
                       return IntervalSet::Precise(value.bits());
                     }
                     if (value.IsToken()) {
                       return IntervalSet::Precise(Bits(0));
                     }
                     XLS_LOG(FATAL) << "Invalid value kind in HandleLiteral";
                   }));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleMap(Map* map) {
  INITIALIZE_OR_SKIP(map);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArrayIndex(ArrayIndex* array_index) {
  INITIALIZE_OR_SKIP(array_index);

  IntervalSetTree array_interval_set_tree =
      GetIntervalSetTree(array_index->array());

  if (array_index->indices().empty()) {
    SetIntervalSetTree(array_index, array_interval_set_tree);
    return absl::OkStatus();
  }

  // The dimension of the multidimensional array, truncated to the number of
  // indexes we are indexing on.
  // For example, if the array has shape [5, 7, 3, 2] and we index with [i, j]
  // then this will contain [5, 7].
  std::vector<int64_t> dimension;

  // The interval set that each index lives in.
  std::vector<IntervalSet> index_intervals;

  {
    ArrayType* array_type = array_index->array()->GetType()->AsArrayOrDie();
    for (Node* index : array_index->indices()) {
      dimension.push_back(array_type->size());
      index_intervals.push_back(GetIntervalSetTree(index).Get({}));
      absl::StatusOr<ArrayType*> array_type_status =
          array_type->element_type()->AsArray();
      array_type = array_type_status.ok() ? *array_type_status : nullptr;
    }
  }

  XLS_ASSIGN_OR_RETURN(
      IntervalSetTree result,
      LeafTypeTree<IntervalSet>::CreateFromFunction(
          array_index->GetType(),
          [](Type* leaf_type,
             absl::Span<const int64_t> index) -> absl::StatusOr<IntervalSet> {
            return IntervalSet(leaf_type->GetFlatBitCount());
          }));

  // Returns true if the given interval set covers the given index value for an
  // array of dimension `dim`. Includes handling of OOB conditions.
  auto intervals_cover_index = [&](const IntervalSet& intervals, int64_t index,
                                   int64_t dim) {
    if (Bits::MinBitCountUnsigned(index) > intervals.BitCount()) {
      // The concrete index value `index` doesn't even fit in the width of the
      // interval set.
      return false;
    }
    if (intervals.Covers(UBits(index, intervals.BitCount()))) {
      return true;
    }

    // An out-of-bound array index operation returns the maximal index element
    // so the maximal array index is considered covered if the interval set
    // covers *any* out-of-bounds index.
    bool index_is_maximal = index == dim - 1;
    std::optional<Bits> ub = intervals.UpperBound();
    return index_is_maximal && ub.has_value() &&
           bits_ops::UGreaterThanOrEqual(ub.value(), dim);
  };

  MixedRadixIterate(dimension, [&](const std::vector<int64_t>& indexes) {
    for (int64_t i = 0; i < indexes.size(); ++i) {
      if (!intervals_cover_index(index_intervals[i], indexes[i],
                                 dimension[i])) {
        return false;
      }
    }
    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        result.AsMutableView(), array_interval_set_tree.AsView(indexes),
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Combine(lhs, rhs);
        });
    return false;
  });

  SetIntervalSetTree(array_index, result);

  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleArraySlice(ArraySlice* slice) {
  INITIALIZE_OR_SKIP(slice);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArrayUpdate(ArrayUpdate* update) {
  INITIALIZE_OR_SKIP(update);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryAnd(NaryOp* and_op) {
  INITIALIZE_OR_SKIP(and_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryNand(NaryOp* nand_op) {
  INITIALIZE_OR_SKIP(nand_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryNor(NaryOp* nor_op) {
  INITIALIZE_OR_SKIP(nor_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryOr(NaryOp* or_op) {
  INITIALIZE_OR_SKIP(or_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryXor(NaryOp* xor_op) {
  INITIALIZE_OR_SKIP(xor_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNe(CompareOp* ne) {
  INITIALIZE_OR_SKIP(ne);

  if (!ne->operand(0)->GetType()->IsBits()) {
    // TODO(meheff): 2022/09/06 Add support for non-bits types.
    return absl::OkStatus();
  }

  IntervalSet lhs_intervals = GetIntervalSetTree(ne->operand(0)).Get({});
  IntervalSet rhs_intervals = GetIntervalSetTree(ne->operand(1)).Get({});

  std::optional<bool> analysis = AnalyzeEq(lhs_intervals, rhs_intervals);
  if (analysis.has_value()) {
    LeafTypeTree<IntervalSet> result(ne->GetType());
    if (analysis == std::optional<bool>(false)) {
      result.Set({}, TrueIntervalSet());
    } else if (analysis == std::optional<bool>(true)) {
      result.Set({}, FalseIntervalSet());
    }
    SetIntervalSetTree(ne, result);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleNeg(UnOp* neg) {
  INITIALIZE_OR_SKIP(neg);
  return HandleAntitoneUnaryOp(bits_ops::Negate, neg);
}

absl::Status RangeQueryVisitor::HandleNot(UnOp* not_op) {
  INITIALIZE_OR_SKIP(not_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleOneHot(OneHot* one_hot) {
  INITIALIZE_OR_SKIP(one_hot);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleOneHotSel(OneHotSelect* sel) {
  INITIALIZE_OR_SKIP(sel);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandlePrioritySel(PrioritySelect* sel) {
  INITIALIZE_OR_SKIP(sel);
  IntervalSet selector_intervals = GetIntervalSetTree(sel->selector()).Get({});

  // Initialize all interval sets to empty
  XLS_ASSIGN_OR_RETURN(
      LeafTypeTree<IntervalSet> result,
      LeafTypeTree<IntervalSet>::CreateFromFunction(
          sel->GetType(), [](Type* leaf_type, absl::Span<const int64_t> index) {
            return IntervalSet(leaf_type->GetFlatBitCount());
          }));

  if (selector_intervals.CoversZero()) {  // possible to see default
    LeafTypeTree<IntervalSet> all_zero_default(sel->GetType());
    for (int64_t i = 0; i < all_zero_default.elements().size(); ++i) {
      // Set all intervals to zero, the default.
      all_zero_default.elements()[i] = IntervalSet::Precise(
          UBits(0, all_zero_default.leaf_types()[i]->GetFlatBitCount()));
    }
    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        result.AsMutableView(), all_zero_default.AsView(),
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Combine(lhs, rhs);
        });
  }
  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    // TODO(vmirian): Make implementation more efficient by considering only the
    // ranges of interest.
    if (selector_intervals.IsTrueWhenMaskWith(
            bits_ops::ShiftLeftLogical(UBits(1, sel->cases().size()), i))) {
      leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
          result.AsMutableView(), GetIntervalSetTree(sel->cases()[i]).AsView(),
          [](IntervalSet& lhs, const IntervalSet& rhs) {
            lhs = IntervalSet::Combine(lhs, rhs);
          });
    }
  }
  for (IntervalSet& intervals : result.elements()) {
    intervals = MinimizeIntervals(intervals);
  }
  SetIntervalSetTree(sel, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleOrReduce(BitwiseReductionOp* or_reduce) {
  INITIALIZE_OR_SKIP(or_reduce);
  IntervalSet intervals = GetIntervalSetTree(or_reduce->operand(0)).Get({});

  LeafTypeTree<IntervalSet> result(or_reduce->GetType());
  bool result_valid = false;

  // Unless the intervals cover 0, the or_reduce of the input must be 1.
  if (!intervals.CoversZero()) {
    result.Set({}, TrueIntervalSet());
    result_valid = true;
  }

  // If the intervals are known to only cover 0, then the result must be 0.
  if (std::optional<Interval> hull = intervals.ConvexHull()) {
    Bits zero = Bits(1);
    if (hull.value() == Interval(zero, zero)) {
      result.Set({}, FalseIntervalSet());
      result_valid = true;
    }
  }

  if (result_valid) {
    SetIntervalSetTree(or_reduce, result);
  }

  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleOutputPort(OutputPort* output_port) {
  INITIALIZE_OR_SKIP(output_port);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleParam(Param* param) {
  INITIALIZE_OR_SKIP(param);
  // We don't know anything about params.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleNext(Next* next) {
  INITIALIZE_OR_SKIP(next);
  // Next values return nothing, represented with an empty tuple.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleReceive(Receive* receive) {
  INITIALIZE_OR_SKIP(receive);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleRegisterRead(RegisterRead* reg_read) {
  INITIALIZE_OR_SKIP(reg_read);
  return absl::OkStatus();  // TODO(taktoa): implement: needs fixed point
}

absl::Status RangeQueryVisitor::HandleRegisterWrite(RegisterWrite* reg_write) {
  INITIALIZE_OR_SKIP(reg_write);
  return absl::OkStatus();  // TODO(taktoa): implement: needs fixed point
}

absl::Status RangeQueryVisitor::HandleReverse(UnOp* reverse) {
  INITIALIZE_OR_SKIP(reverse);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleSDiv(BinOp* div) {
  INITIALIZE_OR_SKIP(div);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSGe(CompareOp* ge) {
  INITIALIZE_OR_SKIP(ge);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSGt(CompareOp* gt) {
  INITIALIZE_OR_SKIP(gt);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSLe(CompareOp* le) {
  INITIALIZE_OR_SKIP(le);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSLt(CompareOp* lt) {
  INITIALIZE_OR_SKIP(lt);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSMod(BinOp* mod) {
  INITIALIZE_OR_SKIP(mod);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSMul(ArithOp* mul) {
  INITIALIZE_OR_SKIP(mul);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSMulp(PartialProductOp* mul) {
  INITIALIZE_OR_SKIP(mul);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleSel(Select* sel) {
  INITIALIZE_OR_SKIP(sel);
  IntervalSet selector_intervals = GetIntervalSetTree(sel->selector()).Get({});
  bool default_possible = false;
  absl::btree_set<uint64_t> selector_values;
  for (const Interval& interval : selector_intervals.Intervals()) {
    uint64_t num_cases = sel->cases().size();
    interval.ForEachElement([&](const Bits& bits) -> bool {
      uint64_t value = *bits.ToUint64();
      // If part of the interval is outside of the valid selector range, then
      // we assume that the remainder of the interval is also out of range.
      if (value >= num_cases) {
        default_possible = true;
        return true;
      }
      selector_values.insert(value);
      return false;
    });
  }

  // TODO(taktoa): check if sel->cases().size() is greater than the number of
  // representable values in the type of the selector, and set default_possible
  // false in that case.

  // Initialize all interval sets to empty
  XLS_ASSIGN_OR_RETURN(
      auto result,
      LeafTypeTree<IntervalSet>::CreateFromFunction(
          sel->GetType(), [](Type* leaf_type, absl::Span<const int64_t> index) {
            return IntervalSet(leaf_type->GetFlatBitCount());
          }));

  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    if (selector_values.contains(i)) {
      leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
          result.AsMutableView(), GetIntervalSetTree(sel->cases()[i]).AsView(),
          [](IntervalSet& lhs, const IntervalSet& rhs) {
            lhs = IntervalSet::Combine(lhs, rhs);
          });
    }
  }
  if (default_possible && sel->default_value().has_value()) {
    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        result.AsMutableView(),
        GetIntervalSetTree(sel->default_value().value()).AsView(),
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Combine(lhs, rhs);
        });
  }
  for (IntervalSet& intervals : result.elements()) {
    intervals = MinimizeIntervals(intervals);
  }
  SetIntervalSetTree(sel, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleSend(Send* send) {
  INITIALIZE_OR_SKIP(send);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleShll(BinOp* shll) {
  INITIALIZE_OR_SKIP(shll);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleShra(BinOp* shra) {
  INITIALIZE_OR_SKIP(shra);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleShrl(BinOp* shrl) {
  INITIALIZE_OR_SKIP(shrl);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleSignExtend(ExtendOp* sign_ext) {
  INITIALIZE_OR_SKIP(sign_ext);
  return HandleMonotoneUnaryOp(
      [sign_ext](const Bits& bits) -> Bits {
        return bits_ops::SignExtend(bits, sign_ext->new_bit_count());
      },
      sign_ext);
}

absl::Status RangeQueryVisitor::HandleSub(BinOp* sub) {
  INITIALIZE_OR_SKIP(sub);
  return HandleMonotoneAntitoneBinOp(
      [](const Bits& lhs, const Bits& rhs) -> std::optional<Bits> {
        if (bits_ops::ULessThanOrEqual(rhs, lhs)) {
          return bits_ops::Sub(lhs, rhs);
        }
        return std::nullopt;
      },
      sub);
}

absl::Status RangeQueryVisitor::HandleTrace(Trace* trace_op) {
  INITIALIZE_OR_SKIP(trace_op);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleTuple(Tuple* tuple) {
  INITIALIZE_OR_SKIP(tuple);
  std::vector<LeafTypeTree<IntervalSet>> children;
  for (Node* element : tuple->operands()) {
    children.push_back(GetIntervalSetTree(element));
  }
  // TODO(meheff): Replace range query API to take/return LeafTypeTree views
  // rather than copying these objects all the time.
  std::vector<LeafTypeTreeView<IntervalSet>> children_views;
  children_views.reserve(children.size());
  for (const LeafTypeTree<IntervalSet>& child : children) {
    children_views.push_back(child.AsView());
  }
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<IntervalSet> result,
                       leaf_type_tree::CreateTuple<IntervalSet>(
                           tuple->GetType()->AsTupleOrDie(), children_views));
  SetIntervalSetTree(tuple, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleTupleIndex(TupleIndex* index) {
  INITIALIZE_OR_SKIP(index);
  LeafTypeTree<IntervalSet> arg = GetIntervalSetTree(index->operand(0));
  SetIntervalSetTree(index,
                     leaf_type_tree::Clone(arg.AsView({index->index()})));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleUDiv(BinOp* div) {
  INITIALIZE_OR_SKIP(div);
  // I (taktoa) verified on 8 bit unsigned integers that UDiv is antitone in
  // its second argument.
  return HandleMonotoneAntitoneBinOp(
      [](const Bits& lhs, const Bits& rhs) -> std::optional<Bits> {
        return bits_ops::UDiv(lhs, rhs);
      },
      div);
}

absl::Status RangeQueryVisitor::HandleUGe(CompareOp* ge) {
  INITIALIZE_OR_SKIP(ge);
  IntervalSet lhs_intervals = GetIntervalSetTree(ge->operand(0)).Get({});
  IntervalSet rhs_intervals = GetIntervalSetTree(ge->operand(1)).Get({});

  std::optional<bool> analysis = AnalyzeLt(rhs_intervals, lhs_intervals);
  if (AnalyzeEq(lhs_intervals, rhs_intervals) == std::optional<bool>(true)) {
    analysis = true;
  }
  if (analysis.has_value()) {
    LeafTypeTree<IntervalSet> result(ge->GetType());
    if (analysis == std::optional<bool>(true)) {
      result.Set({}, TrueIntervalSet());
    } else if (analysis == std::optional<bool>(false)) {
      result.Set({}, FalseIntervalSet());
    }
    SetIntervalSetTree(ge, result);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleUGt(CompareOp* gt) {
  INITIALIZE_OR_SKIP(gt);
  IntervalSet lhs_intervals = GetIntervalSetTree(gt->operand(0)).Get({});
  IntervalSet rhs_intervals = GetIntervalSetTree(gt->operand(1)).Get({});

  std::optional<bool> analysis = AnalyzeLt(rhs_intervals, lhs_intervals);
  if (analysis.has_value()) {
    LeafTypeTree<IntervalSet> result(gt->GetType());
    if (analysis == std::optional<bool>(true)) {
      result.Set({}, TrueIntervalSet());
    } else if (analysis == std::optional<bool>(false)) {
      result.Set({}, FalseIntervalSet());
    }
    SetIntervalSetTree(gt, result);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleULe(CompareOp* le) {
  INITIALIZE_OR_SKIP(le);
  IntervalSet lhs_intervals = GetIntervalSetTree(le->operand(0)).Get({});
  IntervalSet rhs_intervals = GetIntervalSetTree(le->operand(1)).Get({});

  std::optional<bool> analysis = AnalyzeLt(lhs_intervals, rhs_intervals);
  if (AnalyzeEq(lhs_intervals, rhs_intervals) == std::optional<bool>(true)) {
    analysis = true;
  }
  if (analysis.has_value()) {
    LeafTypeTree<IntervalSet> result(le->GetType());
    if (analysis == std::optional<bool>(true)) {
      result.Set({}, TrueIntervalSet());
    } else if (analysis == std::optional<bool>(false)) {
      result.Set({}, FalseIntervalSet());
    }
    SetIntervalSetTree(le, result);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleULt(CompareOp* lt) {
  INITIALIZE_OR_SKIP(lt);
  IntervalSet lhs_intervals = GetIntervalSetTree(lt->operand(0)).Get({});
  IntervalSet rhs_intervals = GetIntervalSetTree(lt->operand(1)).Get({});

  std::optional<bool> analysis = AnalyzeLt(lhs_intervals, rhs_intervals);
  if (analysis.has_value()) {
    LeafTypeTree<IntervalSet> result(lt->GetType());
    if (analysis == std::optional<bool>(true)) {
      result.Set({}, TrueIntervalSet());
    } else if (analysis == std::optional<bool>(false)) {
      result.Set({}, FalseIntervalSet());
    }
    SetIntervalSetTree(lt, result);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleUMod(BinOp* mod) {
  INITIALIZE_OR_SKIP(mod);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleUMul(ArithOp* mul) {
  INITIALIZE_OR_SKIP(mul);
  // NB this is only monotone given we are checking for the overflow condition
  // inside the lambda. Otherwise it would not be monotonic.  Only provably
  // non-overflowing multiplies can be handled properly.
  // TODO(allight) 2023-08-10 google/xls#1098 We should consider replacing
  // potentially overflowing multiplies with multiply followed by truncate,
  // after which this code will work well automatically.
  return HandleMonotoneMonotoneBinOp(
      [mul](const Bits& x, const Bits& y) -> std::optional<Bits> {
        int64_t bit_count = mul->GetType()->GetFlatBitCount();
        Bits result = bits_ops::UMul(x, y);
        if (result.FitsInNBitsUnsigned(bit_count)) {
          return bits_ops::ZeroExtend(bits_ops::DropLeadingZeroes(result),
                                      bit_count);
        }
        return std::nullopt;
      },
      mul);
}

absl::Status RangeQueryVisitor::HandleUMulp(PartialProductOp* mul) {
  INITIALIZE_OR_SKIP(mul);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleXorReduce(
    BitwiseReductionOp* xor_reduce) {
  INITIALIZE_OR_SKIP(xor_reduce);
  // XorReduce determines the parity of the number of 1s in a bitstring.
  // Incrementing a bitstring always outputs in a bitstring with a different
  // parity of 1s (since even + 1 = odd and odd + 1 = even). Therefore, this
  // analysis cannot return anything but unknown when an interval is imprecise.
  // When the given set of intervals only contains precise intervals, we can
  // check whether they all have the same parity of 1s, and return 1 or 0 if
  // they are all the same, or unknown otherwise.
  IntervalSet input_intervals =
      GetIntervalSetTree(xor_reduce->operand(0)).Get({});
  std::optional<Bits> output;
  for (const Interval& interval : input_intervals.Intervals()) {
    if (!interval.IsPrecise()) {
      return absl::OkStatus();
    }
    Bits value = interval.LowerBound();  // guaranteed to be same as upper bound
    Bits reduced = bits_ops::XorReduce(value);
    if (output.has_value()) {
      if (output.value() != reduced) {
        return absl::OkStatus();
      }
    }
    output = reduced;
  }
  if (output.has_value()) {
    LeafTypeTree<IntervalSet> result(xor_reduce->GetType());
    result.Set({}, IntervalSet::Precise(output.value()));
    SetIntervalSetTree(xor_reduce, result);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleZeroExtend(ExtendOp* zero_ext) {
  INITIALIZE_OR_SKIP(zero_ext);
  return HandleMonotoneUnaryOp(
      [zero_ext](const Bits& bits) -> Bits {
        return bits_ops::ZeroExtend(bits, zero_ext->new_bit_count());
      },
      zero_ext);
}

#undef INITIALIZE_OR_SKIP

// Recursive helper function which writes the given IntervalSetTree to the given
// output stream.
static void IntervalSetTreeToStream(const IntervalSetTree& tree, Type* type,
                                    std::vector<int64_t> index,
                                    std::ostream& os) {
  std::string indent(2 * index.size(), ' ');
  if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    os << absl::StreamFormat("%s[\n", indent);
    for (int64_t i = 0; i < array_type->size(); ++i) {
      index.push_back(i);
      IntervalSetTreeToStream(tree, array_type->element_type(), index, os);
      os << "\n";
      index.pop_back();
    }
    os << absl::StreamFormat("%s]\n", indent);
  } else if (type->IsTuple()) {
    TupleType* tuple_type = type->AsTupleOrDie();
    os << absl::StreamFormat("%s(\n", indent);
    for (int64_t i = 0; i < tuple_type->size(); ++i) {
      index.push_back(i);
      IntervalSetTreeToStream(tree, tuple_type->element_type(i), index, os);
      os << "\n";
      index.pop_back();
    }
    os << absl::StreamFormat("%s)\n", indent);
  } else {
    os << absl::StreamFormat("%s%s", indent, tree.Get(index).ToString());
  }
}

std::string IntervalSetTreeToString(const IntervalSetTree& tree) {
  std::stringstream ss;
  IntervalSetTreeToStream(tree, tree.type(), {}, ss);
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const IntervalSetTree& tree) {
  IntervalSetTreeToStream(tree, tree.type(), {}, os);
  return os;
}

}  // namespace xls
