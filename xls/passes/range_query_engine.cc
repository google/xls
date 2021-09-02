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

#include <limits>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/node_iterator.h"

namespace xls {

class RangeQueryVisitor : public DfsVisitor {
 public:
  explicit RangeQueryVisitor(RangeQueryEngine* engine) : engine_(engine) {}

 private:
  // The maximum size of an interval set that can be resident in memory at any
  // one time. When accumulating a result interval set, if the set exceeds this
  // size, `MinimizeIntervals` will be called to reduce its size at the cost of
  // precision of the analysis.
  static constexpr int64_t kMaxResIntervalSetSize = 64;

  // Handles an operation that is monotone, unary (and whose argument is given
  // by `op->operand(0)`), and has an implementation given by the given
  // function.
  //
  // An operation is monotone iff `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(x), impl(y))`.
  absl::Status HandleMonotoneUnaryOp(std::function<Bits(const Bits&)> impl,
                                     Node* op);

  // Handles an operation that is antitone, unary (and whose argument is given
  // by `op->operand(0)`), and has an implementation given by the given
  // function.
  //
  // An operation is antitone iff `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(y), impl(x))`.
  absl::Status HandleAntitoneUnaryOp(std::function<Bits(const Bits&)> impl,
                                     Node* op);

  // Handles an operation that is binary (and whose arguments are given by
  // `op->operand(0)` and `op->operand(1)`), monotone in both arguments, and
  // has an implementation given by the given function. The function may return
  // `absl::nullopt` to indicate that some exception has occurred, like an
  // integer overflow, which invalidates the analysis. When that happens, this
  // function will return without any side effects having occurred.
  //
  // A binary operation is monotone-monotone iff for every `k`,
  // `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(x, k), impl(y, k))`
  // and `bits_ops::ULessThanOrEqual(impl(k, x), impl(k, y))`.
  absl::Status HandleMonotoneMonotoneBinOp(
      std::function<absl::optional<Bits>(const Bits&, const Bits&)> impl,
      Node* op);

  // Handles an operation that is binary (and whose arguments are given by
  // `op->operand(0)` and `op->operand(1)`), monotone in the first argument
  // and antitone in the second argument, and has an implementation given by
  // the given function. The function may return `absl::nullopt` to indicate
  // that some exception has occurred, like an integer overflow, which
  // invalidates the analysis. When that happens, this function will return
  // without any side effects having occurred.
  //
  // A binary operation is monotone-antitone iff for every `k`,
  // `bits_ops::ULessThanOrEqual(x, y)`
  // implies `bits_ops::ULessThanOrEqual(impl(x, k), impl(y, k))`
  // and `bits_ops::ULessThanOrEqual(impl(k, y), impl(k, x))`.
  absl::Status HandleMonotoneAntitoneBinOp(
      std::function<absl::optional<Bits>(const Bits&, const Bits&)> impl,
      Node* op);

  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
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
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleOutputPort(OutputPort* output_port) override;
  absl::Status HandleParam(Param* param) override;
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
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleShll(BinOp* shll) override;
  absl::Status HandleShra(BinOp* shra) override;
  absl::Status HandleShrl(BinOp* shrl) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* index) override;
  absl::Status HandleUDiv(BinOp* div) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleUMod(BinOp* mod) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

  RangeQueryEngine* engine_;
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
std::vector<MergeInterval> ReduceByMerging(absl::Span<Bits const> elements,
                                           int64_t desired_size) {
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

absl::Status RangeQueryEngine::Populate(FunctionBase* f) {
  RangeQueryVisitor visitor(this);
  XLS_RETURN_IF_ERROR(f->Accept(&visitor));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<RangeQueryEngine>> RangeQueryEngine::Run(
    FunctionBase* f) {
  RangeQueryEngine result;
  XLS_RETURN_IF_ERROR(result.Populate(f));
  return absl::make_unique<RangeQueryEngine>(std::move(result));
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
      IntervalSet interval_set(size);
      interval_set.AddInterval(Interval::Maximal(size));
      interval_set.Normalize();
      result.elements()[i] = interval_set;
      ++i;
    }
    return result;
  }

  int64_t size = node->GetType()->GetFlatBitCount();
  IntervalSet interval_set(size);
  interval_set.AddInterval(Interval::Maximal(size));
  interval_set.Normalize();
  IntervalSetTree result(node->GetType());
  result.Set({}, interval_set);
  return result;
}

void RangeQueryEngine::SetIntervalSetTree(
    Node* node, const IntervalSetTree& interval_sets) {
  int64_t size = node->GetType()->GetFlatBitCount();
  if (node->GetType()->IsBits()) {
    IntervalSet interval_set = interval_sets.Get({});
    XLS_CHECK(interval_set.IsNormalized());
    XLS_CHECK(!interval_set.Intervals().empty());
    Bits lcs = bits_ops::LongestCommonPrefixMSB(
        {interval_set.Intervals().front().LowerBound(),
         interval_set.Intervals().back().UpperBound()});
    known_bits_[node] = bits_ops::Concat(
        {Bits::AllOnes(lcs.bit_count()), Bits(size - lcs.bit_count())});
    known_bit_values_[node] =
        bits_ops::Concat({lcs, Bits(size - lcs.bit_count())});
  }
  interval_sets_[node] = interval_sets;
}

void RangeQueryEngine::InitializeNode(Node* node) {
  if (!known_bits_.contains(node) || !known_bit_values_.contains(node)) {
    known_bits_[node] = Bits(node->GetType()->GetFlatBitCount());
    known_bit_values_[node] = Bits(node->GetType()->GetFlatBitCount());
  }
}

absl::Status RangeQueryVisitor::HandleMonotoneUnaryOp(
    std::function<Bits(const Bits&)> impl, Node* op) {
  IntervalSet input_intervals =
      engine_->GetIntervalSetTree(op->operand(0)).Get({});
  IntervalSet result_intervals(op->BitCountOrDie());
  for (const Interval& input_interval : input_intervals.Intervals()) {
    // The essential property of a unary monotone function `f` is that the
    // codomain of `f` applied to `[x, y]` is `[f(x), f(y)]`. For example,
    // the cubing function applied to `[5, 8]` gives a codomain of `[125, 512]`.
    result_intervals.AddInterval(
        {impl(input_interval.LowerBound()), impl(input_interval.UpperBound())});
  }

  result_intervals = MinimizeIntervals(result_intervals);

  IntervalSetTree result(op->GetType());
  result.Set({}, result_intervals);
  engine_->SetIntervalSetTree(op, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleAntitoneUnaryOp(
    std::function<Bits(const Bits&)> impl, Node* op) {
  IntervalSet input_intervals =
      engine_->GetIntervalSetTree(op->operand(0)).Get({});
  IntervalSet result_intervals(op->BitCountOrDie());
  for (const Interval& input_interval : input_intervals.Intervals()) {
    // The essential property of a unary antitone function `f` is that the
    // codomain of `f` applied to `[x, y]` is `[f(y), f(x)]`. For example,
    // the negation function applied to `[10, 20]` gives an codomain of
    // `[-20, -10]`.
    result_intervals.AddInterval(
        {impl(input_interval.UpperBound()), impl(input_interval.LowerBound())});
  }

  result_intervals = MinimizeIntervals(result_intervals);

  IntervalSetTree result(op->GetType());
  result.Set({}, result_intervals);
  engine_->SetIntervalSetTree(op, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleMonotoneMonotoneBinOp(
    std::function<absl::optional<Bits>(const Bits&, const Bits&)> impl,
    Node* op) {
  IntervalSet lhs_intervals =
      engine_->GetIntervalSetTree(op->operand(0)).Get({});
  IntervalSet rhs_intervals =
      engine_->GetIntervalSetTree(op->operand(1)).Get({});
  IntervalSet result_intervals(op->BitCountOrDie());
  for (const Interval& lhs_interval : lhs_intervals.Intervals()) {
    for (const Interval& rhs_interval : rhs_intervals.Intervals()) {
      // The essential property of a binary function `f` that is monotone in
      // both arguments is that if the first argument has range `[xₗ, xᵤ]` and
      // the second argument has range `[yₗ, yᵤ]`, then the image is contained
      // in `[f(xₗ, yₗ), f(xᵤ, yᵤ)]`. For example, the addition function applied
      // to `[20, 30]` and `[5, 10]` gives an image of `[25, 40]`.
      absl::optional<Bits> lower =
          impl(lhs_interval.LowerBound(), rhs_interval.LowerBound());
      absl::optional<Bits> upper =
          impl(lhs_interval.UpperBound(), rhs_interval.UpperBound());
      if (!lower.has_value() || !upper.has_value()) {
        return absl::OkStatus();
      }
      result_intervals.AddInterval({lower.value(), upper.value()});
      if (result_intervals.NumberOfIntervals() > kMaxResIntervalSetSize) {
        result_intervals = MinimizeIntervals(result_intervals);
      }
    }
  }

  result_intervals = MinimizeIntervals(result_intervals);

  IntervalSetTree result(op->GetType());
  result.Set({}, result_intervals);
  engine_->SetIntervalSetTree(op, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleMonotoneAntitoneBinOp(
    std::function<absl::optional<Bits>(const Bits&, const Bits&)> impl,
    Node* op) {
  IntervalSet lhs_intervals =
      engine_->GetIntervalSetTree(op->operand(0)).Get({});
  IntervalSet rhs_intervals =
      engine_->GetIntervalSetTree(op->operand(1)).Get({});
  IntervalSet result_intervals(op->BitCountOrDie());
  for (const Interval& lhs_interval : lhs_intervals.Intervals()) {
    for (const Interval& rhs_interval : rhs_intervals.Intervals()) {
      // The essential property of a binary function `f` that is monotone in
      // the first argument and antitone in the second is that if the first
      // argument has range `[xₗ, xᵤ]` and the second argument has range
      // `[yₗ, yᵤ]`, then the image is contained in `[f(xₗ, yᵤ), f(xᵤ, yₗ)]`.
      // For example, the subtraction function applied to `[20, 30]` and
      // `[5, 10]` gives an image of `[10, 25]`.
      //
      // The proof goes as follows:
      // 1. Assume we have arbitrary x ∈ [xₗ, xᵤ], y ∈ [yₗ, yᵤ], and try to show
      //   that `f(x, y) ∈ [f(xₗ, yᵤ), f(xᵤ, yₗ)]`
      // 2. The definition of monotonicity in the first argument gives
      //    ∀ k. (p ≤ q) ⇒ (f(p, k) ≤ f(q, k))
      // 3. Instantiating that at `p = xₗ` and `q = x` and using `xₗ ≤ x` gives
      //    ∀ k. f(xₗ, k) ≤ f(x, k)
      // 4. Instantiating that at `p = x` and `q = xᵤ` and using `x ≤ xᵤ` gives
      //    ∀ k. f(x, k) ≤ f(xᵤ, k)
      // 5. Together, these imply that ∀ k. f(x, k) ∈ [f(xₗ, k), f(xᵤ, k)]
      // 6. The definition of antitonicity in the second argument gives
      //    ∀ m. (p ≤ q) ⇒ (f(m, q) ≤ f(m, p))
      // 7. Instantiating that at `p = yₗ` and `q = y` and using `yₗ ≤ y` gives
      //    ∀ m. f(m, y) ≤ f(m, yₗ)
      // 8. Instantiating that at `p = y` and `q = yᵤ` and using `y ≤ yᵤ` gives
      //    ∀ m. f(m, yᵤ) ≤ f(m, y)
      // 9. Together, these imply that ∀ m. f(m, y) ∈ [f(m, yᵤ), f(m, yₗ)]
      // 10. Between (5) and (9) we can conclude that
      //     f(x, y) ∈ [f(xₗ, yᵤ), f(xᵤ, yₗ)]
      absl::optional<Bits> lower =
          impl(lhs_interval.LowerBound(), rhs_interval.UpperBound());
      absl::optional<Bits> upper =
          impl(lhs_interval.UpperBound(), rhs_interval.LowerBound());
      if (!lower.has_value() || !upper.has_value()) {
        return absl::OkStatus();
      }
      result_intervals.AddInterval({lower.value(), upper.value()});
      if (result_intervals.NumberOfIntervals() > kMaxResIntervalSetSize) {
        result_intervals = MinimizeIntervals(result_intervals);
      }
    }
  }

  result_intervals = MinimizeIntervals(result_intervals);

  IntervalSetTree result(op->GetType());
  result.Set({}, result_intervals);
  engine_->SetIntervalSetTree(op, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleAdd(BinOp* add) {
  engine_->InitializeNode(add);
  return HandleMonotoneMonotoneBinOp(
      [](const Bits& lhs, const Bits& rhs) -> absl::optional<Bits> {
        int64_t padded_size = std::max(lhs.bit_count(), rhs.bit_count()) + 1;
        Bits padded_lhs = bits_ops::ZeroExtend(lhs, padded_size);
        Bits padded_rhs = bits_ops::ZeroExtend(rhs, padded_size);
        Bits padded_result = bits_ops::Add(padded_lhs, padded_rhs);
        // If the MSB is 1, then we overflowed.
        if (padded_result.msb()) {
          return absl::nullopt;
        }
        return bits_ops::Add(lhs, rhs);
      },
      add);
}

absl::Status RangeQueryVisitor::HandleAfterAll(AfterAll* after_all) {
  engine_->InitializeNode(after_all);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleAndReduce(
    BitwiseReductionOp* and_reduce) {
  engine_->InitializeNode(and_reduce);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArray(Array* array) {
  engine_->InitializeNode(array);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArrayConcat(ArrayConcat* array_concat) {
  engine_->InitializeNode(array_concat);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleAssert(Assert* assert_op) {
  engine_->InitializeNode(assert_op);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleBitSlice(BitSlice* bit_slice) {
  engine_->InitializeNode(bit_slice);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleBitSliceUpdate(BitSliceUpdate* update) {
  engine_->InitializeNode(update);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleConcat(Concat* concat) {
  engine_->InitializeNode(concat);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleCountedFor(CountedFor* counted_for) {
  engine_->InitializeNode(counted_for);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleCover(Cover* cover) {
  engine_->InitializeNode(cover);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleDecode(Decode* decode) {
  engine_->InitializeNode(decode);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  engine_->InitializeNode(dynamic_bit_slice);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleDynamicCountedFor(
    DynamicCountedFor* dynamic_counted_for) {
  engine_->InitializeNode(dynamic_counted_for);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleEncode(Encode* encode) {
  engine_->InitializeNode(encode);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleEq(CompareOp* eq) {
  engine_->InitializeNode(eq);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleGate(Gate* gate) {
  engine_->InitializeNode(gate);
  return HandleMonotoneMonotoneBinOp(
      [](const Bits& cond, const Bits& value) -> absl::optional<Bits> {
        return value;
      },
      gate);
}

absl::Status RangeQueryVisitor::HandleIdentity(UnOp* identity) {
  engine_->InitializeNode(identity);
  return HandleMonotoneUnaryOp([](const Bits& b) { return b; }, identity);
}

absl::Status RangeQueryVisitor::HandleInputPort(InputPort* input_port) {
  engine_->InitializeNode(input_port);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleInvoke(Invoke* invoke) {
  engine_->InitializeNode(invoke);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleLiteral(Literal* literal) {
  engine_->InitializeNode(literal);
  Value v = literal->value();
  if (!v.IsBits()) {
    return absl::OkStatus();
  }
  IntervalSet result_intervals(literal->BitCountOrDie());
  result_intervals.AddInterval(Interval(v.bits(), v.bits()));
  result_intervals.Normalize();
  IntervalSetTree result(literal->GetType());
  result.Set({}, result_intervals);
  engine_->SetIntervalSetTree(literal, result);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleMap(Map* map) {
  engine_->InitializeNode(map);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArrayIndex(ArrayIndex* index) {
  engine_->InitializeNode(index);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArraySlice(ArraySlice* slice) {
  engine_->InitializeNode(slice);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArrayUpdate(ArrayUpdate* update) {
  engine_->InitializeNode(update);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryAnd(NaryOp* and_op) {
  engine_->InitializeNode(and_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryNand(NaryOp* nand_op) {
  engine_->InitializeNode(nand_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryNor(NaryOp* nor_op) {
  engine_->InitializeNode(nor_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryOr(NaryOp* or_op) {
  engine_->InitializeNode(or_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNaryXor(NaryOp* xor_op) {
  engine_->InitializeNode(xor_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNe(CompareOp* ne) {
  engine_->InitializeNode(ne);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleNeg(UnOp* neg) {
  engine_->InitializeNode(neg);
  return HandleAntitoneUnaryOp(bits_ops::Negate, neg);
}

absl::Status RangeQueryVisitor::HandleNot(UnOp* not_op) {
  engine_->InitializeNode(not_op);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleOneHot(OneHot* one_hot) {
  engine_->InitializeNode(one_hot);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleOneHotSel(OneHotSelect* sel) {
  engine_->InitializeNode(sel);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleOrReduce(BitwiseReductionOp* or_reduce) {
  engine_->InitializeNode(or_reduce);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleOutputPort(OutputPort* output_port) {
  engine_->InitializeNode(output_port);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleParam(Param* param) {
  engine_->InitializeNode(param);
  // We don't know anything about params.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleReceive(Receive* receive) {
  engine_->InitializeNode(receive);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleRegisterRead(RegisterRead* reg_read) {
  engine_->InitializeNode(reg_read);
  return absl::OkStatus();  // TODO(taktoa): implement: needs fixed point
}

absl::Status RangeQueryVisitor::HandleRegisterWrite(RegisterWrite* reg_write) {
  engine_->InitializeNode(reg_write);
  return absl::OkStatus();  // TODO(taktoa): implement: needs fixed point
}

absl::Status RangeQueryVisitor::HandleReverse(UnOp* reverse) {
  engine_->InitializeNode(reverse);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleSDiv(BinOp* div) {
  engine_->InitializeNode(div);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSGe(CompareOp* ge) {
  engine_->InitializeNode(ge);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSGt(CompareOp* gt) {
  engine_->InitializeNode(gt);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSLe(CompareOp* le) {
  engine_->InitializeNode(le);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSLt(CompareOp* lt) {
  engine_->InitializeNode(lt);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSMod(BinOp* mod) {
  engine_->InitializeNode(mod);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSMul(ArithOp* mul) {
  engine_->InitializeNode(mul);
  return absl::OkStatus();  // TODO(taktoa): implement: signed
}

absl::Status RangeQueryVisitor::HandleSel(Select* sel) {
  engine_->InitializeNode(sel);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleSend(Send* send) {
  engine_->InitializeNode(send);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleShll(BinOp* shll) {
  engine_->InitializeNode(shll);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleShra(BinOp* shra) {
  engine_->InitializeNode(shra);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleShrl(BinOp* shrl) {
  engine_->InitializeNode(shrl);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleSignExtend(ExtendOp* sign_ext) {
  engine_->InitializeNode(sign_ext);
  return HandleMonotoneUnaryOp(
      [sign_ext](const Bits& bits) -> Bits {
        return bits_ops::SignExtend(bits, sign_ext->new_bit_count());
      },
      sign_ext);
}

absl::Status RangeQueryVisitor::HandleSub(BinOp* sub) {
  engine_->InitializeNode(sub);
  return HandleMonotoneAntitoneBinOp(
      [](const Bits& lhs, const Bits& rhs) -> absl::optional<Bits> {
        if (bits_ops::ULessThanOrEqual(rhs, lhs)) {
          return bits_ops::Sub(lhs, rhs);
        }
        return absl::nullopt;
      },
      sub);
}

absl::Status RangeQueryVisitor::HandleTuple(Tuple* tuple) {
  engine_->InitializeNode(tuple);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleTupleIndex(TupleIndex* index) {
  engine_->InitializeNode(index);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleUDiv(BinOp* div) {
  engine_->InitializeNode(div);
  // I (taktoa) verified on 8 bit unsigned integers that UDiv is antitone in
  // its second argument.
  return HandleMonotoneAntitoneBinOp(
      [](const Bits& lhs, const Bits& rhs) -> absl::optional<Bits> {
        return bits_ops::UDiv(lhs, rhs);
      },
      div);
}

absl::Status RangeQueryVisitor::HandleUGe(CompareOp* ge) {
  engine_->InitializeNode(ge);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleUGt(CompareOp* gt) {
  engine_->InitializeNode(gt);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleULe(CompareOp* le) {
  engine_->InitializeNode(le);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleULt(CompareOp* lt) {
  engine_->InitializeNode(lt);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleUMod(BinOp* mod) {
  engine_->InitializeNode(mod);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleUMul(ArithOp* mul) {
  engine_->InitializeNode(mul);
  // Only provably non-overflowing multiplies can be handled properly.
  // Hopefully soonish we'll replace overflowing multiplies with multiply
  // followed by truncate, after which this code will work well automatically.
  if (mul->GetType()->GetFlatBitCount() >=
      (mul->operand(0)->GetType()->GetFlatBitCount() +
       mul->operand(1)->GetType()->GetFlatBitCount())) {
    return HandleMonotoneMonotoneBinOp(
        [mul](const Bits& x, const Bits& y) -> absl::optional<Bits> {
          return bits_ops::ZeroExtend(bits_ops::UMul(x, y),
                                      mul->GetType()->GetFlatBitCount());
        },
        mul);
  }
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleXorReduce(
    BitwiseReductionOp* xor_reduce) {
  engine_->InitializeNode(xor_reduce);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleZeroExtend(ExtendOp* zero_ext) {
  engine_->InitializeNode(zero_ext);
  return HandleMonotoneUnaryOp(
      [zero_ext](const Bits& bits) -> Bits {
        return bits_ops::ZeroExtend(bits, zero_ext->new_bit_count());
      },
      zero_ext);
}

}  // namespace xls
