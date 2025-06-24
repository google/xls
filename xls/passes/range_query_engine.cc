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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
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
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/query_engine.h"

namespace xls {

namespace {

IntervalSetTree UnconstrainedIntervalSetTree(Type* type) {
  return *IntervalSetTree::CreateFromFunction(
      type, [](Type* leaf_type) -> absl::StatusOr<IntervalSet> {
        return IntervalSet::Maximal(leaf_type->GetFlatBitCount());
      });
}

IntervalSetTree EmptyIntervalSetTree(Type* type) {
  return *IntervalSetTree::CreateFromFunction(
      type, [](Type* leaf_type) -> absl::StatusOr<IntervalSet> {
        return IntervalSet(leaf_type->GetFlatBitCount());
      });
}

IntervalSetTree ZeroIntervalSetTree(Type* type) {
  return *IntervalSetTree::CreateFromFunction(
      type, [](Type* leaf_type) -> absl::StatusOr<IntervalSet> {
        return IntervalSet::Precise(UBits(0, leaf_type->GetFlatBitCount()));
      });
}

}  // namespace

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

  // What size we should minimize interval sets to by default.
  static constexpr int64_t kDefaultIntervalSize = 16;

  // The maximum number of points covered by an interval set that can be
  // iterated over in an analysis.
  static constexpr int64_t kMaxIterationSize = 1024;

  // Wrapper around GetIntervalSetTree for consistency with the
  // SetIntervalSetTree wrapper.
  IntervalSetTree GetIntervalSetTree(Node* node) const {
    return engine_->GetIntervalSetTree(node);
  }

  // Wrapper around GetIntervalSetTreeView. Returns std::nullopt if there is no
  // existing tree (indicating the node is unconstrained).
  std::optional<IntervalSetTreeView> MaybeGetIntervalSetTreeView(Node* node) {
    if (engine_->HasExplicitIntervals(node)) {
      absl::StatusOr<IntervalSetTreeView> view =
          engine_->GetIntervalSetTreeView(node);
      CHECK_OK(view) << node << " has explicit intervals but no tree-view!";
      return *view;
    }
    return std::nullopt;
  }

  // Wrapper that avoids copying interval-sets.
  absl::StatusOr<std::optional<std::reference_wrapper<const IntervalSet>>>
  GetIntervalSet(Node* node) const {
    if (!engine_->HasExplicitIntervals(node)) {
      return std::nullopt;
    }
    XLS_ASSIGN_OR_RETURN(IntervalSetTreeView view,
                         engine_->GetIntervalSetTreeView(node));
    XLS_RET_CHECK(node->GetType()->IsBits());
    return std::ref(view.Get({}));
  }

  // Wrapper around engine_->SetIntervalSetTree that modifies rf_ if necessary.
  void SetIntervalSetTree(Node* node, const IntervalSetTree& interval_sets) {
    if (!engine_->interval_sets_.contains(node)) {
      for (const IntervalSet& set : interval_sets.elements()) {
        if (!set.IsMaximal()) {
          rf_ = ReachedFixpoint::Changed;
          break;
        }
      }
      engine_->SetIntervalSetTree(node, interval_sets);
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
  void SetIntervalSetTree(Node* node, IntervalSetTree&& interval_sets) {
    if (!engine_->interval_sets_.contains(node)) {
      for (const IntervalSet& set : interval_sets.elements()) {
        if (!set.IsMaximal()) {
          rf_ = ReachedFixpoint::Changed;
          break;
        }
      }
      engine_->SetIntervalSetTree(node, std::move(interval_sets));
      return;
    }

    // In the event of a hash collision, it is possible that this could report
    // having reached a fixed point when it hasn't actually. However, this is
    // fairly harmless and very unlikely.
    size_t hash_before =
        absl::Hash<IntervalSetTree>()(engine_->interval_sets_.at(node));
    engine_->SetIntervalSetTree(node, std::move(interval_sets));
    size_t hash_after =
        absl::Hash<IntervalSetTree>()(engine_->interval_sets_.at(node));
    if (!(hash_before == hash_after)) {
      rf_ = ReachedFixpoint::Changed;
    }
  }

  absl::Status SetIntervalSet(Node* node, IntervalSet is) {
    XLS_RET_CHECK(node->GetType()->IsBits());
    SetIntervalSetTree(node, IntervalSetTree::CreateSingleElementTree(
                                 node->GetType(), std::move(is)));
    return absl::OkStatus();
  }

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
  absl::Status HandleStateRead(StateRead* state_read) override;
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
  return UnconstrainedIntervalSetTree(node->GetType());
}

void RangeQueryEngine::SetIntervalSetTree(
    Node* node, const IntervalSetTree& interval_sets) {
  auto [it, inserted] = interval_sets_.try_emplace(node, interval_sets);
  MutableIntervalSetTreeView ist = it->second.AsMutableView();

  if (!inserted) {
    // We already had information for `node`; merge with `interval_sets`.
    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        ist, interval_sets.AsView(),
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Intersect(lhs, rhs);
        });
  }

  if (node->GetType()->IsBits()) {
    interval_ops::KnownBits bits =
        interval_ops::ExtractKnownBits(ist.Get({}), /*source=*/node);
    known_bits_[node] = bits.known_bits;
    known_bit_values_[node] = bits.known_bit_values;
  }
}

void RangeQueryEngine::SetIntervalSetTree(Node* node,
                                          IntervalSetTree&& interval_sets) {
  auto [it, inserted] = interval_sets_.try_emplace(node, interval_sets);
  MutableIntervalSetTreeView ist = it->second.AsMutableView();

  if (!inserted) {
    // We already had information for `node`; merge with `interval_sets`.
    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        ist, interval_sets.AsView(),
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Intersect(lhs, rhs);
        });
  }

  if (node->GetType()->IsBits()) {
    interval_ops::KnownBits bits =
        interval_ops::ExtractKnownBits(ist.Get({}), /*source=*/node);
    known_bits_[node] = bits.known_bits;
    known_bit_values_[node] = bits.known_bit_values;
  }
}

void RangeQueryEngine::InitializeNode(Node* node) {
  if (!known_bits_.contains(node) || !known_bit_values_.contains(node)) {
    known_bits_[node] = Bits(node->GetType()->GetFlatBitCount());
    known_bit_values_[node] = Bits(node->GetType()->GetFlatBitCount());
  }
}

// Helper to just pull result straight from givens if possible.
#define INITIALIZE_OR_SKIP(node)   \
  do {                             \
    if (SetIfGiven(node)) {        \
      return absl::OkStatus();     \
    }                              \
    engine_->InitializeNode(node); \
  } while (false)

#define ASSIGN_INTERVAL_SET_REF_OR_RETURN(target, source)                      \
  XLS_ASSIGN_OR_RETURN(auto __##target##_TEMPORARY, GetIntervalSet(source));   \
  IntervalSet __memory_##target##_TEMPORARY;                                   \
  if (!__##target##_TEMPORARY) {                                               \
    __memory_##target##_TEMPORARY =                                            \
        IntervalSet::Maximal(source->GetType()->GetFlatBitCount());            \
  }                                                                            \
  const IntervalSet& target =                                                  \
      __##target##_TEMPORARY.value_or(std::ref(__memory_##target##_TEMPORARY)) \
          .get()

absl::Status RangeQueryVisitor::HandleAdd(BinOp* add) {
  INITIALIZE_OR_SKIP(add);
  Node* lhs = add->operand(0);
  Node* rhs = add->operand(1);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, lhs);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, rhs);
  // Special case UMulp arguments.
  if (lhs->Is<TupleIndex>() && rhs->Is<TupleIndex>() &&
      lhs->As<TupleIndex>()->index() != rhs->As<TupleIndex>()->index() &&
      lhs->operand(0) == rhs->operand(0) &&
      lhs->operand(0)->op() == Op::kUMulp) {
    return SetIntervalSet(
        add, interval_ops::UMul(l, r, add->GetType()->GetFlatBitCount()));
  }

  return SetIntervalSet(add, interval_ops::Add(l, r));
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
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, and_reduce->operand(0));
  return SetIntervalSet(and_reduce, interval_ops::AndReduce(arg));
}

absl::Status RangeQueryVisitor::HandleArray(Array* array) {
  INITIALIZE_OR_SKIP(array);
  std::vector<IntervalSetTreeView> children;
  std::vector<std::unique_ptr<IntervalSetTree>> temp_storage;
  children.reserve(array->operand_count());
  for (Node* element : array->operands()) {
    std::optional<IntervalSetTreeView> child =
        MaybeGetIntervalSetTreeView(element);
    if (!child.has_value()) {
      temp_storage.push_back(std::make_unique<IntervalSetTree>(
          UnconstrainedIntervalSetTree(element->GetType())));
      child = temp_storage.back()->AsView();
    }
    children.push_back(*child);
  }

  XLS_ASSIGN_OR_RETURN(IntervalSetTree result,
                       leaf_type_tree::CreateArray<IntervalSet>(
                           array->GetType()->AsArrayOrDie(), children));
  SetIntervalSetTree(array, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleArrayConcat(ArrayConcat* array_concat) {
  INITIALIZE_OR_SKIP(array_concat);
  std::vector<IntervalSetTreeView> children;
  std::vector<std::unique_ptr<IntervalSetTree>> temp_storage;
  children.reserve(array_concat->operand_count());
  absl::c_transform(
      array_concat->operands(), std::back_inserter(children), [&](Node* n) {
        std::optional<IntervalSetTreeView> view =
            MaybeGetIntervalSetTreeView(n);
        if (!view.has_value()) {
          temp_storage.push_back(std::make_unique<IntervalSetTree>(
              UnconstrainedIntervalSetTree(n->GetType())));
          view = temp_storage.back()->AsView();
        }
        return *view;
      });

  XLS_ASSIGN_OR_RETURN(IntervalSetTree result,
                       leaf_type_tree::ConcatArray<IntervalSet>(
                           array_concat->GetType()->AsArrayOrDie(), children));
  SetIntervalSetTree(array_concat, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleAssert(Assert* assert_op) {
  INITIALIZE_OR_SKIP(assert_op);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleBitSlice(BitSlice* bit_slice) {
  INITIALIZE_OR_SKIP(bit_slice);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(a, bit_slice->operand(0));
  return SetIntervalSet(bit_slice, interval_ops::BitSlice(a, bit_slice->start(),
                                                          bit_slice->width()));
}

absl::Status RangeQueryVisitor::HandleBitSliceUpdate(BitSliceUpdate* update) {
  INITIALIZE_OR_SKIP(update);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleConcat(Concat* concat) {
  INITIALIZE_OR_SKIP(concat);
  std::vector<IntervalSet> args;
  args.reserve(concat->operand_count());
  for (Node* arg : concat->operands()) {
    ASSIGN_INTERVAL_SET_REF_OR_RETURN(a, arg);
    args.push_back(a);
  }
  return SetIntervalSet(concat, interval_ops::Concat(args));
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
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(a, decode->operand(0));

  return SetIntervalSet(decode,
                        interval_ops::MinimizeIntervals(
                            interval_ops::Decode(a, decode->BitCountOrDie()),
                            kDefaultIntervalSize));
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

  if (eq->operand(0)->GetType()->GetFlatBitCount() == 0) {
    // Empty types are always equal.
    return SetIntervalSet(eq, IntervalSet::Precise(UBits(1, 1)));
  }

  std::optional<IntervalSetTreeView> lhs_intervals =
      MaybeGetIntervalSetTreeView(eq->operand(0));
  std::optional<IntervalSetTreeView> rhs_intervals =
      MaybeGetIntervalSetTreeView(eq->operand(1));
  if (!lhs_intervals.has_value() || !rhs_intervals.has_value()) {
    // One input is unconstrained, so we know nothing about the result.
    return SetIntervalSet(eq, IntervalSet::Maximal(1));
  }
  XLS_RET_CHECK_EQ(lhs_intervals->type(), rhs_intervals->type());

  IntervalSet res(/*bit_count=*/1);
  for (int64_t i = 0; i < lhs_intervals->size(); ++i) {
    res = IntervalSet::Combine(res,
                               interval_ops::Eq(lhs_intervals->elements()[i],
                                                rhs_intervals->elements()[i]));
  }
  return SetIntervalSet(eq, std::move(res));
}

absl::Status RangeQueryVisitor::HandleGate(Gate* gate) {
  INITIALIZE_OR_SKIP(gate);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(cond_intervals, gate->operand(0));

  std::optional<IntervalSetTreeView> value =
      MaybeGetIntervalSetTreeView(gate->operand(1));
  std::optional<IntervalSetTree> temp_value;
  if (!value.has_value()) {
    temp_value = UnconstrainedIntervalSetTree(gate->operand(1)->GetType());
    value = temp_value->AsView();
  }

  SetIntervalSetTree(gate, leaf_type_tree::Map<IntervalSet, IntervalSet>(
                               *value, [&](const IntervalSet& is) {
                                 return interval_ops::Gate(cond_intervals, is);
                               }));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleIdentity(UnOp* identity) {
  INITIALIZE_OR_SKIP(identity);
  SetIntervalSetTree(identity, GetIntervalSetTree(identity->operand(0)));
  return absl::OkStatus();
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
  SetIntervalSetTree(literal,
                     leaf_type_tree::Map<IntervalSet, Value>(
                         v_ltt.AsView(), [](const Value& value) {
                           IntervalSet interval_set(value.GetFlatBitCount());
                           if (value.IsBits()) {
                             return IntervalSet::Precise(value.bits());
                           }
                           if (value.IsToken()) {
                             return IntervalSet::Precise(Bits(0));
                           }
                           LOG(FATAL) << "Invalid value kind in HandleLiteral";
                         }));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleMap(Map* map) {
  INITIALIZE_OR_SKIP(map);
  return absl::OkStatus();  // TODO(taktoa): implement
}

absl::Status RangeQueryVisitor::HandleArrayIndex(ArrayIndex* array_index) {
  INITIALIZE_OR_SKIP(array_index);

  if (array_index->indices().empty()) {
    SetIntervalSetTree(array_index, GetIntervalSetTree(array_index->array()));
    return absl::OkStatus();
  }

  std::optional<IntervalSetTreeView> array_interval_set_tree =
      MaybeGetIntervalSetTreeView(array_index->array());
  if (!array_interval_set_tree.has_value()) {
    // The array is unconstrained, so we can't tell what values we might get.
    SetIntervalSetTree(array_index,
                       UnconstrainedIntervalSetTree(array_index->GetType()));
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
      std::optional<IntervalSetTreeView> index_view =
          MaybeGetIntervalSetTreeView(index);
      if (index_view.has_value()) {
        index_intervals.push_back(index_view->Get({}));
      } else {
        index_intervals.push_back(IntervalSet::Maximal(index->BitCountOrDie()));
      }
      absl::StatusOr<ArrayType*> array_type_status =
          array_type->element_type()->AsArray();
      array_type = array_type_status.ok() ? *array_type_status : nullptr;
    }
  }

  IntervalSetTree result = EmptyIntervalSetTree(array_index->GetType());

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
        result.AsMutableView(), array_interval_set_tree->AsView(indexes),
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Combine(lhs, rhs);
        });
    return false;
  });

  SetIntervalSetTree(array_index, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleArraySlice(ArraySlice* slice) {
  INITIALIZE_OR_SKIP(slice);

  std::optional<IntervalSetTreeView> array_interval_set_tree =
      MaybeGetIntervalSetTreeView(slice->array());
  if (!array_interval_set_tree.has_value()) {
    // The array is unconstrained, so we can't tell what values might be in our
    // slice.
    SetIntervalSetTree(slice, UnconstrainedIntervalSetTree(slice->GetType()));
    return absl::OkStatus();
  }

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(start_interval, slice->start());

  IntervalSetTree result = EmptyIntervalSetTree(slice->GetType());
  absl::Status status;
  start_interval.ForEachElement([&](const Bits& v) -> bool {
    // XLS arrays can't be bigger than this anyway.
    int64_t start = v.FitsInUint64() ? v.ToUint64().value()
                                     : std::numeric_limits<int64_t>::max();
    if (start < 0) {
      // Overflows, clamp back to max int.
      start = std::numeric_limits<int64_t>::max();
    }
    auto slice_ltt = leaf_type_tree::SliceArray<IntervalSet>(
        slice->GetType()->AsArrayOrDie(), array_interval_set_tree->AsView(),
        start);
    if (!slice_ltt.ok()) {
      status.Update(slice_ltt.status());
      return true;
    }
    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        result.AsMutableView(), slice_ltt->AsView(),
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Combine(lhs, rhs);
        });
    return start >= slice->array()->GetType()->AsArrayOrDie()->size();
  });
  XLS_RETURN_IF_ERROR(status);

  SetIntervalSetTree(slice, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleArrayUpdate(ArrayUpdate* update) {
  INITIALIZE_OR_SKIP(update);

  // Record IntervalSets for all indices, and check if they are all precise, all
  // unconstrained, or something else.
  std::vector<std::optional<IntervalSet>> indices;
  indices.reserve(update->indices().size());
  bool all_indices_unconstrained = true;
  bool all_indices_precise = true;
  Type* indexed_type = update->array_to_update()->GetType();
  for (Node* index : update->indices()) {
    ASSIGN_INTERVAL_SET_REF_OR_RETURN(index_intervals, index);

    if (index_intervals.IsMaximal()) {
      all_indices_precise = false;
      if (index_intervals.BitCount() <
          Bits::MinBitCountUnsigned(indexed_type->AsArrayOrDie()->size() - 1)) {
        // Since the interval can only address 2**bit_count items and the array
        // is more than that long the index is constrained.
        all_indices_unconstrained = false;
        indices.push_back(index_intervals);
      } else {
        indices.push_back(std::nullopt);
      }
    } else {
      if (absl::StatusOr<uint64_t> index_lb =
              index_intervals.LowerBound()->ToUint64();
          !index_lb.ok() ||
          *index_lb >=
              static_cast<uint64_t>(indexed_type->AsArrayOrDie()->size())) {
        // Definitely an out-of-bounds access, so no change.
        SetIntervalSetTree(update,
                           GetIntervalSetTree(update->array_to_update()));
        return absl::OkStatus();
      }

      all_indices_unconstrained = false;
      if (!index_intervals.IsPrecise()) {
        all_indices_precise = false;
      }
      indices.push_back(index_intervals);
    }

    indexed_type = indexed_type->AsArrayOrDie()->element_type();
  }

  std::optional<IntervalSetTreeView> update_value =
      MaybeGetIntervalSetTreeView(update->update_value());
  std::optional<IntervalSetTree> temp_update_value;
  if (!update_value.has_value() && all_indices_unconstrained) {
    // The update is fully unconstrained, so we can't tell what values could
    // be in the array anymore.
    SetIntervalSetTree(update, UnconstrainedIntervalSetTree(update->GetType()));
    return absl::OkStatus();
  }

  IntervalSetTree result = GetIntervalSetTree(update->array_to_update());

  if (all_indices_precise) {
    // We know exactly which element is being updated, so we can just replace
    // its possible values with the possible values from `update_value`.
    std::vector<int64_t> precise_index;
    precise_index.reserve(indices.size());
    for (const std::optional<IntervalSet>& index : indices) {
      precise_index.push_back(
          static_cast<int64_t>(*index->GetPreciseValue()->ToUint64()));
    }
    if (update_value.has_value()) {
      XLS_RETURN_IF_ERROR(leaf_type_tree::ReplaceElements(
          result.AsMutableView(precise_index), update_value->AsView()));
    } else {
      // The update value is unconstrained, so remove all constraints on the
      // updated element.
      leaf_type_tree::ForEach(
          result.AsMutableView(precise_index),
          [&](IntervalSet& is) { is = IntervalSet::Maximal(is.BitCount()); });
    }
  } else {
    XLS_RETURN_IF_ERROR(leaf_type_tree::ForEachSubArray<IntervalSet>(
        result.AsMutableView(), indices.size(),
        [&](MutableIntervalSetTreeView element_tree,
            absl::Span<const int64_t> index) {
          for (int64_t i = 0; i < index.size(); ++i) {
            if (indices[i].has_value() &&
                (!UBits(index[i], 64)
                      .FitsInNBitsUnsigned(indices[i]->BitCount()) ||
                 !indices[i]->Covers(
                     UBits(index[i], indices[i]->BitCount())))) {
              // This isn't an index we could possibly be updating, so we make
              // no changes.
              return absl::OkStatus();
            }
          }

          // We might be updating this element, so add the possible values from
          // `update_value`.
          if (update_value.has_value()) {
            leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
                element_tree, update_value->AsView(),
                [](IntervalSet& lhs, const IntervalSet& rhs) {
                  lhs = IntervalSet::Combine(lhs, rhs);
                });
          } else {
            // The update value is unconstrained, so remove all constraints on
            // this element.
            leaf_type_tree::ForEach(element_tree, [&](IntervalSet& is) {
              is = IntervalSet::Maximal(is.BitCount());
            });
          }
          return absl::OkStatus();
        }));
  }
  SetIntervalSetTree(update, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleNaryAnd(NaryOp* and_op) {
  INITIALIZE_OR_SKIP(and_op);
  IntervalSet result =
      IntervalSet::Precise(Bits::AllOnes(and_op->BitCountOrDie()));
  for (Node* operand : and_op->operands()) {
    ASSIGN_INTERVAL_SET_REF_OR_RETURN(operand_intervals, operand);
    result = interval_ops::And(result, operand_intervals);
  }
  return SetIntervalSet(and_op, std::move(result));
}

absl::Status RangeQueryVisitor::HandleNaryNand(NaryOp* nand_op) {
  INITIALIZE_OR_SKIP(nand_op);
  IntervalSet result =
      IntervalSet::Precise(Bits::AllOnes(nand_op->BitCountOrDie()));
  for (Node* operand : nand_op->operands()) {
    ASSIGN_INTERVAL_SET_REF_OR_RETURN(operand_intervals, operand);
    result = interval_ops::And(result, operand_intervals);
  }
  return SetIntervalSet(nand_op, interval_ops::Not(result));
}

absl::Status RangeQueryVisitor::HandleNaryNor(NaryOp* nor_op) {
  INITIALIZE_OR_SKIP(nor_op);
  IntervalSet result = IntervalSet::Precise(Bits(nor_op->BitCountOrDie()));
  for (Node* operand : nor_op->operands()) {
    ASSIGN_INTERVAL_SET_REF_OR_RETURN(operand_intervals, operand);
    result = interval_ops::Or(result, operand_intervals);
  }
  return SetIntervalSet(nor_op, interval_ops::Not(result));
}

absl::Status RangeQueryVisitor::HandleNaryOr(NaryOp* or_op) {
  INITIALIZE_OR_SKIP(or_op);
  IntervalSet result = IntervalSet::Precise(Bits(or_op->BitCountOrDie()));
  for (Node* operand : or_op->operands()) {
    ASSIGN_INTERVAL_SET_REF_OR_RETURN(operand_intervals, operand);
    result = interval_ops::Or(result, operand_intervals);
  }
  return SetIntervalSet(or_op, std::move(result));
}

absl::Status RangeQueryVisitor::HandleNaryXor(NaryOp* xor_op) {
  INITIALIZE_OR_SKIP(xor_op);
  IntervalSet result = IntervalSet::Precise(Bits(xor_op->BitCountOrDie()));
  for (Node* operand : xor_op->operands()) {
    ASSIGN_INTERVAL_SET_REF_OR_RETURN(operand_intervals, operand);
    result = interval_ops::Xor(result, operand_intervals);
  }
  return SetIntervalSet(xor_op, std::move(result));
}

absl::Status RangeQueryVisitor::HandleNe(CompareOp* ne) {
  INITIALIZE_OR_SKIP(ne);

  if (ne->operand(0)->GetType()->GetFlatBitCount() == 0) {
    // Empty types are always equal.
    return SetIntervalSet(ne, IntervalSet::Precise(UBits(0, 1)));
  }

  std::optional<IntervalSetTreeView> lhs_intervals =
      MaybeGetIntervalSetTreeView(ne->operand(0));
  std::optional<IntervalSetTreeView> rhs_intervals =
      MaybeGetIntervalSetTreeView(ne->operand(1));
  if (!lhs_intervals.has_value() || !rhs_intervals.has_value()) {
    // One input is unconstrained, so we know nothing about the result.
    return SetIntervalSet(ne, IntervalSet::Maximal(1));
  }
  XLS_RET_CHECK_EQ(lhs_intervals->type(), rhs_intervals->type());

  IntervalSet res(/*bit_count=*/1);
  for (int64_t i = 0; i < lhs_intervals->size(); ++i) {
    res = IntervalSet::Combine(res,
                               interval_ops::Ne(lhs_intervals->elements()[i],
                                                rhs_intervals->elements()[i]));
  }
  return SetIntervalSet(ne, std::move(res));
}

absl::Status RangeQueryVisitor::HandleNeg(UnOp* neg) {
  INITIALIZE_OR_SKIP(neg);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, neg->operand(0));
  return SetIntervalSet(neg, interval_ops::Neg(arg));
}

absl::Status RangeQueryVisitor::HandleNot(UnOp* not_op) {
  INITIALIZE_OR_SKIP(not_op);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, not_op->operand(0));
  return SetIntervalSet(not_op, interval_ops::Not(arg));
}

absl::Status RangeQueryVisitor::HandleOneHot(OneHot* one_hot) {
  INITIALIZE_OR_SKIP(one_hot);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, one_hot->operand(0));
  return SetIntervalSet(one_hot,
                        interval_ops::OneHot(arg, one_hot->priority()));
}

absl::Status RangeQueryVisitor::HandleOneHotSel(OneHotSelect* sel) {
  INITIALIZE_OR_SKIP(sel);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(selector_intervals, sel->selector());

  // If the selector is always zero, then so is the result.
  if (selector_intervals.IsPrecise() && selector_intervals.CoversZero()) {
    SetIntervalSetTree(sel, ZeroIntervalSetTree(sel->GetType()));
    return absl::OkStatus();
  }

  IntervalSetTree result = EmptyIntervalSetTree(sel->GetType());

  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    TernaryVector case_pattern(sel->cases().size(), TernaryValue::kUnknown);
    case_pattern[i] = TernaryValue::kKnownOne;
    if (!interval_ops::CoversTernary(selector_intervals, case_pattern)) {
      // This case isn't possible, so it won't contribute to the result.
      continue;
    }

    case_pattern[i] = TernaryValue::kKnownZero;
    const bool can_miss_case =
        interval_ops::CoversTernary(selector_intervals, case_pattern);

    std::optional<IntervalSetTreeView> case_i =
        MaybeGetIntervalSetTreeView(sel->cases()[i]);
    if (!case_i.has_value()) {
      result = UnconstrainedIntervalSetTree(sel->GetType());
      break;
    }

    // TODO(epastor): Make more precise by only or-ing cases that can actually
    // occur together; independent cases should be added independently where
    // possible.
    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        result.AsMutableView(), *case_i,
        [can_miss_case](IntervalSet& lhs, const IntervalSet& rhs) {
          if (can_miss_case) {
            IntervalSet new_possibilities = rhs;
            new_possibilities.AddInterval(
                Interval::Precise(UBits(0, new_possibilities.BitCount())));
            new_possibilities.Normalize();

            if (lhs.IsEmpty()) {
              lhs = std::move(new_possibilities);
            } else {
              lhs = interval_ops::Or(lhs, new_possibilities);
            }
          } else {
            if (lhs.IsEmpty()) {
              lhs = rhs;
            } else {
              lhs = interval_ops::Or(lhs, rhs);
            }
          }
        });
  }

  for (IntervalSet& intervals : result.elements()) {
    intervals =
        interval_ops::MinimizeIntervals(intervals, kDefaultIntervalSize);
  }
  SetIntervalSetTree(sel, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandlePrioritySel(PrioritySelect* sel) {
  INITIALIZE_OR_SKIP(sel);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(selector_intervals, sel->selector());

  std::optional<IntervalSetTreeView> default_value =
      MaybeGetIntervalSetTreeView(sel->default_value());
  std::optional<IntervalSetTree> default_value_storage;
  if (!default_value.has_value()) {
    default_value_storage = UnconstrainedIntervalSetTree(sel->GetType());
    default_value = default_value_storage->AsView();
  }

  // Initialize all interval sets to empty - or to the default value if the
  // selector could be zero.
  IntervalSetTree result;
  if (!selector_intervals.CoversZero()) {
    result = EmptyIntervalSetTree(sel->GetType());
  } else if (default_value_storage.has_value()) {
    result = *std::move(default_value_storage);
  } else {
    result = leaf_type_tree::Clone(*default_value);
  }

  TernaryVector case_pattern(sel->cases().size(), TernaryValue::kUnknown);
  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    // TODO(vmirian): Make implementation more efficient by considering only the
    // ranges of interest.
    case_pattern[i] = TernaryValue::kKnownOne;
    if (i > 0) {
      case_pattern[i - 1] = TernaryValue::kKnownZero;
    }
    if (interval_ops::CoversTernary(selector_intervals, case_pattern)) {
      std::optional<IntervalSetTreeView> case_i =
          MaybeGetIntervalSetTreeView(sel->cases()[i]);
      if (!case_i.has_value()) {
        result = UnconstrainedIntervalSetTree(result.type());
        break;
      }
      leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
          result.AsMutableView(), *case_i,
          [](IntervalSet& lhs, const IntervalSet& rhs) {
            lhs = IntervalSet::Combine(lhs, rhs);
          });
    }
  }
  for (IntervalSet& intervals : result.elements()) {
    intervals =
        interval_ops::MinimizeIntervals(intervals, kDefaultIntervalSize);
  }
  SetIntervalSetTree(sel, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleOrReduce(BitwiseReductionOp* or_reduce) {
  INITIALIZE_OR_SKIP(or_reduce);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, or_reduce->operand(0));
  return SetIntervalSet(or_reduce, interval_ops::OrReduce(arg));
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

absl::Status RangeQueryVisitor::HandleStateRead(StateRead* state_read) {
  INITIALIZE_OR_SKIP(state_read);
  // We don't know anything about state element values.
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
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, div->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, div->operand(1));
  return SetIntervalSet(div, interval_ops::SDiv(l, r));
}

absl::Status RangeQueryVisitor::HandleSGe(CompareOp* ge) {
  INITIALIZE_OR_SKIP(ge);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, ge->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, ge->operand(1));
  return SetIntervalSet(ge, interval_ops::SGe(l, r));
}

absl::Status RangeQueryVisitor::HandleSGt(CompareOp* gt) {
  INITIALIZE_OR_SKIP(gt);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, gt->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, gt->operand(1));
  return SetIntervalSet(gt, interval_ops::SGt(l, r));
}

absl::Status RangeQueryVisitor::HandleSLe(CompareOp* le) {
  INITIALIZE_OR_SKIP(le);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, le->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, le->operand(1));
  return SetIntervalSet(le, interval_ops::SLe(l, r));
}

absl::Status RangeQueryVisitor::HandleSLt(CompareOp* lt) {
  INITIALIZE_OR_SKIP(lt);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, lt->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, lt->operand(1));
  return SetIntervalSet(lt, interval_ops::SLt(l, r));
}

absl::Status RangeQueryVisitor::HandleSMod(BinOp* mod) {
  INITIALIZE_OR_SKIP(mod);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, mod->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, mod->operand(1));
  return SetIntervalSet(mod, interval_ops::SMod(l, r));
}

absl::Status RangeQueryVisitor::HandleSMul(ArithOp* mul) {
  INITIALIZE_OR_SKIP(mul);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, mul->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, mul->operand(1));
  return SetIntervalSet(mul, interval_ops::SMul(l, r, mul->width()));
}

absl::Status RangeQueryVisitor::HandleSMulp(PartialProductOp* mul) {
  INITIALIZE_OR_SKIP(mul);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleSel(Select* sel) {
  INITIALIZE_OR_SKIP(sel);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(selector_intervals, sel->selector());

  // Initialize all interval sets to empty
  IntervalSetTree result = EmptyIntervalSetTree(sel->GetType());
  absl::Status status = absl::OkStatus();
  const uint64_t num_cases = sel->cases().size();
  selector_intervals.ForEachElement([&](const Bits& bits) -> bool {
    // Selects can't have a cases size of more than uint64_MAX.
    uint64_t i = bits.ToUint64().value_or(std::numeric_limits<uint64_t>::max());
    std::optional<IntervalSetTreeView> selected_case;
    bool finished = false;
    if (i < num_cases) {
      selected_case = MaybeGetIntervalSetTreeView(sel->cases()[i]);
    } else {
      // Would select the default case; handle it, then end, since all later
      // selectors will also select the default case.
      if (!sel->default_value().has_value()) {
        status = absl::InternalError(
            "Reached default case for a Select that has no default case");
        return true;
      }
      selected_case = MaybeGetIntervalSetTreeView(*sel->default_value());
      finished = true;
    }

    if (!selected_case.has_value()) {
      // Unconstrained case, so the result is unconstrained.
      result = UnconstrainedIntervalSetTree(result.type());

      // No need to check further cases.
      return true;
    }

    leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
        result.AsMutableView(), *selected_case,
        [](IntervalSet& lhs, const IntervalSet& rhs) {
          lhs = IntervalSet::Combine(lhs, rhs);
        });
    return finished;
  });
  XLS_RETURN_IF_ERROR(status);

  for (IntervalSet& intervals : result.elements()) {
    intervals =
        interval_ops::MinimizeIntervals(intervals, kDefaultIntervalSize);
  }
  SetIntervalSetTree(sel, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleSend(Send* send) {
  INITIALIZE_OR_SKIP(send);
  return absl::OkStatus();  // TODO(taktoa): implement: interprocedural
}

absl::Status RangeQueryVisitor::HandleShll(BinOp* shll) {
  INITIALIZE_OR_SKIP(shll);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(a, shll->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(b, shll->operand(1));
  return SetIntervalSet(shll, interval_ops::Shll(a, b));
}

absl::Status RangeQueryVisitor::HandleShra(BinOp* shra) {
  INITIALIZE_OR_SKIP(shra);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(a, shra->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(b, shra->operand(1));
  return SetIntervalSet(shra, interval_ops::Shra(a, b));
}

absl::Status RangeQueryVisitor::HandleShrl(BinOp* shrl) {
  INITIALIZE_OR_SKIP(shrl);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(a, shrl->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(b, shrl->operand(1));
  return SetIntervalSet(shrl, interval_ops::Shrl(a, b));
}

absl::Status RangeQueryVisitor::HandleSignExtend(ExtendOp* sign_ext) {
  INITIALIZE_OR_SKIP(sign_ext);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, sign_ext->operand(0));
  return SetIntervalSet(
      sign_ext, interval_ops::SignExtend(arg, sign_ext->new_bit_count()));
}

absl::Status RangeQueryVisitor::HandleSub(BinOp* sub) {
  INITIALIZE_OR_SKIP(sub);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, sub->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, sub->operand(1));
  return SetIntervalSet(sub, interval_ops::Sub(l, r));
}

absl::Status RangeQueryVisitor::HandleTrace(Trace* trace_op) {
  INITIALIZE_OR_SKIP(trace_op);
  // Produces a token, so maximal range is okay.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleTuple(Tuple* tuple) {
  INITIALIZE_OR_SKIP(tuple);
  std::vector<IntervalSetTreeView> children;
  std::vector<std::unique_ptr<IntervalSetTree>> temp_storage;
  for (Node* element : tuple->operands()) {
    std::optional<IntervalSetTreeView> child =
        MaybeGetIntervalSetTreeView(element);
    if (!child.has_value()) {
      temp_storage.push_back(std::make_unique<IntervalSetTree>(
          UnconstrainedIntervalSetTree(element->GetType())));
      child = temp_storage.back()->AsView();
    }
    children.push_back(*child);
  }
  CHECK(tuple->GetType()->IsTuple());
  XLS_ASSIGN_OR_RETURN(IntervalSetTree result,
                       leaf_type_tree::CreateTuple<IntervalSet>(
                           tuple->GetType()->AsTupleOrDie(), children));
  SetIntervalSetTree(tuple, std::move(result));
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleTupleIndex(TupleIndex* index) {
  INITIALIZE_OR_SKIP(index);
  if (std::optional<IntervalSetTreeView> view =
          MaybeGetIntervalSetTreeView(index->operand(0));
      view.has_value()) {
    SetIntervalSetTree(index,
                       leaf_type_tree::Clone(view->AsView({index->index()})));
  }
  // If we don't have any explicit ranges for the tuple there's nothing for the
  // index either.
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleUDiv(BinOp* div) {
  INITIALIZE_OR_SKIP(div);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, div->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, div->operand(1));
  return SetIntervalSet(div, interval_ops::UDiv(l, r));
}

absl::Status RangeQueryVisitor::HandleUGe(CompareOp* ge) {
  INITIALIZE_OR_SKIP(ge);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, ge->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, ge->operand(1));
  return SetIntervalSet(ge, interval_ops::UGe(l, r));
}

absl::Status RangeQueryVisitor::HandleUGt(CompareOp* gt) {
  INITIALIZE_OR_SKIP(gt);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, gt->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, gt->operand(1));
  return SetIntervalSet(gt, interval_ops::UGt(l, r));
}

absl::Status RangeQueryVisitor::HandleULe(CompareOp* le) {
  INITIALIZE_OR_SKIP(le);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, le->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, le->operand(1));
  return SetIntervalSet(le, interval_ops::ULe(l, r));
}

absl::Status RangeQueryVisitor::HandleULt(CompareOp* lt) {
  INITIALIZE_OR_SKIP(lt);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, lt->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, lt->operand(1));
  return SetIntervalSet(lt, interval_ops::ULt(l, r));
}

absl::Status RangeQueryVisitor::HandleUMod(BinOp* mod) {
  INITIALIZE_OR_SKIP(mod);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, mod->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, mod->operand(1));
  return SetIntervalSet(mod, interval_ops::UMod(l, r));
}

absl::Status RangeQueryVisitor::HandleUMul(ArithOp* mul) {
  INITIALIZE_OR_SKIP(mul);

  ASSIGN_INTERVAL_SET_REF_OR_RETURN(l, mul->operand(0));
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(r, mul->operand(1));
  return SetIntervalSet(mul, interval_ops::UMul(l, r, mul->width()));
}

absl::Status RangeQueryVisitor::HandleUMulp(PartialProductOp* mul) {
  INITIALIZE_OR_SKIP(mul);
  return absl::OkStatus();
}

absl::Status RangeQueryVisitor::HandleXorReduce(
    BitwiseReductionOp* xor_reduce) {
  INITIALIZE_OR_SKIP(xor_reduce);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, xor_reduce->operand(0));
  return SetIntervalSet(xor_reduce, interval_ops::XorReduce(arg));
}

absl::Status RangeQueryVisitor::HandleZeroExtend(ExtendOp* zero_ext) {
  INITIALIZE_OR_SKIP(zero_ext);
  ASSIGN_INTERVAL_SET_REF_OR_RETURN(arg, zero_ext->operand(0));
  return SetIntervalSet(
      zero_ext, interval_ops::ZeroExtend(arg, zero_ext->new_bit_count()));
}

#undef INITIALIZE_OR_SKIP
#undef ASSIGN_INTERVAL_SET_REF_OR_RETURN

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

bool RangeQueryEngine::Covers(Node* node, const Bits& value) const {
  if (!node->GetType()->IsBits() ||
      node->BitCountOrDie() != value.bit_count()) {
    // The type doesn't match, so `node` can't possibly cover it.
    return false;
  }
  if (!IsTracked(node)) {
    return true;
  }
  // If modifications to the function mean we don't have a tree we just fall
  // back to the bit-based version.
  if (!HasExplicitIntervals(node)) {
    return QueryEngine::Covers(node, value);
  }
  return GetIntervalSetTreeView(node)->Get({}).Covers(value);
}

Bits RangeQueryEngine::MaxUnsignedValue(Node* n) const {
  CHECK(n->GetType()->IsBits()) << n;
  // If modifications to the function mean we don't have a tree we just fall
  // back to the bit-based version.
  if (!HasExplicitIntervals(n)) {
    return QueryEngine::MaxUnsignedValue(n);
  }
  std::optional<Interval> hull =
      GetIntervalSetTreeView(n)->Get({}).ConvexHull();
  return hull ? hull->UpperBound() : Bits::AllOnes(n->BitCountOrDie());
}

Bits RangeQueryEngine::MinUnsignedValue(Node* n) const {
  CHECK(n->GetType()->IsBits()) << n;
  // If modifications to the function mean we don't have a tree we just fall
  // back to the bit-based version.
  if (!HasExplicitIntervals(n)) {
    return QueryEngine::MinUnsignedValue(n);
  }
  std::optional<Interval> hull =
      GetIntervalSetTreeView(n)->Get({}).ConvexHull();
  return hull ? hull->LowerBound() : Bits(n->BitCountOrDie());
}

std::optional<int64_t> RangeQueryEngine::KnownLeadingOnes(Node* n) const {
  if (!n->GetType()->IsBits()) {
    return std::nullopt;
  }
  if (!HasExplicitIntervals(n)) {
    return 0;
  }
  std::optional<Bits> lower_bound =
      GetIntervalSetTreeView(n)->Get({}).LowerBound();
  if (!lower_bound) {
    return 0;
  }
  return lower_bound->CountLeadingOnes();
}

std::optional<int64_t> RangeQueryEngine::KnownLeadingZeros(Node* n) const {
  if (!n->GetType()->IsBits()) {
    return std::nullopt;
  }
  if (!HasExplicitIntervals(n)) {
    return 0;
  }
  std::optional<Bits> upper_bound =
      GetIntervalSetTreeView(n)->Get({}).UpperBound();
  if (!upper_bound) {
    return 0;
  }
  return upper_bound->CountLeadingZeros();
}

std::optional<int64_t> RangeQueryEngine::KnownLeadingSignBits(Node* n) const {
  if (!n->GetType()->IsBits() || n->BitCountOrDie() == 0) {
    return std::nullopt;
  }
  if (!HasExplicitIntervals(n)) {
    // Sign bit is always equal to itself
    return 1;
  }
  std::optional<IntervalSet> interval = GetIntervalSetTreeView(n)->Get({});
  if (!interval || interval->IsEmpty()) {
    return 1;
  }
  if (!interval->UpperBound()->msb()) {
    // All values have sign bit of 0
    return KnownLeadingZeros(n);
  }
  if (interval->LowerBound()->msb()) {
    // All values have sign bit of 1.
    return KnownLeadingOnes(n);
  }
  // Need an extra bit since the sign bit is always equal to itself.
  return 1 + n->BitCountOrDie() -
         interval_ops::MinimumSignedBitCount(*interval);
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
