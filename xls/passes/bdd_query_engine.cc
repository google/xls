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

#include "xls/passes/bdd_query_engine.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {

namespace {

// Returns whether the given op should be included in BDD computations.
bool ShouldEvaluate(Node* node) {
  const int64_t kMaxWidth = 64;
  auto is_wide = [](Node* n) {
    return n->GetType()->GetFlatBitCount() > kMaxWidth;
  };

  if (!node->GetType()->IsBits()) {
    return false;
  }
  switch (node->op()) {
    // Logical ops.
    case Op::kAnd:
    case Op::kNand:
    case Op::kNor:
    case Op::kNot:
    case Op::kOr:
    case Op::kXor:
      return true;

    // Extension ops.
    case Op::kSignExt:
    case Op::kZeroExt:
      return true;

    case Op::kLiteral:
      return true;

    // Bit moving ops.
    case Op::kBitSlice:
    case Op::kConcat:
    case Op::kReverse:
    case Op::kIdentity:
      return true;
    case Op::kDynamicBitSlice:
      return !is_wide(node);

    case Op::kOneHot:
      return !is_wide(node);

    // Select operations.
    case Op::kOneHotSel:
    case Op::kPrioritySel:
    case Op::kSel:
      return true;

    // Encode/decode operations:
    case Op::kDecode:
    case Op::kEncode:
      return true;

    // Comparison operation are only expressed if at least one of the operands
    // is a literal. This avoids the potential exponential explosion of BDD
    // nodes which can occur with pathological variable ordering.
    case Op::kUGe:
    case Op::kUGt:
    case Op::kULe:
    case Op::kULt:
    case Op::kEq:
    case Op::kNe:
      return node->operand(0)->Is<Literal>() || node->operand(1)->Is<Literal>();

    // Arithmetic ops
    case Op::kAdd:
    case Op::kSMul:
    case Op::kUMul:
    case Op::kSMulp:
    case Op::kUMulp:
    case Op::kNeg:
    case Op::kSDiv:
    case Op::kSub:
    case Op::kUDiv:
    case Op::kSMod:
    case Op::kUMod:
      return false;

    // Reduction ops.
    case Op::kAndReduce:
    case Op::kOrReduce:
    case Op::kXorReduce:
      return true;

    // Weirdo ops.
    case Op::kAfterAll:
    case Op::kMinDelay:
    case Op::kArray:
    case Op::kArrayConcat:
    case Op::kArrayIndex:
    case Op::kArraySlice:
    case Op::kArrayUpdate:
    case Op::kAssert:
    case Op::kCountedFor:
    case Op::kCover:
    case Op::kDynamicCountedFor:
    case Op::kGate:
    case Op::kInputPort:
    case Op::kInvoke:
    case Op::kMap:
    case Op::kOutputPort:
    case Op::kParam:
    case Op::kStateRead:
    case Op::kNext:
    case Op::kReceive:
    case Op::kRegisterRead:
    case Op::kRegisterWrite:
    case Op::kSend:
    case Op::kTrace:
    case Op::kTuple:
    case Op::kTupleIndex:
    case Op::kInstantiationInput:
    case Op::kInstantiationOutput:
      return false;

    // Unsupported comparison operations.
    case Op::kSGt:
    case Op::kSGe:
    case Op::kSLe:
    case Op::kSLt:
      return false;

    // Shift operations and related ops.
    // Shifts are very intensive to compute because they decompose into many,
    // many gates and they don't seem to provide much benefit. Turn-off for now.
    // TODO(meheff): Consider enabling shifts.
    case Op::kShll:
    case Op::kShra:
    case Op::kShrl:
    case Op::kBitSliceUpdate:
      return false;
  }
  LOG(FATAL) << "Invalid op: " << static_cast<int64_t>(node->op());
}

}  // namespace

bool IsCheapForBdds(const Node* node) {
  // The expense of evaluating a node using a BDD can depend strongly on the
  // width of the inputs or outputs. The nodes are roughly classified into
  // different groups based on their expense with width thresholds set for each
  // group. These values are picked empirically based on benchmark results.
  constexpr int64_t kWideThreshold = 256;
  constexpr int64_t kNarrowThreshold = 16;
  constexpr int64_t kVeryNarrowThreshold = 4;

  auto is_always_cheap = [](const Node* node) {
    return node->Is<ExtendOp>() || node->Is<NaryOp>() || node->Is<BitSlice>() ||
           node->Is<Concat>() || node->Is<Literal>();
  };

  auto is_cheap_when_not_wide = [](const Node* node) {
    return IsBinarySelect(const_cast<Node*>(node)) || node->Is<UnOp>() ||
           node->Is<BitwiseReductionOp>() || node->Is<OneHot>() ||
           node->op() == Op::kEq || node->op() == Op::kNe;
  };

  auto is_cheap_when_narrow = [](const Node* node) {
    return node->Is<CompareOp>() || node->Is<OneHot>() ||
           node->Is<OneHotSelect>() || node->Is<PrioritySelect>();
  };

  int64_t width = node->GetType()->GetFlatBitCount();
  for (Node* operand : node->operands()) {
    width = std::max(operand->GetType()->GetFlatBitCount(), width);
  }

  return is_always_cheap(node) ||
         (is_cheap_when_not_wide(node) && width <= kWideThreshold) ||
         (is_cheap_when_narrow(node) && width <= kNarrowThreshold) ||
         width <= kVeryNarrowThreshold;
}

class BddQueryEngine::AssumingQueryEngine final : public QueryEngine {
 public:
  AssumingQueryEngine(const BddQueryEngine* query_engine,
                      BddNodeIndex assumption)
      : query_engine_(query_engine), assumption_(assumption) {}
  AssumingQueryEngine(std::shared_ptr<BddQueryEngine> query_engine,
                      BddNodeIndex assumption)
      : query_engine_storage_(std::move(query_engine)),
        query_engine_(query_engine_storage_.get()),
        assumption_(assumption) {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;
  bool IsTracked(Node* node) const override;
  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override;
  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const override;
  std::unique_ptr<QueryEngine> SpecializeGiven(
      const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const override;
  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override;
  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override;
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;
  bool IsKnown(const TreeBitLocation& bit) const override;
  std::optional<bool> KnownValue(const TreeBitLocation& bit) const override;
  bool IsAllZeros(Node* n) const override;
  bool IsAllOnes(Node* n) const override;
  bool IsFullyKnown(Node* n) const override;

 private:
  std::shared_ptr<BddQueryEngine> query_engine_storage_;
  const BddQueryEngine* query_engine_;

  BddNodeIndex assumption_;
};

absl::StatusOr<ReachedFixpoint> BddQueryEngine::AssumingQueryEngine::Populate(
    FunctionBase* f) {
  return absl::UnimplementedError("Cannot populate forwarding engine!");
}

bool BddQueryEngine::AssumingQueryEngine::IsTracked(Node* node) const {
  return query_engine_->IsTracked(node);
}
std::optional<SharedLeafTypeTree<TernaryVector>>
BddQueryEngine::AssumingQueryEngine::GetTernary(Node* node) const {
  if (!query_engine_->IsTracked(node)) {
    return std::nullopt;
  }
  absl::StatusOr<TernaryTree> ltt = TernaryTree::CreateFromFunction(
      node->GetType(),
      [&](Type* leaf_type, absl::Span<const int64_t> tree_index)
          -> absl::StatusOr<TernaryVector> {
        TernaryVector ternary(leaf_type->GetFlatBitCount(),
                              TernaryValue::kUnknown);
        for (int64_t bit_index = 0; bit_index < leaf_type->GetFlatBitCount();
             ++bit_index) {
          std::optional<BddNodeIndex> bit = query_engine_->GetBddNode(
              TreeBitLocation(node, bit_index, tree_index));
          if (!bit.has_value()) {
            continue;
          }
          if (query_engine_->Implies(assumption_, *bit)) {
            ternary[bit_index] = TernaryValue::kKnownOne;
          } else if (query_engine_->Implies(assumption_,
                                            query_engine_->bdd().Not(*bit))) {
            ternary[bit_index] = TernaryValue::kKnownZero;
          }
        }
        return ternary;
      });
  if (!ltt.ok()) {
    return std::nullopt;
  }
  return std::move(ltt).value().AsShared();
};

std::unique_ptr<QueryEngine>
BddQueryEngine::AssumingQueryEngine::SpecializeGivenPredicate(
    const absl::flat_hash_set<PredicateState>& state) const {
  std::unique_ptr<AssumingQueryEngine> specialized(
      down_cast<AssumingQueryEngine*>(
          query_engine_->SpecializeGivenPredicate(state).release()));
  specialized->assumption_ =
      query_engine_->bdd().And(assumption_, specialized->assumption_);
  return specialized;
}

std::unique_ptr<QueryEngine>
BddQueryEngine::AssumingQueryEngine::SpecializeGiven(
    const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const {
  std::unique_ptr<AssumingQueryEngine> result;
  if (query_engine_storage_ != nullptr) {
    result = std::make_unique<AssumingQueryEngine>(query_engine_storage_,
                                                   assumption_);
  } else {
    result = std::make_unique<AssumingQueryEngine>(query_engine_, assumption_);
  }
  BddNodeIndex new_assumption =
      query_engine_->bdd().And(assumption_, result->assumption_);
  if (query_engine_->ExceedsPathLimit(result->assumption_)) {
    // Restore the original assumption, with no updates.
    result->assumption_ = assumption_;
  } else {
    result->assumption_ = new_assumption;
  }
  return result;
}

LeafTypeTree<IntervalSet> BddQueryEngine::AssumingQueryEngine::GetIntervals(
    Node* node) const {
  // The default QueryEngine::GetIntervals implementation will inherit the
  // specialization from GetTernary.
  return QueryEngine::GetIntervals(node);
}

bool BddQueryEngine::AssumingQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return query_engine_->AtMostOneTrue(bits, assumption_);
}

bool BddQueryEngine::AssumingQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return query_engine_->AtLeastOneTrue(bits, assumption_);
}

bool BddQueryEngine::AssumingQueryEngine::Implies(
    const TreeBitLocation& a, const TreeBitLocation& b) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = query_engine_->GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = query_engine_->GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  return query_engine_->Implies(query_engine_->bdd().And(*a_bdd, assumption_),
                                *b_bdd);
}

std::optional<Bits> BddQueryEngine::AssumingQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return query_engine_->ImpliedNodeValue(predicate_bit_values, node,
                                         assumption_);
}

std::optional<TernaryVector>
BddQueryEngine::AssumingQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return query_engine_->ImpliedNodeTernary(predicate_bit_values, node,
                                           assumption_);
}

bool BddQueryEngine::AssumingQueryEngine::KnownEquals(
    const TreeBitLocation& a, const TreeBitLocation& b) const {
  return query_engine_->KnownEquals(a, b, assumption_);
}

bool BddQueryEngine::AssumingQueryEngine::KnownNotEquals(
    const TreeBitLocation& a, const TreeBitLocation& b) const {
  return query_engine_->KnownNotEquals(a, b, assumption_);
}

bool BddQueryEngine::AssumingQueryEngine::IsKnown(
    const TreeBitLocation& bit) const {
  return query_engine_->IsKnown(bit, assumption_);
}
std::optional<bool> BddQueryEngine::AssumingQueryEngine::KnownValue(
    const TreeBitLocation& bit) const {
  return query_engine_->KnownValue(bit, assumption_);
}
bool BddQueryEngine::AssumingQueryEngine::IsAllZeros(Node* n) const {
  return query_engine_->IsAllZeros(n, assumption_);
}
bool BddQueryEngine::AssumingQueryEngine::IsAllOnes(Node* n) const {
  return query_engine_->IsAllOnes(n, assumption_);
}
bool BddQueryEngine::AssumingQueryEngine::IsFullyKnown(Node* n) const {
  return query_engine_->IsFullyKnown(n, assumption_);
}

namespace {

using BddVector = std::vector<SaturatingBddNodeIndex>;
using BddSpan = absl::Span<const SaturatingBddNodeIndex>;
using BddTree = LeafTypeTree<BddVector>;
using BddTreeView = LeafTypeTreeView<BddVector>;
using SharedBddTree = SharedLeafTypeTree<BddVector>;

BddVector CreateUnknownVector(int64_t size, BinaryDecisionDiagram* bdd) {
  BddVector bdd_vector;
  bdd_vector.reserve(size);
  absl::c_move(bdd->NewVariables(size), std::back_inserter(bdd_vector));
  return bdd_vector;
}

absl::StatusOr<BddTree> CreateUnknownOfType(Type* type,
                                            BinaryDecisionDiagram* bdd) {
  return BddTree::CreateFromFunction(
      type, [bdd](Type* leaf_type) -> absl::StatusOr<BddVector> {
        return CreateUnknownVector(leaf_type->GetFlatBitCount(), bdd);
      });
}

class BddNodeEvaluator : public AbstractNodeEvaluator<SaturatingBddEvaluator> {
 public:
  BddNodeEvaluator(SaturatingBddEvaluator& evaluator)
      : AbstractNodeEvaluator(evaluator) {}

  absl::Status InjectValue(Node* node, const BddTree* value) {
    if (value == nullptr) {
      XLS_ASSIGN_OR_RETURN(
          BddTree unknown,
          CreateUnknownOfType(node->GetType(), evaluator().bdd()));
      return SetValue(node, std::move(unknown).AsShared());
    }
    return SetValue(node, value->AsView().AsShared());
  }

  absl::Status DefaultHandler(Node* node) override {
    XLS_ASSIGN_OR_RETURN(
        BddTree unknown,
        CreateUnknownOfType(node->GetType(), evaluator().bdd()));
    return SetValue(node, std::move(unknown).AsShared());
  }
};

}  // namespace

BddTree BddQueryEngine::ComputeInfo(
    Node* node, absl::Span<const BddTree* const> operand_infos) const {
  if (!ShouldEvaluate(node)) {
    VLOG(3) << "  node filtered out by generic ShouldEvaluate heuristic.";
    return *CreateUnknownOfType(node->GetType(), bdd_.get());
  }
  if (node_filter_.has_value() && !(*node_filter_)(node)) {
    VLOG(3) << "  node filtered out by configured filter.";
    return *CreateUnknownOfType(node->GetType(), bdd_.get());
  }

  VLOG(3) << "  computing BDD value...";
  BddNodeEvaluator node_evaluator(*evaluator_);
  absl::flat_hash_set<Node*> injected_operands;
  injected_operands.reserve(node->operand_count());
  for (auto [operand, operand_info] :
       iter::zip(node->operands(), operand_infos)) {
    if (auto [_, inserted] = injected_operands.insert(operand); !inserted) {
      // We've already injected the value for this operand.
      continue;
    }
    CHECK_OK(node_evaluator.InjectValue(operand, operand_info));
  }
  CHECK_OK(node->VisitSingleNode(&node_evaluator));
  absl::flat_hash_map<Node*, SharedBddTree> values =
      std::move(node_evaluator).values();
  LeafTypeTree<BddVector> result = std::move(values.at(node)).ToOwned();
  int64_t new_variables = 0;
  leaf_type_tree::ForEach(result.AsMutableView(), [&](BddVector& bdd_vector) {
    for (SaturatingBddNodeIndex& bdd_node : bdd_vector) {
      // Associate a new BDD variable with each bit that exceeded the path
      // limit, so we can continue reasoning while acknowledging that we no
      // longer model this bit's value.
      if (ExceedsPathLimit(bdd_node)) {
        bdd_node = evaluator_->bdd()->NewVariable();
        new_variables++;
      }
    }
  });
  if (new_variables > 0) {
    VLOG(3) << absl::StreamFormat(
        "  introduced %d new variables due to path limit.", new_variables);
  }
  return result;
}

absl::Status BddQueryEngine::MergeWithGiven(BddVector& info,
                                            const BddVector& given) const {
  XLS_RET_CHECK_EQ(info.size(), given.size());
  for (auto [bit, given_bit] : iter::zip(info, given)) {
    if (std::holds_alternative<TooManyPaths>(given_bit)) {
      continue;
    }
    if (ExceedsPathLimit(bit) ||
        std::get<BddNodeIndex>(given_bit) == bdd_->zero() ||
        std::get<BddNodeIndex>(given_bit) == bdd_->one()) {
      bit = given_bit;
    }
  }
  return absl::OkStatus();
}

std::unique_ptr<QueryEngine> BddQueryEngine::SpecializeGivenPredicate(
    const absl::flat_hash_set<PredicateState>& state) const {
  absl::flat_hash_map<Node*, ValueKnowledge> givens;
  for (const PredicateState& ps : state) {
    if (ps.IsBasePredicate()) {
      continue;
    }
    if (ps.IsSelectPredicate()) {
      // Add the assumption that this arm of the select is taken.
      //
      // We need to build the equality assertion.
      Node* selector = ps.selector();
      CHECK(selector->GetType()->IsBits());
      if (ps.node()->Is<Select>()) {
        if (ps.IsDefaultArm()) {
          // The selector can be assumed to be past the end of the cases...
          // unless it doesn't fit, in which case this is impossible.
          int64_t num_cases = ps.node()->As<Select>()->cases().size();
          if (Bits::MinBitCountUnsigned(num_cases) <=
              selector->BitCountOrDie()) {
            IntervalSet intervals(selector->BitCountOrDie());
            intervals.AddInterval(
                Interval::Closed(UBits(ps.node()->As<Select>()->cases().size(),
                                       selector->BitCountOrDie()),
                                 Bits::AllOnes(selector->BitCountOrDie())));
            intervals.Normalize();

            if (auto it = givens.find(selector);
                it != givens.end() && it->second.intervals.has_value()) {
              intervals = IntervalSet::Intersect(intervals,
                                                 it->second.intervals->Get({}));
            }
            givens[selector].intervals =
                IntervalSetTree::CreateSingleElementTree(selector->GetType(),
                                                         std::move(intervals));
          } else {
            // TODO(epastor): Use the impossibility of the default case somehow.
          }
        } else {
          // The selector can be assumed equal to the arm index... unless it
          // doesn't fit, in which case this is impossible.
          if (Bits::MinBitCountUnsigned(ps.arm_index()) <=
              selector->BitCountOrDie()) {
            IntervalSet intervals = IntervalSet::Precise(
                UBits(ps.arm_index(), selector->BitCountOrDie()));
            if (auto it = givens.find(selector);
                it != givens.end() && it->second.intervals.has_value()) {
              intervals = IntervalSet::Intersect(intervals,
                                                 it->second.intervals->Get({}));
            }
            givens[selector].intervals =
                IntervalSetTree::CreateSingleElementTree(selector->GetType(),
                                                         std::move(intervals));
          } else {
            // TODO(epastor): Use the impossibility of this arm somehow.
          }
        }
      } else if (ps.node()->Is<PrioritySelect>()) {
        // PrioritySelect has a selector which is a vector of bits.
        Node* selector = ps.selector();
        CHECK(selector->GetType()->IsBits());
        if (ps.IsDefaultArm()) {
          // The default arm is taken iff all bits are zero.
          IntervalSet intervals =
              IntervalSet::Precise(UBits(0, selector->BitCountOrDie()));
          if (auto it = givens.find(selector);
              it != givens.end() && it->second.intervals.has_value()) {
            intervals = IntervalSet::Intersect(intervals,
                                               it->second.intervals->Get({}));
          }
          givens[selector].intervals = IntervalSetTree::CreateSingleElementTree(
              selector->GetType(), std::move(intervals));
        } else {
          // The arm is taken iff the corresponding bit is one and all previous
          // bits are zero.
          CHECK_LT(ps.arm_index(), selector->BitCountOrDie());
          TernaryVector ternary(selector->BitCountOrDie(),
                                TernaryValue::kUnknown);
          for (int64_t i = 0; i < ps.arm_index(); ++i) {
            ternary[i] = TernaryValue::kKnownZero;
          }
          ternary[ps.arm_index()] = TernaryValue::kKnownOne;

          if (auto it = givens.find(selector);
              it != givens.end() && it->second.ternary.has_value()) {
            if (!ternary_ops::UpdateWithUnion(ternary,
                                              it->second.ternary->Get({}))
                     .ok()) {
              // TODO(epastor): Use the impossibility of this arm somehow.
            }
          }
          givens[selector].ternary = TernaryTree::CreateSingleElementTree(
              selector->GetType(), std::move(ternary));
        }
      } else if (ps.node()->Is<OneHotSelect>()) {
        // OneHotSelect has a selector which is a vector of bits.
        Node* selector = ps.selector();
        CHECK(selector->GetType()->IsBits());
        // The arm is taken iff the corresponding bit is one.
        CHECK_LT(ps.arm_index(), selector->BitCountOrDie());
        TernaryVector ternary(selector->BitCountOrDie(),
                              TernaryValue::kUnknown);
        ternary[ps.arm_index()] = TernaryValue::kKnownOne;
        if (auto it = givens.find(selector);
            it != givens.end() && it->second.ternary.has_value()) {
          if (!ternary_ops::UpdateWithUnion(ternary,
                                            it->second.ternary->Get({}))
                   .ok()) {
            // TODO(epastor): Use the impossibility of this arm somehow.
          }
        }
        givens[selector].ternary = TernaryTree::CreateSingleElementTree(
            selector->GetType(), std::move(ternary));
      } else {
        LOG(FATAL) << "Unsupported select type: " << ps.node();
      }
    } else if (ps.IsArrayUpdatePredicate()) {
      LOG(ERROR) << "ArrayUpdatePredicate for: " << ps.node();
      // Add an assumption that the indices are in or out of bounds.
      std::vector<BddNodeIndex> in_bounds_checks;
      in_bounds_checks.reserve(ps.indices().size());
      Type* array_type = ps.node()->GetType();
      for (Node* index : ps.indices()) {
        CHECK(array_type->IsArray());
        const int64_t array_bound = array_type->AsArrayOrDie()->size();

        CHECK(index->GetType()->IsBits());
        if (index->BitCountOrDie() >= Bits::MinBitCountUnsigned(array_bound)) {
          // Could be out of bounds; add the check.
          IntervalSet intervals(index->BitCountOrDie());
          intervals.AddInterval(
              Interval::RightOpen(UBits(0, index->BitCountOrDie()),
                                  UBits(array_bound, index->BitCountOrDie())));
          intervals.Normalize();

          if (auto it = givens.find(index);
              it != givens.end() && it->second.intervals.has_value()) {
            intervals = IntervalSet::Intersect(intervals,
                                               it->second.intervals->Get({}));
          }
          givens[index].intervals = IntervalSetTree::CreateSingleElementTree(
              index->GetType(), std::move(intervals));
        }

        array_type = array_type->AsArrayOrDie()->element_type();
      }
    }
  }
  return SpecializeGiven(givens);
}

std::unique_ptr<QueryEngine> BddQueryEngine::SpecializeGiven(
    const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const {
  std::vector<BddNodeIndex> assumptions;

  for (const auto& [node, value_knowledge] : givens) {
    if (value_knowledge.ternary.has_value()) {
      VLOG(3) << "Specializing on " << node->GetName() << " = "
              << value_knowledge.ternary->ToString(
                     [](TernarySpan span) { return xls::ToString(span); });
      CHECK_OK(leaf_type_tree::ForEachIndex(
          value_knowledge.ternary->AsView(),
          [&](Type*, TernarySpan ternary,
              absl::Span<const int64_t> tree_index) -> absl::Status {
            // Iterate from high to low bits; that way, if we prune some due to
            // exceeding the path limit, we will preferentially retain
            // high-order bit information.
            for (int64_t bit_index = ternary.size() - 1; bit_index >= 0;
                 --bit_index) {
              if (ternary_ops::IsUnknown(ternary[bit_index])) {
                continue;
              }
              std::optional<BddNodeIndex> bit =
                  GetBddNode(TreeBitLocation(node, bit_index, tree_index));
              if (!bit.has_value()) {
                continue;
              }
              assumptions.push_back(ternary[bit_index] ==
                                            TernaryValue::kKnownOne
                                        ? *bit
                                        : bdd().Not(*bit));
            }
            return absl::OkStatus();
          }));
    }
    if (value_knowledge.intervals.has_value()) {
      VLOG(3) << "Specializing on " << node->GetName() << " in "
              << value_knowledge.intervals->ToString();
      CHECK_OK(leaf_type_tree::ForEachIndex(
          value_knowledge.intervals->AsView(),
          [&](Type*, const IntervalSet& intervals,
              absl::Span<const int64_t> tree_index) -> absl::Status {
            std::vector<SaturatingBddNodeIndex> bits;
            bits.reserve(intervals.BitCount());
            for (int64_t i = 0; i < intervals.BitCount(); ++i) {
              std::optional<BddNodeIndex> bit =
                  GetBddNode(TreeBitLocation(node, i, tree_index));
              if (!bit.has_value()) {
                return absl::OkStatus();
              }
              bits.push_back(*bit);
            }

            SaturatingBddNodeVector in_interval_checks;
            for (const Interval& interval : intervals.Intervals()) {
              if (std::optional<Bits> precise_value =
                      interval.GetPreciseValue();
                  precise_value.has_value()) {
                SaturatingBddNodeIndex is_value = evaluator_->Equals(
                    bits, evaluator_->BitsToVector(*precise_value));
                if (HasTooManyPaths(is_value)) {
                  VLOG(3) << "SpecializeGiven exceeded path limit of "
                          << path_limit_ << " on: " << node->GetName() << " in "
                          << interval.ToString() << " (precise)";
                  // Since one of our checks has too many paths, the result will
                  // also have too many paths - so it will never have any
                  // effect.
                  return absl::OkStatus();
                }
                in_interval_checks.push_back(is_value);
                continue;
              }

              SaturatingBddNodeIndex lower_bound =
                  evaluator_->UGreaterThanOrEqual(
                      bits, evaluator_->BitsToVector(interval.LowerBound()));
              if (HasTooManyPaths(lower_bound)) {
                VLOG(3) << "SpecializeGiven exceeded path limit of "
                        << path_limit_ << " on: " << node->GetName() << " in "
                        << interval.ToString() << " (lower bound)";
                // Since one of our checks has too many paths, the result will
                // also have too many paths - so it will never have any effect.
                return absl::OkStatus();
              }
              SaturatingBddNodeIndex upper_bound = evaluator_->ULessThanOrEqual(
                  bits, evaluator_->BitsToVector(interval.UpperBound()));
              if (HasTooManyPaths(upper_bound)) {
                VLOG(3) << "SpecializeGiven exceeded path limit of "
                        << path_limit_ << " on: " << node->GetName() << " in "
                        << interval.ToString() << " (upper bound)";
                // Since one of our checks has too many paths, the result will
                // also have too many paths - so it will never have any effect.
                return absl::OkStatus();
              }
              SaturatingBddNodeIndex in_interval =
                  evaluator_->And(lower_bound, upper_bound);
              if (HasTooManyPaths(in_interval)) {
                VLOG(3) << "SpecializeGiven exceeded path limit of "
                        << path_limit_ << " on: " << node->GetName() << " in "
                        << interval.ToString() << " (joint)";
                // Since one of our checks has too many paths, the result will
                // also have too many paths - so it will never have any effect.
                return absl::OkStatus();
              }
              in_interval_checks.push_back(in_interval);
            }
            SaturatingBddNodeIndex in_intervals =
                evaluator_->OrReduce(in_interval_checks).front();
            if (HasTooManyPaths(in_intervals)) {
              VLOG(3) << "SpecializeGiven exceeded path limit of "
                      << path_limit_ << " on: " << node->GetName() << " in "
                      << intervals.ToString();
              return absl::OkStatus();
            }
            assumptions.push_back(ToBddNode(in_intervals));

            return absl::OkStatus();
          }));
    }
  }

  bool already_exceeded_limit = false;
  BddNodeIndex assumed = bdd().one();
  for (const BddNodeIndex& assumption : assumptions) {
    BddNodeIndex new_assumed = bdd().And(assumed, assumption);
    if (ExceedsPathLimit(new_assumed)) {
      if (!already_exceeded_limit) {
        VLOG(3) << "SpecializeGiven exceeded path limit of " << path_limit_;
        already_exceeded_limit = true;
      }
      continue;
    }
    assumed = new_assumed;
  }
  return std::make_unique<AssumingQueryEngine>(this, assumed);
}
bool BddQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return AtMostOneTrue(bits, std::nullopt);
}
bool BddQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits,
    std::optional<BddNodeIndex> assumption) const {
  // Computing this property is quadratic (at least) so limit the width.
  const int64_t kMaxWidth = 64;
  if (bits.size() > kMaxWidth) {
    return false;
  }

  BddNodeIndex result = bdd().zero();
  for (const TreeBitLocation& loc : bits) {
    if (!IsTracked(loc.node())) {
      return false;
    }
  }

  // Compute the NOR-reduction of a pairwise AND of all bits. If this value is
  // one then no two bits can be simultaneously true. Equivalently: at most one
  // bit is true.
  for (int64_t i = 0; i < bits.size(); ++i) {
    std::optional<BddNodeIndex> i_bdd = GetBddNode(bits[i]);
    if (!i_bdd.has_value()) {
      return false;
    }
    for (int64_t j = i + 1; j < bits.size(); ++j) {
      std::optional<BddNodeIndex> j_bdd = GetBddNode(bits[j]);
      if (!j_bdd.has_value()) {
        return false;
      }
      result = bdd().Or(result, bdd().And(*i_bdd, *j_bdd));
      if (ExceedsPathLimit(result)) {
        VLOG(3) << "AtMostOneTrue exceeded path limit of " << path_limit_;
        return false;
      }
    }
  }
  result = bdd().Not(result);

  if (assumption.has_value()) {
    return Implies(*assumption, result);
  }
  return result == bdd().one();
}

std::optional<SharedLeafTypeTree<TernaryVector>> BddQueryEngine::GetTernary(
    Node* node) const {
  return GetTernary(node, /*assumption=*/std::nullopt);
}
std::optional<SharedLeafTypeTree<TernaryVector>> BddQueryEngine::GetTernary(
    Node* node, std::optional<BddNodeIndex> assumption) const {
  if (!IsTracked(node)) {
    return std::nullopt;
  }
  if (assumption.has_value() && *assumption == bdd().zero()) {
    return std::nullopt;
  }
  absl::StatusOr<TernaryTree> ltt = TernaryTree::CreateFromFunction(
      node->GetType(),
      [&](Type* leaf_type, absl::Span<const int64_t> tree_index)
          -> absl::StatusOr<TernaryVector> {
        TernaryVector ternary(leaf_type->GetFlatBitCount(),
                              TernaryValue::kUnknown);
        for (int64_t bit_index = 0; bit_index < leaf_type->GetFlatBitCount();
             ++bit_index) {
          std::optional<BddNodeIndex> bit =
              GetBddNode(TreeBitLocation(node, bit_index, tree_index));
          if (!bit.has_value()) {
            continue;
          }
          if (*bit == bdd().zero()) {
            ternary[bit_index] = TernaryValue::kKnownZero;
          } else if (*bit == bdd().one()) {
            ternary[bit_index] = TernaryValue::kKnownOne;
          } else if (assumption.has_value() &&
                     Implies(*assumption, bdd().Not(*bit))) {
            ternary[bit_index] = TernaryValue::kKnownZero;
          } else if (assumption.has_value() && Implies(*assumption, *bit)) {
            ternary[bit_index] = TernaryValue::kKnownOne;
          }
        }
        return ternary;
      });
  if (!ltt.ok()) {
    return std::nullopt;
  }
  return std::move(ltt).value().AsShared();
}

bool BddQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return AtLeastOneTrue(bits, std::nullopt);
}
bool BddQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits,
    std::optional<BddNodeIndex> assumption) const {
  // At least one bit is true is equivalent to an OR-reduction of all the bits.
  SaturatingBddEvaluator evaluator(path_limit_, &bdd());
  SaturatingBddNodeVector bdd_bits;
  for (const TreeBitLocation& location : bits) {
    if (!IsTracked(location.node())) {
      return false;
    }
    std::optional<BddNodeIndex> bdd_node = GetBddNode(location);
    if (!bdd_node.has_value()) {
      return false;
    }
    bdd_bits.push_back(*bdd_node);
  }
  SaturatingBddNodeIndex or_reduce = evaluator.OrReduce(bdd_bits).front();
  if (HasTooManyPaths(or_reduce)) {
    VLOG(3) << "AtLeastOneTrue exceeded path limit of " << path_limit_;
    return false;
  }
  BddNodeIndex result = ToBddNode(or_reduce);
  if (assumption.has_value()) {
    return Implies(*assumption, result);
  }
  return result == bdd().one();
}

bool BddQueryEngine::Implies(const BddNodeIndex& a,
                             const BddNodeIndex& b) const {
  return bdd().Implies(a, b) == bdd().one();
}

bool BddQueryEngine::Implies(const TreeBitLocation& a,
                             const TreeBitLocation& b) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  return Implies(*a_bdd, *b_bdd);
}

std::optional<Bits> BddQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return ImpliedNodeValue(predicate_bit_values, node,
                          /*assumption=*/std::nullopt);
}

std::optional<Bits> BddQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node, std::optional<BddNodeIndex> assumption) const {
  std::optional<TernaryVector> implied_ternary =
      ImpliedNodeTernary(predicate_bit_values, node, assumption);
  if (!implied_ternary.has_value() ||
      !ternary_ops::IsFullyKnown(*implied_ternary)) {
    return std::nullopt;
  }
  return ternary_ops::ToKnownBitsValues(*implied_ternary);
}

std::optional<TernaryVector> BddQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return ImpliedNodeTernary(predicate_bit_values, node,
                            /*assumption=*/std::nullopt);
}

std::optional<TernaryVector> BddQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node, std::optional<BddNodeIndex> assumption) const {
  if (!IsTracked(node) || !node->GetType()->IsBits()) {
    return std::nullopt;
  }

  // Create a Bdd node for the predicate_bit_values.
  SaturatingBddNodeVector bdd_predicate_bits;
  for (const auto& [conjuction_bit_location, conjunction_value] :
       predicate_bit_values) {
    std::optional<BddNodeIndex> conjuction_bit =
        GetBddNode(conjuction_bit_location);
    if (!conjuction_bit.has_value()) {
      // Skip this predicate; we don't recognize the node, so we can't see the
      // effects of assuming it.
      continue;
    }
    bdd_predicate_bits.push_back(
        conjunction_value ? *conjuction_bit : bdd().Not(*conjuction_bit));
  }
  SaturatingBddNodeIndex bdd_predicate =
      evaluator_->AndReduce(bdd_predicate_bits).front();
  if (assumption.has_value()) {
    bdd_predicate = evaluator_->And(bdd_predicate, *assumption);
  }
  if (HasTooManyPaths(bdd_predicate)) {
    return std::nullopt;
  }
  BddNodeIndex bdd_predicate_bit = ToBddNode(bdd_predicate);

  // If the predicate evaluates to false, we can't determine
  // what node value it implies. That is, !predicate || node_bit
  // evaluates to true for both node_bit == 1 and == 0.
  if (bdd_predicate_bit == bdd().zero()) {
    return std::nullopt;
  }

  enum class ImpliedValue : std::uint8_t {
    kNotAnalyzable,
    kUnknown,
    kImpliedTrue,
    kImpliedFalse
  };
  auto implied_value = [&](int node_idx) -> TernaryValue {
    std::optional<BddNodeIndex> bdd_node_bit =
        GetBddNode(TreeBitLocation(node, node_idx));
    if (!bdd_node_bit.has_value()) {
      return TernaryValue::kUnknown;
    }
    if (Implies(bdd_predicate_bit, bdd().Not(*bdd_node_bit))) {
      return TernaryValue::kKnownZero;
    }
    if (Implies(bdd_predicate_bit, *bdd_node_bit)) {
      return TernaryValue::kKnownOne;
    }
    return TernaryValue::kUnknown;
  };

  // Check if bdd_predicate_bit implies anything about the node.
  CHECK(node->GetType()->IsBits());
  TernaryVector ternary(node->BitCountOrDie(), TernaryValue::kUnknown);
  for (int node_idx = 0; node_idx < node->BitCountOrDie(); ++node_idx) {
    ternary[node_idx] = implied_value(node_idx);
  }
  if (ternary_ops::AllUnknown(ternary)) {
    return std::nullopt;
  }
  return ternary;
}

bool BddQueryEngine::KnownEquals(const TreeBitLocation& a,
                                 const TreeBitLocation& b) const {
  return KnownEquals(a, b,
                     /*assumption=*/std::nullopt);
}

bool BddQueryEngine::KnownEquals(const TreeBitLocation& a,
                                 const TreeBitLocation& b,
                                 std::optional<BddNodeIndex> assumption) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  SaturatingBddNodeIndex result = evaluator_->Equals({*a_bdd}, {*b_bdd});
  if (assumption.has_value()) {
    result = evaluator_->Implies(*assumption, result);
  }
  return result == evaluator_->One();
}

bool BddQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                    const TreeBitLocation& b) const {
  return KnownNotEquals(a, b,
                        /*assumption=*/std::nullopt);
}

bool BddQueryEngine::KnownNotEquals(
    const TreeBitLocation& a, const TreeBitLocation& b,
    std::optional<BddNodeIndex> assumption) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  SaturatingBddNodeIndex result =
      evaluator_->Not(evaluator_->Equals({*a_bdd}, {*b_bdd}));
  if (assumption.has_value()) {
    result = evaluator_->Implies(*assumption, result);
  }
  return result == evaluator_->One();
}

bool BddQueryEngine::IsKnown(const TreeBitLocation& bit,
                             std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsKnown(bit);
  }
  return KnownValue(bit, assumption).has_value();
}

std::optional<bool> BddQueryEngine::KnownValue(
    const TreeBitLocation& bit, std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return KnownValue(bit);
  }
  if (!IsTracked(bit.node())) {
    return std::nullopt;
  }
  std::optional<BddNodeIndex> bdd_node = GetBddNode(bit);
  if (!bdd_node.has_value()) {
    return std::nullopt;
  }

  if (assumption.has_value()) {
    if (evaluator_->Implies(*assumption, *bdd_node) == evaluator_->One()) {
      return true;
    }
    if (evaluator_->Implies(*assumption, evaluator_->Not(*bdd_node)) ==
        evaluator_->One()) {
      return false;
    }
    return std::nullopt;
  }

  if (bdd_node == bdd().one()) {
    return true;
  } else if (bdd_node == bdd().zero()) {
    return false;
  } else {
    return std::nullopt;
  }
}

std::optional<Value> BddQueryEngine::KnownValue(
    Node* node, std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return KnownValue(node);
  }

  if (!IsTracked(node)) {
    return std::nullopt;
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
      GetTernary(node, *assumption);
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

bool BddQueryEngine::IsAllZeros(Node* n,
                                std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsAllZeros(n);
  }
  if (!IsTracked(n) || TypeHasToken(n->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(n, *assumption);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsKnownZero(v);
         });
}

bool BddQueryEngine::IsAllOnes(Node* n,
                               std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsAllOnes(n);
  }
  if (!IsTracked(n) || TypeHasToken(n->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(n, *assumption);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsKnownOne(v);
         });
}
bool BddQueryEngine::IsFullyKnown(
    Node* n, std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsFullyKnown(n);
  }
  if (!IsTracked(n) || TypeHasToken(n->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(n, *assumption);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsFullyKnown(v);
         });
}

}  // namespace xls
