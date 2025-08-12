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

#include "xls/passes/ternary_query_engine.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls {
namespace {

// Calls 'f' with a view restricted to each possible index value given the
// ternaries in indices.
template <typename Func>
absl::Status ForEachPossibleIndex(
    LeafTypeTreeView<TernaryEvaluator::Vector> view,
    absl::Span<const TernaryEvaluator::Span> indices, Func f) {
  if (indices.empty()) {
    return f(view);
  }
  // Get each possibility on this level and go on to the next one.
  for (const Bits& possibility : ternary_ops::AllBitsValues(indices.front())) {
    int64_t idx = bits_ops::UnsignedBitsToSaturatedInt64(possibility);
    if (idx >= view.type()->AsArrayOrDie()->size() - 1) {
      // OOB (or exact last-element) on this dimension. Get the results for
      // sub-arrays but then we can stop.
      return ForEachPossibleIndex(
          view.AsView({view.type()->AsArrayOrDie()->size() - 1}),
          indices.subspan(1), f);
    }
    XLS_RETURN_IF_ERROR(
        ForEachPossibleIndex(view.AsView({idx}), indices.subspan(1), f));
  }
  return absl::OkStatus();
}

// Merge 'out' at each of the index combinations in indices with 'update_value'.
absl::Status MergeEntriesForUpdate(
    MutableLeafTypeTreeView<TernaryEvaluator::Vector> out,
    absl::Span<TernaryEvaluator::Span const> indices,
    LeafTypeTreeView<TernaryEvaluator::Vector> update_value) {
  if (indices.empty()) {
    XLS_RET_CHECK(out.type()->IsEqualTo(update_value.type()));
    for (int64_t i = 0; i < out.elements().size(); ++i) {
      ternary_ops::UpdateWithIntersection(out.elements()[i],
                                          update_value.elements()[i]);
    }
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(ArrayType * arr, out.type()->AsArray());
  for (const Bits& v : ternary_ops::AllBitsValues(indices.front())) {
    int64_t idx = bits_ops::UnsignedBitsToSaturatedInt64(v);
    if (idx >= arr->size()) {
      break;
    }
    XLS_RETURN_IF_ERROR(MergeEntriesForUpdate(
        out.AsMutableView({idx}), indices.subspan(1), update_value));
  }
  return absl::OkStatus();
}

// How many bits we allow to be unconstrained when we need to intersect
// multiple possibilities.
static constexpr int64_t kIndexBitLimit = 10;

// Returns whether the operation will be computationally expensive to
// compute. The ternary query engine is intended to be fast so the analysis of
// these expensive operations is skipped with the effect being all bits are
// considered unknown. Operations and limits can be added as needed when
// pathological cases are encountered.
bool IsExpensiveToEvaluate(
    Node* node,
    const absl::flat_hash_map<
        Node*, SharedLeafTypeTree<TernaryEvaluator::Vector>>& known_bits) {
  // How many bits of output we allow for complex evaluations.
  static constexpr int64_t kComplexEvaluationLimit = 256;
  // How many bits of data we are willing to keep track of for compound
  // data-types.
  static constexpr int64_t kCompoundDataTypeSizeLimit = 65536;
  // Shifts are quadratic in the width of the operand so wide shifts are very
  // slow to evaluate in the abstract evaluator.
  bool is_complex_evaluation = node->OpIn({
      Op::kShrl,
      Op::kShll,
      Op::kShra,
      Op::kBitSliceUpdate,
  });
  if (is_complex_evaluation) {
    return node->GetType()->GetFlatBitCount() > kComplexEvaluationLimit;
  }
  // Compound data types can get enormous. Put a limit on how much data we are
  // willing to carry around.
  if (!node->GetType()->IsBits() &&
      node->GetType()->GetFlatBitCount() > kCompoundDataTypeSizeLimit) {
    return true;
  }
  // Array index checks require (worst case) 2^|unknown bits| operations so
  // limit them to 1024 == 2**10 operations.
  bool needs_index_scan =
      node->OpIn({Op::kArrayIndex, Op::kArraySlice, Op::kArrayUpdate});
  if (needs_index_scan) {
    int64_t unknown_index_bits;
    if (node->Is<ArraySlice>()) {
      unknown_index_bits =
          absl::c_count(known_bits.at(node->As<ArraySlice>()->start()).Get({}),
                        TernaryValue::kUnknown);
    } else {
      auto indices = node->Is<ArrayIndex>()
                         ? node->As<ArrayIndex>()->indices()
                         : node->As<ArrayUpdate>()->indices();
      unknown_index_bits =
          absl::c_accumulate(indices, 0, [&](int64_t acc, Node* n) {
            return acc + absl::c_count(known_bits.at(n).Get({}),
                                       TernaryValue::kUnknown);
          });
    }
    return unknown_index_bits >= kIndexBitLimit;
  }

  return false;
}

// Abstract evaluator operating on ternary values.
class TernaryNodeEvaluator : public AbstractNodeEvaluator<TernaryEvaluator> {
 public:
  using CompoundValue = LeafTypeTree<TernaryEvaluator::Vector>;
  using CompoundValueView = LeafTypeTreeView<TernaryEvaluator::Vector>;
  using AbstractNodeEvaluator<TernaryEvaluator>::AbstractNodeEvaluator;

  absl::Status SetGivenValue(Node* n, LeafTypeTree<TernaryVector> v) {
    return SetValue(n, std::move(v));
  }

  // By default everything is considered fully unconstrained.
  absl::Status DefaultHandler(Node* n) final {
    XLS_ASSIGN_OR_RETURN(auto unconstrained, UnconstrainedOf(n->GetType()));
    return SetValue(n, std::move(unconstrained));
  }

  absl::Status HandleArrayIndex(ArrayIndex* index) final {
    XLS_ASSIGN_OR_RETURN(auto indices, GetValueList(index->indices()));
    XLS_ASSIGN_OR_RETURN(auto array, GetCompoundValue(index->array()));
    if (indices.empty()) {
      // Some passes end up creating these index-less array index's with the
      // understanding they are identity ops.
      return SetValue(index, CompoundValue::CreateFromVector(
                                 array.type(), CompoundValue::DataContainerT(
                                                   array.elements().begin(),
                                                   array.elements().end())));
    }

    std::vector<CompoundValueView> possibilities;
    auto collect_possible_element =
        [&](CompoundValueView view) -> absl::Status {
      possibilities.emplace_back(view);
      return absl::OkStatus();
    };
    XLS_RETURN_IF_ERROR(
        ForEachPossibleIndex(array, indices, collect_possible_element));

    XLS_ASSIGN_OR_RETURN(CompoundValue result,
                         MergePossibilities(possibilities));
    return SetValue(index, std::move(result));
  }

  absl::Status HandleArraySlice(ArraySlice* slice) final {
    XLS_ASSIGN_OR_RETURN(auto start, GetValue(slice->start()));
    XLS_ASSIGN_OR_RETURN(CompoundValueView array,
                         GetCompoundValue(slice->array()));
    std::vector<CompoundValue> possibilities;
    for (const Bits& idx : ternary_ops::AllBitsValues(start)) {
      int64_t off = bits_ops::UnsignedBitsToSaturatedInt64(idx);
      XLS_ASSIGN_OR_RETURN(
          auto current_slice,
          leaf_type_tree::SliceArray(slice->GetType()->AsArrayOrDie(),
                                     array.AsView(), off));
      possibilities.push_back(current_slice);
      if (off > array.type()->AsArrayOrDie()->size()) {
        break;
      }
    }
    XLS_ASSIGN_OR_RETURN(CompoundValue result,
                         MergePossibilities(possibilities));
    return SetValue(slice, std::move(result));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) final {
    XLS_ASSIGN_OR_RETURN(auto indices, GetValueList(update->indices()));
    XLS_ASSIGN_OR_RETURN(auto array,
                         GetCompoundValue(update->array_to_update()));
    XLS_ASSIGN_OR_RETURN(auto update_value,
                         GetCompoundValue(update->update_value()));
    CompoundValue result(update->GetType(), array.elements());
    if (absl::c_all_of(indices, ternary_ops::IsFullyKnown)) {
      // Update location is exactly known. We know exactly what that location
      // will be after this.
      std::vector<int64_t> singleton_index;
      singleton_index.reserve(indices.size());
      XLS_ASSIGN_OR_RETURN(ArrayType * arr, array.type()->AsArray());
      // Get the actual location we are writing to.
      for (int64_t j = 0; j < indices.size(); ++j) {
        int64_t int_offset = bits_ops::UnsignedBitsToSaturatedInt64(
            *ternary_ops::AllBitsValues(indices[j]).begin());
        if (int_offset >= arr->size()) {
          // OOB Write. Don't do anything
          return SetValue(update, std::move(result));
        }
        singleton_index.push_back(int_offset);
        if (j + 1 < indices.size()) {
          XLS_ASSIGN_OR_RETURN(arr, arr->element_type()->AsArray());
        }
      }
      // Copy in the new value.
      absl::c_copy(update_value.elements(),
                   result.AsMutableView(singleton_index).elements().begin());
      return SetValue(update, std::move(result));
    }
    XLS_RETURN_IF_ERROR(
        MergeEntriesForUpdate(result.AsMutableView(), indices, update_value));
    return SetValue(update, std::move(result));
  }

 private:
  // Intersect all 'possibilities' together
  absl::StatusOr<CompoundValue> MergePossibilities(
      absl::Span<CompoundValueView const> possibilities) {
    XLS_RET_CHECK(!possibilities.empty());
    CompoundValue result(possibilities.front().type(),
                         possibilities.front().elements());
    for (CompoundValueView possibility : possibilities.subspan(1)) {
      // Bail out early if we are already fully unconstrained.
      //
      // It's judged that going to unconstrained is the most common outcome and
      // it usually will happen quickly so spend a bit of time to check this.
      if (absl::c_all_of(result.elements(),
                         [](const TernaryEvaluator::Vector& elem) {
                           return absl::c_all_of(elem, ternary_ops::IsUnknown);
                         })) {
        return result;
      }
      leaf_type_tree::SimpleUpdateFrom<TernaryVector, TernaryVector>(
          result.AsMutableView(), possibility, [](auto& lhs, const auto& rhs) {
            ternary_ops::UpdateWithIntersection(lhs, rhs);
          });
    }
    return result;
  }

  // Sum up all 'possibilities' using the given 'combine' function.
  absl::StatusOr<CompoundValue> MergePossibilities(
      absl::Span<CompoundValue const> possibilities) {
    std::vector<CompoundValueView> views;
    views.reserve(possibilities.size());
    absl::c_transform(possibilities, std::back_inserter(views),
                      [](const auto& v) { return v.AsView(); });
    return MergePossibilities(views);
  }

  // Returns a LeafTypeTree of all kUnknown values of the given type
  absl::StatusOr<CompoundValue> UnconstrainedOf(Type* type) {
    return CompoundValue::CreateFromFunction(type, [](Type* type) {
      return TernaryEvaluator::Vector(type->GetFlatBitCount(),
                                      TernaryValue::kUnknown);
    });
  }
};

class NoOpGivens final : public TernaryDataProvider {
 public:
  std::optional<LeafTypeTree<TernaryVector>> GetKnownTernary(
      Node* n) const final {
    return std::nullopt;
  }
};

}  // namespace

absl::StatusOr<ReachedFixpoint> TernaryQueryEngine::PopulateWithGivens(
    FunctionBase* f, const TernaryDataProvider& givens) {
  TernaryEvaluator evaluator;
  TernaryNodeEvaluator ternary_visitor(evaluator);
  for (Node* n : TopoSort(f)) {
    std::optional<LeafTypeTree<TernaryVector>> given =
        givens.GetKnownTernary(n);
    if (given) {
      XLS_RETURN_IF_ERROR(ternary_visitor.SetGivenValue(n, *std::move(given)));
      continue;
    }
    if (IsExpensiveToEvaluate(n, ternary_visitor.values())) {
      XLS_RETURN_IF_ERROR(ternary_visitor.DefaultHandler(n));
      continue;
    }
    XLS_RETURN_IF_ERROR(n->VisitSingleNode(&ternary_visitor));
  }

  absl::flat_hash_map<Node*, SharedLeafTypeTree<TernaryVector>> new_values =
      std::move(ternary_visitor).values();
  ReachedFixpoint rf = ReachedFixpoint::Unchanged;
  for (Node* node : f->nodes()) {
    CHECK(new_values.contains(node));
    if (values_.contains(node) &&
        values_[node].type() == new_values.at(node).type()) {
      leaf_type_tree::SimpleUpdateFrom<TernaryVector, TernaryVector>(
          values_[node].AsMutableView(), new_values.at(node).AsView(),
          [&rf](TernaryVector& lhs, const TernaryVector& rhs) {
            if (lhs != rhs) {
              rf = ReachedFixpoint::Changed;
            }
            CHECK_OK(ternary_ops::UpdateWithUnion(lhs, rhs));
          });
    } else {
      values_[node] = std::move(new_values.extract(node).mapped()).ToOwned();
    }
  }
  return rf;
}

absl::StatusOr<ReachedFixpoint> TernaryQueryEngine::Populate(FunctionBase* f) {
  NoOpGivens givens;
  return PopulateWithGivens(f, givens);
}

bool TernaryQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  int64_t maybe_one_count = 0;
  for (const TreeBitLocation& location : bits) {
    if (!IsKnown(location) || IsOne(location)) {
      maybe_one_count++;
    }
  }
  return maybe_one_count <= 1;
}

bool TernaryQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  for (const TreeBitLocation& location : bits) {
    if (IsOne(location)) {
      return true;
    }
  }
  return false;
}

bool TernaryQueryEngine::KnownEquals(const TreeBitLocation& a,
                                     const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) == IsOne(b);
}

bool TernaryQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                        const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) != IsOne(b);
}

}  // namespace xls
