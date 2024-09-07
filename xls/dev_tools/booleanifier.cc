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

// Utility binary to convert input XLS IR to SMTLIB2.
// Adds the handy option of converting the XLS IR into a "fundamental"
// representation, i.e., consisting of only AND/OR/NOT ops.
#include "xls/dev_tools/booleanifier.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Evaluator for converting Nodes representing high-level Ops into
// single-bit AND/OR/NOT-based ones.
class BitEvaluator : public AbstractEvaluator<Node*, BitEvaluator> {
 public:
  explicit BitEvaluator(FunctionBuilder* builder)
      : builder_(builder),
        one_(builder->Literal(UBits(1, 1))),
        zero_(builder->Literal(UBits(0, 1))) {}

  Node* One() const { return one_.node(); }
  Node* Zero() const { return zero_.node(); }
  Node* Not(Node* const& input) const {
    return builder_->Not(BValue(input, builder_)).node();
  }
  Node* And(Node* const& a, Node* const& b) const {
    return builder_->And(BValue(a, builder_), BValue(b, builder_)).node();
  }
  Node* Or(Node* const& a, Node* const& b) const {
    return builder_->Or(BValue(a, builder_), BValue(b, builder_)).node();
  }
  Node* If(Node* sel, Node* consequent, Node* alternate) const {
    // Law of excluded middle applies so this is valid.
    //    sel | con | alt | res
    //     0  |  0  |  0  |  0
    //     0  |  0  |  1  |  1
    //     0  |  1  |  0  |  0
    //     0  |  1  |  1  |  1
    //     1  |  0  |  0  |  0
    //     1  |  0  |  1  |  0
    //     1  |  1  |  0  |  1
    //     1  |  1  |  1  |  1
    return Or(And(sel, consequent), And(Not(sel), alternate));
  }

 private:
  FunctionBuilder* builder_;
  BValue one_;
  BValue zero_;
};

absl::StatusOr<Function*> Booleanifier::Booleanify(
    Function* f, std::string_view boolean_function_name) {
  Booleanifier b(f, boolean_function_name);
  return b.Run();
}

Booleanifier::Booleanifier(Function* f, std::string_view boolean_function_name)
    : input_fn_(f),
      builder_(boolean_function_name.empty()
                   ? absl::StrCat(input_fn_->name(), "_boolean")
                   : boolean_function_name,
               input_fn_->package()),
      evaluator_(std::make_unique<BitEvaluator>(&builder_)) {}

// A node evaluator providing a bit-based implementation of
// array-index/update/slice.
class BooleanifierNodeEvaluator : public AbstractNodeEvaluator<BitEvaluator> {
 public:
  BooleanifierNodeEvaluator(
      BitEvaluator& eval, FunctionBuilder* fb,
      const absl::flat_hash_map<std::string, BValue>& params)
      : AbstractNodeEvaluator<BitEvaluator>(eval),
        builder_(fb),
        params_(params) {}

  // Select the appropriate elements.
  absl::Status HandleArrayIndex(ArrayIndex* index) override {
    XLS_ASSIGN_OR_RETURN(std::vector<BitEvaluator::Span> indexes,
                         GetValueList(index->indices()));
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> array,
                         GetCompoundValue(index->array()));
    LeafTypeTree<LeafValueT> result(array.type(), array.elements());
    for (int64_t i = 0; i < indexes.size(); ++i) {
      if (index->indices().at(i)->Is<Literal>()) {
        int64_t real_index =
            RealIndexFromLiteral(index->indices().at(i)->As<Literal>(),
                                 result.type()->AsArrayOrDie()->size());
        LeafTypeTreeView<LeafValueT> result_view = result.AsView({real_index});
        result = LeafTypeTree<LeafValueT>(result_view.type(),
                                          result_view.elements());
      } else {
        XLS_ASSIGN_OR_RETURN(result, ReadOneIndex(result.AsView(), indexes[i]));
      }
    }
    return SetValue(index, std::move(result));
  }

  // select the appropriate slice.
  absl::Status HandleArraySlice(ArraySlice* slice) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> array,
                         GetCompoundValue(slice->array()));
    XLS_ASSIGN_OR_RETURN(ArrayType * array_type, slice->GetType()->AsArray());
    int64_t input_size = slice->array()->GetType()->AsArrayOrDie()->size();
    if (slice->start()->Is<Literal>()) {
      int64_t start_idx =
          RealIndexFromLiteral(slice->start()->As<Literal>(), input_size);
      XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafValueT> sliced,
                           leaf_type_tree::SliceArray<BitEvaluator::Vector>(
                               array_type, array, start_idx));
      return SetValue(slice, std::move(sliced));
    }
    XLS_ASSIGN_OR_RETURN(BitEvaluator::Span index, GetValue(slice->start()));
    std::vector<LeafTypeTree<LeafValueT>> slices_mem;
    std::vector<LeafTypeTreeView<LeafValueT>> slices;
    slices_mem.reserve(slice->array()->GetType()->AsArrayOrDie()->size());
    slices.reserve(slice->array()->GetType()->AsArrayOrDie()->size());
    int64_t addressable_values = slice->start()->BitCountOrDie() < 64
                                     ? 1 << slice->start()->BitCountOrDie()
                                     : std::numeric_limits<int64_t>::max();
    for (int64_t i = 0; i < input_size && i < addressable_values; ++i) {
      XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafValueT> one_slice,
                           leaf_type_tree::SliceArray(array_type, array, i));
      slices_mem.emplace_back(std::move(one_slice));
      slices.push_back(slices_mem.back().AsView());
    }
    std::vector<BitEvaluator::Span> spans;
    spans.resize(slices.size());
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        (leaf_type_tree::ZipIndex<BitEvaluator::Vector, BitEvaluator::Vector>(
            slices,
            [&](Type* et,
                absl::Span<const BitEvaluator::Vector* const> elements,
                absl::Span<int64_t const> _)
                -> absl::StatusOr<BitEvaluator::Vector> {
              XLS_RET_CHECK_EQ(elements.size(), spans.size());
              absl::c_transform(
                  elements, spans.begin(),
                  [](auto* v) -> BitEvaluator::Span { return *v; });
              return evaluator().Select(
                  index,
                  absl::MakeSpan(spans).subspan(
                      0, std::min<int64_t>(input_size, addressable_values)),
                  input_size < addressable_values
                      ? std::make_optional(spans.back())
                      : std::nullopt);
            })));
    return SetValue(slice, std::move(result));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    XLS_ASSIGN_OR_RETURN(std::vector<BitEvaluator::Span> indexes,
                         GetValueList(update->indices()));
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> array,
                         GetCompoundValue(update->array_to_update()));
    XLS_ASSIGN_OR_RETURN(LeafTypeTreeView<LeafValueT> update_val,
                         GetCompoundValue(update->update_value()));
    // Check for out-of-bounds-write
    XLS_ASSIGN_OR_RETURN(ArrayType * arr, update->GetType()->AsArray());
    for (int64_t i = 0; i < update->indices().size(); ++i) {
      Node* idx = update->indices()[i];
      if (idx->Is<Literal>() &&
          RealIndexFromLiteral(idx->As<Literal>(),
                               std::numeric_limits<int64_t>::max()) >=
              arr->size()) {
        // Write is known-out-of-bounds, no effect.
        return SetValue(
            update, LeafTypeTree<LeafValueT>(array.type(), array.elements()));
      }
      if (i + 1 < update->indices().size()) {
        XLS_ASSIGN_OR_RETURN(arr, arr->element_type()->AsArray());
      }
    }
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<LeafValueT> result,
        PerformUpdateWith(array, indexes, update->indices(), update_val));
    return SetValue(update, std::move(result));
  }

  absl::Status HandleParam(Param* param) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafValueT> result,
                         UnpackParam(params_.at(param->name())));
    return SetValue(param, std::move(result));
  }

 private:
  absl::StatusOr<LeafTypeTree<LeafValueT>> UnpackParam(BValue bv_node) {
    if (bv_node.GetType()->IsBits()) {
      BitEvaluator::Vector res;
      int64_t bit_count = bv_node.GetType()->GetFlatBitCount();
      res.reserve(bit_count);
      for (int64_t i = 0; i < bit_count; ++i) {
        res.push_back(builder_->BitSlice(bv_node, i, 1).node());
      }
      return LeafTypeTree<LeafValueT>(bv_node.GetType(), res);
    }
    if (bv_node.GetType()->IsArray()) {
      ArrayType* arr_type = bv_node.GetType()->AsArrayOrDie();
      std::vector<LeafTypeTree<LeafValueT>> elements_mem;
      std::vector<LeafTypeTreeView<LeafValueT>> elements;
      elements_mem.reserve(arr_type->size());
      elements.reserve(arr_type->size());
      for (int64_t i = 0; i < arr_type->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafValueT> element,
                             UnpackParam(builder_->ArrayIndex(
                                 bv_node, {builder_->Literal(UBits(i, 64))})));
        elements_mem.emplace_back(std::move(element));
        elements.push_back(elements_mem.back().AsView());
      }
      return leaf_type_tree::CreateArray<BitEvaluator::Vector>(arr_type,
                                                               elements);
    }
    if (bv_node.GetType()->IsTuple()) {
      TupleType* tup_type = bv_node.GetType()->AsTupleOrDie();
      std::vector<LeafTypeTree<LeafValueT>> elements_mem;
      std::vector<LeafTypeTreeView<LeafValueT>> elements;
      elements_mem.reserve(tup_type->size());
      elements.reserve(tup_type->size());
      for (int64_t i = 0; i < tup_type->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafValueT> element,
                             UnpackParam(builder_->TupleIndex(bv_node, i)));
        elements_mem.emplace_back(std::move(element));
        elements.push_back(elements_mem.back().AsView());
      }
      return leaf_type_tree::CreateTuple<BitEvaluator::Vector>(tup_type,
                                                               elements);
    }
    XLS_RET_CHECK(bv_node.GetType()->IsToken())
        << bv_node << " type not handled";
    XLS_RET_CHECK_FAIL() << bv_node << " is a token!";
  }
  int64_t RealIndexFromLiteral(Literal* l, int64_t limit) {
    int64_t start_idx = l->value().bits().FitsInUint64()
                            ? l->value().bits().ToUint64().value()
                            : std::numeric_limits<int64_t>::max();
    if (start_idx < 0 || start_idx >= limit) {
      return limit - 1;
    }
    return start_idx;
  }

  // Do an array update on 'array' with index given by indexes and
  // indexes_nodes. Update the value to 'to_update'
  absl::StatusOr<LeafTypeTree<LeafValueT>> PerformUpdateWith(
      LeafTypeTreeView<LeafValueT> array,
      absl::Span<BitEvaluator::Span const> indexes,
      absl::Span<Node* const> indexes_nodes,
      LeafTypeTreeView<LeafValueT> to_update) {
    XLS_RET_CHECK_EQ(indexes.size(), indexes_nodes.size());
    if (indexes.empty()) {
      return LeafTypeTree<LeafValueT>(to_update.type(), to_update.elements());
    }
    if (indexes_nodes.front()->Is<Literal>()) {
      int64_t real_index =
          RealIndexFromLiteral(indexes_nodes.front()->As<Literal>(),
                               std::numeric_limits<int64_t>::max());
      XLS_RET_CHECK_LT(real_index, array.type()->AsArrayOrDie()->size());
      XLS_ASSIGN_OR_RETURN(
          LeafTypeTree<LeafValueT> updated_segment,
          PerformUpdateWith(array.AsView({real_index}), indexes.subspan(1),
                            indexes_nodes.subspan(1), to_update));
      LeafTypeTree<LeafValueT> final(array.type(), array.elements());
      // move the updated values.
      absl::c_move(updated_segment.elements(),
                   final.AsMutableView({real_index}).elements().begin());
      return final;
    }
    int64_t addressable_values = indexes.front().size() < 64
                                     ? 1 << indexes.front().size()
                                     : std::numeric_limits<int64_t>::max();
    int64_t array_size = array.type()->AsArrayOrDie()->size();
    std::vector<LeafTypeTreeView<LeafValueT>> cases;
    std::vector<LeafTypeTree<LeafValueT>> cases_mem;
    cases.reserve(array_size);
    cases_mem.reserve(array_size);
    // For each possible index create the resulting CompoundValue if that slot
    // is the real one.
    for (int64_t i = 0; i < array_size && i < addressable_values; ++i) {
      XLS_ASSIGN_OR_RETURN(
          LeafTypeTree<LeafValueT> updated_segment,
          PerformUpdateWith(array.AsView({i}), indexes.subspan(1),
                            indexes_nodes.subspan(1), to_update));
      // Start with the current array.
      LeafTypeTree<LeafValueT> orig(array.type(), array.elements());
      // std::move in the updated value over the original values.
      absl::c_move(updated_segment.elements(),
                   // Use AsMutableView to select the 'i'th element.
                   orig.AsMutableView({i}).elements().begin());
      // Make sure the view lives long enough.
      cases_mem.emplace_back(std::move(orig));
      cases.push_back(cases_mem.back().AsView());
    }
    // Push the 'unchanged' variant at the end in case its needed.
    cases.push_back(array);
    std::vector<BitEvaluator::Span> spans;
    spans.resize(cases.size());
    return leaf_type_tree::ZipIndex<BitEvaluator::Vector, BitEvaluator::Vector>(
        cases,
        [&](Type* et, absl::Span<const BitEvaluator::Vector* const> elements,
            absl::Span<int64_t const> ltt_location)
            -> absl::StatusOr<BitEvaluator::Vector> {
          XLS_RET_CHECK_EQ(elements.size(), spans.size());
          absl::c_transform(elements, spans.begin(),
                            [](auto* v) -> BitEvaluator::Span { return *v; });
          // back is the unchanged element.
          return evaluator().Select(
              indexes.front(),
              // Make sure that we have no more cases than we can address.
              absl::MakeSpan(spans).subspan(
                  0, std::min<int64_t>(spans.size(), addressable_values)),
              // If we have more possible input values than cases we might
              // overflow so need the 'unchanged' last element as default.
              addressable_values > array_size ? std::make_optional(spans.back())
                                              : std::nullopt);
        });
  }

  // Perform an ArrayIndex with a single index element.
  absl::StatusOr<LeafTypeTree<LeafValueT>> ReadOneIndex(
      LeafTypeTreeView<LeafValueT> source, BitEvaluator::Span index) {
    int64_t array_size = source.type()->AsArrayOrDie()->size();
    std::vector<LeafTypeTreeView<LeafValueT>> cases;
    cases.reserve(array_size);
    int64_t addressable_values = index.size() < 64
                                     ? 1 << index.size()
                                     : std::numeric_limits<int64_t>::max();
    for (int64_t i = 0; i < array_size && i < addressable_values; ++i) {
      cases.push_back(source.AsView({i}));
    }
    std::vector<BitEvaluator::Span> spans;
    spans.resize(cases.size());
    return leaf_type_tree::ZipIndex<BitEvaluator::Vector, BitEvaluator::Vector>(
        cases,
        [&](Type* et, absl::Span<const BitEvaluator::Vector* const> elements,
            absl::Span<int64_t const> _)
            -> absl::StatusOr<BitEvaluator::Vector> {
          XLS_RET_CHECK_EQ(elements.size(), spans.size());
          absl::c_transform(elements, spans.begin(),
                            [](auto* v) -> BitEvaluator::Span { return *v; });
          return evaluator().Select(
              index,
              absl::MakeSpan(spans).subspan(
                  0, std::min<int64_t>(spans.size(), addressable_values)),
              array_size < addressable_values ? std::make_optional(spans.back())
                                              : std::nullopt);
        });
  }
  FunctionBuilder* builder_;
  const absl::flat_hash_map<std::string, BValue>& params_;
};

absl::StatusOr<Function*> Booleanifier::Run() {
  for (const Param* param : input_fn_->params()) {
    params_[param->name()] = builder_.Param(param->name(), param->GetType());
  }

  BooleanifierNodeEvaluator bne(*evaluator_, &builder_, params_);
  XLS_RETURN_IF_ERROR(input_fn_->Accept(&bne));

  Node* return_node = input_fn_->return_value();
  XLS_ASSIGN_OR_RETURN(
      LeafTypeTreeView<BooleanifierNodeEvaluator::LeafValueT> return_val,
      bne.GetCompoundValue(return_node));
  XLS_ASSIGN_OR_RETURN(BValue result, PackReturnValue(return_val));
  return builder_.BuildWithReturnValue(result);
}

// The inverse of UnpackParam - overlays structure on top of a flat bit array.
absl::StatusOr<BValue> Booleanifier::PackReturnValue(
    LeafTypeTreeView<BooleanifierNodeEvaluator::LeafValueT> result) {
  if (result.type()->IsBits()) {
    std::vector<BValue> args;
    args.reserve(result.type()->GetFlatBitCount());
    for (Node* n : result.Get({})) {
      args.push_back(BValue(n, &builder_));
    }
    absl::c_reverse(args);
    return builder_.Concat(args);
  }
  if (result.type()->IsArray()) {
    ArrayType* arr_type = result.type()->AsArrayOrDie();
    std::vector<BValue> args;
    args.reserve(arr_type->size());
    for (int64_t i = 0; i < arr_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(BValue element, PackReturnValue(result.AsView({i})));
      args.push_back(element);
    }
    return builder_.Array(args, arr_type->element_type());
  }
  if (result.type()->IsTuple()) {
    TupleType* tup_type = result.type()->AsTupleOrDie();
    std::vector<BValue> args;
    args.reserve(tup_type->size());
    for (int64_t i = 0; i < tup_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(BValue element, PackReturnValue(result.AsView({i})));
      args.push_back(element);
    }
    return builder_.Tuple(args);
  }
  XLS_RET_CHECK_FAIL() << result.type() << " is unimplemented";
}

}  // namespace xls
