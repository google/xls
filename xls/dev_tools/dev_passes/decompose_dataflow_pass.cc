// Copyright 2026 The XLS Authors
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

#include "xls/dev_tools/dev_passes/decompose_dataflow_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/enumerate.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {
namespace {

bool IsSingleLeaf(Node* node) {
  return node->GetType()->IsBits() || node->GetType()->IsToken() ||
         (node->GetType()->IsTuple() &&
          node->GetType()->AsTupleOrDie()->element_types().empty());
}
class DecomposeDataflowVisitor final : public DataflowVisitor<Node*> {
 public:
  explicit DecomposeDataflowVisitor(const QueryEngine& qe) : qe_(qe) {}
  bool changed() const { return changed_; }

  absl::Status DefaultHandler(Node* node) override {
    XLS_RET_CHECK(absl::c_all_of(node->operands(), IsSingleLeaf))
        << node << " has non-leaf operands";
    std::vector<Node*> new_operands;
    new_operands.reserve(node->operands().size());
    for (Node* operand : node->operands()) {
      auto tree = GetValue(operand);
      new_operands.push_back(tree.Get({}));
    }
    if (new_operands != node->operands()) {
      changed_ = true;
      for (const auto& [idx, arg] : iter::enumerate(new_operands)) {
        XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(idx, arg));
      }
    }
    if (!IsSingleLeaf(node)) {
      changed_ = true;
      // Only literal, param, state_read, receive, umulp and, smulp should be
      // returning a tuple/array and aren't already handled.
      XLS_RET_CHECK(node->OpIn({Op::kLiteral, Op::kUMulp, Op::kSMulp,
                                Op::kParam, Op::kStateRead, Op::kReceive}))
          << node;
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> tree, ToTreeOfNodes(node));
    return SetValue(node, std::move(tree));
  }

  absl::Status HandleEq(CompareOp* eq) override {
    if (IsSingleLeaf(eq->operand(0))) {
      return DefaultHandler(eq);
    }
    auto lhs = GetValue(eq->operand(0));
    auto rhs = GetValue(eq->operand(1));
    changed_ = true;
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Node*> segments,
        (leaf_type_tree::ZipStatus<Node*, Node*, Node*>(
            lhs, rhs,
            [&](Node* lhs_leaf, Node* rhs_leaf) -> absl::StatusOr<Node*> {
              return eq->function_base()->MakeNodeWithName<CompareOp>(
                  eq->loc(), lhs_leaf, rhs_leaf, Op::kEq,
                  NodeNameFormat("%s_decomposed_piece", eq));
            })));
    XLS_ASSIGN_OR_RETURN(
        Node * new_eq,
        NaryAndIfNeeded(eq->function_base(), segments.elements(),
                        NodeNameFormat("%s_decomposed", eq), eq->loc()));
    return SetValue(eq, LeafTypeTree<Node*>(eq->GetType(), {new_eq}));
  }

  absl::Status HandleNe(CompareOp* ne) override {
    if (IsSingleLeaf(ne->operand(0))) {
      return DefaultHandler(ne);
    }
    auto lhs = GetValue(ne->operand(0));
    auto rhs = GetValue(ne->operand(1));
    changed_ = true;
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Node*> segments,
        (leaf_type_tree::ZipStatus<Node*, Node*, Node*>(
            lhs, rhs,
            [&](Node* lhs_leaf, Node* rhs_leaf) -> absl::StatusOr<Node*> {
              return ne->function_base()->MakeNodeWithName<CompareOp>(
                  ne->loc(), lhs_leaf, rhs_leaf, Op::kNe,
                  NodeNameFormat("%s_decomposed_piece", ne));
            })));
    XLS_ASSIGN_OR_RETURN(
        Node * new_ne,
        NaryOrIfNeeded(ne->function_base(), segments.elements(),
                       NodeNameFormat("%s_decomposed", ne), ne->loc()));
    return SetValue(ne, LeafTypeTree<Node*>(ne->GetType(), {new_ne}));
  }

  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    // Normally OHS will go to default-handler if the control is not One hot but
    // we want to always go to the dataflow join. This is because the
    // JoinEelements is where we perform the common 'select-of-aggregate ->
    // aggregate-of-selects' transform so we can't skip it.
    std::vector<LeafTypeTreeView<Node*>> cases;
    for (Node* c : sel->cases()) {
      cases.push_back(GetValue(c));
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> result,
                         Join(cases, {GetValue(sel->selector())}, sel));
    return SetValue(sel, std::move(result));
  }

  absl::Status HandleGate(Gate* gate) override {
    auto data = GetValue(gate->data());
    Node* ctrl = GetValue(gate->condition()).Get({});
    if (IsSingleLeaf(gate) && ctrl == gate->condition() &&
        data.Get({}) == gate->data()) {
      // no need to do anything, not a compound node.
      return SetValue(gate, LeafTypeTree<Node*>(gate->GetType(), {gate}));
    }
    changed_ = true;
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Node*> tree,
        (leaf_type_tree::MapIndex<Node*, Node*>(
            data,
            [&](Type*, Node* data_leaf,
                absl::Span<const int64_t>) -> absl::StatusOr<Node*> {
              return gate->function_base()->MakeNodeWithName<Gate>(
                  gate->loc(), ctrl, data_leaf,
                  NodeNameFormat("%s_decomposed", gate));
            })));
    return SetValue(gate, std::move(tree));
  }

  // Param is handled by DefaultHandler.
  // StateRead is handled by DefaultHandler.
  // Receive is handled by DefaultHandler.

  absl::Status HandleNext(Next* next) override {
    return HandleReconstituteOp(next);
  }

  absl::Status HandleTrace(Trace* trace) override {
    return HandleReconstituteOp(trace);
  }

  absl::Status HandleSend(Send* send) override {
    return HandleReconstituteOp(send);
  }

  absl::Status HandleInvoke(Invoke* invoke) override {
    return HandleReconstituteOp(invoke);
  }

  absl::Status HandleMap(Map* map) override {
    return HandleReconstituteOp(map);
  }
  absl::Status HandleCountedFor(CountedFor* counted_for) override {
    return HandleReconstituteOp(counted_for);
  }

  absl::StatusOr<SharedLeafTypeTree<Node*>> IndexOneArray(
      LeafTypeTreeView<Node*> tree, Node* idx, Node* src) {
    XLS_RET_CHECK(idx->GetType()->IsBits());
    int64_t last_idx = tree.type()->AsArrayOrDie()->size() - 1;
    if (qe_.IsAllZeros(idx)) {
      Bits val = *qe_.KnownValueAsBits(idx);
      if (val.FitsInNBitsUnsigned(63)) {
        XLS_ASSIGN_OR_RETURN(int64_t idx, val.ToUint64());
        if (idx > last_idx) {
          idx = last_idx;
        }
        return tree.AsView({idx}).AsShared();
      } else {
        return tree.AsView({last_idx}).AsShared();
      }
    }
    std::vector<LeafTypeTreeView<Node*>> elements;
    elements.reserve(tree.type()->AsArrayOrDie()->size());
    for (int64_t i = 0; i < tree.type()->AsArrayOrDie()->size(); ++i) {
      elements.push_back(tree.AsView({i}));
    }
    if (Bits::MinBitCountUnsigned(elements.size()) > idx->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          idx, src->function_base()->MakeNodeWithName<ExtendOp>(
                   src->loc(), idx, Bits::MinBitCountUnsigned(elements.size()),
                   Op::kZeroExt, NodeNameFormat("%s_idx_extended", idx)));
    }
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Node*> new_tree,
        (leaf_type_tree::ZipIndex<Node*, Node*>(
            absl::MakeSpan(elements),
            [&](Type*, absl::Span<Node* const* const> elements,
                absl::Span<const int64_t> index) -> absl::StatusOr<Node*> {
              std::vector<Node*> select_cases;
              select_cases.reserve(elements.size());
              for (Node* const* element_ptr :
                   elements.subspan(0, elements.size() - 1)) {
                select_cases.push_back(*element_ptr);
              }
              Node* default_value = *elements.back();
              XLS_ASSIGN_OR_RETURN(
                  Node * new_sel,
                  src->function_base()->MakeNodeWithName<Select>(
                      src->loc(), idx, select_cases, default_value,
                      NodeNameFormat("%s_case_%s_decomposed", src,
                                     absl::StrJoin(index, "_"))));
              return new_sel;
            })));
    return std::move(new_tree).AsShared();
  }
  absl::Status HandleArrayIndex(ArrayIndex* array_index) override {
    changed_ = true;
    SharedLeafTypeTree<Node*> tree = GetValue(array_index->array()).AsShared();
    for (Node* idx : array_index->indices()) {
      XLS_ASSIGN_OR_RETURN(tree,
                           IndexOneArray(tree.AsView(), idx, array_index));
    }
    XLS_RET_CHECK_EQ(tree.type(), array_index->GetType());
    return SetValue(array_index, std::move(tree));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* array_update) override {
    changed_ = true;
    LeafTypeTreeView<Node*> tree = GetValue(array_update->array_to_update());
    LeafTypeTreeView<Node*> update_val = GetValue(array_update->update_value());
    int64_t num_indices = array_update->indices().size();
    XLS_ASSIGN_OR_RETURN(
        LeafTypeTree<Node*> new_array,
        (leaf_type_tree::MapIndex<Node*, Node*>(
            tree,
            [&](Type*, Node* array_leaf,
                absl::Span<const int64_t> index) -> absl::StatusOr<Node*> {
              Node* update_leaf = update_val.Get(index.subspan(num_indices));
              absl::Span<const int64_t> arr_idx = index.subspan(0, num_indices);
              std::vector<Node*> is_idx_match;
              is_idx_match.reserve(num_indices);
              for (int64_t i = 0; i < num_indices; ++i) {
                Node* idx_node = GetValue(array_update->indices()[i]).Get({});
                if (Bits::MinBitCountUnsigned(arr_idx[i]) >
                    idx_node->BitCountOrDie()) {
                  // Cannot possibly be updated.
                  return array_leaf;
                }
                XLS_ASSIGN_OR_RETURN(
                    std::back_inserter(is_idx_match),
                    CompareLiteral(
                        idx_node, arr_idx[i], Op::kEq,
                        NodeNameFormat("%s_idx_%d_match", array_update, i)));
              }
              XLS_ASSIGN_OR_RETURN(
                  Node * all_match,
                  NaryAndIfNeeded(
                      array_update->function_base(), is_idx_match,
                      NodeNameFormat("%s_all_indices_match", array_update)));
              XLS_ASSIGN_OR_RETURN(
                  Node * new_sel,
                  array_update->function_base()->MakeNodeWithName<Select>(
                      array_update->loc(), all_match,
                      absl::Span<Node* const>{array_leaf, update_leaf},
                      /*default_value=*/std::nullopt,
                      NodeNameFormat("%s_decomposed", array_update)));
              return new_sel;
            })));

    return SetValue(array_update, std::move(new_array));
  }

  absl::Status HandleArraySlice(ArraySlice* slice) override {
    changed_ = true;
    LeafTypeTreeView<Node*> tree = GetValue(slice->array());
    std::vector<Node*> new_elements;
    Node* selector = GetValue(slice->start()).Get({});
    int64_t source_array_size =
        slice->array()->GetType()->AsArrayOrDie()->size();
    int64_t selector_width =
        1 + std::max<int64_t>(selector->BitCountOrDie(),
                              Bits::MinBitCountUnsigned(slice->width()));
    std::optional<Node*> selector_ext;
    for (int64_t i = 0; i < slice->width(); ++i) {
      Node* element_idx;
      if (i == 0) {
        element_idx = selector;
      } else if (i >= source_array_size) {
        XLS_ASSIGN_OR_RETURN(
            element_idx,
            slice->function_base()->MakeNode<Literal>(
                slice->loc(), Value(UBits(source_array_size - 1,
                                          Bits::MinBitCountUnsigned(
                                              source_array_size - 1)))));
      } else if (Bits::MinBitCountUnsigned(i) + 1 < selector->BitCountOrDie()) {
        XLS_ASSIGN_OR_RETURN(
            Node * off,
            slice->function_base()->MakeNode<Literal>(
                slice->loc(), Value(UBits(i, selector->BitCountOrDie()))));
        XLS_ASSIGN_OR_RETURN(element_idx,
                             slice->function_base()->MakeNodeWithName<BinOp>(
                                 slice->loc(), selector, off, Op::kAdd,
                                 NodeNameFormat("%s_slice_idx_%d", slice, i)));
      } else {
        XLS_ASSIGN_OR_RETURN(
            Node * off, slice->function_base()->MakeNode<Literal>(
                            slice->loc(), Value(UBits(i, selector_width))));
        if (!selector_ext) {
          XLS_ASSIGN_OR_RETURN(
              selector_ext,
              slice->function_base()->MakeNodeWithName<ExtendOp>(
                  slice->loc(), selector, selector_width, Op::kZeroExt,
                  NodeNameFormat("%s_selector_extended", slice->start())));
        }
        XLS_ASSIGN_OR_RETURN(element_idx,
                             slice->function_base()->MakeNodeWithName<BinOp>(
                                 slice->loc(), *selector_ext, off, Op::kAdd,
                                 NodeNameFormat("%s_slice_idx_%d", slice, i)));
      }
      XLS_ASSIGN_OR_RETURN(auto element,
                           IndexOneArray(tree, element_idx, slice));
      absl::c_copy(element.elements(), std::back_inserter(new_elements));
    }
    return SetValue(slice, LeafTypeTree<Node*>(slice->GetType(), new_elements));
  }

  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override {
    changed_ = true;
    std::vector<Node*> all_leafs;
    all_leafs.reserve(array_concat->GetType()->leaf_count());
    for (Node* arg : array_concat->operands()) {
      LeafTypeTreeView<Node*> tree = GetValue(arg);
      absl::c_copy(tree.elements(), std::back_inserter(all_leafs));
    }
    XLS_RET_CHECK_EQ(all_leafs.size(), array_concat->GetType()->leaf_count());
    return SetValue(array_concat, LeafTypeTree<Node*>(array_concat->GetType(),
                                                      std::move(all_leafs)));
  }

 protected:
  absl::StatusOr<xls::Node*> JoinElements(
      Type* element_type, absl::Span<xls::Node* const* const> data_sources,
      absl::Span<const LeafTypeTreeView<xls::Node*>> control_sources,
      Node* node, absl::Span<const int64_t> index) override {
    // NB: Array ops are handled specially since we can do better than the
    // dataflow visitor default implementation.
    XLS_RET_CHECK(node->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel}));
    XLS_RET_CHECK(!element_type->IsToken());
    if (element_type->IsTuple()) {
      // Special case that empty-tuple is a leaf.
      XLS_RET_CHECK_EQ(element_type->AsTupleOrDie()->size(), 0) << element_type;
      XLS_RET_CHECK_GT(data_sources.size(), 0);
      return *(data_sources[0]);
    }
    XLS_RET_CHECK_EQ(control_sources.size(), 1);
    XLS_ASSIGN_OR_RETURN(auto sel, GenericSelect::From(node));
    // TODO(allight): We could try to use the computed data-sources as our
    // values but that has issues that some are filtered out in complicated
    // ways. It's easier to just recreate the whole select.
    std::vector<Node*> new_cases;
    for (Node* c : sel.cases()) {
      new_cases.push_back(GetValue(c).Get(index));
    }
    Node* selector = GetValue(sel.selector()).Get({});
    std::optional<Node*> new_default =
        sel.default_value()
            ? std::make_optional(GetValue(*sel.default_value()).Get(index))
            : std::nullopt;
    // empty index means this is not a select of a tuple/array.
    if (index.empty() && sel.selector() == selector &&
        sel.cases() == new_cases && sel.default_value() == new_default) {
      // No need to do anything.
      return node;
    }
    changed_ = true;
    XLS_ASSIGN_OR_RETURN(Node * new_sel,
                         sel.CloneSelectLike(selector, new_cases, new_default),
                         _ << "Failed to clone select-like node " << node);
    return new_sel;
  }

 private:
  absl::Status HandleReconstituteOp(Node* node) {
    XLS_RET_CHECK(node->OpIn({Op::kInvoke, Op::kMap, Op::kCountedFor,
                              Op::kTrace, Op::kSend, Op::kNext}));
    std::vector<Node*> new_args;
    new_args.reserve(node->operands().size());
    for (Node* arg : node->operands()) {
      if (node->Is<Next>() &&
          arg ==
              node->function_base()->AsProcOrDie()->GetStateReadByStateElement(
                  node->As<Next>()->state_element())) {
        // Don't decompose state reads. Leave pass-throughs alone and never
        // touch the original read.
        new_args.push_back(arg);
        continue;
      }
      LeafTypeTreeView<Node*> tree = GetValue(arg);
      XLS_ASSIGN_OR_RETURN(*std::back_inserter(new_args),
                           FromTreeOfNodes(node->function_base(), tree,
                                           arg->GetNameView(), arg->loc()));
    }
    if (IsSingleLeaf(node) && new_args == node->operands()) {
      return SetValue(node, LeafTypeTree<Node*>(node->GetType(), {node}));
    }
    changed_ = true;
    for (const auto& [idx, arg] : iter::enumerate(new_args)) {
      XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(idx, arg));
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> tree, ToTreeOfNodes(node));
    return SetValue(node, std::move(tree));
  }

  const QueryEngine& qe_;
  bool changed_ = false;
};

}  // namespace

absl::StatusOr<bool> DecomposeDataflowPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  StatelessQueryEngine qe;
  DecomposeDataflowVisitor visitor(qe);
  XLS_ASSIGN_OR_RETURN(auto topo, context.TopoSort(f));
  for (const auto& n : topo) {
    XLS_RETURN_IF_ERROR(n->VisitSingleNode(&visitor));
  }
  bool changed = visitor.changed();
  if (f->IsFunction()) {
    Function* func = f->AsFunctionOrDie();
    auto ret = visitor.GetValue(func->return_value());
    XLS_ASSIGN_OR_RETURN(
        Node * new_ret,
        FromTreeOfNodes(func, ret, func->return_value()->GetNameView(),
                        func->return_value()->loc()));
    if (new_ret != func->return_value()) {
      changed = true;
      XLS_RETURN_IF_ERROR(func->set_return_value(new_ret));
    }
  }
  return changed;
}

}  // namespace xls
