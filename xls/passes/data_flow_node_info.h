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

#ifndef XLS_PASSES_DATA_FLOW_NODE_INFO_H_
#define XLS_PASSES_DATA_FLOW_NODE_INFO_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_data.h"
#include "xls/passes/lazy_node_info.h"

namespace xls {

// Specialization of the LazyNodeInfo to track data (leaf nodes, eg bits)
// through compounds, eg tuples, and invokes.
//
// CRTP is used to instantiate sub-caches for each invoke, since their
// parameters can vary. The CRTP type is called Derived.
template <typename Derived, typename Info>
class DataFlowLazyNodeInfo : public LazyNodeInfo<Info> {
 public:
  virtual Info ComputeInfoForBitsLiteral(const xls::Bits& literal) const = 0;
  virtual Info ComputeInfoForLeafNode(Node* node) const = 0;
  virtual Info MergeInfos(const absl::Span<const Info>& infos) const = 0;

  DataFlowLazyNodeInfo()
      : LazyNodeInfo<Info>(DagCacheInvalidateDirection::kInvalidatesUsers) {}
  DataFlowLazyNodeInfo(const DataFlowLazyNodeInfo& o)
      : LazyNodeInfo<Info>(o),
        parent_(o.parent_),
        parent_node_(o.parent_node_) {
    for (const auto& [invoke, callee_info] : o.cached_callee_infos_) {
      auto new_info = std::make_unique<Derived>(*callee_info);
      if (!first_cache_for_function_.contains(invoke->to_apply())) {
        first_cache_for_function_[invoke->to_apply()] = new_info.get();
      }
      cached_callee_infos_[invoke] = std::move(new_info);
    }
  }

  Info GetSingleInfoForNode(Node* node) {
    SharedLeafTypeTree<Info> info = LazyNodeInfo<Info>::GetInfo(node);
    return MergeInfos(info.elements());
  }

 private:
  void GetValueInfos(const xls::Value& value,
                     absl::InlinedVector<Info, 1>& infos) const {
    if (value.IsBits()) {
      infos.push_back(ComputeInfoForBitsLiteral(value.bits()));
      return;
    }
    if (value.IsTuple() || value.IsArray()) {
      for (const xls::Value& element : value.elements()) {
        GetValueInfos(element, infos);
      }
      return;
    }
    LOG(FATAL) << "Unsupported value type";
  }

  void DuplicateInfo(xls::Type* type, const Info& info,
                     absl::InlinedVector<Info, 1>& infos) const {
    if (type->IsBits()) {
      infos.push_back(info);
      return;
    }
    if (type->IsTuple()) {
      for (xls::Type* element_type : type->AsTupleOrDie()->element_types()) {
        DuplicateInfo(element_type, info, infos);
      }
      return;
    }
    if (type->IsArray()) {
      for (int64_t e = 0; e < type->AsArrayOrDie()->size(); ++e) {
        DuplicateInfo(type->AsArrayOrDie()->element_type(), info, infos);
      }
      return;
    }
    LOG(FATAL) << "Unsupported value type";
  }

  LeafTypeTree<Info> ComputeInfo(
      Node* node,
      absl::Span<const LeafTypeTree<Info>* const> operand_infos) const {
    absl::InlinedVector<const LeafTypeTree<Info>*, 1> operand_infos_out;
    for (int64_t i = 0; i < operand_infos.size(); ++i) {
      operand_infos_out.push_back(operand_infos[i]);
    }

    xls::Type* ret_type = node->GetType();

    if (operand_infos.size() == 0) {
      if (node->Is<xls::Literal>()) {
        absl::InlinedVector<Info, 1> infos;
        GetValueInfos(node->As<xls::Literal>()->value(), infos);
        return LeafTypeTree<Info>::CreateFromVector(ret_type, std::move(infos));
      }

      // Fall through to merge and duplicate
    }

    switch (node->op()) {
      case xls::Op::kTupleIndex: {
        const int64_t index = node->As<xls::TupleIndex>()->index();
        LeafTypeTreeView<Info> ret =
            operand_infos[xls::TupleIndex::kArgOperand]->AsView({index});
        return ret.AsShared().ToOwned();
      }
      case xls::Op::kArrayIndex: {
        const absl::Span<Node* const> indices =
            node->As<xls::ArrayIndex>()->indices();
        absl::InlinedVector<int64_t, 1> literal_indices;

        if (AllIndicesLiteral(indices, literal_indices,
                              node->operand(xls::ArrayIndex::kArgOperand)
                                  ->GetType()
                                  ->AsArrayOrDie())) {
          LeafTypeTreeView<Info> ret =
              operand_infos[xls::ArrayIndex::kArgOperand]->AsView(
                  literal_indices);
          return ret.AsShared().ToOwned();
        }

        // With dynamic indexing, merge all infos in the array (but not the
        // index)
        operand_infos_out.clear();
        operand_infos_out.push_back(
            operand_infos[xls::ArrayIndex::kArgOperand]);

        // Fall through to merge and duplicate
        break;
      }
      case xls::Op::kArrayUpdate: {
        xls::ArrayUpdate* array_update = node->As<xls::ArrayUpdate>();

        // Should not include replaced element or index
        absl::InlinedVector<int64_t, 1> literal_indices;

        if (AllIndicesLiteral(array_update->indices(), literal_indices,
                              node->operand(xls::ArrayIndex::kArgOperand)
                                  ->GetType()
                                  ->AsArrayOrDie())) {
          LeafTypeTree<Info> ret = leaf_type_tree::Clone(
              operand_infos[xls::ArrayUpdate::kArgOperand]->AsView());
          MutableLeafTypeTreeView<Info> ret_view =
              ret.AsMutableView(literal_indices);
          leaf_type_tree::ReplaceElements(
              ret.AsMutableView(literal_indices),
              operand_infos[xls::ArrayUpdate::kUpdateValueOperand]->AsView());
          return ret;
        }

        // Dynamic indexing
        const LeafTypeTree<Info>* to_update_info =
            operand_infos[xls::ArrayUpdate::kArgOperand];
        const LeafTypeTree<Info>* replace_info =
            operand_infos[xls::ArrayUpdate::kUpdateValueOperand];

        operand_infos_out.clear();
        operand_infos_out.push_back(to_update_info);
        operand_infos_out.push_back(replace_info);

        // Fall through to merge and duplicate
        break;
      }
      case xls::Op::kIdentity: {
        CHECK_EQ(operand_infos.size(), 1);
        CHECK_EQ(operand_infos.size(), node->operand_count());
        return *operand_infos[0];
      }
      case xls::Op::kTuple: {
        xls::Tuple* tuple = node->As<xls::Tuple>();
        absl::InlinedVector<LeafTypeTreeView<Info>, 1> operand_infos_views;
        operand_infos_views.reserve(tuple->operands().size());
        for (const LeafTypeTree<Info>* operand_info : operand_infos) {
          operand_infos_views.push_back(operand_info->AsView());
        }

        auto retret = leaf_type_tree::CreateTuple<Info>(
            ret_type->AsTupleOrDie(), operand_infos_views);
        CHECK(retret.ok());
        LeafTypeTree<Info> ret = retret.value();
        return ret;
      }
      case xls::Op::kArray: {
        xls::Array* array = node->As<xls::Array>();
        absl::InlinedVector<LeafTypeTreeView<Info>, 1> operand_infos_views;
        operand_infos_views.reserve(array->operands().size());
        for (const LeafTypeTree<Info>* operand_info : operand_infos) {
          operand_infos_views.push_back(operand_info->AsView());
        }

        auto retret = leaf_type_tree::CreateArray<Info>(
            ret_type->AsArrayOrDie(), operand_infos_views);
        CHECK(retret.ok());
        LeafTypeTree<Info> ret = retret.value();
        return ret;
      }

      case xls::Op::kInvoke: {
        // Trace through the callee
        xls::Invoke* invoke = node->As<xls::Invoke>();
        xls::Function* callee = invoke->to_apply();

        if (!cached_callee_infos_.contains(invoke)) {
          // For efficiency: Copy the cache for the first invoke on subsequent
          if (first_cache_for_function_.contains(callee)) {
            const Derived* first_cache = first_cache_for_function_.at(callee);
            cached_callee_infos_[invoke] =
                std::make_unique<Derived>(*first_cache);
          } else {
            auto new_cache = std::make_unique<Derived>();
            first_cache_for_function_[callee] = new_cache.get();
            cached_callee_infos_[invoke] = std::move(new_cache);
          }
          // See comment on mutability for cached_callee_infos_
          cached_callee_infos_.at(invoke)->parent_ =
              const_cast<DataFlowLazyNodeInfo<Derived, Info>*>(this);
          cached_callee_infos_.at(invoke)->parent_node_ = invoke;

          CHECK_NE(LazyNodeData<LeafTypeTree<Info>>::bound_function(), nullptr);

          cached_callee_infos_.at(invoke)->Attach(callee);
        }

        Derived* callee_info = cached_callee_infos_.at(invoke).get();

        // Inject params into sub-info
        CHECK_EQ(invoke->operand_count(), operand_infos.size());
        CHECK_EQ(invoke->operand_count(), callee->params().size());

        for (int64_t p = 0; p < callee->params().size(); ++p) {
          callee_info->SetForced(callee->params().at(p), *operand_infos[p]);
        }

        SharedLeafTypeTree<Info> callee_info_opt =
            callee_info->GetInfo(callee->return_value());

        return callee_info_opt.ToOwned();
      }
      default:
        // Fall through to merge and duplicate
        break;
    };

    // Merge all operand infos
    absl::InlinedVector<Info, 1> infos_in;
    for (int64_t op = 0; op < operand_infos_out.size(); ++op) {
      for (int64_t i = 0; i < operand_infos_out[op]->elements().size(); ++i) {
        infos_in.push_back(operand_infos_out[op]->elements()[i]);
      }
    }

    Info ret;
    if (infos_in.size() == 0) {
      ret = ComputeInfoForLeafNode(node);
    } else {
      ret = MergeInfos(infos_in);
    }

    absl::InlinedVector<Info, 1> infos;
    DuplicateInfo(ret_type, ret, infos);
    return LeafTypeTree<Info>::CreateFromVector(ret_type, std::move(infos));
  }

  absl::Status MergeWithGiven(Info& info, const Info& given) const final {
    return absl::UnimplementedError(
        "DataFlowLazyNodeInfo::MergeWithGiven() Not implemented");
  }

  void NodeAdded(Node* node) override {
    LazyNodeInfo<Info>::NodeAdded(node);
    ReportChangeToParent();
  }
  void NodeDeleted(Node* node) override {
    LazyNodeInfo<Info>::NodeDeleted(node);
    // Delete now-unused caches for efficiency
    if (node->Is<xls::Invoke>()) {
      xls::Invoke* invoke = node->As<xls::Invoke>();
      cached_callee_infos_.erase(invoke);
      first_cache_for_function_.erase(invoke->to_apply());
    }
    ReportChangeToParent();
  }

  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override {
    LazyNodeInfo<Info>::OperandChanged(node, old_operand, operand_nos);
    ReportChangeToParent();
  }

  void OperandRemoved(Node* node, Node* old_operand) override {
    LazyNodeInfo<Info>::OperandRemoved(node, old_operand);
    ReportChangeToParent();
  }

  void OperandAdded(Node* node) override {
    LazyNodeInfo<Info>::OperandAdded(node);
    ReportChangeToParent();
  }

  void ReturnValueChanged(Function* function_base,
                          Node* old_return_value) override {
    LazyNodeInfo<Info>::ReturnValueChanged(function_base, old_return_value);
    ReportChangeToParent();
  }
  void NextStateElementChanged(Proc* proc, int64_t state_index,
                               Node* old_next_state_element) override {
    LazyNodeInfo<Info>::NextStateElementChanged(proc, state_index,
                                                old_next_state_element);
    ReportChangeToParent();
  }

  void ReportChangeToParent() {
    if (parent_ != nullptr) {
      CHECK_NE(parent_node_, nullptr);
      parent_->ForceRecompute(parent_node_);
    }
  }

  bool AllIndicesLiteral(
      absl::Span<Node* const> indices,
      absl::InlinedVector<int64_t, 1>& literal_indices,
      std::optional<xls::ArrayType*> array_type_opt = std::nullopt) const {
    xls::ArrayType* array_type = array_type_opt.value_or(nullptr);

    bool all_indices_literal = true;
    for (Node* index_node : indices) {
      if (!index_node->Is<xls::Literal>()) {
        all_indices_literal = false;
        break;
      }
      absl::StatusOr<uint64_t> index_ret =
          index_node->As<xls::Literal>()->value().bits().ToUint64();
      CHECK(index_ret.ok());
      uint64_t index = index_ret.value();
      // Clamp out of bounds literal index
      if (array_type != nullptr && index >= array_type->size()) {
        index = array_type->size() - 1;
      }
      literal_indices.push_back(index);
      if (array_type != nullptr && array_type->element_type()->IsArray()) {
        array_type = array_type->element_type()->AsArrayOrDie();
      }
    }
    return all_indices_literal;
  }

  DataFlowLazyNodeInfo<Derived, Info>* parent_ = nullptr;
  xls::Node* parent_node_ = nullptr;

  // Each invoke can have different input parameters
  // This is mutable because ComputeInfo() is const. The cache has
  // immutable characteristics, however, so it is safe.
  mutable absl::flat_hash_map<xls::Invoke*, std::unique_ptr<Derived>>
      cached_callee_infos_;

  mutable absl::flat_hash_map<xls::Function*, const Derived*>
      first_cache_for_function_;
};

}  // namespace xls

#endif  // XLS_PASSES_DATA_FLOW_NODE_INFO_H_
