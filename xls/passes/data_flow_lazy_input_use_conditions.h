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

#ifndef XLS_PASSES_DATA_FLOW_LAZY_INPUT_USE_CONDITIONS_H_
#define XLS_PASSES_DATA_FLOW_LAZY_INPUT_USE_CONDITIONS_H_

#include <cstdint>
#include <list>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_info.h"
#include "xls/passes/query_engine.h"

namespace xls {

struct ConditionsBySinkNodeOrder {
  bool operator()(const Node* a, const Node* b) const {
    return a->id() < b->id();
  }
};

typedef absl::btree_map<Node*, Node*, ConditionsBySinkNodeOrder>
    ConditionsBySinkNode;

// This class lazily computes the conditions in which inputs, such as
// parameters, are used by a given output. It does this by working backwards
// from outputs, such as a function's return value.
//
// For example:
// fn Select(s: bits[2] id=1, x: bits[32] id=2, y: bits[32] id=3, z: bits[32]
// id=4) -> bits[32] {
//   sel: bits[32] = sel(s, cases=[y, x], default=z, id=5)
//   ret ret_node: bits[32] = identity(sel, id=6)
// }
//
// The parameter x has one sink, ret_node, and it is used by that sink under
// the condition that s == 1.
//
// NOTE: Currently only functions are supported.
// NOTE: This class may create temporary nodes to express condition expressions.
// These are deleted when the class is destroyed, so if they are used to
// generate IR output, they should be cloned first.
class DataFlowLazyInputUseConditions
    : public LazyNodeInfo<ConditionsBySinkNode> {
 public:
  explicit DataFlowLazyInputUseConditions()
      : LazyNodeInfo<ConditionsBySinkNode>(
            DagCacheInvalidateDirection::kInvalidatesOperands) {}

  // Don't double delete temporary nodes
  DataFlowLazyInputUseConditions(const DataFlowLazyInputUseConditions&) =
      delete;

  ~DataFlowLazyInputUseConditions() {
    std::list<Node*> nodes_to_remove_ordered;
    for (Node* node : temporary_nodes_) {
      nodes_to_remove_ordered.push_back(node);
    }

    // Remove temporary nodes according to their topology.
    // Nodes with users cannot be removed.
    while (!nodes_to_remove_ordered.empty()) {
      nodes_to_remove_ordered.sort([](Node* a, Node* b) {
        return a->users().size() < b->users().size();
      });

      CHECK(nodes_to_remove_ordered.front()->users().empty());

      while (!nodes_to_remove_ordered.empty()) {
        Node* node = nodes_to_remove_ordered.front();

        if (!node->users().empty()) {
          break;
        }

        CHECK_OK(node->function_base()->RemoveNode(node));

        nodes_to_remove_ordered.erase(nodes_to_remove_ordered.begin());
      }
    }
  }

  void set_query_engine(QueryEngine* query_engine) {
    query_engine_ = query_engine;
  }

  absl::Status MergeWithGiven(ConditionsBySinkNode& info,
                              const ConditionsBySinkNode& given) const final {
    return absl::UnimplementedError(
        "DataFlowLazyInputUseConditions::MergeWithGiven() Not implemented");
  }

  ConditionsBySinkNode GetConditionsForNode(Node* node) const {
    SharedLeafTypeTree<ConditionsBySinkNode> info =
        LazyNodeInfo<ConditionsBySinkNode>::GetInfo(node);
    absl::InlinedVector<ConditionsBySinkNode, 1> infos_flat;
    infos_flat.reserve(info.elements().size());
    for (const ConditionsBySinkNode& info_in : info.elements()) {
      infos_flat.push_back(info_in);
    }
    return MergeInfosFlat(absl::MakeSpan(infos_flat));
  }

  Literal* GetLiteralOne(FunctionBase* function) const {
    if (literal_ones_by_func_.contains(function)) {
      return literal_ones_by_func_.at(function);
    }
    Value one(UBits(1, 1));
    Literal* literal = CreateTemporaryNode<Literal>(function, one);
    literal_ones_by_func_[function] = literal;
    return literal;
  }

 protected:
  template <typename NodeT, typename... Args>
  NodeT* CreateTemporaryNode(FunctionBase* function, Args&&... args) const {
    absl::StatusOr<NodeT*> node = function->MakeNode<NodeT>(
        xls::SourceInfo(), std::forward<Args>(args)...);
    CHECK_OK(node.status());
    temporary_nodes_.insert(*node);
    return *node;
  }

  // Backward analysis: "Inputs" are the users of the node in the IR.
  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->users();
  }

  // Backward analysis: "Users" are the operands of the node in the IR.
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->operands();
  }

  LeafTypeTree<ConditionsBySinkNode> DuplicateInfo(
      Type* out_type, const ConditionsBySinkNode& info) const {
    LeafTypeTree<std::monostate> layout(out_type);
    absl::InlinedVector<ConditionsBySinkNode, 1> infos(layout.elements().size(),
                                                       info);
    return LeafTypeTree<ConditionsBySinkNode>::CreateFromVector(
        out_type, std::move(infos));
  }

  ConditionsBySinkNode MergeInfosFlat(
      const absl::Span<ConditionsBySinkNode>& infos_in) const {
    ConditionsBySinkNode out_info;
    for (const ConditionsBySinkNode& info_in : infos_in) {
      for (const auto& [sink_in, condition_in] : info_in) {
        if (!out_info.contains(sink_in) ||
            out_info.at(sink_in) == condition_in) {
          out_info[sink_in] = condition_in;
        } else {
          out_info[sink_in] = CreateTemporaryNode<NaryOp>(
              condition_in->function_base(),
              std::vector<Node*>{out_info.at(sink_in), condition_in}, Op::kOr);
        }
      }
    }
    return out_info;
  }

  LeafTypeTree<ConditionsBySinkNode> MergeInfosElementWise(
      Type* out_type,
      const absl::InlinedVector<LeafTypeTree<ConditionsBySinkNode>, 1>&
          in_infos) const {
    LeafTypeTree<ConditionsBySinkNode> out_tree(out_type);
    for (int64_t i = 0; i < out_tree.elements().size(); ++i) {
      absl::InlinedVector<ConditionsBySinkNode, 1> infos_this_elem;
      for (const LeafTypeTree<ConditionsBySinkNode>& in_tree : in_infos) {
        CHECK_EQ(out_tree.elements().size(), in_tree.elements().size());
        infos_this_elem.push_back(in_tree.elements().at(i));
      }
      out_tree.elements()[i] = MergeInfosFlat(absl::MakeSpan(infos_this_elem));
    }
    return out_tree;
  }

  bool NodeIsOutput(Node* node) const {
    CHECK(node->function_base()->IsFunction());
    return node->function_base()->AsFunctionOrDie()->return_value() == node;
  }

  void ApplyCondition(LeafTypeTree<ConditionsBySinkNode>& user_tree,
                      Node* condition_node) const {
    for (ConditionsBySinkNode& sinks : user_tree.elements()) {
      ConditionsBySinkNode original_sinks = std::move(sinks);
      sinks.clear();
      for (const auto& [orig_sink, orig_condition] : original_sinks) {
        sinks[orig_sink] = CreateTemporaryNode<NaryOp>(
            condition_node->function_base(),
            std::vector<Node*>{condition_node, orig_condition}, Op::kAnd);
      }
    }
  }

  LeafTypeTree<ConditionsBySinkNode> ComputeInfo(
      Node* node,
      absl::Span<const LeafTypeTree<ConditionsBySinkNode>* const> operand_infos)
      const override {
    if (temporary_nodes_.contains(node)) {
      return LeafTypeTree<ConditionsBySinkNode>(node->GetType());
    }

    // For clarify, while preserving parameter name
    absl::InlinedVector<LeafTypeTree<ConditionsBySinkNode>, 1> user_infos;
    user_infos.reserve(node->users().size());
    for (const LeafTypeTree<ConditionsBySinkNode>* const user_info_ptr :
         operand_infos) {
      user_infos.push_back(*user_info_ptr);
    }
    absl::Span<Node* const> users = node->users();
    CHECK_EQ(user_infos.size(), users.size());

    if (NodeIsOutput(node)) {
      ConditionsBySinkNode ret = {{node, GetLiteralOne(node->function_base())}};
      return DuplicateInfo(node->GetType(), ret);
    }

    if (node->Is<Literal>()) {
      return LeafTypeTree<ConditionsBySinkNode>(node->GetType());
    }

    for (int64_t user_idx = 0; user_idx < users.size(); ++user_idx) {
      Node* user_node = users.at(user_idx);
      LeafTypeTree<ConditionsBySinkNode>& user_tree = user_infos.at(user_idx);
      switch (user_node->op()) {
        case Op::kSel: {
          Select* user_select = user_node->As<Select>();
          Node* selector = user_select->selector();

          if (node != selector) {
            CHECK(selector->GetType()->IsBits());

            std::optional<Value> selector_value_known =
                query_engine_->KnownValue(selector);

            if (selector_value_known.has_value()) {
              // Static selector
              CHECK(selector_value_known.value().IsBits());
              CHECK_EQ(selector_value_known.value().GetFlatBitCount(),
                       selector->GetType()->GetFlatBitCount());
              int64_t known_case_index =
                  selector_value_known.value().bits().ToInt64().value();
              CHECK_GE(known_case_index, 0);
              Node* case_selected_node =
                  (known_case_index < user_select->cases().size())
                      ? user_select->cases().at(known_case_index)
                      : user_select->default_value().value();
              if (node != case_selected_node) {
                // Clear sinks for all but selected case.
                user_tree = LeafTypeTree<ConditionsBySinkNode>(node->GetType());
              } else {
                CHECK(
                    node->GetType()->IsEqualTo(case_selected_node->GetType()));
              }
            } else {
              // Dynamic selector: Add the condition based on which case it is
              std::optional<int64_t> case_index_found = std::nullopt;
              for (int64_t case_idx = 0; case_idx < user_select->cases().size();
                   ++case_idx) {
                if (user_select->cases().at(case_idx) == node) {
                  case_index_found = case_idx;
                  break;
                }
              }

              Node* condition_node = nullptr;

              if (case_index_found.has_value()) {
                Literal* case_literal = CreateTemporaryNode<Literal>(
                    user_node->function_base(),
                    Value(UBits(case_index_found.value(),
                                selector->BitCountOrDie())));
                condition_node = CreateTemporaryNode<CompareOp>(
                    user_node->function_base(), selector, case_literal,
                    Op::kEq);
              } else {
                CHECK(user_select->default_value().has_value());
                CHECK_EQ(node, user_select->default_value().value());
                Literal* case_literal = CreateTemporaryNode<Literal>(
                    user_node->function_base(),
                    Value(UBits(user_select->cases().size(),
                                selector->BitCountOrDie())));
                condition_node = CreateTemporaryNode<CompareOp>(
                    user_node->function_base(), selector, case_literal,
                    Op::kUGe);
              }

              CHECK(node->GetType()->IsEqualTo(user_node->GetType()));

              ApplyCondition(user_tree, condition_node);
            }
          }

          break;
        }
        default: {
          // If type is different from user, then merge and duplicate
          for (int64_t user_idx = 0; user_idx < users.size(); ++user_idx) {
            Node* user_node = users.at(user_idx);
            LeafTypeTree<ConditionsBySinkNode>& user_tree =
                user_infos.at(user_idx);

            if (user_node->GetType()->IsEqualTo(node->GetType())) {
              continue;
            }

            user_tree = DuplicateInfo(node->GetType(),
                                      MergeInfosFlat(user_tree.elements()));
          }

          break;
        }
      }
    }

    // Merge the elements of all users' infos
    return MergeInfosElementWise(node->GetType(), user_infos);
  }

 private:
  xls::QueryEngine* query_engine_ = nullptr;

  mutable absl::flat_hash_map<FunctionBase*, Literal*> literal_ones_by_func_;
  mutable absl::flat_hash_set<Node*> temporary_nodes_;
};

}  // namespace xls

#endif  // XLS_PASSES_DATA_FLOW_LAZY_INPUT_USE_CONDITIONS_H_
