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

#include "xls/passes/expression_builder.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/query_engine.h"

namespace xls {

ExpressionBuilder::ExpressionBuilder(FunctionBase* func,
                                     const BddQueryEngine* bdd_engine)
    : unary_op_cache_(),
      binary_op_cache_(),
      bitslice_cache_(),
      func_(func),
      bdd_engine_(bdd_engine),
      tmp_package_("ExpressionBuilderCostEstimatorPkg") {
  tmp_bdd_engine_ = BddQueryEngine::MakeDefault();
  tmp_func_ = tmp_package_.AddFunction(std::make_unique<Function>(
      absl::StrCat("ExpressionBuilderCostEstimatorFn",
                   tmp_package_.functions().size()),
      &tmp_package_));
  CHECK_OK(tmp_bdd_engine_->Populate(tmp_func_).status());
}

bool ExpressionBuilder::Equivalent(Node* one, Node* other) {
  if (one == other) {
    return true;
  }
  if (!one->IsDefinitelyEqualTo(other)) {
    return false;
  }
  if (one->Is<Literal>()) {
    return one->As<Literal>()->value() == other->As<Literal>()->value();
  } else if (one->Is<BitSlice>()) {
    return one->operand(0) == other->operand(0) &&
           one->As<BitSlice>()->start() == other->As<BitSlice>()->start() &&
           one->As<BitSlice>()->width() == other->As<BitSlice>()->width();
  } else if (!one->Is<CompareOp>() && !one->Is<NaryOp>() && !one->Is<UnOp>()) {
    return false;
  }
  for (int i = 0; i < one->operand_count(); ++i) {
    if (!Equivalent(one->operand(i), other->operand(i))) {
      return false;
    }
  }
  return true;
}

bool ExpressionBuilder::Implies(Node* one, Node* other) {
  if (!one->GetType()->IsBits() || !other->GetType()->IsBits()) {
    return false;
  }
  if (one->BitCountOrDie() != other->BitCountOrDie() ||
      one->BitCountOrDie() == 0) {
    return false;
  }
  if (one->function_base() != other->function_base()) {
    return false;
  }
  const BddQueryEngine* bdd_engine = GetBddEngine(one);
  if (!bdd_engine) {
    return false;
  }
  for (int bit = 0; bit < one->BitCountOrDie(); ++bit) {
    if (!bdd_engine->Implies(TreeBitLocation{one, bit},
                             TreeBitLocation{other, bit})) {
      return false;
    }
  }
  return true;
}

std::vector<Node*> ExpressionBuilder::UniqueAndNotImplied(
    std::vector<Node*>& operands, bool keep_node_that_implies_other) {
  if (operands.size() < 2) {
    return operands;
  }
  if (operands.size() == 2) {
    if (Equivalent(operands[0], operands[1]) ||
        Implies(operands[0], operands[1])) {
      if (keep_node_that_implies_other) {
        return {operands[0]};
      } else {
        return {operands[1]};
      }
    }
    if (Implies(operands[1], operands[0])) {
      if (keep_node_that_implies_other) {
        return {operands[1]};
      } else {
        return {operands[0]};
      }
    }
    return operands;
  }

  absl::flat_hash_map<Node*, int> node_to_index;
  std::vector<std::unique_ptr<InlineBitmap>> implieds;
  std::vector<int> implication_count;
  node_to_index.reserve(operands.size());
  implieds.reserve(operands.size());
  implication_count.reserve(operands.size());
  for (int i = 0; i < operands.size(); ++i) {
    node_to_index[operands[i]] = i;
    implieds.push_back(std::make_unique<InlineBitmap>(
        static_cast<int64_t>(operands.size()), false));
    implication_count.push_back(0);
  }
  for (int i = 0; i < operands.size(); ++i) {
    auto one = operands[i];
    for (int j = 0; j < operands.size(); ++j) {
      auto other = operands[j];
      bool other_implied = Equivalent(one, other) || Implies(one, other);
      if (keep_node_that_implies_other) {
        implieds[i]->Set(j, other_implied);
        implication_count[i] += other_implied;
      } else {
        implieds[j]->Set(i, other_implied);
        implication_count[j] += other_implied;
      }
    }
  }
  // Prioritize keeping operands that imply the most others.
  std::vector<Node*> operands_vec{operands.begin(), operands.end()};
  absl::c_sort(operands_vec, [&](Node* a, Node* b) {
    return implication_count[node_to_index[a]] >
           implication_count[node_to_index[b]];
  });
  std::vector<Node*> unique_ops;
  for (int i = 0; i < operands_vec.size(); ++i) {
    auto operand = operands_vec[i];
    bool is_unique = true;
    for (auto other : unique_ops) {
      if (implieds[node_to_index[other]]->Get(node_to_index[operand])) {
        is_unique = false;
        break;
      }
    }
    if (is_unique) {
      unique_ops.push_back(operand);
    }
  }
  return unique_ops;
}

absl::StatusOr<Node*> ExpressionBuilder::FindOrMakeBitSlice(Node* selector,
                                                            int64_t start,
                                                            int64_t width) {
  auto key = std::make_tuple(selector, start, width);
  if (auto it = bitslice_cache_.find(key); it != bitslice_cache_.end()) {
    return it->second;
  }
  for (Node* user : selector->users()) {
    if (user->Is<BitSlice>()) {
      auto bitslice = user->As<BitSlice>();
      if (bitslice->start() == start && bitslice->width() == width) {
        bitslice_cache_[key] = bitslice;
        return bitslice;
      }
    }
  }
  XLS_ASSIGN_OR_RETURN(Node * selector_bits,
                       selector->function_base()->MakeNode<BitSlice>(
                           selector->loc(), selector, start, width));
  bitslice_cache_[key] = selector_bits;
  return selector_bits;
}

absl::StatusOr<Node*> ExpressionBuilder::FindOrMakeUnaryNode(Op op,
                                                             Node* operand) {
  auto key = std::make_pair(op, operand);
  if (auto it = unary_op_cache_.find(key); it != unary_op_cache_.end()) {
    return it->second;
  }
  XLS_ASSIGN_OR_RETURN(Node * result, operand->function_base()->MakeNode<UnOp>(
                                          operand->loc(), operand, op));
  unary_op_cache_[key] = result;
  return result;
}

absl::StatusOr<Node*> ExpressionBuilder::FindOrMakeBinaryNode(Op op, Node* lhs,
                                                              Node* rhs) {
  auto key = std::make_tuple(op, lhs, rhs);
  if (auto it = binary_op_cache_.find(key); it != binary_op_cache_.end()) {
    return it->second;
  }
  bool is_and = op == Op::kAnd || op == Op::kNand;
  bool is_or = op == Op::kOr || op == Op::kNor;
  if (Equivalent(lhs, rhs) || (is_and && Implies(lhs, rhs)) ||
      (is_or && Implies(rhs, lhs))) {
    binary_op_cache_[key] = lhs;
    return lhs;
  }
  if ((is_and && Implies(rhs, lhs)) || (is_or && Implies(lhs, rhs))) {
    binary_op_cache_[key] = rhs;
    return rhs;
  }
  XLS_ASSIGN_OR_RETURN(Node * result,
                       lhs->function_base()->MakeNode<NaryOp>(
                           lhs->loc(), std::vector<Node*>{lhs, rhs}, op));
  binary_op_cache_[key] = result;
  return result;
}

absl::StatusOr<Node*> ExpressionBuilder::FindOrMakeNaryNode(
    Op op, std::vector<Node*>&& operands) {
  if (operands.empty()) {
    return nullptr;
  }
  if (operands.size() == 1) {
    return operands[0];
  }
  absl::c_sort(operands, [](Node* a, Node* b) { return a->id() < b->id(); });
  Node* accum = operands[0];
  for (int i = 1; i < operands.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(accum, FindOrMakeBinaryNode(op, accum, operands[i]));
  }
  return accum;
}

absl::StatusOr<Node*> ExpressionBuilder::OrOperands(
    std::vector<Node*>& operands) {
  if (operands.empty()) {
    return nullptr;
  }
  // For A | B, if A -> B, then A | B is equivalent to B.
  auto needed =
      UniqueAndNotImplied(operands, /*keep_node_that_implies_other=*/false);
  if (needed.size() == 1) {
    return needed[0];
  }
  return FindOrMakeNaryNode(Op::kOr, std::move(needed));
}

absl::StatusOr<Node*> ExpressionBuilder::AndOperands(
    std::vector<Node*>& operands) {
  if (operands.empty()) {
    return nullptr;
  }
  // For A & B, if A -> B, then A & B is equivalent to A.
  auto needed =
      UniqueAndNotImplied(operands, /*keep_node_that_implies_other=*/true);
  if (needed.size() == 1) {
    return needed[0];
  }
  return FindOrMakeNaryNode(Op::kAnd, std::move(needed));
}

absl::StatusOr<Node*> ExpressionBuilder::TmpFuncNodeOrParam(Node* node) {
  if (node->function_base() == tmp_func_) {
    return node;
  }
  return tmp_func_->MakeNode<Param>(
      node->loc(),
      tmp_package_.GetBitsType(node->GetType()->GetFlatBitCount()));
}

}  // namespace xls
