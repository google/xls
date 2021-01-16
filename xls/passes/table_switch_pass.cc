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
#include "xls/passes/table_switch_pass.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"

namespace xls {

// SelectChain represents a "chain" of Select nodes, i.e., a graph of Select
// nodes that choose between a Select node and a Literal which terminates in a
// Select node that chooses between two literals. This sort of structure maps to
// if A, then X else if B then Y else if C then Z, ...
// This sort of structure maps to an ArrayIndex node - trivially if the indices
// being checked have unit increment starting from 0, so this is used to
// determine if an IR graph contains any chained Selects that could be
// simplified in that manner.
class SelectChain {
 public:
  // If a chain is shorter than this, don't bother to tableize.
  static constexpr int kMinimumSize = 3;

  // Potential enhancements: This currently performs only the most basic
  // matching and could be significantly enhanced by:
  //  - Handling non-zero start indexes.
  //  - Handling descending increments (from N -> 0).
  //  - Support non-1 index increments:
  //    - Extract a constant factor, e.g., 0, 4, 8, ... -> 0, 1, 2, ...
  //    - Re-order indexes to be monotonically ascending/descending, i.e.,
  //      0, 3, 1, 2, ... -> 0, 1, 2, 3, ...
  //  - Handle "partial" chains - those that only cover part of the match space
  //    (i.e., table-switch indexes 0-N of a 0-M range (where M > N).
  static absl::StatusOr<absl::optional<SelectChain>> TryCreate(FunctionBase* f,
                                                               Select* root) {
    SelectChain chain;
    XLS_ASSIGN_OR_RETURN(bool success, chain.Init(root));
    if (!success) {
      return absl::nullopt;
    }
    return chain;
  }

  bool Contains(Select* node) const { return elements_.contains(node); }

  int64 size() const { return nodes_.size(); }

  // Attempts to do the actual replacement of the Select nodes in the chain with
  // a literal array and ArrayIndex. At this point, the SelectChain should be
  // valid (checked in Init() below), so any failure should be an error.
  absl::Status ReplaceWithArrayIndex() const {
    FunctionBase* f = nodes_.front()->function_base();

    // If increment_ is -1, that means we reversed the vector, so then the last
    // node in the chain is now the back (remember, we start a chain w/the
    // "last" node, i.e., the one on which no other chain element depends), and
    // the first is in front.
    int first_chain_index = nodes_.size() - 1;
    if (bits_ops::SLessThan(increment_, 0)) {
      first_chain_index = 0;
    }

    std::vector<Value> values;
    values.reserve(nodes_.size() + 1);
    for (int i = 0; i < nodes_.size(); i++) {
      // Each node will have a single literal as the RHS of the select,
      // except for the last, which will have one on both;
      values.push_back(nodes_[i]->operand(2)->As<Literal>()->value());
    }
    // The "false"/0 case in the terminal chain element is essentially the
    // "else" for any indices not matched by the chain. ArrayIndex semantics is
    // for OOB accesses to return the last element in the array, so we put that
    // OOB-capturing else case there.
    values.push_back(
        nodes_[first_chain_index]->operand(1)->As<Literal>()->value());

    XLS_ASSIGN_OR_RETURN(Value array, Value::Array(values));
    XLS_ASSIGN_OR_RETURN(Literal * literal,
                         f->MakeNode<Literal>(nodes_.front()->loc(), array));
    if (bits_ops::SGreaterThan(increment_, 0)) {
      return nodes_.front()
          ->ReplaceUsesWithNew<ArrayIndex>(literal,
                                           std::vector<Node*>({eq_var_}))
          .status();
    }
    return nodes_.back()
        ->ReplaceUsesWithNew<ArrayIndex>(literal, std::vector<Node*>({eq_var_}))
        .status();
  }

 private:
  enum class MatchChainResult {
    kNoMatch,
    kInnerMatch,
    kTerminalMatch,
  };

  // Returns true if we were able to create a chain from this root node, or a
  // Status if an error occurred in that evaluation.
  absl::StatusOr<bool> Init(Select* root) {
    // Walk this candidate, returning an error if it's not a valid chain.
    if (!root->operand(0)->Is<CompareOp>()) {
      return absl::InvalidArgumentError(
          "Select node selector needs a compare op!");
    }

    CompareOp* selector = root->operand(0)->As<CompareOp>();
    if (selector->op() != Op::kEq) {
      return absl::InvalidArgumentError("Chains only match on equality");
    }

    // The variable arg of the select comparison; must be the same for all
    // elements in the chain.
    XLS_ASSIGN_OR_RETURN(eq_var_, GetNonliteralArg(selector));

    bool done = false;
    Add(root);
    Node* curr = root;
    Select* prev = nullptr;
    while (!done) {
      // Walk up the chain, verifying that all elements are links or a terminal
      // and compare to the same value.
      // IsChainTerminal and IsChainElement verify that current is a Select.
      prev = curr->As<Select>();
      curr = curr->operand(1);
      XLS_ASSIGN_OR_RETURN(MatchChainResult mr, MatchChain(curr, prev));
      if (mr == MatchChainResult::kNoMatch) {
        return false;
      }

      Add(curr->As<Select>());
      if (mr == MatchChainResult::kTerminalMatch) {
        break;
      }
    }

    if (size() < kMinimumSize) {
      XLS_VLOG(3) << "Chain is too short to tablefy.";
      return false;
    }

    Bits nodes_size = SBits(nodes_.size(), increment_.bit_count());
    Bits one = SBits(1, increment_.bit_count());
    for (int i = 0; i < nodes_.size(); i++) {
      XLS_ASSIGN_OR_RETURN(
          Literal * literal,
          GetLiteralArg(nodes_[i]->operand(0)->As<CompareOp>()));
      Bits index = literal->value().bits();
      Bits expected_index = SBits(i, index.bit_count());
      if (bits_ops::SLessThan(increment_, 0)) {
        Bits bits_i = SBits(i, increment_.bit_count());
        expected_index = bits_ops::Sub(bits_ops::Sub(nodes_size, bits_i), one);
      }
      if (expected_index != index) {
        return false;
      }
    }

    // Validate this is 0-N or N-0; reverse if necessary, so we can emit as 0-N.
    if (increment_ == SBits(-1, increment_.bit_count())) {
      std::reverse(nodes_.begin(), nodes_.end());
    }

    return true;
  }

  absl::StatusOr<MatchChainResult> MatchChain(Node* curr_node, Select* prev) {
    if (!curr_node->Is<Select>()) {
      return MatchChainResult::kNoMatch;
    }

    Select* curr = curr_node->As<Select>();
    if (IsChainTerminal(curr)) {
      return MatchChainResult::kTerminalMatch;
    }

    if (!SelectIsChainShaped(curr)) {
      return MatchChainResult::kNoMatch;
    }

    // Determine the expected increment based on the indices of the 1st and
    // 2nd nodes.
    XLS_ASSIGN_OR_RETURN(Literal * prev_literal,
                         GetLiteralArg(prev->operand(0)->As<CompareOp>()));
    XLS_ASSIGN_OR_RETURN(Literal * curr_literal,
                         GetLiteralArg(curr->operand(0)->As<CompareOp>()));
    // This check isn't _really_ necessary, since comparisons must be between
    // Bits types, but defense is the best offense (really, this protects us in
    // case those semantics change in the future).
    if (!prev_literal->value().IsBits() || !curr_literal->value().IsBits()) {
      return MatchChainResult::kNoMatch;
    }
    Bits prev_bits = prev_literal->value().bits();
    Bits curr_bits = curr_literal->value().bits();
    if (curr_bits.bit_count() != prev_bits.bit_count()) {
      return MatchChainResult::kNoMatch;
    }
    if (prev == nodes_.front()) {
      // On the first iter, just set the expected increment.
      // TODO(rspringer): Matching increments shouldn't be necessary (and really
      // makes some of this code inelegant. Instead, we can keep a map of
      // matched literal -> value along with the "else" case, and use that to A)
      // simplify this code and B) match sparse index sets. It's a win-win-win.
      increment_ = bits_ops::Sub(curr_bits, prev_bits);
      // Verify the initial increment is 1 or -1.
      if (bits_ops::Abs(increment_).Get(0) != 1) {
        XLS_VLOG(3) << "Index increment isn't unit: " << increment_;
        return MatchChainResult::kNoMatch;
      }
    } else {
      // On subsequent iters, make sure it's consistent.
      Bits new_increment = bits_ops::Sub(curr_bits, prev_bits);
      if (new_increment != increment_) {
        XLS_VLOG(3) << "Index increment isn't consistent: " << new_increment
                    << " vs. " << increment_;
        return MatchChainResult::kNoMatch;
      }
    }

    return MatchChainResult::kInnerMatch;
  }

  // Returns true of the given node is a "chain-shaped" Select, i.e., that it's
  // a Select and chooses between a Select and a Literal.
  bool SelectIsChainShaped(Select* select) {
    if (VerifyCompareOp(select->selector()) && select->cases().size() == 2 &&
        select->get_case(0)->op() == Op::kSel &&
        select->get_case(1)->op() == Op::kLiteral) {
      return true;
    }

    return false;
  }

  // Returns true if this node is the terminal node in a select chain, i.e.,
  // that chooses between two Literals, instead of a Select and a Literal.
  bool IsChainTerminal(Select* select) const {
    // 1. Make sure the node is a select-eq, and that the comparison var is
    //    the same node as the rest in the chain,
    // 2. Make sure both options are literals.
    if (VerifyCompareOp(select->selector()) && select->cases().size() == 2 &&
        select->get_case(0)->op() == Op::kLiteral &&
        select->get_case(1)->op() == Op::kLiteral) {
      return true;
    }

    return false;
  }

  // Adds a node to the back of the select chain.
  void Add(Select* node) {
    nodes_.push_back(node);
    elements_.insert(node);
  }

  // Verifies that the given Node is a select, its op is equals, and that its
  // free var is the same as the given.
  bool VerifyCompareOp(Node* node) const {
    if (!node->Is<CompareOp>()) {
      return false;
    }

    const CompareOp* op = node->As<CompareOp>();
    if (op->op() != Op::kEq) {
      return false;
    }

    return op->operand(0) == eq_var_;
  }

  // Returns the literal parameter to the given comparison op (if any).
  absl::StatusOr<Literal*> GetLiteralArg(CompareOp* op) const {
    Node* eq_lhs = op->operand(0);
    Node* eq_rhs = op->operand(1);
    if (eq_lhs->Is<Literal>() && !eq_rhs->Is<Literal>()) {
      return eq_lhs->As<Literal>();
    } else if (!eq_lhs->Is<Literal>() && eq_rhs->Is<Literal>()) {
      return eq_rhs->As<Literal>();
    } else {
      return absl::InvalidArgumentError(
          "Op either has no or both literal operands.");
    }
  }

  // Returns the nonliteral parameter to the given comparison op (if any).
  absl::StatusOr<Node*> GetNonliteralArg(CompareOp* op) const {
    Node* eq_lhs = op->operand(0);
    Node* eq_rhs = op->operand(1);
    if (eq_lhs->Is<Literal>() && !eq_rhs->Is<Literal>()) {
      return eq_rhs;
    } else if (!eq_lhs->Is<Literal>() && eq_rhs->Is<Literal>()) {
      return eq_lhs;
    } else {
      return absl::InvalidArgumentError(
          "Op either has no or both literal operands.");
    }
  }

  absl::flat_hash_set<Select*> elements_;
  std::vector<Select*> nodes_;
  Node* eq_var_;
  Bits increment_;
};

// This excludes the first/last element in a chain (sel(eq, literal, literal)),
// but there's no point in table-switching a two-element table.
bool IsChainCandidate(Node* node) {
  if (!node->Is<Select>()) {
    return false;
  }

  Select* select = node->As<Select>();
  return select->selector()->op() == Op::kEq && select->cases().size() == 2 &&
         select->get_case(0)->op() == Op::kSel &&
         select->get_case(1)->op() == Op::kLiteral;
}

// Returns true if this node is already in another SelectChain.
bool IsInChain(Select* select, const std::vector<SelectChain>& chains) {
  for (const auto& chain : chains) {
    if (chain.Contains(select)) {
      return true;
    }
  }

  return false;
}

absl::StatusOr<bool> TableSwitchPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  // Find all candidate starts - sel(eq, sel, lit). Walk up from each (I guess);
  // but we only need the first in each chain, so let's reverse toposort,
  // and if node X is already in the chain of Y, then skip it.

  std::vector<SelectChain> chains;
  for (Node* node : ReverseTopoSort(f)) {
    if (IsChainCandidate(node) && !IsInChain(node->As<Select>(), chains)) {
      XLS_ASSIGN_OR_RETURN(absl::optional<SelectChain> chain,
                           SelectChain::TryCreate(f, node->As<Select>()));
      if (chain) {
        chains.push_back(*chain);
      }
    }
  }

  XLS_VLOG(2) << "Potential chains: " << chains.size();
  bool changed = false;
  for (const SelectChain& chain : chains) {
    XLS_RETURN_IF_ERROR(chain.ReplaceWithArrayIndex());
    changed = true;
  }

  return changed;
}

}  // namespace xls
