// Copyright 2020 Google LLC
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

#include "xls/passes/bdd_simplification_pass.h"

#include "absl/container/inlined_vector.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_query_engine.h"

namespace xls {

namespace {

// Returns a conscise string representation of the given node if it is a
// comparator. For example, a kEq with a literal operand might produce:
// "x == 42".
std::string SelectorToString(Node* node) {
  std::string op;
  switch (node->op()) {
    case (OP_EQ):
      op = "==";
      break;
    case (OP_ULT):
      op = "<";
      break;
    case (OP_UGT):
      op = ">";
      break;
    case (OP_ULE):
      op = "<=";
      break;
    case (OP_UGE):
      op = ">=";
      break;
    default:
      return "<unknown>";
  }
  auto node_to_string = [](Node* n) {
    return n->Is<Literal>() ? n->As<Literal>()->value().ToString()
                            : n->GetName();
  };
  return absl::StrFormat("%s %s %s", node_to_string(node->operand(0)), op,
                         node_to_string(node->operand(1)));
}

// Collapse chain of selects with disjoint (one-hot or zero) selectors into a
// single one-hot-select.
xabsl::StatusOr<bool> CollapseSelectChains(Function* f,
                                           const QueryEngine& query_engine) {
  auto is_binary_select = [](Node* node) {
    if (!node->Is<Select>()) {
      return false;
    }
    Select* sel = node->As<Select>();
    return (sel->cases().size() == 2 && !sel->default_value().has_value());
  };
  // A set containing the select instructions collapsed so far so we don't waste
  // time considering selects which have already been optimized.
  absl::flat_hash_set<Select*> collapsed_selects;
  bool modified = false;

  // Walk the graph in reverse order looking for chains of binary selects where
  // case 0 of one select is another binary select. The diagram below shows the
  // relationships in the graph between the nodes in the vectors as the chain is
  // built:
  //   'select_chain' : vector of binary kSelects
  //   'selectors'    : vector of selectors for the kSelects in 'select_chains'
  //   'cases'        : vector of the case=1 operands of the kSelects.
  //
  //                         |  cases[n-1]
  //                         |     |
  //                         V     V
  //  selectors[n-1] ->  select_chain[n-1]
  //                         |
  //                        ...
  //                         |  cases[1]
  //                         |     |
  //                         V     V
  //  selectors[1]}  ->  select_chain[1]
  //                         |
  //                         |  cases[0]
  //                         |     |
  //                         V     V
  //  selectors[0]   ->  select_chain[0]
  //                         |
  //                         V
  // TODO(meheff): Also merge OneHotSelects.
  for (Node* node : ReverseTopoSort(f)) {
    if (!is_binary_select(node) ||
        collapsed_selects.contains(node->As<Select>())) {
      continue;
    }
    std::vector<Select*> select_chain;
    for (Node* s = node; is_binary_select(s); s = s->As<Select>()->cases()[0]) {
      select_chain.push_back(s->As<Select>());
    }
    if (select_chain.size() <= 1) {
      continue;
    }
    XLS_VLOG(4) << absl::StreamFormat("Considering select chain rooted at %s:",
                                      node->ToString());
    if (XLS_VLOG_IS_ON(4)) {
      for (Select* s : select_chain) {
        XLS_VLOG(4) << absl::StreamFormat("  %s // selector: %s", s->ToString(),
                                          SelectorToString(s->selector()));
      }
    }

    std::vector<Node*> selectors(select_chain.size());
    std::transform(select_chain.begin(), select_chain.end(), selectors.begin(),
                   [](Select* s) { return s->selector(); });
    if (!query_engine.AtMostOneNodeTrue(selectors)) {
      XLS_VLOG(4) << "Cannot collapse: more than one selector may be true.";
      continue;
    }
    std::vector<Node*> cases(selectors.size());
    std::transform(select_chain.begin(), select_chain.end(), cases.begin(),
                   [](Select* s) { return s->cases()[1]; });
    if (!query_engine.AtLeastOneNodeTrue(selectors)) {
      // All the selectors may be simultaneously false, so we need to add a
      // "fall-through" case whose selector is the NOR of all the other
      // selectors.
      XLS_VLOG(4) << "All selectors may be false.";
      XLS_ASSIGN_OR_RETURN(Node * nor_of_selectors,
                           node->function()->MakeNode<NaryOp>(
                               node->loc(), std::vector{selectors}, OP_NOR));
      selectors.push_back(nor_of_selectors);
      cases.push_back(select_chain.back()->cases()[0]);
    }
    XLS_VLOG(4) << "Replacing select chain with one-hot-select.";
    XLS_ASSIGN_OR_RETURN(Node * ohs_selector,
                         node->function()->MakeNode<Concat>(
                             node->loc(), std::vector{selectors}));

    // Reverse the direction of the cases so cases[0] which is at bottom of the
    // chain is last in the vector of cases. The concat operation naturally has
    // a reversing effect so no need to reverse the selectors.
    std::reverse(cases.begin(), cases.end());
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<OneHotSelect>(ohs_selector, cases).status());
    collapsed_selects.insert(select_chain.begin(), select_chain.end());
    modified = true;
  }
  return modified;
}

xabsl::StatusOr<bool> SimplifyNode(Node* node, const QueryEngine& query_engine,
                                   bool split_ops) {
  // Replace a sequence of known bits at the least-significant (suffix) or
  // most-significant (prefix) bits of a value.
  if (!node->Is<Literal>() && node->GetType()->IsBits()) {
    // Sequence of known bits at the most-significant end of the value.
    absl::InlinedVector<bool, 1> known_prefix;
    int64 i = node->BitCountOrDie() - 1;
    while (i >= 0 && query_engine.IsKnown(BitLocation{node, i})) {
      known_prefix.push_back(query_engine.IsOne(BitLocation{node, i}));
      --i;
    }
    std::reverse(known_prefix.begin(), known_prefix.end());

    // Sequence of known bits at the least-significant end of the value.
    absl::InlinedVector<bool, 1> known_suffix;
    if (known_prefix.size() != node->BitCountOrDie()) {
      i = 0;
      while (query_engine.IsKnown(BitLocation{node, i})) {
        known_suffix.push_back(query_engine.IsOne(BitLocation{node, i}));
        ++i;
      }
    }
    // If the op has known prefix and/or suffix replace the known bits with a
    // literal concatted with a slice of the original instruction. For example,
    // if the 4 msb of the value 'x' is known, then 'x' is replaced with the
    // following expression:
    //
    //   x: bits[42] = ...
    //   known_prefix: bits[4] = literal(value=...)
    //   sliced_x: bits[38] = bit_slice(x, start=0, width=38)
    //   replacement: bits[42] = concat(known_prefix, sliced_x)
    //
    // If 'x' is already a concat containing a literal operand at the known bit
    // positions then this transformation is not worthwhile as it does no
    // simplification.
    if (node->Is<Concat>()) {
      if (node->operand(0)->Is<Literal>() &&
          node->operand(0)->BitCountOrDie() >= known_prefix.size()) {
        // The known prefix is entirely coming from a literal operand of this
        // concat, don't replace the known prefix as this transformation does no
        // simplication.
        known_prefix.clear();
      }
      if (node->operands().back()->Is<Literal>() &&
          node->operand(0)->BitCountOrDie() >= known_suffix.size()) {
        // The known suffix is entirely coming from a literal operand of this
        // concat, don't replace the known suffix as this transformation does no
        // simplication.
        known_suffix.clear();
      }
    }

    if (!known_prefix.empty() || !known_suffix.empty()) {
      if (known_prefix.size() == node->BitCountOrDie()) {
        XLS_VLOG(1)
            << "Replacing node with its (entirely known) bits: " << node
            << " as "
            << Value(Bits(known_prefix)).ToString(FormatPreference::kBinary);
        XLS_RETURN_IF_ERROR(
            node->ReplaceUsesWithNew<Literal>(Value(Bits(known_prefix)))
                .status());
      } else if (split_ops) {
        std::vector<Node*> old_users(node->users().begin(),
                                     node->users().end());
        std::vector<Node*> concat_elements;
        if (!known_prefix.empty()) {
          XLS_VLOG(1) << node->GetName()
                      << " has known bits prefix: " << Bits(known_prefix);
          XLS_ASSIGN_OR_RETURN(Node * prefix_literal,
                               node->function()->MakeNode<Literal>(
                                   node->loc(), Value(Bits(known_prefix))));
          concat_elements.push_back(prefix_literal);
        }
        XLS_ASSIGN_OR_RETURN(
            Node * sliced_node,
            node->function()->MakeNode<BitSlice>(
                node->loc(), node, /*start=*/known_suffix.size(),
                /*width=*/node->BitCountOrDie() - known_prefix.size() -
                    known_suffix.size()));
        concat_elements.push_back(sliced_node);
        if (!known_suffix.empty()) {
          XLS_VLOG(1) << node->GetName()
                      << " has known bits suffix: " << Bits(known_suffix);
          XLS_ASSIGN_OR_RETURN(Node * suffix_literal,
                               node->function()->MakeNode<Literal>(
                                   node->loc(), Value(Bits(known_suffix))));
          concat_elements.push_back(suffix_literal);
        }
        XLS_ASSIGN_OR_RETURN(
            Node * replacement,
            node->function()->MakeNode<Concat>(node->loc(), concat_elements));
        for (Node* user : old_users) {
          user->ReplaceOperand(node, replacement);
        }
        if (node == node->function()->return_value()) {
          node->function()->set_return_value(replacement);
        }
      }
      return true;
    }
  }

  // Replace two-way kOneHotSelect that has an actual one-hot selector with a
  // kSelect. This can only be done if the selector is one-hot.
  if (split_ops && node->Is<OneHotSelect>() &&
      node->As<OneHotSelect>()->selector()->BitCountOrDie() == 2) {
    OneHotSelect* ohs = node->As<OneHotSelect>();
    if (query_engine.AtLeastOneBitTrue(ohs->selector()) &&
        query_engine.AtMostOneBitTrue(ohs->selector())) {
      XLS_ASSIGN_OR_RETURN(
          Node * bit0_selector,
          node->function()->MakeNode<BitSlice>(node->loc(), ohs->selector(),
                                               /*start=*/0, /*width=*/1));
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Select>(
                  /*selector=*/bit0_selector,
                  std::vector<Node*>{ohs->cases()[1], ohs->cases()[0]},
                  /*default_value*/ absl::nullopt)
              .status());
    }
    return true;
  }

  // Remove kOneHot operations with an input that is already one-hot.
  if (node->Is<OneHot>() && query_engine.AtMostOneBitTrue(node->operand(0))) {
    XLS_ASSIGN_OR_RETURN(
        Node * zero,
        node->function()->MakeNode<Literal>(
            node->loc(),
            Value(UBits(0, /*bit_count=*/node->operand(0)->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(Node * operand_eq_zero,
                         node->function()->MakeNode<CompareOp>(
                             node->loc(), node->operand(0), zero, OP_EQ));
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Concat>(
                                std::vector{operand_eq_zero, node->operand(0)})
                            .status());
    return true;
  }

  return false;
}

}  // namespace

xabsl::StatusOr<bool> BddSimplificationPass::RunOnFunction(
    Function* f, const PassOptions& options, PassResults* results) const {
  XLS_VLOG(2) << "Running BDD simplifier on function " << f->name();
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

  // TODO(meheff): Try tuning the minterm limit.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BddQueryEngine> query_engine,
                       BddQueryEngine::Run(f, /*minterm_limit=*/4096));
  bool modified = false;
  for (Node* node : TopoSort(f)) {
    XLS_ASSIGN_OR_RETURN(bool node_modified,
                         SimplifyNode(node, *query_engine, split_ops_));
    modified |= node_modified;
  }
  XLS_ASSIGN_OR_RETURN(bool selects_collapsed,
                       CollapseSelectChains(f, *query_engine));

  XLS_VLOG(3) << "After:";
  XLS_VLOG_LINES(3, f->DumpIr());

  return modified || selects_collapsed;
}

}  // namespace xls
