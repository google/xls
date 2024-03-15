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

#include "xls/passes/bdd_simplification_pass.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"

namespace xls {

namespace {

// Returns a concise string representation of the given node if it is a
// comparator. For example, a kEq with a literal operand might produce:
// "x == 42".
std::string SelectorToString(Node* node) {
  std::string op;
  switch (node->op()) {
    case (Op::kEq):
      op = "==";
      break;
    case (Op::kULt):
      op = "<";
      break;
    case (Op::kUGt):
      op = ">";
      break;
    case (Op::kULe):
      op = "<=";
      break;
    case (Op::kUGe):
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
absl::StatusOr<bool> CollapseSelectChains(FunctionBase* f,
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
        collapsed_selects.contains(node->As<Select>()) ||
        !node->GetType()->IsBits()) {
      continue;
    }
    std::vector<Select*> select_chain;
    for (Node* s = node; is_binary_select(s);
         s = s->As<Select>()->get_case(0)) {
      select_chain.push_back(s->As<Select>());
    }
    // Only transform if the select chain is sufficiently long to avoid
    // interfering with select optimizations as plain selects are generally
    // easier to analyse/transform.
    // TODO(meheff): 2021/12/23 Consider tuning this value.
    if (select_chain.size() <= 4) {
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
                   [](Select* s) { return s->get_case(1); });
    if (!query_engine.AtLeastOneNodeTrue(selectors)) {
      // All the selectors may be simultaneously false, so we need to add a
      // "fall-through" case whose selector is the NOR of all the other
      // selectors.
      XLS_VLOG(4) << "All selectors may be false.";
      XLS_ASSIGN_OR_RETURN(Node * nor_of_selectors,
                           node->function_base()->MakeNode<NaryOp>(
                               node->loc(), std::vector{selectors}, Op::kNor));
      selectors.push_back(nor_of_selectors);
      cases.push_back(select_chain.back()->get_case(0));
    }
    XLS_VLOG(4) << "Replacing select chain with one-hot-select.";
    XLS_ASSIGN_OR_RETURN(Node * ohs_selector,
                         node->function_base()->MakeNode<Concat>(
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

absl::StatusOr<bool> SimplifyNode(Node* node, const QueryEngine& query_engine,
                                  int64_t opt_level) {
  // Replace a sequence of known bits at the least-significant (suffix) or
  // most-significant (prefix) bits of a value.
  if (!node->Is<Literal>() && node->GetType()->IsBits()) {
    // Sequence of known bits at the most-significant end of the value.
    absl::InlinedVector<bool, 1> known_prefix;
    int64_t i = node->BitCountOrDie() - 1;
    while (i >= 0 && query_engine.IsKnown(TreeBitLocation(node, i))) {
      known_prefix.push_back(query_engine.IsOne(TreeBitLocation(node, i)));
      --i;
    }
    std::reverse(known_prefix.begin(), known_prefix.end());

    // Sequence of known bits at the least-significant end of the value.
    absl::InlinedVector<bool, 1> known_suffix;
    if (known_prefix.size() != node->BitCountOrDie()) {
      i = 0;
      while (query_engine.IsKnown(TreeBitLocation(node, i))) {
        known_suffix.push_back(query_engine.IsOne(TreeBitLocation(node, i)));
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
        XLS_VLOG(2)
            << "Replacing node with its (entirely known) bits: " << node
            << " as "
            << Value(Bits(known_prefix)).ToString(FormatPreference::kBinary);
        XLS_RETURN_IF_ERROR(
            node->ReplaceUsesWithNew<Literal>(Value(Bits(known_prefix)))
                .status());
        return true;
      }
      if (SplitsEnabled(opt_level)) {
        std::vector<Node*> old_users(node->users().begin(),
                                     node->users().end());
        std::vector<Node*> concat_elements;
        if (!known_prefix.empty()) {
          XLS_VLOG(2) << node->GetName()
                      << " has known bits prefix: " << Bits(known_prefix);
          XLS_ASSIGN_OR_RETURN(Node * prefix_literal,
                               node->function_base()->MakeNode<Literal>(
                                   node->loc(), Value(Bits(known_prefix))));
          concat_elements.push_back(prefix_literal);
        }
        XLS_ASSIGN_OR_RETURN(
            Node * sliced_node,
            node->function_base()->MakeNode<BitSlice>(
                node->loc(), node, /*start=*/known_suffix.size(),
                /*width=*/node->BitCountOrDie() - known_prefix.size() -
                    known_suffix.size()));
        concat_elements.push_back(sliced_node);
        if (!known_suffix.empty()) {
          XLS_VLOG(2) << node->GetName()
                      << " has known bits suffix: " << Bits(known_suffix);
          XLS_ASSIGN_OR_RETURN(Node * suffix_literal,
                               node->function_base()->MakeNode<Literal>(
                                   node->loc(), Value(Bits(known_suffix))));
          concat_elements.push_back(suffix_literal);
        }
        XLS_ASSIGN_OR_RETURN(Node * replacement,
                             node->function_base()->MakeNode<Concat>(
                                 node->loc(), concat_elements));
        for (Node* user : old_users) {
          user->ReplaceOperand(node, replacement);
        }
        XLS_RETURN_IF_ERROR(
            node->ReplaceImplicitUsesWith(replacement).status());
        return true;
      }
    }
  }

  // Replace two-way kOneHotSelect that has an actual one-hot selector with a
  // kSelect. This can only be done if the selector is one-hot.
  if (SplitsEnabled(opt_level) && node->Is<OneHotSelect>() &&
      node->As<OneHotSelect>()->selector()->BitCountOrDie() == 2) {
    OneHotSelect* ohs = node->As<OneHotSelect>();
    if (query_engine.AtLeastOneBitTrue(ohs->selector()) &&
        query_engine.AtMostOneBitTrue(ohs->selector())) {
      XLS_VLOG(2) << absl::StreamFormat(
          "Replacing one-hot-select %swith two-way select", node->GetName());
      XLS_ASSIGN_OR_RETURN(Node * bit0_selector,
                           node->function_base()->MakeNode<BitSlice>(
                               node->loc(), ohs->selector(),
                               /*start=*/0, /*width=*/1));
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Select>(
                  /*selector=*/bit0_selector,
                  std::vector<Node*>{ohs->get_case(1), ohs->get_case(0)},
                  /*default_value*/ std::nullopt)
              .status());
      return true;
    }
  }

  // Remove kOneHot operations with an input that is already one-hot.
  if (node->Is<OneHot>() && query_engine.AtMostOneBitTrue(node->operand(0))) {
    XLS_ASSIGN_OR_RETURN(
        Node * zero,
        node->function_base()->MakeNode<Literal>(
            node->loc(),
            Value(UBits(0, /*bit_count=*/node->operand(0)->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(Node * operand_eq_zero,
                         node->function_base()->MakeNode<CompareOp>(
                             node->loc(), node->operand(0), zero, Op::kEq));
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Concat>(
                                std::vector{operand_eq_zero, node->operand(0)})
                            .status());
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> BddSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  BddQueryEngine query_engine(BddFunction::kDefaultPathLimit, IsCheapForBdds);
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  bool modified = false;
  for (Node* node : TopoSort(f)) {
    XLS_ASSIGN_OR_RETURN(bool node_modified,
                         SimplifyNode(node, query_engine, opt_level_));
    modified |= node_modified;
  }

  XLS_ASSIGN_OR_RETURN(bool selects_collapsed,
                       CollapseSelectChains(f, query_engine));

  return modified || selects_collapsed;
}

XLS_REGISTER_MODULE_INITIALIZER(bdd_simp, {
  CHECK_OK(RegisterOptimizationPass<BddSimplificationPass>(
      "bdd_simp", pass_config::kOptLevel));
  CHECK_OK(RegisterOptimizationPass<BddSimplificationPass>(
      "bdd_simp(2)", pass_config::CappedOptLevel{2}));
  CHECK_OK(RegisterOptimizationPass<BddSimplificationPass>(
      "bdd_simp(3)", pass_config::CappedOptLevel{3}));
});

}  // namespace xls
