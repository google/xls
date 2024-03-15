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

#include "xls/data_structures/binary_decision_diagram.h"

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"

namespace xls {

BinaryDecisionDiagram::BinaryDecisionDiagram() {
  // Leaf node 0.
  nodes_.push_back(BddNode(BddVariable(-1), BddNodeIndex(-1), BddNodeIndex(-1),
                           /*p=*/1));
  // Leaf node 1.
  nodes_.push_back(BddNode(BddVariable(-1), BddNodeIndex(-1), BddNodeIndex(-1),
                           /*p=*/1));
}

BddNodeIndex BinaryDecisionDiagram::GetOrCreateNode(BddVariable var,
                                                    BddNodeIndex high,
                                                    BddNodeIndex low) {
  NodeKey key = std::make_tuple(var, high, low);
  auto it = node_map_.find(key);
  if (it != node_map_.end()) {
    return it->second;
  }
  if (low == high) {
    return low;
  }
  // Compute the number of paths that the new node will have to the terminal
  // nodes 0 and 1. Use int64s to avoid overflowing and saturate at INT32_MAX.
  int32_t paths = std::min(
      static_cast<int64_t>(GetNode(low).path_count) + GetNode(high).path_count,
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  nodes_.emplace_back(var, high, low, paths);
  BddNodeIndex node_index = BddNodeIndex(nodes_.size() - 1);
  node_map_[key] = node_index;
  return node_index;
}

BddNodeIndex BinaryDecisionDiagram::Restrict(BddNodeIndex expr, BddVariable var,
                                             bool value) {
  if (expr == zero() || expr == one()) {
    return expr;
  }

  const BddNode& node = GetNode(expr);
  CHECK_LE(var, node.variable);
  if (node.variable == var) {
    return value ? node.high : node.low;
  }
  return expr;
}

BddNodeIndex BinaryDecisionDiagram::IfThenElse(BddNodeIndex cond,
                                               BddNodeIndex if_true,
                                               BddNodeIndex if_false) {
  if (cond == one()) {
    return if_true;
  }
  if (cond == zero()) {
    return if_false;
  }
  if (if_true == one() && if_false == zero()) {
    return cond;
  }
  if (if_true == if_false) {
    return if_true;
  }
  auto key = std::make_tuple(cond, if_true, if_false);
  auto it = ite_map_.find(key);
  if (it != ite_map_.end()) {
    return it->second;
  }

  // The expression is non-trivial and has not been computed before. Recursively
  // decompose the expression by peeling away the first variable and performing
  // a Shannon decomposition.

  // First, find the lowest-index variable amongst all expressions. In all paths
  // through the BDD the variable indices are strictly increasing.
  BddVariable min_var = GetNode(cond).variable;
  // Only non-leaf nodes (not zero or one) have associated variables.
  if (if_true != zero() && if_true != one()) {
    min_var = std::min(min_var, GetNode(if_true).variable);
  }
  if (if_false != zero() && if_false != one()) {
    min_var = std::min(min_var, GetNode(if_false).variable);
  }

  // Perform a Shannon expansion about the variable where Shannon expansion is
  // the identity:
  //
  //   F(x0, x1, ..) = !x0 && F(0, x1, ...) + x0 && F(1, x1, ...)
  //
  BddNodeIndex true_cofactor = IfThenElse(Restrict(cond, min_var, true),
                                          Restrict(if_true, min_var, true),
                                          Restrict(if_false, min_var, true));
  BddNodeIndex false_cofactor = IfThenElse(Restrict(cond, min_var, false),
                                           Restrict(if_true, min_var, false),
                                           Restrict(if_false, min_var, false));

  if (true_cofactor == false_cofactor) {
    ite_map_[key] = true_cofactor;
    return true_cofactor;
  }
  BddNodeIndex expr = GetOrCreateNode(min_var, true_cofactor, false_cofactor);
  ite_map_[key] = expr;
  return expr;
}

BddNodeIndex BinaryDecisionDiagram::NewVariable() {
  BddVariable var = next_var_;
  ++next_var_;
  return GetOrCreateNode(var, one(), zero());
}

BddNodeIndex BinaryDecisionDiagram::Not(BddNodeIndex expr) {
  return IfThenElse(expr, zero(), one());
}

BddNodeIndex BinaryDecisionDiagram::Or(BddNodeIndex a, BddNodeIndex b) {
  return IfThenElse(a, one(), b);
}

BddNodeIndex BinaryDecisionDiagram::And(BddNodeIndex a, BddNodeIndex b) {
  return IfThenElse(a, b, zero());
}

absl::StatusOr<bool> BinaryDecisionDiagram::Evaluate(
    BddNodeIndex expr,
    const absl::flat_hash_map<BddNodeIndex, bool>& variable_values) const {
  BddNodeIndex result = expr;
  XLS_VLOG(2) << "Evaluating node: " << static_cast<int64_t>(expr);
  XLS_VLOG(2) << "  expression = " << ToStringDnf(expr, /*minterm_limit=*/5);
  if (VLOG_IS_ON(3)) {
    XLS_VLOG(3) << "  variable values: ";
    std::vector<BddNodeIndex> variables;
    for (const auto& pair : variable_values) {
      variables.push_back(pair.first);
    }
    std::sort(variables.begin(), variables.end());
    for (BddNodeIndex node : variables) {
      XLS_VLOG(3) << "    variable " << GetNode(node).variable << ": "
                  << variable_values.at(node);
    }
  }
  while (result != zero() && result != one()) {
    BddNodeIndex var_node = GetVariableBaseNode(GetNode(result).variable);
    if (!variable_values.contains(var_node)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Missing value for BDD variable %d (node index %d)",
                          GetNode(result).variable.value(), var_node.value()));
    }
    result = variable_values.at(var_node) ? GetNode(result).high
                                          : GetNode(result).low;
  }
  XLS_VLOG(2) << "  result = " << (result == one() ? true : false);
  return result == one();
}

void BinaryDecisionDiagram::ToStringDnfHelper(BddNodeIndex expr,
                                              int64_t* minterms_to_emit,
                                              std::vector<std::string>* terms,
                                              std::string* str) const {
  if (*minterms_to_emit < 0) {
    return;
  }

  if (expr == zero()) {
    return;
  }
  if (expr == one()) {
    if (*minterms_to_emit == 0) {
      absl::StrAppend(str, " + ...");
    } else {
      if (str->empty()) {
        absl::StrAppend(str, absl::StrJoin(*terms, "."));
      } else {
        absl::StrAppend(str, " + ", absl::StrJoin(*terms, "."));
      }
    }
    --*minterms_to_emit;
    return;
  }

  const BddNode& node = GetNode(expr);
  terms->push_back(absl::StrCat("x", node.variable.value()));
  ToStringDnfHelper(node.high, minterms_to_emit, terms, str);
  terms->back() = absl::StrCat("!x", node.variable.value());
  ToStringDnfHelper(node.low, minterms_to_emit, terms, str);
  terms->pop_back();
}

std::string BinaryDecisionDiagram::ToStringDnf(BddNodeIndex expr,
                                               int64_t minterm_limit) const {
  if (expr == zero()) {
    return "0";
  }
  if (expr == one()) {
    return "1";
  }
  std::string result;
  std::vector<std::string> terms;
  int64_t minterms_to_emit =
      minterm_limit == 0 ? std::numeric_limits<int64_t>::max() : minterm_limit;
  ToStringDnfHelper(expr, &minterms_to_emit, &terms, &result);
  return result;
}

}  // namespace xls
