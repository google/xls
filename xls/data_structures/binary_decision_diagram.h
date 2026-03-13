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

#ifndef XLS_DATA_STRUCTURES_BINARY_DECISION_DIAGRAM_H_
#define XLS_DATA_STRUCTURES_BINARY_DECISION_DIAGRAM_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/strong_int.h"
#include "xls/data_structures/inline_bitmap.h"

namespace xls {

// An efficient implementation of binary decision diagrams (BDD). This
// implementation allows an arbitrary number of expressions over a set of
// variables to be represented in a single BDD.
//
// Based on:
//   K.S. Brace, R.L. Rudell, and R.E. Bryant,
//   "Efficient Implementation of a BDD package"
//   https://ieeexplore.ieee.org/document/114826

namespace internal {
using BddIdTy = int32_t;
}
// For efficiency variables and nodes are referred to by indices into vector
// data members in the BDD.
XLS_DEFINE_STRONG_INT_TYPE(BddVariable, internal::BddIdTy);
XLS_DEFINE_STRONG_INT_TYPE(BddNodeIndex, internal::BddIdTy);

// A node in the BDD. The node is associated with a single variable and has
// children corresponding to when the variable is true (high) and when it is
// false (low).
struct BddNode {
  BddNode() : variable(0), high(0), low(0), path_count(0) {}
  BddNode(BddVariable v, BddNodeIndex h, BddNodeIndex l, int32_t p)
      : variable(v), high(h), low(l), path_count(p) {}

  BddVariable variable;
  BddNodeIndex high;
  BddNodeIndex low;

  // Number of paths from this node to the terminal nodes 0 and 1. Used to limit
  // the growth of the BDD by halting evaluation if the number of paths gets too
  // large. Saturates at INT32_MAX.
  int32_t path_count;
};

class BinaryDecisionDiagram {
 public:
  static constexpr int64_t kDefaultMaxPaths =
      std::numeric_limits<int32_t>::max() - 1;

  static constexpr BddNodeIndex kInfeasible = BddNodeIndex(-1);

  // Creates an empty BDD. Initialize the BDD contains only the nodes
  // corresponding to zero and one.
  explicit BinaryDecisionDiagram();
  explicit BinaryDecisionDiagram(int64_t max_paths);

  // Adds a new variable to the BDD and returns the node corresponding the
  // variable's value.
  BddNodeIndex NewVariable();

  // Adds `count` new variables to the BDD and returns the nodes corresponding
  // to the variables' values. This is more efficient than calling NewVariable
  // repeatedly for large `count`. This is efficient for both large and small
  // `count`.
  std::vector<BddNodeIndex> NewVariables(int64_t count);

  // Returns the inverse of the given expression.
  BddNodeIndex Not(BddNodeIndex expr);

  // Returns the OR/AND of the given expressions.
  BddNodeIndex And(BddNodeIndex a, BddNodeIndex b);
  BddNodeIndex Or(BddNodeIndex a, BddNodeIndex b);

  // Returns the expression a -> b (i.e., (!a || b))
  BddNodeIndex Implies(BddNodeIndex a, BddNodeIndex b);

  // Returns whether a implies b.
  // Equivalent to checking if `Implies(a, b) == one()`. However, this will not
  // create any new BDD nodes & has more early-return conditions, so it's
  // usually faster & more efficient.
  bool DoesImply(BddNodeIndex a, BddNodeIndex b);

  // Returns the leaf node corresponding to zero or one.
  BddNodeIndex zero() const { return BddNodeIndex(0); }
  BddNodeIndex one() const { return BddNodeIndex(1); }

  // Evaluates the given expression with the given variable values. The keys in
  // the map are the *node* indices of the respective variable (value returned
  // by NewVariable). An error is returned if the map is missing a variable
  // required to evaluate the expression.
  absl::StatusOr<bool> Evaluate(
      BddNodeIndex expr,
      const absl::flat_hash_map<BddNodeIndex, bool>& variable_values) const;

  // Returns the BDD node with the given index.
  const BddNode& GetNode(BddNodeIndex node_index) const {
    CHECK_GT(nodes_.size(), node_index.value()) << "bad index";
    const auto& e = nodes_.at(node_index.value());
    CHECK(!IsFreeNode(e)) << "Got a free list index at " << node_index;
    return e;
  }

  // Returns the number of nodes in the graph.
  int64_t size() const { return nodes_size_; }
  int64_t capacity() const { return nodes_.capacity(); }
  // Get a (very) rough approximation of how much memory is being used by the
  // BDD. This can be used to decide to perform a GC to try to avoid using too
  // much memory.
  int64_t approximate_memory_use() const {
    return nodes_.capacity() * sizeof(decltype(nodes_)::value_type) +
           variable_base_nodes_.capacity() *
               sizeof(decltype(variable_base_nodes_)::value_type) +
           node_map_.capacity() * sizeof(decltype(node_map_)::value_type) +
           ite_cache_.approximate_memory_use();
  }

  // Returns the number of variables in the graph.
  int64_t variable_count() const { return variable_base_nodes_.size(); }

  // Returns the number of paths in the given expression.
  int64_t path_count(BddNodeIndex expr) const {
    if (expr == kInfeasible) {
      SaturatedResult<int64_t> too_many_paths =
          SaturatingAdd(max_paths_, static_cast<int64_t>(1));
      return too_many_paths.result;
    }
    return GetNode(expr).path_count;
  }

  // Returns the given expression in disjunctive normal form (sum of products).
  // The expression is not minimal. 'minterm_limit' is the maximum number of
  // minterms to emit before truncating the output.
  std::string ToStringDnf(BddNodeIndex expr, int64_t minterm_limit = 0) const;

  // Returns true if the given node is the "base" node for its respective
  // variable. The expression of a base node is exactly equal to the value of
  // the variable.
  bool IsVariableBaseNode(BddNodeIndex expr) const {
    return GetNode(expr).high == one() && GetNode(expr).low == zero();
  }

  // Returns the node corresponding to the given if-then-else expression.
  BddNodeIndex IfThenElse(BddNodeIndex cond, BddNodeIndex if_true,
                          BddNodeIndex if_false);

  // Returns whether the given if-then-else expression is known true or false,
  // without computing any new intermediate BDD nodes.
  std::optional<bool> IfThenElseConstant(BddNodeIndex cond,
                                         BddNodeIndex if_true,
                                         BddNodeIndex if_false);

  // Perform a garbage collection and clear all bdd node information that isn't
  // referenced (transitively or directly) by one of the bdd-nodes in the roots
  // list. After calling this BddNodeIndex values might be reused. A node holds
  // live any variables that point directly to it as well. Note that variable
  // ids are never reused. Unreferenced variables will be marked as freed
  // however and CHECK fail if used. This is done to allow for easier
  // maintenance of internal invariants related to the shape of the diagram as
  // a whole. Even on exceptionally large designs the memory cost of not
  // compating the variables is O(10s of mb) so easily affordable.
  //
  // This is intended to combat the issues with excessive memory usage we have
  // been seeing on some large designs.
  //
  // GC will only be performed if more than gc_threshold portion of the
  // allocated nodes are dead.
  //
  // Returns a list of all node indexs which were freed dead.
  absl::StatusOr<std::vector<BddNodeIndex>> GarbageCollect(
      absl::Span<BddNodeIndex const> roots, double gc_threshold = 0.5);

  int64_t last_gc_node_size() { return prev_nodes_size_; }

 private:
  static constexpr BddVariable kFreeNodeVariable = BddVariable(-2);

  // A special index value, used only for the free list. that indicates this is
  // in a list where all consecutive numbers up to the last element in the free
  // list are free. After hitting this every subsequent value in the free list
  // must be this value.
  static constexpr BddNodeIndex kNextFreeIsConsecutive = BddNodeIndex(-1);

  static bool IsFreeNode(const BddNode& node) {
    return node.variable == kFreeNodeVariable;
  }
  static void SetFreeNode(BddNode& node, BddNodeIndex next) {
    node.variable = BddVariable(kFreeNodeVariable);
    node.high = BddNodeIndex(next);
  }
  static BddNodeIndex GetNextFreeNode(const BddNode& node) {
    CHECK(IsFreeNode(node));
    return node.high;
  }
  static BddNodeIndex NextFree(BddNodeIndex next, BddNodeIndex current_addr) {
    if (next == kNextFreeIsConsecutive) {
      return BddNodeIndex(current_addr.value() + 1);
    }
    return next;
  }

  // If-Then-Else helper for trivial cases. Returns std::nullopt if non-trivial.
  std::optional<BddNodeIndex> IfThenElseTrivial(BddNodeIndex cond,
                                                BddNodeIndex if_true,
                                                BddNodeIndex if_false);

  // Helper for constructing a DNF string representation.
  void ToStringDnfHelper(BddNodeIndex expr, int64_t* minterms_to_emit,
                         std::vector<std::string>* terms,
                         std::string* str) const;

  // Get the node corresponding to the given variable with the given low/high
  // children. Creates it if it does not exist.
  BddNodeIndex GetOrCreateNode(BddVariable var, BddNodeIndex high,
                               BddNodeIndex low);

  // Returns the node corresponding to the value of the given variable.
  BddNodeIndex GetVariableBaseNode(BddVariable variable) const {
    auto res = variable_base_nodes_.at(variable.value());
    CHECK(res) << "Variable has been garbage collected: " << variable.value();
    return *res;
  }

  // Creates the base BddNode corresponding to the given variable.
  BddNodeIndex CreateVariableBaseNode(BddVariable var);

  // Adds the node to the address space and returns the id.
  BddNodeIndex CreateNode(BddNode node);

  struct IfThenElseExpr {
    BddNodeIndex cond;
    BddNodeIndex if_true;
    BddNodeIndex if_false;
  };

  struct ITEDecomposition {
    BddVariable min_var;
    IfThenElseExpr true_cofactor;
    IfThenElseExpr false_cofactor;
  };

  // Performs a Shannon expansion about the minimum variable, where Shannon
  // expansion is the identity:
  //
  //   F(x0, x1, ..) = !x0 && F(0, x1, ...) + x0 && F(1, x1, ...)
  //
  ITEDecomposition DecomposeITE(BddNodeIndex cond, BddNodeIndex if_true,
                                BddNodeIndex if_false);

  // NodeIndexes corresponding to the base nodes (var, one, zero) for each
  // variable. std::nullopt if the variable has been garbage collected.
  //
  // NB To make maintenance of the requirement that variables be increasing
  // easier these are never collected or reused.
  std::vector<std::optional<BddNodeIndex>> variable_base_nodes_;

  // # Explanation of the node free-lists.
  //
  // Since bdd nodes are created quite rapidly and can quickly end up
  // unreachable (due to nodes being removed or other simplifications) we need
  // to be able to GC this BDD to avoid using an excessive amount of memory. To
  // do this we use a mark-sweep GC and reuse node and variable ids using
  // free-lists. Each of the nodes lists is an address space for their
  // corresponding 'Index' pointers. For simplicity the node state is held in a
  // cpp variant. If the variant is a 'FreeListNode' then that index is free to
  // be allocated to a new node and functions as a singlely-linked-list, with
  // each node having the pointer of the next free node held within it. The head
  // of the list is held in the '_head_' field. Furthermore to make extending
  // the address space easier the ending values of the address_space can have
  // free nodes with the fixed 'kNextIsConsecutive' value which indicates that
  // the address space is free from that node to the end of the address space.
  //
  // The address spaces are never compacted so care should be taken to call
  // GarbageCollect often enough that they don't grow too large.
  //
  // The vector of all the nodes in the BDD and the free list of ids.
  std::vector<BddNode> nodes_;
  // The first free offset in the nodes_ list.
  BddNodeIndex free_node_head_;
  internal::BddIdTy nodes_size_;

  // A map from BDD node content (variable id, high child, low child) to the
  // index of the respective node. This map is used to ensure that no duplicate
  // nodes are created.
  using NodeKey = std::tuple<BddVariable, BddNodeIndex, BddNodeIndex>;
  absl::flat_hash_map<NodeKey, BddNodeIndex> node_map_;

  // A cache from if-then-else expression to the node corresponding to that
  // expression. The key elements are (condition, if-true, if-false). This
  // enables fast lookup for expressions.
  class DynamicIteCache {
   public:
    using IteKey = std::tuple<BddNodeIndex, BddNodeIndex, BddNodeIndex>;
    using ExactMap = absl::flat_hash_map<IteKey, BddNodeIndex>;
    struct IteCacheEntry {
      BddNodeIndex cond = kInfeasible;
      BddNodeIndex if_true = kInfeasible;
      BddNodeIndex if_false = kInfeasible;
      BddNodeIndex result = kInfeasible;

      friend bool operator==(const IteCacheEntry& a,
                             const IteCacheEntry& b) = default;
    };
    using LossyArray = std::vector<IteCacheEntry>;

    static constexpr size_t kLossyCacheSize = size_t{1} << 20;
    static constexpr size_t kCutoverThreshold = size_t{1} << 20;

    DynamicIteCache() : cache_(ExactMap()) {}

    std::optional<BddNodeIndex> Get(BddNodeIndex cond, BddNodeIndex if_true,
                                    BddNodeIndex if_false) const;
    void Insert(BddNodeIndex cond, BddNodeIndex if_true, BddNodeIndex if_false,
                BddNodeIndex result);
    void GarbageCollect(const InlineBitmap& live_nodes);
    size_t approximate_memory_use() const;

   private:
    size_t Index(BddNodeIndex cond, BddNodeIndex if_true,
                 BddNodeIndex if_false) const;
    void TransitionToLossy();

    std::variant<ExactMap, LossyArray> cache_;
  };
  DynamicIteCache ite_cache_;

  int64_t max_paths_;
  int64_t prev_nodes_size_ = 2;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_BINARY_DECISION_DIAGRAM_H_
