// Copyright 2021 The XLS Authors
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

#ifndef XLS_DSLX_SYMBOLIC_TYPE_H_
#define XLS_DSLX_SYMBOLIC_TYPE_H_

#include <string>
#include <variant>

#include "absl/types/variant.h"
#include "xls/dslx/ast.h"
#include "xls/ir/bits.h"

namespace xls::dslx {

// Holds a binary expression tree for a symbolic variable.
//
// Leaves are either program inputs stored symbolically as string or "Bits"
// corresponding to the literal numbers. Interior nodes are a pair of
// expression trees with the binary operator stored in the root.
//
// Attributes:
//   expr_tree: The subtree this represents (either a pair of subtrees or a
//   leaf).
//   root_op: binary operator for this subtree.
class SymbolicType {
 public:
  struct Nodes {
    SymbolicType* left;
    SymbolicType* right;
  };

  using Leaf = absl::variant<Param*, Bits>;

  SymbolicType() {}
  SymbolicType(Leaf expr_tree_leaf) : expr_tree_(expr_tree_leaf) {}
  SymbolicType(Nodes expr_tree_nodes, BinopKind root_op)
      : expr_tree_(expr_tree_nodes), root_op_(root_op) {}

  ~SymbolicType() = default;

  bool IsLeaf() const { return absl::holds_alternative<Leaf>(expr_tree_); }
  absl::StatusOr<Leaf> leaf() const {
    if (!absl::holds_alternative<Leaf>(expr_tree_))
      return absl::NotFoundError("Expression tree does not have a leaf.");
    return absl::get<Leaf>(expr_tree_);
  }
  absl::StatusOr<std::string> LeafToString();
  absl::StatusOr<uint64_t> LeafToUint64();

  absl::StatusOr<Nodes> nodes();

  bool IsBits();

  // Prints the symbolic expression tree via inorder traversal
  absl::StatusOr<std::string> ToString();

  // Performs an inorder tree traversal under this node in the expression tree.
  // Inorder traversal generates the original program sequence
  absl::Status DoInorder(const std::function<absl::Status(SymbolicType*)>& f);

  const absl::variant<Nodes, Leaf>& tree() const { return expr_tree_; }

 private:
  absl::variant<Nodes, Leaf> expr_tree_;
  BinopKind root_op_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_SYMBOLIC_TYPE_H_
