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
  // TODO(akalan): Eventually, it makes more sense if SymbolicType stores the
  // leaves as Bits type and distinguishes the function parameters from
  // constants via a flag.
  SymbolicType(Leaf expr_tree_leaf, int64_t bit_count, bool is_signed)
      : expr_tree_(expr_tree_leaf),
        bit_count_(bit_count),
        is_signed_(is_signed) {}
  SymbolicType(Nodes expr_tree_nodes, BinopKind root_op, int64_t bit_count,
               bool is_signed)
      : expr_tree_(expr_tree_nodes),
        root_op_(root_op),
        bit_count_(bit_count),
        is_signed_(is_signed) {}

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

  BinopKind op() { return root_op_; }

  bool IsBits();
  absl::StatusOr<Bits> GetBits();

  int64_t GetBitCount() { return bit_count_; }
  bool IsSigned() { return is_signed_; }

  // Prints the symbolic expression tree via inorder traversal
  absl::StatusOr<std::string> ToString();

  // Performs a postorder tree traversal under this node in the expression tree.
  absl::Status DoPostorder(const std::function<absl::Status(SymbolicType*)>& f);

  const absl::variant<Nodes, Leaf>& tree() const { return expr_tree_; }

 private:
  absl::variant<Nodes, Leaf> expr_tree_;
  BinopKind root_op_;
  // Concrete information used for Z3 translation.
  int64_t bit_count_;
  bool is_signed_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_SYMBOLIC_TYPE_H_
