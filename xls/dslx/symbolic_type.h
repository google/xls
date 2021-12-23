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

namespace xls::dslx {

// Tags a node in the expression tree to denote its type.
//
// Note that kArray represents arrays, tuples, and structs at the InterpValue
// level.
//
// Only function parameters that are casted or bit-sliced need a different tag,
// since we need to differentiate the bit width/sign when translating the node
// to Z3 vs when generating the DSLX test. Other literals after cast or slicing
// stay as kNumber, since we don't need to retain the original bit width/sign.
enum class SymbolicNodeTag {
  kInternalOp,
  kInternalTernary,
  kFnParam,
  kCastedParam,
  kSlicedParam,
  kNumber,
  kArray
};

// Represents the concrete info of the nodes in the expression tree. Z3 module
// later uses this information to translate the nodes.
//
// Note that internal nodes are also annotated as signed/unsigned which is
// needed for translating operations such as "<".
//
// "id" is used for Z3 translation of function parameters.
// "slice_idx" stores the index that the variable was sliced from.
// "cast_width" stores the new bit width for casted or sliced parameter.
struct ConcreteInfo {
  bool is_signed;
  int64_t bit_count;
  int64_t bit_value = 0;
  std::string id;
  int64_t slice_idx = 0;
  int64_t cast_width = 0;
};

// Holds a binary expression tree for a symbolic variable.
//
// Leaves are either program inputs stored symbolically as string or "Bits"
// corresponding to the literal numbers. Interior nodes are a pair of
// expression trees representing binary/unary operations or "ternary if"s.
class SymbolicType {
 public:
  using OpKind = absl::variant<BinopKind, UnopKind>;
  using Root = absl::variant<OpKind, SymbolicType*>;

  // Represents an internal node.
  //
  // Binary and unary operations store the operands as children and the operator
  // as root (unary operations store the single operand as the left child).
  //
  // For the ternary ifs, left and right children are symbolic nodes
  // representing consequent and alternate expressions in the ternary statement
  // respectively and the root is the symbolic node representing the if
  // constraint.
  struct Nodes {
    Root root;
    SymbolicType* left;
    SymbolicType* right;
  };

  SymbolicType() {}

  static SymbolicType MakeUnary(Nodes expr_tree, ConcreteInfo concrete_info);
  static SymbolicType MakeBinary(Nodes expr_tree, ConcreteInfo concrete_info);
  static SymbolicType MakeTernary(Nodes expr_tree, ConcreteInfo concrete_info);

  static SymbolicType MakeLiteral(ConcreteInfo concrete_info);
  static SymbolicType MakeParam(ConcreteInfo concrete_info);
  static SymbolicType MakeCastedParam(ConcreteInfo concrete_info);
  static SymbolicType MakeSlicedParam(ConcreteInfo concrete_info);

  static SymbolicType MakeArray(std::vector<SymbolicType*> children);

  ~SymbolicType() = default;

  absl::StatusOr<OpKind> op();
  absl::StatusOr<std::string> OpToString();

  // Queries
  bool IsBits() { return tag_ == SymbolicNodeTag::kNumber; }
  bool IsTernary() { return tag_ == SymbolicNodeTag::kInternalTernary; }
  bool isInternalOp() { return tag_ == SymbolicNodeTag::kInternalOp; }
  bool IsSigned() { return concrete_info_.is_signed; }
  bool IsArray() { return tag_ == SymbolicNodeTag::kArray; }
  bool IsParam() {
    return tag_ == SymbolicNodeTag::kFnParam ||
           tag_ == SymbolicNodeTag::kCastedParam ||
           tag_ == SymbolicNodeTag::kSlicedParam;
  }
  bool IsCasted() { return tag_ == SymbolicNodeTag::kCastedParam; }
  bool IsSliced() { return tag_ == SymbolicNodeTag::kSlicedParam; }
  bool IsLeaf() {
    return tag_ != SymbolicNodeTag::kInternalOp &&
           tag_ != SymbolicNodeTag::kInternalTernary;
  }
  SymbolicNodeTag tag() { return tag_; }
  int64_t bit_count() { return concrete_info_.bit_count; }
  int64_t cast_bit_count() { return concrete_info_.cast_width; }
  int64_t slice_index() { return concrete_info_.slice_idx; }
  int64_t bit_value() { return concrete_info_.bit_value; }
  std::string id() { return concrete_info_.id; }
  std::vector<SymbolicType*> GetChildren() { return children_; }

  void MarkAsFnParam(std::string id);

  absl::StatusOr<SymbolicType*> TernaryRoot();

  // Prints the symbolic expression tree via inorder traversal
  absl::StatusOr<std::string> ToString();

  // Performs a postorder tree traversal under this node in the expression tree.
  absl::Status DoPostorder(const std::function<absl::Status(SymbolicType*)>& f);

  static SymbolicType CreateLogicalOp(SymbolicType* lhs, SymbolicType* rhs,
                                      BinopKind op);
  absl::StatusOr<Nodes> tree() const;

 private:
  // Constructor for internal nodes i.e. binary/unary operations or "ternary
  // if"s.
  SymbolicType(Nodes expr_tree, ConcreteInfo concrete_info, SymbolicNodeTag tag)
      : expr_tree_(expr_tree), concrete_info_(concrete_info), tag_(tag) {}

  // Constructor for leaves i.e. function parameters or literal values.
  SymbolicType(ConcreteInfo concrete_info, SymbolicNodeTag tag)
      : concrete_info_(concrete_info), tag_(tag) {}

  // Constructor for array/tuple/struct sort.
  SymbolicType(std::vector<SymbolicType*> children, SymbolicNodeTag tag)
      : tag_(tag), children_(children) {}

  absl::StatusOr<Root> root();

  // The subtree this represents (either binary/unary operations or ternary if)
  Nodes expr_tree_;
  ConcreteInfo concrete_info_;
  SymbolicNodeTag tag_;

  std::vector<SymbolicType*> children_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_SYMBOLIC_TYPE_H_
