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

#include "xls/dslx/symbolic_type.h"

#include "absl/strings/str_cat.h"

namespace xls::dslx {

inline absl::StatusOr<std::string> SymbolicType::LeafToString() {
  XLS_ASSIGN_OR_RETURN(Leaf leaf, leaf());
  return absl::get<Param*>(leaf)->identifier();
}

inline absl::StatusOr<uint64_t> SymbolicType::LeafToUint64() {
  XLS_ASSIGN_OR_RETURN(Leaf leaf, leaf());
  return absl::get<Bits>(leaf).ToUint64();
}

absl::StatusOr<SymbolicType::Nodes> SymbolicType::nodes() {
  Nodes* nodes = absl::get_if<Nodes>(&expr_tree_);
  if (nodes == nullptr)
    return absl::NotFoundError("Expression tree does not hold any nodes.");

  return absl::get<Nodes>(expr_tree_);
}

bool SymbolicType::IsBits() {
  return absl::holds_alternative<Leaf>(expr_tree_) &&
         absl::holds_alternative<Bits>(leaf().value());
}

absl::StatusOr<std::string> SymbolicType::ToString() {
  if (IsLeaf()) {
    if (IsBits()) {
      XLS_ASSIGN_OR_RETURN(uint64_t bits_value, LeafToUint64());
      return std::to_string(bits_value);
    }
    return LeafToString();
  } else {
    XLS_ASSIGN_OR_RETURN(Nodes tree_nodes, nodes());
    XLS_ASSIGN_OR_RETURN(std::string node_left, tree_nodes.left->ToString());
    XLS_ASSIGN_OR_RETURN(std::string node_right, tree_nodes.right->ToString());

    std::string guts = absl::StrCat(node_left, ", ", node_right);
    return absl::StrCat(absl::StrFormat("(%s)", guts),
                        BinopKindFormat(this->root_op_));
  }
}

absl::Status SymbolicType::DoInorder(
    const std::function<absl::Status(SymbolicType*)>& f) {
  if (IsLeaf()) {
    XLS_RETURN_IF_ERROR(f(this));
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(Nodes tree_nodes, nodes());

  XLS_RETURN_IF_ERROR(tree_nodes.left->DoInorder(f));
  XLS_RETURN_IF_ERROR(f(this));
  XLS_RETURN_IF_ERROR(tree_nodes.right->DoInorder(f));

  return absl::OkStatus();
}

}  // namespace xls::dslx
