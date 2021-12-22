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
#include "xls/common/status/ret_check.h"

namespace xls::dslx {

/*static*/ SymbolicType SymbolicType::MakeUnary(Nodes expr_tree,
                                                ConcreteInfo concrete_info) {
  return SymbolicType(expr_tree, concrete_info, SymbolicNodeTag::kInternalOp);
}
/*static*/ SymbolicType SymbolicType::MakeBinary(Nodes expr_tree,
                                                 ConcreteInfo concrete_info) {
  return SymbolicType(expr_tree, concrete_info, SymbolicNodeTag::kInternalOp);
}
/*static*/ SymbolicType SymbolicType::MakeTernary(Nodes expr_tree,
                                                  ConcreteInfo concrete_info) {
  return SymbolicType(expr_tree, concrete_info,
                      SymbolicNodeTag::kInternalTernary);
}

/*static*/ SymbolicType SymbolicType::MakeLiteral(ConcreteInfo concrete_info) {
  return SymbolicType(concrete_info, SymbolicNodeTag::kNumber);
}

/*static*/ SymbolicType SymbolicType::MakeParam(ConcreteInfo concrete_info) {
  return SymbolicType(concrete_info, SymbolicNodeTag::kFnParam);
}

/*static*/ SymbolicType SymbolicType::MakeArray(
    std::vector<SymbolicType*> children) {
  return SymbolicType(children, SymbolicNodeTag::kArray);
}

absl::StatusOr<SymbolicType::OpKind> SymbolicType::op() {
  XLS_RET_CHECK(tag_ == SymbolicNodeTag::kInternalOp);
  XLS_ASSIGN_OR_RETURN(Root root, root());
  return absl::get<SymbolicType::OpKind>(root);
}

absl::StatusOr<std::string> SymbolicType::OpToString() {
  XLS_ASSIGN_OR_RETURN(OpKind op, op());
  if (absl::holds_alternative<BinopKind>(op)) {
    return BinopKindFormat(absl::get<BinopKind>(op));
  }
  return UnopKindToString(absl::get<UnopKind>(op));
}
absl::StatusOr<SymbolicType*> SymbolicType::TernaryRoot() {
  XLS_RET_CHECK(tag_ == SymbolicNodeTag::kInternalTernary);
  XLS_ASSIGN_OR_RETURN(Root root, root());
  return absl::get<SymbolicType*>(root);
}

void SymbolicType::MarkAsFnParam(std::string id) {
  tag_ = SymbolicNodeTag::kFnParam;
  concrete_info_.id = id;
}

SymbolicType SymbolicType::CreateLogicalOp(SymbolicType* lhs, SymbolicType* rhs,
                                          BinopKind op) {
  return SymbolicType(SymbolicType::Nodes{op, lhs, rhs},
                      ConcreteInfo{/*is_signed=*/false, /*bit_count=*/1},
                      SymbolicNodeTag::kInternalOp);
}

absl::StatusOr<SymbolicType::Nodes> SymbolicType::tree() const {
  XLS_RET_CHECK(tag_ == SymbolicNodeTag::kInternalOp ||
                tag_ == SymbolicNodeTag::kInternalTernary);
  return expr_tree_;
}

absl::StatusOr<SymbolicType::Root> SymbolicType::root() {
  XLS_RET_CHECK(tag_ == SymbolicNodeTag::kInternalOp ||
                tag_ == SymbolicNodeTag::kInternalTernary);
  return expr_tree_.root;
}

absl::StatusOr<std::string> SymbolicType::ToString() {
  switch (tag_) {
    case SymbolicNodeTag::kFnParam:
    case SymbolicNodeTag::kArray:
      return concrete_info_.id;
    case SymbolicNodeTag::kNumber:
      return std::to_string(concrete_info_.bit_value);
    case SymbolicNodeTag::kInternalOp:
    case SymbolicNodeTag::kInternalTernary: {
      XLS_ASSIGN_OR_RETURN(std::string node_left,
                           this->expr_tree_.left->ToString());
      // Unary operations have null pointer as their right child.
      std::string node_right = "";
      if (expr_tree_.right != nullptr) {
        XLS_ASSIGN_OR_RETURN(node_right, this->expr_tree_.right->ToString());
      }
      std::string guts = absl::StrCat(node_left, ", ", node_right);
      if (IsTernary()) {
        XLS_ASSIGN_OR_RETURN(SymbolicType * root, TernaryRoot());
        XLS_ASSIGN_OR_RETURN(std::string root_string, root->ToString());
        return absl::StrCat(absl::StrFormat("(%s)", guts), "if ", root_string);
      }
      XLS_ASSIGN_OR_RETURN(std::string op_string, OpToString());
      return absl::StrCat(absl::StrFormat("(%s)", guts), op_string);
    }
  }
}

absl::Status SymbolicType::DoPostorder(
    const std::function<absl::Status(SymbolicType*)>& f) {
  if (IsLeaf()) {
    XLS_RETURN_IF_ERROR(f(this));
    return absl::OkStatus();
  }
  XLS_RETURN_IF_ERROR(expr_tree_.left->DoPostorder(f));
  if (expr_tree_.right == nullptr) {
    XLS_RETURN_IF_ERROR(f(nullptr));
  } else {
    XLS_RETURN_IF_ERROR(expr_tree_.right->DoPostorder(f));
  }
  XLS_RETURN_IF_ERROR(f(this));
  return absl::OkStatus();
}

}  // namespace xls::dslx
