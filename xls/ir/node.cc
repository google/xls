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

#include "xls/ir/node.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {

Node::Node(Op op, Type* type, absl::optional<SourceLocation> loc,
           Function* function)
    : function_(function),
      id_(function_->package()->GetNextNodeId()),
      op_(op),
      type_(type),
      loc_(loc) {}

void Node::AddOperand(Node* operand) {
  XLS_VLOG(3) << " Adding operand " << operand->GetName() << " as #"
              << operands_.size() << " operand of " << GetName();
  operands_.push_back(operand);
  operand->AddUser(this);
  XLS_VLOG(3) << " " << operand->GetName()
              << " user now: " << operand->GetUsersString();
}

void Node::AddOperands(absl::Span<Node* const> operands) {
  for (Node* operand : operands) {
    AddOperand(operand);
  }
}

void Node::AddOptionalOperand(absl::optional<Node*> operand) {
  if (operand.has_value()) {
    AddOperand(*operand);
  }
}

absl::Status Node::AddNodeToFunctionAndReplace(
    std::unique_ptr<Node> replacement) {
  Node* replacement_ptr = function()->AddNode(std::move(replacement));
  XLS_RETURN_IF_ERROR(Verify(replacement_ptr));
  return ReplaceUsesWith(replacement_ptr).status();
}

void Node::AddUser(Node* user) {
  auto insert_result = users_set_.insert(user);
  if (insert_result.second) {
    users_.push_back(user);
    // Keep the users sequence sorted by ordinal for stability.
    absl::c_sort(users_, [](Node* a, Node* b) { return a->id() < b->id(); });
  }
}

void Node::RemoveUser(Node* user) {
  users_set_.erase(user);
  users_.erase(std::remove(users_.begin(), users_.end(), user), users_.end());
}

absl::Status Node::VisitSingleNode(DfsVisitor* visitor) {
  switch (op()) {
    case Op::kAdd:
      XLS_RETURN_IF_ERROR(visitor->HandleAdd(down_cast<BinOp*>(this)));
      break;
    case Op::kAnd:
      XLS_RETURN_IF_ERROR(visitor->HandleNaryAnd(down_cast<NaryOp*>(this)));
      break;
    case Op::kAndReduce:
      XLS_RETURN_IF_ERROR(
          visitor->HandleAndReduce(down_cast<BitwiseReductionOp*>(this)));
      break;
    case Op::kNand:
      XLS_RETURN_IF_ERROR(visitor->HandleNaryNand(down_cast<NaryOp*>(this)));
      break;
    case Op::kNor:
      XLS_RETURN_IF_ERROR(visitor->HandleNaryNor(down_cast<NaryOp*>(this)));
      break;
    case Op::kArray:
      XLS_RETURN_IF_ERROR(visitor->HandleArray(down_cast<Array*>(this)));
      break;
    case Op::kBitSlice:
      XLS_RETURN_IF_ERROR(visitor->HandleBitSlice(down_cast<BitSlice*>(this)));
      break;
    case Op::kConcat:
      XLS_RETURN_IF_ERROR(visitor->HandleConcat(down_cast<Concat*>(this)));
      break;
    case Op::kDecode:
      XLS_RETURN_IF_ERROR(visitor->HandleDecode(down_cast<Decode*>(this)));
      break;
    case Op::kEncode:
      XLS_RETURN_IF_ERROR(visitor->HandleEncode(down_cast<Encode*>(this)));
      break;
    case Op::kEq:
      XLS_RETURN_IF_ERROR(visitor->HandleEq(down_cast<CompareOp*>(this)));
      break;
    case Op::kIdentity:
      XLS_RETURN_IF_ERROR(visitor->HandleIdentity(down_cast<UnOp*>(this)));
      break;
    case Op::kArrayIndex:
      XLS_RETURN_IF_ERROR(
          visitor->HandleArrayIndex(down_cast<ArrayIndex*>(this)));
      break;
    case Op::kArrayUpdate:
      XLS_RETURN_IF_ERROR(
          visitor->HandleArrayUpdate(down_cast<ArrayUpdate*>(this)));
      break;
    case Op::kInvoke:
      XLS_RETURN_IF_ERROR(visitor->HandleInvoke(down_cast<Invoke*>(this)));
      break;
    case Op::kCountedFor:
      XLS_RETURN_IF_ERROR(
          visitor->HandleCountedFor(down_cast<CountedFor*>(this)));
      break;
    case Op::kLiteral:
      XLS_RETURN_IF_ERROR(visitor->HandleLiteral(down_cast<Literal*>(this)));
      break;
    case Op::kMap:
      XLS_RETURN_IF_ERROR(visitor->HandleMap(down_cast<Map*>(this)));
      break;
    case Op::kNe:
      XLS_RETURN_IF_ERROR(visitor->HandleNe(down_cast<CompareOp*>(this)));
      break;
    case Op::kNeg:
      XLS_RETURN_IF_ERROR(visitor->HandleNeg(down_cast<UnOp*>(this)));
      break;
    case Op::kNot:
      XLS_RETURN_IF_ERROR(visitor->HandleNot(down_cast<UnOp*>(this)));
      break;
    case Op::kOneHot:
      XLS_RETURN_IF_ERROR(visitor->HandleOneHot(down_cast<OneHot*>(this)));
      break;
    case Op::kOneHotSel:
      XLS_RETURN_IF_ERROR(
          visitor->HandleOneHotSel(down_cast<OneHotSelect*>(this)));
      break;
    case Op::kOr:
      XLS_RETURN_IF_ERROR(visitor->HandleNaryOr(down_cast<NaryOp*>(this)));
      break;
    case Op::kOrReduce:
      XLS_RETURN_IF_ERROR(
          visitor->HandleOrReduce(down_cast<BitwiseReductionOp*>(this)));
      break;
    case Op::kParam:
      XLS_RETURN_IF_ERROR(visitor->HandleParam(down_cast<Param*>(this)));
      break;
    case Op::kReverse:
      XLS_RETURN_IF_ERROR(visitor->HandleReverse(down_cast<UnOp*>(this)));
      break;
    case Op::kSDiv:
      XLS_RETURN_IF_ERROR(visitor->HandleSDiv(down_cast<BinOp*>(this)));
      break;
    case Op::kSel:
      XLS_RETURN_IF_ERROR(visitor->HandleSel(down_cast<Select*>(this)));
      break;
    case Op::kSGt:
      XLS_RETURN_IF_ERROR(visitor->HandleSGt(down_cast<CompareOp*>(this)));
      break;
    case Op::kSGe:
      XLS_RETURN_IF_ERROR(visitor->HandleSGe(down_cast<CompareOp*>(this)));
      break;
    case Op::kShll:
      XLS_RETURN_IF_ERROR(visitor->HandleShll(down_cast<BinOp*>(this)));
      break;
    case Op::kShra:
      XLS_RETURN_IF_ERROR(visitor->HandleShra(down_cast<BinOp*>(this)));
      break;
    case Op::kShrl:
      XLS_RETURN_IF_ERROR(visitor->HandleShrl(down_cast<BinOp*>(this)));
      break;
    case Op::kSLe:
      XLS_RETURN_IF_ERROR(visitor->HandleSLe(down_cast<CompareOp*>(this)));
      break;
    case Op::kSLt:
      XLS_RETURN_IF_ERROR(visitor->HandleSLt(down_cast<CompareOp*>(this)));
      break;
    case Op::kSMul:
      XLS_RETURN_IF_ERROR(visitor->HandleSMul(down_cast<ArithOp*>(this)));
      break;
    case Op::kSub:
      XLS_RETURN_IF_ERROR(visitor->HandleSub(down_cast<BinOp*>(this)));
      break;
    case Op::kTupleIndex:
      XLS_RETURN_IF_ERROR(
          visitor->HandleTupleIndex(down_cast<TupleIndex*>(this)));
      break;
    case Op::kTuple:
      XLS_RETURN_IF_ERROR(visitor->HandleTuple(down_cast<Tuple*>(this)));
      break;
    case Op::kUDiv:
      XLS_RETURN_IF_ERROR(visitor->HandleUDiv(down_cast<BinOp*>(this)));
      break;
    case Op::kUGe:
      XLS_RETURN_IF_ERROR(visitor->HandleUGe(down_cast<CompareOp*>(this)));
      break;
    case Op::kUGt:
      XLS_RETURN_IF_ERROR(visitor->HandleUGt(down_cast<CompareOp*>(this)));
      break;
    case Op::kULe:
      XLS_RETURN_IF_ERROR(visitor->HandleULe(down_cast<CompareOp*>(this)));
      break;
    case Op::kULt:
      XLS_RETURN_IF_ERROR(visitor->HandleULt(down_cast<CompareOp*>(this)));
      break;
    case Op::kUMul:
      XLS_RETURN_IF_ERROR(visitor->HandleUMul(down_cast<ArithOp*>(this)));
      break;
    case Op::kXor:
      XLS_RETURN_IF_ERROR(visitor->HandleNaryXor(down_cast<NaryOp*>(this)));
      break;
    case Op::kXorReduce:
      XLS_RETURN_IF_ERROR(
          visitor->HandleXorReduce(down_cast<BitwiseReductionOp*>(this)));
      break;
    case Op::kSignExt:
      XLS_RETURN_IF_ERROR(
          visitor->HandleSignExtend(down_cast<ExtendOp*>(this)));
      break;
    case Op::kZeroExt:
      XLS_RETURN_IF_ERROR(
          visitor->HandleZeroExtend(down_cast<ExtendOp*>(this)));
      break;
  }
  return absl::OkStatus();
}

absl::Status Node::Accept(DfsVisitor* visitor) {
  if (visitor->IsVisited(this)) {
    return absl::OkStatus();
  }
  if (visitor->IsTraversing(this)) {
    return absl::InternalError(
        absl::StrFormat("Cycle detected which includes node %s", GetName()));
  }
  visitor->SetTraversing(this);
  for (Node* operand : operands()) {
    XLS_RETURN_IF_ERROR(operand->Accept(visitor));
  }
  visitor->UnsetTraversing(this);
  visitor->MarkVisited(this);
  return VisitSingleNode(visitor);
}

bool Node::IsDefinitelyEqualTo(const Node* other) const {
  if (op() != other->op()) {
    return false;
  }
  auto same_type = [&](const Node* a, const Node* b) {
    return a->GetType()->IsEqualTo(b->GetType());
  };

  // Must have the same operand count, and each operand must be the same type.
  if (operand_count() != other->operand_count()) {
    return false;
  }
  for (int64 i = 0; i < operand_count(); ++i) {
    if (!same_type(operand(i), other->operand(i))) {
      return false;
    }
  }
  return same_type(this, other);
}

std::string Node::GetName() const {
  if (Is<Param>()) {
    return As<Param>()->name();
  }
  return absl::StrFormat("%s.%d", OpToString(op_), id_);
}

std::string Node::ToStringInternal(bool include_operand_types) const {
  auto node_to_name = [](const Node* n, bool include_type) -> std::string {
    std::string name = n->GetName();
    if (include_type) {
      absl::StrAppend(&name, ": ", n->GetType()->ToString());
    }
    return name;
  };

  std::string ret = node_to_name(this, false);
  Type* t = GetType();
  absl::StrAppend(&ret, ": ", t->ToString());
  absl::StrAppend(&ret, " = ", OpToString(op_));
  std::vector<std::string> args;
  for (Node* operand : operands()) {
    args.push_back(node_to_name(operand, include_operand_types));
  }
  switch (op_) {
    case Op::kParam:
      args.push_back(GetName());
      break;
    case Op::kLiteral:
      args.push_back(
          absl::StrFormat("value=%s", As<Literal>()->value().ToHumanString()));
      break;
    case Op::kCountedFor:
      for (int64 i = 0; i < As<CountedFor>()->invariant_args().size(); ++i) {
        args.pop_back();
      }
      args.push_back(
          absl::StrFormat("trip_count=%d", As<CountedFor>()->trip_count()));
      args.push_back(absl::StrFormat("stride=%d", As<CountedFor>()->stride()));
      args.push_back(
          absl::StrFormat("body=%s", As<CountedFor>()->body()->name()));
      if (!As<CountedFor>()->invariant_args().empty()) {
        args.push_back(absl::StrFormat(
            "invariant_args=[%s]",
            absl::StrJoin(As<CountedFor>()->invariant_args(), ", ",
                          [](std::string* out, const Node* node) {
                            absl::StrAppend(out, node->GetName());
                          })));
      }
      break;
    case Op::kMap:
      args.push_back(
          absl::StrFormat("to_apply=%s", As<Map>()->to_apply()->name()));
      break;
    case Op::kInvoke:
      args.push_back(
          absl::StrFormat("to_apply=%s", As<Invoke>()->to_apply()->name()));
      break;
    case Op::kTupleIndex:
      args.push_back(absl::StrFormat("index=%d", As<TupleIndex>()->index()));
      break;
    case Op::kOneHot:
      args.push_back(absl::StrFormat(
          "lsb_prio=%s",
          As<OneHot>()->priority() == LsbOrMsb::kLsb ? "true" : "false"));
      break;
    case Op::kOneHotSel: {
      const OneHotSelect* sel = As<OneHotSelect>();
      args = {operand(0)->GetName()};
      args.push_back(absl::StrFormat(
          "cases=[%s]", absl::StrJoin(sel->cases(), ", ",
                                      [](std::string* out, const Node* node) {
                                        absl::StrAppend(out, node->GetName());
                                      })));
      break;
    }
    case Op::kSel: {
      const Select* sel = As<Select>();
      args = {operand(0)->GetName()};
      args.push_back(absl::StrFormat(
          "cases=[%s]", absl::StrJoin(sel->cases(), ", ",
                                      [](std::string* out, const Node* node) {
                                        absl::StrAppend(out, node->GetName());
                                      })));
      if (sel->default_value().has_value()) {
        args.push_back(
            absl::StrFormat("default=%s", (*sel->default_value())->GetName()));
      }
      break;
    }
    case Op::kSignExt:
    case Op::kZeroExt:
      args.push_back(
          absl::StrFormat("new_bit_count=%d", As<ExtendOp>()->new_bit_count()));
      break;
    case Op::kBitSlice:
      args.push_back(absl::StrFormat("start=%d", As<BitSlice>()->start()));
      args.push_back(absl::StrFormat("width=%d", As<BitSlice>()->width()));
      break;
    case Op::kDecode:
      args.push_back(absl::StrFormat("width=%d", As<Decode>()->width()));
      break;
    default:
      break;
  }
  if (loc()) {
    // Tack the location on as an optional keyword attribute.
    args.push_back(absl::StrFormat("pos=%d,%d,%d", loc()->fileno().value(),
                                   loc()->lineno().value(),
                                   loc()->colno().value()));
  }
  absl::StrAppendFormat(&ret, "(%s)", absl::StrJoin(args, ", "));
  return ret;
}

std::string Node::GetOperandsString() const {
  return absl::StrFormat(
      "[%s]",
      absl::StrJoin(operands_, ", ", [](std::string* out, const Node* n) {
        absl::StrAppend(out, n->GetName());
      }));
}

std::string Node::GetUsersString() const {
  return absl::StrFormat(
      "[%s]", absl::StrJoin(users_, ", ", [](std::string* out, const Node* n) {
        absl::StrAppend(out, n->GetName());
      }));
}

bool Node::HasUser(const Node* target) const {
  return users_set_.find(const_cast<Node*>(target)) != users_set_.end();
}

bool Node::HasOperand(const Node* target) const {
  for (const Node* operand : operands_) {
    if (operand == target) {
      return true;
    }
  }
  return false;
}

int64 Node::OperandInstanceCount(const Node* target) const {
  int64 count = 0;
  for (const Node* operand : operands_) {
    if (operand == target) {
      ++count;
    }
  }
  return count;
}

bool Node::ReplaceOperand(Node* old_operand, Node* new_operand) {
  // The following test is necessary, because of the following scenario
  // during IR manipulation. Assume we want to replace a node 'sub' with
  // another node 'neg' that has as an operand the node sub. With function
  // builder, this would be:
  //    Node *neg = f->AddNode(absl::make_unique<UnOp>(
  //      Op::kNeg, n->loc(), n, n->package()));
  //    f->ReplaceSingleNode(n, neg);
  // At the time of ReplaceSingleNode, neg is a user of n, and so replacing it
  // would create a cycle, with 'neg' getting a user 'neg'.
  if (this == new_operand) {
    return true;
  }
  bool did_replace = false;
  for (int64 i = 0; i < operand_count(); ++i) {
    if (operands_[i] == old_operand) {
      if (!did_replace && new_operand != nullptr) {
        // Now we know we're definitely using this new operand.
        new_operand->AddUser(this);
      }
      did_replace = true;
      operands_[i] = new_operand;
    }
  }
  old_operand->RemoveUser(this);
  return did_replace;
}

absl::Status Node::ReplaceOperandNumber(int64 operand_no, Node* new_operand) {
  Node* old_operand = operands_[operand_no];
  XLS_RET_CHECK(old_operand->GetType() == new_operand->GetType())
      << "old operand type: " << old_operand->GetType()->ToString()
      << " new operand type: " << new_operand->GetType()->ToString();

  // AddUser is idempotent so even if the new operand is already used by this
  // node in another operand slot, it is safe to call.
  new_operand->AddUser(this);
  operands_[operand_no] = new_operand;

  for (Node* operand : operands()) {
    if (operand == old_operand) {
      return absl::OkStatus();
    }
  }
  // old_operand is no longer an operand of this node.
  old_operand->RemoveUser(this);
  return absl::OkStatus();
}

xabsl::StatusOr<bool> Node::ReplaceUsesWith(Node* replacement) {
  XLS_RET_CHECK(GetType() == replacement->GetType())
      << "type was: " << GetType()->ToString()
      << " replacement: " << replacement->GetType()->ToString();
  bool changed = false;
  std::vector<Node*> orig_users(users().begin(), users().end());
  for (Node* user : orig_users) {
    XLS_RET_CHECK(user->ReplaceOperand(this, replacement));
    changed = true;
  }
  if (this == function()->return_value()) {
    function()->set_return_value(replacement);
    changed = true;
  }
  return changed;
}

bool Node::OpIn(const std::vector<Op>& choices) {
  for (auto& c : choices) {
    if (c == op()) {
      return true;
    }
  }
  return false;
}

Package* Node::package() const { return function()->package(); }

}  // namespace xls
