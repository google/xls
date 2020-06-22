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

#ifndef XLS_IR_NODE_H_
#define XLS_IR_NODE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class Package;
class Node;
class Function;

// Forward decaration to avoid circular dependency.
class DfsVisitor;

// Abstract type for a node (representing an expression) in the high level IR.
//
// Node is subtyped and can be checked-converted via the As* methods below.
class Node {
 public:
  virtual ~Node() = default;

  // Accepts the visitor, instructing it to visit this node.
  //
  // The visitor is instructed to visit this node with:
  //
  // * This node with kPre
  // * Each operand of this node with kIn and the operand number
  //   * After calling kIn with an operand number returns, that operand is
  //     visited.
  // * This node with kPost
  absl::Status Accept(DfsVisitor* visitor);

  // Visits this node with the given visitor. Visits only this node and does not
  // traverse the graph.
  absl::Status VisitSingleNode(DfsVisitor* visitor);

  Type* GetType() const { return type_; }

  int64 BitCountOrDie() const { return GetType()->AsBitsOrDie()->bit_count(); }

  Op op() const { return op_; }
  Function* function() const { return function_; }
  Package* package() const;
  const absl::optional<SourceLocation> loc() const { return loc_; }

  // Returns the sequence of operands used by this node.
  //
  // Note: we make the view of the operands immutable in order to ensure that we
  // preserve invariants across mutations (so you can't forget to update
  // operand/user symmetry); e.g. see helpers ReplaceOperands() and
  // SwapOperands() below.
  absl::Span<Node* const> operands() const { return operands_; }

  // Returns whether at least one operand was replaced.
  bool ReplaceOperand(Node* old_operand, Node* new_operand);

  // Replaces the existing operand at position 'operand_no' with 'new_operand'.
  absl::Status ReplaceOperandNumber(int64 operand_no, Node* new_operand);

  // Replace all uses of this node with 'replacement'. If this node is the
  // return value of the function, then 'replacement' is made the return
  // value. This node is not deleted and remains in the graph. Returns true if
  // the graph was changed, equivalently, whether 'this' has any users or is the
  // return value of the function.
  xabsl::StatusOr<bool> ReplaceUsesWith(Node* replacement);

  // Constructs a new node and replaces all uses of 'this' with the newly
  // constructed node. NodeT is the node subclass (e.g., 'Param') and the
  // variadic args are the constructor arguments with the exception of the first
  // loc argument and the final Function* argument which are inherited from this
  // node. Returns a pointer to the newly constructed node.
  template <typename NodeT, typename... Args>
  xabsl::StatusOr<NodeT*> ReplaceUsesWithNew(Args&&... args) {
    std::unique_ptr<NodeT> new_node = absl::make_unique<NodeT>(
        loc(), std::forward<Args>(args)..., function());
    NodeT* ptr = new_node.get();
    XLS_RETURN_IF_ERROR(AddNodeToFunctionAndReplace(std::move(new_node)));
    return ptr;
  }

  // Swaps the operands at indices 'a' and 'b' in the operands sequence.
  void SwapOperands(int64 a, int64 b) {
    // Operand/user chains already set up properly.
    std::swap(operands_[a], operands_[b]);
  }

  // Returns true if analysis indicates that this node always produces the
  // same value as 'other' when run with the same operands. The analysis is
  // conservative and false may be returned for some "equivalent" nodes.
  virtual bool IsDefinitelyEqualTo(const Node* other) const;

  // Returns whether this Op is of the template argument subclass. For example:
  // Is<Param>().
  template <typename OpT>
  bool Is() const {
    return IsOpClass<OpT>(op());
  }

  // Returns a down_cast pointer of the given template argument type. XLS_CHECK
  // fails if the object is not of the given type. For example: As<Param>().
  template <typename OpT>
  const OpT* As() const {
    XLS_CHECK(Is<OpT>());
    return down_cast<const OpT*>(this);
  }
  template <typename OpT>
  OpT* As() {
    XLS_CHECK(Is<OpT>());
    return down_cast<OpT*>(this);
  }

  // Returns the ordinal-annotated name of this node; e.g. "add.2".
  std::string GetName() const;

  // Returns the name of the node and any concise supplementary information.
  std::string ToString() const { return ToStringInternal(false); }

  // As above, but includes the type of the operands in the string.
  std::string ToStringWithOperandTypes() const {
    return ToStringInternal(true);
  }

  // Returns a string of operand names; e.g. "[param.2, literal.7]".
  std::string GetOperandsString() const;

  // Returns a string with the user of users; e.g. "{add.3, sub.7}".
  std::string GetUsersString() const;

  // Returns the unique set of users of this node sorted by id.
  absl::Span<Node* const> users() const { return users_; }

  // Helper for querying whether "target" is a user of this node.
  bool HasUser(const Node* target) const;

  // Returns true when the Op is in the list of choices
  bool OpIn(const std::vector<Op>& choices);

  Node* operand(int64 i) const {
    XLS_CHECK_LT(i, operands_.size());
    return operands_[i];
  }
  int64 operand_count() const { return operands_.size(); }

  // Returns true if target is an operand of this node.
  bool HasOperand(const Node* target) const;

  // Returns the number of times that 'target' appears in the operand list of
  // this node.
  int64 OperandInstanceCount(const Node* target) const;

  int64 id() const { return id_; }

  // Note: use with caution, the id should be unique among all nodes in a
  // function.
  void set_id(int64 id) { id_ = id; }

  // Clones the node with the new operands. Returns the newly created
  // instruction. The instruction is owned by new_function which must also
  // contain all of new_operands (if any).
  virtual xabsl::StatusOr<Node*> Clone(absl::Span<Node* const> new_operands,
                                       Function* new_function) const = 0;

 protected:
  // Function needs to be a friend to access RemoveUser for deleting nodes from
  // the graph.
  friend class Function;

  explicit Node(Op op, Type* type, absl::optional<SourceLocation> loc,
                Function* function);

  std::string ToStringInternal(bool include_operand_types) const;

  // Adds an operand to the operand list with a symmetric "user" link added to
  // those operands, noting that this node is a user.
  void AddOperand(Node* operand);

  // Adds the set of operands 'operands'. Updating user links as with
  // AddOperand.
  void AddOperands(absl::Span<Node* const> operands);

  // Adds an optional operand to the set of operands 'operands'. Updating user
  // links as with AddOperand.
  void AddOptionalOperand(absl::optional<Node*> operand);

  // Adds the given node to this node's function and replaces this node's uses
  // with the node.
  absl::Status AddNodeToFunctionAndReplace(std::unique_ptr<Node> replacement);

 private:
  void AddUser(Node* user);
  void RemoveUser(Node* user);

  Function* function_;
  int64 id_;
  Op op_;
  Type* type_;

  absl::optional<SourceLocation> loc_;
  std::vector<Node*> operands_;

#include "xls/ir/container_hack.inc"
  UnorderedSet<Node*> users_set_;
#include "xls/ir/container_hack_undef.inc"
  std::vector<Node*> users_;
};

inline std::ostream& operator<<(std::ostream& os, const Node& node) {
  os << node.ToString();
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const Node* node) {
  os << (node == nullptr ? std::string("<nullptr Node*>") : node->ToString());
  return os;
}

inline void NodeAppend(std::string* out, const Node* n) {
  absl::StrAppend(out, n->ToString());
}

}  // namespace xls

#endif  // XLS_IR_NODE_H_
