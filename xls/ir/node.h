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

#ifndef XLS_IR_NODE_H_
#define XLS_IR_NODE_H_

#include <compare>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/pointer_utils.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"

namespace xls {

class Package;
class Node;
class FunctionBase;

absl::Span<ChangeListener* const> GetChangeListeners(
    FunctionBase* function_base);

// Forward declaration to avoid circular dependency.
class DfsVisitor;

// A (non-owning) reference to a node; contains both the node's ID and the
// pointer to the node, enabling efficient access & safe comparison.
class NodeRef {
 public:
  explicit NodeRef(Node* node);

  int64_t id() const { return id_; }
  Node* node() const { return node_; }

  // On dereference, behaves exactly like the underlying node.
  Node& operator*() const { return *node_; }
  Node* operator->() const { return node_; }

  friend bool operator==(const NodeRef& a, const NodeRef& b) {
    DCHECK(a.id() != b.id() || a.node() == b.node())
        << "False node match due to reused ID";
    return a.id() == b.id();
  }

  template <typename H>
  friend H AbslHashValue(H h, const NodeRef& node) {
    return h.combine(std::move(h), node.id());
  }

 private:
  int64_t id_;
  Node* node_;
};

// Abstract type for a node (representing an expression) in the high level IR.
//
// Node is subtyped and can be checked-converted via the As* methods below.
class Node {
 public:
  virtual ~Node() = default;

  // Accepts the visitor, instructing it to visit this node.
  absl::Status Accept(DfsVisitor* visitor);

  // Visits this node with the given visitor. Visits only this node and does not
  // traverse the graph.
  absl::Status VisitSingleNode(DfsVisitor* visitor);

  Type* GetType() const { return type_; }

  // Precondition: this node's result type must be known to be a `BitsType`.
  //
  // Convenience helper for getting the bit count of the result type. Many
  // operations produce bits-typed outputs, so this is a useful helper even
  // though it is not properly polymorphic. It must only be called on nodes that
  // we are sure are producing bits-typed results.
  int64_t BitCountOrDie() const {
    return GetType()->AsBitsOrDie()->bit_count();
  }

  Op op() const { return op_; }
  FunctionBase* function_base() const { return function_base_; }
  Package* package() const;
  const SourceInfo& loc() const { return loc_; }

  // Returns the sequence of operands used by this node.
  //
  // Note: we make the view of the operands immutable in order to ensure that we
  // preserve invariants across mutations (so you can't forget to update
  // operand/user symmetry); e.g. see helpers ReplaceOperands() and
  // SwapOperands() below.
  absl::Span<Node* const> operands() const { return operands_; }

  // Replaces all instances of 'old_operand' with 'new_operand', and returns
  // whether at least one operand was replaced.
  bool ReplaceOperand(Node* old_operand, Node* new_operand);

  // Replaces the existing operand at position 'operand_no' with 'new_operand'.
  absl::Status ReplaceOperandNumber(int64_t operand_no, Node* new_operand,
                                    bool type_must_match = true);

  // Replace all uses of this node with 'replacement', except within those users
  // where 'filter' returns false. If replace_implicit_uses is false implicit
  // uses (such as return-values/next line uses) will not be replaced.
  //
  // TODO(allight): The remove_implicit_uses should be removed once next-node is
  // complete since its only there because in functions using next-node there is
  // an implicit 'param' on the next value line internally which if it is
  // removed it messes up the verifier.
  absl::Status ReplaceUsesWith(Node* replacement,
                               const std::function<bool(Node*)>& filter,
                               bool replace_implicit_uses = true);
  // Replace all uses of this node with 'replacement'. If replace_implicit_uses
  // is false implicit uses (such as return-values/next line uses) will not be
  // replaced.
  //
  // TODO(allight): The remove_implicit_uses should be removed once next-node is
  // complete since its only there because in functions using next-node there is
  // an implicit 'param' on the next value line internally which if it is
  // removed it messes up the verifier.
  absl::Status ReplaceUsesWith(Node* replacement,
                               bool replace_implicit_uses = true) {
    return ReplaceUsesWith(
        replacement, [](Node*) { return true; }, replace_implicit_uses);
  }

  // Constructs a new node and replaces all uses of 'this' with the newly
  // constructed node. NodeT is the node subclass (e.g., 'Param') and the
  // variadic args are the constructor arguments with the exception of the first
  // loc argument and the final Function* argument which are inherited from this
  // node. Returns a pointer to the newly constructed node.
  template <typename NodeT, typename... Args>
  absl::StatusOr<NodeT*> ReplaceUsesWithNew(Args&&... args) {
    std::unique_ptr<NodeT> new_node = std::make_unique<NodeT>(
        loc(), std::forward<Args>(args)..., /*name=*/"", function_base());
    NodeT* ptr = new_node.get();
    XLS_RETURN_IF_ERROR(AddNodeToFunctionAndReplace(std::move(new_node)));
    return ptr;
  }

  // Replaces any implicit uses of this node with the given
  // replacement. Implicit uses include function return values and proc next
  // state and tokens. For example, if this node is the return value of a
  // function, then calling ReplaceImplicitUsesWith(foo) will make foo the
  // return value. Returns whether any implicit uses were replaced.
  absl::StatusOr<bool> ReplaceImplicitUsesWith(Node* replacement);

  // Swaps the operands at indices 'a' and 'b' in the operands sequence.
  void SwapOperands(int64_t a, int64_t b);

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

  // Returns a down_cast pointer of the given template argument type. CHECK
  // fails if the object is not of the given type. For example: As<Param>().
  template <typename OpT>
  const OpT* As() const {
    CHECK(Is<OpT>());
    return down_cast<const OpT*>(this);
  }
  template <typename OpT>
  OpT* As() {
    CHECK(Is<OpT>());
    return down_cast<OpT*>(this);
  }

  // Returns whether this node was assigned a name at construction. Nodes
  // without assigned names will have names generated from the opcode and unique
  // id.
  bool HasAssignedName() const { return name_ != nullptr; }

  // Returns the name of this node. If not assigned at construction time, the
  // name is generated from the opcode and unique id (e.g. "add.2");
  std::string GetName() const;

  // Returns the name of this node, if assigned. Will be empty iff
  // HasAssignedName returns false.
  std::string_view GetNameView() const;

  // Sets the name of this node. After this method is called. HasAssignedName
  // will return true.
  void SetName(std::string_view name);

  // Sets the name of this node. Makes no attempt to unique-ify the name so care
  // must be taken that there are no collisions.
  void SetNameDirectly(std::string_view name);

  // Clears the name of this node. The node will have a generate name based on
  // the opcode and ID. After this method is called. HasAssignedName will return
  // false.
  void ClearName();

  // Set source location.
  void SetLoc(const SourceInfo& loc);

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

  // Comparator used for sorting by node ID.
  struct NodeIdLessThan {
    bool operator()(const Node* a, const Node* b) const {
      // Params may have duplicate node ids so compare pointers if id's match.
      // TODO(b/138805124): 2019/08/19 Figure out how to represent param uids in
      // the textual IR so parameters can get unique ids like all other nodes
      // rather than being unconditionally numbered from zero.
      if (a->id() == b->id()) {
        return a < b;
      }
      return a->id() < b->id();
    }

    std::strong_ordering Compare(const Node* a, const Node* b) const {
      if (std::strong_ordering id_cmp = a->id() <=> b->id();
          id_cmp != std::strong_ordering::equal) {
        return id_cmp;
      }
      return a <=> b;
    }
  };

  // Returns the unique set of users of this node sorted by id.
  absl::Span<Node* const> users() const { return users_; }

  // Helper for querying whether "target" is a user of this node.
  bool HasUser(const Node* target) const;

  // Returns true if the node has no explicit uses (users() is empty) and no
  // implicit uses (e.g., is root node of a function).
  bool IsDead() const;

  // Returns true when the Op is in the list of choices
  bool OpIn(absl::Span<const Op> choices) const;

  Node* operand(int64_t i) const {
    CHECK_LT(i, operands_.size());
    return operands_[i];
  }
  int64_t operand_count() const { return operands_.size(); }

  // Returns true if target is an operand of this node.
  bool HasOperand(const Node* target) const;

  // Returns the number of times that 'target' appears in the operand list of
  // this node.
  int64_t OperandInstanceCount(const Node* target) const;

  int64_t id() const { return id_; }

  // Sets the id of the node. Mutates the user sets of the operands of the node
  // because user sets are sorted by id.  Note: this should only be used by the
  // parser and ideally not even there.
  // TODO(meheff): 2021/05/05 Remove this method.
  void SetId(int64_t id);

  // Clones the node with the new operands. Returns the newly created
  // instruction.
  absl::StatusOr<Node*> Clone(absl::Span<Node* const> new_operands) const {
    return CloneInNewFunction(new_operands, function_base());
  }

  // As above but clones the instruction into the function 'new_function'. The
  // instruction is owned by new_function which must also contain all of
  // new_operands (if any).
  virtual absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Node& node) {
    absl::Format(&sink, "%s", node.GetName());
  }

  // User-data access functions. Should not be directly used. Use NodeMap
  // instead.
  //
  // Extreme care should be used when interacting with these functions and the
  // Package ones since this is basically doing manual memory management.

  // Get the pointer associated with this indexes user data or nullptr if
  // never set. Use HasUserData to see if anything has ever been set.
  //
  // idx must be a value returned by Package::AllocateNodeUserData which has not
  // had ReleaseNodeUserDataId called on it.
  void* GetUserData(int64_t idx);
  // Sets user data at idx to 'data'.
  void SetUserData(int64_t idx, TypeErasedUniquePtr data);
  // Removes user data at idx from the node. Returns std::nullopt if nothing has
  // been set.
  std::optional<TypeErasedUniquePtr> TakeUserData(int64_t idx);
  // Checks if anything has ever been set at the given user data.
  bool HasUserData(int64_t idx) {
    return user_data_.size() > idx && user_data_[idx].has_value();
  }

 protected:
  // FunctionBase needs to be a friend to access RemoveUser for deleting nodes
  // from the graph.
  friend class FunctionBase;
  // Block needs to be a friend to strongly name ports (guarantee name has no
  // uniquifying prefix).
  friend class Block;

  Node(Op op, Type* type, const SourceInfo& loc, std::string_view name,
       FunctionBase* function);

  std::string ToStringInternal(bool include_operand_types) const;

  // Adds an operand to the operand list with a symmetric "user" link added to
  // those operands, noting that this node is a user.
  void AddOperand(Node* operand);

  // Adds the set of operands 'operands'. Updating user links as with
  // AddOperand.
  void AddOperands(absl::Span<Node* const> operands);

  // Adds an optional operand to the set of operands 'operands'. Updating user
  // links as with AddOperand.
  void AddOptionalOperand(std::optional<Node*> operand);

  // Removes the optional operand at position 'operand_no'. Notes that this node
  // is no longer a user of the operand if this is its last use.
  absl::Status RemoveOptionalOperand(int64_t operand_no);

  // Adds the given node to this node's function and replaces this node's uses
  // with the node.
  absl::Status AddNodeToFunctionAndReplace(std::unique_ptr<Node> replacement);

  void AddUser(Node* user);
  void RemoveUser(Node* user);

  // The number of users that we consider small enough to perform linear-time
  // algorithms on.
  static constexpr int64_t kSmallUserCount = 8;

  FunctionBase* function_base_;
  int64_t id_;
  Op op_;
  Type* type_;
  SourceInfo loc_;
  std::unique_ptr<std::string> name_;  // Non-null if name has been assigned.

  // Most nodes have <= 2 operands, so we keep those locally if we can.
  absl::InlinedVector<Node*, 2> operands_;

  // Set of users sorted by node_id for stability.
  absl::InlinedVector<Node*, 2> users_;

 private:
  std::vector<std::optional<TypeErasedUniquePtr>> user_data_;

  // Clear all user data.
  void ClearUserData() { user_data_.clear(); };

  // for ClearUserData
  friend class Package;
};

inline NodeRef::NodeRef(Node* node)
    : id_(node == nullptr ? -1 : node->id()), node_(node) {}

inline std::ostream& operator<<(std::ostream& os, const Node& node) {
  os << node.ToString();
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const Node* node) {
  os << (node == nullptr ? std::string("<nullptr Node*>") : node->ToString());
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const NodeRef& node) {
  os << node->ToString();
  return os;
}

inline bool operator==(const NodeRef& a, Node* b) {
  if (b == nullptr) {
    return a.node() == nullptr;
  }
  DCHECK(a.id() != b->id() || a.node() == b)
      << "False node match due to reused ID";
  return a.id() == b->id();
}
inline bool operator==(Node* a, const NodeRef& b) { return b == a; }

}  // namespace xls

#endif  // XLS_IR_NODE_H_
