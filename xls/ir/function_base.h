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

#ifndef XLS_IR_FUNCTION_BASE_H_
#define XLS_IR_FUNCTION_BASE_H_

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/iterator_range.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/unwrapping_iterator.h"
#include "xls/ir/verify_node.h"

namespace xls {

class Function;
class Proc;

// Base class for Functions and Procs. A holder of a set of nodes.
class FunctionBase {
 protected:
  using NodeList = std::list<std::unique_ptr<Node>>;

 public:
  enum class Kind {
    kFunction,
    kProc,
    kBlock,
  };
  FunctionBase(std::string_view name, Package* package)
      : name_(name), package_(package) {}
  FunctionBase(const FunctionBase& other) = delete;
  void operator=(const FunctionBase& other) = delete;

  virtual ~FunctionBase() = default;

  Package* package() const { return package_; }
  const std::string& name() const { return name_; }
  void SetName(std::string_view name) { name_ = name; }
  std::string qualified_name() const {
    return absl::StrCat(package_->name(), "::", name_);
  }

  std::optional<int64_t> GetInitiationInterval() const {
    return initiation_interval_;
  }

  void SetInitiationInterval(int64_t ii) { initiation_interval_ = ii; }

  void ClearInitiationInterval() { initiation_interval_ = std::nullopt; }

  // Returns true if this is the top FunctionBase of the package.
  bool IsTop() const { return package()->IsTop(this); }

  // DumpIr emits the IR in a parsable, hierarchical text format.
  virtual std::string DumpIr() const = 0;

  // Get the kind of function base this is as an enum.
  virtual Kind kind() const = 0;

  // Get FunctionBase attributes suitable for putting in #[...] in the IR.
  std::vector<std::string> AttributeIrStrings() const;

  // Return Span of parameters.
  absl::Span<Param* const> params() const { return params_; }

  // Return the parameter at the given index.
  Param* param(int64_t index) const { return params_.at(index); }

  // Return the parameter with the given name.
  absl::StatusOr<Param*> GetParamByName(std::string_view param_name) const;

  absl::StatusOr<int64_t> GetParamIndex(Param* param) const;

  absl::Span<Next* const> next_values() const { return next_values_; }

  const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values(
      StateRead* state_read) const {
    return next_values_by_state_read_.at(state_read);
  }

  // Moves the given param to the given index in the parameter list.
  absl::Status MoveParamToIndex(Param* param, int64_t index);

  int64_t node_count() const { return nodes_.size(); }

  // Expose Nodes, so that transformation passes can operate
  // on this function.
  xabsl::iterator_range<UnwrappingIterator<NodeList::iterator>> nodes() {
    return xabsl::make_range(MakeUnwrappingIterator(nodes_.begin()),
                             MakeUnwrappingIterator(nodes_.end()));
  }
  xabsl::iterator_range<UnwrappingIterator<NodeList::const_iterator>> nodes()
      const {
    return xabsl::make_range(MakeUnwrappingIterator(nodes_.begin()),
                             MakeUnwrappingIterator(nodes_.end()));
  }
  xabsl::iterator_range<UnwrappingIterator<NodeList::reverse_iterator>>
  nodes_reversed() {
    return xabsl::make_range(MakeUnwrappingIterator(nodes_.rbegin()),
                             MakeUnwrappingIterator(nodes_.rend()));
  }
  xabsl::iterator_range<UnwrappingIterator<NodeList::const_reverse_iterator>>
  nodes_reversed() const {
    return xabsl::make_range(MakeUnwrappingIterator(nodes_.rbegin()),
                             MakeUnwrappingIterator(nodes_.rend()));
  }

  // Adds a node to the set owned by this function.
  template <typename T>
    requires(std::is_base_of_v<Node, T>)
  T* AddNode(std::unique_ptr<T> n) {
    T* ptr = n.get();
    AddNodeInternal(std::move(n));
    return ptr;
  }

  // Creates a new node and adds it to the function. NodeT is the node subclass
  // (e.g., 'Param') and the variadic args are the constructor arguments with
  // the exception of the final FunctionBase* argument. This method verifies the
  // newly constructed node after it is added to the function. Returns a pointer
  // to the newly constructed node.
  template <typename NodeT, typename... Args>
    requires(std::is_base_of_v<Node, NodeT>)
  absl::StatusOr<NodeT*> MakeNode(Args&&... args) {
    NodeT* new_node = AddNode(std::make_unique<NodeT>(
        std::forward<Args>(args)..., /*name=*/"", this));
    XLS_RETURN_IF_ERROR(VerifyNode(new_node));
    return new_node;
  }

  template <typename NodeT, typename... Args>
    requires(std::is_base_of_v<Node, NodeT>)
  absl::StatusOr<NodeT*> MakeNodeWithName(Args&&... args) {
    NodeT* new_node =
        AddNode(std::make_unique<NodeT>(std::forward<Args>(args)..., this));
    XLS_RETURN_IF_ERROR(VerifyNode(new_node));
    return new_node;
  }

  // Find a node by its name, as generated by DumpIr.
  std::optional<Node*> MaybeGetNode(std::string_view standard_node_name) const;
  absl::StatusOr<Node*> GetNode(std::string_view standard_node_name) const;
  bool HasNode(std::string_view standard_node_name) const {
    return MaybeGetNode(standard_node_name).has_value();
  }

  // Find a node with the given id.
  absl::StatusOr<Node*> GetNodeById(int64_t id) const;

  // Removes the node from the function. The node must have no users.
  // Warning: if you remove a parameter node via this method you will change the
  // function type signature.
  virtual absl::Status RemoveNode(Node* n);

  // Visit all nodes (including nodes not reachable from the root) in the
  // function using the given visitor.
  absl::Status Accept(DfsVisitor* visitor);

  // Sanitizes and uniquifies the given name using the function's name
  // uniquer. Registers the uniquified name in the uniquer so it is not handed
  // out again.
  std::string UniquifyNodeName(std::string_view name) {
    return node_name_uniquer_.GetSanitizedUniqueName(name);
  }

  // Returns whether this FunctionBase is a function, proc, or block.
  bool IsFunction() const { return kind() == Kind::kFunction; }
  bool IsProc() const { return kind() == Kind::kProc; }
  bool IsBlock() const { return kind() == Kind::kBlock; }

  const Function* AsFunctionOrDie() const;
  Function* AsFunctionOrDie();
  const Proc* AsProcOrDie() const;
  Proc* AsProcOrDie();
  const Block* AsBlockOrDie() const;
  Block* AsBlockOrDie();

  // Returns true if the given node has implicit uses in the function. Implicit
  // uses include return values of functions and the recurrent token/state in
  // procs.
  virtual bool HasImplicitUse(Node* node) const = 0;

  // Set information about foreign function
  void SetForeignFunctionData(const std::optional<ForeignFunctionData>& ff) {
    foreign_function_ = ff;
  }

  // If this is to be expressed as foreign function call, returns the necessary
  // call information.
  const std::optional<xls::ForeignFunctionData>& ForeignFunctionData() const {
    return foreign_function_;
  }

  absl::Span<ChangeListener* const> ChangeListeners() const {
    return change_listeners_;
  }
  void RegisterChangeListener(ChangeListener* listener) {
    change_listeners_.push_back(listener);
  }
  void UnregisterChangeListener(ChangeListener* listener) {
    std::erase(change_listeners_, listener);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const FunctionBase& fb) {
    absl::Format(&sink, "%s", fb.name());
  }

  // Comparator used for sorting by name.
  static bool NameLessThan(const FunctionBase* a, const FunctionBase* b) {
    return a->name() < b->name();
  }

  // Struct form for passing comparators as template arguments.
  struct NameLessThan {
    bool operator()(const FunctionBase* a, const FunctionBase* b) const {
      return FunctionBase::NameLessThan(a, b);
    }
  };

  // Rebuild any side tables (to the maximum extent possible) using only the
  // node graph and unchangeable metadata.
  // TODO(allgiht): This is a terrible API since you basically just need to know
  // when it needs to be called. Ideally this should be called regularly in some
  // fashion.
  // TODO(allight): Significant parts of the function/proc/block is included in
  // the 'unchangeable metadata' and some of it is actually pretty malleable. We
  // should move to a more node-y future where almost all
  // configuration/definition data is embedded in the node graph.
  absl::Status RebuildSideTables();

 protected:
  // Many function-types have side-tables that store various pieces of
  // information. This function should, as much as possible, rebuild any using
  // only data from the nodes. If data is missing or corrupted or a valid setup
  // cannot be created then an error may be returned.
  virtual absl::Status InternalRebuildSideTables() = 0;
  // Internal virtual helper for adding a node. Returns a pointer to the newly
  // added node.
  virtual Node* AddNodeInternal(std::unique_ptr<Node> node);

  // Returns a vector containing the reserved words in the IR.
  static std::vector<std::string> GetIrReservedWords();

  std::string name_;
  Package* package_;
  std::optional<int64_t> initiation_interval_;

  // Store Nodes in std::list as they can be added and removed arbitrarily and
  // we want a stable iteration order. Keep a map from instruction pointer to
  // location in the list for fast lookup.
  NodeList nodes_;
  absl::flat_hash_map<const Node*, NodeList::iterator> node_iterators_;

  std::vector<Param*> params_;
  std::vector<Next*> next_values_;
  absl::flat_hash_map<StateRead*, absl::btree_set<Next*, Node::NodeIdLessThan>>
      next_values_by_state_read_;

  NameUniquer node_name_uniquer_ =
      NameUniquer(/*separator=*/"__", GetIrReservedWords());

  std::optional<xls::ForeignFunctionData> foreign_function_;

  std::vector<ChangeListener*> change_listeners_;
};

inline absl::Span<ChangeListener* const> GetChangeListeners(
    FunctionBase* function_base) {
  return function_base->ChangeListeners();
}

std::ostream& operator<<(std::ostream& os, const FunctionBase& function);
std::ostream& operator<<(std::ostream& os, const FunctionBase::Kind& kind);

}  // namespace xls

#endif  // XLS_IR_FUNCTION_BASE_H_
