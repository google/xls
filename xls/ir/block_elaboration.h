// Copyright 2024 The XLS Authors
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

#ifndef XLS_IR_BLOCK_ELABORATION_H_
#define XLS_IR_BLOCK_ELABORATION_H_

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {
class BlockInstance;
class ElaboratedBlockDfsVisitor;

// A node and its associated hierarchical instance.
struct ElaboratedNode {
  Node* node;
  // Owned by the top-level elaboration.
  BlockInstance* instance;

  bool operator==(const ElaboratedNode& other) const {
    return node == other.node && instance == other.instance;
  }
  bool operator!=(const ElaboratedNode& other) const {
    return !(*this == other);
  }

  std::string ToString() const;
  absl::Status Accept(ElaboratedBlockDfsVisitor& visitor) const;
  absl::Status VisitSingleNode(ElaboratedBlockDfsVisitor& visitor) const;
};

std::ostream& operator<<(std::ostream& os,
                         const ElaboratedNode& node_and_instance);

template <typename H>
H AbslHashValue(H h, const ElaboratedNode& node_and_instance) {
  return H::combine(std::move(h), node_and_instance.node->id(),
                    node_and_instance.instance);
}

// Representation of an instance of a block. This is a recursive data structure
// which also owns all block instances instantiated by this block (and
// transitively their instances).
class BlockInstance {
 public:
  BlockInstance(
      std::optional<Block*> block, std::optional<Instantiation*> instantiation,
      BlockInstantiationPath&& path,
      std::vector<std::unique_ptr<BlockInstance>> instantiated_blocks);

  // Returns the block associated with this instance if it exists. Some
  // instantiations (e.g. fifo) do not have an associated block and will
  // return std::nullopt.
  std::optional<Block*> block() const { return block_; }

  // Prefix for referencing entities hierarchically.
  //
  // Empty string for top inst, "inst_name::inst_name::...::" for everyone else.
  // This is useful for e.g. referring to a register hierarchically, e.g.
  // `inst_a::inst_b::reg_a`.
  std::string_view RegisterPrefix() const { return register_prefix_; }
  std::string ToString() const;

  std::optional<Instantiation*> instantiation() const { return instantiation_; }

  // The path to this block instance through the block hierarchy.
  const BlockInstantiationPath& path() const { return path_; }

  // The BlockInstances instantiated by this block instance.
  absl::Span<const std::unique_ptr<BlockInstance>> child_instances() const {
    return child_instances_;
  }

  // Nodes that are connected to child instances. These are
  // InstantiationInput/InstantiationOutput nodes connected to child
  // InputPort/OutputPort nodes.
  const absl::flat_hash_map<Node*, ElaboratedNode>& parent_to_child_ports()
      const {
    return parent_to_child_ports_;
  }
  // Nodes that are connected to the parent instance. These are
  // InputPort/OutputPort nodes connected to parent
  // InstantiationInput/InstantiationOutput.
  const absl::flat_hash_map<Node*, ElaboratedNode>& child_to_parent_ports()
      const {
    return child_to_parent_ports_;
  }

 private:
  std::optional<Block*> block_;
  std::optional<Instantiation*> instantiation_;
  BlockInstantiationPath path_;
  std::string register_prefix_;

  // Child instances of this instance. Unique pointers are used for pointer
  // stability as pointers to these objects are handed out.
  std::vector<std::unique_ptr<BlockInstance>> child_instances_;

  absl::flat_hash_map<Node*, ElaboratedNode> parent_to_child_ports_;
  absl::flat_hash_map<Node*, ElaboratedNode> child_to_parent_ports_;
};

// Data structure representing the elaboration tree starting from a root block.
class BlockElaboration {
 public:
  static absl::StatusOr<BlockElaboration> Elaborate(Block* top);

  BlockInstance* top() const { return top_.get(); }

  std::string ToString() const { return top_->ToString(); }

  absl::StatusOr<BlockInstance*> GetInstance(
      const BlockInstantiationPath& path) const;
  absl::StatusOr<BlockInstance*> GetInstance(std::string_view path_str) const;

  absl::Span<BlockInstance* const> instances() const { return instance_ptrs_; }
  absl::Span<Block* const> blocks() const { return blocks_; }

  // Return all instances of a particular `FunctionT`.
  absl::Span<BlockInstance* const> GetInstances(Block* block) const;

  // Return the unique instance of the given proc/channel. Returns an error if
  // there is not exactly one instance associated with the IR object.
  absl::StatusOr<BlockInstance*> GetUniqueInstance(Block* function) const;

  Package* package() const { return package_; }

  // Create path from the given path string serialization. Example input:
  //
  //    top_proc::inst1->other_proc::inst2->that_proc
  //
  // The return path will have the Proc pointer to `top_proc` as the top of
  // the path, with an instantiation path containing the BlockInstantiation
  // pointers: {inst1, inst2}.
  //
  // Returns an error if the path does not exist in the elaboration.
  absl::StatusOr<BlockInstantiationPath> CreatePath(
      std::string_view path_str) const;

  absl::Status Accept(ElaboratedBlockDfsVisitor& visitor) const;

 private:
  BlockElaboration() = default;

  Package* package_;
  // The top-level instance. All other BlockInstances are contained within this.
  std::unique_ptr<BlockInstance> top_;
  // Pointers to all instances (including the top).
  std::vector<BlockInstance*> instance_ptrs_;
  // List of all blocks that are instantiated.
  std::vector<Block*> blocks_;

  // All proc instances in the elaboration indexed by instantiation path.
  absl::flat_hash_map<BlockInstantiationPath, BlockInstance*>
      instances_by_path_;

  absl::flat_hash_map<Block*, std::vector<BlockInstance*>>
      instances_of_function_;
};

// Returns a list of every (Node, BlockInstance) in the elaboration in topo
// order.
std::vector<ElaboratedNode> ElaboratedTopoSort(
    const BlockElaboration& elaboration);

// As above, but returns a reverse topo order.
std::vector<ElaboratedNode> ElaboratedReverseTopoSort(
    const BlockElaboration& elaboration);

}  // namespace xls

#endif  // XLS_IR_BLOCK_ELABORATION_H_
