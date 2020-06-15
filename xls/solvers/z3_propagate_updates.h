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

// Defines routines for propagating a Z3 AST change down through any
// affected/downstream nodes.
#ifndef XLS_SOLVERS_Z3_PROPAGATE_UPDATES_H_
#define XLS_SOLVERS_Z3_PROPAGATE_UPDATES_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "../z3/src/api/z3.h"

namespace xls {
namespace solvers {
namespace z3 {

// Maps a node to the set of nodes that it depends on as input.
template <typename T>
using DownstreamNodes = absl::flat_hash_map<T, absl::flat_hash_set<T>>;

// Holds the node-to-Z3_ast mapping for an AST update. Z3_substitute needs both
// the AST being replaced as well as the replacement AST, so this struct holds
// them all together.
template <typename T>
struct UpdatedNode {
  T node;
  Z3_ast old_ast;
  Z3_ast new_ast;
};

// Forward declaration. GetDownstreamNodes returns the set of all nodes
// downstream of the set of "updated_nodes", along with their dependees.
template <typename T>
DownstreamNodes<T> GetDownstreamNodes(
    absl::Span<const UpdatedNode<T>> updated_nodes,
    std::function<std::vector<T>(T parent)> get_users);

// Propagates any updated Z3_asts through their downstream Z3 nodes.
// When a Z3_ast in a tree is replaced (via Z3_substitute), it results in a new
// Z3_ast. The replacement doesn't affect any nodes downstream of the original -
// it instead creates a new "sibling" of the original node.
// In our uses, we want any updates to be reflected across the entire tree,
// so we must manually propagate any changes to all downstream nodes. That means
// we must create replacements for any nodes downstream of the originals.
// To do this, we do a standard toposort down from the original updated nodes
// and process each downstream node as its inputs are satisfied.
//
// This process is nearly entirely uniform between IR and netlist translations,
// so it's been templatized for both.
//
// Args:
//  - ctx: The owning Z3 context.
//  - translations: Mapping of XLS node (IR or NetRef) to Z3_ast.
//  - get_inputs: Function returning the set of inputs for an [XLS] node (IR
//    Node or netlist Cell).
//  - get_inputs: Function returning the set of users for an [XLS] node (IR
//    Node or netlist Cell).
//  - downstream_nodes: The updated Z3_asts to propagate down.
template <typename T>
void PropagateAstUpdates(Z3_context ctx,
                         absl::flat_hash_map<T, Z3_ast>& translations,
                         std::function<std::vector<T>(T parent)> get_inputs,
                         std::function<std::vector<T>(T parent)> get_users,
                         const std::vector<UpdatedNode<T>>& input_nodes) {
  DownstreamNodes<T> downstream_nodes =
      GetDownstreamNodes(absl::MakeSpan(input_nodes), get_users);

  std::deque<T> active_nodes;
  absl::flat_hash_map<T, UpdatedNode<T>> updated_nodes;
  for (const UpdatedNode<T>& input_node : input_nodes) {
    active_nodes.push_back(input_node.node);
    updated_nodes.insert({input_node.node, {input_node}});
  }

  while (!active_nodes.empty()) {
    T active_node = active_nodes.front();
    active_nodes.pop_front();
    for (auto& pair : downstream_nodes) {
      if (!pair.second.contains(active_node)) {
        continue;
      }

      T downstream_node = pair.first;
      pair.second.erase(active_node);
      if (pair.second.empty()) {
        // We replace all updated references at once - that means we need to
        // collect all old/new Z3_ast pairs.
        std::vector<Z3_ast> old_inputs;
        std::vector<Z3_ast> new_inputs;
        for (const auto& input : get_inputs(downstream_node)) {
          if (updated_nodes.contains(input)) {
            old_inputs.push_back(updated_nodes[input].old_ast);
            new_inputs.push_back(updated_nodes[input].new_ast);
          }
        }

        // Update this node, put its users on the list
        Z3_ast old_ast = translations[downstream_node];
        Z3_ast new_ast = Z3_substitute(ctx, old_ast, old_inputs.size(),
                                       old_inputs.data(), new_inputs.data());
        updated_nodes[downstream_node] = {downstream_node, old_ast, new_ast};
        translations[downstream_node] = new_ast;

        for (const auto& user : get_users(downstream_node)) {
          active_nodes.push_back(user);
        }
      }
    }
  }
}

template <typename T>
DownstreamNodes<T> GetDownstreamNodes(
    absl::Span<const UpdatedNode<T>> updated_nodes,
    std::function<std::vector<T>(T parent)> get_users) {
  DownstreamNodes<T> affected_nodes;

  // The NetRefs that need to be updated - those downstream of the input refs.
  std::deque<T> live_nodes;
  for (const auto& updated_node : updated_nodes) {
    live_nodes.push_back(updated_node.node);
  }

  absl::flat_hash_set<T> seen_nodes;
  while (!live_nodes.empty()) {
    T node = live_nodes.front();
    live_nodes.pop_front();

    // Each cell with an updated ref as its input will need all of its outputs
    // to be updated.
    for (T user : get_users(node)) {
      affected_nodes[user].insert(node);
      if (!seen_nodes.contains(user)) {
        seen_nodes.insert(user);
        for (const auto& subuser : get_users(user)) {
          live_nodes.push_back(subuser);
        }
      }
    }
  }

  return affected_nodes;
}

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // XLS_SOLVERS_Z3_PROPAGATE_UPDATES_H_
