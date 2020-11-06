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

#ifndef XLS_INTEGRATOR_IR_INTEGRATOR_H_
#define XLS_INTEGRATOR_IR_INTEGRATOR_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"

namespace xls {

// Class that represents an integration function i.e. a function combining the
// IR of other functions. This class tracks which original function nodes are
// mapped to which integration function nodes. It also provides some utilities
// that are useful for constructing the integrated function.
class IntegrationFunction {
 public:
  IntegrationFunction() {}

  IntegrationFunction(const IntegrationFunction& other) = delete;
  void operator=(const IntegrationFunction& other) = delete;

  // Create an IntegrationFunction object that is empty expect for
  // parameters. Each initial parameter of the function is a tuple
  // which holds inputs corresponding to the paramters of one
  // of the source_functions.
  static absl::StatusOr<std::unique_ptr<IntegrationFunction>>
  MakeIntegrationFunctionWithParamTuples(
      Package* package, absl::Span<const Function* const> source_functions,
      std::string function_name = "IntegrationFunction");

  Function* function() const { return function_.get(); }

  // Add the external node 'to_insert' into the integration function. Mapped
  // operands are automatically discovered and connected.
  absl::StatusOr<Node*> InsertNode(const Node* to_insert);

  // Estimate the cost of merging node_a and node_b. If nodes cannot be
  // merged, not value is returned.
  absl::StatusOr<std::optional<int64>> GetMergeNodesCost(const Node* node_a,
                                                         const Node* node_b);

  // Merge node_a and node_b. Returns the nodes that node_a and node_b
  // map to after merging (vector contains a single node if they map
  // to the same node).
  absl::StatusOr<std::vector<Node*>> MergeNodes(Node* node_a, Node* node_b);

  // For the integration function nodes node_a and node_b,
  // returns a UnifiedNode. UnifiedNode.node points to a single integration
  // function node that combines the two nodes. This may involve adding a mux
  // and a parameter to serve as the mux select signal.
  // UnifiedNode.new_mux_added will be be set to true if this call added a new
  // mux. Otherwise, false.
  struct UnifiedNode {
    Node* node;
    bool new_mux_added;
  };
  absl::StatusOr<UnifiedNode> UnifyIntegrationNodes(Node* node_a, Node* node_b);

  // Return a UnifiedOperands struct in which the 'operands' vector holds nodes
  // where each node unifies the corresponding operands of 'node_a' and
  // 'node_b'.  The 'added_muxes' field lists all new muxes created by this
  // call.
  struct UnifiedOperands {
    std::vector<Node*> operands;
    std::vector<Node*> added_muxes;
  };
  absl::StatusOr<UnifiedOperands> UnifyNodeOperands(const Node* node_a,
                                                    const Node* node_b);

  // For a mux produced by UnifyIntegrationNodes, remove the mux and
  // the select paramter. Updates internal unification / mux book-keeping
  // accordingly. This function should only be called if the mux has no users
  // and mux's select signal is only used by the mux.
  absl::Status DeUnifyIntegrationNodes(Node* mux);

  // Declares that node 'source' from a source function maps
  // to node 'map_target' in the integrated_function.
  absl::Status SetNodeMapping(const Node* source, Node* map_target);

  // Returns the integrated node that 'original' maps to, if it
  // exists. Otherwise, return an error status.
  absl::StatusOr<Node*> GetNodeMapping(const Node* original) const;

  // Returns the original nodes that map to 'map_target' in the integrated
  // function.
  absl::StatusOr<const absl::flat_hash_set<const Node*>*> GetNodesMappedToNode(
      const Node* map_target) const;

  // Returns a vector of Nodes to which the operands of the node
  // 'node' map. If node is owned by the integrated function, these are just
  // node's operands. If an operand does not yet have a mapping, the operand is
  // temporarily mapped to a new parameter(not yet implemented). Use of this
  // temporary will be replaced with the real mapping when it is set.
  absl::StatusOr<std::vector<Node*>> GetIntegratedOperands(
      const Node* node) const;

  // Returns true if 'node' is mapped to a node in the integrated function.
  bool HasMapping(const Node* node) const;

  // Returns true if other nodes map to 'node'
  bool IsMappingTarget(const Node* node) const;

  // Returns true if 'node' is in the integrated function.
  bool IntegrationFunctionOwnsNode(const Node* node) const {
    return function_.get() == node->function_base();
  }

  // Returns an estimate of the (gate count? area?) cost of a node.
  int64 GetNodeCost(const Node* node) const;

 private:
  // Helper function that implements the logic for merging nodes,
  // allowing for either the merge to be performed or for the cost
  // of the merge to be estimated.  A MergeNodesBackendResult struct is
  // returned. The 'can_merge' field indicates if the nodes can be merged. If
  // they can be merged, target_a and target_b point to resulting nodes that
  // represent the values of 'node_a' and 'node_b' in the integrated graph
  // (note: these will not necessarily point to the same merged node e.g. if the
  // merge node has a wider bitwidth than one of the original nodes, the target
  // pointer may instead point to a bitslice that takes in the wider node as an
  // operand). New muxes created by this call are placed in 'added_muxes'. Other
  // nodes created by this call are placed in 'other_added_nodes'.
  struct MergeNodesBackendResult {
    bool can_merge;
    Node* target_a = nullptr;
    Node* target_b = nullptr;
    std::vector<Node*> added_muxes;
    // We use a list rather than a vector here because
    // we will later want to remove elements in a (currently)
    // unknown order.  This would involve wastefule data copying
    // if we used a vector.
    std::list<Node*> other_added_nodes;
  };
  absl::StatusOr<MergeNodesBackendResult> MergeNodesBackend(const Node* node_a,
                                                            const Node* node_b);

  // Track mapping of original function nodes to integrated function nodes.
  absl::flat_hash_map<const Node*, Node*> original_node_to_integrated_node_map_;
  absl::flat_hash_map<const Node*, absl::flat_hash_set<const Node*>>
      integrated_node_to_original_nodes_map_;

  // Track which node-pairs have an associated mux.
  absl::flat_hash_map<std::pair<const Node*, const Node*>, Node*>
      node_pair_to_mux_;

  // Integrated function.
  std::unique_ptr<Function> function_;
  Package* package_;
};

// Class used to integrate separate functions into a combined, reprogrammable
// circuit that can be configured to have the same functionality as the
// input functions. The builder will attempt to construct the integrated
// function such that hardware common to the input functions is consolidated.
// Note that this is distinct from function inlining. With inlining, a function
// call is replaced by the body of the function that is called.  With function
// integration, we take separate functions that do not call each other and
// combine the hardware used to implement the functions.
class IntegrationBuilder {
 public:
  // Creates an IntegrationBuilder and uses it to produce an integrated function
  // implementing all functions in source_functions_.
  static absl::StatusOr<std::unique_ptr<IntegrationBuilder>> Build(
      absl::Span<const Function* const> input_functions);

  Package* package() { return package_.get(); }
  Function* integrated_function() { return integrated_function_; }

 private:
  IntegrationBuilder(absl::Span<const Function* const> input_functions) {
    original_package_source_functions_.insert(
        original_package_source_functions_.end(), input_functions.begin(),
        input_functions.end());
    // TODO(jbaileyhandle): Make package name an optional argument.
    package_ = absl::make_unique<Package>("IntegrationPackage");
  }

 private:
  // Copy the source functions into a common package.
  absl::Status CopySourcesToIntegrationPackage();

  // Recursively copy a function into the common package_.
  absl::StatusOr<Function*> CloneFunctionRecursive(
      const Function* function,
      absl::flat_hash_map<const Function*, Function*>* call_remapping);

  NameUniquer function_name_uniquer_ = NameUniquer(/*separator=*/"__");

  // Common package for to-be integrated functions
  // and integrated function.
  std::unique_ptr<Package> package_;

  Function* integrated_function_;

  // Functions to be integrated, in the integration package.
  std::vector<Function*> source_functions_;
  // Functions to be integrated, in their original packages.
  std::vector<const Function*> original_package_source_functions_;
};

}  // namespace xls

#endif  // XLS_INTEGRATOR_IR_INTEGRATOR_H_
