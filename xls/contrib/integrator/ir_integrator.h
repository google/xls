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

#ifndef XLS_CONTRIB_INTEGRATOR_IR_INTEGRATOR_H_
#define XLS_CONTRIB_INTEGRATOR_IR_INTEGRATOR_H_

#include <list>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/contrib/integrator/integration_options.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"

namespace xls {

// Class that represents an integration function i.e. a function combining the
// IR of other functions. This class tracks which original function nodes are
// mapped to which integration function nodes. It also provides some utilities
// that are useful for constructing the integrated function.
// TODO(jbaileyhandle): Consider breaking this up into 2 or more smaller
// classes (e.g. could have a class for handling muxes, another for tracking
// mappings etc). Should move these classes and their tests into a new file(s).
class IntegrationFunction {
 public:
  IntegrationFunction(const IntegrationFunction& other) = delete;
  void operator=(const IntegrationFunction& other) = delete;

  // Create an IntegrationFunction object that is empty expect for
  // parameters. Each initial parameter of the function is a tuple
  // which holds inputs corresponding to the parameters of one
  // of the source_functions.
  static absl::StatusOr<std::unique_ptr<IntegrationFunction>>
  MakeIntegrationFunctionWithParamTuples(
      Package* package, absl::Span<const Function* const> source_functions,
      const IntegrationOptions& options = IntegrationOptions(),
      std::string function_name = "IntegrationFunction");

  // Create a tuple of the nodes that are the map targets of
  // the return values of the source functions. Set this value
  // as the return value of the integration function and return the tuple.
  // Should not be called if a return value is already set for
  // the integration function.
  absl::StatusOr<Node*> MakeTupleReturnValue();

  Function* function() const { return function_.get(); }
  Node* global_mux_select() const { return global_mux_select_param_; }

  // Estimate the cost of inserting the node 'to_insert'.
  absl::StatusOr<float> GetInsertNodeCost(const Node* to_insert);

  // Add the external node 'to_insert' into the integration function. Mapped
  // operands are automatically discovered and connected.
  absl::StatusOr<Node*> InsertNode(Node* to_insert);

  // Estimate the cost of merging node_a and node_b. If nodes cannot be
  // merged, not value is returned.
  absl::StatusOr<std::optional<int64_t>> GetMergeNodesCost(const Node* node_a,
                                                           const Node* node_b);

  // Merge node_a and node_b. Operands are automatically multiplexed.
  // Returns the nodes that node_a and node_b map to after merging (vector
  // contains a single node if they map to the same node).
  absl::StatusOr<std::vector<Node*>> MergeNodes(Node* node_a, Node* node_b);

  enum class UnificationChange {
    kNoChange,
    kNewMuxAdded,
    kExistingMuxCasesModified,
  };
  struct UnifiedNode {
    Node* node;
    UnificationChange change;
  };
  // For the integration function nodes node_a and node_b,
  // returns a UnifiedNode. UnifiedNode.node points to a single integration
  // function node that combines the two nodes. This may involve adding a mux
  // and a parameter to serve as the mux select signal.
  // UnifiedNode.change will indicate if the ir graph is unchaged, if a new
  // mux was added by this call, or if an existing mux was modified by this
  // call.
  absl::StatusOr<UnifiedNode> UnifyIntegrationNodes(Node* node_a, Node* node_b);

  // Return a UnifiedOperands struct in which the 'operands' vector holds nodes
  // where each node unifies the corresponding operands of 'node_a' and
  // 'node_b'.  The 'changed_muxes' field lists all muxes created by or modified
  // by this call.
  struct UnifiedOperands {
    std::vector<Node*> operands;
    std::vector<UnifiedNode> changed_muxes;
  };
  absl::StatusOr<UnifiedOperands> UnifyNodeOperands(const Node* node_a,
                                                    const Node* node_b);

  // Return the case indexes which are in active use for a mux
  // whose selector is global_mux_select_param_;
  absl::StatusOr<const std::set<int64_t>*> GetGlobalMuxOccupiedCaseIndexes(
      const Node* node) const;

  // Return the case indices which were most recently added to a mux
  // whose selector is global_mux_select_param_;
  absl::StatusOr<const std::set<int64_t>*> GetGlobalMuxLastCaseIndexesAdded(
      const Node* node) const;

  // Returns how many muxes whose selector is global_mux_select_param_
  // we track metadata for.
  int64_t GetNumberOfGlobalMuxesTracked() const;

  // For a mux produced by UnifyIntegrationNodes, undoes the previous
  // call to UnifyIntegrationNodes. Also updates
  // internal unification / mux book-keeping accordingly. If the mux was
  // modified / replaced, the new mux is returned. If the mux was removed,
  // nullptr is returned.
  absl::StatusOr<Node*> DeUnifyIntegrationNodes(Node* node);

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

  // Returns the source function index for the given node.
  absl::StatusOr<int64_t> GetSourceFunctionIndexOfNode(const Node* node) const;

  // Returns the source function index of all nodes that map to 'map_target'.
  absl::StatusOr<std::set<int64_t>> GetSourceFunctionIndexesOfNodesMappedToNode(
      const Node* map_target) const;

  // Returns true if node_a and node_b are both map targets for nodes from a
  // common source function (or are themselves in the common function).
  absl::StatusOr<bool> NodeSourceFunctionsCollide(const Node* node_a,
                                                  const Node* node_b) const;

  // Returns a vector of Nodes to which the operands of the node
  // 'node' map. If node is owned by the integrated function, these are just
  // node's operands. If an operand does not yet have a mapping, the operand is
  // temporarily mapped to a new parameter(not yet implemented). Use of this
  // temporary will be replaced with the real mapping when it is set.
  absl::StatusOr<std::vector<Node*>> GetIntegratedOperands(
      const Node* node) const;

  // Returns true if 'node' is mapped to a node in the integrated function.
  bool HasMapping(const Node* node) const;

  // Returns true if all operands of 'node' are mapped to a node in the
  // integrated function.
  bool AllOperandsHaveMapping(const Node* node) const;

  // Returns true if other nodes map to 'node'
  bool IsMappingTarget(const Node* node) const;

  // Returns true if 'node' is in the integrated function.
  bool IntegrationFunctionOwnsNode(const Node* node) const {
    return function_.get() == node->function_base();
  }

  // Returns an estimate of the (gate count? area?) cost of a node.
  int64_t GetNodeCost(const Node* node) const;

 private:
  IntegrationFunction(Package* package, const IntegrationOptions& options)
      : package_(package), integration_options_(options) {}

  // Helper function that implements the logic for merging nodes,
  // allowing for either the merge to be performed or for the cost
  // of the merge to be estimated.  A MergeNodesBackendResult struct is
  // returned. The 'can_merge' field indicates if the nodes can be merged. If
  // they can be merged, target_a and target_b point to resulting nodes that
  // represent the values of 'node_a' and 'node_b' in the integrated graph
  // (note: these will not necessarily point to the same merged node e.g. if the
  // merge node has a wider bitwidth than one of the original nodes, the target
  // pointer may instead point to a bitslice that takes in the wider node as an
  // operand). New muxes created or muxes modified by this call are placed in
  // 'changed_muxes'. Other nodes created by this call are placed in
  // 'other_added_nodes'.
  struct MergeNodesBackendResult {
    bool can_merge;
    // Separate targets for node_a and node_b because we may derive
    // map target nodes from the merged node;
    Node* target_a = nullptr;
    Node* target_b = nullptr;
    std::vector<UnifiedNode> changed_muxes;
    // We use a list rather than a vector here because
    // we will later want to remove elements in a (currently)
    // unknown order.  This would involve wastefule data copying
    // if we used a vector.
    std::list<Node*> other_added_nodes;
  };
  absl::StatusOr<MergeNodesBackendResult> MergeNodesBackend(const Node* node_a,
                                                            const Node* node_b);

  // For the integration function nodes node_a and node_b,
  // returns a single integration function node that combines the two
  // nodes. If a node combining node_a and node_b does not already
  // exist, a new mux and a per-mux select parameter are added.
  // If provided, the bool pointed to  by 'new_mux_added' will be
  // set to true if this call added a new mux.  Otherwise, false.
  absl::StatusOr<UnifiedNode> UnifyIntegrationNodesWithPerMuxSelect(
      Node* node_a, Node* node_b);

  // For the integration function nodes node_a and node_b,
  // returns a single integration function node that combines the two
  // nodes. If neither node_a or node_b is a mux added by a previous call,
  // a new mux is added whose select signal global_mux_select_param_.
  // If one of the nodes is such a mux, the other node is added as an
  // input to the mux.
  absl::StatusOr<UnifiedNode> UnifyIntegrationNodesWithGlobalMuxSelect(
      Node* node_a, Node* node_b);

  // Helper function for UnifyIntegrationNodesWithGlobalMuxSelect that handles
  // the case that neither input is a pre-existing mux.
  absl::StatusOr<UnifiedNode> UnifyIntegrationNodesWithGlobalMuxSelectArgIsMux(
      Node* mux, Node* case_node);

  // Helper function for UnifyIntegrationNodesWithGlobalMuxSelect that handles
  // the cases that one of the input nodes is a pre-existing mux. The other
  // input is a node that will be added as a case(s) to the mux.
  absl::StatusOr<UnifiedNode> UnifyIntegrationNodesWithGlobalMuxSelectNoMuxArg(
      Node* mux, Node* case_node);

  // For a mux produced by UnifyIntegrationNodesWithPerMuxSelect, remove the mux
  // and the select parameter. Updates internal unification / mux book-keeping
  // accordingly. This function should only be called if the mux has no users
  // and mux's select signal is only used by the mux.
  absl::Status DeUnifyIntegrationNodesWithPerMuxSelect(Node* node);

  // For a mux produced by UnifyIntegrationNodesWithGlobalMuxSelect,
  // remove the most recently added cases. If no cases are in use after this,
  // then the mux is removed. Otherwise, the revert mux is returned. Also
  // updates internal unification / mux book-keeping.
  absl::StatusOr<Node*> DeUnifyIntegrationNodesWithGlobalMuxSelect(Node* node);

  // Replaces 'mux' with a new mux which is identical except that
  // the the cases at the indexes in 'source_index_to_case'
  // are replaced with the nodes specified.
  absl::StatusOr<Node*> ReplaceMuxCases(
      Node* mux_node,
      const absl::flat_hash_map<int64_t, Node*>& source_index_to_case);

  // Track mapping of original function nodes to integrated function nodes.
  absl::flat_hash_map<const Node*, Node*> original_node_to_integrated_node_map_;
  // Use node_hash_map because we rely on pointer stability in SetNodeMapping.
  absl::node_hash_map<const Node*, absl::flat_hash_set<const Node*>>
      integrated_node_to_original_nodes_map_;

  // Track which node-pairs have an associated mux.
  absl::flat_hash_map<std::pair<const Node*, const Node*>, Node*>
      node_pair_to_mux_;

  // If integration_options_ does not specify that each mux has a unique
  // select signal, this shared parameter is the select for all integration
  // muxes.
  Node* global_mux_select_param_ = nullptr;

  struct GlobalMuxMetadata {
    // Prefer to use sets rather than flat_hash_sets so that
    // the order of indexes reflects the order of the mux inputs.

    // Track which select arms map meaningful nodes for muxes
    // using the global_mux_select_param_ select signal.
    std::set<int64_t> occupied_case_indexes;

    // Track which select arms were most recently added for muxes
    // using the global_mux_select_param_ select signal. Note that
    // we only need to preserve enough history to call DeUnifyIntegrationNodes
    // after calling UnifyIntegrationNodes, with no other calls to
    // UnifyIntegrationNodes for a given mux inbetween. Further, there should
    // not be repeated calls to DeUnifyIntegrationNodes for a given node.
    std::set<int64_t> last_case_indexes_added;
  };
  // Track information about muxes that use global_mux_select_param_ as their
  // select signal.
  absl::flat_hash_map<Node*, GlobalMuxMetadata> global_mux_to_metadata_;

  // Source function in the integration package.
  std::vector<const Function*> source_functions_;

  // Maps each source function to a unique index.
  absl::flat_hash_map<const FunctionBase*, int64_t>
      source_function_base_to_index_;

  // Integrated function.
  std::unique_ptr<Function> function_;
  Package* package_;
  const IntegrationOptions integration_options_;
};

}  // namespace xls

#endif  // XLS_CONTRIB_INTEGRATOR_IR_INTEGRATOR_H_
