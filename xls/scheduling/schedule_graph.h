// Copyright 2025 The XLS Authors
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

#ifndef XLS_SCHEDULING_SCHEDULE_GRAPH_H_
#define XLS_SCHEDULING_SCHEDULE_GRAPH_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/proc_elaboration.h"

namespace xls {

// Returns true if `node` is untimed, i.e., it should not be a subject of the
// scheduling problem and exists outside the scope of the schedule. For now,
// it's just literals, but it could be other notions of "constant" or
// "time-invariant" nodes, like a runtime constant.
bool IsUntimed(Node* node);

// The ScheduleGraph abstracts the information required by the scheduler from
// the IR.

struct ScheduleNode {
  Node* node;

  // The set of immediate predecessors of the node in the graph (e.g., IR node
  // operands).
  std::vector<Node*> predecessors;

  // The set of immediate successors of the node in the graph (e.g., IR node
  // users).
  std::vector<Node*> successors;

  // Whether this node will be dead after synthesis. Specifically the only
  // effect of the node is to compute a value that is used for asserts, covers,
  // or traces.
  bool is_dead_after_synthesis;

  // Whether the node must be scheduled in the first/last stage.
  bool schedule_in_first_stage;
  bool schedule_in_last_stage;

  // Whether the node is live in/out of the graph. Nodes which are live in (out)
  // of the graph and are not scheduled in the first (last) stage require
  // additional pipeline registers to carry the value from the first (last)
  // stage to the stage in which the node is scheduled.
  bool is_live_in;
  bool is_live_out;
};

struct LessThanInitiationInterval : public std::monostate {};

// A ScheduleBackedge represents a backwards-flowing edge in the data flow graph
// such proc state. These edges often impose additional constraints on the
// schedule.
struct ScheduleBackedge {
  Node* source;
  Node* destination;

  // If specified, this is a constraint on the distance in stages between
  // `source` and `destination` in the schedule. Possible values:
  //
  //    int64_t: `source` must be exactly this number of stages later than
  //      `destination`
  //
  //    LessThanInitiationInterval: `source` must be scheduled less than the
  //      initiation interval stages after `destination`
  //
  // If not specified then no constraint on the distance between `source` and
  // `destination`.
  std::optional<std::variant<int64_t, LessThanInitiationInterval>> distance;
};

class ScheduleGraph {
 public:
  // Return a ScheduleGraph representing a single FunctionBase (Function or
  // Proc).
  static ScheduleGraph Create(
      FunctionBase* f, const absl::flat_hash_set<Node*>& dead_after_synthesis);

  // Return a ScheduleGraph representing the procs in `elab` which can be used
  // for synchronous scheduling. Non-loopback channels are treated as regular
  // dataflow edges. Loopback channels and state are represented using
  // ScheduleBackedge in the returned graph (though they need not technically be
  // backedges in the dataflow graph).
  static absl::StatusOr<ScheduleGraph> CreateSynchronousGraph(
      Package* p, absl::Span<Channel* const> loopback_channels,
      const ProcElaboration& elab,
      const absl::flat_hash_set<Node*>& dead_after_synthesis);

  std::string_view name() const { return name_; }

  // Returns the nodes of the graph in topological sorted order.
  absl::Span<const ScheduleNode> nodes() const { return nodes_; }

  absl::Span<const ScheduleBackedge> backedges() const { return backedges_; }

  // The IR scope which the ScheduleGraph represents.
  std::variant<Package*, FunctionBase*> ir_scope() const { return ir_scope_; }

  bool IsFunctionBaseScoped() const {
    return std::holds_alternative<FunctionBase*>(ir_scope_);
  }

  // Returns the ScheduleNode representing `node`. CHECK fails if `node` is
  // outside the scope of the graph.
  const ScheduleNode& GetScheduleNode(Node* node) const {
    auto it = node_map_.find(node);
    CHECK(it != node_map_.end()) << node;
    return nodes_[it->second];
  }

  // Whether this graph represents IR which includes a proc.
  bool IncludesProc() const {
    return std::holds_alternative<Package*>(ir_scope_) ||
           (std::holds_alternative<FunctionBase*>(ir_scope_) &&
            std::get<FunctionBase*>(ir_scope_)->IsProc());
  }

  // Whether this graph represents a single proc.
  bool IsSingleProc() const {
    return std::holds_alternative<FunctionBase*>(ir_scope_) &&
           std::get<FunctionBase*>(ir_scope_)->IsProc();
  }

  std::string ToString() const;

 private:
  ScheduleGraph(std::string_view name,
                std::variant<Package*, FunctionBase*> ir_scope,
                std::vector<ScheduleNode> nodes,
                std::vector<ScheduleBackedge> backedges)
      : name_(name),
        ir_scope_(ir_scope),
        nodes_(std::move(nodes)),
        backedges_(std::move(backedges)) {
    node_map_.reserve(nodes_.size());
    for (int64_t i = 0; i < nodes_.size(); ++i) {
      node_map_[nodes_[i].node] = i;
    }
  }

  std::string name_;
  std::variant<Package*, FunctionBase*> ir_scope_;
  std::vector<ScheduleNode> nodes_;
  std::vector<ScheduleBackedge> backedges_;

  // Map from xls::Node to its respective index in `nodes_`.
  absl::flat_hash_map<Node*, int64_t> node_map_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULE_GRAPH_H_
