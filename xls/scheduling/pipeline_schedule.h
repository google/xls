// Copyright 2022 The XLS Authors
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

#ifndef XLS_SCHEDULING_PIPELINE_SCHEDULE_H_
#define XLS_SCHEDULING_PIPELINE_SCHEDULE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/proc.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

// Abstraction describing the binding of Nodes to cycles.
class PipelineSchedule {
 public:
  // Produces a feed-forward pipeline schedule using the given delay model and
  // scheduling options.
  static absl::StatusOr<PipelineSchedule> Run(
      FunctionBase* f, const DelayEstimator& delay_estimator,
      const SchedulingOptions& options);

  // Reconstructs a PipelineSchedule object from a proto representation.
  static absl::StatusOr<PipelineSchedule> FromProto(
      FunctionBase* function, const PipelineScheduleProto& proto);

  // Constructs a schedule for the given function with the given cycle map. If
  // length is not given, then the length equal to the largest cycle in cycle
  // map minus one.
  PipelineSchedule(FunctionBase* function_base, ScheduleCycleMap cycle_map,
                   std::optional<int64_t> length = absl::nullopt);

  FunctionBase* function_base() const { return function_base_; }

  // Returns whether the given node is contained in this schedule.
  bool IsScheduled(Node* node) const { return cycle_map_.contains(node); }

  // Returns the cycle in which the node is placed. Dies if node has not
  // been placed in this schedule.
  int64_t cycle(const Node* node) const { return cycle_map_.at(node); }

  // Returns the nodes scheduled in the given cycle. The node order is
  // guaranteed to be topological.
  absl::Span<Node* const> nodes_in_cycle(int64_t cycle) const;

  std::string ToString() const;

  // Computes and returns the set of Nodes which are live out of the given
  // cycle. A node is live out of cycle N if it is scheduled at or before cycle
  // N and has users after cycle N.
  std::vector<Node*> GetLiveOutOfCycle(int64_t c) const;

  // Returns true if the given node is live out of the given cycle.
  bool IsLiveOutOfCycle(Node* node, int64_t c) const;

  // Returns the number of stages in the pipeline. Use 'length' instead of
  // 'size' as 'size' is ambiguous in this context (number of resources? number
  // of nodes? number of cycles?). Note that codegen may add flops to the input
  // or output of the pipeline so this value may not be the same as the latency
  // of the pipeline.
  int64_t length() const { return cycle_to_nodes_.size(); }

  // Verifies various invariants of the schedule (each node scheduled exactly
  // once, node not scheduled before operands, etc.).
  absl::Status Verify() const;

  // Verifies that no path of nodes scheduled in the same cycle exceeds the
  // given clock period.
  absl::Status VerifyTiming(int64_t clock_period_ps,
                            const DelayEstimator& delay_estimator) const;

  // Returns a protobuf holding this object's scheduling info.
  PipelineScheduleProto ToProto() const;

  // Returns the number of internal registers in this schedule.
  int64_t CountFinalInteriorPipelineRegisters() const;

 private:
  FunctionBase* function_base_;

  // Map from node to the cycle in which it is scheduled.
  ScheduleCycleMap cycle_map_;

  // The nodes scheduled each cycle.
  std::vector<std::vector<Node*>> cycle_to_nodes_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_PIPELINE_SCHEDULE_H_
