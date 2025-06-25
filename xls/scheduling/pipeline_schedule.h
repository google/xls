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
//
// Abstractions for describing the binding of Nodes to cycles.
// PackageSchedule map FunctionBases to a PipelineSchedule, and
// PipelineSchedule maps nodes in a FunctionBase to an integer value
// representing a pipeline stage.

#ifndef XLS_SCHEDULING_PIPELINE_SCHEDULE_H_
#define XLS_SCHEDULING_PIPELINE_SCHEDULE_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/fdo/delay_manager.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

// Abstraction describing the binding of Nodes to cycles for a FunctionBase.
class PipelineSchedule {
 public:
  PipelineSchedule() = default;

  // Reconstructs a PipelineSchedule object from a proto representation.
  static absl::StatusOr<PipelineSchedule> FromProto(
      FunctionBase* function, const PackageScheduleProto& proto);

  // Builds trivial pipeline schedule with all nodes in a single stage
  static absl::StatusOr<PipelineSchedule> SingleStage(FunctionBase* function);

  // Constructs a schedule for the given function with the given cycle map. If
  // length is not given, then the length equal to the largest cycle in cycle
  // map minus one.
  PipelineSchedule(FunctionBase* function_base, ScheduleCycleMap cycle_map,
                   std::optional<int64_t> length = std::nullopt,
                   std::optional<int64_t> min_clock_period_ps = std::nullopt);

  FunctionBase* function_base() const { return function_base_; }

  // Returns whether the given node is contained in this schedule.
  bool IsScheduled(Node* node) const { return cycle_map_.contains(node); }

  // Remove a node from the schedule.
  void RemoveNode(Node* node);

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

  // Returns the minimum possible clock period, if this was computed while
  // creating the schedule. This is purely for tracing purposes.
  const std::optional<int64_t>& min_clock_period_ps() const {
    return min_clock_period_ps_;
  }

  // Verifies various invariants of the schedule (each node scheduled exactly
  // once, node not scheduled before operands, etc.).
  absl::Status Verify() const;

  // Verifies that no path of nodes scheduled in the same cycle exceeds the
  // given clock period.
  absl::Status VerifyTiming(int64_t clock_period_ps,
                            const DelayEstimator& delay_estimator) const;
  absl::Status VerifyTiming(int64_t clock_period_ps,
                            const DelayManager& delay_manager) const;

  // Verifies that all scheduling constraints are followed.
  absl::Status VerifyConstraints(
      absl::Span<const SchedulingConstraint> constraints,
      std::optional<int64_t> worst_case_throughput) const;

  // Returns a protobuf holding this object's scheduling info.
  PipelineScheduleProto ToProto(const DelayEstimator& delay_estimator) const;

  // Returns the number of internal registers in this schedule.
  int64_t CountFinalInteriorPipelineRegisters() const;

  // Returns the underlying cycle map.
  const ScheduleCycleMap& GetCycleMap() const { return cycle_map_; }

 private:
  FunctionBase* function_base_ = nullptr;

  // Map from node to the cycle in which it is scheduled.
  ScheduleCycleMap cycle_map_;

  // The nodes scheduled each cycle.
  std::vector<std::vector<Node*>> cycle_to_nodes_;

  // The minimum possible clock period, if known.
  std::optional<int64_t> min_clock_period_ps_;
};

// A collection of FunctionBase schedules necessary to generate pipelined
// implementation for a Package.
class PackageSchedule {
 public:
  using ScheduleMap = absl::flat_hash_map<FunctionBase*, PipelineSchedule>;

  explicit PackageSchedule(Package* package) : package_(package) {}

  explicit PackageSchedule(PipelineSchedule schedule)
      : package_(schedule.function_base()->package()) {
    schedules_[schedule.function_base()] = std::move(schedule);
  }
  PackageSchedule(Package* package, ScheduleMap schedules,
                  std::optional<absl::flat_hash_map<FunctionBase*, int64_t>>
                      synchronous_offsets = std::nullopt)
      : package_(package),
        schedules_(std::move(schedules)),
        synchronous_offsets_(synchronous_offsets) {}

  // Reconstructs a PackageSchedule object from a proto representation.
  // Will return an error status if the proto schedules reference nodes that
  // don't exist in the package.
  static absl::StatusOr<PackageSchedule> FromProto(
      Package* p, const PackageScheduleProto& proto);

  // Returns whether the given FunctionBase has a schedule.
  bool HasSchedule(FunctionBase* fb) const { return schedules_.contains(fb); }

  const ScheduleMap& GetSchedules() const { return schedules_; }

  // Return the PipelineSchedule for the given FunctionBase.
  PipelineSchedule& GetSchedule(FunctionBase* function_base) {
    return schedules_.at(function_base);
  }
  const PipelineSchedule& GetSchedule(FunctionBase* function_base) const {
    return schedules_.at(function_base);
  }

  // Adds a schedule for the given FunctionBase. Returns an error if there is
  // already a schedule for the given FunctionBase.
  absl::Status AddSchedule(FunctionBase* fb, PipelineSchedule&& schedule) {
    XLS_RET_CHECK(!HasSchedule(fb))
        << absl::StrFormat("`%s` already has a schedule", fb->name());
    XLS_RET_CHECK_EQ(fb, schedule.function_base());
    schedules_[fb] = schedule;
    return absl::OkStatus();
  }

  absl::Status RemoveSchedule(FunctionBase* fb) {
    if (!HasSchedule(fb)) {
      return absl::NotFoundError(
          absl::StrFormat("FunctionBase `%s` has no schedule", fb->name()));
    }
    schedules_.erase(fb);
    if (synchronous_offsets_.has_value()) {
      synchronous_offsets_->erase(fb);
    }
    return absl::OkStatus();
  }

  // Sets the schedule associated with `fb` to `schedule`.
  absl::Status UpdateSchedule(FunctionBase* fb, PipelineSchedule&& schedule) {
    if (HasSchedule(fb)) {
      XLS_RETURN_IF_ERROR(RemoveSchedule(fb));
    }
    return AddSchedule(fb, std::move(schedule));
  }

  // Returns the FunctionBases with a schedule in a stable sort.
  std::vector<FunctionBase*> GetScheduledFunctionBases() const {
    std::vector<FunctionBase*> function_bases;
    for (const auto& [fb, _] : schedules_) {
      function_bases.push_back(fb);
    }
    std::sort(function_bases.begin(), function_bases.end(),
              FunctionBase::NameLessThan);
    return function_bases;
  }

  bool IsSynchronousSchedule() const {
    return synchronous_offsets_.has_value();
  }

  // Returns the global stage the nodes is in within a synchronous schedule of
  // the package. CHECK fails if this is PackageSchedule is not a synchronous
  // schedule.
  int64_t GetSynchronousCycle(Node* node) const {
    CHECK(IsSynchronousSchedule());
    return GetSchedule(node->function_base()).cycle(node) +
           synchronous_offsets_->at(node->function_base());
  }

  void Clear() {
    schedules_.clear();
    synchronous_offsets_.reset();
  }

  PackageScheduleProto ToProto(const DelayEstimator& delay_estimator) const;
  std::string ToString() const;

 private:
  Package* package_ = nullptr;
  ScheduleMap schedules_;

  // If this package represents a synchronous schedule of FunctionBases (procs)
  // in the package, this map contains the stage offset of each function base in
  // the synchronous schedule.
  std::optional<absl::flat_hash_map<FunctionBase*, int64_t>>
      synchronous_offsets_;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_PIPELINE_SCHEDULE_H_
