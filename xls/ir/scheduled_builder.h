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

#ifndef XLS_IR_SCHEDULED_BUILDER_H_
#define XLS_IR_SCHEDULED_BUILDER_H_

#include <cstdint>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {

class ScheduledFunctionBuilder : public FunctionBuilder {
 public:
  ScheduledFunctionBuilder(std::string_view name, Package* package,
                           bool should_verify = true)
      : FunctionBuilder(name, package, ScheduledFunctionTag{}, should_verify) {}
  ~ScheduledFunctionBuilder() override = default;

  // Sets the stage that newly added nodes will be assigned to.
  // Defaults to 0.
  void SetCurrentStage(int64_t stage);
  int64_t GetCurrentStage() const { return current_stage_; }

  // End the current stage and start a new one.
  void EndStage();

  // Applies all collected stage assignments to the function and builds it.
  // The function is created as a scheduled function via the constructor.
  // Hides FunctionBuilder::Build().
  absl::StatusOr<ScheduledFunction*> Build();

  // Build function using given return value.
  // Hides FunctionBuilder::BuildWithReturnValue().
  absl::StatusOr<ScheduledFunction*> BuildWithReturnValue(BValue return_value);

  // Overrides the stage assignment for the given node, without affecting
  // current_stage_. Returns the node for chaining.
  BValue AssignNodeToStage(BValue node, int64_t stage);

 protected:
  void OnNodeAdded(Node* node) override;

 private:
  int64_t current_stage_ = 0;
  absl::flat_hash_map<Node*, int64_t> node_to_stage_;
};

class ScheduledProcBuilder : public ProcBuilder {
 public:
  ScheduledProcBuilder(std::string_view name, Package* package,
                       bool should_verify = true)
      : ProcBuilder(name, package, ScheduledProcTag{}, should_verify) {}
  ScheduledProcBuilder(NewStyleProc tag, std::string_view name,
                       Package* package, bool should_verify = true)
      : ProcBuilder(NewStyleProc{}, name, package, ScheduledProcTag{},
                    should_verify) {}
  ~ScheduledProcBuilder() override = default;

  // Sets the stage that newly added nodes will be assigned to.
  // Defaults to 0.
  void SetCurrentStage(int64_t stage);
  int64_t GetCurrentStage() const { return current_stage_; }

  // End the current stage and start a new one.
  void EndStage();

  // Applies all collected stage assignments to the proc and builds it.
  // The proc is created as a scheduled proc via the constructor.
  // Hides ProcBuilder::Build().
  absl::StatusOr<ScheduledProc*> Build();
  absl::StatusOr<ScheduledProc*> Build(absl::Span<const BValue> next_state);

  // Overrides the stage assignment for the given node, without affecting
  // current_stage_. Returns the node for chaining.
  BValue AssignNodeToStage(BValue node, int64_t stage);

 protected:
  void OnNodeAdded(Node* node) override;

 private:
  int64_t current_stage_ = 0;
  absl::flat_hash_map<Node*, int64_t> node_to_stage_;
};

}  // namespace xls

#endif  // XLS_IR_SCHEDULED_BUILDER_H_
