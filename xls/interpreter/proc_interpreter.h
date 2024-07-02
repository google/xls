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

#ifndef XLS_INTERPRETER_PROC_INTERPRETER_H_
#define XLS_INTERPRETER_PROC_INTERPRETER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"

namespace xls {

// A interpreter for an individual proc. Incrementally executes Procs a single
// tick at a time. Data is fed to the proc via ChannelQueues.  ProcInterpreters
// are thread-safe if called with different continuations.
class ProcInterpreter : public ProcEvaluator {
 public:
  ProcInterpreter(Proc* proc, ChannelQueueManager* queue_manager);
  ProcInterpreter(const ProcInterpreter&) = delete;
  ProcInterpreter operator=(const ProcInterpreter&) = delete;

  ~ProcInterpreter() override = default;

  std::unique_ptr<ProcContinuation> NewContinuation(
      ProcInstance* proc_instance) const override;
  absl::StatusOr<TickResult> Tick(
      ProcContinuation& continuation) const override;

 private:
  ChannelQueueManager* queue_manager_;

  // A topological sort of the nodes of the proc which determines the execution
  // order of the proc.
  std::vector<Node*> execution_order_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_INTERPRETER_H_
