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

#include <deque>
#include <limits>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/channel.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {

// A stateful interpreter for an individual proc. Incrementally executes Procs a
// single iteration at a time. Data is fed to the proc via
// ChannelQueues. ProcInterpreters are thread-compatible, but not thread-safe.
class ProcInterpreter {
 public:
  ProcInterpreter(Proc* proc, ChannelQueueManager* queue_manager);

  ProcInterpreter(const ProcInterpreter&) = delete;
  ProcInterpreter operator=(const ProcInterpreter&) = delete;

  // Advances the proc interpreter to the next iteration. This method must be
  // called prior to calling RunIterationUntilCompleteOrBlocked() for the first
  // time. An error is returned if the the interpreter has not completed the
  // previous iteration.
  // absl::Status NextIteration();

  // Data structure holding the result of a single call to
  // RunIterationUntilCompleteOrBlocked.
  struct RunResult {
    // Whether the proc completed executing for this iteration. Execution is not
    // completed if (and only if) a receive operation is blocked.
    bool iteration_complete;

    // Whether any progress was made (at least one instruction was executed).
    bool progress_made;

    // If the proc is blocked on receive (iteration complete is false), this
    // vector includes the channels which have blocked operations.
    std::vector<Channel*> blocked_channels;

    bool operator==(const RunResult& other) const;
    bool operator!=(const RunResult& other) const;

    std::string ToString() const;
  };

  // Runs the proc until the iteration is complete or execution is blocked on a
  // receive operation. If the previous call to this method completed the
  // iteration (the returned RunResult field iteration_complete was true), a new
  // iteration of the proc will be initiated.
  absl::StatusOr<RunResult> RunIterationUntilCompleteOrBlocked();

  // Whether the previous call to RunUntilIterationCompleteOrBlocked completed
  // an iteration of the proc (the returned RunResult field iteration_complete
  // was true).
  bool IsIterationComplete() const;

  Proc* proc() { return proc_; }

  // Returns the computed next state values. Returns NotFoundError if not all
  // next state values have been computed (due to a blocked channel).
  absl::StatusOr<std::vector<Value>> ResolveState() const;

  // Resets the proc state elements to their initial values.
  void ResetState();

 private:
  Proc* proc_;
  ChannelQueueManager* queue_manager_;

  // A topological sort of the nodes of the proc.
  NodeIterator topo_sort_;

  // A monotonically increasing value holding the number of complete iterations
  // that the proc has executed.
  int64_t current_iteration_;

  // The interpreter used for evaluating nodes in the proc.
  std::unique_ptr<IrInterpreter> visitor_;
};

std::ostream& operator<<(std::ostream& os,
                         const ProcInterpreter::RunResult& result);

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_INTERPRETER_H_
