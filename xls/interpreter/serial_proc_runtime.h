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

#ifndef XLS_INTERPRETER_SERIAL_PROC_RUNTIME_H_
#define XLS_INTERPRETER_SERIAL_PROC_RUNTIME_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/ir/package.h"

namespace xls {

// Class for interpreting a network of procs. Simultaneously interprets all
// procs in a package handling all interproc communication via a channel queues.
// SerialProcRuntimes are thread-compatible, but not thread-safe.
class SerialProcRuntime : public ProcRuntime {
 public:
  // Creates and returns an proc network interpreter for the given
  // evaluators.
  static absl::StatusOr<std::unique_ptr<SerialProcRuntime>> Create(
      std::vector<std::unique_ptr<ProcEvaluator>>&& evaluators,
      std::unique_ptr<ChannelQueueManager>&& queue_manager,
      const EvaluatorOptions& options = EvaluatorOptions());

 private:
  SerialProcRuntime(
      absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>>&& evaluators,
      std::unique_ptr<ChannelQueueManager>&& queue_manager,
      const EvaluatorOptions& options = EvaluatorOptions())
      : ProcRuntime(std::move(evaluators), std::move(queue_manager), options) {}

  absl::StatusOr<SerialProcRuntime::NetworkTickResult> TickInternal() override;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_SERIAL_PROC_RUNTIME_H_
