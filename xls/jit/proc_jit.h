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

#ifndef XLS_JIT_PROC_JIT_H_
#define XLS_JIT_PROC_JIT_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_evaluator_options.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {

// This class provides a facility to execute XLS procs (on the host) by
// converting them to LLVM IR, compiling it, and finally executing it.
class ProcJit : public ProcEvaluator {
 public:
  // Returns an object containing a host-compiled version of the specified XLS
  // proc.
  static absl::StatusOr<std::unique_ptr<ProcJit>> Create(
      Proc* proc, JitRuntime* jit_runtime, JitChannelQueueManager* queue_mgr,
      const EvaluatorOptions& options = EvaluatorOptions(),
      const JitEvaluatorOptions& jit_options = JitEvaluatorOptions());

  static absl::StatusOr<std::unique_ptr<ProcJit>> CreateFromAot(
      Proc* proc, JitRuntime* jit_runtime, JitChannelQueueManager* queue_mgr,
      const AotEntrypointProto& entrypoint, JitFunctionType unpacked,
      std::optional<JitFunctionType> packed = std::nullopt,
      const EvaluatorOptions& options = EvaluatorOptions());

  ~ProcJit() override = default;

  std::unique_ptr<ProcContinuation> NewContinuation(
      ProcInstance* proc_instance) const override;
  absl::StatusOr<TickResult> Tick(
      ProcContinuation& continuation) const override;

  JitRuntime* runtime() const { return jit_runtime_; }

  OrcJit& GetOrcJit() { return *orc_jit_; }

 private:
  explicit ProcJit(Proc* proc, JitRuntime* jit_runtime,
                   JitChannelQueueManager* queue_mgr,
                   std::unique_ptr<OrcJit> orc_jit, bool has_observer_callbacks,
                   const EvaluatorOptions& options)
      : ProcEvaluator(proc, options),
        jit_runtime_(jit_runtime),
        queue_mgr_(queue_mgr),
        orc_jit_(std::move(orc_jit)),
        has_observer_callbacks_(has_observer_callbacks) {}

  JitRuntime* jit_runtime_;
  JitChannelQueueManager* queue_mgr_;
  std::unique_ptr<OrcJit> orc_jit_;
  JittedFunctionBase jitted_function_base_;
  // We need to have compiled in the callbacks in order to support the
  // Evaluation/RuntimeObserver apis.
  bool has_observer_callbacks_;

  // The set of channel queues used in each proc instance. The vector is in a
  // predetermined order assigned at JIT compile time. The JITted code looks for
  // the queue at a particular index.
  absl::flat_hash_map<ProcInstance*, std::vector<JitChannelQueue*>>
      channel_queues_;
};

}  // namespace xls

#endif  // XLS_JIT_PROC_JIT_H_
