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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/events.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {

// A continuation used by the ProcJit. Stores control and data state of proc
// execution for the JIT.
class ProcJitContinuation : public ProcContinuation {
 public:
  // Construct a new continuation. Execution the proc begins with the state set
  // to its initial values with no proc nodes yet executed. `temp_buffer_size`
  // specifies the size of a flat buffer used to hold temporary xls::Node values
  // during execution of the JITed function. The size of the buffer is
  // determined at JIT compile time and known by the ProcJit.
  explicit ProcJitContinuation(Proc* proc, int64_t temp_buffer_size,
                               JitRuntime* jit_runtime);

  ~ProcJitContinuation() override = default;

  std::vector<Value> GetState() const override;
  const InterpreterEvents& GetEvents() const override { return events_; }
  InterpreterEvents& GetEvents() override { return events_; }
  bool AtStartOfTick() const override { return continuation_point_ == 0; }

  // Get/Set the point at which execution will resume in the proc in the next
  // call to Tick.
  int64_t GetContinuationPoint() const { return continuation_point_; }
  void SetContinuationPoint(int64_t value) { continuation_point_ = value; }

  // Return the various buffers passed to the top-level function implementing
  // the proc.
  absl::Span<uint8_t*> GetInputBuffers() { return absl::MakeSpan(input_ptrs_); }
  absl::Span<uint8_t*> GetOutputBuffers() {
    return absl::MakeSpan(output_ptrs_);
  }
  absl::Span<uint8_t> GetTempBuffer() { return absl::MakeSpan(temp_buffer_); }

  // Sets the continuation to resume execution at the entry of the proc. Updates
  // state to the "next" value computed in the previous tick.
  void NextTick();

  Proc* proc() const { return proc_; }

 private:
  Proc* proc_;
  int64_t continuation_point_;
  JitRuntime* jit_runtime_;

  InterpreterEvents events_;

  // Buffers to hold inputs, outputs, and temporary storage. This is allocated
  // once and then re-used with each invocation of Run. Not thread-safe.
  std::vector<std::vector<uint8_t>> input_buffers_;
  std::vector<std::vector<uint8_t>> output_buffers_;

  // Raw pointers to the buffers held in `input_buffers_` and `output_buffers_`.
  std::vector<uint8_t*> input_ptrs_;
  std::vector<uint8_t*> output_ptrs_;
  std::vector<uint8_t> temp_buffer_;
};

// This class provides a facility to execute XLS procs (on the host) by
// converting them to LLVM IR, compiling it, and finally executing it.
class ProcJit : public ProcEvaluator {
 public:
  // Returns an object containing a host-compiled version of the specified XLS
  // proc.
  static absl::StatusOr<std::unique_ptr<ProcJit>> Create(
      Proc* proc, JitRuntime* jit_runtime, JitChannelQueueManager* queue_mgr);

  virtual ~ProcJit() = default;

  std::unique_ptr<ProcContinuation> NewContinuation() const override;
  absl::StatusOr<TickResult> Tick(
      ProcContinuation& continuation) const override;
  Proc* proc() const override { return proc_; }

  JitRuntime* runtime() const { return jit_runtime_; }

  OrcJit& GetOrcJit() { return *orc_jit_; }

 private:
  explicit ProcJit(Proc* proc, JitRuntime* jit_runtime,
                   std::unique_ptr<OrcJit> orc_jit)
      : proc_(proc), jit_runtime_(jit_runtime), orc_jit_(std::move(orc_jit)) {}

  Proc* proc_;
  JitRuntime* jit_runtime_;
  std::unique_ptr<OrcJit> orc_jit_;
  JittedFunctionBase jitted_function_base_;
};

}  // namespace xls

#endif  // XLS_JIT_PROC_JIT_H_
