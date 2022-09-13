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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/events.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {

// This class provides a facility to execute XLS procs (on the host) by
// converting them to LLVM IR, compiling it, and finally executing it.
class ProcJit {
 public:
  // Returns an object containing a host-compiled version of the specified XLS
  // proc.
  static absl::StatusOr<std::unique_ptr<ProcJit>> Create(
      Proc* proc, JitChannelQueueManager* queue_mgr, RecvFnT recv_fn,
      SendFnT send_fn, int64_t opt_level = 3);

  // Executes a single tick of the compiled proc. `state` are the initial state
  // values. The returned values are the next state value.  The optional opaque
  // "user_data" argument is passed into Proc send/recv callbacks. Returns both
  // the resulting value and events that happened during evaluation.
  // TODO(meheff) 2022/05/24 Add way to run with packed values.
  absl::StatusOr<InterpreterResult<std::vector<Value>>> Run(
      absl::Span<const Value> state, void* user_data = nullptr);

  // Executes the compiled proc with the given state and updates the
  // next state view.
  //
  // "views" - flat buffers onto which structures layouts can be applied (see
  // value_view.h).
  absl::Status RunWithViews(absl::Span<const uint8_t* const> state,
                            absl::Span<uint8_t* const> next_state,
                            void* user_data = nullptr);

  // Converts the state respresented as xls Values to the native LLVM data
  // layout.
  //  - If initialize_with_value is false, then the native LLVM data
  //   is only allocated and not initialized.
  absl::StatusOr<std::vector<std::vector<uint8_t>>> ConvertStateToView(
      absl::Span<const Value> state_value, bool initialize_with_value = true);

  // Convert the state represented in native LLVM data layout to xls Values.
  std::vector<Value> ConvertStateViewToValue(
      absl::Span<uint8_t const* const> state_buffers);

  // Returns the function that the JIT executes.
  Proc* proc() { return proc_; }

  JitRuntime* runtime() { return ir_runtime_.get(); }

  OrcJit& GetOrcJit() { return *orc_jit_; }

  LlvmTypeConverter* type_converter() const {
    return &orc_jit_->GetTypeConverter();
  }

 private:
  explicit ProcJit(Proc* proc) : proc_(proc) {}

  Proc* proc_;

  std::unique_ptr<OrcJit> orc_jit_;

  JittedFunctionBase jitted_function_base_;

  // Buffers to hold inputs, outputs, and temporary storage. This is allocated
  // once and then re-used with each invocation of Run. Not thread-safe.
  std::vector<std::vector<uint8_t>> input_buffers_;
  std::vector<std::vector<uint8_t>> output_buffers_;
  std::vector<uint8_t> temp_buffer_;

  // Raw pointers to the buffers held in `input_buffers_` and `output_buffers_`.
  std::vector<uint8_t*> input_ptrs_;
  std::vector<uint8_t*> output_ptrs_;

  std::unique_ptr<JitRuntime> ir_runtime_;
};

}  // namespace xls

#endif  // XLS_JIT_PROC_JIT_H_
