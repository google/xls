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
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/orc_jit.h"

namespace xls {

// This class provides a facility to execute XLS procs (on the host) by
// converting them to LLVM IR, compiling it, and finally executing it.
class ProcJit {
 public:
  // Function types for send and receive actions. The caller should provide
  // callables of this type.
  //
  // The receive function has the following prototype:
  //
  // void recv_fn(JitChannelQueue* queue, uint64_t recv, uint8_t* buffer,
  //              int64_t data_sz, void* user_data);
  // where:
  //  - queue is a pointer to a JitChannelQueue,
  //  - recv is a pointer to a Receive node,
  //  - buffer is a pointer to the data to fill (with incoming data), and
  //  - data_sz is the size of the receive buffer.
  //  - user_data is an opaque pointer to user-provided data needed for
  //    processing, e.g., thread/queue info.
  //
  // The send function has the following prototype:
  // void send_fn(uint64_t queue_ptr, uint64_t send_ptr, uint8_t* buffer,
  //              int64_t data_sz, void* user_data);
  // where:
  //  - queue_ptr is a pointer to a JitChannelQueue,
  //  - send_ptr is a pointer to a Send node,
  //  - buffer is a pointer to the data to fill (with incoming data), and
  //  - data_sz is the size of the receive buffer.
  //  - user_data is an opaque pointer to user-provided data needed for
  //    processing, e.g., thread/queue info.
  using RecvFnT = bool (*)(JitChannelQueue*, Receive*, uint8_t*, int64_t,
                           void*);
  using SendFnT = void (*)(JitChannelQueue*, Send*, uint8_t*, int64_t, void*);

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
  absl::Status RunWithViews(absl::Span<uint8_t const* const> state,
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

  LlvmTypeConverter* type_converter() { return &orc_jit_->GetTypeConverter(); }

  OrcJit& GetOrcJit() { return *orc_jit_; }

 private:
  explicit ProcJit(Proc* proc);

  // Builds a function which wraps the natively compiled XLS proc `callee` with
  // another function which accepts the input arguments as an array of pointers
  // to buffers and the output as a pointer to a buffer. The input/output values
  // are in the native LLVM data layout. The function signature is:
  //
  //   void f(uint8_t*[] state, uint8_t* next_state,
  //          void* events, void* user_data, void* jit_runtime)
  //
  // `state` is an array containing a pointer for each state element argument.
  // The pointer points to a buffer containing the respective argument in the
  // native LLVM data layout.
  //
  // `next_state` points to an empty buffer appropriately sized to accept the
  // next_state value in the native LLVM data layout.
  absl::StatusOr<llvm::Function*> BuildWrapper(llvm::Function* callee);

  std::unique_ptr<OrcJit> orc_jit_;

  Proc* proc_;

  std::unique_ptr<JitRuntime> ir_runtime_;

  // When initialized, this points to the compiled output.
  using JitFunctionType = void (*)(const uint8_t* const* state,
                                   uint8_t* const* next_state,
                                   InterpreterEvents* events, void* user_data,
                                   JitRuntime* jit_runtime);
  JitFunctionType invoker_;
};

}  // namespace xls

#endif  // XLS_JIT_PROC_JIT_H_
