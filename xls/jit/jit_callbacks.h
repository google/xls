// Copyright 2024 The XLS Authors
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

#ifndef XLS_JIT_JIT_CALLBACKS_H_
#define XLS_JIT_JIT_CALLBACKS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xls/ir/events.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/type.h"
#include "xls/ir/type_manager.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/observer.h"

namespace xls {

struct InstanceContext;
// Manual vtable of an InstanceContext. Called directly from LLVM jit code.
//
// TODO(allight): Instead of using this Vtable passed as an argument we could
// use the ORC-jit dynamic linking functionality we use to implement MSAN
// support. Unfortunately that would require us to ensure that symbols are
// exported correctly which can be tricky. This way we don't need to deal with
// the linker at all.
struct InstanceContextVTable {
 public:
  explicit InstanceContextVTable();

  using PerformStringStepFn = void (*)(InstanceContext* thiz, char* step_string,
                                       std::string* buffer);
  // This is a shim to let JIT code add a new trace fragment to an existing
  // trace buffer.
  const PerformStringStepFn perform_string_step;
  using PerformFormatStepFn =
      void (*)(InstanceContext* thiz, JitRuntime* runtime,
               const uint8_t* type_proto_data, int64_t type_proto_data_size,
               const uint8_t* value, uint64_t format_u64, std::string* buffer);
  const PerformFormatStepFn perform_format_step;

  using RecordTraceFn = void (*)(InstanceContext* thiz, std::string* buffer,
                                 int64_t verbosity, InterpreterEvents* events);
  // This a shim to let JIT code record a completed trace as an interpreter
  // event.
  const RecordTraceFn record_trace;

  using CreateTraceBufferFn = std::string* (*)(InstanceContext * thiz);
  // This is a shim to let JIT code create a buffer for accumulating trace
  // fragments.
  const CreateTraceBufferFn create_trace_buffer;

  using RecordAssertionFn = void (*)(InstanceContext* thiz, const char* msg,
                                     InterpreterEvents* events);
  // This a shim to let JIT code record an assertion failure as an interpreter
  // event.
  const RecordAssertionFn record_assertion;

  using QueueReceiveWrapperFn = bool (*)(InstanceContext* thiz,
                                         int64_t queue_index, uint8_t* buffer);
  const QueueReceiveWrapperFn queue_receive_wrapper;

  using QueueSendWrapperFn = void (*)(InstanceContext* instance_context,
                                      int64_t queue_index, const uint8_t* data);
  const QueueSendWrapperFn queue_send_wrapper;

  using RecordActiveNextValueFn = void (*)(InstanceContext* thiz,
                                           int64_t param_id, int64_t next_id);
  // This is a shim to let JIT code record the activation of a `next_value`
  // node.
  const RecordActiveNextValueFn record_active_next_value;

  using RecordNodeResultFn = void (*)(InstanceContext* thiz, int64_t node_ptr,
                                      const uint8_t* data);
  // This is a shim to let JIT record what data was recorded for each node. The
  // 'node_ptr' is the constant pointer value of the node at the time the jitted
  // code was created. Whether it is a valid pointer to dereference depends on
  // execution context. It will be globally unique and consistent however.
  //
  // Data is in JIT data format and can be read using the appropriate type
  // information for the node.
  const RecordNodeResultFn record_node_result;

  using AllocateBufferFn = void* (*)(InstanceContext * thiz, int64_t byte_size,
                                     int64_t alignment);
  // This is a shim to let the JIT allocate a large buffer on the heap for cases
  // where its required to avoid blowing out the stack.
  const AllocateBufferFn allocate_buffer;

  using DeallocateBufferFn = void (*)(InstanceContext* thiz, void* buffer);
  // This is a shim to let the JIT deallocate the buffer returned by
  // allocate_buffer.
  const DeallocateBufferFn deallocate_buffer;

  using RecordActiveRegisterWriteFn = void (*)(InstanceContext* thiz,
                                               int64_t register_no,
                                               int64_t register_write_no);
  // This is a shim to let JIT code record the activation of a register write
  // node.
  const RecordActiveRegisterWriteFn record_active_register_write;
};

// Data structure passed to the JITted function which contains instance-specific
// execution-relevant information and function pointers. Used for JITted procs.
struct InstanceContext {
 public:
  static InstanceContext CreateForFunc() { return InstanceContext(); }
  static InstanceContext CreateForBlock() { return InstanceContext(); }
  static InstanceContext CreateForProc(ProcInstance* inst,
                                       std::vector<JitChannelQueue*> queues) {
    InstanceContext ret;
    ret.instance = inst;
    ret.channel_queues = std::move(queues);
    return ret;
  }

  // Offsets in the vtable the LLVM can use to grab the actual function pointer.
  static constexpr int64_t kPerformStringStepOffset =
      offsetof(InstanceContextVTable, perform_string_step);
  static constexpr int64_t kPerformFormatStepOffset =
      offsetof(InstanceContextVTable, perform_format_step);
  static constexpr int64_t kRecordTraceOffset =
      offsetof(InstanceContextVTable, record_trace);
  static constexpr int64_t kCreateTraceBufferOffset =
      offsetof(InstanceContextVTable, create_trace_buffer);
  static constexpr int64_t kRecordAssertionOffset =
      offsetof(InstanceContextVTable, record_assertion);
  static constexpr int64_t kQueueReceiveWrapperOffset =
      offsetof(InstanceContextVTable, queue_receive_wrapper);
  static constexpr int64_t kQueueSendWrapperOffset =
      offsetof(InstanceContextVTable, queue_send_wrapper);
  static constexpr int64_t kRecordActiveNextValueOffset =
      offsetof(InstanceContextVTable, record_active_next_value);
  static constexpr int64_t kRecordNodeResultOffset =
      offsetof(InstanceContextVTable, record_node_result);
  static constexpr int64_t kAllocateBufferOffset =
      offsetof(InstanceContextVTable, allocate_buffer);
  static constexpr int64_t kDeallocateBufferOffset =
      offsetof(InstanceContextVTable, deallocate_buffer);
  static constexpr int64_t kRecordActiveRegisterWrite =
      offsetof(InstanceContextVTable, record_active_register_write);
  static constexpr int64_t kVTableLength = 12;
  using VTableArrayType = std::array<void (*)(), kVTableLength>;

  static constexpr bool IsVtableOffset(int64_t v) {
    return v == kPerformFormatStepOffset || v == kPerformStringStepOffset ||
           v == kRecordTraceOffset || v == kCreateTraceBufferOffset ||
           v == kRecordAssertionOffset || v == kQueueReceiveWrapperOffset ||
           v == kQueueSendWrapperOffset || v == kRecordActiveNextValueOffset ||
           v == kRecordNodeResultOffset || v == kAllocateBufferOffset ||
           v == kDeallocateBufferOffset || v == kRecordActiveRegisterWrite;
  }

  Type* ParseTypeFromProto(absl::Span<uint8_t const> data);

  InstanceContextVTable vtable;

  // The proc instance being evaluated (if we are evaluating a proc).
  ProcInstance* instance = nullptr;

  // The active next values for each parameter (if we are evaluating a proc).
  // Map from node-id of the param to node-id of the next
  absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>> active_next_values;

  // The active register writes for each register (if we are evaluating a
  // block). Map from register index (in Block::GetRegisters) to register write
  // index (in Block::GetRegisterWrites). This is only recorded for registers
  // which have multiple writes.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> active_register_writes;

  // The channel queues used by the proc instance. The order of queues is
  // assigned at JIT compile time. The indices of particular queues is baked
  // into the JITted code for sends and receives.
  std::vector<JitChannelQueue*> channel_queues;

  // Arena used to materialize types that are passed to callbacks.
  std::unique_ptr<TypeManager> type_manager = std::make_unique<TypeManager>();

  RuntimeObserver* observer = nullptr;
};

static_assert(offsetof(InstanceContext, vtable) == 0);
static_assert(sizeof(InstanceContextVTable) ==
              sizeof(InstanceContext::VTableArrayType));

}  // namespace xls

#endif  // XLS_JIT_JIT_CALLBACKS_H_
