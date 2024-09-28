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

#include "xls/jit/jit_callbacks.h"

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/jit/jit_runtime.h"

namespace xls {

namespace {
void PerformStringStep(InstanceContext* thiz, char* step_string,
                       std::string* buffer) {
  buffer->append(step_string);
}

void PerformFormatStep(InstanceContext* thiz, JitRuntime* runtime,
                       const uint8_t* proto_data, int64_t proto_data_size,
                       const uint8_t* value, uint64_t format_u64,
                       std::string* buffer) {
  Type* type = thiz->ParseTypeFromProto(
      absl::Span<uint8_t const>(proto_data, proto_data_size));
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  __msan_unpoison(value, runtime->GetTypeByteSize(type));
#endif
  FormatPreference format = static_cast<FormatPreference>(format_u64);
  Value ir_value = runtime->UnpackBuffer(value, type);
  absl::StrAppend(buffer, ir_value.ToHumanString(format));
}

void RecordTrace(InstanceContext* thiz, std::string* buffer, int64_t verbosity,
                 InterpreterEvents* events) {
  events->trace_msgs.push_back(
      TraceMessage{.message = *buffer, .verbosity = verbosity});
  delete buffer;
}
std::string* CreateTraceBuffer(InstanceContext* thiz) {
  return new std::string();
}
void RecordAssertion(InstanceContext* thiz, const char* msg,
                     InterpreterEvents* events) {
  events->assert_msgs.push_back(msg);
}

bool QueueReceiveWrapper(InstanceContext* thiz, int64_t queue_index,
                         uint8_t* buffer) {
  return thiz->channel_queues[queue_index]->ReadRaw(buffer);
}

void QueueSendWrapper(InstanceContext* thiz, int64_t queue_index,
                      const uint8_t* data) {
  thiz->channel_queues[queue_index]->WriteRaw(data);
}

void RecordActiveNextValue(InstanceContext* thiz, int64_t param_id,
                           int64_t next_id) {
  thiz->active_next_values[param_id].insert(next_id);
}

void RecordNodeResult(InstanceContext* thiz, int64_t node_ptr,
                      const uint8_t* data) {
  if (thiz->observer != nullptr) {
    thiz->observer->RecordNodeValue(node_ptr, data);
  }
}
}  // namespace

InstanceContextVTable::InstanceContextVTable()
    : perform_string_step(&PerformStringStep),
      perform_format_step(&PerformFormatStep),
      record_trace(&RecordTrace),
      create_trace_buffer(&CreateTraceBuffer),
      record_assertion(&RecordAssertion),
      queue_receive_wrapper(&QueueReceiveWrapper),
      queue_send_wrapper(&QueueSendWrapper),
      record_active_next_value(&RecordActiveNextValue),
      record_node_result(&RecordNodeResult) {}

Type* InstanceContext::ParseTypeFromProto(absl::Span<uint8_t const> data) {
  TypeProto proto;
  CHECK(proto.ParseFromArray(data.data(), data.size()));
  auto type_or = type_manager->GetTypeFromProto(proto);
  CHECK_OK(type_or);
  return *type_or;
}
}  // namespace xls
