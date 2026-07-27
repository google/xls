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

#ifndef XLS_INTERPRETER_TRACE_RECORDER_H_
#define XLS_INTERPRETER_TRACE_RECORDER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "riegeli/records/record_writer.h"
#include "xls/interpreter/trace.pb.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"

namespace xls {

// Class for recording trace events during IR evaluation.
class TraceRecorder {
 public:
  explicit TraceRecorder(riegeli::RecordWriterBase& writer);

  // Records a NodeValue event.
  absl::Status RecordNodeValue(Node* node, const xls::Value& value);

  // Increments the simulation time by one tick.
  void Tick() { tick_++; }

 private:
  riegeli::RecordWriterBase& writer_;
  absl::flat_hash_set<int64_t> seen_node_ids_;
  int64_t tick_ = 0;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_TRACE_RECORDER_H_
