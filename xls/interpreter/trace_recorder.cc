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

#include "xls/interpreter/trace_recorder.h"

#include "google/protobuf/timestamp.pb.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/util/time_util.h"
#include "riegeli/records/record_writer.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/trace.pb.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_value.pb.h"

namespace xls {

TraceRecorder::TraceRecorder(riegeli::RecordWriterBase& writer)
    : writer_(writer) {}

absl::Status TraceRecorder::RecordNodeValue(Node* node, const Value& value) {
  google::protobuf::Timestamp now = google::protobuf::util::TimeUtil::GetCurrentTime();
  auto [it, inserted] = seen_node_ids_.insert(node->id());
  if (inserted) {
    // Add a node id to name mapping packet.
    TracePacketProto name_packet;
    NodeIdNameMappingProto* node_id_name_mapping =
        name_packet.mutable_node_id_name_mapping();
    node_id_name_mapping->set_name(
        absl::StrCat(node->function_base()->name(), ".", node->GetName()));
    node_id_name_mapping->set_id(node->id());
    XLS_RET_CHECK(writer_.WriteRecord(name_packet));
  }
  TracePacketProto packet;
  NodeTraceProto* node_value = packet.mutable_node_value();
  node_value->set_node_id(node->id());

  TimeProto* time = node_value->mutable_time();
  *time->mutable_wall_time() = now;
  if (node->function_base()->IsProc()) {
    time->mutable_simulation_time()->set_proc_tick(tick_);
  }
  if (node->function_base()->IsBlock()) {
    time->mutable_simulation_time()->set_block_cycle(tick_);
  }
  XLS_ASSIGN_OR_RETURN(*node_value->mutable_value(), value.AsProto());
  XLS_RET_CHECK(writer_.WriteRecord(packet));
  return absl::OkStatus();
}

}  // namespace xls
