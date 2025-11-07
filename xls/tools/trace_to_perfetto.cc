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

#include "xls/tools/trace_to_perfetto.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "google/protobuf/timestamp.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "protos/perfetto/trace/trace.pb.h"
#include "protos/perfetto/trace/trace_packet.pb.h"
#include "protos/perfetto/trace/track_event/debug_annotation.pb.h"
#include "protos/perfetto/trace/track_event/track_descriptor.pb.h"
#include "protos/perfetto/trace/track_event/track_event.pb.h"
#include "riegeli/records/record_reader.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/trace.pb.h"
#include "xls/ir/value.h"

namespace xls {

// Unique ID identifying the trace writer.
constexpr uint32_t kPerfettoProgramTracerSequenceId = 1;

namespace {

absl::Status Validate(const google::protobuf::Timestamp& t) {
  const auto sec = t.seconds();
  const auto ns = t.nanos();
  // sec must be [0001-01-01T00:00:00Z, 9999-12-31T23:59:59.999999999Z]
  if (sec < -62135596800 || sec > 253402300799) {
    return absl::InvalidArgumentError(absl::StrCat("seconds=", sec));
  }
  if (ns < 0 || ns > 999999999) {
    return absl::InvalidArgumentError(absl::StrCat("nanos=", ns));
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Time> TimestampToTime(
    const google::protobuf::Timestamp& timestamp) {
  XLS_RETURN_IF_ERROR(Validate(timestamp));
  return absl::FromUnixSeconds(timestamp.seconds()) +
         absl::Nanoseconds(timestamp.nanos());
}

absl::StatusOr<uint64_t> GetTimestamp(const xls::TimeProto& time_proto) {
  if (time_proto.has_simulation_time()) {
    if (time_proto.simulation_time().has_block_cycle()) {
      return time_proto.simulation_time().block_cycle();
    }
    if (time_proto.simulation_time().has_proc_tick()) {
      return time_proto.simulation_time().proc_tick();
    }
  }
  XLS_ASSIGN_OR_RETURN(absl::Time wall_time,
                       TimestampToTime(time_proto.wall_time()));
  return absl::ToUnixNanos(wall_time);
}

}  // namespace

absl::StatusOr<perfetto::protos::Trace> TraceToPerfetto(
    riegeli::RecordReaderBase& xls_trace_reader) {
  perfetto::protos::Trace perfetto_trace;
  absl::flat_hash_map<int64_t, std::string> id_to_name;
  absl::flat_hash_map<int64_t, uint64_t> node_id_to_track_uuid;
  uint64_t next_uuid = 1;
  uint64_t max_timestamp = 0;

  // TODO(rigge): Add a clock track.

  absl::flat_hash_map<uint64_t, xls::Value> track_to_last_value;
  absl::flat_hash_map<uint64_t, uint64_t> track_to_slice_start_ts;

  TracePacketProto packet;
  while (xls_trace_reader.ReadRecord(packet)) {
    if (packet.has_node_id_name_mapping()) {
      id_to_name[packet.node_id_name_mapping().id()] =
          packet.node_id_name_mapping().name();
    } else if (packet.has_node_value()) {
      const auto& node_event = packet.node_value();
      int64_t node_id = node_event.node_id();

      if (!node_id_to_track_uuid.contains(node_id)) {
        uint64_t track_uuid = next_uuid++;
        node_id_to_track_uuid[node_id] = track_uuid;

        perfetto::protos::TracePacket* track_packet =
            perfetto_trace.add_packet();
        track_packet->set_trusted_packet_sequence_id(
            kPerfettoProgramTracerSequenceId);
        perfetto::protos::TrackDescriptor* track_descriptor =
            track_packet->mutable_track_descriptor();
        track_descriptor->set_uuid(track_uuid);
        track_descriptor->set_name(id_to_name.at(node_id));
      }

      uint64_t track_uuid = node_id_to_track_uuid.at(node_id);

      XLS_ASSIGN_OR_RETURN(uint64_t current_timestamp,
                           GetTimestamp(node_event.time()));
      max_timestamp = std::max(max_timestamp, current_timestamp);
      XLS_ASSIGN_OR_RETURN(xls::Value current_value,
                           xls::Value::FromProto(node_event.value()));

      if (!track_to_last_value.contains(track_uuid)) {
        // First event for this track
        track_to_last_value[track_uuid] = current_value;
        track_to_slice_start_ts[track_uuid] = current_timestamp;

        perfetto::protos::TracePacket* begin_packet =
            perfetto_trace.add_packet();
        begin_packet->set_trusted_packet_sequence_id(
            kPerfettoProgramTracerSequenceId);
        begin_packet->set_timestamp(current_timestamp);
        perfetto::protos::TrackEvent* begin_event =
            begin_packet->mutable_track_event();
        begin_event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
        begin_event->set_track_uuid(track_uuid);
        begin_event->set_name(current_value.ToString());
        perfetto::protos::DebugAnnotation* debug_annotation =
            begin_event->add_debug_annotations();
        debug_annotation->set_name("value");
        debug_annotation->set_string_value(current_value.ToString());
      } else {
        const xls::Value& last_value = track_to_last_value.at(track_uuid);
        if (current_value != last_value) {
          // Value changed, end previous slice and start a new one.

          perfetto::protos::TracePacket* end_packet =
              perfetto_trace.add_packet();
          end_packet->set_trusted_packet_sequence_id(
              kPerfettoProgramTracerSequenceId);
          end_packet->set_timestamp(current_timestamp);
          perfetto::protos::TrackEvent* end_event =
              end_packet->mutable_track_event();
          end_event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
          end_event->set_track_uuid(track_uuid);

          perfetto::protos::TracePacket* begin_packet =
              perfetto_trace.add_packet();
          begin_packet->set_trusted_packet_sequence_id(
              kPerfettoProgramTracerSequenceId);
          begin_packet->set_timestamp(current_timestamp);
          perfetto::protos::TrackEvent* begin_event =
              begin_packet->mutable_track_event();
          begin_event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
          begin_event->set_track_uuid(track_uuid);
          begin_event->set_name(current_value.ToString());
          perfetto::protos::DebugAnnotation* debug_annotation =
              begin_event->add_debug_annotations();
          debug_annotation->set_name("value");
          debug_annotation->set_string_value(current_value.ToString());

          track_to_last_value[track_uuid] = current_value;
          track_to_slice_start_ts[track_uuid] = current_timestamp;
        }
        // If value is the same, do nothing.
      }
    }
  }

  // Finalizing slices: End any open slices at max_timestamp.
  for (const auto& pair : track_to_slice_start_ts) {
    uint64_t track_uuid = pair.first;
    perfetto::protos::TracePacket* end_packet = perfetto_trace.add_packet();
    end_packet->set_trusted_packet_sequence_id(
        kPerfettoProgramTracerSequenceId);
    end_packet->set_timestamp(max_timestamp);
    perfetto::protos::TrackEvent* end_event = end_packet->mutable_track_event();
    end_event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
    end_event->set_track_uuid(track_uuid);
  }

  return perfetto_trace;
}

}  // namespace xls
