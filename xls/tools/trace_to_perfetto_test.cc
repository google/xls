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

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "protos/perfetto/trace/trace.pb.h"
#include "protos/perfetto/trace/trace_packet.pb.h"
#include "protos/perfetto/trace/track_event/track_descriptor.pb.h"
#include "protos/perfetto/trace/track_event/track_event.pb.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/trace.pb.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::ResultOf;
using ::xls::proto_testing::EqualsProto;
using ::xls::proto_testing::Partially;

TracePacketProto ParseTracePacketOrDie(std::string_view text_proto) {
  TracePacketProto packet;
  CHECK_OK(xls::ParseTextProto(text_proto, {}, &packet));
  return packet;
}

TEST(TraceToPerfettoTest, ConvertEmptyTrace) {
  riegeli::RecordReader reader{riegeli::StringReader("")};
  EXPECT_THAT(TraceToPerfetto(reader), IsOkAndHolds(EqualsProto(R"pb()pb")));
}

TEST(TraceToPerfettoTest, ConvertNodeValueTrace) {
  std::string file_content;
  riegeli::RecordWriter writer{riegeli::StringWriter(&file_content)};
  std::vector<TracePacketProto> packets = {
      ParseTracePacketOrDie(
          R"pb(node_id_name_mapping { name: "my_func.add.1" id: 1 })pb"),
      ParseTracePacketOrDie(
          R"pb(node_id_name_mapping { name: "my_func.add.2" id: 2 })pb"),
      ParseTracePacketOrDie(
          R"pb(node_id_name_mapping { name: "my_func.sub.3" id: 3 })pb"),
      ParseTracePacketOrDie(R"pb(
        node_value {
          node_id: 1
          time { simulation_time { block_cycle: 10 } }
          value { bits { bit_count: 32 data: "\x01\x00\x00\x00" } }
        }
      )pb"),
      ParseTracePacketOrDie(R"pb(
        node_value {
          node_id: 2
          time { simulation_time { block_cycle: 10 } }
          value { bits { bit_count: 32 data: "\x02\x00\x00\x00" } }
        }
      )pb"),
      ParseTracePacketOrDie(R"pb(
        node_value {
          node_id: 3
          time { simulation_time { block_cycle: 10 } }
          value { bits { bit_count: 32 data: "\xff\xff\xff\xff" } }
        }
      )pb"),
      // Same value as before for node_id 1, should not start new slice
      ParseTracePacketOrDie(R"pb(
        node_value {
          node_id: 1
          time { simulation_time { block_cycle: 11 } }
          value { bits { bit_count: 32 data: "\x01\x00\x00\x00" } }
        }
      )pb"),
      // Value changes for node_id 1, new slice
      ParseTracePacketOrDie(R"pb(
        node_value {
          node_id: 1
          time { simulation_time { block_cycle: 12 } }
          value { bits { bit_count: 32 data: "\x03\x00\x00\x00" } }
        }
      )pb")};
  for (const TracePacketProto& packet : packets) {
    writer.WriteRecord(packet);
  }
  writer.Close();
  riegeli::RecordReader reader{riegeli::StringReader(file_content)};

  XLS_ASSERT_OK_AND_ASSIGN(perfetto::protos::Trace perfetto_trace,
                           TraceToPerfetto(reader));

  std::vector<perfetto::protos::TracePacket> descriptor_packets;
  std::vector<perfetto::protos::TracePacket> event_packets;
  for (const auto& packet : perfetto_trace.packet()) {
    if (packet.has_track_descriptor()) {
      descriptor_packets.push_back(packet);
    } else if (packet.has_track_event()) {
      event_packets.push_back(packet);
    }
  }

  EXPECT_THAT(descriptor_packets,
              testing::UnorderedElementsAre(
                  Partially(EqualsProto(R"pb(
                    track_descriptor { uuid: 1 name: "my_func.add.1" }
                    trusted_packet_sequence_id: 1
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    track_descriptor { uuid: 2 name: "my_func.add.2" }
                    trusted_packet_sequence_id: 1
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    track_descriptor { uuid: 3 name: "my_func.sub.3" }
                    trusted_packet_sequence_id: 1
                  )pb"))));

  EXPECT_THAT(event_packets,
              testing::UnorderedElementsAre(
                  Partially(EqualsProto(R"pb(
                    timestamp: 10
                    track_event {
                      type: TYPE_SLICE_BEGIN
                      track_uuid: 1
                      name: "bits[32]:1"
                    }
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    timestamp: 10
                    track_event {
                      type: TYPE_SLICE_BEGIN
                      track_uuid: 2
                      name: "bits[32]:2"
                    }
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    timestamp: 10
                    track_event {
                      type: TYPE_SLICE_BEGIN
                      track_uuid: 3
                      name: "bits[32]:4294967295"
                    }
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    timestamp: 12
                    track_event { type: TYPE_SLICE_END track_uuid: 1 }
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    timestamp: 12
                    track_event {
                      type: TYPE_SLICE_BEGIN
                      track_uuid: 1
                      name: "bits[32]:3"
                    }
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    timestamp: 12
                    track_event { type: TYPE_SLICE_END track_uuid: 1 }
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    timestamp: 12
                    track_event { type: TYPE_SLICE_END track_uuid: 2 }
                  )pb")),
                  Partially(EqualsProto(R"pb(
                    timestamp: 12
                    track_event { type: TYPE_SLICE_END track_uuid: 3 }
                  )pb"))));
}

TEST(TraceToPerfettoTest, ConvertTraceFromFile) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path path,
      GetXlsRunfilePath("xls/tools/testdata/eval_proc_main_test.tr"));
  riegeli::RecordReader reader{riegeli::FdReader(path)};
  XLS_ASSERT_OK_AND_ASSIGN(perfetto::protos::Trace perfetto_trace,
                           TraceToPerfetto(reader));
  EXPECT_GT(perfetto_trace.packet_size(), 0);

  std::vector<perfetto::protos::TracePacket> descriptor_packets;
  std::vector<perfetto::protos::TracePacket> event_packets;
  for (const auto& packet : perfetto_trace.packet()) {
    if (packet.has_track_descriptor()) {
      descriptor_packets.push_back(packet);
    } else if (packet.has_track_event()) {
      event_packets.push_back(packet);
    }
  }
  EXPECT_FALSE(descriptor_packets.empty());
  EXPECT_FALSE(event_packets.empty());

  EXPECT_THAT(descriptor_packets,
              Contains(ResultOf(
                  [](const auto& p) { return p.track_descriptor().name(); },
                  HasSubstr("test_proc"))));

  absl::flat_hash_map<uint64_t, int> slice_balance;
  for (const auto& packet : event_packets) {
    if (packet.track_event().type() ==
        perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN) {
      slice_balance[packet.track_event().track_uuid()]++;
    } else if (packet.track_event().type() ==
               perfetto::protos::TrackEvent::TYPE_SLICE_END) {
      slice_balance[packet.track_event().track_uuid()]--;
    }
  }

  for (const auto& [track_uuid, balance] : slice_balance) {
    EXPECT_EQ(balance, 0) << "track_uuid " << track_uuid
                          << " has unbalanced slices";
  }
}

}  // namespace
}  // namespace xls
