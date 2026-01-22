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

#include <cstdint>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/trace.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

class TraceRecorderTest : public IrTestBase {};

TEST_F(TraceRecorderTest, ProcRecording) {
  auto p = std::make_unique<Package>(TestName());
  ProcBuilder pb(TestName(), p.get());
  BValue st_bval = pb.StateElement("st", Value(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({st_bval}));
  XLS_ASSERT_OK_AND_ASSIGN(Node * state_node, proc->GetNode("st"));

  std::string trace_buffer;
  riegeli::RecordWriter writer{riegeli::StringWriter(&trace_buffer)};

  TraceRecorder recorder(writer);
  XLS_ASSERT_OK(recorder.RecordNodeValue(state_node, Value(UBits(42, 32))));
  recorder.Tick();
  XLS_ASSERT_OK(recorder.RecordNodeValue(state_node, Value(UBits(123, 32))));

  ASSERT_TRUE(writer.Close());

  riegeli::RecordReader reader{riegeli::StringReader(trace_buffer)};

  absl::flat_hash_map<std::string, int64_t> name_to_id;
  // Copy of trace.packets() with name mappings removed.
  std::vector<NodeTraceProto> node_values;
  TracePacketProto packet;
  while (reader.ReadRecord(packet)) {
    if (packet.has_node_id_name_mapping()) {
      name_to_id[packet.node_id_name_mapping().name()] =
          packet.node_id_name_mapping().id();
    } else if (packet.has_node_value()) {
      node_values.push_back(packet.node_value());
    }
  }
  ASSERT_TRUE(reader.Close());

  ASSERT_TRUE(name_to_id.contains("ProcRecording.st"));
  int64_t st_id = name_to_id.at("ProcRecording.st");

  NodeTraceProto expected0;
  expected0.set_node_id(st_id);
  expected0.mutable_time()->mutable_simulation_time()->set_proc_tick(0);
  XLS_ASSERT_OK_AND_ASSIGN(*expected0.mutable_value(),
                           Value(UBits(42, 32)).AsProto());

  NodeTraceProto expected1;
  expected1.set_node_id(st_id);
  expected1.mutable_time()->mutable_simulation_time()->set_proc_tick(1);
  XLS_ASSERT_OK_AND_ASSIGN(*expected1.mutable_value(),
                           Value(UBits(123, 32)).AsProto());

  EXPECT_THAT(
      node_values,
      testing::UnorderedElementsAre(
          proto_testing::Partially(proto_testing::EqualsProto(expected0)),
          proto_testing::Partially(proto_testing::EqualsProto(expected1))));
}

TEST_F(TraceRecorderTest, BlockRecording) {
  auto p = std::make_unique<Package>(TestName());
  BlockBuilder bb(TestName(), p.get());
  bb.InputPort("in", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Node * port_node, block->GetNode("in"));

  std::string trace_buffer;
  riegeli::RecordWriter writer{riegeli::StringWriter(&trace_buffer)};
  TraceRecorder recorder(writer);
  XLS_ASSERT_OK(recorder.RecordNodeValue(port_node, Value(UBits(42, 32))));
  recorder.Tick();
  XLS_ASSERT_OK(recorder.RecordNodeValue(port_node, Value(UBits(123, 32))));

  ASSERT_TRUE(writer.Close());

  riegeli::RecordReader reader{riegeli::StringReader(trace_buffer)};
  absl::flat_hash_map<std::string, int64_t> name_to_id;
  std::vector<NodeTraceProto> node_values;
  TracePacketProto packet;
  while (reader.ReadRecord(packet)) {
    if (packet.has_node_id_name_mapping()) {
      name_to_id[packet.node_id_name_mapping().name()] =
          packet.node_id_name_mapping().id();
    } else if (packet.has_node_value()) {
      node_values.push_back(packet.node_value());
    }
  }
  ASSERT_TRUE(reader.Close());

  ASSERT_TRUE(name_to_id.contains("BlockRecording.in"));
  int64_t in_id = name_to_id.at("BlockRecording.in");

  NodeTraceProto expected0;
  expected0.set_node_id(in_id);
  expected0.mutable_time()->mutable_simulation_time()->set_block_cycle(0);
  XLS_ASSERT_OK_AND_ASSIGN(*expected0.mutable_value(),
                           Value(UBits(42, 32)).AsProto());

  NodeTraceProto expected1;
  expected1.set_node_id(in_id);
  expected1.mutable_time()->mutable_simulation_time()->set_block_cycle(1);
  XLS_ASSERT_OK_AND_ASSIGN(*expected1.mutable_value(),
                           Value(UBits(123, 32)).AsProto());

  EXPECT_THAT(
      node_values,
      testing::UnorderedElementsAre(
          proto_testing::Partially(proto_testing::EqualsProto(expected0)),
          proto_testing::Partially(proto_testing::EqualsProto(expected1))));
}

}  // namespace
}  // namespace xls
