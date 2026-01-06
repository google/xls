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

#include "xls/interpreter/tracing_observer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/timestamp.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/trace.pb.h"
#include "xls/interpreter/trace_recorder.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/jit/block_jit.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_evaluator_options.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/jit/observer.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;
using ::xls::proto_testing::EqualsProto;
using ::xls::proto_testing::Partially;

class TracingObserverTest : public IrTestBase,
                            public ::testing::WithParamInterface<bool> {
 protected:
  // Runs the given function with the given observer, using either the JIT or
  // interpreter depending on the test parameter.
  absl::StatusOr<Value> RunFunction(Function* f, absl::Span<const Value> args,
                                    TracingObserver* observer) {
    if (GetParam()) {  // JIT
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<FunctionJit> function_jit,
          FunctionJit::Create(
              f, EvaluatorOptions(),
              JitEvaluatorOptions().set_include_observer_callbacks(
                  /*value=*/true)));
      RuntimeEvaluationObserverAdapter adapter(
          observer,
          [](uint64_t idx) {
            return reinterpret_cast<Node*>(static_cast<intptr_t>(idx));
          },
          function_jit->runtime());
      XLS_RETURN_IF_ERROR(function_jit->SetRuntimeObserver(&adapter));
      return DropInterpreterEvents(function_jit->Run(args));
    }
    // Interpreter
    return DropInterpreterEvents(
        InterpretFunction(f, args, EvaluatorOptions(), observer));
  }

  // Creates a proc runtime, using either the JIT or interpreter depending on
  // the test parameter.
  absl::StatusOr<std::unique_ptr<ProcRuntime>> CreateProcRuntime(Package* p) {
    if (GetParam()) {  // JIT
      return CreateJitSerialProcRuntime(
          p, EvaluatorOptions().set_support_observers(true));
    }
    // Interpreter
    return CreateInterpreterSerialProcRuntime(p);
  }

  const BlockEvaluator& GetBlockEvaluator() {
    if (GetParam()) {  // JIT
      return kObservableJitBlockEvaluator;
    }
    // Interpreter
    return kInterpreterBlockEvaluator;
  }
};

TEST_P(TracingObserverTest, SimpleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y, SourceInfo(), "my_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add));

  std::string trace_buffer;
  riegeli::RecordWriter writer{riegeli::StringWriter(&trace_buffer)};
  TraceRecorder recorder(writer);
  TracingObserver observer(recorder);

  XLS_ASSERT_OK(
      RunFunction(f, {Value(UBits(10, 32)), Value(UBits(32, 32))}, &observer)
          .status());

  ASSERT_TRUE(writer.Close());
  riegeli::RecordReader reader{riegeli::StringReader(trace_buffer)};
  std::vector<TracePacketProto> packets;
  TracePacketProto packet_reader;
  while (reader.ReadRecord(packet_reader)) {
    packets.push_back(packet_reader);
  }
  ASSERT_TRUE(reader.Close());

  absl::flat_hash_map<std::string, int64_t> name_to_id;
  std::vector<NodeTraceProto> node_values;
  for (const TracePacketProto& packet : packets) {
    if (packet.has_node_id_name_mapping()) {
      name_to_id[packet.node_id_name_mapping().name()] =
          packet.node_id_name_mapping().id();
    } else if (packet.has_node_value()) {
      node_values.push_back(packet.node_value());
    }
  }

  std::string x_name = absl::Substitute("$0.x", TestName());
  std::string y_name = absl::Substitute("$0.y", TestName());
  std::string my_add_name = absl::Substitute("$0.my_add", TestName());

  ASSERT_TRUE(name_to_id.contains(x_name));
  ASSERT_TRUE(name_to_id.contains(y_name));
  ASSERT_TRUE(name_to_id.contains(my_add_name));

  NodeTraceProto expected_x;
  expected_x.set_node_id(name_to_id.at(x_name));
  XLS_ASSERT_OK_AND_ASSIGN(*expected_x.mutable_value(),
                           Value(UBits(10, 32)).AsProto());

  NodeTraceProto expected_y;
  expected_y.set_node_id(name_to_id.at(y_name));
  XLS_ASSERT_OK_AND_ASSIGN(*expected_y.mutable_value(),
                           Value(UBits(32, 32)).AsProto());

  NodeTraceProto expected_add;
  expected_add.set_node_id(name_to_id.at(my_add_name));
  XLS_ASSERT_OK_AND_ASSIGN(*expected_add.mutable_value(),
                           Value(UBits(42, 32)).AsProto());

  EXPECT_THAT(node_values,
              UnorderedElementsAre(Partially(EqualsProto(expected_x)),
                                   Partially(EqualsProto(expected_y)),
                                   Partially(EqualsProto(expected_add))));

  // Check that time is populated.
  for (const auto& nv : node_values) {
    EXPECT_TRUE(nv.has_time());
    EXPECT_TRUE(nv.time().has_wall_time());
    // We can't know the exact time, but check that it's not zero.
    EXPECT_TRUE(nv.time().wall_time().seconds() > 0 ||
                nv.time().wall_time().nanos() > 0);
  }
}

TEST_P(TracingObserverTest, ProcTick) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue x_state = pb.StateElement("x", Value(UBits(10, 32)));
  XLS_ASSERT_OK(pb.Build({x_state}).status());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcRuntime> runtime,
                           CreateProcRuntime(p.get()));
  std::string trace_buffer;
  riegeli::RecordWriter writer{riegeli::StringWriter(&trace_buffer)};
  TraceRecorder recorder(writer);
  TracingObserver observer(recorder);

  XLS_ASSERT_OK(runtime->SetObserver(&observer));

  XLS_ASSERT_OK(runtime->Tick());

  ASSERT_TRUE(writer.Close());
  riegeli::RecordReader reader{riegeli::StringReader(trace_buffer)};
  std::vector<TracePacketProto> packets;
  TracePacketProto packet_reader;
  while (reader.ReadRecord(packet_reader)) {
    packets.push_back(packet_reader);
  }
  ASSERT_TRUE(reader.Close());

  std::vector<NodeTraceProto> node_values;
  for (const auto& packet : packets) {
    if (packet.has_node_value()) {
      node_values.push_back(packet.node_value());
    }
  }
  // Next state param evaluation & state register read.
  ASSERT_EQ(node_values.size(), 2);
  EXPECT_EQ(node_values[0].time().simulation_time().proc_tick(), 1);
  EXPECT_FALSE(node_values[0].time().simulation_time().has_block_cycle());
  EXPECT_EQ(node_values[1].time().simulation_time().proc_tick(), 1);
  EXPECT_FALSE(node_values[1].time().simulation_time().has_block_cycle());
}

TEST_P(TracingObserverTest, BlockCycle) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue x = bb.InputPort("x", p->GetBitsType(32));
  bb.OutputPort("o", x);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  std::string trace_buffer;
  riegeli::RecordWriter writer{riegeli::StringWriter(&trace_buffer)};
  TraceRecorder recorder(writer);
  TracingObserver observer(recorder);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BlockContinuation> continuation,
                           GetBlockEvaluator().NewContinuation(block));
  XLS_ASSERT_OK(continuation->SetObserver(&observer));

  absl::flat_hash_map<std::string, Value> inputs;
  inputs["x"] = Value(UBits(10, 32));

  XLS_ASSERT_OK(continuation->RunOneCycle(inputs));

  ASSERT_TRUE(writer.Close());
  riegeli::RecordReader reader{riegeli::StringReader(trace_buffer)};
  std::vector<TracePacketProto> packets;
  TracePacketProto packet_reader;
  while (reader.ReadRecord(packet_reader)) {
    packets.push_back(packet_reader);
  }
  ASSERT_TRUE(reader.Close());

  std::vector<NodeTraceProto> node_values;
  for (const auto& packet : packets) {
    if (packet.has_node_value()) {
      node_values.push_back(packet.node_value());
    }
  }
  // Input port and output port.
  ASSERT_EQ(node_values.size(), 2);
  // Cycle info should be present.
  EXPECT_FALSE(node_values[0].time().simulation_time().has_proc_tick());
  EXPECT_TRUE(node_values[0].time().simulation_time().has_block_cycle());
  EXPECT_EQ(node_values[0].time().simulation_time().block_cycle(), 1);
  EXPECT_FALSE(node_values[1].time().simulation_time().has_proc_tick());
  EXPECT_TRUE(node_values[1].time().simulation_time().has_block_cycle());
  EXPECT_EQ(node_values[1].time().simulation_time().block_cycle(), 1);
}

TEST_P(TracingObserverTest, ScopedTracingObserverWritesFile) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y, SourceInfo(), "my_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add));

  XLS_ASSERT_OK_AND_ASSIGN(TempFile temp, TempFile::Create());

  std::vector<Value> args = {Value(UBits(10, 32)), Value(UBits(32, 32))};
  {
    auto writer = std::make_unique<riegeli::RecordWriter<riegeli::FdWriter<>>>(
        riegeli::FdWriter(temp.path().string()));
    CHECK_OK(writer->status());
    ScopedTracingObserver observer(std::move(writer));
    if (GetParam()) {  // JIT
      XLS_ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<FunctionJit> function_jit,
          FunctionJit::Create(
              f, EvaluatorOptions(),
              JitEvaluatorOptions().set_include_observer_callbacks(true)));
      RuntimeEvaluationObserverAdapter adapter(
          &observer,
          [](uint64_t idx) {
            return reinterpret_cast<Node*>(static_cast<intptr_t>(idx));
          },
          function_jit->runtime());
      XLS_ASSERT_OK(function_jit->SetRuntimeObserver(&adapter));
      XLS_ASSERT_OK(DropInterpreterEvents(function_jit->Run(args)).status());
    } else {  // Interpreter
      XLS_ASSERT_OK(
          InterpretFunction(f, args, EvaluatorOptions(), &observer).status());
    }
  }

  riegeli::RecordReader reader{riegeli::FdReader(temp.path().string())};
  std::vector<TracePacketProto> packets;
  TracePacketProto packet_reader;
  while (reader.ReadRecord(packet_reader)) {
    packets.push_back(packet_reader);
  }
  ASSERT_TRUE(reader.Close());

  absl::flat_hash_map<std::string, int64_t> name_to_id;
  std::vector<NodeTraceProto> node_values;
  for (const TracePacketProto& packet : packets) {
    if (packet.has_node_id_name_mapping()) {
      name_to_id[packet.node_id_name_mapping().name()] =
          packet.node_id_name_mapping().id();
    } else if (packet.has_node_value()) {
      node_values.push_back(packet.node_value());
    }
  }

  std::string x_name = absl::Substitute("$0.x", TestName());
  std::string y_name = absl::Substitute("$0.y", TestName());
  std::string my_add_name = absl::Substitute("$0.my_add", TestName());

  ASSERT_TRUE(name_to_id.contains(x_name));
  ASSERT_TRUE(name_to_id.contains(y_name));
  ASSERT_TRUE(name_to_id.contains(my_add_name));

  NodeTraceProto expected_x;
  expected_x.set_node_id(name_to_id.at(x_name));
  XLS_ASSERT_OK_AND_ASSIGN(*expected_x.mutable_value(),
                           Value(UBits(10, 32)).AsProto());

  NodeTraceProto expected_y;
  expected_y.set_node_id(name_to_id.at(y_name));
  XLS_ASSERT_OK_AND_ASSIGN(*expected_y.mutable_value(),
                           Value(UBits(32, 32)).AsProto());

  NodeTraceProto expected_add;
  expected_add.set_node_id(name_to_id.at(my_add_name));
  XLS_ASSERT_OK_AND_ASSIGN(*expected_add.mutable_value(),
                           Value(UBits(42, 32)).AsProto());

  EXPECT_THAT(node_values,
              UnorderedElementsAre(Partially(EqualsProto(expected_x)),
                                   Partially(EqualsProto(expected_y)),
                                   Partially(EqualsProto(expected_add))));
}

INSTANTIATE_TEST_SUITE_P(JitOrInterpreter, TracingObserverTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Jit" : "Interpreter";
                         });

}  // namespace
}  // namespace xls
