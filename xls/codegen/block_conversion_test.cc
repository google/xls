// Copyright 2021 The XLS Authors
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

#include "xls/codegen/block_conversion.h"

#include <cstdint>
#include <memory>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_options.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/verifier.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace m = xls::op_matchers;

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using testing::ElementsAre;
using testing::Pair;
using testing::UnorderedElementsAre;

// Specialization of IrTestBase for testing of simple blocks.
class BlockConversionTest : public IrTestBase {
 protected:
  // Returns the unique output port of the block (send over a port
  // channel). Check fails if no such unique send exists.
  OutputPort* GetOutputPort(Block* block) {
    OutputPort* output_port = nullptr;
    for (Node* node : block->nodes()) {
      if (node->Is<OutputPort>()) {
        output_port = node->As<OutputPort>();
      }
    }
    XLS_CHECK(output_port != nullptr);
    return output_port;
  }

  CodegenOptions codegen_options() {
    return CodegenOptions().module_name(TestName());
  }
};

// Unit delay delay estimator.
class TestDelayEstimator : public DelayEstimator {
 public:
  TestDelayEstimator() : DelayEstimator("test") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kAfterAll:
      case Op::kParam:
      case Op::kInputPort:
      case Op::kOutputPort:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kConcat:
      case Op::kTupleIndex:
      case Op::kReceive:
      case Op::kSend:
        return 0;
      default:
        return 1;
    }
  }
};

template <typename T>
class IsPrefixOf : public ::testing::MatcherInterface<std::vector<T>> {
 public:
  using is_gtest_matcher = void;
  explicit IsPrefixOf(const std::vector<T>& needed) : needed_(needed) {}

  bool MatchAndExplain(
      std::vector<T> array,
      ::testing::MatchResultListener* listener) const override {
    if (array.size() > needed_.size()) {
      return false;
    }
    for (int64_t i = 0; i < array.size(); ++i) {
      if (array.at(i) != needed_.at(i)) {
        return false;
      }
    }
    return true;
  }

  void DescribeTo(::std::ostream* os) const override {
    *os << "is a prefix of ";
    std::vector<std::string> strings;
    for (const T& element : needed_) {
      std::stringstream ss;
      ss << element;
      strings.push_back(ss.str());
    }
    *os << "[" << absl::StrJoin(strings, ", ") << "]";
  }

 private:
  std::vector<T> needed_;
};

// Convenience functions for sensitizing and analyzing procs used to
// test pipelined proc to block conversion
class ProcConversionTestFixture : public BlockConversionTest {
 protected:
  // A pair of cycle and value for returning traces.
  struct CycleAndValue {
    int64_t cycle;
    uint64_t value;
  };

  enum class SignalType { kInput, kOutput, kExpectedOutput };

  // Specification for a column when printing out a signal trace.
  struct SignalSpec {
    std::string signal_name;
    SignalType signal_type;
    bool active_low_reset = false;
  };

  // Creates a simple pipelined block named "the_proc" within a package.
  //
  // Returns the newly created package.
  virtual absl::StatusOr<std::unique_ptr<Package>> BuildBlockInPackage(
      int64_t stage_count, const CodegenOptions& options) {
    return absl::UnimplementedError("BuildBlockInPackage() unimplemented");
  }

  // For cycles in range [first_cycle, last_cycle] inclusive,
  // add the IO signals as described in signals to io.
  absl::Status SetSignalsOverCycles(
      int64_t first_cycle, int64_t last_cycle,
      const absl::flat_hash_map<std::string, uint64_t>& signals,
      std::vector<absl::flat_hash_map<std::string, uint64_t>>& io) const {
    XLS_CHECK_GE(first_cycle, 0);
    XLS_CHECK_GE(last_cycle, 0);
    XLS_CHECK_LE(first_cycle, last_cycle);

    if (io.size() <= last_cycle) {
      io.resize(last_cycle + 1);
    }

    for (auto [name, value] : signals) {
      for (int64_t i = first_cycle; i <= last_cycle; ++i) {
        io.at(i)[name] = value;
      }
    }

    return absl::OkStatus();
  }

  // For cycles in range [first_cycle, last_cycle] inclusive,
  // set given input signal to a incrementing value starting with start_val.
  //
  // One after the last signal value used is returned.
  absl::StatusOr<uint64_t> SetIncrementingSignalOverCycles(
      int64_t first_cycle, int64_t last_cycle, std::string_view signal_name,
      uint64_t signal_val,
      std::vector<absl::flat_hash_map<std::string, uint64_t>>& io) const {
    XLS_CHECK_GE(first_cycle, 0);
    XLS_CHECK_GE(last_cycle, 0);
    XLS_CHECK_LE(first_cycle, last_cycle);

    if (io.size() <= last_cycle) {
      io.resize(last_cycle + 1);
    }

    for (int64_t i = first_cycle; i <= last_cycle; ++i) {
      io.at(i)[signal_name] = signal_val;
      ++signal_val;
    }

    return signal_val;
  }

  // For cycles in range [first_cycle, last_cycle] inclusive,  set given
  // input signal to uniformly random input in range [min_value, max_value].
  absl::Status SetRandomSignalOverCycles(
      int64_t first_cycle, int64_t last_cycle, std::string_view signal_name,
      uint64_t min_value, uint64_t max_value, std::minstd_rand& rnd_engine,
      std::vector<absl::flat_hash_map<std::string, uint64_t>>& io) const {
    XLS_CHECK_GE(first_cycle, 0);
    XLS_CHECK_GE(last_cycle, 0);
    XLS_CHECK_LE(first_cycle, last_cycle);

    if (io.size() <= last_cycle) {
      io.resize(last_cycle + 1);
    }

    std::uniform_int_distribution<uint64_t> rnd_generator(min_value, max_value);

    for (int64_t i = first_cycle; i <= last_cycle; ++i) {
      io.at(i)[signal_name] = rnd_generator(rnd_engine);
    }

    return absl::OkStatus();
  }

  // From either an input or output channel, retrieve the sequence of
  // sent/received data.
  //
  // Data is deemed sent/received if not under reset and valid and ready are 1.
  absl::StatusOr<std::vector<CycleAndValue>> GetChannelSequenceFromIO(
      const SignalSpec& data_signal, const SignalSpec& valid_signal,
      const SignalSpec& ready_signal, const SignalSpec& reset_signal,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> outputs)
      const {
    XLS_CHECK_EQ(inputs.size(), outputs.size());

    std::vector<CycleAndValue> sequence;

    for (int64_t i = 0; i < inputs.size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          bool rst,
          FindWithinIOHashMaps(reset_signal, inputs.at(i), outputs.at(i)));

      XLS_ASSIGN_OR_RETURN(
          uint64_t data,
          FindWithinIOHashMaps(data_signal, inputs.at(i), outputs.at(i)));
      XLS_ASSIGN_OR_RETURN(
          bool data_vld,
          FindWithinIOHashMaps(valid_signal, inputs.at(i), outputs.at(i)));
      XLS_ASSIGN_OR_RETURN(
          bool data_rdy,
          FindWithinIOHashMaps(ready_signal, inputs.at(i), outputs.at(i)));

      bool rst_active = reset_signal.active_low_reset ? !rst : rst;

      if (data_vld && data_rdy && !rst_active) {
        sequence.push_back({i, data});
      }
    }

    return sequence;
  }

  // Log at verbose level 1, a table of signals and their values.
  absl::Status VLogTestPipelinedIO(
      absl::Span<const SignalSpec> table_spec, int64_t column_width,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> outputs,
      std::optional<
          absl::Span<const absl::flat_hash_map<std::string, uint64_t>>>
          expected_outputs = std::nullopt) const {
    XLS_CHECK_EQ(inputs.size(), outputs.size());
    if (expected_outputs.has_value()) {
      XLS_CHECK_EQ(inputs.size(), expected_outputs->size());
    }

    std::string header;
    for (const SignalSpec& col : table_spec) {
      if (col.signal_type == SignalType::kExpectedOutput) {
        std::string signal_name_with_suffix =
            absl::StrCat(col.signal_name, "_e");
        absl::StrAppend(&header, absl::StrFormat(" %*s", column_width,
                                                 signal_name_with_suffix));
      } else {
        absl::StrAppend(&header,
                        absl::StrFormat(" %*s", column_width, col.signal_name));
      }
    }

    XLS_VLOG(1) << header;

    for (int64_t i = 0; i < inputs.size(); ++i) {
      std::string row;

      for (const SignalSpec& col : table_spec) {
        std::string_view signal_name = col.signal_name;
        SignalType signal_type = col.signal_type;

        XLS_CHECK(signal_type == SignalType::kInput ||
                  signal_type == SignalType::kOutput ||
                  signal_type == SignalType::kExpectedOutput);

        uint64_t signal_value = 0;
        if (signal_type == SignalType::kInput) {
          signal_value = inputs.at(i).at(signal_name);
        } else if (signal_type == SignalType::kOutput) {
          signal_value = outputs.at(i).at(signal_name);
        } else {
          XLS_CHECK(expected_outputs.has_value());
          signal_value = expected_outputs->at(i).at(signal_name);
        }

        absl::StrAppend(&row,
                        absl::StrFormat(" %*d", column_width, signal_value));
      }

      XLS_VLOG(1) << row;
    }

    return absl::OkStatus();
  }

  // Find signal value either the input or output hash maps depending on the
  // spec.
  absl::StatusOr<uint64_t> FindWithinIOHashMaps(
      const SignalSpec& signal,
      const absl::flat_hash_map<std::string, uint64_t>& inputs,
      const absl::flat_hash_map<std::string, uint64_t>& outputs) const {
    SignalType signal_type = signal.signal_type;
    std::string_view signal_name = signal.signal_name;

    if (signal_type == SignalType::kInput) {
      if (!inputs.contains(signal_name)) {
        return absl::InternalError(
            absl::StrFormat("%s not found in input", signal_name));
      }
      return inputs.at(signal_name);
    } else if (signal_type == SignalType::kOutput ||
               signal_type == SignalType::kExpectedOutput) {
      if (!outputs.contains(signal_name)) {
        return absl::InternalError(
            absl::StrFormat("%s not found in output", signal_name));
      }
      return outputs.at(signal_name);
    }

    return absl::InternalError(absl::StrFormat(
        "Unsupported SignalType %d for %s", signal_type, signal_name));
  }

  // Name of the block created by BuildBlockInPackage().
  const std::string_view kBlockName = "the_proc";
};

TEST_F(BlockConversionTest, SimpleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block,
      FunctionToCombinationalBlock(
          f, codegen_options().module_name("SimpleFunctionBlock")));

  EXPECT_EQ(block->name(), "SimpleFunctionBlock");
  EXPECT_EQ(block->GetPorts().size(), 3);

  EXPECT_THAT(GetOutputPort(block),
              m::OutputPort(m::Add(m::InputPort("x"), m::InputPort("y"))));
}

TEST_F(BlockConversionTest, ZeroInputs) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Literal(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           FunctionToCombinationalBlock(f, codegen_options()));

  EXPECT_EQ(block->GetPorts().size(), 1);

  EXPECT_THAT(GetOutputPort(block), m::OutputPort("out", m::Literal(42)));
}

TEST_F(BlockConversionTest, ZeroWidthInputsAndOutput) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetTupleType({}));
  BValue y = fb.Param("y", p->GetBitsType(0));
  fb.Param("z", p->GetBitsType(1234));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({x, y})));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           FunctionToCombinationalBlock(f, codegen_options()));

  EXPECT_EQ(block->GetPorts().size(), 4);
}

TEST_F(BlockConversionTest, SimplePipelinedFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Negate(fb.Not(fb.Add(x, y)))));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block,
      FunctionToPipelinedBlock(
          schedule,
          CodegenOptions().flop_inputs(false).flop_outputs(false).clock_name(
              "clk"),
          f));

  EXPECT_THAT(GetOutputPort(block),
              m::OutputPort(m::Neg(m::Register(m::Not(m::Register(
                  m::Add(m::InputPort("x"), m::InputPort("y"))))))));
}

TEST_F(BlockConversionTest, TrivialPipelinedFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Negate(fb.Not(fb.Add(x, y)))));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));
  {
    // No flopping inputs or outputs.
    XLS_ASSERT_OK_AND_ASSIGN(
        Block * block,
        FunctionToPipelinedBlock(
            schedule,
            CodegenOptions().flop_inputs(false).flop_outputs(false).clock_name(
                "clk"),
            f));

    EXPECT_THAT(GetOutputPort(block),
                m::OutputPort(m::Neg(m::Register(m::Not(m::Register(
                    m::Add(m::InputPort("x"), m::InputPort("y"))))))));
    XLS_ASSERT_OK(p->RemoveBlock(block));
  }
  {
    // Flop inputs.
    XLS_ASSERT_OK_AND_ASSIGN(
        Block * block,
        FunctionToPipelinedBlock(
            schedule,
            CodegenOptions().flop_inputs(true).flop_outputs(false).clock_name(
                "clk"),
            f));

    EXPECT_THAT(GetOutputPort(block),
                m::OutputPort(m::Neg(m::Register(m::Not(
                    m::Register(m::Add(m::Register(m::InputPort("x")),
                                       m::Register(m::InputPort("y")))))))));
    XLS_ASSERT_OK(p->RemoveBlock(block));
  }
  {
    // Flop outputs.
    XLS_ASSERT_OK_AND_ASSIGN(
        Block * block,
        FunctionToPipelinedBlock(
            schedule,
            CodegenOptions().flop_inputs(false).flop_outputs(true).clock_name(
                "clk"),
            f));

    EXPECT_THAT(GetOutputPort(block),
                m::OutputPort(m::Register(m::Neg(m::Register(m::Not(m::Register(
                    m::Add(m::InputPort("x"), m::InputPort("y")))))))));
    XLS_ASSERT_OK(p->RemoveBlock(block));
  }
  {
    // Flop inputs and outputs.
    XLS_ASSERT_OK_AND_ASSIGN(
        Block * block,
        FunctionToPipelinedBlock(
            schedule,
            CodegenOptions().flop_inputs(true).flop_outputs(true).clock_name(
                "clk"),
            f));

    EXPECT_THAT(GetOutputPort(block),
                m::OutputPort(m::Register(m::Neg(m::Register(m::Not(
                    m::Register(m::Add(m::Register(m::InputPort("x")),
                                       m::Register(m::InputPort("y"))))))))));
    XLS_ASSERT_OK(p->RemoveBlock(block));
  }
}

TEST_F(BlockConversionTest, ZeroWidthPipeline) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetTupleType({}));
  BValue y = fb.Param("y", p->GetBitsType(0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({x, y})));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block,
      FunctionToPipelinedBlock(
          schedule,
          CodegenOptions().flop_inputs(false).flop_outputs(false).clock_name(
              "clk"),
          f));

  EXPECT_EQ(block->GetRegisters().size(), 4);
}

// Verifies that an implicit token, as generated by the DSLX IR converter, is
// appropriately plumbed into the wrapping block during conversion.
TEST_F(BlockConversionTest, ImplicitToken) {
  const std::string kIrText = R"(
package implicit_token

fn __itok__implicit_token__main(__token: token, __activated: bits[1]) ->
(token, ()) {
  after_all.7: token = after_all(__token, id=7)
  tuple.6: () = tuple(id=6)
  ret tuple.8: (token, ()) = tuple(after_all.7, tuple.6, id=8)
}

fn __implicit_token__main() -> () {
  after_all.9: token = after_all(id=9)
  literal.10: bits[1] = literal(value=1, id=10)
  invoke.11: (token, ()) = invoke(after_all.9, literal.10,
  to_apply=__itok__implicit_token__main, id=11) tuple_index.12: token =
  tuple_index(invoke.11, index=0, id=12) invoke.13: (token, ()) =
  invoke(tuple_index.12, literal.10, to_apply=__itok__implicit_token__main,
  id=13) ret tuple_index.14: () = tuple_index(invoke.13, index=1, id=14)
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, p->GetFunction("__implicit_token__main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto block,
                           FunctionToCombinationalBlock(f, codegen_options()));
  XLS_ASSERT_OK(VerifyBlock(block));
}

TEST_F(BlockConversionTest, SimpleProc) {
  const std::string ir_text = R"(package test

chan in(bits[32], id=0, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 1 }""")
chan out(bits[32], id=1, kind=single_value, ops=send_only,
         metadata="""module_port { flopped: false,  port_order: 0 }""")

proc my_proc(my_token: token, my_state: (), init={()}) {
  rcv: (token, bits[32]) = receive(my_token, channel_id=0)
  data: bits[32] = tuple_index(rcv, index=1)
  negate: bits[32] = neg(data)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, negate, channel_id=1)
  next (send, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));
  EXPECT_THAT(FindNode("out", block),
              m::OutputPort("out", m::Neg(m::InputPort("in"))));
}

TEST_F(BlockConversionTest, ProcWithVariousNextStateNodes) {
  // A block with corner-case next state nodes (e.g., not dependent on state
  // param, same as state param, and shared next state nodes).
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      p->CreateStreamingChannel("input", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_out,
      p->CreateStreamingChannel("x_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_out,
      p->CreateStreamingChannel("y_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * z_out,
      p->CreateStreamingChannel("z_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_out,
      p->CreateStreamingChannel("in_out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder b(TestName(), "tkn", p.get());
  BValue x = b.StateElement("x", Value(UBits(0, 32)));
  BValue y = b.StateElement("y", Value(UBits(0, 32)));
  BValue z = b.StateElement("z", Value(UBits(0, 32)));
  BValue x_plus_one = b.Add(x, b.Literal(UBits(1, 32)));

  b.Send(in_out, b.Identity(b.Receive(in)));
  b.Send(x_out, x);
  b.Send(y_out, y);
  b.Send(z_out, z);

  // `x_plus_one` is the next state value for both `x` and `y` state elements.
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({x_plus_one, x_plus_one, z}));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));

  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToPipelinedBlock(schedule, options, proc));

  std::vector<ChannelSource> sources{
      ChannelSource("input_data", "input_valid", "input_ready", 1.0, block),
  };
  XLS_ASSERT_OK(sources.front().SetDataSequence({10, 20, 30}));
  std::vector<ChannelSink> sinks{
      ChannelSink("x_out_data", "x_out_valid", "x_out_ready", 1.0, block),
      ChannelSink("y_out_data", "y_out_valid", "y_out_ready", 1.0, block),
      ChannelSink("z_out_data", "z_out_valid", "z_out_ready", 1.0, block),
      ChannelSink("in_out_data", "in_out_valid", "in_out_ready", 1.0, block),
  };
  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs(10,
                                                                 {{"rst", 0}});
  XLS_ASSERT_OK_AND_ASSIGN(
      BlockIOResultsAsUint64 results,
      InterpretChannelizedSequentialBlockWithUint64(
          block, absl::MakeSpan(sources), absl::MakeSpan(sinks), inputs));

  EXPECT_THAT(sinks.at(0).GetOutputSequenceAsUint64(),
              IsOkAndHolds(ElementsAre(0, 1, 2)));
  EXPECT_THAT(sinks.at(1).GetOutputSequenceAsUint64(),
              IsOkAndHolds(ElementsAre(0, 1, 2)));
  EXPECT_THAT(sinks.at(2).GetOutputSequenceAsUint64(),
              IsOkAndHolds(ElementsAre(0, 0, 0)));
  EXPECT_THAT(sinks.at(3).GetOutputSequenceAsUint64(),
              IsOkAndHolds(ElementsAre(10, 20, 30)));
}

TEST_F(BlockConversionTest, ChannelDefaultAndNonDefaultSuffixName) {
  const std::string ir_text = R"(package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only,
        flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid, metadata="")
chan in2(bits[32], id=2, kind=single_value, ops=receive_only, metadata="")
chan out2(bits[32], id=3, kind=single_value, ops=send_only, metadata="")

proc my_proc(my_token: token, my_state: (), init={()}) {
  rcv: (token, bits[32]) = receive(my_token, channel_id=0)
  rcv2: (token, bits[32]) = receive(my_token, channel_id=2)

  data: bits[32] = tuple_index(rcv, index=1)
  rcv_token: token = tuple_index(rcv, index=0)
  negate: bits[32] = neg(data)

  data2: bits[32] = tuple_index(rcv2, index=1)
  rcv2_token: token = tuple_index(rcv2, index=0)
  negate2: bits[32] = neg(data2)

  send: token = send(rcv_token, negate, channel_id=1)
  send2: token = send(rcv2_token, negate2, channel_id=3)
  fin : token = after_all(send, send2)
  next (fin, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block_default_suffix,
                           ProcToCombinationalBlock(proc, codegen_options()));

  EXPECT_TRUE(HasNode("in", block_default_suffix));
  EXPECT_TRUE(HasNode("in_rdy", block_default_suffix));
  EXPECT_TRUE(HasNode("in_vld", block_default_suffix));

  EXPECT_TRUE(HasNode("out", block_default_suffix));
  EXPECT_TRUE(HasNode("out_rdy", block_default_suffix));
  EXPECT_TRUE(HasNode("out_vld", block_default_suffix));

  EXPECT_TRUE(HasNode("in2", block_default_suffix));
  EXPECT_FALSE(HasNode("in2_rdy", block_default_suffix));
  EXPECT_FALSE(HasNode("in2_vld", block_default_suffix));

  EXPECT_TRUE(HasNode("out2", block_default_suffix));
  EXPECT_FALSE(HasNode("out2_rdy", block_default_suffix));
  EXPECT_FALSE(HasNode("out2_vld", block_default_suffix));

  CodegenOptions options = codegen_options()
                               .streaming_channel_data_suffix("_data")
                               .streaming_channel_ready_suffix("_ready")
                               .streaming_channel_valid_suffix("_valid");
  XLS_ASSERT_OK_AND_ASSIGN(Block * block_nondefault_suffix,
                           ProcToCombinationalBlock(proc, options));

  XLS_VLOG_LINES(3, block_nondefault_suffix->DumpIr());

  EXPECT_TRUE(HasNode("in_data", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("in_ready", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("in_valid", block_nondefault_suffix));

  EXPECT_TRUE(HasNode("out_data", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("out_ready", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("out_valid", block_nondefault_suffix));

  // Non-streaming / ready-valid channels are not impacted by suffix.
  EXPECT_TRUE(HasNode("in2", block_nondefault_suffix));
  EXPECT_FALSE(HasNode("in2_data", block_nondefault_suffix));
  EXPECT_FALSE(HasNode("in2_ready", block_nondefault_suffix));
  EXPECT_FALSE(HasNode("in2_valid", block_nondefault_suffix));

  EXPECT_TRUE(HasNode("out2", block_nondefault_suffix));
  EXPECT_FALSE(HasNode("out2_data", block_nondefault_suffix));
  EXPECT_FALSE(HasNode("out2_ready", block_nondefault_suffix));
  EXPECT_FALSE(HasNode("out2_valid", block_nondefault_suffix));
}

TEST_F(BlockConversionTest, ProcWithMultipleInputChannels) {
  const std::string ir_text = R"(package test

chan in0(bits[32], id=0, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 0 }""")
chan in1(bits[32], id=1, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 2 }""")
chan in2(bits[32], id=2, kind=single_value, ops=receive_only,
        metadata="""module_port { flopped: false,  port_order: 1 }""")
chan out(bits[32], id=3, kind=single_value, ops=send_only,
         metadata="""module_port { flopped: false,  port_order: 0 }""")

proc my_proc(my_token: token, my_state: (), init={()}) {
  rcv0: (token, bits[32]) = receive(my_token, channel_id=0)
  rcv0_token: token = tuple_index(rcv0, index=0)
  rcv1: (token, bits[32]) = receive(rcv0_token, channel_id=1)
  rcv1_token: token = tuple_index(rcv1, index=0)
  rcv2: (token, bits[32]) = receive(rcv1_token, channel_id=2)
  rcv2_token: token = tuple_index(rcv2, index=0)
  data0: bits[32] = tuple_index(rcv0, index=1)
  data1: bits[32] = tuple_index(rcv1, index=1)
  data2: bits[32] = tuple_index(rcv2, index=1)
  neg_data1: bits[32] = neg(data1)
  two: bits[32] = literal(value=2)
  data2_times_two: bits[32] = umul(data2, two)
  tmp: bits[32] = add(neg_data1, data2_times_two)
  sum: bits[32] = add(tmp, data0)
  send: token = send(rcv2_token, sum, channel_id=3)
  next (send, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));
  EXPECT_THAT(
      FindNode("out", block),
      m::OutputPort("out",
                    m::Add(m::Add(m::Neg(m::InputPort("in1")),
                                  m::UMul(m::InputPort("in2"), m::Literal(2))),
                           m::InputPort("in0"))));
}

TEST_F(BlockConversionTest, OnlyFIFOOutProc) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=single_value, ops=receive_only, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

proc my_proc(tkn: token, st: (), init={()}) {
  receive.13: (token, bits[32]) = receive(tkn, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  literal.21: bits[1] = literal(value=1, id=21, pos=[(1,8,3)])
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15, predicate=literal.21, channel_id=1, id=20, pos=[(1,5,1)])
  next (send.20, st)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));
  EXPECT_THAT(FindNode("out", block), m::OutputPort("out", m::InputPort("in")));
  EXPECT_THAT(FindNode("out_vld", block),
              m::OutputPort("out_vld", m::And(m::Literal(1), m::Literal(1),
                                              m::Literal(1), m::Literal(1))));
}

TEST_F(BlockConversionTest, OnlyFIFOInProcGateRecvsTrue) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only,
flow_control=ready_valid, metadata="""module_port { flopped: false
port_order: 0 }""") chan out(bits[32], id=1, kind=single_value,
ops=send_only, metadata="""module_port { flopped: false port_order: 1 }""")

proc my_proc(tkn: token, st: (), init={()}) {
  literal.21: bits[1] = literal(value=1, id=21, pos=[(1,8,3)])
  receive.13: (token, bits[32]) = receive(tkn, predicate=literal.21, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15,
                        channel_id=1, id=20, pos=[(1,5,1)])
  next (send.20, st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));

  // A select node is inferred by the `receive.13` node.
  EXPECT_THAT(
      FindNode("out", block),
      m::OutputPort("out", m::Select(m::Literal(1),
                                     {m::Literal(0), m::InputPort("in")})));
  EXPECT_THAT(FindNode("in", block), m::InputPort("in"));
  EXPECT_THAT(FindNode("in_vld", block), m::InputPort("in_vld"));
  EXPECT_THAT(FindNode("in_rdy", block),
              m::OutputPort("in_rdy", m::And(m::Literal(1), m::Literal(1))));
}

TEST_F(BlockConversionTest, OnlyFIFOInProcGateRecvsFalse) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only,
flow_control=ready_valid, metadata="""module_port { flopped: false
port_order: 0 }""") chan out(bits[32], id=1, kind=single_value,
ops=send_only, metadata="""module_port { flopped: false port_order: 1 }""")

proc my_proc(tkn: token, st: (), init={()}) {
  literal.21: bits[1] = literal(value=1, id=21, pos=[(1,8,3)])
  receive.13: (token, bits[32]) = receive(tkn, predicate=literal.21, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15,
                        channel_id=1, id=20, pos=[(1,5,1)])
  next (send.20, st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  CodegenOptions options = codegen_options();
  options.gate_recvs(false);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, options));

  EXPECT_THAT(FindNode("out", block), m::OutputPort("out", m::InputPort("in")));
  EXPECT_THAT(FindNode("in", block), m::InputPort("in"));
  EXPECT_THAT(FindNode("in_vld", block), m::InputPort("in_vld"));
  EXPECT_THAT(FindNode("in_rdy", block),
              m::OutputPort("in_rdy", m::And(m::Literal(1), m::Literal(1))));
}

TEST_F(BlockConversionTest, UnconditionalSendRdyVldProc) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=single_value, ops=receive_only, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

proc my_proc(tkn: token, st: (), init={()}) {
  receive.13: (token, bits[32]) = receive(tkn, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15, channel_id=1, id=20) next (send.20, st)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));

  EXPECT_THAT(FindNode("out", block), m::OutputPort("out", m::InputPort("in")));
  EXPECT_THAT(FindNode("out_vld", block),
              m::OutputPort("out_vld", m::And(m::Literal(1), m::Literal(1),
                                              m::Literal(1))));
  EXPECT_THAT(FindNode("out_rdy", block), m::InputPort("out_rdy"));
}

// Ensure that the output of the receive is zero when the predicate is false.
TEST_F(BlockConversionTest, ReceiveIfIsZeroWhenPredicateIsFalse) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_pred,
      package.CreateSingleValueChannel("pred", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue ch_pred_value = pb.Receive(ch_pred);
  BValue a = pb.ReceiveIf(ch_in, ch_pred_value);
  pb.Send(ch_out, a);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));

  // Assert the predicate to false. Note out contains the value of 0 although
  // the value of in is 42.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 0}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 0), Pair("out_vld", 1),
                                        Pair("out", 0))));

  // Assert the predicate to true. Note out contains the value of in (42).
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 1}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 1), Pair("out_vld", 1),
                                        Pair("out", 42))));
}

// Ensure that the output of the receive is passthrough when the predicate is
// false.
TEST_F(BlockConversionTest, ReceiveIfIsPassthroughWhenPredicateIsFalse) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_pred,
      package.CreateSingleValueChannel("pred", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue ch_pred_value = pb.Receive(ch_pred);
  BValue a = pb.ReceiveIf(ch_in, ch_pred_value);
  pb.Send(ch_out, a);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  CodegenOptions options = codegen_options();
  options.gate_recvs(false);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, options));

  // Assert the predicate to false. Note out contains the value of in (42).
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 0}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 0), Pair("out_vld", 1),
                                        Pair("out", 42))));

  // Assert the predicate to true. Note out contains the value of in (42).
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 1}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 1), Pair("out_vld", 1),
                                        Pair("out", 42))));
}

// Ensure that the output of the receive is zero when the data is not valid.
TEST_F(BlockConversionTest, NonblockingReceiveIsZeroWhenDataInvalid) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue in = pb.ReceiveNonBlocking(ch_in, pb.GetTokenParam());
  BValue in_tkn = pb.TupleIndex(in, 0);
  BValue in_data = pb.TupleIndex(in, 1);
  BValue tok_fin = pb.Send(ch_out, in_tkn, in_data);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(tok_fin, {}));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));

  // `in`'s valid signal is deasserted. Note `out` contains the value of 0
  // although the value of `in` is 42.
  EXPECT_THAT(InterpretCombinationalBlock(
                  block, {{"in", 42}, {"in_vld", 0}, {"out_rdy", 1}}),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair("in_rdy", 1), Pair("out_vld", 1), Pair("out", 0))));

  // `in`'s valid signal is asserted. Note `out` contains the value of `in`
  // which is 42.
  EXPECT_THAT(InterpretCombinationalBlock(
                  block, {{"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair("in_rdy", 1), Pair("out_vld", 1), Pair("out", 42))));
}

// Ensure that the output of the receive is passthrough when the data is not
// valid.
TEST_F(BlockConversionTest, NonblockingReceiveIsPassthroughWhenDataInvalid) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue in = pb.ReceiveNonBlocking(ch_in, pb.GetTokenParam());
  BValue in_tkn = pb.TupleIndex(in, 0);
  BValue in_data = pb.TupleIndex(in, 1);
  BValue tok_fin = pb.Send(ch_out, in_tkn, in_data);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(tok_fin, {}));

  CodegenOptions options = codegen_options();
  options.gate_recvs(false);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, options));

  // `in`'s valid signal is deasserted. Note `out` contains the value of `in`
  // which is 42.
  EXPECT_THAT(InterpretCombinationalBlock(
                  block, {{"in", 42}, {"in_vld", 0}, {"out_rdy", 1}}),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair("in_rdy", 1), Pair("out_vld", 1), Pair("out", 42))));

  // `in`'s valid signal is asserted. Note `out` contains the value of `in`
  // which is 42.
  EXPECT_THAT(InterpretCombinationalBlock(
                  block, {{"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair("in_rdy", 1), Pair("out_vld", 1), Pair("out", 42))));
}

// Ensure that the output of the receive is zero when the predicate is false.
TEST_F(BlockConversionTest, NonblockingReceiveIsZeroWhenPredicateIsFalse) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_pred,
      package.CreateSingleValueChannel("pred", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue ch_pred_response = pb.Receive(ch_pred, pb.GetTokenParam());
  BValue ch_pred_tkn = pb.TupleIndex(ch_pred_response, 0);
  BValue ch_pred_value = pb.TupleIndex(ch_pred_response, 1);
  BValue in_response =
      pb.ReceiveIfNonBlocking(ch_in, ch_pred_tkn, ch_pred_value);
  BValue in_tkn = pb.TupleIndex(in_response, 0);
  BValue in_data = pb.TupleIndex(in_response, 1);
  BValue tok_fin = pb.Send(ch_out, in_tkn, in_data);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(tok_fin, {}));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));

  // Assert the predicate to false. Note `out` contains the value of 0 although
  // the value of `in` is 42.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 0}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 0), Pair("out_vld", 1),
                                        Pair("out", 0))));

  // Assert the predicate to true. Note `out` contains the value of `in` (42).
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 1}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 1), Pair("out_vld", 1),
                                        Pair("out", 42))));
}

// Ensure that the output of the receive is passthrough when the predicate is
// false.
TEST_F(BlockConversionTest,
       NonblockingReceiveIsPassthroughWhenPredicateIsFalse) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_pred,
      package.CreateSingleValueChannel("pred", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue ch_pred_response = pb.Receive(ch_pred, pb.GetTokenParam());
  BValue ch_pred_tkn = pb.TupleIndex(ch_pred_response, 0);
  BValue ch_pred_value = pb.TupleIndex(ch_pred_response, 1);
  BValue in_response =
      pb.ReceiveIfNonBlocking(ch_in, ch_pred_tkn, ch_pred_value);
  BValue in_tkn = pb.TupleIndex(in_response, 0);
  BValue in_data = pb.TupleIndex(in_response, 1);
  BValue tok_fin = pb.Send(ch_out, in_tkn, in_data);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(tok_fin, {}));

  CodegenOptions options = codegen_options();
  options.gate_recvs(false);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, options));

  // Assert the predicate to false. Note out contains the value of in (42).
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 0}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 0), Pair("out_vld", 1),
                                        Pair("out", 42))));

  // Assert the predicate to true. Note `out` contains the value of `in` (42).
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block, {{"pred", 1}, {"in", 42}, {"in_vld", 1}, {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("in_rdy", 1), Pair("out_vld", 1),
                                        Pair("out", 42))));
}

TEST_F(BlockConversionTest, TwoToOneProc) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_dir,
      package.CreateSingleValueChannel("dir", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue dir = pb.Receive(ch_dir);
  BValue a = pb.ReceiveIf(ch_a, dir);
  BValue b = pb.ReceiveIf(ch_b, pb.Not(dir));
  pb.Send(ch_out, pb.Select(dir, {b, a}));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));

  // Input B selected, input valid and output ready asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(block, {{"dir", 0},
                                          {"a", 123},
                                          {"b", 42},
                                          {"a_vld", 1},
                                          {"b_vld", 1},
                                          {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 1), Pair("b_rdy", 1),
                                        Pair("out", 42), Pair("a_rdy", 0))));

  // Input A selected, input valid and output ready asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(block, {{"dir", 1},
                                          {"a", 123},
                                          {"b", 42},
                                          {"a_vld", 1},
                                          {"b_vld", 0},
                                          {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 1), Pair("b_rdy", 0),
                                        Pair("out", 123), Pair("a_rdy", 1))));

  // Input A selected, input valid asserted, and output ready *not*
  // asserted. Input ready should be zero.
  EXPECT_THAT(
      InterpretCombinationalBlock(block, {{"dir", 1},
                                          {"a", 123},
                                          {"b", 42},
                                          {"a_vld", 1},
                                          {"b_vld", 1},
                                          {"out_rdy", 0}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 1), Pair("b_rdy", 0),
                                        Pair("out", 123), Pair("a_rdy", 0))));

  // Input A selected, input valid *not* asserted, and output ready
  // asserted. Output valid should be zero.
  EXPECT_THAT(
      InterpretCombinationalBlock(block, {{"dir", 1},
                                          {"a", 123},
                                          {"b", 42},
                                          {"a_vld", 0},
                                          {"b_vld", 1},
                                          {"out_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", 0), Pair("b_rdy", 0),
                                        Pair("out", 123), Pair("a_rdy", 1))));
}

TEST_F(BlockConversionTest, OneToTwoProc) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_dir,
      package.CreateSingleValueChannel("dir", ChannelOps::kReceiveOnly,
                                       package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a,
      package.CreateStreamingChannel("a", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b,
      package.CreateStreamingChannel("b", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
  BValue dir = pb.Receive(ch_dir);
  BValue in = pb.Receive(ch_in);
  pb.SendIf(ch_a, dir, in);
  pb.SendIf(ch_b, pb.Not(dir), in);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToCombinationalBlock(proc, codegen_options()));

  // Output B selected. Input valid and output readies asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block,
          {{"dir", 0}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 1),
                                        Pair("in_rdy", 1), Pair("a_vld", 0),
                                        Pair("b", 123))));

  // Output A selected. Input valid and output readies asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 1), Pair("a_vld", 1),
                                        Pair("b", 123))));

  // Output A selected. Input *not* valid and output readies asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 0}, {"a_rdy", 1}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 1), Pair("a_vld", 0),
                                        Pair("b", 123))));

  // Output A selected. Input valid and output ready *not* asserted.
  EXPECT_THAT(
      InterpretCombinationalBlock(
          block,
          {{"dir", 1}, {"in", 123}, {"in_vld", 1}, {"a_rdy", 0}, {"b_rdy", 1}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("a", 123), Pair("b_vld", 0),
                                        Pair("in_rdy", 0), Pair("a_vld", 1),
                                        Pair("b", 123))));
}

TEST_F(BlockConversionTest, FlopSingleValueChannelProc) {
  const std::string ir_text = R"(package test
chan in(bits[32], id=0, kind=single_value, ops=receive_only, metadata="")
chan out(bits[32], id=1, kind=single_value, ops=send_only, metadata="")

proc my_proc(tkn: token, st: (), init={()}) {
  receive.13: (token, bits[32]) = receive(tkn, channel_id=0, id=13)
  tuple_index.14: token = tuple_index(receive.13, index=0, id=14)
  tuple_index.15: bits[32] = tuple_index(receive.13, index=1, id=15)
  send.20: token = send(tuple_index.14, tuple_index.15, channel_id=1, id=20) next (send.20, st)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));

  CodegenOptions options;
  options.module_name("my_proc");
  options.flop_inputs(true).flop_outputs(true).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst_n", false, /*active_low=*/true, false);

  {
    options.flop_single_value_channels(true);

    XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                             ProcToPipelinedBlock(schedule, options, proc));

    XLS_VLOG_LINES(2, block->DumpIr());

    EXPECT_TRUE(HasNode("__out_reg", block));
    EXPECT_TRUE(HasNode("__in_reg", block));
    EXPECT_THAT(FindNode("out", block),
                m::OutputPort("out", m::RegisterRead("__out_reg")));
    EXPECT_THAT(FindNode("__in_reg", block), m::RegisterRead("__in_reg"));
  }

  {
    options.flop_single_value_channels(false);

    XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                             ProcToPipelinedBlock(schedule, options, proc));

    XLS_VLOG_LINES(2, block->DumpIr());

    EXPECT_FALSE(HasNode("__out_reg", block));
    EXPECT_FALSE(HasNode("__in_reg", block));
  }
}

// Fixture used to test pipelined BlockConversion on a simple
// identity block.
class SimplePipelinedProcTest : public ProcConversionTestFixture {
 protected:
  absl::StatusOr<std::unique_ptr<Package>> BuildBlockInPackage(
      int64_t stage_count, const CodegenOptions& options) override {
    // Simple streaming one input and one output pipeline.
    auto package_ptr = std::make_unique<Package>(TestName());
    Package& package = *package_ptr;

    Type* u32 = package.GetBitsType(32);
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in,
        package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_out,
        package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

    TokenlessProcBuilder pb(TestName(),
                            /*token_name=*/"tkn", &package);

    BValue in_val = pb.Receive(ch_in);

    BValue buffered_in_val = pb.Not(pb.Not(in_val));
    pb.Send(ch_out, buffered_in_val);
    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build({}));

    XLS_VLOG(2) << "Simple streaming proc";
    XLS_VLOG_LINES(2, proc->DumpIr());

    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
            proc, TestDelayEstimator(),
            SchedulingOptions()
                .pipeline_stages(stage_count)
                .add_constraint(RecvsFirstSendsLastConstraint())));

    CodegenOptions codegen_options = options;
    codegen_options.module_name(kBlockName);

    XLS_RET_CHECK_OK(ProcToPipelinedBlock(schedule, codegen_options, proc));

    return package_ptr;
  }
};

TEST_F(SimplePipelinedProcTest, ChannelDefaultSuffixName) {
  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package_default_suffix,
                           BuildBlockInPackage(/*stage_count=*/4, options));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block_default_suffix,
                           package_default_suffix->GetBlock(kBlockName));

  EXPECT_TRUE(HasNode("in", block_default_suffix));
  EXPECT_TRUE(HasNode("in_rdy", block_default_suffix));
  EXPECT_TRUE(HasNode("in_vld", block_default_suffix));

  EXPECT_TRUE(HasNode("out", block_default_suffix));
  EXPECT_TRUE(HasNode("out_rdy", block_default_suffix));
  EXPECT_TRUE(HasNode("out_vld", block_default_suffix));
}

TEST_F(SimplePipelinedProcTest, ChannelNonDefaultSuffixName) {
  CodegenOptions options;
  options.flop_inputs(false)
      .flop_outputs(false)
      .clock_name("clk")
      .valid_control("input_valid", "output_valid")
      .reset("rst", false, false, false)
      .streaming_channel_data_suffix("_data")
      .streaming_channel_ready_suffix("_ready")
      .streaming_channel_valid_suffix("_valid");

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package_nondefault_suffix,
                           BuildBlockInPackage(/*stage_count=*/4, options));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block_nondefault_suffix,
                           package_nondefault_suffix->GetBlock(kBlockName));

  EXPECT_TRUE(HasNode("in_data", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("in_ready", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("in_valid", block_nondefault_suffix));

  EXPECT_TRUE(HasNode("out_data", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("out_ready", block_nondefault_suffix));
  EXPECT_TRUE(HasNode("out_valid", block_nondefault_suffix));
}

TEST_F(SimplePipelinedProcTest, BasicDatapathResetAndInputFlop) {
  CodegenOptions options;
  options.flop_inputs(true).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, /*reset_data_path=*/true);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           BuildBlockInPackage(/*stage_count=*/4, options));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Simple streaming pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  std::vector<absl::flat_hash_map<std::string, uint64_t>> expected_outputs;

  uint64_t running_in_val = 1;
  uint64_t running_out_val = 0;

  // During reset, the output will be 0 due to reset also resetting the
  // datapath
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(0, 9, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      0, 9, {{"rst", 1}, {"in_vld", 1}, {"out_rdy", 1}}, inputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(
      0, 9, {{"in_rdy", 1}, {"out_vld", 0}, {"out", 0}}, expected_outputs));

  // Once reset is deasserted, then the pipeline is closed, no further changes
  // in the output is expected if the input is not valid.
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(10, 19, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 19, {{"rst", 0}, {"in_vld", 0}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 19, {{"in_rdy", 0}, {"out_vld", 0}, {"out", 0}}, expected_outputs));

  // Returning input_valid, output will reflect valid input upon pipeline delay.
  uint64_t prior_running_out_val = running_out_val;
  running_out_val = running_in_val;
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(20, 29, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      20, 29, {{"rst", 0}, {"in_vld", 1}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      20, 22, {{"in_rdy", 1}, {"out_vld", 0}, {"out", prior_running_out_val}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      23, 23, {{"in_rdy", 1}, {"out_vld", 0}, {"out", 0}}, expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(24, 29, {{"in_rdy", 1}, {"out_vld", 1}},
                                     expected_outputs));
  XLS_ASSERT_OK_AND_ASSIGN(
      running_out_val, SetIncrementingSignalOverCycles(
                           24, 29, "out", running_out_val, expected_outputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, expected_outputs.size() - 1,
                                                "cycle", 0, expected_outputs));
  ASSERT_EQ(inputs.size(), expected_outputs.size());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  ASSERT_EQ(outputs.size(), expected_outputs.size());

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                              {"rst", SignalType::kInput},
                              {"in", SignalType::kInput},
                              {"in_vld", SignalType::kInput},
                              {"in_rdy", SignalType::kExpectedOutput},
                              {"in_rdy", SignalType::kOutput},
                              {"out", SignalType::kExpectedOutput},
                              {"out", SignalType::kOutput},
                              {"out_vld", SignalType::kExpectedOutput},
                              {"out_vld", SignalType::kOutput},
                              {"out_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs, expected_outputs));

  for (int64_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(outputs.at(i), expected_outputs.at(i));
  }
}

TEST_F(SimplePipelinedProcTest, BasicResetAndStall) {
  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           BuildBlockInPackage(/*stage_count=*/4, options));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Simple streaming pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  std::vector<absl::flat_hash_map<std::string, uint64_t>> expected_outputs;

  uint64_t running_in_val = 1;
  uint64_t running_out_val = 1;

  // During reset, the output will be invalid, but the pipeline
  // is open and the in data will flow through to the output.
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(0, 9, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      0, 9, {{"rst", 1}, {"in_vld", 1}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      0, 2, {{"in_rdy", 1}, {"out_vld", 0}, {"out", 0}}, expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(1, 1, {{"out", 0}}, expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(3, 9, {{"in_rdy", 1}, {"out_vld", 0}},
                                     expected_outputs));
  XLS_ASSERT_OK_AND_ASSIGN(running_out_val,
                           SetIncrementingSignalOverCycles(
                               3, 9, "out", running_out_val, expected_outputs));
  // The way SDC schedules this requires this, because there's a not node after
  // a register in stage 0, so on reset that outputs !0 = INT_MAX.
  // We can't easily change SimplePipelinedProcTest to use a manual schedule
  // because it accepts the number of stages as a parameter.
  expected_outputs.at(2).at("out") = std::numeric_limits<uint32_t>::max();

  // Once reset is deasserted, then the pipeline is closed, no further changes
  // in the output is expected if the input is not valid.
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(10, 19, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 19, {{"rst", 0}, {"in_vld", 0}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 19, {{"in_rdy", 0}, {"out_vld", 0}, {"out", running_out_val}},
      expected_outputs));

  // Returning input_valid, output will reflect valid input upon pipeline delay.
  uint64_t prior_running_out_val = running_out_val;
  running_out_val = running_in_val;
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(20, 29, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      20, 29, {{"rst", 0}, {"in_vld", 1}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      20, 22, {{"in_rdy", 1}, {"out_vld", 0}, {"out", prior_running_out_val}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(23, 29, {{"in_rdy", 1}, {"out_vld", 1}},
                                     expected_outputs));
  XLS_ASSERT_OK_AND_ASSIGN(
      running_out_val, SetIncrementingSignalOverCycles(
                           23, 29, "out", running_out_val, expected_outputs));

  // Output can stall the pipeline. and in_vld will reflect that as the pipe
  // is currently full, out_rdy will continue to assert ready data.
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(30, 35, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      30, 35, {{"rst", 0}, {"in_vld", 1}, {"out_rdy", 0}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      30, 35, {{"in_rdy", 0}, {"out_vld", 1}, {"out", running_out_val}},
      expected_outputs));

  // output_rdy becoming true will allow pipeline to drain even if input_vld
  // becomes false.  Once the pipe is drained, out_vld is deasserted.
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(36, 39, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      36, 39, {{"rst", 0}, {"in_vld", 0}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(36, 38, {{"in_rdy", 0}, {"out_vld", 1}},
                                     expected_outputs));
  XLS_ASSERT_OK_AND_ASSIGN(
      running_out_val, SetIncrementingSignalOverCycles(
                           36, 38, "out", running_out_val, expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      39, 39, {{"in_rdy", 0}, {"out_vld", 0}, {"out", running_out_val - 1}},
      expected_outputs));

  // Input rdy becoming true will allow the pipeline to fill.
  // Create a single bubble and allow pipeline to fill it
  //  Cycle    S0 | S1 | S2 | S3
  //   40      41 | [] | [] | [] -- in_vld = 1 , in_rdy = 1,  out_rdy = 1
  //   41      42 | 41 | [] | []
  //   42      [] | 42 | 41 | [] -- in_vld = 0, in_rdy = 0
  //   43      44 | [] | 42 | 41 -- in_vld = 1, in_rdy = 1, out_rdy = 0
  //   44      45 | 44 | 42 | 41 -- in_vld = 1, in_rdy = 0
  //   45      46 | 44 | 42 | 41
  //   46      47 | 44 | 42 | 41 -- in_vld = 1, in_rdy =  1, out_rdy = 1
  //   47      48 | 47 | 44 | 42
  //   48      49 | 48 | 47 | 44
  //   49      50 | 49 | 48 | 47
  prior_running_out_val = running_out_val - 1;
  running_out_val = running_in_val;
  XLS_ASSERT_OK_AND_ASSIGN(
      running_in_val,
      SetIncrementingSignalOverCycles(40, 59, "in", running_in_val, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(40, 59, {{"rst", 0}}, inputs));
  XLS_ASSERT_OK(
      SetSignalsOverCycles(40, 41, {{"in_vld", 1}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(
      SetSignalsOverCycles(42, 42, {{"in_vld", 0}, {"out_rdy", 1}}, inputs));
  XLS_ASSERT_OK(
      SetSignalsOverCycles(43, 45, {{"in_vld", 1}, {"out_rdy", 0}}, inputs));
  XLS_ASSERT_OK(
      SetSignalsOverCycles(46, 59, {{"in_vld", 1}, {"out_rdy", 1}}, inputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(
      40, 41, {{"in_rdy", 1}, {"out_vld", 0}, {"out", prior_running_out_val}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      42, 42, {{"in_rdy", 0}, {"out_vld", 0}, {"out", prior_running_out_val}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      43, 43, {{"in_rdy", 1}, {"out_vld", 1}, {"out", running_out_val}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      44, 45, {{"in_rdy", 0}, {"out_vld", 1}, {"out", running_out_val}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      46, 46, {{"in_rdy", 1}, {"out_vld", 1}, {"out", running_out_val}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      47, 47, {{"in_rdy", 1}, {"out_vld", 1}, {"out", running_out_val + 1}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      48, 48, {{"in_rdy", 1}, {"out_vld", 1}, {"out", running_out_val + 3}},
      expected_outputs));
  running_out_val = running_out_val + 6;
  XLS_ASSERT_OK(SetSignalsOverCycles(49, 59, {{"in_rdy", 1}, {"out_vld", 1}},
                                     expected_outputs));
  XLS_ASSERT_OK_AND_ASSIGN(
      running_out_val, SetIncrementingSignalOverCycles(
                           49, 59, "out", running_out_val, expected_outputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, expected_outputs.size() - 1,
                                                "cycle", 0, expected_outputs));
  ASSERT_EQ(inputs.size(), expected_outputs.size());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  ASSERT_EQ(outputs.size(), expected_outputs.size());

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                              {"rst", SignalType::kInput},
                              {"in", SignalType::kInput},
                              {"in_vld", SignalType::kInput},
                              {"in_rdy", SignalType::kExpectedOutput},
                              {"in_rdy", SignalType::kOutput},
                              {"out", SignalType::kExpectedOutput},
                              {"out", SignalType::kOutput},
                              {"out_vld", SignalType::kExpectedOutput},
                              {"out_vld", SignalType::kOutput},
                              {"out_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs, expected_outputs));

  for (int64_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(outputs.at(i), expected_outputs.at(i));
  }
}

// Fixture to sweep SimplePipelinedProcTest
class SimplePipelinedProcTestSweepFixture
    : public SimplePipelinedProcTest,
      public testing::WithParamInterface<
          std::tuple<int64_t, bool, bool, CodegenOptions::IOKind,
                     CodegenOptions::IOKind>> {
 public:
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    int64_t stage_count = std::get<0>(info.param);
    bool flop_inputs = std::get<1>(info.param);
    bool flop_outputs = std::get<2>(info.param);
    CodegenOptions::IOKind flop_inputs_kind = std::get<3>(info.param);
    CodegenOptions::IOKind flop_outputs_kind = std::get<4>(info.param);

    return absl::StrFormat(
        "stage_count_%d_flop_inputs_%d_flop_outputs_%d_"
        "flop_inputs_kind_%s_flop_outputs_kind_%s",
        stage_count, flop_inputs, flop_outputs,
        CodegenOptions::IOKindToString(flop_inputs_kind),
        CodegenOptions::IOKindToString(flop_outputs_kind));
  }
};

TEST_P(SimplePipelinedProcTestSweepFixture, RandomStalling) {
  int64_t stage_count = std::get<0>(GetParam());
  bool flop_inputs = std::get<1>(GetParam());
  bool flop_outputs = std::get<2>(GetParam());
  CodegenOptions::IOKind flop_inputs_kind = std::get<3>(GetParam());
  CodegenOptions::IOKind flop_outputs_kind = std::get<4>(GetParam());

  CodegenOptions options;
  options.flop_inputs(flop_inputs).flop_outputs(flop_outputs).clock_name("clk");
  options.flop_inputs_kind(flop_inputs_kind);
  options.flop_outputs_kind(flop_outputs_kind);
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      BuildBlockInPackage(/*stage_count=*/stage_count, options));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Simple streaming pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  // The input stimulus to this test are
  //  1. 10 cycles of reset
  //  2. Randomly varying in_vld and out_rdy.
  //  3. in_vld = 0 and out_rdy = 1 for 10 cycles to drain the pipeline
  int64_t simulation_cycle_count = 10000;
  int64_t max_random_cycle = simulation_cycle_count - 10 - 1;

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  XLS_ASSERT_OK(SetSignalsOverCycles(0, 9, {{"rst", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(10, simulation_cycle_count - 1,
                                     {{"rst", 0}}, inputs));

  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in", 1, inputs));

  std::minstd_rand rng_engine;
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "out_rdy", 0, 1,
                                          rng_engine, inputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(max_random_cycle + 1,
                                     simulation_cycle_count - 1,
                                     {{"in_vld", 0}, {"out_rdy", 1}}, inputs));

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                              {"rst", SignalType::kInput},
                              {"in", SignalType::kInput},
                              {"in_vld", SignalType::kInput},
                              {"in_rdy", SignalType::kOutput},
                              {"out", SignalType::kOutput},
                              {"out_vld", SignalType::kOutput},
                              {"out_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs));

  // Check the following property
  // 1. The sequence of inputs where (in_vld && in_rdy && !rst) is true
  //    is strictly monotone increasing with no duplicates.
  // 2. The sequence of outputs where out_vld && out_rdy is true
  //    is strictly monotone increasing with no duplicates.
  // 3. Both sequences in #1 and #2 are identical.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input_sequence,
      GetChannelSequenceFromIO({"in", SignalType::kInput},
                               {"in_vld", SignalType::kInput},
                               {"in_rdy", SignalType::kOutput},
                               {"rst", SignalType::kInput}, inputs, outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> output_sequence,
      GetChannelSequenceFromIO({"out", SignalType::kOutput},
                               {"out_vld", SignalType::kOutput},
                               {"out_rdy", SignalType::kInput},
                               {"rst", SignalType::kInput}, inputs, outputs));

  std::vector<uint64_t> input_value_sequence;
  std::vector<uint64_t> output_value_sequence;

  for (int64_t i = 0; i < input_sequence.size(); ++i) {
    uint64_t curr_value = input_sequence[i].value;

    if (i >= 1) {
      int64_t curr_cycle = input_sequence[i].cycle;
      uint64_t prior_value = input_sequence[i - 1].value;

      EXPECT_LT(prior_value, curr_value) << absl::StreamFormat(
          "Input not strictly monotone cycle %d "
          "got %d prior %d",
          curr_cycle, curr_value, prior_value);
    }

    input_value_sequence.push_back(curr_value);
  }

  for (int64_t i = 0; i < output_sequence.size(); ++i) {
    uint64_t curr_value = output_sequence[i].value;
    if (i >= 1) {
      int64_t curr_cycle = output_sequence[i].cycle;
      uint64_t prior_value = output_sequence[i - 1].value;

      EXPECT_LT(prior_value, curr_value) << absl::StreamFormat(
          "Output not strictly monotone cycle %d "
          "got %d prior %d",
          curr_cycle, curr_value, prior_value);
    }

    output_value_sequence.push_back(curr_value);
  }

  EXPECT_EQ(input_value_sequence, output_value_sequence);
}

INSTANTIATE_TEST_SUITE_P(
    SimplePipelinedProcTestSweep, SimplePipelinedProcTestSweepFixture,
    testing::Combine(
        testing::Values(1, 2, 3, 4), testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer)),
    SimplePipelinedProcTestSweepFixture::PrintToStringParamName);

// Fixture used to test pipelined BlockConversion on a simple
// block with a running counter
class SimpleRunningCounterProcTestSweepFixture
    : public ProcConversionTestFixture,
      public testing::WithParamInterface<
          std::tuple<int64_t, bool, bool, bool, CodegenOptions::IOKind,
                     CodegenOptions::IOKind>> {
 public:
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    int64_t stage_count = std::get<0>(info.param);
    bool active_low_reset = std::get<1>(info.param);
    bool flop_inputs = std::get<2>(info.param);
    bool flop_outputs = std::get<3>(info.param);
    CodegenOptions::IOKind flop_inputs_kind = std::get<4>(info.param);
    CodegenOptions::IOKind flop_outputs_kind = std::get<5>(info.param);

    return absl::StrFormat(
        "stage_count_%d_active_low_reset_%d_"
        "flop_inputs_%d_flop_outputs_%d_"
        "flop_inputs_kind_%s_flop_outputs_kind_%s",
        stage_count, active_low_reset, flop_inputs, flop_outputs,
        CodegenOptions::IOKindToString(flop_inputs_kind),
        CodegenOptions::IOKindToString(flop_outputs_kind));
  }

 protected:
  absl::StatusOr<std::unique_ptr<Package>> BuildBlockInPackage(
      int64_t stage_count, const CodegenOptions& options) override {
    // Simple streaming one input and one output pipeline.
    auto package_ptr = std::make_unique<Package>(TestName());
    Package& package = *package_ptr;

    Type* u32 = package.GetBitsType(32);
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in,
        package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_out,
        package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

    TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
    BValue state = pb.StateElement("st", Value(UBits(0, 32)));
    BValue in_val = pb.Receive(ch_in);

    BValue next_state = pb.Add(in_val, state, SourceInfo(), "increment");

    BValue buffered_state = pb.Not(pb.Not(pb.Not(pb.Not(next_state))));
    pb.Send(ch_out, buffered_state);
    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build({next_state}));

    XLS_VLOG(2) << "Simple counting proc";
    XLS_VLOG_LINES(2, proc->DumpIr());

    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::Run(proc, TestDelayEstimator(),
                                               SchedulingOptions()
                                                   .pipeline_stages(stage_count)
                                                   .clock_period_ps(10)));

    CodegenOptions codegen_options = options;
    codegen_options.module_name(kBlockName);

    XLS_RET_CHECK_OK(ProcToPipelinedBlock(schedule, codegen_options, proc));

    return package_ptr;
  }
};

TEST_P(SimpleRunningCounterProcTestSweepFixture, RandomStalling) {
  int64_t stage_count = std::get<0>(GetParam());
  bool active_low_reset = std::get<1>(GetParam());
  bool flop_inputs = std::get<2>(GetParam());
  bool flop_outputs = std::get<3>(GetParam());
  CodegenOptions::IOKind flop_inputs_kind = std::get<4>(GetParam());
  CodegenOptions::IOKind flop_outputs_kind = std::get<5>(GetParam());

  CodegenOptions options;
  options.flop_inputs(flop_inputs).flop_outputs(flop_outputs).clock_name("clk");
  options.flop_inputs_kind(flop_inputs_kind);
  options.flop_outputs_kind(flop_outputs_kind);
  options.valid_control("input_valid", "output_valid");
  options.reset(active_low_reset ? "rst_n" : "rst", false,
                /*active_low=*/active_low_reset, false);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      BuildBlockInPackage(/*stage_count=*/stage_count, options));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Simple counting pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  // The input stimulus to this test are
  //  1. 10 cycles of reset
  //  2. Randomly varying in_vld and out_rdy.
  //  3. in_vld = 0 and out_rdy = 1 for 10 cycles to drain the pipeline

  const char* reset_signal = active_low_reset ? "rst_n" : "rst";
  int64_t reset_active = active_low_reset ? 0 : 1;
  int64_t reset_inactive = active_low_reset ? 1 : 0;

  int64_t simulation_cycle_count = 10000;
  int64_t max_random_cycle = simulation_cycle_count - 10 - 1;

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  XLS_ASSERT_OK(
      SetSignalsOverCycles(0, 9, {{reset_signal, reset_active}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(10, simulation_cycle_count - 1,
                                     {{reset_signal, reset_inactive}}, inputs));

  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in", 1, inputs));

  std::minstd_rand rng_engine;
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "out_rdy", 0, 1,
                                          rng_engine, inputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(max_random_cycle + 1,
                                     simulation_cycle_count - 1,
                                     {{"in_vld", 0}, {"out_rdy", 1}}, inputs));

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{
          {"cycle", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset},
          {"in", SignalType::kInput},
          {"in_vld", SignalType::kInput},
          {"in_rdy", SignalType::kOutput},
          {"out", SignalType::kOutput},
          {"out_vld", SignalType::kOutput},
          {"out_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs));

  // Check the following property
  // 1. The sequence of inputs where (in_vld && in_rdy && !rst) is true
  //    is strictly monotone increasing with no duplicates.
  // 2. The sequence of outputs where out_vld && out_rdy is true
  //    is strictly monotone increasing with no duplicates.
  // 3. The sum of input_sequence is the last element of the output_sequence.
  // 4. The first value of the input and output sequences are the same.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input_sequence,
      GetChannelSequenceFromIO(
          {"in", SignalType::kInput}, {"in_vld", SignalType::kInput},
          {"in_rdy", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> output_sequence,
      GetChannelSequenceFromIO(
          {"out", SignalType::kOutput}, {"out_vld", SignalType::kOutput},
          {"out_rdy", SignalType::kInput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  for (int64_t i = 0; i < input_sequence.size(); ++i) {
    uint64_t curr_value = input_sequence[i].value;

    if (i >= 1) {
      int64_t curr_cycle = input_sequence[i].cycle;
      uint64_t prior_value = input_sequence[i - 1].value;

      EXPECT_LT(prior_value, curr_value) << absl::StreamFormat(
          "Input not strictly monotone cycle %d "
          "got %d prior %d",
          curr_cycle, curr_value, prior_value);
    }
  }

  for (int64_t i = 0; i < output_sequence.size(); ++i) {
    uint64_t curr_value = output_sequence[i].value;
    if (i >= 1) {
      int64_t curr_cycle = output_sequence[i].cycle;
      uint64_t prior_value = output_sequence[i - 1].value;

      EXPECT_LT(prior_value, curr_value) << absl::StreamFormat(
          "Output not strictly monotone cycle %d "
          "got %d prior %d",
          curr_cycle, curr_value, prior_value);
    }
  }

  int64_t in_sum = 0;
  for (CycleAndValue cv : input_sequence) {
    in_sum += cv.value;
  }

  EXPECT_EQ(input_sequence.front().value, output_sequence.front().value);
  EXPECT_EQ(in_sum, output_sequence.back().value);
}

INSTANTIATE_TEST_SUITE_P(
    SimpleRunningCounterProcTestSweep, SimpleRunningCounterProcTestSweepFixture,
    testing::Combine(
        testing::Values(1, 2, 3), testing::Values(false, true),
        testing::Values(false, true), testing::Values(false, true),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer)),
    SimpleRunningCounterProcTestSweepFixture::PrintToStringParamName);

// Fixture used to test pipelined BlockConversion on a multi input  block.
class MultiInputPipelinedProcTest : public ProcConversionTestFixture {
 protected:
  absl::StatusOr<std::unique_ptr<Package>> BuildBlockInPackage(
      int64_t stage_count, const CodegenOptions& options) override {
    // Simple streaming one input and one output pipeline.
    auto package_ptr = std::make_unique<Package>(TestName());
    Package& package = *package_ptr;

    Type* u32 = package.GetBitsType(32);
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in0,
        package.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in1,
        package.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_out,
        package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

    TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);

    BValue in0_val = pb.Receive(ch_in0);
    BValue in1_val = pb.Receive(ch_in1);

    BValue sum_val = pb.Add(in0_val, in1_val);
    pb.Send(ch_out, sum_val);
    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build({}));

    XLS_VLOG(2) << "Multi input streaming proc";
    XLS_VLOG_LINES(2, proc->DumpIr());

    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
            proc, TestDelayEstimator(),
            SchedulingOptions()
                .pipeline_stages(stage_count)
                .add_constraint(RecvsFirstSendsLastConstraint())));

    CodegenOptions codegen_options = options;
    codegen_options.module_name(kBlockName);

    XLS_RET_CHECK_OK(ProcToPipelinedBlock(schedule, codegen_options, proc));

    return package_ptr;
  }
};

// Fixture to sweep MultiInputProcPipelinedTest
class MultiInputPipelinedProcTestSweepFixture
    : public MultiInputPipelinedProcTest,
      public testing::WithParamInterface<
          std::tuple<int64_t, bool, bool, CodegenOptions::IOKind,
                     CodegenOptions::IOKind>> {
 public:
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    int64_t stage_count = std::get<0>(info.param);
    bool flop_inputs = std::get<1>(info.param);
    bool flop_outputs = std::get<2>(info.param);
    CodegenOptions::IOKind flop_inputs_kind = std::get<3>(info.param);
    CodegenOptions::IOKind flop_outputs_kind = std::get<4>(info.param);

    return absl::StrFormat(
        "stage_count_%d_flop_inputs_%d_flop_outputs_%d_"
        "flop_inputs_kind_%s_flop_outputs_kind_%s",
        stage_count, flop_inputs, flop_outputs,
        CodegenOptions::IOKindToString(flop_inputs_kind),
        CodegenOptions::IOKindToString(flop_outputs_kind));
  }
};

TEST_P(MultiInputPipelinedProcTestSweepFixture, RandomStalling) {
  int64_t stage_count = std::get<0>(GetParam());
  bool flop_inputs = std::get<1>(GetParam());
  bool flop_outputs = std::get<2>(GetParam());
  CodegenOptions::IOKind flop_inputs_kind = std::get<3>(GetParam());
  CodegenOptions::IOKind flop_outputs_kind = std::get<4>(GetParam());
  bool active_low_reset = true;

  CodegenOptions options;
  options.flop_inputs(flop_inputs).flop_outputs(flop_outputs).clock_name("clk");
  options.flop_inputs_kind(flop_inputs_kind);
  options.flop_outputs_kind(flop_outputs_kind);
  options.valid_control("input_valid", "output_valid");
  options.reset("rst_n", false, /*active_low=*/active_low_reset, false);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      BuildBlockInPackage(/*stage_count=*/stage_count, options));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Multi input counting pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  // The input stimulus to this test are
  //  1. 10 cycles of reset
  //  2. Randomly varying in_vld and out_rdy.
  //  3. in_vld = 0 and out_rdy = 1 for 10 cycles to drain the pipeline

  const char* reset_signal = "rst_n";
  int64_t reset_active = 0;
  int64_t reset_inactive = 1;

  int64_t simulation_cycle_count = 10000;
  int64_t max_random_cycle = simulation_cycle_count - 10 - 1;

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  XLS_ASSERT_OK(
      SetSignalsOverCycles(0, 9, {{reset_signal, reset_active}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(10, simulation_cycle_count - 1,
                                     {{reset_signal, reset_inactive}}, inputs));

  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in0", 1, inputs));
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in1", 1, inputs));

  std::minstd_rand rng_engine;
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in0_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in1_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "out_rdy", 0, 1,
                                          rng_engine, inputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(
      max_random_cycle + 1, simulation_cycle_count - 1,
      {{"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}}, inputs));

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{
          {"cycle", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset},
          {"in0", SignalType::kInput},
          {"in0_vld", SignalType::kInput},
          {"in0_rdy", SignalType::kOutput},
          {"in1", SignalType::kInput},
          {"in1_vld", SignalType::kInput},
          {"in1_rdy", SignalType::kOutput},
          {"out", SignalType::kOutput},
          {"out_vld", SignalType::kOutput},
          {"out_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs));

  // Check the following property
  // 1. The number of outputs is the same as the number of inputs within
  //    at most one additional input.
  //    is strictly monotone increasing with no duplicates.
  // 2. The sequence of outputs is implied by the sum of the
  //    sequennce of inputs.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input0_sequence,
      GetChannelSequenceFromIO(
          {"in0", SignalType::kInput}, {"in0_vld", SignalType::kInput},
          {"in0_rdy", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input1_sequence,
      GetChannelSequenceFromIO(
          {"in1", SignalType::kInput}, {"in1_vld", SignalType::kInput},
          {"in1_rdy", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> output_sequence,
      GetChannelSequenceFromIO(
          {"out", SignalType::kOutput}, {"out_vld", SignalType::kOutput},
          {"out_rdy", SignalType::kInput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  EXPECT_TRUE(output_sequence.size() == input0_sequence.size() ||
              output_sequence.size() + 1 == input0_sequence.size());
  EXPECT_TRUE(output_sequence.size() == input1_sequence.size() ||
              output_sequence.size() + 1 == input1_sequence.size());

  for (int64_t i = 0; i < output_sequence.size(); ++i) {
    int64_t in0_val = input0_sequence.at(i).value;
    int64_t in1_val = input1_sequence.at(i).value;
    int64_t out_val = output_sequence.at(i).value;

    int64_t expected_sum = in0_val + in1_val;

    EXPECT_EQ(out_val, expected_sum) << absl::StreamFormat(
        "Expected output index %d val %d == %d + %d, got %d, expected %d", i,
        out_val, in0_val, in1_val, out_val, expected_sum);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultiInputPipelinedProcTestSweep, MultiInputPipelinedProcTestSweepFixture,
    testing::Combine(
        testing::Values(1, 2, 3), testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer)),
    MultiInputPipelinedProcTestSweepFixture::PrintToStringParamName);

TEST_F(MultiInputPipelinedProcTest, IdleSignalNoFlops) {
  int64_t stage_count = 4;
  bool active_low_reset = true;

  CodegenOptions options;
  options.flop_inputs(true).flop_outputs(true).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst_n", false, /*active_low=*/active_low_reset, false);
  options.add_idle_output(true);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      BuildBlockInPackage(/*stage_count=*/stage_count, options));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Multi input counting pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  // The input stimulus to this test are
  //  8. 10 cycles of idle
  //  9. 10 cycles of data on in0 and in1
  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  std::vector<absl::flat_hash_map<std::string, uint64_t>> expected_outputs;

  // During reset, the output will be invalid, but the pipeline
  // is open and the in data will flow through to the output.

  //  1. 10 cycles of reset - idle will be 1
  XLS_ASSERT_OK(SetSignalsOverCycles(
      0, 9, {{"rst_n", 0}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      0, 9, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 1}},
      expected_outputs));

  //  2. 10 cycles of idle -- idle remains 1
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 19, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 19, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 1}},
      expected_outputs));

  //  3. 1 cycle of data on in0 - idle immediately becomes 0 due to
  //  combinational path
  //  4. 20 cycles of idle - idle continues to remain 0 as the pipeline is not
  //  flowing.
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 10, {{"rst_n", 1}, {"in0_vld", 1}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      10, 10, {{"in0_rdy", 1}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      11, 29, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      11, 29, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));

  //  5. 1 cycle of data on in1 -- allows 4-stage pipeline to drain
  //  6. After 5 more cycles (on 36th cycle), pipeline drains and block becomes
  //  idle
  XLS_ASSERT_OK(SetSignalsOverCycles(
      30, 30, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 1}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      30, 30, {{"in0_rdy", 0}, {"in1_rdy", 1}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      31, 39, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      31, 34, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      35, 35, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 1}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      36, 39, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 1}},
      expected_outputs));

  //  7. 1 cycle of data on in1 - idle immediately becomes 0 due to
  //  combinational path
  //  8. 20 cycles of idle - idle continues to remain 0 as the pipeline is not
  //  flowing.
  XLS_ASSERT_OK(SetSignalsOverCycles(
      40, 40, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 1}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      40, 40, {{"in0_rdy", 0}, {"in1_rdy", 1}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      41, 69, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      41, 69, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));

  //  9. 1 cycle of data on in0 -- allows 4-stage pipeline to drain
  // 10. After 5 more cycles (on 76th cycle), pipeline drains and block becomes
  // idle
  XLS_ASSERT_OK(SetSignalsOverCycles(
      70, 70, {{"rst_n", 1}, {"in0_vld", 1}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      70, 70, {{"in0_rdy", 1}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      71, 79, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      71, 74, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      75, 75, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 1}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      76, 79, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 1}},
      expected_outputs));

  // 11. Skipping cycle of valid input data, then drain the pipeline
  //   input on cycle 80 appears on the output on cycle 85
  //   input on cycle 83 appears on the output on cycle 88
  //   idle aserts on cycle 89
  XLS_ASSERT_OK(SetSignalsOverCycles(
      80, 80, {{"rst_n", 1}, {"in0_vld", 1}, {"in1_vld", 1}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      80, 80, {{"in0_rdy", 1}, {"in1_rdy", 1}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(
      81, 82, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      81, 82, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(
      83, 83, {{"rst_n", 1}, {"in0_vld", 1}, {"in1_vld", 1}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      83, 83, {{"in0_rdy", 1}, {"in1_rdy", 1}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      84, 84, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(
      84, 89, {{"rst_n", 1}, {"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      85, 85, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 1}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      86, 87, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      88, 88, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 1}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      89, 89, {{"in0_rdy", 0}, {"in1_rdy", 0}, {"out_vld", 0}, {"idle", 1}},
      expected_outputs));

  // 12. Continuous data for 10 cycles means that idle becomes 0 again.
  //     input on cycle 90 appears on the output on cycle 94
  XLS_ASSERT_OK(SetSignalsOverCycles(
      90, 99, {{"rst_n", 1}, {"in0_vld", 1}, {"in1_vld", 1}, {"out_rdy", 1}},
      inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      90, 94, {{"in0_rdy", 1}, {"in1_rdy", 1}, {"out_vld", 0}, {"idle", 0}},
      expected_outputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(
      95, 99, {{"in0_rdy", 1}, {"in1_rdy", 1}, {"out_vld", 1}, {"idle", 0}},
      expected_outputs));

  // Fill in the input data
  uint64_t running_in_val = 0;
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, inputs.size() - 1, "in0",
                                                running_in_val, inputs));
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, inputs.size() - 1, "in1",
                                                running_in_val, inputs));

  // Run interpreter
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, expected_outputs.size() - 1,
                                                "cycle", 0, expected_outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                              {"rst_n", SignalType::kInput, true},
                              {"in0", SignalType::kInput},
                              {"in0_vld", SignalType::kInput},
                              {"in0_rdy", SignalType::kOutput},
                              {"in1", SignalType::kInput},
                              {"in1_vld", SignalType::kInput},
                              {"in1_rdy", SignalType::kOutput},
                              {"out", SignalType::kOutput},
                              {"out_vld", SignalType::kOutput},
                              {"out_rdy", SignalType::kInput},
                              {"idle", SignalType::kOutput}},
      /*column_width=*/10, inputs, outputs));

  ASSERT_EQ(inputs.size(), expected_outputs.size());
  ASSERT_EQ(outputs.size(), expected_outputs.size());

  for (int64_t i = 0; i < outputs.size(); ++i) {
    // ignore the actual value of the output
    outputs[i].erase("out");

    EXPECT_EQ(outputs.at(i), expected_outputs.at(i));
  }
}

// Fixture used to test pipelined BlockConversion on a multi input block,
// that has multiple state elements.
class MultiInputWithStatePipelinedProcTest : public ProcConversionTestFixture {
 protected:
  absl::StatusOr<std::unique_ptr<Package>> BuildBlockInPackage(
      int64_t stage_count, const CodegenOptions& options) override {
    // Simple streaming one input and one output pipeline.
    auto package_ptr = std::make_unique<Package>(TestName());
    Package& package = *package_ptr;

    Type* u32 = package.GetBitsType(32);
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in0,
        package.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in1,
        package.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_out,
        package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

    Value initial_state = Value(UBits(0, 32));
    TokenlessProcBuilder pb(TestName(),
                            /*token_name=*/"tkn", &package);

    BValue accum0 = pb.StateElement("accum0", Value(UBits(0, 32)));
    BValue accum1 = pb.StateElement("accum1", Value(UBits(0, 32)));

    BValue in0_val = pb.Receive(ch_in0);
    BValue in1_val = pb.Receive(ch_in1);

    BValue next_accum0 = pb.Add(accum0, in0_val);
    BValue next_accum1 = pb.Add(accum1, in1_val);
    BValue sum = pb.Add(next_accum0, next_accum1);

    pb.Send(ch_out, sum);
    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build({next_accum0, next_accum1}));

    XLS_VLOG(2) << "Multi input streaming proc";
    XLS_VLOG_LINES(2, proc->DumpIr());

    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::Run(
                             proc, TestDelayEstimator(),
                             SchedulingOptions().pipeline_stages(stage_count)));

    CodegenOptions codegen_options = options;
    codegen_options.module_name(kBlockName);

    XLS_RET_CHECK_OK(ProcToPipelinedBlock(schedule, codegen_options, proc));

    return package_ptr;
  }
};

// Fixture to sweep MultiInputWithStatePipelinedProcTest
class MultiInputWithStatePipelinedProcTestSweepFixture
    : public MultiInputWithStatePipelinedProcTest,
      public testing::WithParamInterface<
          std::tuple<int64_t, bool, bool, CodegenOptions::IOKind,
                     CodegenOptions::IOKind>> {
 public:
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    int64_t stage_count = std::get<0>(info.param);
    bool flop_inputs = std::get<1>(info.param);
    bool flop_outputs = std::get<2>(info.param);
    CodegenOptions::IOKind flop_inputs_kind = std::get<3>(info.param);
    CodegenOptions::IOKind flop_outputs_kind = std::get<4>(info.param);

    return absl::StrFormat(
        "stage_count_%d_flop_inputs_%d_flop_outputs_%d_"
        "flop_inputs_kind_%s_flop_outputs_kind_%s",
        stage_count, flop_inputs, flop_outputs,
        CodegenOptions::IOKindToString(flop_inputs_kind),
        CodegenOptions::IOKindToString(flop_outputs_kind));
  }
};

TEST_P(MultiInputWithStatePipelinedProcTestSweepFixture, RandomStalling) {
  int64_t stage_count = std::get<0>(GetParam());
  bool flop_inputs = std::get<1>(GetParam());
  bool flop_outputs = std::get<2>(GetParam());
  CodegenOptions::IOKind flop_inputs_kind = std::get<3>(GetParam());
  CodegenOptions::IOKind flop_outputs_kind = std::get<4>(GetParam());
  bool add_idle_output = true;
  bool active_low_reset = true;

  CodegenOptions options;
  options.flop_inputs(flop_inputs).flop_outputs(flop_outputs).clock_name("clk");
  options.flop_inputs_kind(flop_inputs_kind);
  options.flop_outputs_kind(flop_outputs_kind);
  options.add_idle_output(add_idle_output);
  // This block has no single_value_channels so this is simply testing a NOP.
  options.flop_single_value_channels(false);
  options.valid_control("input_valid", "output_valid");
  options.reset("rst_n", false, /*active_low=*/active_low_reset, false);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      BuildBlockInPackage(/*stage_count=*/stage_count, options));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Multi input counting pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  // The input stimulus to this test are
  //  1. 10 cycles of reset
  //  2. Randomly varying in_vld and out_rdy.
  //  3. in_vld = 0 and out_rdy = 1 for 10 cycles to drain the pipeline

  const char* reset_signal = "rst_n";
  int64_t reset_active = 0;
  int64_t reset_inactive = 1;

  int64_t simulation_cycle_count = 10000;
  int64_t max_random_cycle = simulation_cycle_count - 10 - 1;

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  XLS_ASSERT_OK(
      SetSignalsOverCycles(0, 9, {{reset_signal, reset_active}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(10, simulation_cycle_count - 1,
                                     {{reset_signal, reset_inactive}}, inputs));

  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in0", 1, inputs));
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in1", 1, inputs));

  std::minstd_rand rng_engine;
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in0_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in1_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "out_rdy", 0, 1,
                                          rng_engine, inputs));

  XLS_ASSERT_OK(SetSignalsOverCycles(
      max_random_cycle + 1, simulation_cycle_count - 1,
      {{"in0_vld", 0}, {"in1_vld", 0}, {"out_rdy", 1}}, inputs));

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{
          {"cycle", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset},
          {"in0", SignalType::kInput},
          {"in0_vld", SignalType::kInput},
          {"in0_rdy", SignalType::kOutput},
          {"in1", SignalType::kInput},
          {"in1_vld", SignalType::kInput},
          {"in1_rdy", SignalType::kOutput},
          {"out", SignalType::kOutput},
          {"out_vld", SignalType::kOutput},
          {"out_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs));

  // Check the following property
  // 1. The sequence of outputs is implied by the running
  //    sum of the sequennce of inputs.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input0_sequence,
      GetChannelSequenceFromIO(
          {"in0", SignalType::kInput}, {"in0_vld", SignalType::kInput},
          {"in0_rdy", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input1_sequence,
      GetChannelSequenceFromIO(
          {"in1", SignalType::kInput}, {"in1_vld", SignalType::kInput},
          {"in1_rdy", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> output_sequence,
      GetChannelSequenceFromIO(
          {"out", SignalType::kOutput}, {"out_vld", SignalType::kOutput},
          {"out_rdy", SignalType::kInput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  int64_t prior_sum = 0;

  for (int64_t i = 0; i < output_sequence.size(); ++i) {
    int64_t in0_val = input0_sequence.at(i).value;
    int64_t in1_val = input1_sequence.at(i).value;
    int64_t out_val = output_sequence.at(i).value;

    int64_t expected_sum = in0_val + in1_val + prior_sum;

    EXPECT_EQ(out_val, expected_sum) << absl::StreamFormat(
        "Expected output index %d val %d == %d + %d + %d, got %d, expected %d",
        i, out_val, in0_val, prior_sum, in1_val, out_val, expected_sum);

    prior_sum = expected_sum;
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultiInputWithStatePipelinedProcTestSweep,
    MultiInputWithStatePipelinedProcTestSweepFixture,
    testing::Combine(
        testing::Values(1, 2, 3), testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer)),
    MultiInputWithStatePipelinedProcTestSweepFixture::PrintToStringParamName);

TEST_F(BlockConversionTest, BlockWithNonMutuallyExclusiveSends) {
  auto package_ptr = std::make_unique<Package>(TestName());
  Package& package = *package_ptr;

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));

  Value initial_state = Value(UBits(0, 32));
  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);

  BValue in_val = pb.Receive(in);

  BValue two = pb.Literal(UBits(2, 32));
  BValue three = pb.Literal(UBits(3, 32));

  pb.SendIf(out0, pb.ULt(in_val, two), in_val);
  pb.SendIf(out1, pb.ULt(in_val, three), in_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_VLOG_LINES(2, proc->DumpIr());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(2)));

  // Pipelined test
  {
    CodegenOptions options;
    options.module_name(TestName());
    options.flop_inputs(true).flop_outputs(true).clock_name("clk");
    options.valid_control("input_valid", "output_valid");
    options.reset("rst_n", false, /*active_low=*/true, false);

    XLS_EXPECT_OK(ProcToPipelinedBlock(schedule, options, proc).status());
  }

  // Combinational test
  {
    CodegenOptions options = codegen_options();
    options.module_name(TestName());
    options.valid_control("input_valid", "output_valid");

    EXPECT_THAT(ProcToCombinationalBlock(proc, options).status(),
                status_testing::StatusIs(
                    absl::StatusCode::kUnimplemented,
                    testing::HasSubstr("not proven to be mutually exclusive")));
  }
}

// Fixture used to test pipelined BlockConversion on a multi input and
// output block, that has state.
class MultiIOWithStatePipelinedProcTest : public ProcConversionTestFixture {
 protected:
  absl::StatusOr<std::unique_ptr<Package>> BuildBlockInPackage(
      int64_t stage_count, const CodegenOptions& options) override {
    auto package_ptr = std::make_unique<Package>(TestName());
    Package& package = *package_ptr;

    Type* u32 = package.GetBitsType(32);
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in0,
        package.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_in1,
        package.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_out0,
        package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * ch_out1,
        package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));

    TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);
    BValue state = pb.StateElement("st", Value(UBits(0, 32)));

    BValue in0_val = pb.Receive(ch_in0);
    BValue in1_val = pb.Receive(ch_in1);

    BValue increment = pb.Add(in0_val, in1_val);
    BValue next_state = pb.Add(state, increment);

    pb.Send(ch_out0, next_state);

    BValue state_plus_in1 = pb.Add(state, in1_val);
    pb.Send(ch_out1, state_plus_in1);

    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build({next_state}));

    XLS_VLOG(2) << "Multi io streaming proc";
    XLS_VLOG_LINES(2, proc->DumpIr());

    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::Run(
                             proc, TestDelayEstimator(),
                             SchedulingOptions().pipeline_stages(stage_count)));

    CodegenOptions codegen_options = options;
    codegen_options.module_name(kBlockName);

    XLS_RET_CHECK_OK(ProcToPipelinedBlock(schedule, codegen_options, proc));

    return package_ptr;
  }
};

// Fixture to sweep MultiIOWithStatePipelinedProcTest
//
// Sweep parameters are (stage_count, flop_inputs, flop_outputs,
// flop_output_kind).
class MultiIOWithStatePipelinedProcTestSweepFixture
    : public MultiIOWithStatePipelinedProcTest,
      public testing::WithParamInterface<
          std::tuple<int64_t, bool, bool, CodegenOptions::IOKind,
                     CodegenOptions::IOKind>> {
 public:
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    int64_t stage_count = std::get<0>(info.param);
    bool flop_inputs = std::get<1>(info.param);
    bool flop_outputs = std::get<2>(info.param);
    CodegenOptions::IOKind flop_inputs_kind = std::get<3>(info.param);
    CodegenOptions::IOKind flop_outputs_kind = std::get<4>(info.param);

    return absl::StrFormat(
        "stage_count_%d_flop_inputs_%d_flop_outputs_%d_"
        "flop_inputs_kind_%s_flop_outputs_kind_%s",
        stage_count, flop_inputs, flop_outputs,
        CodegenOptions::IOKindToString(flop_inputs_kind),
        CodegenOptions::IOKindToString(flop_outputs_kind));
  }
};

TEST_P(MultiIOWithStatePipelinedProcTestSweepFixture, RandomStalling) {
  int64_t stage_count = std::get<0>(GetParam());
  bool flop_inputs = std::get<1>(GetParam());
  bool flop_outputs = std::get<2>(GetParam());
  CodegenOptions::IOKind flop_inputs_kind = std::get<3>(GetParam());
  CodegenOptions::IOKind flop_outputs_kind = std::get<4>(GetParam());
  bool add_idle_output = true;
  bool active_low_reset = true;

  CodegenOptions options;
  options.flop_inputs(flop_inputs).flop_outputs(flop_outputs).clock_name("clk");
  options.flop_inputs_kind(flop_inputs_kind);
  options.flop_outputs_kind(flop_outputs_kind);
  options.add_idle_output(add_idle_output);
  options.valid_control("input_valid", "output_valid");
  options.reset("rst_n", false, /*active_low=*/active_low_reset, false);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      BuildBlockInPackage(/*stage_count=*/stage_count, options));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Multi io counting pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  // The input stimulus to this test are
  //  1. 10 cycles of reset
  //  2. Randomly varying in_vld and out_rdy.
  //  3. in_vld = 0 and out_rdy = 1 for 10 cycles to drain the pipeline

  const char* reset_signal = "rst_n";
  int64_t reset_active = 0;
  int64_t reset_inactive = 1;

  int64_t simulation_cycle_count = 10000;

  std::vector<absl::flat_hash_map<std::string, uint64_t>> non_streaming_inputs;
  XLS_ASSERT_OK(SetSignalsOverCycles(0, 9, {{reset_signal, reset_active}},
                                     non_streaming_inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(10, simulation_cycle_count - 1,
                                     {{reset_signal, reset_inactive}},
                                     non_streaming_inputs));

  std::vector<uint64_t> in_values(simulation_cycle_count);
  std::iota(in_values.begin(), in_values.end(), 0);

  std::vector<ChannelSource> sources{
      ChannelSource("in0", "in0_vld", "in0_rdy", 0.5, block),
      ChannelSource("in1", "in1_vld", "in1_rdy", 0.5, block),
  };
  XLS_ASSERT_OK(sources.at(0).SetDataSequence(in_values));
  XLS_ASSERT_OK(sources.at(1).SetDataSequence(in_values));

  std::vector<ChannelSink> sinks{
      ChannelSink("out0", "out0_vld", "out0_rdy", 0.5, block),
      ChannelSink("out1", "out1_vld", "out1_rdy", 0.5, block),
  };

  BlockIOResultsAsUint64 io_results;
  std::vector<absl::flat_hash_map<std::string, uint64_t>>& inputs =
      io_results.inputs;
  std::vector<absl::flat_hash_map<std::string, uint64_t>>& outputs =
      io_results.outputs;

  XLS_ASSERT_OK_AND_ASSIGN(
      io_results, InterpretChannelizedSequentialBlockWithUint64(
                      block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
                      non_streaming_inputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{
          {"cycle", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset},
          {"in0", SignalType::kInput},
          {"in0_vld", SignalType::kInput},
          {"in0_rdy", SignalType::kOutput},
          {"in1", SignalType::kInput},
          {"in1_vld", SignalType::kInput},
          {"in1_rdy", SignalType::kOutput},
          {"out0", SignalType::kOutput},
          {"out0_vld", SignalType::kOutput},
          {"out0_rdy", SignalType::kInput},
          {"out1", SignalType::kOutput},
          {"out1_vld", SignalType::kOutput},
          {"out1_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs));

  // Check the following property
  // 1. The sequence of outputs is implied by the running
  //    sum of the sequennce of inputs.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input0_sequence,
      GetChannelSequenceFromIO(
          {"in0", SignalType::kInput}, {"in0_vld", SignalType::kInput},
          {"in0_rdy", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input1_sequence,
      GetChannelSequenceFromIO(
          {"in1", SignalType::kInput}, {"in1_vld", SignalType::kInput},
          {"in1_rdy", SignalType::kOutput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> output0_sequence,
      GetChannelSequenceFromIO(
          {"out0", SignalType::kOutput}, {"out0_vld", SignalType::kOutput},
          {"out0_rdy", SignalType::kInput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> output1_sequence,
      GetChannelSequenceFromIO(
          {"out1", SignalType::kOutput}, {"out1_vld", SignalType::kOutput},
          {"out1_rdy", SignalType::kInput},
          {reset_signal, SignalType::kInput, active_low_reset}, inputs,
          outputs));

  EXPECT_GT(output0_sequence.size(), 1000);
  EXPECT_GT(output1_sequence.size(), 1000);

  int64_t min_output_count = output0_sequence.size() > output1_sequence.size()
                                 ? output1_sequence.size()
                                 : output0_sequence.size();

  int64_t prior_sum = 0;

  for (int64_t i = 0; i < min_output_count; ++i) {
    int64_t in0_val = input0_sequence.at(i).value;
    int64_t in1_val = input1_sequence.at(i).value;
    int64_t out0_val = output0_sequence.at(i).value;
    int64_t out1_val = output1_sequence.at(i).value;

    int64_t expected0_sum = in0_val + in1_val + prior_sum;
    int64_t expected1_sum = prior_sum + in1_val;

    EXPECT_EQ(out0_val, expected0_sum) << absl::StreamFormat(
        "Expected output0 index %d val %d == %d + %d + %d, got %d, expected %d",
        i, out0_val, in0_val, prior_sum, in1_val, out0_val, expected0_sum);

    EXPECT_EQ(out1_val, expected1_sum) << absl::StreamFormat(
        "Expected output0 index %d val %d == %d + %d + %d, got %d, expected %d",
        i, out1_val, in0_val, prior_sum, in1_val, out1_val, expected1_sum);

    prior_sum = expected0_sum;
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultiIOWithStatePipelinedProcTestSweep,
    MultiIOWithStatePipelinedProcTestSweepFixture,
    testing::Combine(
        testing::Values(1, 2, 3, 4), testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer),
        testing::Values(CodegenOptions::IOKind::kFlop,
                        CodegenOptions::IOKind::kSkidBuffer,
                        CodegenOptions::IOKind::kZeroLatencyBuffer)),
    MultiIOWithStatePipelinedProcTestSweepFixture::PrintToStringParamName);

TEST_F(BlockConversionTest, IOSignatureProcToPipelinedBlock) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * in_single_val,
                           package.CreateSingleValueChannel(
                               "in_single_val", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_streaming_rv,
      package.CreateStreamingChannel(
          "in_streaming", ChannelOps::kReceiveOnly, u32,
          /*initial_values=*/{}, /*fifo_depth=*/std::nullopt,
          FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_single_val,
                           package.CreateSingleValueChannel(
                               "out_single_val", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_streaming_rv,
      package.CreateStreamingChannel(
          "out_streaming", ChannelOps::kSendOnly, u32,
          /*initial_values=*/{}, /*fifo_depth=*/std::nullopt,
          FlowControl::kReadyValid));

  TokenlessProcBuilder pb(TestName(),
                          /*token_name=*/"tkn", &package);
  BValue in0 = pb.Receive(in_single_val);
  BValue in1 = pb.Receive(in_streaming_rv);
  pb.Send(out_single_val, in0);
  pb.Send(out_streaming_rv, in1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  EXPECT_FALSE(in_single_val->HasCompletedBlockPortNames());
  EXPECT_FALSE(out_single_val->HasCompletedBlockPortNames());
  EXPECT_FALSE(in_streaming_rv->HasCompletedBlockPortNames());
  EXPECT_FALSE(out_streaming_rv->HasCompletedBlockPortNames());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(1)));
  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToPipelinedBlock(schedule, options, proc));
  XLS_VLOG_LINES(2, block->DumpIr());

  EXPECT_TRUE(in_single_val->HasCompletedBlockPortNames());
  EXPECT_TRUE(out_single_val->HasCompletedBlockPortNames());
  EXPECT_TRUE(in_streaming_rv->HasCompletedBlockPortNames());
  EXPECT_TRUE(out_streaming_rv->HasCompletedBlockPortNames());

  EXPECT_EQ(in_single_val->GetBlockName().value(), "pipelined_proc");
  EXPECT_EQ(in_single_val->GetDataPortName().value(), "in_single_val");

  EXPECT_EQ(out_single_val->GetBlockName().value(), "pipelined_proc");
  EXPECT_EQ(out_single_val->GetDataPortName().value(), "out_single_val");

  EXPECT_EQ(in_streaming_rv->GetBlockName().value(), "pipelined_proc");
  EXPECT_EQ(in_streaming_rv->GetDataPortName().value(), "in_streaming_data");
  EXPECT_EQ(in_streaming_rv->GetValidPortName().value(), "in_streaming_valid");
  EXPECT_EQ(in_streaming_rv->GetReadyPortName().value(), "in_streaming_ready");

  EXPECT_EQ(out_streaming_rv->GetBlockName().value(), "pipelined_proc");
  EXPECT_EQ(out_streaming_rv->GetDataPortName().value(), "out_streaming_data");
  EXPECT_EQ(out_streaming_rv->GetValidPortName().value(),
            "out_streaming_valid");
  EXPECT_EQ(out_streaming_rv->GetReadyPortName().value(),
            "out_streaming_ready");
}

TEST_F(BlockConversionTest, IOSignatureProcToCombBlock) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * in_single_val,
                           package.CreateSingleValueChannel(
                               "in_single_val", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_streaming_rv,
      package.CreateStreamingChannel(
          "in_streaming", ChannelOps::kReceiveOnly, u32,
          /*initial_values=*/{}, /*fifo_depth=*/std::nullopt,
          FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_single_val,
                           package.CreateSingleValueChannel(
                               "out_single_val", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_streaming_rv,
      package.CreateStreamingChannel(
          "out_streaming", ChannelOps::kSendOnly, u32,
          /*initial_values=*/{}, /*fifo_depth=*/std::nullopt,
          FlowControl::kReadyValid));

  TokenlessProcBuilder pb(TestName(),
                          /*token_name=*/"tkn", &package);
  BValue in0 = pb.Receive(in_single_val);
  BValue in1 = pb.Receive(in_streaming_rv);
  pb.Send(out_single_val, in0);
  pb.Send(out_streaming_rv, in1);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  EXPECT_FALSE(in_single_val->HasCompletedBlockPortNames());
  EXPECT_FALSE(out_single_val->HasCompletedBlockPortNames());
  EXPECT_FALSE(in_streaming_rv->HasCompletedBlockPortNames());
  EXPECT_FALSE(out_streaming_rv->HasCompletedBlockPortNames());

  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block, ProcToCombinationalBlock(
                         proc, codegen_options().module_name("the_proc")));
  XLS_VLOG_LINES(2, block->DumpIr());

  EXPECT_TRUE(in_single_val->HasCompletedBlockPortNames());
  EXPECT_TRUE(out_single_val->HasCompletedBlockPortNames());
  EXPECT_TRUE(in_streaming_rv->HasCompletedBlockPortNames());
  EXPECT_TRUE(out_streaming_rv->HasCompletedBlockPortNames());

  EXPECT_EQ(in_single_val->GetBlockName().value(), "the_proc");
  EXPECT_EQ(in_single_val->GetDataPortName().value(), "in_single_val");

  EXPECT_EQ(out_single_val->GetBlockName().value(), "the_proc");
  EXPECT_EQ(out_single_val->GetDataPortName().value(), "out_single_val");

  EXPECT_EQ(in_streaming_rv->GetBlockName().value(), "the_proc");
  EXPECT_EQ(in_streaming_rv->GetDataPortName().value(), "in_streaming");
  EXPECT_EQ(in_streaming_rv->GetValidPortName().value(), "in_streaming_vld");
  EXPECT_EQ(in_streaming_rv->GetReadyPortName().value(), "in_streaming_rdy");

  EXPECT_EQ(out_streaming_rv->GetBlockName().value(), "the_proc");
  EXPECT_EQ(out_streaming_rv->GetDataPortName().value(), "out_streaming");
  EXPECT_EQ(out_streaming_rv->GetValidPortName().value(), "out_streaming_vld");
  EXPECT_EQ(out_streaming_rv->GetReadyPortName().value(), "out_streaming_rdy");
}

TEST_F(ProcConversionTestFixture, SimpleProcRandomScheduler) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));

  TokenlessProcBuilder pb(TestName(), "tkn", &package);
  BValue recv = pb.Receive(in);
  pb.Send(out, pb.Not(pb.Not(pb.Not(pb.Not(recv)))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  for (int32_t i = 0; i < 100; ++i) {
    int32_t seed = 100000 + 5000 * i;

    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(proc, TestDelayEstimator(),
                              SchedulingOptions(SchedulingStrategy::RANDOM)
                                  .pipeline_stages(20)
                                  .seed(seed)));
    CodegenOptions options;
    options.flop_inputs(false).flop_outputs(false).clock_name("clk");
    options.valid_control("input_valid", "output_valid");
    options.reset("rst", false, false, true);
    options.streaming_channel_data_suffix("_data");
    options.streaming_channel_valid_suffix("_valid");
    options.streaming_channel_ready_suffix("_ready");
    options.module_name("pipelined_proc");

    XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                             ProcToPipelinedBlock(schedule, options, proc));
    XLS_VLOG_LINES(2, block->DumpIr());

    int64_t input_count = 40;
    int64_t simulation_cycle_count = 200;

    std::vector<absl::flat_hash_map<std::string, uint64_t>>
        non_streaming_inputs(simulation_cycle_count, {{"rst", 0}});

    std::vector<uint64_t> in_values(input_count);
    std::iota(in_values.begin(), in_values.end(), 0);

    std::vector<ChannelSource> sources{
        ChannelSource("in_data", "in_valid", "in_ready", 0.5, block),
    };
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(in_values));

    std::vector<ChannelSink> sinks{
        ChannelSink("out_data", "out_valid", "out_ready", 0.5, block),
    };

    XLS_ASSERT_OK_AND_ASSIGN(BlockIOResultsAsUint64 io_results,
                             InterpretChannelizedSequentialBlockWithUint64(
                                 block, absl::MakeSpan(sources),
                                 absl::MakeSpan(sinks), non_streaming_inputs));

    std::vector<absl::flat_hash_map<std::string, uint64_t>>& inputs =
        io_results.inputs;
    std::vector<absl::flat_hash_map<std::string, uint64_t>>& outputs =
        io_results.outputs;

    // Add a cycle count for easier comparison with simulation results.
    XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1,
                                                  "cycle", 0, outputs));

    XLS_ASSERT_OK(VLogTestPipelinedIO(
        std::vector<SignalSpec>{
            {"cycle", SignalType::kOutput},
            {"rst", SignalType::kInput, /*active_low_reset=*/false},
            {"in_data", SignalType::kInput},
            {"in_valid", SignalType::kInput},
            {"in_ready", SignalType::kOutput},
            {"out_data", SignalType::kOutput},
            {"out_valid", SignalType::kOutput},
            {"out_ready", SignalType::kInput}},
        /*column_width=*/10, inputs, outputs));

    EXPECT_THAT(sinks.at(0).GetOutputSequenceAsUint64(),
                IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                         32, 33, 34, 35, 36, 37, 38, 39)));
  }
}

TEST_F(ProcConversionTestFixture, AddRandomScheduler) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_a,
      package.CreateStreamingChannel("in_a", ChannelOps::kReceiveOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_b,
      package.CreateStreamingChannel("in_b", ChannelOps::kReceiveOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));

  TokenlessProcBuilder pb(TestName(), "tkn", &package);
  BValue a = pb.Receive(in_a);
  BValue b = pb.Receive(in_b);
  pb.Send(out, pb.Not(pb.Not(pb.Not(pb.Not(pb.Add(a, b))))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  for (int32_t i = 0; i < 100; ++i) {
    int32_t seed = 100000 + 5000 * i;

    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(proc, TestDelayEstimator(),
                              SchedulingOptions(SchedulingStrategy::RANDOM)
                                  .pipeline_stages(20)
                                  .seed(seed)));
    CodegenOptions options;
    options.flop_inputs(false).flop_outputs(false).clock_name("clk");
    options.valid_control("input_valid", "output_valid");
    options.reset("rst", false, false, true);
    options.streaming_channel_data_suffix("_data");
    options.streaming_channel_valid_suffix("_valid");
    options.streaming_channel_ready_suffix("_ready");
    options.module_name("pipelined_proc");

    XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                             ProcToPipelinedBlock(schedule, options, proc));
    XLS_VLOG_LINES(2, block->DumpIr());

    int64_t input_count = 40;
    int64_t simulation_cycle_count = 200;

    std::vector<absl::flat_hash_map<std::string, uint64_t>>
        non_streaming_inputs(simulation_cycle_count, {{"rst", 0}});

    std::vector<uint64_t> in_values(input_count);
    std::iota(in_values.begin(), in_values.end(), 0);

    std::vector<ChannelSource> sources{
        ChannelSource("in_a_data", "in_a_valid", "in_a_ready", 0.5, block),
        ChannelSource("in_b_data", "in_b_valid", "in_b_ready", 0.5, block),
    };
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(in_values));
    XLS_ASSERT_OK(sources.at(1).SetDataSequence(in_values));

    std::vector<ChannelSink> sinks{
        ChannelSink("out_data", "out_valid", "out_ready", 0.5, block),
    };

    XLS_ASSERT_OK_AND_ASSIGN(BlockIOResultsAsUint64 io_results,
                             InterpretChannelizedSequentialBlockWithUint64(
                                 block, absl::MakeSpan(sources),
                                 absl::MakeSpan(sinks), non_streaming_inputs));

    std::vector<absl::flat_hash_map<std::string, uint64_t>>& inputs =
        io_results.inputs;
    std::vector<absl::flat_hash_map<std::string, uint64_t>>& outputs =
        io_results.outputs;

    // Add a cycle count for easier comparison with simulation results.
    XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1,
                                                  "cycle", 0, outputs));

    XLS_ASSERT_OK(VLogTestPipelinedIO(
        std::vector<SignalSpec>{
            {"cycle", SignalType::kOutput},
            {"rst", SignalType::kInput, /*active_low_reset=*/false},
            {"in_a_data", SignalType::kInput},
            {"in_a_valid", SignalType::kInput},
            {"in_a_ready", SignalType::kOutput},
            {"in_b_data", SignalType::kInput},
            {"in_b_valid", SignalType::kInput},
            {"in_b_ready", SignalType::kOutput},
            {"out_data", SignalType::kOutput},
            {"out_valid", SignalType::kOutput},
            {"out_ready", SignalType::kInput}},
        /*column_width=*/10, inputs, outputs));

    EXPECT_THAT(sinks.at(0).GetOutputSequenceAsUint64(),
                IsOkAndHolds(ElementsAre(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                                         22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                         42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
                                         62, 64, 66, 68, 70, 72, 74, 76, 78)));
  }
}

TEST_F(ProcConversionTestFixture, TwoReceivesTwoSendsRandomScheduler) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_a,
      package.CreateStreamingChannel("in_a", ChannelOps::kReceiveOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_b,
      package.CreateStreamingChannel("in_b", ChannelOps::kReceiveOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_a,
      package.CreateStreamingChannel("out_a", ChannelOps::kSendOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_b,
      package.CreateStreamingChannel("out_b", ChannelOps::kSendOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));

  TokenlessProcBuilder pb(TestName(), "tkn", &package);
  pb.Send(out_a, pb.Not(pb.Not(pb.Not(pb.Not(pb.Receive(in_a))))));
  pb.Send(out_b, pb.Not(pb.Not(pb.Not(pb.Not(pb.Receive(in_b))))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  for (int32_t i = 0; i < 100; ++i) {
    int32_t seed = 100000 + 5000 * i;

    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(proc, TestDelayEstimator(),
                              SchedulingOptions(SchedulingStrategy::RANDOM)
                                  .pipeline_stages(20)
                                  .seed(seed)));
    CodegenOptions options;
    options.flop_inputs(false).flop_outputs(false).clock_name("clk");
    options.valid_control("input_valid", "output_valid");
    options.reset("rst", false, false, true);
    options.streaming_channel_data_suffix("_data");
    options.streaming_channel_valid_suffix("_valid");
    options.streaming_channel_ready_suffix("_ready");
    options.module_name("pipelined_proc");

    XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                             ProcToPipelinedBlock(schedule, options, proc));
    XLS_VLOG_LINES(2, block->DumpIr());

    int64_t input_count = 40;
    int64_t simulation_cycle_count = 200;

    std::vector<absl::flat_hash_map<std::string, uint64_t>>
        non_streaming_inputs(simulation_cycle_count, {{"rst", 0}});

    std::vector<uint64_t> in_values(input_count);
    std::iota(in_values.begin(), in_values.end(), 0);

    std::vector<ChannelSource> sources{
        ChannelSource("in_a_data", "in_a_valid", "in_a_ready", 0.5, block),
        ChannelSource("in_b_data", "in_b_valid", "in_b_ready", 0.5, block),
    };
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(in_values));
    XLS_ASSERT_OK(sources.at(1).SetDataSequence(in_values));

    std::vector<ChannelSink> sinks{
        ChannelSink("out_a_data", "out_a_valid", "out_a_ready", 0.5, block),
        ChannelSink("out_b_data", "out_b_valid", "out_b_ready", 0.5, block),
    };

    XLS_ASSERT_OK_AND_ASSIGN(BlockIOResultsAsUint64 io_results,
                             InterpretChannelizedSequentialBlockWithUint64(
                                 block, absl::MakeSpan(sources),
                                 absl::MakeSpan(sinks), non_streaming_inputs));

    std::vector<absl::flat_hash_map<std::string, uint64_t>>& inputs =
        io_results.inputs;
    std::vector<absl::flat_hash_map<std::string, uint64_t>>& outputs =
        io_results.outputs;

    // Add a cycle count for easier comparison with simulation results.
    XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1,
                                                  "cycle", 0, outputs));

    XLS_ASSERT_OK(VLogTestPipelinedIO(
        std::vector<SignalSpec>{
            {"cycle", SignalType::kOutput},
            {"rst", SignalType::kInput, /*active_low_reset=*/false},
            {"in_a_data", SignalType::kInput},
            {"in_a_valid", SignalType::kInput},
            {"in_a_ready", SignalType::kOutput},
            {"in_b_data", SignalType::kInput},
            {"in_b_valid", SignalType::kInput},
            {"in_b_ready", SignalType::kOutput},
            {"out_a_data", SignalType::kOutput},
            {"out_a_valid", SignalType::kOutput},
            {"out_a_ready", SignalType::kInput},
            {"out_b_data", SignalType::kOutput},
            {"out_b_valid", SignalType::kOutput},
            {"out_b_ready", SignalType::kInput}},
        /*column_width=*/10, inputs, outputs));

    EXPECT_THAT(sinks.at(0).GetOutputSequenceAsUint64(),
                IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                         32, 33, 34, 35, 36, 37, 38, 39)));
    EXPECT_THAT(sinks.at(1).GetOutputSequenceAsUint64(),
                IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                         32, 33, 34, 35, 36, 37, 38, 39)));
  }
}

// Fixture used to test non-blocking receives.
class NonblockingReceivesProcTest : public ProcConversionTestFixture {
 protected:
  absl::StatusOr<std::unique_ptr<Package>> BuildBlockInPackage(
      int64_t stage_count, const CodegenOptions& options) override {
    auto package_ptr = std::make_unique<Package>(TestName());
    Package& package = *package_ptr;

    Type* u32 = package.GetBitsType(32);
    XLS_ASSIGN_OR_RETURN(
        Channel * in0,
        package.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * in1,
        package.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * in2,
        package.CreateSingleValueChannel("in2", ChannelOps::kReceiveOnly, u32));
    XLS_ASSIGN_OR_RETURN(
        Channel * out0,
        package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

    ProcBuilder pb(TestName(), /*token_name=*/"tok", &package);

    BValue in0_data_and_valid = pb.ReceiveNonBlocking(in0, pb.GetTokenParam());
    BValue in1_data_and_valid = pb.ReceiveNonBlocking(in1, pb.GetTokenParam());
    BValue in2_data_and_valid = pb.ReceiveNonBlocking(in2, pb.GetTokenParam());

    BValue sum = pb.Literal(UBits(0, 32));

    BValue in0_tok = pb.TupleIndex(in0_data_and_valid, 0);
    BValue in0_data = pb.TupleIndex(in0_data_and_valid, 1);
    BValue in0_valid = pb.TupleIndex(in0_data_and_valid, 2);
    BValue add_sum_in0 = pb.Add(sum, in0_data);
    BValue sum0 = pb.Select(in0_valid, {sum, add_sum_in0});

    BValue in1_tok = pb.TupleIndex(in1_data_and_valid, 0);
    BValue in1_data = pb.TupleIndex(in1_data_and_valid, 1);
    BValue in1_valid = pb.TupleIndex(in1_data_and_valid, 2);
    BValue add_sum0_in1 = pb.Add(sum0, in1_data);
    BValue sum1 = pb.Select(in1_valid, {sum0, add_sum0_in1});

    BValue in2_tok = pb.TupleIndex(in2_data_and_valid, 0);
    BValue in2_data = pb.TupleIndex(in2_data_and_valid, 1);
    BValue in2_valid = pb.TupleIndex(in2_data_and_valid, 2);
    BValue add_sum1_in2 = pb.Add(sum1, in2_data);
    BValue sum2 = pb.Select(in2_valid, {sum1, add_sum1_in2});

    BValue after_in_tok = pb.AfterAll({in0_tok, in1_tok, in2_tok});
    BValue tok_fin = pb.Send(out0, after_in_tok, sum2);

    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build(tok_fin, {}));

    XLS_VLOG(2) << "Non-blocking proc";
    XLS_VLOG_LINES(2, proc->DumpIr());

    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
            proc, TestDelayEstimator(),
            SchedulingOptions()
                .pipeline_stages(stage_count)
                .add_constraint(RecvsFirstSendsLastConstraint())));

    CodegenOptions codegen_options = options;
    codegen_options.module_name(kBlockName);

    XLS_RET_CHECK_OK(ProcToPipelinedBlock(schedule, codegen_options, proc));

    return package_ptr;
  }
};

// Fixture to sweep NonblockingReceivesProcTest
//
// Sweep parameters are (stage_count, flop_inputs, flop_outputs,
// flop_output_kind).
class NonblockingReceivesProcTestSweepFixture
    : public NonblockingReceivesProcTest,
      public testing::WithParamInterface<
          std::tuple<int64_t, bool, bool, CodegenOptions::IOKind,
                     CodegenOptions::IOKind>> {
 public:
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    int64_t stage_count = std::get<0>(info.param);
    bool flop_inputs = std::get<1>(info.param);
    bool flop_outputs = std::get<2>(info.param);
    CodegenOptions::IOKind flop_inputs_kind = std::get<3>(info.param);
    CodegenOptions::IOKind flop_outputs_kind = std::get<4>(info.param);

    return absl::StrFormat(
        "stage_count_%d_flop_inputs_%d_flop_outputs_%d_"
        "flop_inputs_kind_%s_flop_outputs_kind_%s",
        stage_count, flop_inputs, flop_outputs,
        CodegenOptions::IOKindToString(flop_inputs_kind),
        CodegenOptions::IOKindToString(flop_outputs_kind));
  }
};

TEST_P(NonblockingReceivesProcTestSweepFixture, RandomInput) {
  int64_t stage_count = std::get<0>(GetParam());
  bool flop_inputs = std::get<1>(GetParam());
  bool flop_outputs = std::get<2>(GetParam());
  CodegenOptions::IOKind flop_inputs_kind = std::get<3>(GetParam());
  CodegenOptions::IOKind flop_outputs_kind = std::get<4>(GetParam());
  bool add_idle_output = true;
  bool active_low_reset = false;

  CodegenOptions options;
  options.flop_inputs(flop_inputs).flop_outputs(flop_outputs).clock_name("clk");
  options.flop_inputs_kind(flop_inputs_kind);
  options.flop_outputs_kind(flop_outputs_kind);
  options.add_idle_output(add_idle_output);
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, /*active_low=*/active_low_reset, false);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      BuildBlockInPackage(/*stage_count=*/stage_count, options));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock(kBlockName));

  XLS_VLOG(2) << "Nonblocking receives pipelined block";
  XLS_VLOG_LINES(2, block->DumpIr());

  // The input stimulus to this test are
  //  1. 10 cycles of reset
  //  2a. Randomly varying in0_vld, in1_vld
  //  2b. Constant out_rdy
  //  2c. Constant in2 data
  //  3. in_vld = 0 and out_rdy = 1 for 10 cycles to drain the pipeline
  int64_t simulation_cycle_count = 10000;
  int64_t max_random_cycle = simulation_cycle_count - 10 - 1;
  uint64_t in2_val = 77;

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  XLS_ASSERT_OK(SetSignalsOverCycles(0, 9, {{"rst", 1}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(10, simulation_cycle_count - 1,
                                     {{"rst", 0}}, inputs));

  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in0", 1, inputs));
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, simulation_cycle_count - 1,
                                                "in1", 1, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(0, simulation_cycle_count - 1,
                                     {{"in2", in2_val}}, inputs));

  std::minstd_rand rng_engine;
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in0_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetRandomSignalOverCycles(0, max_random_cycle, "in1_vld", 0, 1,
                                          rng_engine, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(max_random_cycle + 1,
                                     simulation_cycle_count - 1,
                                     {{"in0_vld", 0}, {"in1_vld", 0}}, inputs));
  XLS_ASSERT_OK(SetSignalsOverCycles(0, simulation_cycle_count - 1,
                                     {{"out_rdy", 1}}, inputs));

  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, InterpretSequentialBlock(block, inputs));

  // Add a cycle count for easier comparison with simulation results.
  XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1, "cycle",
                                                0, outputs));

  XLS_VLOG(1) << "Signal Trace";
  XLS_ASSERT_OK(VLogTestPipelinedIO(
      std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                              {"rst", SignalType::kInput},
                              {"in0", SignalType::kInput},
                              {"in0_vld", SignalType::kInput},
                              {"in0_rdy", SignalType::kOutput},
                              {"in1", SignalType::kInput},
                              {"in1_vld", SignalType::kInput},
                              {"in1_rdy", SignalType::kOutput},
                              {"in2", SignalType::kInput},
                              {"out", SignalType::kOutput},
                              {"out_vld", SignalType::kOutput},
                              {"out_rdy", SignalType::kInput}},
      /*column_width=*/10, inputs, outputs));

  // Check the following property
  // 1. The sequence of inputs where (in_vld && in_rdy && !rst) is true
  //    is strictly monotone increasing with no duplicates.
  // 2. The sequence of outputs where out_vld && out_rdy is true
  //    is strictly monotone increasing with no duplicates.
  // 3. Both sequences in #1 and #2 are identical.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input0_sequence,
      GetChannelSequenceFromIO({"in0", SignalType::kInput},
                               {"in0_vld", SignalType::kInput},
                               {"in0_rdy", SignalType::kOutput},
                               {"rst", SignalType::kInput}, inputs, outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> input1_sequence,
      GetChannelSequenceFromIO({"in1", SignalType::kInput},
                               {"in1_vld", SignalType::kInput},
                               {"in1_rdy", SignalType::kOutput},
                               {"rst", SignalType::kInput}, inputs, outputs));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CycleAndValue> output_sequence,
      GetChannelSequenceFromIO({"out", SignalType::kOutput},
                               {"out_vld", SignalType::kOutput},
                               {"out_rdy", SignalType::kInput},
                               {"rst", SignalType::kInput}, inputs, outputs));

  // Because things are non-blocking there should be an output value every cycle
  // except for the reset time and pipeline fill time.
  int64_t pipe_latency = stage_count - 1;

  // If we are flopping the input (with either flop or skid buffer), then the
  // input has a once cycle delay before arriving at the pipeline.  This results
  // in an extra output. So flopping input does not increase the time it takes
  // for a valid output to appear.
  if (flop_outputs) {
    ++pipe_latency;
  }
  EXPECT_EQ(output_sequence.size(), simulation_cycle_count - pipe_latency - 10);

  int64_t input0_index = 0;
  int64_t input1_index = 0;
  int64_t output_index = 0;

  if (flop_inputs) {
    EXPECT_EQ(output_sequence.at(0).value, in2_val);
    ++output_index;
  }

  for (int64_t cycle = 10; cycle < simulation_cycle_count; ++cycle) {
    uint64_t in0_val = 0;
    uint64_t in1_val = 0;

    if (input0_index < input0_sequence.size() &&
        input0_sequence[input0_index].cycle == cycle) {
      in0_val = input0_sequence[input0_index].value;
      ++input0_index;
    }

    if (input1_index < input1_sequence.size() &&
        input1_sequence[input1_index].cycle == cycle) {
      in1_val = input1_sequence[input1_index].value;
      ++input1_index;
    }

    uint64_t expected_out_val = in0_val + in1_val + in2_val;
    if (output_index < output_sequence.size()) {
      uint64_t out_val = output_sequence[output_index].value;

      EXPECT_EQ(expected_out_val, out_val) << absl::StreamFormat(
          "Output cycle %d from input cycle %d expected %d+%d+%d=%d, got %d",
          output_sequence[output_index].cycle, cycle, in0_val, in1_val, in2_val,
          out_val, expected_out_val);
      ++output_index;
    }
  }
}

class ProcWithStateTest : public BlockConversionTest {
 public:
  void TestBlockWithSchedule(const xls::SchedulingOptions& scheduling_options) {
    const std::string ir_text = R"(package my_package

  chan a_in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
  chan a_out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
  chan b_in(bits[32], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
  chan b_out(bits[32], id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
  chan c_in(bits[32], id=4, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
  chan c_out(bits[32], id=5, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

  top proc test_proc(tkn: token, st_0: bits[32], st_1: bits[32], st_2: bits[32], init={3, 5, 9}) {
    receive.107: (token, bits[32]) = receive(tkn, channel_id=0, id=107)
    receive.108: (token, bits[32]) = receive(tkn, channel_id=2, id=108)
    receive.109: (token, bits[32]) = receive(tkn, channel_id=4, id=109)
    tuple_index.34: token = tuple_index(receive.107, index=0, id=34)
    send.110: token = send(tkn, st_0, channel_id=1, id=110)
    tuple_index.43: token = tuple_index(receive.108, index=0, id=43)
    send.111: token = send(tkn, st_1, channel_id=3, id=111)
    tuple_index.52: token = tuple_index(receive.109, index=0, id=52)
    send.112: token = send(tkn, st_2, channel_id=5, id=112)
    tuple_index.35: bits[32] = tuple_index(receive.107, index=1, id=35)
    tuple_index.44: bits[32] = tuple_index(receive.108, index=1, id=44)
    tuple_index.53: bits[32] = tuple_index(receive.109, index=1, id=53)
    after_all.59: token = after_all(tuple_index.34, send.110, tuple_index.43, send.111, tuple_index.52, send.112, id=59)
    add.100: bits[32] = add(st_0, tuple_index.35, id=100)
    add.101: bits[32] = add(st_1, tuple_index.44, id=101)
    add.102: bits[32] = add(st_2, tuple_index.53, id=102)
    next (after_all.59, add.100, add.101, add.102)
  }
  )";
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(ir_text));

    XLS_ASSERT_OK_AND_ASSIGN(xls::Proc * proc, package->GetProc("test_proc"));

    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(proc, TestDelayEstimator(), scheduling_options));

    CodegenOptions options;
    options.module_name(TestName());
    options.flop_inputs(true).flop_outputs(true).clock_name("clk");
    options.valid_control("input_valid", "output_valid");
    options.reset("rst", false, false, false);

    XLS_ASSERT_OK_AND_ASSIGN(xls::Block * block,
                             ProcToPipelinedBlock(schedule, options, proc));

    const double io_probability = 0.5;
    const uint64_t run_cycles = 128;

    std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
        {{"rst", 1}}};
    for (uint64_t c = 0; c < run_cycles; ++c) {
      inputs.push_back({{"rst", 0}});
    }

    std::vector<ChannelSource> sources{
        ChannelSource("a_in", "a_in_vld", "a_in_rdy", io_probability, block),
        ChannelSource("b_in", "b_in_vld", "b_in_rdy", io_probability, block),
        ChannelSource("c_in", "c_in_vld", "c_in_rdy", io_probability, block),
    };

    XLS_ASSERT_OK(sources.at(0).SetDataSequence(
        {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}));
    XLS_ASSERT_OK(sources.at(1).SetDataSequence(
        {11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121}));
    XLS_ASSERT_OK(sources.at(2).SetDataSequence(
        {50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325}));

    std::vector<ChannelSink> sinks{
        ChannelSink("a_out", "a_out_vld", "a_out_rdy", io_probability, block),
        ChannelSink("b_out", "b_out_vld", "b_out_rdy", io_probability, block),
        ChannelSink("c_out", "c_out_vld", "c_out_rdy", io_probability, block),
    };

    XLS_ASSERT_OK_AND_ASSIGN(
        BlockIOResultsAsUint64 results,
        InterpretChannelizedSequentialBlockWithUint64(
            block, absl::MakeSpan(sources), absl::MakeSpan(sinks), inputs,
            options.reset()));
    EXPECT_THAT(
        sinks.at(0).GetOutputSequenceAsUint64(),
        IsOkAndHolds(IsPrefixOf<uint64_t>(std::vector<uint64_t>{
            3, 13, 33, 63, 103, 153, 213, 283, 363, 453, 553, 663, 783})));
    EXPECT_THAT(
        sinks.at(1).GetOutputSequenceAsUint64(),
        IsOkAndHolds(IsPrefixOf<uint64_t>(std::vector<uint64_t>{
            5, 16, 37, 68, 109, 160, 221, 292, 373, 464, 565, 676, 797})));
    EXPECT_THAT(sinks.at(2).GetOutputSequenceAsUint64(),
                IsOkAndHolds(IsPrefixOf<uint64_t>(
                    std::vector<uint64_t>{9, 59, 134, 234, 359, 509, 684, 884,
                                          1109, 1359, 1634, 1934, 2259})));
  }
};

TEST_F(ProcWithStateTest, ProcWithStateSingleCycle) {
  xls::SchedulingOptions scheduling_options;
  scheduling_options.pipeline_stages(1);

  TestBlockWithSchedule(scheduling_options);
}

TEST_F(ProcWithStateTest, ProcWithStateBackedgesIn2Stages) {
  xls::SchedulingOptions scheduling_options;
  scheduling_options.pipeline_stages(2);

  scheduling_options.add_constraint(xls::IOConstraint(
      "a_in", xls::IODirection::kReceive, "b_in", xls::IODirection::kReceive,
      /*min_latency=*/1, /* max_latency=*/1));

  TestBlockWithSchedule(scheduling_options);
}

TEST_F(ProcWithStateTest, ProcWithStateBackedgesIn3Stages) {
  xls::SchedulingOptions scheduling_options;
  scheduling_options.pipeline_stages(3);

  scheduling_options.add_constraint(xls::IOConstraint(
      "a_in", xls::IODirection::kReceive, "b_in", xls::IODirection::kReceive,
      /*min_latency=*/1, /* max_latency=*/1));
  scheduling_options.add_constraint(xls::IOConstraint(
      "b_in", xls::IODirection::kReceive, "c_in", xls::IODirection::kReceive,
      /*min_latency=*/1, /* max_latency=*/1));

  TestBlockWithSchedule(scheduling_options);
}

TEST_F(ProcWithStateTest, ProcWithStateBackedgesIn3StagesWithExtra) {
  xls::SchedulingOptions scheduling_options;
  scheduling_options.pipeline_stages(4);

  scheduling_options.add_constraint(xls::IOConstraint(
      "a_in", xls::IODirection::kReceive, "b_in", xls::IODirection::kReceive,
      /*min_latency=*/1, /* max_latency=*/1));
  scheduling_options.add_constraint(xls::IOConstraint(
      "b_in", xls::IODirection::kReceive, "c_in", xls::IODirection::kReceive,
      /*min_latency=*/1, /* max_latency=*/1));

  TestBlockWithSchedule(scheduling_options);
}

INSTANTIATE_TEST_SUITE_P(
    NonblockingReceivesProcTestSweep, NonblockingReceivesProcTestSweepFixture,
    testing::Combine(testing::Values(1, 2, 3), testing::Values(false, true),
                     testing::Values(false, true),
                     testing::Values(CodegenOptions::IOKind::kFlop,
                                     CodegenOptions::IOKind::kSkidBuffer),
                     testing::Values(CodegenOptions::IOKind::kFlop,
                                     CodegenOptions::IOKind::kSkidBuffer)),
    NonblockingReceivesProcTestSweepFixture::PrintToStringParamName);

TEST_F(ProcConversionTestFixture, RecvDataFeedingSendPredicate) {
  Package package(TestName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32, {},
                                     std::nullopt, FlowControl::kReadyValid));

  TokenlessProcBuilder pb(TestName(), "tkn", &package);
  BValue recv = pb.Receive(in);

  BValue two_five = pb.Literal(UBits(25, 32));
  BValue one_five = pb.Literal(UBits(15, 32));

  BValue lt_two_five = pb.ULt(recv, two_five);
  BValue gt_one_five = pb.UGt(recv, one_five);

  pb.SendIf(out0, lt_two_five, recv);
  pb.SendIf(out1, gt_one_five, recv);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  for (int32_t i = 0; i < 100; ++i) {
    int32_t seed = 100000 + 5000 * i;

    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
            proc, TestDelayEstimator(),
            SchedulingOptions().pipeline_stages(1).seed(seed)));
    CodegenOptions options;
    options.flop_inputs(false).flop_outputs(false).clock_name("clk");
    options.valid_control("input_valid", "output_valid");
    options.reset("rst", false, false, true);
    options.streaming_channel_data_suffix("_data");
    options.streaming_channel_valid_suffix("_valid");
    options.streaming_channel_ready_suffix("_ready");
    options.module_name("pipelined_proc");

    XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                             ProcToPipelinedBlock(schedule, options, proc));
    XLS_VLOG_LINES(2, block->DumpIr());

    int64_t input_count = 80;
    int64_t simulation_cycle_count = 500;

    std::vector<absl::flat_hash_map<std::string, uint64_t>>
        non_streaming_inputs(simulation_cycle_count, {{"rst", 0}});

    std::vector<uint64_t> in_values(input_count);
    std::vector<uint64_t> sequence_values(input_count / 2);

    std::iota(sequence_values.begin(), sequence_values.end(), 0);
    std::copy(sequence_values.begin(), sequence_values.end(),
              in_values.begin());
    std::reverse_copy(sequence_values.begin(), sequence_values.end(),
                      in_values.begin() + sequence_values.size());

    std::vector<ChannelSource> sources{
        ChannelSource("in_data", "in_valid", "in_ready", 0.25, block),
    };
    XLS_ASSERT_OK(sources.at(0).SetDataSequence(in_values));

    std::vector<ChannelSink> sinks{
        ChannelSink("out0_data", "out0_valid", "out0_ready", 1, block),
        ChannelSink("out1_data", "out1_valid", "out1_ready", 0.5, block),
    };

    XLS_ASSERT_OK_AND_ASSIGN(
        BlockIOResultsAsUint64 io_results,
        InterpretChannelizedSequentialBlockWithUint64(
            block, absl::MakeSpan(sources), absl::MakeSpan(sinks),
            non_streaming_inputs, std::nullopt, seed));

    std::vector<absl::flat_hash_map<std::string, uint64_t>>& inputs =
        io_results.inputs;
    std::vector<absl::flat_hash_map<std::string, uint64_t>>& outputs =
        io_results.outputs;

    // Add a cycle count for easier comparison with simulation results.
    XLS_ASSERT_OK(SetIncrementingSignalOverCycles(0, outputs.size() - 1,
                                                  "cycle", 0, outputs));

    XLS_ASSERT_OK(VLogTestPipelinedIO(
        std::vector<SignalSpec>{
            {"cycle", SignalType::kOutput},
            {"rst", SignalType::kInput, /*active_low_reset=*/false},
            {"in_data", SignalType::kInput},
            {"in_valid", SignalType::kInput},
            {"in_ready", SignalType::kOutput},
            {"out0_data", SignalType::kOutput},
            {"out0_valid", SignalType::kOutput},
            {"out0_ready", SignalType::kInput},
            {"out1_data", SignalType::kOutput},
            {"out1_valid", SignalType::kOutput},
            {"out1_ready", SignalType::kInput}},
        /*column_width=*/10, inputs, outputs));

    EXPECT_THAT(
        sinks.at(0).GetOutputSequenceAsUint64(),
        IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 24,
                                 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                                 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)));
    EXPECT_THAT(
        sinks.at(1).GetOutputSequenceAsUint64(),
        IsOkAndHolds(ElementsAre(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30,
            29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16)));
  }
}

}  // namespace
}  // namespace verilog
}  // namespace xls
