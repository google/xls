// Copyright 2020 The XLS Authors
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

#include "xls/simulation/module_simulator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/testbench_signal_capture.h"
#include "xls/tools/eval_utils.h"

namespace xls {
namespace verilog {
namespace {

// Converts a map of named Values to a map of named Bits via flattening.
ModuleSimulator::BitsMap ValueMapToBitsMap(
    const absl::flat_hash_map<std::string, Value>& inputs) {
  ModuleSimulator::BitsMap outputs;
  for (const auto& pair : inputs) {
    outputs[pair.first] = FlattenValueToBits(pair.second);
  }
  return outputs;
}

// Converts a list of IR Value to a list of IR Bits.
std::vector<Bits> ValueListToBitsList(absl::Span<const Value> values) {
  std::vector<Bits> bits_list;
  for (const Value& value : values) {
    bits_list.push_back(FlattenValueToBits(value));
  }
  return bits_list;
}

// Converts a list of IR Bits to a list of IR Values.
absl::StatusOr<std::vector<Value>> BitsListToValueList(
    absl::Span<const Bits> bits_list, const TypeProto& type) {
  std::vector<Value> values_list;
  for (const Bits& bits : bits_list) {
    XLS_ASSIGN_OR_RETURN(Value value, UnflattenBitsToValue(bits, type));
    values_list.push_back(value);
  }
  return values_list;
}

absl::Status VerifyReadyValidHoldoffs(
    const ReadyValidHoldoffs& holdoffs,
    const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
    const ModuleSignature& signature) {
  for (const auto& [channel_name, valid_holdoffs] : holdoffs.valid_holdoffs) {
    XLS_RET_CHECK(channel_inputs.contains(channel_name)) << absl::StreamFormat(
        "Valid holdoff specified for channel `%s` which is not an input",
        channel_name);
    XLS_RET_CHECK_EQ(channel_inputs.at(channel_name).size(),
                     valid_holdoffs.size())
        << absl::StreamFormat(
               "Number of valid holdoffs does not match number of inputs for "
               "channel `%s` which has no inputs",
               channel_name);
    for (const ValidHoldoff& valid_holdoff : valid_holdoffs) {
      XLS_RET_CHECK_GE(valid_holdoff.cycles, 0);
      if (!valid_holdoff.driven_values.empty()) {
        XLS_RET_CHECK_EQ(valid_holdoff.cycles,
                         valid_holdoff.driven_values.size())
            << absl::StreamFormat(
                   "Mismatch between holdoff length and number of driven "
                   "values for  channel `%s` which has no inputs",
                   channel_name);
      }
    }
  }

  for (const auto& [channel_name, _] : holdoffs.ready_holdoffs) {
    XLS_RET_CHECK(output_channel_counts.contains(channel_name))
        << absl::StreamFormat(
               "Ready holdoff specified for channel `%s` is not an output",
               channel_name);
  }
  return absl::OkStatus();
}

// Holds-off the valid signal of the indicated channel based on the hold-off
// behavior specified in `holdoffs`.
absl::Status HoldoffValid(int64_t input_number, int64_t input_bit_count,
                          const ValidHoldoff& valid_holdoff,
                          std::string_view channel_name,
                          const BlockPortMappingProto& port_mapping_proto,
                          SequentialBlock& seq_block) {
  if (valid_holdoff.cycles == 0) {
    // Nothing to do as the valid holdoff is zero cycles.
  } else if (valid_holdoff.driven_values.empty()) {
    // Values to drive on the data port not specified. Drive X and wait
    // the specified number of cycles.
    seq_block.Set(port_mapping_proto.valid_port_name(), UBits(0, 1));
    seq_block.SetX(port_mapping_proto.data_port_name());
    seq_block.AdvanceNCycles(valid_holdoff.cycles);
  } else {
    XLS_RET_CHECK_EQ(valid_holdoff.driven_values.size(), valid_holdoff.cycles)
        << absl::StreamFormat(
               "Unexpected number of driven values for channel `%s`",
               channel_name);
    seq_block.Set(port_mapping_proto.valid_port_name(), UBits(0, 1));
    for (const BitsOrX& bits_or_x : valid_holdoff.driven_values) {
      if (std::holds_alternative<IsX>(bits_or_x)) {
        seq_block.SetX(port_mapping_proto.data_port_name());
      } else {
        seq_block.Set(port_mapping_proto.data_port_name(),
                      std::get<Bits>(bits_or_x));
      }
      seq_block.NextCycle();
    }
  }
  return absl::OkStatus();
}

// Sets up the test bench to drive the specified inputs on to the given channel.
absl::Status DriveInputChannel(absl::Span<const Bits> inputs,
                               std::string_view channel_name,
                               const BlockPortMappingProto& port_mapping_proto,
                               absl::Span<const ValidHoldoff> valid_holdoffs,
                               ModuleTestbench& tb) {
  std::vector<DutInput> dut_inputs;
  std::string_view data_port_name = port_mapping_proto.data_port_name();
  dut_inputs.push_back(DutInput{.port_name = std::string{data_port_name},
                                .initial_value = IsX()});
  std::optional<std::string> valid_port_name;
  std::optional<std::string> ready_port_name;
  if (port_mapping_proto.has_valid_port_name()) {
    valid_port_name = port_mapping_proto.valid_port_name();
    dut_inputs.push_back(DutInput{.port_name = valid_port_name.value(),
                                  .initial_value = UBits(0, 1)});
  }
  if (port_mapping_proto.has_ready_port_name()) {
    ready_port_name = port_mapping_proto.ready_port_name();
  }
  XLS_ASSIGN_OR_RETURN(
      ModuleTestbenchThread * tbt,
      tb.CreateThread(absl::StrFormat("%s driver", channel_name), dut_inputs,
                      /*wait_until_done=*/false));
  SequentialBlock& seq_block = tbt->MainBlock();
  for (int64_t input_number = 0; input_number < inputs.size(); ++input_number) {
    const Bits& value = inputs[input_number];
    if (!valid_holdoffs.empty()) {
      XLS_RET_CHECK(valid_port_name.has_value()) << absl::StreamFormat(
          "Valid hold-off specified for channel without a valid signal: "
          "`%s`",
          channel_name);
      XLS_RETURN_IF_ERROR(HoldoffValid(
          input_number, value.bit_count(), valid_holdoffs[input_number],
          channel_name, port_mapping_proto, seq_block));
    }
    seq_block.Set(data_port_name, value);
    if (valid_port_name.has_value()) {
      seq_block.Set(valid_port_name.value(), UBits(1, 1));
    }
    if (ready_port_name.has_value()) {
      seq_block.WaitForCycleAfter(ready_port_name.value());
    }
  }
  // After driving all inputs, set valid to 0 and data port to X.
  if (valid_port_name.has_value()) {
    seq_block.Set(valid_port_name.value(), UBits(0, 1));
  }
  seq_block.SetX(data_port_name);
  return absl::OkStatus();
}

// Sets up the test bench to captures `output_count` outputs on the given output
// channel. Returns the sequence of placeholder Bits objects which will hold the
// values after the test bench is run.
absl::StatusOr<std::vector<std::unique_ptr<Bits>>> CaptureOutputChannel(
    std::string_view channel_name,
    const BlockPortMappingProto& port_mapping_proto, int64_t output_count,
    absl::Span<const int64_t> ready_holdoffs, ModuleTestbench& tb) {
  if (port_mapping_proto.has_ready_port_name()) {
    std::string_view ready_port_name = port_mapping_proto.ready_port_name();
    // Create a separate thread to drive ready and any necessary holdoffs
    std::vector<DutInput> dut_inputs = {
        DutInput{.port_name = std::string{ready_port_name},
                 .initial_value = UBits(0, 1)}};
    XLS_ASSIGN_OR_RETURN(
        ModuleTestbenchThread * ready_tbt,
        tb.CreateThread(absl::StrFormat("%s driver", channel_name), dut_inputs,
                        /*wait_until_done=*/false));
    // Collapse consecutive 0's in the ready holdoff sequence into a single
    // multi-cycle assertion of ready. E.g., the sequence {7, 0, 0, 42, 0}
    // should lower to:
    //
    //   ready = 0;
    //   <wait 7 cycles>
    //   ready = 1;
    //   <wait 3 cycles>
    //   ready = 0;
    //   <wait 42 cycles>
    //   ready = 1;
    int64_t assertion_length = 1;
    for (int64_t holdoff : ready_holdoffs) {
      if (holdoff == 0) {
        ++assertion_length;
      } else {
        if (assertion_length > 0) {
          ready_tbt->MainBlock().Set(ready_port_name, UBits(1, 1));
          ready_tbt->MainBlock().AdvanceNCycles(assertion_length);
        }
        ready_tbt->MainBlock().Set(ready_port_name, UBits(0, 1));
        ready_tbt->MainBlock().AdvanceNCycles(holdoff);
        assertion_length = 1;
      }
    }
    ready_tbt->MainBlock().Set(ready_port_name, UBits(1, 1));
  }

  // Create thread for capturing outputs. This thread drives no signals.
  XLS_ASSIGN_OR_RETURN(
      ModuleTestbenchThread * tbt,
      tb.CreateThread(absl ::StrFormat("output %s capture", channel_name),
                      /*dut_inputs=*/{}));
  std::vector<std::string> flow_control_signals;
  if (port_mapping_proto.has_valid_port_name()) {
    flow_control_signals.push_back(port_mapping_proto.valid_port_name());
  }
  if (port_mapping_proto.has_ready_port_name()) {
    flow_control_signals.push_back(port_mapping_proto.ready_port_name());
  }
  std::vector<std::unique_ptr<Bits>> outputs;
  for (int64_t read_count = 0; read_count < output_count; ++read_count) {
    outputs.push_back(std::make_unique<Bits>());
    if (flow_control_signals.empty()) {
      tbt->MainBlock().AtEndOfCycle().Capture(
          port_mapping_proto.data_port_name(), outputs.back().get());
    } else {
      tbt->MainBlock()
          .AtEndOfCycleWhenAll(flow_control_signals)
          .Capture(port_mapping_proto.data_port_name(), outputs.back().get());
    }
  }
  return outputs;
}

absl::StatusOr<const BlockPortMappingProto*> BlockPortMappingForModule(
    const ChannelProto& channel_proto, std::string_view module_name) {
  auto iter = absl::c_find_if(
      channel_proto.metadata().block_ports(),
      [&](const BlockPortMappingProto& block_port_mapping_proto) {
        return block_port_mapping_proto.block_name() == module_name;
      });
  if (iter == channel_proto.metadata().block_ports().end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block port mapping for channel '%s' not found for module '%s'.",
        channel_proto.name(), module_name));
  }
  return &*iter;
}

}  // namespace

std::vector<DutInput> ModuleSimulator::DeassertControlSignals() const {
  std::vector<DutInput> dut_inputs;
  if (signature_.proto().has_pipeline() &&
      signature_.proto().pipeline().has_pipeline_control()) {
    const PipelineControl& pipeline_control =
        signature_.proto().pipeline().pipeline_control();
    if (pipeline_control.has_valid()) {
      dut_inputs.push_back(
          DutInput{.port_name = pipeline_control.valid().input_name(),
                   .initial_value = UBits(0, 1)});
    }
  }
  return dut_inputs;
}

absl::StatusOr<ModuleSimulator::BitsMap> ModuleSimulator::RunFunction(
    const ModuleSimulator::BitsMap& inputs) const {
  XLS_ASSIGN_OR_RETURN(auto outputs, RunBatched({inputs}));
  XLS_RET_CHECK_EQ(outputs.size(), 1);
  return outputs[0];
}

absl::StatusOr<Bits> ModuleSimulator::RunAndReturnSingleOutput(
    const BitsMap& inputs) const {
  BitsMap outputs;
  XLS_ASSIGN_OR_RETURN(outputs, RunFunction(inputs));
  if (outputs.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected exactly one data output, got %d", outputs.size()));
  }
  return outputs.begin()->second;
}

absl::StatusOr<std::vector<ModuleSimulator::BitsMap>>
ModuleSimulator::RunBatched(absl::Span<const BitsMap> inputs) const {
  VLOG(1) << "Running Verilog module with signature:\n"
          << signature_.ToString();
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Arguments:\n";
    for (int64_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      VLOG(1) << "  Set " << i << ":";
      for (const auto& pair : input) {
        VLOG(1) << "    " << pair.first << " : " << pair.second.ToDebugString();
      }
    }
  }
  VLOG(2) << "Verilog:\n" << verilog_text_;

  if (inputs.empty()) {
    return std::vector<BitsMap>();
  }

  for (auto& input : inputs) {
    XLS_RETURN_IF_ERROR(signature_.ValidateInputs(input));
  }

  if (!signature_.proto().has_clock_name() &&
      !signature_.proto().has_combinational()) {
    return absl::InvalidArgumentError("Expected clock in signature");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ModuleTestbench> tb,
                       ModuleTestbench::CreateFromVerilogText(
                           verilog_text_, file_type_, signature_, simulator_,
                           /*reset_dut=*/true, includes_));

  // Drive any control signals to an unasserted state so the all control inputs
  // are non-X when the device comes out of reset.
  std::vector<DutInput> dut_inputs = DeassertControlSignals();
  for (const auto& [name, _] : inputs.front()) {
    dut_inputs.push_back(DutInput{name, IsX()});
  }

  XLS_ASSIGN_OR_RETURN(ModuleTestbenchThread * tbt,
                       tb->CreateThread("input driver", dut_inputs));
  SequentialBlock& seq_block = tbt->MainBlock();

  // Drive data inputs. Values are flattened before using.
  auto drive_data = [&](int64_t index) {
    for (const PortProto& input : signature_.data_inputs()) {
      seq_block.Set(input.name(), inputs[index].at(input.name()));
    }
  };

  // Lambda which captures outputs into a map. Use std::unique_ptr for pointer
  // stability necessary for ModuleTestbench::Capture().
  using OutputMap = absl::flat_hash_map<std::string, std::unique_ptr<Bits>>;
  std::vector<OutputMap> stable_outputs(inputs.size());
  auto capture_outputs = [&](int64_t index, EndOfCycleEvent& event) {
    OutputMap& outputs = stable_outputs[index];
    for (const PortProto& output : signature_.data_outputs()) {
      outputs[output.name()] = std::make_unique<Bits>();
      event.Capture(output.name(), outputs.at(output.name()).get());
    }
  };

  if (signature_.proto().has_fixed_latency()) {
    for (int64_t i = 0; i < inputs.size(); ++i) {
      drive_data(i);
      // Fixed latency interface: just wait for compute to complete.
      seq_block.AdvanceNCycles(signature_.proto().fixed_latency().latency());
      EndOfCycleEvent& event = seq_block.AtEndOfCycle();
      capture_outputs(i, event);

      // The input data cannot be changed in the same cycle that the output is
      // being read so hold for one more cycle while output is read.
      seq_block.NextCycle();
    }
  } else if (signature_.proto().has_pipeline()) {
    const int64_t latency = signature_.proto().pipeline().latency();
    int64_t cycle = 0;
    int64_t captured_outputs = 0;
    std::optional<PipelineControl> pipeline_control;
    if (signature_.proto().pipeline().has_pipeline_control()) {
      pipeline_control = signature_.proto().pipeline().pipeline_control();
    }

    if (pipeline_control.has_value() && pipeline_control->has_manual()) {
      // Drive the pipeline register load-enable signals high.
      seq_block.Set(pipeline_control->manual().input_name(),
                    Bits::AllOnes(latency));
    }

    // Expect the output_valid signal (if it exists) to be the given value or X.
    auto maybe_expect_output_valid = [&](bool expect_x, bool expected_value,
                                         EndOfCycleEvent& event) {
      if (pipeline_control.has_value() && pipeline_control->has_valid() &&
          pipeline_control->valid().has_output_name()) {
        if (expect_x) {
          event.ExpectX(pipeline_control->valid().output_name());
        } else {
          event.ExpectEq(pipeline_control->valid().output_name(),
                         static_cast<int64_t>(expected_value));
        }
      }
    };
    while (cycle < inputs.size()) {
      drive_data(cycle);
      if (pipeline_control.has_value() && pipeline_control->has_valid()) {
        seq_block.Set(pipeline_control->valid().input_name(), 1);
      }
      // Pipelined interface: drive inputs for a cycle, then wait for compute to
      // complete.  A pipelined interface should not require that the inputs be
      // held for more than a cycle.
      EndOfCycleEvent& event = seq_block.AtEndOfCycle();

      if (cycle >= latency) {
        maybe_expect_output_valid(/*expect_x=*/false, /*expected_value=*/true,
                                  event);
        capture_outputs(captured_outputs, event);
        captured_outputs++;
      } else {
        // The initial inputs have not yet reached the end of the pipeline. The
        // output_valid signal (if it exists) should still be X if there is no
        // reset signal.
        maybe_expect_output_valid(/*expect_x=*/!signature_.proto().has_reset(),
                                  /*expected_value=*/false, event);
      }
      cycle++;
    }
    for (const PortProto& input : signature_.data_inputs()) {
      seq_block.SetX(input.name());
    }
    if (pipeline_control.has_value() && pipeline_control->has_valid()) {
      seq_block.Set(pipeline_control->valid().input_name(), 0);
    }
    if (cycle < latency) {
      seq_block.AdvanceNCycles(latency - cycle);
    }
    while (captured_outputs < inputs.size()) {
      EndOfCycleEvent& event = seq_block.AtEndOfCycle();
      maybe_expect_output_valid(/*expect_x=*/false, /*expected_value=*/true,
                                event);
      capture_outputs(captured_outputs, event);
      captured_outputs++;
    }
    // valid == 0 should have propagated all the way through the pipeline to
    // output_valid.
    EndOfCycleEvent& event = seq_block.AtEndOfCycle();
    maybe_expect_output_valid(/*expect_x=*/false, /*expected_value=*/false,
                              event);
  } else if (signature_.proto().has_combinational()) {
    for (int64_t i = 0; i < inputs.size(); ++i) {
      drive_data(i);
      capture_outputs(i, seq_block.AtEndOfCycle());
    }
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "Unsupported interface: ", signature_.proto().interface_oneof_case()));
  }

  XLS_RETURN_IF_ERROR(tb->Run());

  // Transfer outputs to an ArgumentSet for return.
  std::vector<BitsMap> outputs(inputs.size());
  for (int64_t i = 0; i < inputs.size(); ++i) {
    for (const auto& pair : stable_outputs[i]) {
      outputs[i][pair.first] = *pair.second;
    }
  }

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Results:\n";
    for (int64_t i = 0; i < outputs.size(); ++i) {
      VLOG(1) << "  Set " << i << ":";
      for (const auto& pair : outputs[i]) {
        VLOG(1) << "    " << pair.first << " : " << pair.second.ToDebugString();
      }
    }
  }

  return outputs;
}

absl::StatusOr<Value> ModuleSimulator::RunFunction(
    const absl::flat_hash_map<std::string, Value>& inputs) const {
  absl::flat_hash_map<std::string, Value> input_map(inputs.begin(),
                                                    inputs.end());
  XLS_ASSIGN_OR_RETURN(auto outputs, RunBatched({input_map}));
  XLS_RET_CHECK_EQ(outputs.size(), 1);
  return outputs[0];
}

absl::StatusOr<std::vector<Value>> ModuleSimulator::RunBatched(
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs) const {
  std::vector<BitsMap> bits_inputs;
  for (const auto& input : inputs) {
    XLS_RETURN_IF_ERROR(signature_.ValidateInputs(input));
    bits_inputs.push_back(ValueMapToBitsMap(input));
  }
  XLS_ASSIGN_OR_RETURN(std::vector<BitsMap> bits_outputs,
                       RunBatched(bits_inputs));
  CHECK_EQ(signature_.data_outputs().size(), 1);
  std::vector<Value> outputs;
  for (const BitsMap& bits_output : bits_outputs) {
    XLS_RET_CHECK_EQ(bits_output.size(), 1);
    XLS_ASSIGN_OR_RETURN(
        Value output,
        UnflattenBitsToValue(bits_output.begin()->second,
                             signature_.data_outputs().begin()->type()));
    outputs.push_back(std::move(output));
  }
  return outputs;
}

absl::StatusOr<std::string> ModuleSimulator::GenerateProcTestbenchVerilog(
    const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
    std::optional<ReadyValidHoldoffs> holdoffs) const {
  XLS_ASSIGN_OR_RETURN(
      ProcTestbench proc_tb,
      CreateProcTestbench(channel_inputs, output_channel_counts,
                          std::move(holdoffs)));
  return proc_tb.testbench->GenerateVerilog();
}

absl::StatusOr<ModuleSimulator::ProcTestbench>
ModuleSimulator::CreateProcTestbench(
    const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
    std::optional<ReadyValidHoldoffs> holdoffs) const {
  VLOG(1) << "Generating testbench for Verilog module with signature:\n"
          << signature_.ToString();
  if (VLOG_IS_ON(1)) {
    absl::flat_hash_map<std::string, std::vector<Value>> channel_inputs_values;
    for (const auto& [channel_name, channel_values] : channel_inputs) {
      XLS_ASSIGN_OR_RETURN(ChannelProto channel_proto,
                           signature_.GetInputChannelProtoByName(channel_name));
      XLS_ASSIGN_OR_RETURN(
          const BlockPortMappingProto* block_port_proto,
          BlockPortMappingForModule(channel_proto, signature_.module_name()));
      XLS_ASSIGN_OR_RETURN(PortProto data_port,
                           signature_.GetInputPortProtoByName(
                               block_port_proto->data_port_name()));
      XLS_ASSIGN_OR_RETURN(
          channel_inputs_values[channel_name],
          BitsListToValueList(channel_values, data_port.type()));
    }
    VLOG(1) << "Input channel values:\n";
    VLOG(1) << ChannelValuesToString(channel_inputs_values);
  }
  VLOG(2) << "Verilog:\n" << verilog_text_;

  for (const auto& [channel_name, channel_values] : channel_inputs) {
    XLS_RETURN_IF_ERROR(
        signature_.ValidateChannelBitsInputs(channel_name, channel_values));
  }

  // Ensure all output channels have an expected read count.
  for (const ChannelProto& channel_proto : signature_.GetOutputChannels()) {
    std::string_view channel_name = channel_proto.name();
    if (!output_channel_counts.contains(channel_name)) {
      return absl::NotFoundError(absl::StrFormat(
          "Channel '%s' not found in expected output channel counts map.",
          channel_name));
    }
    int64_t read_count = output_channel_counts.at(channel_name);
    if (read_count < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Output channel '%s' has a negative read count.", channel_name));
    }
  }

  // Verify ready/valid holdoffs if specified.
  if (holdoffs.has_value()) {
    XLS_RETURN_IF_ERROR(VerifyReadyValidHoldoffs(
        holdoffs.value(), channel_inputs, output_channel_counts, signature_));
  }

  if (!signature_.proto().has_clock_name() &&
      !signature_.proto().has_combinational()) {
    return absl::InvalidArgumentError("Expected clock in signature");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ModuleTestbench> tb,
                       ModuleTestbench::CreateFromVerilogText(
                           verilog_text_, file_type_, signature_, simulator_,
                           /*reset_dut=*/true, includes_));

  int64_t max_channel_reads = 0;
  for (const auto& [_, read_count] : output_channel_counts) {
    max_channel_reads = std::max(max_channel_reads, read_count);
  }

  // TODO(vmirian): 10-30-2022 Ensure semantics work for single value channel.
  for (const ChannelProto& channel_proto : signature_.GetInputChannels()) {
    std::string_view channel_name = channel_proto.name();
    absl::Span<const ValidHoldoff> valid_holdoffs;
    if (holdoffs.has_value() &&
        holdoffs->valid_holdoffs.contains(channel_name)) {
      valid_holdoffs = holdoffs->valid_holdoffs.at(channel_name);
    }
    XLS_ASSIGN_OR_RETURN(
        const BlockPortMappingProto* block_port_proto,
        BlockPortMappingForModule(channel_proto, signature_.module_name()));
    XLS_RETURN_IF_ERROR(DriveInputChannel(channel_inputs.at(channel_name),
                                          channel_name, *block_port_proto,
                                          valid_holdoffs, *tb));
  }

  // Use std::unique_ptr for pointer stability necessary for
  // ModuleTestbench::Capture().
  absl::flat_hash_map<std::string, std::vector<std::unique_ptr<Bits>>>
      stable_outputs;
  for (const ChannelProto& channel_proto : signature_.GetOutputChannels()) {
    std::string_view channel_name = channel_proto.name();
    absl::Span<const int64_t> ready_holdoffs;
    if (holdoffs.has_value() &&
        holdoffs->ready_holdoffs.contains(channel_name)) {
      ready_holdoffs = holdoffs->ready_holdoffs.at(channel_name);
    }
    XLS_ASSIGN_OR_RETURN(
        const BlockPortMappingProto* block_port_proto,
        BlockPortMappingForModule(channel_proto, signature_.module_name()));
    XLS_ASSIGN_OR_RETURN(
        stable_outputs[channel_name],
        CaptureOutputChannel(channel_name, *block_port_proto,
                             output_channel_counts.at(channel_proto.name()),
                             ready_holdoffs, *tb));
  }
  return ProcTestbench{
      .testbench = std::move(tb),
      .outputs = std::move(stable_outputs),
  };
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Bits>>>
ModuleSimulator::RunInputSeriesProc(
    const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
    std::optional<ReadyValidHoldoffs> holdoffs) const {
  XLS_ASSIGN_OR_RETURN(
      ProcTestbench proc_tb,
      CreateProcTestbench(channel_inputs, output_channel_counts,
                          std::move(holdoffs)));
  XLS_RETURN_IF_ERROR(proc_tb.testbench->Run());

  absl::flat_hash_map<std::string, std::vector<Bits>> outputs;
  for (const ChannelProto& channel_proto : signature_.GetOutputChannels()) {
    std::string_view channel_name = channel_proto.name();
    outputs[channel_name] = std::vector<Bits>();
    for (std::unique_ptr<Bits>& bits : proc_tb.outputs.at(channel_name)) {
      outputs[channel_name].push_back(std::move(*bits));
    }
  }

  if (VLOG_IS_ON(1)) {
    absl::flat_hash_map<std::string, std::vector<Value>> result_channel_values;
    for (const auto& [channel_name, channel_values] : outputs) {
      XLS_ASSIGN_OR_RETURN(
          ChannelProto channel_proto,
          signature_.GetOutputChannelProtoByName(channel_name));
      XLS_ASSIGN_OR_RETURN(
          const BlockPortMappingProto* block_port_proto,
          BlockPortMappingForModule(channel_proto, signature_.module_name()));
      XLS_ASSIGN_OR_RETURN(PortProto data_port,
                           signature_.GetOutputPortProtoByName(
                               block_port_proto->data_port_name()));
      XLS_ASSIGN_OR_RETURN(
          result_channel_values[channel_name],
          BitsListToValueList(channel_values, data_port.type()));
    }
    VLOG(1) << "Result channel values:\n";
    VLOG(1) << ChannelValuesToString(result_channel_values);
  }
  return outputs;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ModuleSimulator::RunInputSeriesProc(
    const absl::flat_hash_map<std::string, std::vector<Value>>& channel_inputs,
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
    std::optional<ReadyValidHoldoffs> holdoffs) const {
  absl::flat_hash_map<std::string, std::vector<Value>> channel_outputs;

  using MapT = absl::flat_hash_map<std::string, std::vector<Bits>>;
  MapT channel_inputs_bits;
  for (const auto& [channel_name, channel_values] : channel_inputs) {
    XLS_RETURN_IF_ERROR(
        signature_.ValidateChannelValueInputs(channel_name, channel_values));
    channel_inputs_bits[channel_name] = ValueListToBitsList(channel_values);
  }

  XLS_ASSIGN_OR_RETURN(
      MapT channel_outputs_bits,
      RunInputSeriesProc(channel_inputs_bits, output_channel_counts,
                         std::move(holdoffs)));

  for (const auto& [channel_name, channel_bits] : channel_outputs_bits) {
    XLS_ASSIGN_OR_RETURN(ChannelProto channel_proto,
                         signature_.GetOutputChannelProtoByName(channel_name));
    XLS_ASSIGN_OR_RETURN(
        const BlockPortMappingProto* block_port_proto,
        BlockPortMappingForModule(channel_proto, signature_.module_name()));
    XLS_ASSIGN_OR_RETURN(PortProto data_port,
                         signature_.GetOutputPortProtoByName(
                             block_port_proto->data_port_name()));
    XLS_ASSIGN_OR_RETURN(std::vector<Value> values,
                         BitsListToValueList(channel_bits, data_port.type()));
    channel_outputs[channel_proto.name()] = values;
  }
  return channel_outputs;
}

absl::StatusOr<Value> ModuleSimulator::RunFunction(
    absl::Span<const Value> inputs) const {
  absl::flat_hash_map<std::string, Value> kwargs;
  XLS_ASSIGN_OR_RETURN(kwargs, signature_.ToKwargs(inputs));
  return RunFunction(kwargs);
}

}  // namespace verilog
}  // namespace xls
