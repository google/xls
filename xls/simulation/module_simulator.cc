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

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/flattening.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/simulation/module_testbench.h"
#include "xls/tools/eval_helpers.h"

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

absl::flat_hash_map<std::string, std::optional<Bits>> InitValuesToX(
    absl::Span<const ModuleSimulator::BitsMap> inputs) {
  absl::flat_hash_map<std::string, std::optional<Bits>> init_values;
  for (const auto& [name, _] : inputs.front()) {
    init_values[name] = std::nullopt;
  }
  return init_values;
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
    absl::Span<const Bits> bits, const TypeProto& type) {
  std::vector<Value> values_list;
  for (const Bits& bits : bits) {
    XLS_ASSIGN_OR_RETURN(Value value, UnflattenBitsToValue(bits, type));
    values_list.push_back(value);
  }
  return values_list;
}

// Returns `true`, if the map is empty or all counts are zero. Otherwise,
// returns `false`.
bool IsEmptyOrAreAllCountsZero(
    const absl::flat_hash_map<std::string, int64_t>& channel_counts) {
  for (const auto& [_, count] : channel_counts) {
    if (count != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::flat_hash_map<std::string, Bits> ModuleSimulator::DeassertControlSignals()
    const {
  absl::flat_hash_map<std::string, Bits> control_signals;
  if (signature_.proto().has_pipeline() &&
      signature_.proto().pipeline().has_pipeline_control()) {
    const PipelineControl& pipeline_control =
        signature_.proto().pipeline().pipeline_control();
    if (pipeline_control.has_valid()) {
      control_signals[pipeline_control.valid().input_name()] = UBits(0, 1);
    }
  }
  return control_signals;
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
  XLS_VLOG(1) << "Running Verilog module with signature:\n"
              << signature_.ToString();
  if (XLS_VLOG_IS_ON(1)) {
    XLS_VLOG(1) << "Arguments:\n";
    for (int64_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      XLS_VLOG(1) << "  Set " << i << ":";
      for (const auto& pair : input) {
        XLS_VLOG(1) << "    " << pair.first << " : " << pair.second;
      }
    }
  }
  XLS_VLOG(2) << "Verilog:\n" << verilog_text_;

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

  ModuleTestbench tb(verilog_text_, file_type_, signature_, simulator_,
                     includes_);

  // Drive any control signals to an unasserted state so the all control inputs
  // are non-X when the device comes out of reset.
  absl::flat_hash_map<std::string, Bits> control_signals =
      DeassertControlSignals();

  absl::flat_hash_map<std::string, std::optional<Bits>>
      init_values_after_reset = InitValuesToX(inputs);

  for (const auto& [name, bit_value] : control_signals) {
    init_values_after_reset[name] = bit_value;
  }

  ModuleTestbenchThread& tbt = tb.CreateThread(init_values_after_reset);

  // Drive data inputs. Values are flattened before using.
  auto drive_data = [&](int64_t index) {
    for (const PortProto& input : signature_.data_inputs()) {
      tbt.Set(input.name(), inputs[index].at(input.name()));
    }
  };

  // Lambda which captures outputs into a map. Use std::unique_ptr for pointer
  // stability necessary for ModuleTestbench::Capture().
  using OutputMap = absl::flat_hash_map<std::string, std::unique_ptr<Bits>>;
  std::vector<OutputMap> stable_outputs(inputs.size());
  auto capture_outputs = [&](int64_t index) {
    OutputMap& outputs = stable_outputs[index];
    for (const PortProto& output : signature_.data_outputs()) {
      outputs[output.name()] = std::make_unique<Bits>();
      tbt.Capture(output.name(), outputs.at(output.name()).get());
    }
  };

  if (signature_.proto().has_fixed_latency()) {
    for (int64_t i = 0; i < inputs.size(); ++i) {
      drive_data(i);
      // Fixed latency interface: just wait for compute to complete.
      tbt.AdvanceNCycles(signature_.proto().fixed_latency().latency());
      capture_outputs(i);

      // The input data cannot be changed in the same cycle that the output is
      // being read so hold for one more cycle while output is read.
      tbt.NextCycle();
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
      tbt.Set(pipeline_control->manual().input_name(), Bits::AllOnes(latency));
    }

    // Expect the output_valid signal (if it exists) to be the given value or X.
    auto maybe_expect_output_valid = [&](bool expect_x, bool expected_value) {
      if (pipeline_control.has_value() && pipeline_control->has_valid() &&
          pipeline_control->valid().has_output_name()) {
        if (expect_x) {
          tbt.ExpectX(pipeline_control->valid().output_name());
        } else {
          tbt.ExpectEq(pipeline_control->valid().output_name(),
                       static_cast<int64_t>(expected_value));
        }
      }
    };
    while (cycle < inputs.size()) {
      drive_data(cycle);
      if (pipeline_control.has_value() && pipeline_control->has_valid()) {
        tbt.Set(pipeline_control->valid().input_name(), 1);
      }
      // Pipelined interface: drive inputs for a cycle, then wait for compute to
      // complete.  A pipelined interface should not require that the inputs be
      // held for more than a cycle.
      tbt.NextCycle();
      cycle++;

      if (cycle >= latency) {
        maybe_expect_output_valid(/*expect_x=*/false, /*expected_value=*/true);
        capture_outputs(captured_outputs);
        captured_outputs++;
      } else {
        // The initial inputs have not yet reached the end of the pipeline. The
        // output_valid signal (if it exists) should still be X if there is no
        // reset signal.
        maybe_expect_output_valid(/*expect_x=*/!signature_.proto().has_reset(),
                                  /*expected_value=*/false);
      }
    }
    for (const PortProto& input : signature_.data_inputs()) {
      tbt.SetX(input.name());
    }
    if (pipeline_control.has_value() && pipeline_control->has_valid()) {
      tbt.Set(pipeline_control->valid().input_name(), 0);
    }
    if (cycle < latency - 1) {
      tbt.AdvanceNCycles(latency - 1 - cycle);
    }
    while (captured_outputs < inputs.size()) {
      tbt.NextCycle();
      maybe_expect_output_valid(/*expect_x=*/false, /*expected_value=*/true);
      capture_outputs(captured_outputs);
      captured_outputs++;
    }
    // valid == 0 should have propagated all the way through the pipeline to
    // output_valid.
    tbt.NextCycle();
    maybe_expect_output_valid(/*expect_x=*/false, /*expected_value=*/false);
  } else if (signature_.proto().has_combinational()) {
    for (int64_t i = 0; i < inputs.size(); ++i) {
      drive_data(i);
      capture_outputs(i);
      tbt.NextCycle();
    }
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "Unsupported interface: ", signature_.proto().interface_oneof_case()));
  }

  XLS_RETURN_IF_ERROR(tb.Run());

  // Transfer outputs to an ArgumentSet for return.
  std::vector<BitsMap> outputs(inputs.size());
  for (int64_t i = 0; i < inputs.size(); ++i) {
    for (const auto& pair : stable_outputs[i]) {
      outputs[i][pair.first] = *pair.second;
    }
  }

  if (XLS_VLOG_IS_ON(1)) {
    XLS_VLOG(1) << "Results:\n";
    for (int64_t i = 0; i < outputs.size(); ++i) {
      XLS_VLOG(1) << "  Set " << i << ":";
      for (const auto& pair : outputs[i]) {
        XLS_VLOG(1) << "    " << pair.first << " : " << pair.second;
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
  XLS_CHECK_EQ(signature_.data_outputs().size(), 1);
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

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Bits>>>
ModuleSimulator::RunInputSeriesProc(
    const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts)
    const {
  XLS_VLOG(1) << "Running Verilog module with signature:\n"
              << signature_.ToString();
  if (XLS_VLOG_IS_ON(1)) {
    absl::flat_hash_map<std::string, std::vector<Value>> channel_inputs_values;
    for (const auto& [channel_name, channel_values] : channel_inputs) {
      XLS_ASSIGN_OR_RETURN(ChannelProto channel_proto,
                           signature_.GetInputChannelProtoByName(channel_name));
      XLS_ASSIGN_OR_RETURN(
          PortProto data_port,
          signature_.GetInputPortProtoByName(channel_proto.data_port_name()));
      XLS_ASSIGN_OR_RETURN(
          channel_inputs_values[channel_name],
          BitsListToValueList(channel_values, data_port.type()));
    }
    XLS_VLOG(1) << "Input channel values:\n";
    XLS_VLOG(1) << ChannelValuesToString(channel_inputs_values);
  }
  XLS_VLOG(2) << "Verilog:\n" << verilog_text_;

  absl::flat_hash_map<std::string, std::vector<Bits>> channel_outputs;

  // When there are no output channels or no expected value from the output
  // channels, the simulation simply terminates and there is no data collected.
  // For this special case, we exit early for performance.
  if (IsEmptyOrAreAllCountsZero(output_channel_counts)) {
    for (const auto& [name, _] : output_channel_counts) {
      channel_outputs[name] = std::vector<Bits>();
    }
    return channel_outputs;
  }

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

  if (!signature_.proto().has_clock_name() &&
      !signature_.proto().has_combinational()) {
    return absl::InvalidArgumentError("Expected clock in signature");
  }

  ModuleTestbench tb(verilog_text_, file_type_, signature_, simulator_,
                     includes_);

  int64_t max_channel_reads = 0;
  for (const auto& [_, read_count] : output_channel_counts) {
    max_channel_reads = std::max(max_channel_reads, read_count);
  }

  // TODO(vmirian) : 10-30-2022 Ensure semantics work for single value channel.
  for (const auto& [channel_name, channel_values] : channel_inputs) {
    XLS_ASSIGN_OR_RETURN(ChannelProto channel_proto,
                         signature_.GetInputChannelProtoByName(channel_name));
    absl::flat_hash_map<std::string, std::optional<Bits>>
        owned_signals_to_drive;
    std::string_view data_port_name = channel_proto.data_port_name();
    owned_signals_to_drive[data_port_name] = std::nullopt;
    std::optional<std::string> valid_port_name;
    std::optional<std::string> ready_port_name;
    if (channel_proto.has_valid_port_name()) {
      valid_port_name = channel_proto.valid_port_name();
      owned_signals_to_drive[valid_port_name.value()] = UBits(0, 1);
    }
    if (channel_proto.has_ready_port_name()) {
      ready_port_name = channel_proto.ready_port_name();
    }
    ModuleTestbenchThread& tbt =
        tb.CreateThread(owned_signals_to_drive, /*emit_done_signal=*/false);
    for (const Bits& value : channel_values) {
      tbt.Set(data_port_name, value);
      if (valid_port_name.has_value()) {
        tbt.Set(valid_port_name.value(), UBits(1, 1));
      }
      tbt.NextCycle();
      if (ready_port_name.has_value()) {
        tbt.WaitFor(ready_port_name.value());
      }
    }
    if (valid_port_name.has_value()) {
      tbt.SetX(valid_port_name.value());
    }
  }
  // Use std::unique_ptr for pointer stability necessary for
  // ModuleTestbench::Capture().
  using OutputMap = absl::flat_hash_map<std::string, std::unique_ptr<Bits>>;
  std::vector<OutputMap> stable_outputs(max_channel_reads);
  for (const ChannelProto& channel_proto : signature_.GetOutputChannels()) {
    std::string_view data_port_name = channel_proto.data_port_name();
    absl::flat_hash_map<std::string, std::optional<Bits>>
        owned_signals_to_drive;
    std::vector<std::string> input_driver_names;
    std::optional<std::string> valid_port_name;
    std::optional<std::string> ready_port_name;
    if (channel_proto.has_valid_port_name()) {
      valid_port_name = channel_proto.valid_port_name();
    }
    if (channel_proto.has_ready_port_name()) {
      ready_port_name = channel_proto.ready_port_name();
      owned_signals_to_drive[ready_port_name.value()] = UBits(0, 1);
    }
    ModuleTestbenchThread& tbt = tb.CreateThread(owned_signals_to_drive);
    for (int64_t read_count = 0;
         read_count < output_channel_counts.at(channel_proto.name());
         ++read_count) {
      if (ready_port_name.has_value()) {
        tbt.Set(ready_port_name.value(), UBits(1, 1));
      }
      if (valid_port_name.has_value()) {
        tbt.WaitFor(valid_port_name.value());
      }
      OutputMap& outputs = stable_outputs[read_count];
      outputs[data_port_name] = std::make_unique<Bits>();
      tbt.Capture(data_port_name, outputs.at(data_port_name).get());
      tbt.NextCycle();
    }
    if (ready_port_name.has_value()) {
      tbt.SetX(ready_port_name.value());
    }
  }

  XLS_RETURN_IF_ERROR(tb.Run());

  for (const ChannelProto& channel_proto : signature_.GetOutputChannels()) {
    channel_outputs[channel_proto.name()] = std::vector<Bits>();
  }

  for (const OutputMap& outputs : stable_outputs) {
    for (const auto& [data_port_name, value] : outputs) {
      XLS_ASSIGN_OR_RETURN(std::string channel_name,
                           signature_.GetChannelNameWith(data_port_name));
      XLS_ASSIGN_OR_RETURN(
          ChannelProto channel_proto,
          signature_.GetOutputChannelProtoByName(channel_name));
      channel_outputs[channel_proto.name()].push_back(*value);
    }
  }

  if (XLS_VLOG_IS_ON(1)) {
    absl::flat_hash_map<std::string, std::vector<Value>> result_channel_values;
    for (const auto& [channel_name, channel_values] : channel_outputs) {
      XLS_ASSIGN_OR_RETURN(
          ChannelProto channel_proto,
          signature_.GetOutputChannelProtoByName(channel_name));
      XLS_ASSIGN_OR_RETURN(
          PortProto data_port,
          signature_.GetOutputPortProtoByName(channel_proto.data_port_name()));
      XLS_ASSIGN_OR_RETURN(
          result_channel_values[channel_name],
          BitsListToValueList(channel_values, data_port.type()));
    }
    XLS_VLOG(1) << "Result channel values:\n";
    XLS_VLOG(1) << ChannelValuesToString(result_channel_values);
  }
  return channel_outputs;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ModuleSimulator::RunInputSeriesProc(
    const absl::flat_hash_map<std::string, std::vector<Value>>& channel_inputs,
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts)
    const {
  absl::flat_hash_map<std::string, std::vector<Value>> channel_outputs;

  // When there are no output channels or no expected value from the output
  // channels, the simulation simply terminates and there is no data collected.
  // For this special case, we exit early for performance. Moreover, although
  // this special case is handled in the overload with with xls::Bits, by
  // handling the case in this function, the translation between xls::Value and
  // xls::Bits for the input/output channel values are not performed, increasing
  // the performance further.
  if (IsEmptyOrAreAllCountsZero(output_channel_counts)) {
    for (const auto& [name, _] : output_channel_counts) {
      channel_outputs[name] = std::vector<Value>();
    }
    return channel_outputs;
  }

  using MapT = absl::flat_hash_map<std::string, std::vector<Bits>>;
  MapT channel_inputs_bits;
  for (const auto& [channel_name, channel_values] : channel_inputs) {
    XLS_RETURN_IF_ERROR(
        signature_.ValidateChannelValueInputs(channel_name, channel_values));
    channel_inputs_bits[channel_name] = ValueListToBitsList(channel_values);
  }

  XLS_ASSIGN_OR_RETURN(
      MapT channel_outputs_bits,
      RunInputSeriesProc(channel_inputs_bits, output_channel_counts));

  for (const auto& [channel_name, channel_bits] : channel_outputs_bits) {
    XLS_ASSIGN_OR_RETURN(ChannelProto channel_proto,
                         signature_.GetOutputChannelProtoByName(channel_name));
    XLS_ASSIGN_OR_RETURN(
        PortProto data_port,
        signature_.GetOutputPortProtoByName(channel_proto.data_port_name()));
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
