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

#ifndef XLS_CODEGEN_CODEGEN_OPTIONS_H_
#define XLS_CODEGEN_CODEGEN_OPTIONS_H_

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// Options describing how codegen should be performed.
class CodegenOptions {
 public:
  // Enum to describe how IO should be registered.
  enum class IOKind { kFlop = 0, kSkidBuffer, kZeroLatencyBuffer };

  // Convert IOKind enum to a string.
  static absl::string_view IOKindToString(IOKind kind) {
    switch (kind) {
      case IOKind::kFlop:
        return "kFlop";
      case IOKind::kSkidBuffer:
        return "kSkidBuffer";
      case IOKind::kZeroLatencyBuffer:
        return "kZeroLatencyBuffer";
      default:
        return "UnknownKind";
    }
  }

  // Added latency for each IOKind.
  static int64_t IOKindLatency(IOKind kind) {
    switch (kind) {
      case IOKind::kFlop:
        return 1;
      case IOKind::kSkidBuffer:
        return 1;
      case IOKind::kZeroLatencyBuffer:
        return 0;
      default:
        return 9999;
    }
  }

  // The name of the top-level function or proc to generate a Verilog module
  // for. Required.
  // TODO(meheff): 2021/04/21 As this is required, perhaps this should be made a
  // constructor argument.
  CodegenOptions& entry(absl::string_view name);
  absl::optional<absl::string_view> entry() const { return entry_; }

  // Name to use for the generated module. If not given, the name of the XLS
  // function/proc is used.
  CodegenOptions& module_name(absl::string_view name);
  const absl::optional<std::string_view> module_name() const {
    return module_name_;
  }

  // Reset signal to use for any registers with initial values. Required if the
  // proc contains any registers with initial values.
  CodegenOptions& reset(absl::string_view name, bool asynchronous,
                        bool active_low, bool reset_data_path);
  const absl::optional<ResetProto>& reset() const { return reset_proto_; }

  // Specifies manual pipeline register load-enable control.
  CodegenOptions& manual_control(absl::string_view input_name);
  absl::optional<ManualPipelineControl> manual_control() const;

  // Specifies pipeline register load-enable controlled by a valid signal.
  CodegenOptions& valid_control(absl::string_view input_name,
                                absl::optional<absl::string_view> output_name);
  absl::optional<ValidProto> valid_control() const;

  // Returns the proto describing the pipeline control scheme.
  const absl::optional<PipelineControl>& control() const {
    return pipeline_control_;
  }

  // Name of the clock signal. Required if the block has any registers.
  CodegenOptions& clock_name(absl::string_view clock_name);
  absl::optional<absl::string_view> clock_name() const { return clock_name_; }

  // Whether to use SystemVerilog in the generated code otherwise Verilog is
  // used. The default is to use SystemVerilog.
  CodegenOptions& use_system_verilog(bool value);
  bool use_system_verilog() const { return use_system_verilog_; }

  // Whether to flop inputs into a register at the beginning of the pipeline. If
  // true, adds a single cycle to the latency of the pipline.
  CodegenOptions& flop_inputs(bool value);
  bool flop_inputs() const { return flop_inputs_; }

  // Whether to flop outputs into a register at the end of the pipeline. If
  // true, adds a single cycle to the latency of the pipline.
  CodegenOptions& flop_outputs(bool value);
  bool flop_outputs() const { return flop_outputs_; }

  // When flop_inputs() is true, determines the type of flop to add.
  CodegenOptions& flop_inputs_kind(IOKind value);
  IOKind flop_inputs_kind() const { return flop_inputs_kind_; }

  // When flop_outputs() is true, determines the type of flop to add.
  CodegenOptions& flop_outputs_kind(IOKind value);
  IOKind flop_outputs_kind() const { return flop_outputs_kind_; }

  // Returns the input latency, if any, associated with registring the input.
  int64_t GetInputLatency() const {
    return flop_inputs() ? IOKindLatency(flop_inputs_kind_) : 0;
  }

  // Returns the output latency, if any, associated with registring the input.
  int64_t GetOutputLatency() const {
    return flop_outputs() ? IOKindLatency(flop_outputs_kind_) : 0;
  }

  // When false, single value channels are not registered even if
  // flop_inputs() or flop_outputs() is true.
  CodegenOptions& flop_single_value_channels(bool value);
  bool flop_single_value_channels() const {
    return flop_single_value_channels_;
  }

  // If the output is tuple-typed, generate an output port for each element of
  // the output tuple.
  CodegenOptions& split_outputs(bool value);
  bool split_outputs() const { return split_outputs_; }

  // Add a single idle signal output, tied to the nor of all valid signals.
  CodegenOptions& add_idle_output(bool value);
  bool add_idle_output() const { return add_idle_output_; }

  // Format string to use when emitting assert operations in Verilog. Supports
  // the following placeholders:
  //
  //  {message}   : Message of the assert operation.
  //  {condition} : Condition of the assert.
  //  {label}     : Label of the assert operation. Returns error if the
  //                operation has no label.
  //  {clk}       : Name of the clock signal. Returns error if no clock is
  //                specified.
  //  {rst}       : Name of the reset signal. Returns error if no reset is
  //                specified.
  //
  // For example, the format string:
  //
  //    '{label}: `MY_ASSERT({condition}, "{message}")'
  //
  // Might result in the following in the emitted Verilog:
  //
  //    my_label: `MY_ASSERT(foo < 8'h42, "Oh noes!");
  CodegenOptions& assert_format(absl::string_view value);
  absl::optional<absl::string_view> assert_format() const {
    return assert_format_;
  }

  // Format string to use when emitting gate operations in Verilog. Supports the
  // following placeholders:
  //
  //  {condition} : Identifier (or expression) of the condition of the assert.
  //  {input}     : The identifier (or expression) for the data input of the
  //                gate operation.
  //  {output}    : The identifier of the gate operation.
  //  {width}     : The bit width of the gate operation.
  //
  // For example, consider a format string which instantiates a particular
  // custom AND gate for gating:
  //
  //    'my_and gated_{output} [{width}-1:0] (.Z({output}), .A({condition}),
  //    .B({input}))'
  //
  // And the IR gate operations is:
  //
  //    the_result: bits[32] = gate(the_cond, the_data)
  //
  // This results in the following emitted Verilog:
  //
  //    my_and gated_the_result [32-1:0] (.Z(the_result), .A(the cond),
  //    .B(the_data));
  //
  // To ensure valid Verilog, the instantiated template must declare a value
  // named {output} (e.g., `the_result` in the example).
  //
  // If no format value is given, then a logical AND with the condition value is
  // generated. For example:
  //
  //   wire the_result [31:0];
  //   assign the_result = {32{the_cond}} & the_data;
  CodegenOptions& gate_format(absl::string_view value);
  absl::optional<absl::string_view> gate_format() const { return gate_format_; }

  // Emit the signal declarations and logic in the Verilog as a sequence of
  // pipeline stages separated by per-stage comment headers. The option does not
  // functionally change the generated Verilog but rather affects its layout.
  // The registers must be strictly layered or an error is returned during code
  // generation.
  CodegenOptions& emit_as_pipeline(bool value);
  bool emit_as_pipeline() const { return emit_as_pipeline_; }

  // For ready_valid channels append value to the channel's name for the
  // signals corresponding to data.
  //
  // Default is no suffix so that the signal name matches the channel name.
  CodegenOptions& streaming_channel_data_suffix(absl::string_view value);
  absl::string_view streaming_channel_data_suffix() const {
    return streaming_channel_data_suffix_;
  }

  // For ready_valid channels (data, ready, valid), append value to the
  // channel's name for the signals corresponding to valid.
  //
  // Default is "_vld"
  CodegenOptions& streaming_channel_valid_suffix(absl::string_view value);
  absl::string_view streaming_channel_valid_suffix() const {
    return streaming_channel_valid_suffix_;
  }

  // For ready_valid channels (data, ready, valid), append value to the
  // channel's name for the signals corresponding to ready.
  //
  // Default is "_rdy"
  CodegenOptions& streaming_channel_ready_suffix(absl::string_view value);
  absl::string_view streaming_channel_ready_suffix() const {
    return streaming_channel_ready_suffix_;
  }

 private:
  absl::optional<std::string> entry_;
  absl::optional<std::string> module_name_;
  absl::optional<ResetProto> reset_proto_;
  absl::optional<PipelineControl> pipeline_control_;
  absl::optional<std::string> clock_name_;
  bool use_system_verilog_ = true;
  bool flop_inputs_ = false;
  bool flop_outputs_ = false;
  IOKind flop_inputs_kind_ = IOKind::kFlop;
  IOKind flop_outputs_kind_ = IOKind::kFlop;
  bool split_outputs_ = false;
  bool add_idle_output_ = false;
  bool flop_single_value_channels_ = false;
  absl::optional<std::string> assert_format_;
  absl::optional<std::string> gate_format_;
  bool emit_as_pipeline_ = false;
  std::string streaming_channel_data_suffix_ = "";
  std::string streaming_channel_ready_suffix_ = "_rdy";
  std::string streaming_channel_valid_suffix_ = "_vld";
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_OPTIONS_H_
