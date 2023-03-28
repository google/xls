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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/op_override.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"

namespace xls::verilog {

// Options describing how codegen should be performed.
class CodegenOptions {
 public:
  explicit CodegenOptions() = default;

  CodegenOptions(const CodegenOptions& options);
  CodegenOptions& operator=(const CodegenOptions& options);
  CodegenOptions(CodegenOptions&& options) = default;
  CodegenOptions& operator=(CodegenOptions&& options) = default;
  ~CodegenOptions() = default;

  // Enum to describe how IO should be registered.
  enum class IOKind { kFlop = 0, kSkidBuffer, kZeroLatencyBuffer };

  // Convert IOKind enum to a string.
  static std::string_view IOKindToString(IOKind kind);

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
  CodegenOptions& entry(std::string_view name);
  std::optional<std::string_view> entry() const { return entry_; }

  // Name to use for the generated module. If not given, the name of the XLS
  // function/proc is used.
  CodegenOptions& module_name(std::string_view name);
  const std::optional<std::string_view> module_name() const {
    return module_name_;
  }

  // Reset signal to use for any registers with initial values. Required if the
  // proc contains any registers with initial values.
  CodegenOptions& reset(std::string_view name, bool asynchronous,
                        bool active_low, bool reset_data_path);
  const std::optional<ResetProto>& reset() const { return reset_proto_; }

  // Returns an xls::Reset constructed from the reset() proto.
  std::optional<xls::Reset> ResetBehavior() const;

  // Specifies manual pipeline register load-enable control.
  CodegenOptions& manual_control(std::string_view input_name);
  std::optional<ManualPipelineControl> manual_control() const;

  // Specifies pipeline register load-enable controlled by a valid signal.
  CodegenOptions& valid_control(std::string_view input_name,
                                std::optional<std::string_view> output_name);
  std::optional<ValidProto> valid_control() const;

  // Returns the proto describing the pipeline control scheme.
  const std::optional<PipelineControl>& control() const {
    return pipeline_control_;
  }

  // Name of the clock signal. Required if the block has any registers.
  CodegenOptions& clock_name(std::string_view clock_name);
  std::optional<std::string_view> clock_name() const { return clock_name_; }

  // Whether to use SystemVerilog in the generated code otherwise Verilog is
  // used. The default is to use SystemVerilog.
  CodegenOptions& use_system_verilog(bool value);
  bool use_system_verilog() const { return use_system_verilog_; }

  // Whether to emit everything on a separate line, which is useful when
  // using the area profiler.
  CodegenOptions& separate_lines(bool value);
  bool separate_lines() const { return separate_lines_; }

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

  // Set an OpOverride to customize codegen for an Op.
  CodegenOptions& SetOpOverride(Op kind,
                                std::unique_ptr<OpOverride> configuration);
  // Get the OpOverride for an op, if it's defined.
  std::optional<OpOverride*> GetOpOverride(Op kind) const {
    auto itr = op_overrides_.find(kind);
    if (itr == op_overrides_.end()) {
      return std::nullopt;
    }
    return itr->second.get();
  }

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
  CodegenOptions& streaming_channel_data_suffix(std::string_view value);
  std::string_view streaming_channel_data_suffix() const {
    return streaming_channel_data_suffix_;
  }

  // For ready_valid channels (data, ready, valid), append value to the
  // channel's name for the signals corresponding to valid.
  //
  // Default is "_vld"
  CodegenOptions& streaming_channel_valid_suffix(std::string_view value);
  std::string_view streaming_channel_valid_suffix() const {
    return streaming_channel_valid_suffix_;
  }

  // For ready_valid channels (data, ready, valid), append value to the
  // channel's name for the signals corresponding to ready.
  //
  // Default is "_rdy"
  CodegenOptions& streaming_channel_ready_suffix(std::string_view value);
  std::string_view streaming_channel_ready_suffix() const {
    return streaming_channel_ready_suffix_;
  }

  // Emit bounds checking on array-index operations in Verilog. In the XLS IR an
  // out of bounds array-index operation returns the maximal index element in
  // the array. This can be expensive to synthesize. Setting this value to false
  // may result in mismatches between IR-level evaluation and Verilog
  // simulation.
  CodegenOptions& array_index_bounds_checking(bool value);
  bool array_index_bounds_checking() const {
    return array_index_bounds_checking_;
  }

  // Emit logic to gate the data value of a receive operation in Verilog. In the
  // XLS IR, the receive operation has the semantics that the data value is zero
  // when the predicate is `false`. Moreover, for a non-blocking receive, the
  // data value is zero when the data is invalid. When set to true, the data is
  // gated and has the previously described semantics. However, the latter does
  // utilize more resource/area. Setting this value to false may reduce the
  // resource/area utilization, but may also result in mismatches between
  // IR-level evaluation and Verilog simulation.
  CodegenOptions& gate_recvs(bool value);
  bool gate_recvs() const { return gate_recvs_; }

  // List of channels to rewrite for RAMs.
  CodegenOptions& ram_configurations(
      absl::Span<const std::unique_ptr<RamConfiguration>> ram_configurations);
  absl::Span<const std::unique_ptr<RamConfiguration>> ram_configurations()
      const {
    return ram_configurations_;
  }

 private:
  std::optional<std::string> entry_;
  std::optional<std::string> module_name_;
  std::optional<ResetProto> reset_proto_;
  std::optional<PipelineControl> pipeline_control_;
  std::optional<std::string> clock_name_;
  bool use_system_verilog_ = true;
  bool separate_lines_ = false;
  bool flop_inputs_ = false;
  bool flop_outputs_ = false;
  IOKind flop_inputs_kind_ = IOKind::kFlop;
  IOKind flop_outputs_kind_ = IOKind::kFlop;
  bool split_outputs_ = false;
  bool add_idle_output_ = false;
  bool flop_single_value_channels_ = false;
  absl::flat_hash_map<Op, std::unique_ptr<OpOverride>> op_overrides_;
  bool emit_as_pipeline_ = false;
  std::string streaming_channel_data_suffix_ = "";
  std::string streaming_channel_ready_suffix_ = "_rdy";
  std::string streaming_channel_valid_suffix_ = "_vld";
  bool array_index_bounds_checking_ = true;
  bool gate_recvs_ = true;
  std::vector<std::unique_ptr<RamConfiguration>> ram_configurations_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_OPTIONS_H_
