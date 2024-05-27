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

#include "xls/codegen/codegen_options.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/op_override.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/ir/bits.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"

namespace xls::verilog {

/* static */ std::string_view CodegenOptions::IOKindToString(IOKind kind) {
  switch (kind) {
    case IOKind::kFlop:
      return "kFlop";
    case IOKind::kSkidBuffer:
      return "kSkidBuffer";
    case IOKind::kZeroLatencyBuffer:
      return "kZeroLatencyBuffer";
  }
  LOG(FATAL) << "Invalid IOKind: " << static_cast<int64_t>(kind);
}

CodegenOptions::CodegenOptions(const CodegenOptions& options)
    : entry_(options.entry_),
      module_name_(options.module_name_),
      output_port_name_(options.output_port_name_),
      reset_proto_(options.reset_proto_),
      pipeline_control_(options.pipeline_control_),
      clock_name_(options.clock_name_),
      use_system_verilog_(options.use_system_verilog_),
      separate_lines_(options.separate_lines_),
      flop_inputs_(options.flop_inputs_),
      flop_outputs_(options.flop_outputs_),
      flop_inputs_kind_(options.flop_inputs_kind_),
      flop_outputs_kind_(options.flop_outputs_kind_),
      split_outputs_(options.split_outputs_),
      add_idle_output_(options.add_idle_output_),
      flop_single_value_channels_(options.flop_single_value_channels_),
      emit_as_pipeline_(options.emit_as_pipeline_),
      streaming_channel_data_suffix_(options.streaming_channel_data_suffix_),
      streaming_channel_ready_suffix_(options.streaming_channel_ready_suffix_),
      streaming_channel_valid_suffix_(options.streaming_channel_valid_suffix_),
      array_index_bounds_checking_(options.array_index_bounds_checking_),
      gate_recvs_(options.gate_recvs_),
      register_merge_strategy_(options.register_merge_strategy_) {
  for (auto& [op, op_override] : options.op_overrides_) {
    op_overrides_.insert_or_assign(op, op_override->Clone());
  }
  ram_configurations_.reserve(options.ram_configurations().size());
  for (auto& option : options.ram_configurations()) {
    ram_configurations_.push_back(option->Clone());
  }
}

CodegenOptions& CodegenOptions::operator=(const CodegenOptions& options) {
  entry_ = options.entry_;
  module_name_ = options.module_name_;
  output_port_name_ = options.output_port_name_;
  reset_proto_ = options.reset_proto_;
  pipeline_control_ = options.pipeline_control_;
  clock_name_ = options.clock_name_;
  use_system_verilog_ = options.use_system_verilog_;
  separate_lines_ = options.separate_lines_;
  flop_inputs_ = options.flop_inputs_;
  flop_outputs_ = options.flop_outputs_;
  flop_inputs_kind_ = options.flop_inputs_kind_;
  flop_outputs_kind_ = options.flop_outputs_kind_;
  split_outputs_ = options.split_outputs_;
  add_idle_output_ = options.add_idle_output_;
  flop_single_value_channels_ = options.flop_single_value_channels_;
  emit_as_pipeline_ = options.emit_as_pipeline_;
  streaming_channel_data_suffix_ = options.streaming_channel_data_suffix_;
  streaming_channel_ready_suffix_ = options.streaming_channel_ready_suffix_;
  streaming_channel_valid_suffix_ = options.streaming_channel_valid_suffix_;
  array_index_bounds_checking_ = options.array_index_bounds_checking_;
  gate_recvs_ = options.gate_recvs_;
  register_merge_strategy_ = options.register_merge_strategy_;
  for (auto& [op, op_override] : options.op_overrides_) {
    op_overrides_.insert_or_assign(op, op_override->Clone());
  }
  ram_configurations_.reserve(options.ram_configurations().size());
  for (auto& option : options.ram_configurations()) {
    ram_configurations_.push_back(option->Clone());
  }
  return *this;
}

CodegenOptions& CodegenOptions::entry(std::string_view name) {
  entry_ = name;
  return *this;
}

CodegenOptions& CodegenOptions::module_name(std::string_view name) {
  module_name_ = name;
  return *this;
}

CodegenOptions& CodegenOptions::output_port_name(std::string_view name) {
  output_port_name_ = name;
  return *this;
}

CodegenOptions& CodegenOptions::reset(std::string_view name, bool asynchronous,
                                      bool active_low, bool reset_data_path) {
  reset_proto_ = ResetProto();
  reset_proto_->set_name(ToProtoString(name));
  reset_proto_->set_asynchronous(asynchronous);
  reset_proto_->set_active_low(active_low);
  reset_proto_->set_reset_data_path(reset_data_path);
  return *this;
}

std::optional<xls::Reset> CodegenOptions::ResetBehavior() const {
  if (!reset_proto_.has_value()) {
    return std::nullopt;
  }
  return xls::Reset{
      .reset_value = Value(UBits(0, 1)),
      .asynchronous = reset_proto_->asynchronous(),
      .active_low = reset_proto_->active_low(),
  };
}

CodegenOptions& CodegenOptions::manual_control(std::string_view input_name) {
  if (!pipeline_control_.has_value()) {
    pipeline_control_ = PipelineControl();
  }
  pipeline_control_->mutable_manual()->set_input_name(
      ToProtoString(input_name));
  return *this;
}

std::optional<ManualPipelineControl> CodegenOptions::manual_control() const {
  if (!pipeline_control_.has_value() || !pipeline_control_->has_manual()) {
    return std::nullopt;
  }
  return pipeline_control_->manual();
}

CodegenOptions& CodegenOptions::valid_control(
    std::string_view input_name, std::optional<std::string_view> output_name) {
  if (!pipeline_control_.has_value()) {
    pipeline_control_ = PipelineControl();
  }
  ValidProto* valid = pipeline_control_->mutable_valid();
  valid->set_input_name(ToProtoString(input_name));
  if (output_name.has_value()) {
    valid->set_output_name(ToProtoString(*output_name));
  }
  return *this;
}

std::optional<ValidProto> CodegenOptions::valid_control() const {
  if (!pipeline_control_.has_value() || !pipeline_control_->has_valid()) {
    return std::nullopt;
  }
  return pipeline_control_->valid();
}

CodegenOptions& CodegenOptions::clock_name(std::string_view clock_name) {
  clock_name_ = clock_name;
  return *this;
}

CodegenOptions& CodegenOptions::use_system_verilog(bool value) {
  use_system_verilog_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::separate_lines(bool value) {
  separate_lines_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::flop_inputs(bool value) {
  flop_inputs_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::flop_outputs(bool value) {
  flop_outputs_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::flop_inputs_kind(IOKind value) {
  flop_inputs_kind_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::flop_outputs_kind(IOKind value) {
  flop_outputs_kind_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::flop_single_value_channels(bool value) {
  flop_single_value_channels_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::split_outputs(bool value) {
  split_outputs_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::add_idle_output(bool value) {
  add_idle_output_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::SetOpOverride(
    Op kind, std::unique_ptr<OpOverride> configuration) {
  op_overrides_.insert_or_assign(kind, std::move(configuration));
  return *this;
}

CodegenOptions& CodegenOptions::emit_as_pipeline(bool value) {
  emit_as_pipeline_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::streaming_channel_data_suffix(
    std::string_view value) {
  streaming_channel_data_suffix_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::streaming_channel_valid_suffix(
    std::string_view value) {
  streaming_channel_valid_suffix_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::streaming_channel_ready_suffix(
    std::string_view value) {
  streaming_channel_ready_suffix_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::array_index_bounds_checking(bool value) {
  array_index_bounds_checking_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::gate_recvs(bool value) {
  gate_recvs_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::ram_configurations(
    absl::Span<const std::unique_ptr<RamConfiguration>> ram_configurations) {
  ram_configurations_.clear();
  for (auto& config : ram_configurations) {
    ram_configurations_.push_back(config->Clone());
  }
  return *this;
}

CodegenOptions& CodegenOptions::register_merge_strategy(
    CodegenOptions::RegisterMergeStrategy strategy) {
  register_merge_strategy_ = strategy;
  return *this;
}

}  // namespace xls::verilog
