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
#include <optional>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/op_override.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"

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

std::optional<ResetBehavior> CodegenOptions::GetResetBehavior() const {
  if (!reset_proto_.has_value()) {
    return std::nullopt;
  }
  return ResetBehavior{
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

CodegenOptions& CodegenOptions::max_inline_depth(int64_t value) {
  max_inline_depth_ = value;
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

CodegenOptions& CodegenOptions::SetOpOverride(Op kind,
                                              OpOverride configuration) {
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
    absl::Span<const RamConfiguration> ram_configurations) {
  ram_configurations_.clear();
  for (auto& config : ram_configurations) {
    ram_configurations_.push_back(config);
  }
  return *this;
}

CodegenOptions& CodegenOptions::register_merge_strategy(
    CodegenOptions::RegisterMergeStrategy strategy) {
  register_merge_strategy_ = strategy;
  return *this;
}

CodegenOptions& CodegenOptions::add_invariant_assertions(bool value) {
  add_invariant_assertions_ = value;
  return *this;
}

}  // namespace xls::verilog
