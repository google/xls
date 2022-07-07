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

#include "xls/codegen/module_signature.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"

namespace xls {
namespace verilog {

ModuleSignatureBuilder& ModuleSignatureBuilder::WithClock(
    absl::string_view name) {
  XLS_CHECK(!proto_.has_clock_name());
  proto_.set_clock_name(ToProtoString(name));
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithReset(
    absl::string_view name, bool asynchronous, bool active_low) {
  XLS_CHECK(!proto_.has_reset());
  ResetProto* reset = proto_.mutable_reset();
  reset->set_name(ToProtoString(name));
  reset->set_asynchronous(asynchronous);
  reset->set_active_low(active_low);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithReadyValidInterface(
    absl::string_view input_ready, absl::string_view input_valid,
    absl::string_view output_ready, absl::string_view output_valid) {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  ReadyValidInterface* interface = proto_.mutable_ready_valid();
  interface->set_input_ready(ToProtoString(input_ready));
  interface->set_input_valid(ToProtoString(input_valid));
  interface->set_output_ready(ToProtoString(output_ready));
  interface->set_output_valid(ToProtoString(output_valid));
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithFixedLatencyInterface(
    int64_t latency) {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  FixedLatencyInterface* interface = proto_.mutable_fixed_latency();
  interface->set_latency(latency);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithCombinationalInterface() {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  proto_.mutable_combinational();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithUnknownInterface() {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  proto_.mutable_unknown();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithPipelineInterface(
    int64_t latency, int64_t initiation_interval,
    std::optional<PipelineControl> pipeline_control) {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  PipelineInterface* interface = proto_.mutable_pipeline();
  interface->set_latency(latency);
  interface->set_initiation_interval(initiation_interval);
  if (pipeline_control.has_value()) {
    *interface->mutable_pipeline_control() = *pipeline_control;
  }
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithFunctionType(
    FunctionType* function_type) {
  XLS_CHECK(!proto_.has_function_type());
  *proto_.mutable_function_type() = function_type->ToProto();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddDataInput(
    absl::string_view name, int64_t width) {
  PortProto* port = proto_.add_data_ports();
  port->set_direction(DIRECTION_INPUT);
  port->set_name(ToProtoString(name));
  port->set_width(width);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddDataOutput(
    absl::string_view name, int64_t width) {
  PortProto* port = proto_.add_data_ports();
  port->set_direction(DIRECTION_OUTPUT);
  port->set_name(ToProtoString(name));
  port->set_width(width);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddSingleValueChannel(
    absl::string_view name, ChannelOps supported_ops,
    absl::string_view port_name) {
  ChannelProto* channel = proto_.add_data_channels();
  channel->set_name(ToProtoString(name));
  channel->set_kind(CHANNEL_KIND_SINGLE_VALUE);

  if (supported_ops == ChannelOps::kSendOnly) {
    channel->set_supported_ops(CHANNEL_OPS_SEND_ONLY);
  } else if (supported_ops == ChannelOps::kReceiveOnly) {
    channel->set_supported_ops(CHANNEL_OPS_RECEIVE_ONLY);
  } else {
    channel->set_supported_ops(CHANNEL_OPS_SEND_RECEIVE);
  }

  channel->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  channel->set_data_port_name(ToProtoString(port_name));

  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddStreamingChannel(
    absl::string_view name, ChannelOps supported_ops, FlowControl flow_control,
    std::optional<int64_t> fifo_depth, absl::string_view port_name,
    std::optional<absl::string_view> valid_port_name,
    std::optional<absl::string_view> ready_port_name) {
  ChannelProto* channel = proto_.add_data_channels();
  channel->set_name(ToProtoString(name));
  channel->set_kind(CHANNEL_KIND_STREAMING);

  if (supported_ops == ChannelOps::kSendOnly) {
    channel->set_supported_ops(CHANNEL_OPS_SEND_ONLY);
  } else if (supported_ops == ChannelOps::kReceiveOnly) {
    channel->set_supported_ops(CHANNEL_OPS_RECEIVE_ONLY);
  } else {
    channel->set_supported_ops(CHANNEL_OPS_SEND_RECEIVE);
  }

  if (flow_control == FlowControl::kReadyValid) {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_READY_VALID);
  } else {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  }

  if (fifo_depth.has_value()) {
    channel->set_fifo_depth(fifo_depth.value());
  }

  channel->set_data_port_name(ToProtoString(port_name));

  if (ready_port_name.has_value()) {
    channel->set_ready_port_name(ToProtoString(ready_port_name.value()));
  }

  if (valid_port_name.has_value()) {
    channel->set_valid_port_name(ToProtoString(valid_port_name.value()));
  }

  return *this;
}

absl::StatusOr<ModuleSignature> ModuleSignatureBuilder::Build() {
  return ModuleSignature::FromProto(proto_);
}

/*static*/ absl::StatusOr<ModuleSignature> ModuleSignature::FromProto(
    const ModuleSignatureProto& proto) {
  // TODO(meheff): do more validation here.
  // Validate widths/number of function type.
  if ((proto.has_pipeline() || proto.has_ready_valid()) &&
      !proto.has_clock_name()) {
    return absl::InvalidArgumentError("Missing clock signal");
  }

  ModuleSignature signature;
  signature.proto_ = proto;
  for (const PortProto& port : proto.data_ports()) {
    if (port.direction() == DIRECTION_INPUT) {
      signature.data_inputs_.push_back(port);
    } else if (port.direction() == DIRECTION_OUTPUT) {
      signature.data_outputs_.push_back(port);
    } else {
      return absl::InvalidArgumentError("Invalid port direction.");
    }
  }

  for (const ChannelProto& channel : proto.data_channels()) {
    if (channel.kind() == CHANNEL_KIND_SINGLE_VALUE) {
      signature.single_value_channels_.push_back(channel);
    } else if (channel.kind() == CHANNEL_KIND_STREAMING) {
      signature.streaming_channels_.push_back(channel);
    } else {
      return absl::InvalidArgumentError("Invalid channel kind.");
    }
  }

  return signature;
}

int64_t ModuleSignature::TotalDataInputBits() const {
  int64_t total = 0;
  for (const PortProto& port : data_inputs()) {
    total += port.width();
  }
  return total;
}

int64_t ModuleSignature::TotalDataOutputBits() const {
  int64_t total = 0;
  for (const PortProto& port : data_outputs()) {
    total += port.width();
  }
  return total;
}

// Checks that the given inputs match one-to-one to the input ports (matched by
// name). Returns a vector containing the inputs in the same order as the input
// ports.
template <typename T>
static absl::StatusOr<std::vector<const T*>> CheckAndReturnOrderedInputs(
    absl::Span<const PortProto> input_ports,
    const absl::flat_hash_map<std::string, T>& inputs) {
  absl::flat_hash_set<std::string> port_names;
  std::vector<const T*> ordered_inputs;
  for (const PortProto& port : input_ports) {
    port_names.insert(port.name());

    if (!inputs.contains(port.name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input '%s' was not passed as an argument.", port.name()));
    }
    ordered_inputs.push_back(&inputs.at(port.name()));
  }

  // Verify every passed in input is accounted for.
  for (const auto& pair : inputs) {
    if (!port_names.contains(pair.first)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unexpected input value named '%s'.", pair.first));
    }
  }
  return ordered_inputs;
}

absl::Status ModuleSignature::ValidateInputs(
    const absl::flat_hash_map<std::string, Bits>& input_bits) const {
  XLS_ASSIGN_OR_RETURN(std::vector<const Bits*> ordered_inputs,
                       CheckAndReturnOrderedInputs(data_inputs(), input_bits));
  for (int64_t i = 0; i < ordered_inputs.size(); ++i) {
    const PortProto& port = data_inputs()[i];
    const Bits* input = ordered_inputs[i];
    if (port.width() != input->bit_count()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected input '%s' to have width %d, has width %d",
                          port.name(), port.width(), input->bit_count()));
    }
  }
  return absl::OkStatus();
}

static std::string TypeProtoToString(const TypeProto& proto) {
  // Create a dummy package for creating Type*'s from a proto.
  // TODO(meheff): Find a better way to manage types. We need types disconnected
  // from any IR package.
  Package p("dummy_package");
  auto type_status = p.GetTypeFromProto(proto);
  if (!type_status.ok()) {
    return "<invalid>";
  }
  return type_status.value()->ToString();
}

static absl::StatusOr<bool> TypeProtosEqual(const TypeProto& a,
                                            const TypeProto& b) {
  // Create a dummy package for creating Type*'s from a proto.
  // TODO(meheff): Find a better way to manage types. We need types disconnected
  // from any IR package.
  Package p("dummy_package");
  XLS_ASSIGN_OR_RETURN(Type * a_type, p.GetTypeFromProto(a));
  XLS_ASSIGN_OR_RETURN(Type * b_type, p.GetTypeFromProto(b));
  return a_type == b_type;
}

absl::Status ModuleSignature::ValidateInputs(
    const absl::flat_hash_map<std::string, Value>& input_values) const {
  if (!proto().has_function_type()) {
    return absl::InvalidArgumentError(
        "Cannot validate Value inputs because signature has no function_type "
        "field");
  }
  XLS_ASSIGN_OR_RETURN(
      std::vector<const Value*> ordered_inputs,
      CheckAndReturnOrderedInputs(data_inputs(), input_values));
  XLS_RET_CHECK_EQ(data_inputs().size(),
                   proto().function_type().parameters_size());
  for (int64_t i = 0; i < ordered_inputs.size(); ++i) {
    const Value* input = ordered_inputs[i];
    const TypeProto& expected_type_proto =
        proto().function_type().parameters(i);
    XLS_ASSIGN_OR_RETURN(TypeProto value_type_proto, input->TypeAsProto());
    XLS_ASSIGN_OR_RETURN(bool types_equal, TypeProtosEqual(expected_type_proto,
                                                           value_type_proto));
    if (!types_equal) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input value '%s' is wrong type. Expected '%s', got '%s'",
          data_inputs()[i].name(), TypeProtoToString(expected_type_proto),
          TypeProtoToString(value_type_proto)));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
ModuleSignature::ToKwargs(absl::Span<const Value> inputs) const {
  if (inputs.size() != data_inputs().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %d arguments, got %d.", data_inputs().size(), inputs.size()));
  }
  absl::flat_hash_map<std::string, Value> kwargs;
  for (int64_t i = 0; i < data_inputs().size(); ++i) {
    kwargs[data_inputs()[i].name()] = inputs[i];
  }
  return kwargs;
}

absl::Status ModuleSignature::ReplaceBlockMetrics(
    BlockMetricsProto block_metrics) {
  *proto_.mutable_metrics()->mutable_block_metrics() = std::move(block_metrics);
  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os, const ModuleSignature& signature) {
  os << signature.ToString();
  return os;
}

}  // namespace verilog
}  // namespace xls
