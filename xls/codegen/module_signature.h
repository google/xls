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

#ifndef XLS_CODEGEN_MODULE_SIGNATURE_H_
#define XLS_CODEGEN_MODULE_SIGNATURE_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace verilog {

class ModuleSignature;

// A builder for constructing ModuleSignatures (descriptions of Verilog module
// interfaces).
class ModuleSignatureBuilder {
 public:
  explicit ModuleSignatureBuilder(absl::string_view module_name) {
    proto_.set_module_name(ToProtoString(module_name));
  }

  // Sets the clock as having the given name.
  ModuleSignatureBuilder& WithClock(absl::string_view name);

  // Sets the reset signal as having the given name and properties.
  ModuleSignatureBuilder& WithReset(absl::string_view name, bool asynchronous,
                                    bool active_low);

  // Defines the module interface as using ready/valid flow control with signals
  // of the given names.
  ModuleSignatureBuilder& WithReadyValidInterface(
      absl::string_view input_ready, absl::string_view input_valid,
      absl::string_view output_ready, absl::string_view output_valid);

  // Defines the module interface as fixed latency.
  ModuleSignatureBuilder& WithFixedLatencyInterface(int64_t latency);

  // Defines the module interface as pipelined with the given latency and
  // initiation interval.
  ModuleSignatureBuilder& WithPipelineInterface(
      int64_t latency, int64_t initiation_interval,
      std::optional<PipelineControl> pipeline_control = absl::nullopt);

  // Defines the module interface as purely combinational.
  ModuleSignatureBuilder& WithCombinationalInterface();

  // Defines the module interface with an unspecified or non-existent
  // protocol. The code generator supports emission of arbitrary RTL and not all
  // blocks conform to the predefined interfaces: feed-forward pipelines,
  // ready/valid, fixed latency or purely combinational. This option is a
  // catch-all for such nonconforming cases.
  ModuleSignatureBuilder& WithUnknownInterface();

  // Sets the type of the function to the given string. The expected form is
  // defined by xls::FunctionType::ToString.
  ModuleSignatureBuilder& WithFunctionType(FunctionType* function_type);

  // Add data input/outputs to the interface. Control signals such as the clock,
  // reset, ready/valid signals, etc should not be added using these methods.
  ModuleSignatureBuilder& AddDataInput(absl::string_view name, int64_t width);
  ModuleSignatureBuilder& AddDataOutput(absl::string_view name, int64_t width);

  // Add a single value channel to the interface.
  ModuleSignatureBuilder& AddSingleValueChannel(absl::string_view name,
                                                ChannelOps supported_ops,
                                                absl::string_view port_name);

  // Add a streaming channel to the interface
  ModuleSignatureBuilder& AddStreamingChannel(
      absl::string_view name, ChannelOps supported_ops,
      FlowControl flow_control, std::optional<int64_t> fifo_depth,
      absl::string_view port_name,
      std::optional<absl::string_view> valid_port_name =
          std::optional<absl::string_view>(),
      std::optional<absl::string_view> ready_port_name =
          std::optional<absl::string_view>());

  absl::StatusOr<ModuleSignature> Build();

 private:
  ModuleSignatureProto proto_;
};

// An abstraction describing the interface to a Verilog module. At the moment
// this is a thin wrapper around a proto and most of the fields are accessed
// directly through the proto (ModuleSignature::proto). However the class has
// the benefit of invariant enforcement, convenience methods, and is a framework
// to expand the interface.
class ModuleSignature {
 public:
  static absl::StatusOr<ModuleSignature> FromProto(
      const ModuleSignatureProto& proto);

  const std::string& module_name() const { return proto_.module_name(); }

  const ModuleSignatureProto& proto() const { return proto_; }

  // Return the signature as the proto in text form.
  std::string AsTextProto() const { return proto_.DebugString(); }

  // Returns the data inputs/outputs of module. This does not include clock,
  // reset, etc. These ports necessarily exist in the proto as well but are
  // duplicated here for convenience.
  absl::Span<const PortProto> data_inputs() const { return data_inputs_; }
  absl::Span<const PortProto> data_outputs() const { return data_outputs_; }

  // Returns the single value channels of the module.
  absl::Span<const ChannelProto> single_value_channels() {
    return single_value_channels_;
  }

  // Returns the streaming channels of the module.
  absl::Span<const ChannelProto> streaming_channels() {
    return streaming_channels_;
  }

  // Returns the total number of bits of the data input/outputs.
  int64_t TotalDataInputBits() const;
  int64_t TotalDataOutputBits() const;

  std::string ToString() const { return proto_.DebugString(); }

  // Verifies that the given data input Bits(Values) are exactly the expected
  // set and of the appropriate type for the module.
  absl::Status ValidateInputs(
      const absl::flat_hash_map<std::string, Bits>& input_bits) const;
  absl::Status ValidateInputs(
      const absl::flat_hash_map<std::string, Value>& input_values) const;

  // Converts the ordered set of Value arguments to the module of the signature
  // into an argument name-value map.
  absl::StatusOr<absl::flat_hash_map<std::string, Value>> ToKwargs(
      absl::Span<const Value> inputs) const;

  // Replace the signature with block metrics created during codegen.
  // TODO(tedhong): 2022-01-28 Support incremental update of metrics.
  absl::Status ReplaceBlockMetrics(BlockMetricsProto block_metrics);

 private:
  ModuleSignatureProto proto_;

  // These ports also exist in the proto, but are duplicated here to enable the
  // convenience methods data_inputs() and data_outputs().
  std::vector<PortProto> data_inputs_;
  std::vector<PortProto> data_outputs_;

  // These channels also exist in the proto, but are duplicated here to enable
  // the convenience methods single_value_channels() and streaming_channels()
  std::vector<ChannelProto> single_value_channels_;
  std::vector<ChannelProto> streaming_channels_;
};

// Abstraction gathering the Verilog text and module signature produced by the
// generator.
struct ModuleGeneratorResult {
  std::string verilog_text;
  VerilogLineMap verilog_line_map;
  ModuleSignature signature;
};

std::ostream& operator<<(std::ostream& os, const ModuleSignature& signature);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_MODULE_SIGNATURE_H_
