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

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace verilog {

class ModuleSignature;

// A builder for constructing ModuleSignatures (descriptions of Verilog module
// interfaces).
class ModuleSignatureBuilder {
 public:
  explicit ModuleSignatureBuilder(std::string_view module_name) {
    proto_.set_module_name(ToProtoString(module_name));
  }

  static ModuleSignatureBuilder FromProto(ModuleSignatureProto proto) {
    return ModuleSignatureBuilder(std::move(proto));
  }

  // Sets the clock as having the given name.
  ModuleSignatureBuilder& WithClock(std::string_view name);

  // Sets the reset signal as having the given name and properties.
  ModuleSignatureBuilder& WithReset(std::string_view name, bool asynchronous,
                                    bool active_low);

  // Defines the module interface as fixed latency.
  ModuleSignatureBuilder& WithFixedLatencyInterface(int64_t latency);

  // Defines the module interface as pipelined with the given latency and
  // initiation interval.
  ModuleSignatureBuilder& WithPipelineInterface(
      int64_t latency, int64_t initiation_interval,
      std::optional<PipelineControl> pipeline_control = std::nullopt);

  // Defines the module interface as purely combinational.
  ModuleSignatureBuilder& WithCombinationalInterface();

  // Defines the module interface with an unspecified or non-existent
  // protocol. The code generator supports emission of arbitrary RTL and not all
  // blocks conform to the predefined interfaces: feed-forward pipelines,
  // ready/valid, fixed latency or purely combinational. This option is a
  // catch-all for such nonconforming cases.
  ModuleSignatureBuilder& WithUnknownInterface();

  // Add data input/outputs to the interface. Control signals such as the clock,
  // reset, ready/valid signals, etc should not be added using these methods.
  ModuleSignatureBuilder& AddDataInput(std::string_view name, Type* type);
  ModuleSignatureBuilder& AddDataOutput(std::string_view name, Type* type);
  ModuleSignatureBuilder& AddDataInputAsBits(std::string_view name,
                                             int64_t width);
  ModuleSignatureBuilder& AddDataOutputAsBits(std::string_view name,
                                              int64_t width);

  // Removes data input/output from the interface by name.
  absl::Status RemoveData(std::string_view name);

  // Add a single value channel to the interface.
  ModuleSignatureBuilder& AddSingleValueChannel(
      std::string_view name, ChannelOps supported_ops,
      const ChannelMetadataProto& metadata);

  // Add a streaming channel to the interface.
  ModuleSignatureBuilder& AddStreamingChannel(
      std::string_view name, ChannelOps supported_ops, FlowControl flow_control,
      Type* type, std::optional<FifoConfig> fifo_config,
      const ChannelMetadataProto& metadata);

  struct NameAndType {
    std::string_view name;
    Type* type;
  };
  struct FifoPorts {
    std::string_view push_ready_name;
    NameAndType push_data;
    std::string_view push_valid_name;
    std::string_view pop_ready_name;
    NameAndType pop_data;
    std::string_view pop_valid_name;
  };

  ModuleSignatureBuilder& AddFifoInstantiation(
      Package* package, std::string_view instance_name,
      std::optional<std::string_view> channel_name, const Type* data_type,
      FifoConfig fifo_config);

  // Remove a streaming channel from the interface.
  absl::Status RemoveChannel(std::string_view name);

  // Struct to emulate named arguments for AddRam1RW as there are a lot of
  // arguments with the same type.
  struct Ram1RWArgs {
    Package* package;
    Type* data_type;
    std::string_view ram_name;
    std::string_view req_name;
    std::string_view resp_name;
    int64_t address_width;
    int64_t read_mask_width;
    int64_t write_mask_width;
    std::string_view address_name;
    std::string_view read_enable_name;
    std::string_view write_enable_name;
    std::string_view read_data_name;
    std::string_view write_data_name;
    std::string_view write_mask_name;
    std::string_view read_mask_name;
  };
  ModuleSignatureBuilder& AddRam1RW(const Ram1RWArgs& args);

  // Struct to emulate named arguments for AddRam1R1W as there are a lot of
  // arguments with the same type.
  struct Ram1R1WArgs {
    Package* package;
    Type* data_type;
    std::string_view ram_name;
    std::string_view rd_req_name;
    std::string_view rd_resp_name;
    std::string_view wr_req_name;
    int64_t address_width;
    int64_t read_mask_width;
    int64_t write_mask_width;
    std::string_view read_address_name;
    std::string_view read_data_name;
    std::string_view read_mask_name;
    std::string_view read_enable_name;
    std::string_view write_address_name;
    std::string_view write_data_name;
    std::string_view write_mask_name;
    std::string_view write_enable_name;
  };
  ModuleSignatureBuilder& AddRam1R1W(const Ram1R1WArgs& args);

  absl::StatusOr<ModuleSignature> Build();

 private:
  explicit ModuleSignatureBuilder(ModuleSignatureProto&& proto)
      : proto_(std::move(proto)) {}

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
  std::string AsTextProto() const {
    std::string text_proto;
    google::protobuf::TextFormat::PrintToString(proto_, &text_proto);
    return text_proto;
  }

  // Returns the data inputs/outputs of module. This does not include clock,
  // reset, etc. These ports necessarily exist in the proto as well but are
  // duplicated here for convenience.
  absl::Span<const PortProto> data_inputs() const { return data_inputs_; }
  absl::Span<const PortProto> data_outputs() const { return data_outputs_; }

  // Returns the single value channels of the module.
  absl::Span<const ChannelProto> single_value_channels() const {
    return single_value_channels_;
  }

  // Returns the streaming channels of the module.
  absl::Span<const ChannelProto> streaming_channels() const {
    return streaming_channels_;
  }

  absl::Span<const RamProto> rams() { return rams_; }

  absl::Span<const InstantiationProto> instantiations() {
    return instantiations_;
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
  absl::Status ValidateChannelBitsInputs(std::string_view channel_name,
                                         absl::Span<const Bits> values) const;
  absl::Status ValidateChannelValueInputs(std::string_view channel_name,
                                          absl::Span<const Value> values) const;

  // Converts the ordered set of Value arguments to the module of the signature
  // into an argument name-value map.
  absl::StatusOr<absl::flat_hash_map<std::string, Value>> ToKwargs(
      absl::Span<const Value> inputs) const;

  absl::StatusOr<PortProto> GetInputPortProtoByName(
      std::string_view name) const;
  absl::StatusOr<PortProto> GetOutputPortProtoByName(
      std::string_view name) const;
  absl::StatusOr<ChannelProto> GetInputChannelProtoByName(
      std::string_view name) const;
  absl::StatusOr<ChannelProto> GetOutputChannelProtoByName(
      std::string_view name) const;
  absl::Span<const ChannelProto> GetInputChannels() const {
    return input_channels_;
  }
  absl::Span<const ChannelProto> GetOutputChannels() const {
    return output_channels_;
  }
  absl::StatusOr<std::string> GetChannelNameWith(
      std::string_view port_name) const;

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
  std::vector<ChannelProto> input_channels_;
  std::vector<ChannelProto> output_channels_;

  // Like the channels above, duplicate rams to enable a convenience method.
  std::vector<RamProto> rams_;

  // Duplicate instantiations to enable a convenience method.
  std::vector<InstantiationProto> instantiations_;

  // Map from port name to channel name.
  absl::flat_hash_map<std::string, std::string> port_name_to_channel_name;

  // TODO(vmirian): 10-28-2022 Support I/O channels and I/O ports.
  // Map from input/output port name to index in the input/output port vector.
  absl::flat_hash_map<std::string, int64_t> input_port_map_;
  absl::flat_hash_map<std::string, int64_t> output_port_map_;
  // Map from channel name to index in the channel input/output vector.
  absl::flat_hash_map<std::string, int64_t> input_channel_map_;
  absl::flat_hash_map<std::string, int64_t> output_channel_map_;
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
