// Copyright 2022 The XLS Authors
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

#include "xls/codegen/ram_rewrite_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

namespace {
absl::StatusOr<Channel*> GetStreamingChannel(Block* block,
                                             std::string_view channel_name) {
  XLS_ASSIGN_OR_RETURN(Channel * channel,
                       block->package()->GetChannel(channel_name));
  XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming)
      << absl::StreamFormat("Channel %s must be a streaming channel.",
                            channel_name);
  XLS_RET_CHECK(channel->GetDataPortName().has_value())
      << "data port not found- channel " << channel->name()
      << " should be streaming with ready/valid flow control.";
  XLS_RET_CHECK(channel->GetValidPortName().has_value())
      << "valid port not found- channel " << channel->name()
      << " should be streaming with ready/valid flow control.";
  XLS_RET_CHECK(channel->GetReadyPortName().has_value())
      << "ready port not found- channel " << channel->name()
      << " should be streaming with ready/valid flow control.";

  return channel;
}

struct ReqBlockPorts {
  OutputPort* req_data;
  OutputPort* req_valid;
  InputPort* req_ready;
};

struct RespBlockPorts {
  InputPort* resp_data;
  InputPort* resp_valid;
  OutputPort* resp_ready;
};

struct RamRWPortBlockPorts {
  ReqBlockPorts req_ports;
  RespBlockPorts resp_ports;
  RespBlockPorts write_completion_ports;
};

struct RamRPortBlockPorts {
  ReqBlockPorts req_ports;
  RespBlockPorts resp_ports;
};

struct RamWPortBlockPorts {
  ReqBlockPorts req_ports;
  RespBlockPorts write_completion_ports;
};

absl::Status CheckDataPortType(
    Type* tpe, std::string_view channel_name,
    absl::Span<const std::optional<TypeKind>> expected_types,
    absl::Span<const std::string_view> expected_names) {
  if (expected_types.size() != expected_names.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "expected_types size (%d) must match expected_names size (%d).",
        expected_types.size(), expected_names.size()));
  }
  int64_t num_tuple_elements = expected_types.size();

  // data should be a tuple with the length of expected_types.
  if (!tpe->IsTuple()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s must be a tuple type.", channel_name));
  }
  TupleType* tuple_tpe = tpe->AsTupleOrDie();
  if (tuple_tpe->size() != expected_types.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s must be a tuple type with %d elements, found %d.",
                        channel_name, num_tuple_elements, tuple_tpe->size()));
  }
  // Check each element of the tuple.
  for (int64_t element_idx = 0; element_idx < tuple_tpe->size();
       ++element_idx) {
    auto* element = tuple_tpe->element_type(element_idx);
    if (!expected_types[element_idx].has_value()) {
      if (TypeHasToken(element)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s element %s (idx=%d) must not contain token, got %s.",
            channel_name, expected_names[element_idx], element_idx,
            element->ToString()));
      }
      continue;
    }
    if (element->kind() != expected_types[element_idx].value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s element %s (idx=%d) must be type %s, got %s.", channel_name,
          expected_names[element_idx], element_idx,
          TypeKindToString(expected_types[element_idx].value()),
          element->ToString()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<ReqBlockPorts> GetReqBlockPorts(Block* block,
                                               std::string_view channel_name) {
  XLS_ASSIGN_OR_RETURN(Channel * req_channel,
                       GetStreamingChannel(block, channel_name));
  XLS_ASSIGN_OR_RETURN(
      auto* req_data_port,
      block->GetOutputPort(req_channel->GetDataPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* req_valid_port,
      block->GetOutputPort(req_channel->GetValidPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* req_ready_port,
      block->GetInputPort(req_channel->GetReadyPortName().value()));
  return ReqBlockPorts{.req_data = req_data_port,
                       .req_valid = req_valid_port,
                       .req_ready = req_ready_port};
}

absl::StatusOr<RespBlockPorts> GetRespBlockPorts(
    Block* block, std::string_view channel_name) {
  XLS_ASSIGN_OR_RETURN(Channel * resp_channel,
                       GetStreamingChannel(block, channel_name));
  XLS_ASSIGN_OR_RETURN(
      auto* resp_data_port,
      block->GetInputPort(resp_channel->GetDataPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* resp_valid_port,
      block->GetInputPort(resp_channel->GetValidPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* resp_ready_port,
      block->GetOutputPort(resp_channel->GetReadyPortName().value()));
  return RespBlockPorts{.resp_data = resp_data_port,
                        .resp_valid = resp_valid_port,
                        .resp_ready = resp_ready_port};
}

absl::StatusOr<RamRWPortBlockPorts> GetRWBlockPorts(
    Block* const block, const RamRWPortConfiguration& port_config) {
  RamRWPortBlockPorts ports;
  XLS_ASSIGN_OR_RETURN(
      ports.req_ports,
      GetReqBlockPorts(block, port_config.request_channel_name));
  XLS_ASSIGN_OR_RETURN(
      ports.resp_ports,
      GetRespBlockPorts(block, port_config.response_channel_name));
  XLS_ASSIGN_OR_RETURN(
      ports.write_completion_ports,
      GetRespBlockPorts(block, port_config.write_completion_channel_name));
  return ports;
}

absl::StatusOr<RamRPortBlockPorts> GetRBlockPorts(
    Block* const block, const RamRPortConfiguration& port_config) {
  RamRPortBlockPorts ports;
  XLS_ASSIGN_OR_RETURN(
      ports.req_ports,
      GetReqBlockPorts(block, port_config.request_channel_name));
  XLS_ASSIGN_OR_RETURN(
      ports.resp_ports,
      GetRespBlockPorts(block, port_config.response_channel_name));
  return ports;
}

absl::StatusOr<RamWPortBlockPorts> GetWBlockPorts(
    Block* const block, const RamWPortConfiguration& port_config) {
  RamWPortBlockPorts ports;
  XLS_ASSIGN_OR_RETURN(
      ports.req_ports,
      GetReqBlockPorts(block, port_config.request_channel_name));
  XLS_ASSIGN_OR_RETURN(
      ports.write_completion_ports,
      GetRespBlockPorts(block, port_config.write_completion_channel_name));
  return ports;
}

// When we rewrite Ram ports, we invalidate the metadata in StreamingIOPipeline.
// To avoid having dangling pointers, we remove the StreamingInputs and
// StreamingOutputs associated with Ram channels.
// TODO: github/xls#1300 - remove when metadata is refactored.
void ClearRewrittenMetadata(
    StreamingIOPipeline& streaming_io,
    absl::flat_hash_set<std::string_view> channel_names) {
  for (std::vector<StreamingInput>& inputs : streaming_io.inputs) {
    inputs.erase(
        std::remove_if(inputs.begin(), inputs.end(),
                       [&channel_names](const StreamingInput& input) {
                         return channel_names.contains(input.channel->name());
                       }),
        inputs.end());
  }
  for (std::vector<StreamingOutput>& outputs : streaming_io.outputs) {
    outputs.erase(
        std::remove_if(outputs.begin(), outputs.end(),
                       [&channel_names](const StreamingOutput& output) {
                         return channel_names.contains(output.channel->name());
                       }),
        outputs.end());
  }
}

// The write completion channel exists to model the behavior of a write
// completion and does not actually drive any bits. It is useful as a touchpoint
// for scheduling constraints.
// The ready signal is unneeded, the valid signal should be replaced with a true
// literal, and the data usages should be replaced with empty tuple literals.
absl::Status WriteCompletionRewrite(
    Block* block, const RespBlockPorts& write_completion_ports,
    std::string_view ram_name) {
  // Data should be an empty tuple for write completion.
  XLS_RETURN_IF_ERROR(CheckDataPortType(
      write_completion_ports.resp_data->GetType(), "Write completion", {}, {}));

  std::string data_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_write_completion_data"));

  XLS_ASSIGN_OR_RETURN(xls::Literal * empty_tuple,
                       block->MakeNodeWithName<xls::Literal>(
                           write_completion_ports.resp_data->loc(),
                           Value::Tuple({}), data_name));
  XLS_RETURN_IF_ERROR(
      write_completion_ports.resp_data->ReplaceUsesWith(empty_tuple));

  XLS_ASSIGN_OR_RETURN(
      xls::Literal * true_lit,
      block->MakeNode<xls::Literal>(write_completion_ports.resp_valid->loc(),
                                    Value(UBits(1, 1))));
  XLS_RETURN_IF_ERROR(
      write_completion_ports.resp_valid->ReplaceUsesWith(true_lit));

  XLS_RETURN_IF_ERROR(block->RemoveNode(write_completion_ports.resp_data));
  XLS_RETURN_IF_ERROR(block->RemoveNode(write_completion_ports.resp_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(write_completion_ports.resp_ready));

  return absl::OkStatus();
}

absl::Status Ram1RWUpdateSignature(
    ModuleSignature& signature, const Ram1RWConfiguration& ram_config,
    std::string_view ram_name, Package* package, OutputPort* req_addr_port,
    OutputPort* req_re_port, OutputPort* req_we_port,
    OutputPort* req_wr_data_port, OutputPort* req_wr_mask_port,
    OutputPort* req_rd_mask_port, InputPort* resp_rd_data_port) {
  auto builder = ModuleSignatureBuilder::FromProto(signature.proto());
  XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(
      ram_config.rw_port_configuration().request_channel_name));
  XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(
      ram_config.rw_port_configuration().response_channel_name));
  XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(
      ram_config.rw_port_configuration().write_completion_channel_name));

  for (std::string_view channel_name : {
           ram_config.rw_port_configuration().request_channel_name,
           ram_config.rw_port_configuration().response_channel_name,
           ram_config.rw_port_configuration().write_completion_channel_name,
       }) {
    XLS_ASSIGN_OR_RETURN(Channel * channel, package->GetChannel(channel_name));
    if (channel->GetReadyPortName().has_value()) {
      XLS_RETURN_IF_ERROR(
          builder.RemoveData(channel->GetReadyPortName().value()));
    }
    if (channel->GetDataPortName().has_value()) {
      XLS_RETURN_IF_ERROR(
          builder.RemoveData(channel->GetDataPortName().value()));
    }
    if (channel->GetValidPortName().has_value()) {
      XLS_RETURN_IF_ERROR(
          builder.RemoveData(channel->GetValidPortName().value()));
    }
  }

  for (const OutputPort* port : {
           req_addr_port,
           req_re_port,
           req_we_port,
           req_wr_data_port,
           req_wr_mask_port,
           req_rd_mask_port,
       }) {
    if (port->operand(0)->GetType()->GetFlatBitCount() > 0) {
      builder.AddDataOutput(port->name(), port->operand(0)->GetType());
    }
  }
  for (const xls::InputPort* port : {resp_rd_data_port}) {
    if (resp_rd_data_port->GetType()->GetFlatBitCount() > 0) {
      builder.AddDataInput(port->name(), port->GetType());
    }
  }

  builder.AddRam1RW({
      .package = package,
      .data_type = req_wr_data_port->GetType(),
      .ram_name = ram_name,
      .req_name = ram_config.rw_port_configuration().request_channel_name,
      .resp_name = ram_config.rw_port_configuration().response_channel_name,
      .address_width = req_addr_port->GetType()->GetFlatBitCount(),
      .read_mask_width = req_rd_mask_port->GetType()->GetFlatBitCount(),
      .write_mask_width = req_wr_mask_port->GetType()->GetFlatBitCount(),
      .address_name = req_addr_port->GetName(),
      .read_enable_name = req_re_port->GetName(),
      .write_enable_name = req_we_port->GetName(),
      .read_data_name = resp_rd_data_port->GetName(),
      .write_data_name = req_wr_data_port->GetName(),
      .write_mask_name = req_wr_mask_port->GetName(),
      .read_mask_name = req_rd_mask_port->GetName(),
  });

  XLS_ASSIGN_OR_RETURN(signature, builder.Build());
  return absl::OkStatus();
}

absl::StatusOr<bool> Ram1RWRewrite(
    CodegenPassUnit* unit, const CodegenPassOptions& pass_options,
    const RamConfiguration& base_ram_configuration) {
  auto& ram_config =
      down_cast<const Ram1RWConfiguration&>(base_ram_configuration);
  Block* block = unit->top_block;

  XLS_ASSIGN_OR_RETURN(
      RamRWPortBlockPorts rw_block_ports,
      GetRWBlockPorts(block, ram_config.rw_port_configuration()));

  // req is (addr, wr_data, we, re)
  XLS_RETURN_IF_ERROR(CheckDataPortType(
      rw_block_ports.req_ports.req_data->operand(0)->GetType(), "Request",
      {TypeKind::kBits, std::nullopt, std::nullopt, std::nullopt,
       TypeKind::kBits, TypeKind::kBits},
      {"addr", "wr_data", "wr_mask", "rd_mask", "we", "re"}));
  // resp is (rd_data)
  XLS_RETURN_IF_ERROR(
      CheckDataPortType(rw_block_ports.resp_ports.resp_data->GetType(),
                        "Response", {std::nullopt}, {"rd_data"}));

  auto tuple_index = [block](Node* node, int idx) {
    return block->MakeNode<TupleIndex>(
        /*loc=*/SourceInfo(), node, /*index=*/idx);
  };

  // Peel off each field from the data port's operand.
  VLOG(3) << "req_data_port op = "
          << rw_block_ports.req_ports.req_data->operand(0)
                 ->ToStringWithOperandTypes();
  XLS_ASSIGN_OR_RETURN(
      Node * req_addr,
      tuple_index(rw_block_ports.req_ports.req_data->operand(0), 0));
  XLS_ASSIGN_OR_RETURN(
      Node * req_wr_data,
      tuple_index(rw_block_ports.req_ports.req_data->operand(0), 1));
  XLS_ASSIGN_OR_RETURN(
      Node * req_wr_mask,
      tuple_index(rw_block_ports.req_ports.req_data->operand(0), 2));
  XLS_ASSIGN_OR_RETURN(
      Node * req_rd_mask,
      tuple_index(rw_block_ports.req_ports.req_data->operand(0), 3));
  XLS_ASSIGN_OR_RETURN(
      Node * req_we,
      tuple_index(rw_block_ports.req_ports.req_data->operand(0), 4));
  XLS_ASSIGN_OR_RETURN(
      Node * req_re,
      tuple_index(rw_block_ports.req_ports.req_data->operand(0), 5));

  // Earlier, we checked that we and re have type bits. Now, also check that
  // they have width=1.
  if (req_we->GetType()->GetFlatBitCount() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Request element we (idx=2) must be type bits[1], got %s.",
        req_we->GetType()->ToString()));
  }
  if (req_re->GetType()->GetFlatBitCount() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Request element re (idx=3) must be type bits[1], got %s.",
        req_re->GetType()->ToString()));
  }

  Node* req_valid = rw_block_ports.req_ports.req_valid->operand(0);

  // Make names for each element of the request tuple. They will end up each
  // having their own port.
  std::string_view ram_name = ram_config.ram_name();
  std::string req_addr_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_addr"));
  std::string req_wr_data_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_wr_data"));
  std::string req_wr_mask_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_wr_mask"));
  std::string req_rd_mask_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_rd_mask"));
  std::string req_we_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_we"));
  std::string req_re_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_re"));

  // we is asserted when req.we is asserted and when the request is valid.
  XLS_ASSIGN_OR_RETURN(
      Node * req_we_valid,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(), std::vector<Node*>({req_we, req_valid}),
          Op::kAnd, req_we_name));
  // re is asserted when req.re is asserted and when the request is valid.
  XLS_ASSIGN_OR_RETURN(
      Node * req_re_valid,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(), std::vector<Node*>({req_re, req_valid}),
          Op::kAnd, req_re_name));

  std::string req_re_valid_buf_name = block->UniquifyNodeName(
      absl::StrFormat("__%s_buffer", req_re_valid->GetName()));
  XLS_ASSIGN_OR_RETURN(Node * req_re_valid_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), req_re_valid, Op::kIdentity,
                           req_re_valid_buf_name));

  std::optional<xls::Reset> reset_behavior =
      pass_options.codegen_options.ResetBehavior();

  XLS_ASSIGN_OR_RETURN(
      Node * ram_resp_valid,
      AddRegisterAfterNode(absl::StrCat(req_valid->GetName(), "_delay"),
                           reset_behavior, std::nullopt, req_re_valid_buf,
                           block));

  // Make a new response port with a new name.
  std::string resp_rd_data_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_rd_data"));
  XLS_ASSIGN_OR_RETURN(
      InputPort * resp_rd_data_port,
      block->AddInputPort(resp_rd_data_name,
                          rw_block_ports.resp_ports.resp_data->GetType()));
  XLS_RETURN_IF_ERROR(
      rw_block_ports.resp_ports.resp_data->ReplaceUsesWith(resp_rd_data_port));

  // Add buffer before resp_ready
  std::string resp_ready_port_buf_name = absl::StrFormat(
      "__%s_buffer", rw_block_ports.resp_ports.resp_ready->name());
  XLS_ASSIGN_OR_RETURN(Node * resp_ready_port_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(),
                           rw_block_ports.resp_ports.resp_ready->operand(0),
                           Op::kIdentity, resp_ready_port_buf_name));

  // Update channel ready/valid ports usages with new internal signals.
  absl::flat_hash_map<Node*, Node*> replaced_nodes = {
      {rw_block_ports.resp_ports.resp_data, resp_rd_data_port},
      {rw_block_ports.resp_ports.resp_ready, resp_ready_port_buf},
      {rw_block_ports.resp_ports.resp_ready->operand(0), resp_ready_port_buf},
      {rw_block_ports.resp_ports.resp_valid, ram_resp_valid},
      {rw_block_ports.req_ports.req_ready, resp_ready_port_buf},
  };
  for (auto& [_, metadata] : unit->metadata) {
    for (Node*& node :
         metadata.streaming_io_and_pipeline.all_active_outputs_ready) {
      auto itr = replaced_nodes.find(node);
      if (itr != replaced_nodes.end()) {
        node = itr->second;
      }
    }
  }
  XLS_RETURN_IF_ERROR(
      rw_block_ports.resp_ports.resp_ready->ReplaceOperandNumber(
          0, resp_ready_port_buf));
  XLS_RETURN_IF_ERROR(
      rw_block_ports.resp_ports.resp_valid->ReplaceUsesWith(ram_resp_valid));
  XLS_RETURN_IF_ERROR(
      rw_block_ports.req_ports.req_ready->ReplaceUsesWith(resp_ready_port_buf));

  // Add zero-latency buffer at output of ram
  std::vector<std::optional<Node*>> valid_nodes;
  std::string zero_latency_buffer_name =
      absl::StrCat(ram_name, "_ram_zero_latency0");
  XLS_RETURN_IF_ERROR(AddZeroLatencyBufferToRDVNodes(
                          resp_rd_data_port, ram_resp_valid,
                          resp_ready_port_buf, zero_latency_buffer_name,
                          reset_behavior, block, valid_nodes)
                          .status());

  // Add output ports for expanded req.data.
  XLS_ASSIGN_OR_RETURN(auto* req_addr_port,
                       block->AddOutputPort(req_addr_name, req_addr));
  XLS_ASSIGN_OR_RETURN(auto* req_wr_data_port,
                       block->AddOutputPort(req_wr_data_name, req_wr_data));
  XLS_ASSIGN_OR_RETURN(auto* req_wr_mask_port,
                       block->AddOutputPort(req_wr_mask_name, req_wr_mask));
  XLS_ASSIGN_OR_RETURN(auto* req_rd_mask_port,
                       block->AddOutputPort(req_rd_mask_name, req_rd_mask));
  XLS_ASSIGN_OR_RETURN(auto* req_we_port,
                       block->AddOutputPort(req_we_name, req_we_valid));
  XLS_ASSIGN_OR_RETURN(auto* req_re_port,
                       block->AddOutputPort(req_re_name, req_re_valid));

  // Remove ports that have been replaced.
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.req_ports.req_data));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.req_ports.req_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.req_ports.req_ready));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.resp_ports.resp_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.resp_ports.resp_ready));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.resp_ports.resp_data));

  XLS_RETURN_IF_ERROR(WriteCompletionRewrite(
      block, rw_block_ports.write_completion_ports, ram_name));

  auto metadata_itr = unit->metadata.find(unit->top_block);
  if (metadata_itr != unit->metadata.end()) {
    ClearRewrittenMetadata(
        metadata_itr->second.streaming_io_and_pipeline,
        {ram_config.rw_port_configuration().request_channel_name,
         ram_config.rw_port_configuration().response_channel_name,
         ram_config.rw_port_configuration().write_completion_channel_name});

    if (metadata_itr->second.signature.has_value()) {
      ModuleSignature& signature = *metadata_itr->second.signature;
      XLS_RETURN_IF_ERROR(
          Ram1RWUpdateSignature(signature, ram_config, ram_name, unit->package,
                                /*req_addr_port=*/req_addr_port,
                                /*req_re_port=*/req_re_port,
                                /*req_we_port=*/req_we_port,
                                /*req_wr_data_port=*/req_wr_data_port,
                                /*req_wr_mask_port=*/req_wr_mask_port,
                                /*req_rd_mask_port=*/req_rd_mask_port,
                                /*resp_rd_data_port=*/resp_rd_data_port));
    }
  }

  return true;
}

absl::StatusOr<bool> Ram1R1WRewrite(
    CodegenPassUnit* unit, const CodegenPassOptions& pass_options,
    const RamConfiguration& base_ram_configuration) {
  auto& ram_config =
      down_cast<const Ram1R1WConfiguration&>(base_ram_configuration);
  Block* block = unit->top_block;

  XLS_ASSIGN_OR_RETURN(
      RamRPortBlockPorts r_block_ports,
      GetRBlockPorts(block, ram_config.r_port_configuration()));
  XLS_ASSIGN_OR_RETURN(
      RamWPortBlockPorts w_block_ports,
      GetWBlockPorts(block, ram_config.w_port_configuration()));

  // rd_req is (rd_addr, rd_mask)
  XLS_RETURN_IF_ERROR(CheckDataPortType(
      r_block_ports.req_ports.req_data->operand(0)->GetType(), "rd_req",
      {TypeKind::kBits, std::nullopt}, {"rd_addr", "rd_mask"}));
  // rd_resp is (rd_data)
  XLS_RETURN_IF_ERROR(
      CheckDataPortType(r_block_ports.resp_ports.resp_data->GetType(),
                        "rd_resp", {std::nullopt}, {"rd_data"}));
  // wr_req is (wr_addr, wr_data, wr_mask)
  XLS_RETURN_IF_ERROR(
      CheckDataPortType(w_block_ports.req_ports.req_data->operand(0)->GetType(),
                        "wr_req", {TypeKind::kBits, std::nullopt, std::nullopt},
                        {"wr_addr", "wr_data", "wr_mask"}));

  auto tuple_index = [block](Node* node, int idx) {
    return block->MakeNode<TupleIndex>(
        /*loc=*/SourceInfo(), node, /*index=*/idx);
  };

  // Peel off fields from the data ports' operands.
  XLS_ASSIGN_OR_RETURN(
      Node * rd_addr,
      tuple_index(r_block_ports.req_ports.req_data->operand(0), 0));
  XLS_ASSIGN_OR_RETURN(
      Node * rd_mask,
      tuple_index(r_block_ports.req_ports.req_data->operand(0), 1));
  XLS_ASSIGN_OR_RETURN(Node * rd_data,
                       tuple_index(r_block_ports.resp_ports.resp_data, 0));
  XLS_ASSIGN_OR_RETURN(
      Node * wr_addr,
      tuple_index(w_block_ports.req_ports.req_data->operand(0), 0));
  XLS_ASSIGN_OR_RETURN(
      Node * wr_data,
      tuple_index(w_block_ports.req_ports.req_data->operand(0), 1));
  XLS_ASSIGN_OR_RETURN(
      Node * wr_mask,
      tuple_index(w_block_ports.req_ports.req_data->operand(0), 2));

  Node* rd_en = r_block_ports.req_ports.req_valid->operand(0);
  Node* wr_en = w_block_ports.req_ports.req_valid->operand(0);

  // Make names for each element of the request tuple. They will end up each
  // having their own port.
  std::string_view ram_name = ram_config.ram_name();
  std::string rd_addr_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_rd_addr"));
  std::string rd_mask_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_rd_mask"));
  std::string rd_en_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_rd_en"));
  std::string wr_addr_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_wr_addr"));
  std::string wr_data_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_wr_data"));
  std::string wr_mask_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_wr_mask"));
  std::string wr_en_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_wr_en"));

  wr_en->SetName(wr_en_name);
  rd_en->SetName(rd_en_name);

  std::string req_re_valid_buf_name =
      block->UniquifyNodeName(absl::StrFormat("__%s_buffer", rd_en->GetName()));
  XLS_ASSIGN_OR_RETURN(
      Node * req_re_valid_buf,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), rd_en, Op::kIdentity, req_re_valid_buf_name));

  std::optional<xls::Reset> reset_behavior =
      pass_options.codegen_options.ResetBehavior();

  XLS_ASSIGN_OR_RETURN(
      Node * rd_resp_valid,
      AddRegisterAfterNode(absl::StrCat(rd_en->GetName(), "_delay"),
                           reset_behavior, std::nullopt, req_re_valid_buf,
                           block));

  // Make a new response port with a new name.
  std::string rd_data_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_rd_data"));
  XLS_ASSIGN_OR_RETURN(InputPort * rd_data_port,
                       block->AddInputPort(rd_data_name, rd_data->GetType()));
  XLS_ASSIGN_OR_RETURN(
      Tuple * rd_data_tuple,
      block->MakeNode<Tuple>(/*loc=*/SourceInfo(),
                             std::vector<Node*>{rd_data_port}));
  XLS_RETURN_IF_ERROR(
      r_block_ports.resp_ports.resp_data->ReplaceUsesWith(rd_data_tuple));

  // Add buffer before resp_ready
  std::string resp_ready_port_buf_name = absl::StrFormat(
      "__%s_buffer", r_block_ports.resp_ports.resp_ready->name());
  XLS_ASSIGN_OR_RETURN(
      Node * resp_ready_port_buf,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), r_block_ports.resp_ports.resp_ready->operand(0),
          Op::kIdentity, resp_ready_port_buf_name));

  // Update channel ready/valid ports usages with new internal signals.
  XLS_RETURN_IF_ERROR(r_block_ports.resp_ports.resp_ready->ReplaceOperandNumber(
      0, resp_ready_port_buf));
  XLS_RETURN_IF_ERROR(
      r_block_ports.resp_ports.resp_valid->ReplaceUsesWith(rd_resp_valid));
  XLS_RETURN_IF_ERROR(
      r_block_ports.req_ports.req_ready->ReplaceUsesWith(resp_ready_port_buf));

  // Add zero-latency buffer at output of ram
  std::vector<std::optional<Node*>> valid_nodes;
  std::string zero_latency_buffer_name =
      absl::StrCat(ram_name, "_ram_zero_latency0");
  XLS_RETURN_IF_ERROR(AddZeroLatencyBufferToRDVNodes(
                          rd_data_port, rd_resp_valid, resp_ready_port_buf,
                          zero_latency_buffer_name, reset_behavior, block,
                          valid_nodes)
                          .status());
  // Replace write ready with literal 1 (RAM is always ready for write).
  // TODO(rigge): should this signal check for hazards?
  XLS_ASSIGN_OR_RETURN(
      xls::Literal * literal_1,
      block->MakeNode<xls::Literal>(/*loc=*/SourceInfo(), Value(UBits(1, 1))));
  XLS_RETURN_IF_ERROR(
      w_block_ports.req_ports.req_ready->ReplaceUsesWith(literal_1));

  // Add output ports for expanded req.data.
  XLS_ASSIGN_OR_RETURN(auto* rd_addr_port,
                       block->AddOutputPort(rd_addr_name, rd_addr));
  XLS_ASSIGN_OR_RETURN(auto* rd_mask_port,
                       block->AddOutputPort(rd_mask_name, rd_mask));
  XLS_ASSIGN_OR_RETURN(auto* rd_en_port,
                       block->AddOutputPort(rd_en_name, rd_en));
  XLS_ASSIGN_OR_RETURN(auto* wr_addr_port,
                       block->AddOutputPort(wr_addr_name, wr_addr));
  XLS_ASSIGN_OR_RETURN(auto* wr_data_port,
                       block->AddOutputPort(wr_data_name, wr_data));
  XLS_ASSIGN_OR_RETURN(auto* wr_mask_port,
                       block->AddOutputPort(wr_mask_name, wr_mask));
  XLS_ASSIGN_OR_RETURN(auto* wr_en_port,
                       block->AddOutputPort(wr_en_name, wr_en));

  // Remove ports that have been replaced.
  XLS_RETURN_IF_ERROR(block->RemoveNode(r_block_ports.req_ports.req_data));
  XLS_RETURN_IF_ERROR(block->RemoveNode(r_block_ports.req_ports.req_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(r_block_ports.req_ports.req_ready));
  XLS_RETURN_IF_ERROR(block->RemoveNode(r_block_ports.resp_ports.resp_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(r_block_ports.resp_ports.resp_ready));
  XLS_RETURN_IF_ERROR(block->RemoveNode(r_block_ports.resp_ports.resp_data));
  XLS_RETURN_IF_ERROR(block->RemoveNode(w_block_ports.req_ports.req_data));
  XLS_RETURN_IF_ERROR(block->RemoveNode(w_block_ports.req_ports.req_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(w_block_ports.req_ports.req_ready));

  XLS_RETURN_IF_ERROR(WriteCompletionRewrite(
      block, w_block_ports.write_completion_ports, ram_name));

  auto metadata_itr = unit->metadata.find(unit->top_block);
  if (metadata_itr != unit->metadata.end()) {
    if (metadata_itr->second.signature.has_value()) {
      ClearRewrittenMetadata(
          metadata_itr->second.streaming_io_and_pipeline,
          {
              ram_config.r_port_configuration().request_channel_name,
              ram_config.r_port_configuration().response_channel_name,
              ram_config.w_port_configuration().request_channel_name,
              ram_config.w_port_configuration().write_completion_channel_name,
          });
      ModuleSignature& signature = metadata_itr->second.signature.value();
      auto builder = ModuleSignatureBuilder::FromProto(signature.proto());
      XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(
          ram_config.r_port_configuration().request_channel_name));
      XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(
          ram_config.r_port_configuration().response_channel_name));
      XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(
          ram_config.w_port_configuration().request_channel_name));
      XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(
          ram_config.w_port_configuration().write_completion_channel_name));

      for (std::string_view channel_name : {
               ram_config.r_port_configuration().request_channel_name,
               ram_config.r_port_configuration().response_channel_name,
               ram_config.w_port_configuration().request_channel_name,
               ram_config.w_port_configuration().write_completion_channel_name,
           }) {
        XLS_ASSIGN_OR_RETURN(Channel * channel,
                             block->package()->GetChannel(channel_name));
        if (channel->GetReadyPortName().has_value()) {
          XLS_RETURN_IF_ERROR(
              builder.RemoveData(channel->GetReadyPortName().value()));
        }
        if (channel->GetDataPortName().has_value()) {
          XLS_RETURN_IF_ERROR(
              builder.RemoveData(channel->GetDataPortName().value()));
        }
        if (channel->GetValidPortName().has_value()) {
          XLS_RETURN_IF_ERROR(
              builder.RemoveData(channel->GetValidPortName().value()));
        }
      }
      for (const xls::OutputPort* port : {
               rd_addr_port,
               rd_mask_port,
               rd_en_port,
               wr_addr_port,
               wr_data_port,
               wr_mask_port,
               wr_en_port,
           }) {
        if (port->operand(0)->GetType()->GetFlatBitCount() > 0) {
          builder.AddDataOutput(port->name(), port->operand(0)->GetType());
        }
      }
      for (const xls::InputPort* port : {rd_data_port}) {
        if (port->GetType()->GetFlatBitCount() > 0) {
          builder.AddDataInput(port->name(), port->GetType());
        }
      }

      builder.AddRam1R1W({
          .package = unit->package,
          .data_type = rd_data->GetType(),
          .ram_name = ram_name,
          .rd_req_name = ram_config.r_port_configuration().request_channel_name,
          .rd_resp_name =
              ram_config.r_port_configuration().response_channel_name,
          .wr_req_name = ram_config.w_port_configuration().request_channel_name,
          .address_width = rd_addr_port->GetType()->GetFlatBitCount(),
          .read_mask_width = rd_mask_port->GetType()->GetFlatBitCount(),
          .write_mask_width = wr_mask_port->GetType()->GetFlatBitCount(),
          .read_address_name = rd_addr_port->GetName(),
          .read_data_name = rd_data_port->GetName(),
          .read_mask_name = rd_mask_port->GetName(),
          .read_enable_name = rd_en_port->GetName(),
          .write_address_name = wr_addr_port->GetName(),
          .write_data_name = wr_data_port->GetName(),
          .write_mask_name = wr_mask_port->GetName(),
          .write_enable_name = wr_en_port->GetName(),
      });

      XLS_ASSIGN_OR_RETURN(signature, builder.Build());
    }
  }

  return true;
}

absl::flat_hash_map<std::string, ram_rewrite_function_t>*
GetRamRewriteFunctionMap() {
  static auto* singleton =
      new absl::flat_hash_map<std::string, ram_rewrite_function_t>{
          {"1RW", Ram1RWRewrite},
          {"1R1W", Ram1R1WRewrite},
      };
  return singleton;
}

absl::StatusOr<bool> RamRewrite(CodegenPassUnit* unit,
                                const CodegenPassOptions& pass_options,
                                const RamConfiguration& ram_configuration) {
  absl::flat_hash_map<std::string, ram_rewrite_function_t>*
      rewrite_function_map = GetRamRewriteFunctionMap();
  auto itr = rewrite_function_map->find(ram_configuration.ram_kind());
  if (itr == rewrite_function_map->end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No RAM rewriter found for RAM kind %s.",
                        ram_configuration.ram_kind()));
  }
  return itr->second(unit, pass_options, ram_configuration);
}

}  // namespace

// This function finds all the ports related to each channel specified in
// `codegen_options.ram_channels()` and invokes `RewriteRamChannel()` on them.
absl::StatusOr<bool> RamRewritePass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  bool changed = false;

  XLS_RET_CHECK_NE(unit->top_block, nullptr)
      << "RamRewritePass requires top_block to be set.";

  for (const std::unique_ptr<RamConfiguration>& ram_configuration :
       options.codegen_options.ram_configurations()) {
    VLOG(2) << "Rewriting channels for ram " << ram_configuration->ram_name()
            << ".";
    XLS_ASSIGN_OR_RETURN(bool this_one_changed,
                         RamRewrite(unit, options, *ram_configuration));
    changed = changed || this_one_changed;
  }

  if (changed) {
    unit->GcMetadata();
  }

  return changed;
}

}  // namespace xls::verilog
