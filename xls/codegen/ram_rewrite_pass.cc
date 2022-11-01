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

#include <stdint.h>

#include <array>
#include <optional>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/codegen/block_conversion.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/block.h"
#include "xls/ir/nodes.h"

namespace xls::verilog {

namespace {
absl::StatusOr<Channel*> GetStreamingChannel(Block* block,
                                             std::string_view channel_name) {
  XLS_ASSIGN_OR_RETURN(Channel * channel,
                       block->package()->GetChannel(channel_name));
  XLS_RET_CHECK(channel->GetDataPortName().has_value());
  XLS_RET_CHECK(channel->GetValidPortName().has_value())
      << "valid port not found- channels should be streaming with "
         "ready/valid flow control.";
  XLS_RET_CHECK(channel->GetReadyPortName().has_value())
      << "ready port not found- channels should be streaming with "
         "ready/valid flow control.";

  return channel;
}

constexpr std::array<std::string_view, 4> kRwRequestTupleElementNames = {
    "addr", "wr_data", "we", "re"};

struct Ram1RWPortBlockPorts {
  OutputPort* req_data;
  OutputPort* req_valid;
  InputPort* req_ready;
  InputPort* resp_data;
  InputPort* resp_valid;
  OutputPort* resp_ready;
};

absl::StatusOr<Ram1RWPortBlockPorts> GetRWBlockPorts(
    Block* const block, const RamRWPortConfiguration& port_config) {
  XLS_ASSIGN_OR_RETURN(
      Channel * req_channel,
      GetStreamingChannel(block, port_config.request_channel_name));
  XLS_ASSIGN_OR_RETURN(
      Channel * resp_channel,
      GetStreamingChannel(block, port_config.response_channel_name));

  XLS_ASSIGN_OR_RETURN(
      auto* req_data_port,
      block->GetOutputPort(req_channel->GetDataPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* req_valid_port,
      block->GetOutputPort(req_channel->GetValidPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* req_ready_port,
      block->GetInputPort(req_channel->GetReadyPortName().value()));

  XLS_ASSIGN_OR_RETURN(
      auto* resp_data_port,
      block->GetInputPort(resp_channel->GetDataPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* resp_valid_port,
      block->GetInputPort(resp_channel->GetValidPortName().value()));
  XLS_ASSIGN_OR_RETURN(
      auto* resp_ready_port,
      block->GetOutputPort(resp_channel->GetReadyPortName().value()));

  // req_data should have a tuple type, where tuple elements are (addr, wr_data,
  // we, re)
  auto* req_tpe = req_data_port->operand(0)->GetType();
  if (!req_tpe->IsTuple()) {
    return absl::InvalidArgumentError("Request must be a tuple type.");
  }
  auto* req_tuple_tpe = req_tpe->AsTupleOrDie();
  if (req_tuple_tpe->size() != 4) {  // (addr, data, we, re)
    return absl::InvalidArgumentError(absl::StrFormat(
        "Request must be a tuple type with 4 elements, found %d.",
        req_tuple_tpe->size()));
  }
  // Each element must be of type bits
  for (int element_idx = 0; element_idx < req_tuple_tpe->element_types().size();
       ++element_idx) {
    auto* element = req_tuple_tpe->element_type(element_idx);
    if (!element->IsBits()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Request %s element must be type bits, got %s.",
          kRwRequestTupleElementNames[element_idx], element->ToString()));
    }
  }

  int64_t data_width = req_tuple_tpe->element_type(1)->GetFlatBitCount();
  // we and re must each be one bit wide
  if (req_tuple_tpe->element_type(2)->GetFlatBitCount() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Request we element must have width 1, got %d.",
                        req_tuple_tpe->element_type(2)->GetFlatBitCount()));
  }
  if (req_tuple_tpe->element_type(3)->GetFlatBitCount() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Request re element must have width 1, got %d.",
                        req_tuple_tpe->element_type(3)->GetFlatBitCount()));
  }

  // resp_data should be a single-element tuple consisting of (rd_data).
  auto* resp_tpe = resp_data_port->GetType();
  if (!resp_tpe->IsTuple()) {
    return absl::InvalidArgumentError("Response must be a tuple type.");
  }
  auto* resp_tuple_tpe = resp_tpe->AsTupleOrDie();
  if (resp_tuple_tpe->size() != 1) {  // (rd_data,)
    return absl::InvalidArgumentError(absl::StrFormat(
        "Response must be a tuple type with 1 element, found %d.",
        resp_tuple_tpe->size()));
  }
  if (!resp_tuple_tpe->element_type(0)->IsBits()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Response rd_data element must be type bits, got %s.",
                        resp_tuple_tpe->element_type(0)->ToString()));
  }
  if (resp_tuple_tpe->element_type(0)->GetFlatBitCount() != data_width) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Response rd_data element (width=%d) must have the same width as "
        "request wr_data element (width=%d)",
        resp_tuple_tpe->element_type(0)->GetFlatBitCount(), data_width));
  }
  return Ram1RWPortBlockPorts{
      .req_data = req_data_port,
      .req_valid = req_valid_port,
      .req_ready = req_ready_port,
      .resp_data = resp_data_port,
      .resp_valid = resp_valid_port,
      .resp_ready = resp_ready_port,
  };
}

// After modifying request/response channels to drive a RAM, the channels no
// longer fit neatly into how ModuleSignature represents channels. Instead,
// there is a separate place in the signature that describes RAM ports. This
// function removes channels from the "streaming channel" list and adds them,
// along with other metadata, to the ram list.
absl::Status UpdateModuleSignatureChannelsToRams(
    ModuleSignature* signature, std::string_view ram_name,
    std::string_view req_name, std::string_view resp_name, OutputPort* address,
    OutputPort* write_data, OutputPort* read_enable, OutputPort* write_enable,
    InputPort* read_data) {
  auto builder = ModuleSignatureBuilder::FromProto(signature->proto());
  XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(req_name));
  XLS_RETURN_IF_ERROR(builder.RemoveStreamingChannel(resp_name));
  builder.AddRamRWPort(
      /*name=*/ram_name,
      /*req_name=*/req_name, /*resp_name=*/resp_name,
      /*address_width=*/address->GetType()->GetFlatBitCount(),
      /*data_width=*/write_data->GetType()->GetFlatBitCount(),
      /*address_name=*/address->GetName(),
      /*read_enable_name=*/read_enable->GetName(),
      /*write_enable_name=*/write_enable->GetName(),
      /*read_data_name=*/read_data->GetName(),
      /*write_data_name=*/write_data->GetName());

  XLS_ASSIGN_OR_RETURN(*signature, builder.Build());
  return absl::OkStatus();
}

absl::StatusOr<bool> Ram1RWRewrite(
    CodegenPassUnit* unit, const CodegenPassOptions& pass_options,
    const RamConfiguration& base_ram_configuration) {
  auto& ram_config =
      down_cast<const Ram1RWConfiguration&>(base_ram_configuration);
  XLS_VLOG(2) << "Rewriting channels for ram " << ram_config.ram_name() << ".";
  Block* block = unit->block;

  XLS_ASSIGN_OR_RETURN(
      Ram1RWPortBlockPorts rw_block_ports,
      GetRWBlockPorts(block, ram_config.rw_port_configuration()));
  auto tuple_index = [block](Node* node, int idx) {
    return block->MakeNode<TupleIndex>(
        /*loc=*/SourceInfo(), node, /*index=*/idx);
  };

  // Peel off each field from the data port's operand.
  XLS_VLOG(3)
      << "req_data_port op = "
      << rw_block_ports.req_data->operand(0)->ToStringWithOperandTypes();
  XLS_ASSIGN_OR_RETURN(Node * req_addr,
                       tuple_index(rw_block_ports.req_data->operand(0), 0));
  XLS_ASSIGN_OR_RETURN(Node * req_wr_data,
                       tuple_index(rw_block_ports.req_data->operand(0), 1));
  XLS_ASSIGN_OR_RETURN(Node * req_we,
                       tuple_index(rw_block_ports.req_data->operand(0), 2));
  XLS_ASSIGN_OR_RETURN(Node * req_re,
                       tuple_index(rw_block_ports.req_data->operand(0), 3));

  Node* req_valid = rw_block_ports.req_valid->operand(0);

  // Make names for each element of the request tuple. They will end up each
  // having their own port.
  std::string_view ram_name = ram_config.ram_name();
  std::string req_addr_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_addr"));
  std::string req_wr_data_name =
      block->UniquifyNodeName(absl::StrCat(ram_name, "_wr_data"));
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
                          rw_block_ports.resp_data->GetType()));
  XLS_RETURN_IF_ERROR(
      rw_block_ports.resp_data->ReplaceUsesWith(resp_rd_data_port));

  // Add buffer before resp_ready
  std::string resp_ready_port_buf_name =
      absl::StrFormat("__%s_buffer", rw_block_ports.resp_ready->name());
  XLS_ASSIGN_OR_RETURN(
      Node * resp_ready_port_buf,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), rw_block_ports.resp_ready->operand(0),
          Op::kIdentity, resp_ready_port_buf_name));

  // Update channel ready/valid ports usages with new internal signals.
  XLS_RETURN_IF_ERROR(
      rw_block_ports.resp_ready->ReplaceOperandNumber(0, resp_ready_port_buf));
  XLS_RETURN_IF_ERROR(
      rw_block_ports.resp_valid->ReplaceUsesWith(ram_resp_valid));
  XLS_RETURN_IF_ERROR(
      rw_block_ports.req_ready->ReplaceUsesWith(resp_ready_port_buf));

  // Add zero-latency buffer at output of ram
  std::vector<Node*> valid_nodes;
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
  XLS_ASSIGN_OR_RETURN(auto* req_we_port,
                       block->AddOutputPort(req_we_name, req_we_valid));
  XLS_ASSIGN_OR_RETURN(auto* req_re_port,
                       block->AddOutputPort(req_re_name, req_re_valid));

  // Remove ports that have been replaced.
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.req_data));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.req_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.req_ready));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.resp_valid));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.resp_ready));
  XLS_RETURN_IF_ERROR(block->RemoveNode(rw_block_ports.resp_data));

  if (unit->signature.has_value()) {
    XLS_RETURN_IF_ERROR(UpdateModuleSignatureChannelsToRams(
        &unit->signature.value(),
        /*ram_name=*/ram_config.ram_name(),
        /*req_name=*/ram_config.rw_port_configuration().request_channel_name,
        /*resp_name=*/ram_config.rw_port_configuration().response_channel_name,
        /*address=*/req_addr_port,
        /*write_data=*/req_wr_data_port,
        /*read_enable=*/req_re_port,
        /*write_enable=*/req_we_port,
        /*read_data=*/resp_rd_data_port));
  }

  return true;
}

absl::flat_hash_map<std::string, ram_rewrite_function_t>*
GetRamRewriteFunctionMap() {
  static auto* singleton =
      new absl::flat_hash_map<std::string, ram_rewrite_function_t>{
          {"1RW", Ram1RWRewrite},
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

  for (auto& ram_configuration : options.codegen_options.ram_configurations()) {
    XLS_ASSIGN_OR_RETURN(bool this_one_changed,
                         RamRewrite(unit, options, *ram_configuration));
    changed |= this_one_changed;
  }

  return changed;
}

}  // namespace xls::verilog
