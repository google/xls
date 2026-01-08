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

#include "xls/codegen_v_1_5/signature_generation_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast/vast.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

FlopKindProto ToProtoFlop(FlopKind f) {
  switch (f) {
    case FlopKind::kNone:
      return FLOP_KIND_NONE;
    case FlopKind::kFlop:
      return FLOP_KIND_FLOP;
    case FlopKind::kSkid:
      return FLOP_KIND_SKID;
    case FlopKind::kZeroLatency:
      return FLOP_KIND_ZERO_LATENCY;
  }
}

absl::StatusOr<verilog::ModuleSignature> GenerateSignature(
    const verilog::CodegenOptions& options, Block* block,
    std::optional<int64_t> pipeline_stages) {
  verilog::ModuleSignatureBuilder b(
      verilog::SanitizeVerilogIdentifier(block->name()));

  // Optionally add clock and reset.
  if (options.clock_name().has_value()) {
    b.WithClock(options.clock_name().value());
  }
  if (options.reset().has_value()) {
    b.WithReset(options.reset()->name(), options.reset()->asynchronous(),
                options.reset()->active_low());
  }

  std::vector<Type*> input_types;
  std::vector<Type*> output_types;

  // Returns true if the given node is a data port (reset and valid in/out are
  // not considered data ports).
  auto is_data_port = [&](Node* node) {
    if (options.reset().has_value() &&
        node->GetName() == options.reset()->name()) {
      return false;
    }
    if (options.valid_control().has_value() &&
        (node->GetName() == options.valid_control()->input_name() ||
         node->GetName() == options.valid_control()->output_name())) {
      return false;
    }
    return true;
  };

  for (const Block::Port& port : block->GetPorts()) {
    if (std::holds_alternative<InputPort*>(port)) {
      InputPort* input_port = std::get<InputPort*>(port);
      if (!is_data_port(input_port)) {
        continue;
      }
      input_types.push_back(input_port->GetType());
      b.AddDataInput(input_port->GetName(), input_port->GetType());
    } else if (std::holds_alternative<OutputPort*>(port)) {
      OutputPort* output_port = std::get<OutputPort*>(port);
      if (!is_data_port(output_port)) {
        continue;
      }
      Type* type = output_port->operand(0)->GetType();
      output_types.push_back(type);
      b.AddDataOutput(output_port->GetName(), type);
    } else {
      // No need to do anything for the clock port.
      XLS_RET_CHECK(std::holds_alternative<Block::ClockPort*>(port));
    }
  }

  Package* p = block->package();
  VLOG(5) << "GenerateSignature called on package:";
  XLS_VLOG_LINES(5, p->DumpIr());

  // Add internal channels and block instantiations. Iterate through the fifo
  // instantiations.
  for (const ::xls::Instantiation* instantiation : block->GetInstantiations()) {
    if (instantiation->kind() == ::xls::InstantiationKind::kFifo) {
      const FifoInstantiation* fifo =
          down_cast<const FifoInstantiation*>(instantiation);
      if (fifo->channel_name().has_value()) {
        b.AddStreamingChannel(fifo->channel_name().value(), fifo->data_type(),
                              FlowControl::kReadyValid, fifo->fifo_config());
      }
      b.AddFifoInstantiation(p, fifo->name(), fifo->channel_name(),
                             fifo->data_type(), fifo->fifo_config());
    } else if (instantiation->kind() == ::xls::InstantiationKind::kBlock) {
      const BlockInstantiation* block_instantiation =
          down_cast<const BlockInstantiation*>(instantiation);
      b.AddBlockInstantiation(p,
                              block_instantiation->instantiated_block()->name(),
                              block_instantiation->name());
    }
  }

  // Add interface channels. Iterate through the block metadata.
  for (const auto& [channel, direction] : block->GetChannelsWithMappedPorts()) {
    XLS_ASSIGN_OR_RETURN(ChannelPortMetadata metadata,
                         block->GetChannelPortMetadata(channel, direction));
    verilog::ChannelDirectionProto direction_proto =
        metadata.direction == ChannelDirection::kSend
            ? verilog::ChannelDirectionProto::CHANNEL_DIRECTION_SEND
            : verilog::ChannelDirectionProto::CHANNEL_DIRECTION_RECEIVE;
    switch (metadata.channel_kind) {
      case ChannelKind::kStreaming: {
        FlowControl flow_control =
            (metadata.ready_port.has_value() && metadata.valid_port.has_value())
                ? FlowControl::kReadyValid
                : FlowControl::kNone;
        b.AddStreamingChannelInterface(
            metadata.channel_name, direction_proto, metadata.type, flow_control,
            metadata.data_port, metadata.ready_port, metadata.valid_port,
            ToProtoFlop(metadata.flop_kind), metadata.stage);
        break;
      }
      case ChannelKind::kSingleValue:
        b.AddSingleValueChannelInterface(
            metadata.channel_name, direction_proto, metadata.type,
            metadata.data_port.value(), ToProtoFlop(metadata.flop_kind),
            metadata.stage);
        break;
    }
  }

  VLOG(3) << "Computing register levels for block: " << block->name();
  int64_t register_levels = 0;

  if (pipeline_stages.has_value() && *pipeline_stages > 0) {
    int64_t pipeline_register_levels = *pipeline_stages - 1;
    VLOG(3) << absl::StreamFormat("Adding pipeline registers = %d.",
                                  pipeline_register_levels);
    register_levels += pipeline_register_levels;
  }

  // If the block is a scheduled function, we should account for I/O flopping as
  // configured in our options; otherwise, we'll rely entirely on channel
  // metadata to account for this.
  if (block->IsScheduled() &&
      down_cast<ScheduledBlock*>(block)->source() != nullptr &&
      down_cast<ScheduledBlock*>(block)->source()->IsFunction()) {
    if (options.flop_inputs()) {
      register_levels +=
          verilog::CodegenOptions::IOKindLatency(options.flop_inputs_kind());
    }
    if (options.flop_outputs()) {
      register_levels +=
          verilog::CodegenOptions::IOKindLatency(options.flop_outputs_kind());
    }
  } else {
    int64_t max_receive_latency = 0;
    int64_t max_send_latency = 0;
    for (const auto& [channel, metadata] : block->GetChannelPortMetadata()) {
      if (metadata.direction == ChannelDirection::kReceive) {
        VLOG(3) << absl::StreamFormat(
            "Receive latency for channel %s: %d.", channel.first,
            verilog::CodegenOptions::FlopKindLatency(metadata.flop_kind));
        max_receive_latency = std::max(
            max_receive_latency,
            verilog::CodegenOptions::FlopKindLatency(metadata.flop_kind));
      }

      if (metadata.direction == ChannelDirection::kSend) {
        VLOG(3) << absl::StreamFormat(
            "Send latency for channel %s: %d.", channel.first,
            verilog::CodegenOptions::FlopKindLatency(metadata.flop_kind));
        max_send_latency = std::max(
            max_send_latency,
            verilog::CodegenOptions::FlopKindLatency(metadata.flop_kind));
      }
    }

    if (max_receive_latency > 0) {
      VLOG(3) << absl::StreamFormat("Adding input latency = %d.",
                                    max_receive_latency);
      register_levels += max_receive_latency;
    }
    if (max_send_latency > 0) {
      VLOG(3) << absl::StreamFormat("Adding output latency = %d.",
                                    max_send_latency);
      register_levels += max_send_latency;
    }
  }

  VLOG(3) << absl::StreamFormat("Register levels = %d.", register_levels);
  if (register_levels == 0 && options.generate_combinational()) {
    // Block has no registers. The block is combinational.
    b.WithCombinationalInterface();
  } else {
    std::optional<verilog::PipelineControl> pipeline_control;
    if (options.valid_control().has_value()) {
      pipeline_control = verilog::PipelineControl();
      *(pipeline_control->mutable_valid()) = options.valid_control().value();
    }
    b.WithPipelineInterface(
        register_levels,
        /*initiation_interval=*/block->GetInitiationInterval().value_or(1),
        pipeline_control);
  }

  return b.Build();
}

}  // namespace

absl::StatusOr<bool> SignatureGenerationPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (block->GetSignature().has_value()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        verilog::ModuleSignature signature,
        GenerateSignature(options.codegen_options, block.get(),
                          /*pipeline_stages=*/block->IsScheduled()
                              ? std::make_optional(block->stages().size())
                              : std::nullopt));
    block->SetSignature(signature.proto());
    changed = true;
  }
  return changed;
}

}  // namespace xls::codegen
