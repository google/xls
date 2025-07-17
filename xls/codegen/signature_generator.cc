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

#include "xls/codegen/signature_generator.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast/vast.h"
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

namespace xls::verilog {
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
}  // namespace

absl::StatusOr<ModuleSignature> GenerateSignature(
    const CodegenOptions& options, Block* block,
    const absl::flat_hash_map<Node*, Stage>& stage_map) {
  ModuleSignatureBuilder b(SanitizeVerilogIdentifier(block->name()));

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
    ChannelDirectionProto direction_proto =
        metadata.direction == ChannelDirection::kSend
            ? CHANNEL_DIRECTION_SEND
            : CHANNEL_DIRECTION_RECEIVE;
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

  int64_t register_levels = 0;
  if (options.flop_inputs()) {
    VLOG(3) << absl::StreamFormat("Adding input latency = %d - 1.",
                                  options.GetInputLatency());
    // Schedule has been adjusted by MaybeAddInputOutputFlopsToSchedule(), but
    // this does not take zero-latency buffers into account. We subtract 1 from
    // latency only if inputs are zero-latency.
    register_levels += options.GetInputLatency() - 1;
  }
  if (options.flop_outputs()) {
    VLOG(3) << absl::StreamFormat("Adding output latency = %d - 1.",
                                  options.GetOutputLatency());
    // Similar to above, if the output is flopped with zero-latency buffers we
    // should shorten the latency.
    register_levels += options.GetOutputLatency() - 1;
  }
  if (!stage_map.empty()) {
    // The stage map generally comes from a `PipelineSchedule`. In the past, we
    // directly used the `PipelineSchedule`, but the schedule is really an
    // artifact of the original Function/Proc, not the converted Block. We
    // derive the latency of the pipeline from the stage map by finding the
    // maximum value.
    int64_t pipeline_registers =
        absl::c_max_element(stage_map, [](const std::pair<Node*, Stage>& lhs,
                                          const std::pair<Node*, Stage>& rhs) {
          return lhs.second < rhs.second;
        })->second;
    VLOG(3) << absl::StreamFormat("Adding pipeline registers = %d.",
                                  pipeline_registers - 1);
    register_levels += pipeline_registers;
  }
  VLOG(3) << absl::StreamFormat("Register levels = %d .", register_levels);
  if (register_levels == 0 && !options.emit_as_pipeline()) {
    // Block has no registers. The block is combinational.
    b.WithCombinationalInterface();
  } else {
    std::optional<PipelineControl> pipeline_control;
    if (options.valid_control().has_value()) {
      pipeline_control = PipelineControl();
      *(pipeline_control->mutable_valid()) = options.valid_control().value();
    }
    b.WithPipelineInterface(
        register_levels,
        /*initiation_interval=*/block->GetInitiationInterval().value_or(1),
        pipeline_control);
  }

  return b.Build();
}

}  // namespace xls::verilog
