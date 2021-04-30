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

#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_util.h"

namespace xls::verilog {

absl::StatusOr<ModuleSignature> GenerateSignature(
    const CodegenOptions& options, FunctionBase* func_base,
    absl::optional<PipelineSchedule> schedule) {
  std::string module_name = options.module_name().has_value()
                                ? std::string{options.module_name().value()}
                                : func_base->name();
  ModuleSignatureBuilder b(module_name);

  // Optionally add clock and reset.
  if (options.clock_name().has_value()) {
    b.WithClock(options.clock_name().value());
  }
  if (options.reset().has_value()) {
    b.WithReset(options.reset()->name(), options.reset()->asynchronous(),
                options.reset()->active_low());
  }

  if (Function* func = dynamic_cast<Function*>(func_base)) {
    // Function given, use to function params and output type to generate
    // type signature.
    for (Param* param : func->params()) {
      b.AddDataInput(param->name(), param->GetType()->GetFlatBitCount());
    }
    b.AddDataOutput("out", func->return_value()->GetType()->GetFlatBitCount());
    b.WithFunctionType(func->GetType());
  } else {
    // Given func_base is a proc. Generate signature from input/output ports.
    Proc* proc = down_cast<Proc*>(func_base);
    // Generate signature from inputs and outputs.
    std::vector<Type*> input_types;
    std::vector<Type*> output_types;
    XLS_ASSIGN_OR_RETURN(std::vector<Proc::Port> ports, proc->GetPorts());
    for (const Proc::Port& port : ports) {
      if (port.direction == Proc::PortDirection::kInput) {
        input_types.push_back(port.channel->type());
        b.AddDataInput(port.channel->name(),
                       port.channel->type()->GetFlatBitCount());
      } else {
        output_types.push_back(port.channel->type());
        b.AddDataOutput(port.channel->name(),
                        port.channel->type()->GetFlatBitCount());
      }
    }

    if (output_types.size() == 1) {
      b.WithFunctionType(
          proc->package()->GetFunctionType(input_types, output_types.front()));
    }
  }

  // If proc has a send/receive over any kind of channel except ports, then
  // generate an unknown signature type.
  // TODO(meheff): 2021/04/27 Add support for streaming channels.
  for (Node* node : func_base->nodes()) {
    if (IsChannelNode(node)) {
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(node));
      if (!ch->IsPort()) {
        return b.WithUnknownInterface().Build();
      }
    }
  }

  int64_t register_levels = 0;
  if (options.flop_inputs()) {
    register_levels++;
  }
  if (options.flop_outputs()) {
    register_levels++;
  }
  if (schedule.has_value()) {
    register_levels += schedule.value().length() - 1;
  }
  if (register_levels == 0) {
    // Block has no registers. The block is combinational.
    b.WithCombinationalInterface();
  } else {
    // We assume initiation-interval of one because that is all we generate at
    // the moment.
    b.WithPipelineInterface(register_levels, /*initiation_interval=*/1);
  }

  return b.Build();
}

}  // namespace xls::verilog
