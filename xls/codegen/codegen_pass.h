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

#ifndef XLS_CODEGEN_CODEGEN_PASS_H_
#define XLS_CODEGEN_CODEGEN_PASS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// Defines the pass types for passes involved in lowering and optimizing prior
// to codegen.

// Options passed to each pass.
struct CodegenPassOptions : public PassOptions {
  // Options to use for codegen.
  CodegenOptions codegen_options;

  // Optional schedule. If given, a feedforward pipeline is generated based on
  // the schedule.
  std::optional<PipelineSchedule> schedule;
};

using Stage = int64_t;

// Data structures holding the data and (optional) predicate nodes representing
// streaming inputs (receive over streaming channel) and streaming outputs (send
// over streaming channel) in the generated block.
struct StreamingInput {
  InputPort* port;
  InputPort* port_valid;
  OutputPort* port_ready;

  // signal_data and signal_valid respresent the internal view of the
  // streaming input.  These are used (ex. for handling non-blocking receives)
  // as additional logic are placed between the ports and the pipeline use of
  // these signals.
  //
  // Pictorially:
  //      | port_data   | port_valid   | port_ready
  //  ----|-------------|--------------|-------------
  //  |   |             |              |            |
  //  | |--------------------------|   |            |
  //  | |  Logic / Adapter         |   |            |
  //  | |                          |   |            |
  //  | |--------------------------|   |            |
  //  |   | signal_data | signal_valid |            |
  //  |   |             |              |            |
  //  |                                             |
  //  -----------------------------------------------
  Node* signal_data;
  Node* signal_valid;

  Channel* channel;
  std::optional<Node*> predicate;
};

struct StreamingOutput {
  OutputPort* port;
  OutputPort* port_valid;
  InputPort* port_ready;
  Channel* channel;
  std::optional<Node*> predicate;
};

// Data structures holding the port representing single value inputs/outputs
// in the generated block.
struct SingleValueInput {
  InputPort* port;
  Channel* channel;
};

struct SingleValueOutput {
  OutputPort* port;
  Channel* channel;
};

// A data structure representing a pipeline register for a single XLS IR value.
struct PipelineRegister {
  Register* reg;
  RegisterWrite* reg_write;
  RegisterRead* reg_read;
};

// A data structure representing a state register for a single XLS IR value.
// If the state is always populated with a valid value, reg_full.* == nullptr.
// (This should always be true for non-pipelined code.)
//
// If the state may not always have a valid value (e.g., a non-trivial backedge
// from the next-state value to the param node), reg_full should be a 1-bit
// register, and should be set (by reg_full_write) to 1 when the state is valid
// and 0 when it is not. Stages in flow control may wait on reg_full_read to
// determine when they can run.
struct StateRegister {
  std::string name;
  Value reset_value;
  Stage read_stage;
  Stage write_stage;
  Register* reg;
  RegisterWrite* reg_write;
  RegisterRead* reg_read;
  Register* reg_full;
  RegisterWrite* reg_full_write;
  RegisterRead* reg_full_read;
};

// The collection of pipeline registers for a single stage.
using PipelineStageRegisters = std::vector<PipelineRegister>;

struct StreamingIOPipeline {
  std::vector<std::vector<StreamingInput>> inputs;
  std::vector<std::vector<StreamingOutput>> outputs;
  std::vector<std::vector<int64_t>> input_states;
  std::vector<std::vector<int64_t>> output_states;
  std::vector<SingleValueInput> single_value_inputs;
  std::vector<SingleValueOutput> single_value_outputs;
  std::vector<PipelineStageRegisters> pipeline_registers;
  // `state_registers` includes an element for each state element in the
  // proc. The vector element is nullopt if the state element is an empty tuple.
  std::vector<std::optional<StateRegister>> state_registers;
  std::optional<OutputPort*> idle_port;

  // Node in block that represents when all output channels (that
  // are predicated true) are ready.
  // See MakeInputReadyPortsForOutputChannels().
  std::vector<Node*> all_active_outputs_ready;
  std::vector<Node*> all_active_inputs_valid;
  std::vector<Node*> all_active_states_ready;
  std::vector<Node*> all_active_states_valid;

  std::vector<Node*> pipeline_valid;
  std::vector<Node*> stage_valid;
  std::vector<Node*> stage_done;

  absl::flat_hash_map<Node*, Stage> node_to_stage_map;
};

// Plumbs a valid signal through the block. This includes:
// (1) Add an input port for a single-bit valid signal.
// (2) Add a pipeline register for the valid signal at each pipeline stage.
// (3) Add an output port for the valid signal from the final stage of the
//     pipeline.
// (4) Use the (pipelined) valid signal as the load enable signal for other
//     pipeline registers in each stage. This is a power optimization
//     which reduces switching in the data path when the valid signal is
//     deasserted.
// TODO(meheff): 2021/08/21 This might be better performed as a codegen pass.
struct ValidPorts {
  InputPort* input;
  OutputPort* output;
};

struct FunctionConversionMetadata {
  std::optional<ValidPorts> valid_ports;
};
struct ProcConversionMetadata {
  std::vector<Node*> valid_flops;
};


// Data structure operated on by codegen passes. Contains the IR and associated
// metadata which may be used and mutated by passes.
struct CodegenPassUnit {
  CodegenPassUnit(Package* p, Block* b) : package(p), block(b) {}

  // The package containing IR to lower.
  Package* package;

  // The top-level block to generate a Verilog module for.
  Block* block;

  // Metadata for pipelined blocks.
  // TODO(google/xls#1060): refactor so conversion_metadata is in
  // StreamingIOPipeline and more elements are split as function- or proc-only.
  StreamingIOPipeline streaming_io_and_pipeline;
  // Only set when converting functions.
  std::variant<FunctionConversionMetadata, ProcConversionMetadata>
      conversion_metadata;


  // The signature is generated (and potentially mutated) during the codegen
  // process.
  // TODO(https://github.com/google/xls/issues/410): 2021/04/27 Consider adding
  // a "block" contruct which corresponds to a verilog module. This block could
  // hold its own signature. This would help prevent the signature from getting
  // out-of-sync with the IR.
  std::optional<ModuleSignature> signature;

  // These methods are required by CompoundPassBase.
  std::string DumpIr() const;
  const std::string& name() const { return block->name(); }
};

using CodegenPass = PassBase<CodegenPassUnit, CodegenPassOptions, PassResults>;
using CodegenCompoundPass =
    CompoundPassBase<CodegenPassUnit, CodegenPassOptions, PassResults>;
using CodegenInvariantChecker = CodegenCompoundPass::InvariantChecker;

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_PASS_H_
