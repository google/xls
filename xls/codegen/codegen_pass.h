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

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/codegen/module_signature.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// Defines the pass types for passes involved in lowering and optimizing prior
// to codegen.

// Options passed to each pass.
struct CodegenPassOptions : public PassOptionsBase {
  // Options to use for codegen.
  CodegenOptions codegen_options;

  // Optional schedule. If given, a feedforward pipeline is generated based on
  // the schedule.
  std::optional<PipelineSchedule> schedule;

  // Optional delay estimator. If given, block delay metrics will be added to
  // the signature.
  const DelayEstimator* delay_estimator = nullptr;
};

using Stage = int64_t;

// Data structures holding the data and (optional) predicate nodes representing
// streaming inputs (receive over streaming channel) and streaming outputs (send
// over streaming channel) in the generated block.
struct StreamingInput {
  // Note that these ports can either be external I/Os or be ports from a FIFO
  // instantiation.
  std::optional<Node*> port;
  Node* port_valid;
  Node* port_ready;

  // signal_data and signal_valid represent the internal view of the streaming
  // input.  These are used (ex. for handling non-blocking receives) as
  // additional logic are placed between the ports and the pipeline use of these
  // signals.
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
  std::optional<Node*> signal_data;
  std::optional<Node*> signal_valid;

  Channel* channel;
  std::optional<FifoInstantiation*> fifo_instantiation;
  std::optional<Node*> predicate;

  bool IsExternal() const {
    return (!port.has_value() || port.value()->op() == Op::kInputPort) &&
           port_valid->op() == Op::kInputPort &&
           port_ready->op() == Op::kOutputPort;
  }
  bool IsInstantiation() const {
    return (!port.has_value() ||
            port.value()->op() == Op::kInstantiationOutput) &&
           port_valid->op() == Op::kInstantiationOutput &&
           port_ready->op() == Op::kInstantiationInput;
  }
};

struct StreamingOutput {
  // Note that these ports can either be external I/Os or be ports from a FIFO
  // instantiation.
  std::optional<Node*> port;
  Node* port_valid;
  Node* port_ready;
  Channel* channel;
  std::optional<FifoInstantiation*> fifo_instantiation;
  std::optional<Node*> predicate;

  bool IsExternal() const {
    return port.value()->op() == Op::kOutputPort &&
           port_valid->op() == Op::kOutputPort &&
           port_ready->op() == Op::kInputPort;
  }
  bool IsInstantiation() const {
    return port.value()->op() == Op::kInstantiationInput &&
           port_valid->op() == Op::kInstantiationInput &&
           port_ready->op() == Op::kInstantiationOutput;
  }
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
  struct NextValue {
    Stage stage;

    // If absent, this is a next value that leaves the previous value unchanged.
    std::optional<Node*> value;

    // If absent, this is an unpredicated next value, and is always used.
    std::optional<Node*> predicate;
  };

  std::string name;
  Value reset_value;
  Stage read_stage;
  std::vector<NextValue> next_values;
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
  // Map of stage# -> input descriptor
  std::vector<std::vector<StreamingInput>> inputs;
  // Map of stage# -> output descriptor
  std::vector<std::vector<StreamingOutput>> outputs;
  // Map of stage# -> state register vector. (Values are pointers into
  // state_registers array). Order of inner list is not meaningful.
  std::vector<std::vector<int64_t>> input_states;
  // Map of stage# -> state register vector. (Values are pointers into
  // state_registers array). Order of inner list is not meaningful.
  std::vector<std::vector<int64_t>> output_states;
  // List linking the single-value input port to the channel it implements.
  std::vector<SingleValueInput> single_value_inputs;
  // List linking the single-value output port to the channel it implements.
  std::vector<SingleValueOutput> single_value_outputs;
  // Map of stage# -> pipeline registers which hold each value that lives beyond
  // that stage.
  std::vector<PipelineStageRegisters> pipeline_registers;
  // `state_registers` includes an element for each state element in the
  // proc. The vector element is nullopt if the state element is an empty tuple.
  // This is pointed to by the input_states and output_states fields.
  std::vector<std::optional<StateRegister>> state_registers;
  std::optional<OutputPort*> idle_port;

  // Map of stage# -> node which denotes if the stages input data from the
  // previous stage is valid at this stage (i.e. the stage does not contain a
  // bubble). See MakePipelineStagesForValid().
  std::vector<std::optional<Node*>> pipeline_valid;
  // Node denoting if all of the specific stage's input data is valid.
  std::vector<std::optional<Node*>> stage_valid;
  // Node denoting if a specific stage is finished.
  std::vector<std::optional<Node*>> stage_done;

  // Map from node to stage.
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
  std::vector<std::optional<Node*>> valid_flops;
};

// Per-block metadata used for codegen.
struct CodegenMetadata {
  StreamingIOPipeline streaming_io_and_pipeline;
  std::variant<FunctionConversionMetadata, ProcConversionMetadata>
      conversion_metadata;

  // The signature is generated (and potentially mutated) during the codegen
  // process.
  // TODO(https://github.com/google/xls/issues/410): 2021/04/27 Consider adding
  // a "block" construct which corresponds to a verilog module. This block could
  // hold its own signature. This would help prevent the signature from getting
  // out-of-sync with the IR.
  std::optional<ModuleSignature> signature;

  // Proven knowledge about which stages are active concurrently.
  //
  // If absent all stages should be considered potentially concurrently active
  // with one another.
  std::optional<ConcurrentStageGroups> concurrent_stages;
};

// Data structure operated on by codegen passes. Contains the IR and associated
// metadata which may be used and mutated by passes.
struct CodegenPassUnit {
  // Ordering for blocks lexicographically by name.
  struct BlockByName {
    std::strong_ordering operator()(const Block* lhs, const Block* rhs) const {
      return lhs->name() <=> rhs->name();
    }
  };
  CodegenPassUnit(Package* p, Block* b) : package(p), top_block(b) {}

  // The package containing IR to lower.
  Package* package;

  // The top-level block to generate a Verilog module for.
  absl::Nonnull<Block*> top_block;

  // Metadata for pipelined blocks.
  // TODO(google/xls#1060): refactor so conversion_metadata is in
  // StreamingIOPipeline and more elements are split as function- or proc-only.
  using MetadataMap = absl::btree_map<Block*, CodegenMetadata, BlockByName>;
  MetadataMap metadata;

  // These methods are required by CompoundPassBase.
  std::string DumpIr() const;
  const std::string& name() const {
    CHECK_NE(top_block, nullptr);
    return top_block->name();
  }
  int64_t GetNodeCount() const;
  const TransformMetrics& transform_metrics() const {
    return package->transform_metrics();
  }

  // Clean up any dangling pointers in codegen metadata.
  void GcMetadata();
};

using CodegenPass = PassBase<CodegenPassUnit, CodegenPassOptions, PassResults>;
using CodegenCompoundPass =
    CompoundPassBase<CodegenPassUnit, CodegenPassOptions, PassResults>;
using CodegenInvariantChecker = CodegenCompoundPass::InvariantChecker;

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_PASS_H_
