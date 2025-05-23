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
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"
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

  // The config to use for loopback channels that are fabricated during codegen
  // to implement proc state members.
  ChannelConfig state_channel_config = ChannelConfig{};
};

using Stage = int64_t;

enum class ConnectionKind { kInternal, kExternal };

// Data structures holding the data and (optional) predicate nodes representing
// streaming inputs (receive over streaming channel) and streaming outputs (send
// over streaming channel) in the generated block.
class StreamingInput {
  // Note that these ports can either be external I/Os or be ports from a FIFO
  // instantiation.
 public:
  StreamingInput(Block* block, std::string_view channel_name,
                 ConnectionKind kind)
      : block_(block), channel_name_(channel_name), kind_(kind) {}

  std::optional<Node*> GetDataPort() const;
  Node* GetValidPort() const;
  Node* GetReadyPort() const;
  std::string_view GetChannelName() const { return channel_name_; }
  ChannelRef GetChannel() const;

  const std ::optional<Node*>& GetPredicate() const { return predicate_; }
  void SetPredicate(std::optional<Node*> value) { predicate_ = value; }

  std ::optional<FifoInstantiation*> GetFifoInstantiation() const {
    if (IsExternal()) {
      return std::nullopt;
    }
    return block_->GetFifoInstantiationForChannel(channel_name_).value();
  }

  const std ::optional<Node*>& GetSignalData() const { return signal_data_; }
  void SetSignalData(std::optional<Node*> value) { signal_data_ = value; }
  const std ::optional<Node*>& GetSignalValid() const { return signal_valid_; }
  void SetSignalValid(std::optional<Node*> value) { signal_valid_ = value; }

  bool IsExternal() const { return kind_ == ConnectionKind::kExternal; }
  bool IsInstantiation() const { return kind_ == ConnectionKind::kInternal; }

 private:
  Block* block_;
  std::string channel_name_;
  ConnectionKind kind_;

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
  std::optional<Node*> signal_data_;
  std::optional<Node*> signal_valid_;
  std::optional<Node*> predicate_;
};

class StreamingOutput {
 public:
  StreamingOutput(Block* block, std::string_view channel_name,
                  ConnectionKind kind)
      : block_(block), channel_name_(channel_name), kind_(kind) {}

  std::optional<Node*> GetDataPort() const;
  Node* GetValidPort() const;
  Node* GetReadyPort() const;
  std::string_view GetChannelName() const { return channel_name_; }

  ChannelRef GetChannel() const;

  const std ::optional<Node*>& GetPredicate() const { return predicate_; }
  void SetPredicate(std::optional<Node*> value) { predicate_ = value; }

  const std::optional<Node*>& GetData() const { return data_; }
  void SetData(Node* value) { data_ = value; }

  std ::optional<FifoInstantiation*> GetFifoInstantiation() const {
    if (IsExternal()) {
      return std::nullopt;
    }
    return block_->GetFifoInstantiationForChannel(channel_name_).value();
  }

  bool IsExternal() const { return kind_ == ConnectionKind::kExternal; }
  bool IsInstantiation() const { return kind_ == ConnectionKind::kInternal; }

 private:
  // Note that these ports can either be external I/Os or be ports from a FIFO
  // instantiation.
  Block* block_;
  std::string channel_name_;
  ConnectionKind kind_;

  std::optional<Node*> predicate_;
  std::optional<Node*> data_;
};

// Data structures holding the port representing single value inputs/outputs
// in the generated block.
struct SingleValueInput {
 public:
  SingleValueInput(Block* block, std::string_view channel_name)
      : block_(block), channel_name_(channel_name) {}

  std::string_view GetChannelName() const { return channel_name_; }
  ChannelRef GetChannel() const;
  std::optional<InputPort*> GetDataPort() const;

 private:
  Block* block_;
  std::string channel_name_;
};

class SingleValueOutput {
 public:
  SingleValueOutput(Block* block, std::string_view channel_name)
      : block_(block), channel_name_(channel_name) {}

  std::string_view GetChannelName() const { return channel_name_; }
  ChannelRef GetChannel() const;
  std::optional<OutputPort*> GetDataPort() const;

 private:
  Block* block_;
  std::string channel_name_;
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
  Node* read_predicate;
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

  // Proven knowledge about which stages are active concurrently.
  //
  // If absent all stages should be considered potentially concurrently active
  // with one another.
  std::optional<ConcurrentStageGroups> concurrent_stages;
};

// Mutable context data structure operated on by codegen passes.
class CodegenContext {
 public:
  // Ordering for blocks lexicographically by name.
  struct BlockByName {
    std::strong_ordering operator()(const Block* lhs, const Block* rhs) const {
      return lhs->name() <=> rhs->name();
    }
  };

  using MetadataMap = absl::btree_map<Block*, CodegenMetadata, BlockByName>;

  explicit CodegenContext(Block* b = nullptr) : top_block_(b) {}

  // Has a top block been set?
  bool HasTopBlock() const { return top_block_ != nullptr; }

  // Returns the top block if it is set.  Dies if it does not.
  Block* top_block() {
    CHECK_NE(top_block_, nullptr);

    return top_block_;
  }

  // Sets the top block.
  void SetTopBlock(Block* ABSL_NONNULL b) { top_block_ = b; }

  // Adds necessary maps from the FunctionBase to the Schedule.
  void AssociateSchedule(FunctionBase* fb, PipelineSchedule schedule) {
    function_base_to_schedule_.emplace(fb, std::move(schedule));
  }

  // Returns the metadata map.
  MetadataMap& metadata() { return metadata_; }
  const MetadataMap& metadata() const { return metadata_; }

  // Returns true if we have metadata forthe block.
  bool HasMetadataForBlock(Block* ABSL_NONNULL block) const {
    return metadata_.contains(block);
  }

  // Returns the metadata for the given block (will create one if it doesn't
  // exist).
  CodegenMetadata& GetMetadataForBlock(Block* ABSL_NONNULL block) {
    return metadata_[block];
  }

  void SetMetadataForBlock(Block* ABSL_NONNULL block,
                           CodegenMetadata metadata) {
    metadata_[block] = std::move(metadata);
  }

  // Clean up any dangling pointers in codegen metadata.
  void GcMetadata();

  // Returns function to schedule mapping.
  const absl::btree_map<FunctionBase*, PipelineSchedule,
                        struct FunctionBase::NameLessThan>&
  function_base_to_schedule() const {
    return function_base_to_schedule_;
  }
  absl::btree_map<FunctionBase*, PipelineSchedule,
                  struct FunctionBase::NameLessThan>&
  function_base_to_schedule() {
    return function_base_to_schedule_;
  }

  // Returns the manager of stage conversion metadata.
  StageConversionMetadata& stage_conversion_metadata() {
    return stage_conversion_metadata_;
  }

  absl::flat_hash_map<std::string, std::string>& register_renames() {
    return register_renames_;
  }

  absl::flat_hash_map<std::string, xls::Type*>& inserted_registers() {
    return inserted_registers_;
  }

 private:
  // Metadata for pipelined blocks.
  // TODO(google/xls#1060): refactor so conversion_metadata is in
  // StreamingIOPipeline and more elements are split as function- or proc-only.
  MetadataMap metadata_;

  // Sorted map from FunctionBase to schedule.
  absl::btree_map<FunctionBase*, PipelineSchedule,
                  struct FunctionBase::NameLessThan>
      function_base_to_schedule_;

  // The top-level block to generate a Verilog module for.
  Block* top_block_;

  // Object that associates function bases to metadata created during
  // stage conversion.
  StageConversionMetadata stage_conversion_metadata_;

  // A map from original register names to renamed register. Note that the
  // register names are generally going to include the elaboration-prefix if not
  // in the top block. Generally pass should only add to this and only users who
  // have tight control and knowledge of the pass pipeline should attempt to
  // interpret the values in this map. For example if you have 2 passes the
  // first renames 'a' -> 'b' and the second 'b' -> 'c' you would see both
  // renames in this map (i.e. 2 elements).
  absl::flat_hash_map<std::string, std::string> register_renames_;

  // Map from register name (including instantiation path) to type of the
  // register of registers some pass inserted which was not initially a part of
  // the design (eg FIFO implementation registers).
  absl::flat_hash_map<std::string, xls::Type*> inserted_registers_;
};

using CodegenPass = PassBase<CodegenPassOptions, CodegenContext>;
using CodegenFunctionBasePass =
    FunctionBasePass<CodegenPassOptions, CodegenContext>;
using CodegenCompoundPass =
    CompoundPassBase<CodegenPassOptions, CodegenContext>;
using CodegenInvariantChecker = CodegenCompoundPass::InvariantChecker;

// Abstract base class for a pass that runs on all scheduled functions.
class CodegenFunctionPass : public CodegenFunctionBasePass {
 public:
  using CodegenFunctionBasePass::CodegenFunctionBasePass;

  virtual ~CodegenFunctionPass() {}

  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* fb, const CodegenPassOptions& options, PassResults* results,
      CodegenContext& context) const override {
    if (!fb->IsFunction()) {
      return false;
    }
    const auto schedule_it = context.function_base_to_schedule().find(fb);
    if (schedule_it == context.function_base_to_schedule().end()) {
      return false;
    }
    return RunOnFunctionInternal(fb->AsFunctionOrDie(), schedule_it->second,
                                 options, results, context);
  }

  void GcAfterFunctionBaseChange(Package* p,
                                 CodegenContext& context) const override {
    context.GcMetadata();
  }

  // Must be implemented by a subclass to do the logic for the pass and return
  // whether anything was changed.
  virtual absl::StatusOr<bool> RunOnFunctionInternal(
      Function* f, const PipelineSchedule& schedule,
      const CodegenPassOptions& options, PassResults* results,
      CodegenContext& context) const = 0;
};

// Abstract base class for a pass that runs on all procs.
class CodegenProcPass : public ProcPass<CodegenPassOptions, CodegenContext> {
 public:
  using ProcPass::ProcPass;

  virtual void GcAfterProcChange(Package* p,
                                 CodegenContext& context) const override {
    context.GcMetadata();
  }
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_PASS_H_
