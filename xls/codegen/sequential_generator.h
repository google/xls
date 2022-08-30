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

#ifndef XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_
#define XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

// Configuration options for a sequential module.
class SequentialOptions {
 public:
  // Delay estimator to use for loop body pipeline generation.
  SequentialOptions& delay_estimator(const DelayEstimator* delay_estimator) {
    delay_estimator_ = delay_estimator;
    return *this;
  }
  const DelayEstimator* delay_estimator() const { return delay_estimator_; }

  // Name to use for the generated module. If not given, the name is derived
  // from the CountedFor node.
  SequentialOptions& module_name(absl::string_view name) {
    module_name_ = name;
    return *this;
  }
  const std::optional<std::string> module_name() const { return module_name_; }

  // Reset logic to use.
  SequentialOptions& reset(const ResetProto& reset_proto) {
    reset_proto_ = reset_proto;
    return *this;
  }
  const std::optional<ResetProto>& reset() const { return reset_proto_; }

  // Scheduling options for the loop body pipeline.
  SequentialOptions& pipeline_scheduling_options(
      const SchedulingOptions& sched_options) {
    pipeline_scheduling_options_ = sched_options;
    return *this;
  }
  SchedulingOptions& pipeline_scheduling_options() {
    return pipeline_scheduling_options_;
  }
  const SchedulingOptions& pipeline_scheduling_options() const {
    return pipeline_scheduling_options_;
  }

  // Whether to use SystemVerilog in the generated code, otherwise Verilog is
  // used. The default is to use SystemVerilog.
  SequentialOptions& use_system_verilog(bool value) {
    use_system_verilog_ = value;
    return *this;
  }
  bool use_system_verilog() const { return use_system_verilog_; }

 private:
  const DelayEstimator* delay_estimator_ = &GetStandardDelayEstimator();
  std::optional<std::string> module_name_;
  std::optional<ResetProto> reset_proto_;
  SchedulingOptions pipeline_scheduling_options_;
  bool use_system_verilog_ = true;
  // TODO(jbaileyhandle): Interface options.
};

class SequentialModuleBuilder {
 public:
  SequentialModuleBuilder(const SequentialOptions& options,
                          const CountedFor* loop)
      : file_(options.use_system_verilog() ? FileType::kSystemVerilog
                                           : FileType::kVerilog),
        loop_(loop),
        sequential_options_(options) {}

  // Container for logical references to the ports of the sequential module.
  struct PortReferences {
    LogicRef* clk;
    std::vector<LogicRef*> data_in;
    std::vector<LogicRef*> data_out;
    std::optional<LogicRef*> reset;
    std::optional<LogicRef*> ready_in;
    std::optional<LogicRef*> valid_in;
    std::optional<LogicRef*> ready_out;
    std::optional<LogicRef*> valid_out;
  };

  // Container for logical references to strided counter I/O.
  struct StridedCounterReferences {
    // Inputs.
    // When set high, synchronosly sets the counter value to 0.
    LogicRef* set_zero;
    // When set high and set_zero is not high, synchronously adds 'stride' to
    // the counter value.
    LogicRef* increment;

    // Outputs.
    // Holds the current value of the counter.
    LogicRef* value;
    // Driven high iff the counter currently holds the largest allowed value
    // (inclusive).
    LogicRef* holds_max_inclusive_value;
  };

  // Adds the FSM that orchestrates the sequential module's execution.
  absl::Status AddFsm(int64_t pipeline_latency,
                      LogicRef* index_holds_max_inclusive_value,
                      LogicRef* last_pipeline_cycle);

  // Adds a strided counter with statically determined value_limit_exclusive to
  // the module. Note that this is not a saturating counter.
  absl::StatusOr<StridedCounterReferences> AddStaticStridedCounter(
      std::string name, int64_t stride, int64_t value_limit_exclusive,
      LogicRef* clk, LogicRef* set_zero_arg, LogicRef* increment_arg);

  // Assign lhs to rhs (flat bit types only).
  void AddContinuousAssignment(LogicRef* lhs, Expression* rhs) {
    module_builder()->assignment_section()->Add<ContinuousAssignment>(
        SourceInfo(), lhs, rhs);
  }

  // Constructs the sequential module.
  absl::StatusOr<ModuleGeneratorResult> Build();

  // Generates a pipeline module that implements the loop's body.
  absl::StatusOr<std::unique_ptr<ModuleGeneratorResult>>
  GenerateLoopBodyPipeline();

  // Generates the signature for the top-level module.
  absl::StatusOr<std::unique_ptr<ModuleSignature>> GenerateModuleSignature();

  // Initializes the module builder according to the signature.
  absl::Status InitializeModuleBuilder(const ModuleSignature& signature);

  // Accessor methods.
  const VerilogFile* file() const { return &file_; }
  const ModuleGeneratorResult* loop_result() const {
    return loop_body_pipeline_result_.get();
  }
  Module* module() { return module_builder_->module(); }
  ModuleBuilder* module_builder() { return module_builder_.get(); }
  const ModuleSignature* module_signature() const {
    return module_signature_.get();
  }
  const PortReferences* ports() const { return &port_references_; }

 private:
  // Adds all interal logic to the sequential module.
  absl::Status AddSequentialLogic();

  // Declares and assigns a wire, returning a logical reference to the wire.
  LogicRef* DeclareVariableAndAssign(absl::string_view name, Expression* rhs,
                                     int64_t bit_count) {
    LogicRef* wire = module_builder_->DeclareVariable(name, bit_count);
    AddContinuousAssignment(wire, rhs);
    return wire;
  }

  // Instantiates the loop body.
  absl::Status InstantiateLoopBody(
      LogicRef* index_value, const ModuleBuilder::Register& accumulator_reg,
      absl::Span<const ModuleBuilder::Register> invariant_registers,
      LogicRef* pipeline_output);

  VerilogFile file_;
  const CountedFor* loop_;
  std::unique_ptr<ModuleGeneratorResult> loop_body_pipeline_result_;
  std::unique_ptr<ModuleBuilder> module_builder_;
  std::unique_ptr<ModuleSignature> module_signature_;
  absl::flat_hash_map<LogicRef*, Expression*> output_reg_to_assignment_;
  PortReferences port_references_;
  const SequentialOptions sequential_options_;
};

// Emits the given function as a verilog module which reuses the same hardware
// over time to executed loop iterations.
absl::StatusOr<ModuleGeneratorResult> ToSequentialModuleText(Function* func);

// Emits the given CountedFor as a verilog module which reuses the same hardware
// over time to executed loop iterations.
absl::StatusOr<ModuleGeneratorResult> ToSequentialModuleText(
    const SequentialOptions& options, const CountedFor* loop);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_
