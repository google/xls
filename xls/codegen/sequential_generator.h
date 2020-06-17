// Copyright 2020 Google LLC
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

#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/vast.h"
#include "xls/common/integral_types.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

// Configuration options for a sequential module.
class SequentialOptions {
 public:
  // Reset logic to use.
  SequentialOptions& reset(const ResetProto& reset_proto) {
    reset_proto_ = reset_proto;
    return *this;
  }
  const absl::optional<ResetProto>& reset() const { return reset_proto_; }

  // Name to use for the generated module. If not given, the name is derived
  // from the CountedFor node.
  SequentialOptions& module_name(absl::string_view name) {
    module_name_ = name;
    return *this;
  }
  const absl::optional<std::string> module_name() const { return module_name_; }

  // Whether to use SystemVerilog in the generated code, otherwise Verilog is
  // used. The default is to use SystemVerilog.
  SequentialOptions& use_system_verilog(bool value) {
    use_system_verilog_ = value;
    return *this;
  }
  bool use_system_verilog() const { return use_system_verilog_; }

 private:
  absl::optional<std::string> module_name_;
  absl::optional<ResetProto> reset_proto_;
  bool use_system_verilog_ = true;
  // TODO(jbaileyhandle): Flop ouptut option?
  // TODO(jbaileyhandle): Interface options.
};

class SequentialModuleBuilder {
 public:
  SequentialModuleBuilder(const SequentialOptions& options,
                          const CountedFor* loop)
      : loop_(loop), sequential_options_(options) {}

  // Container for logical references to the ports of the sequential module.
  struct PortReferences {
    LogicRef* clk;
    std::vector<LogicRef*> data_in;
    std::vector<LogicRef*> data_out;
    absl::optional<LogicRef*> reset;
    absl::optional<LogicRef*> ready_in;
    absl::optional<LogicRef*> valid_in;
    absl::optional<LogicRef*> ready_out;
    absl::optional<LogicRef*> valid_out;
  };

  // Generates a pipeline module that implements the loop's body.
  xabsl::StatusOr<std::unique_ptr<ModuleGeneratorResult>>
  GenerateLoopBodyPipeline(const SchedulingOptions& scheduling_options,
                           const DelayEstimator& = GetStandardDelayEstimator());

  // Generates the signature for the top-level module.
  xabsl::StatusOr<std::unique_ptr<ModuleSignature>> GenerateModuleSignature();

  // Initializes the module builder according to the signature.
  absl::Status InitializeModuleBuilder(const ModuleSignature& signature);

  // Accessor methods.
  const VerilogFile* file() const { return &file_; }
  const ModuleGeneratorResult* loop_result() const {
    return loop_body_pipeline_result_.get();
  }
  const Module* module() const { return module_builder_->module(); }
  const ModuleSignature* module_signature() const {
    return module_signature_.get();
  }
  const PortReferences* port_references() const { return &port_references_; }

 private:
  VerilogFile file_;
  const CountedFor* loop_;
  std::unique_ptr<ModuleGeneratorResult> loop_body_pipeline_result_;
  std::unique_ptr<ModuleBuilder> module_builder_;
  std::unique_ptr<ModuleSignature> module_signature_;
  PortReferences port_references_;
  const SequentialOptions sequential_options_;
};

// Emits the given function as a verilog module which reuses the same hardware
// over time to executed loop iterations.
xabsl::StatusOr<ModuleGeneratorResult> ToSequentialModuleText(Function* func);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_
