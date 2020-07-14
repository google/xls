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

#ifndef XLS_CODEGEN_PIPELINE_GENERATOR_H_
#define XLS_CODEGEN_PIPELINE_GENERATOR_H_

#include <string>

#include "absl/types/optional.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/name_to_bit_count.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

// Class describing the options passed to the pipeline generator.
class PipelineOptions {
 public:
  // Reset logic to use for the pipeline.
  PipelineOptions& reset(const ResetProto& reset_proto) {
    reset_proto_ = reset_proto;
    return *this;
  }
  const absl::optional<ResetProto>& reset() const { return reset_proto_; }

  // Name to use for the generated module. If not given, the name of the XLS
  // function is used.
  PipelineOptions& module_name(absl::string_view name) {
    module_name_ = name;
    return *this;
  }
  const absl::optional<std::string> module_name() const { return module_name_; }

  // Specifies manual pipeline register load-enable control.
  PipelineOptions& manual_control(absl::string_view input_name);
  absl::optional<ManualPipelineControl> manual_control() const;

  // Specifies pipeline register load-enable controlled by a valid signal.
  PipelineOptions& valid_control(absl::string_view input_name,
                                 absl::optional<absl::string_view> output_name);
  absl::optional<ValidProto> valid_control() const;

  // Returns the proto describing the pipeline control scheme.
  const absl::optional<PipelineControl>& control() const {
    return pipeline_control_;
  }

  // Whether to use SystemVerilog in the generated code otherwise Verilog is
  // used. The default is to use SystemVerilog.
  PipelineOptions& use_system_verilog(bool value);
  bool use_system_verilog() const { return use_system_verilog_; }

  // Whether to flop inputs into a register at the beginning of the pipeline. If
  // true, adds a single cycle to the latency of the pipline.
  PipelineOptions& flop_inputs(bool value);
  bool flop_inputs() const { return flop_inputs_; }

  // Whether to flop outputs into a register at the end of the pipeline. If
  // true, adds a single cycle to the latency of the pipline.
  PipelineOptions& flop_outputs(bool value);
  bool flop_outputs() const { return flop_outputs_; }

 private:
  absl::optional<std::string> module_name_;
  absl::optional<ResetProto> reset_proto_;
  absl::optional<PipelineControl> pipeline_control_;
  bool use_system_verilog_ = true;
  bool flop_inputs_ = true;
  bool flop_outputs_ = true;
};

// Emits the given function as a verilog module which follows the given
// schedule. The module is pipelined with a latency and initiation interval
// given in the signature.
xabsl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const PipelineOptions& options = PipelineOptions());

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_PIPELINE_GENERATOR_H_
