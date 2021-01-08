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

#ifndef XLS_CODEGEN_PROC_GENERATOR_H_
#define XLS_CODEGEN_PROC_GENERATOR_H_

#include "absl/status/statusor.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/ir/proc.h"

namespace xls {
namespace verilog {

// Options to pass to the proc generator.
class GeneratorOptions {
 public:
  // Reset logic to use for the pipeline.
  GeneratorOptions& reset(const ResetProto& reset_proto) {
    reset_proto_ = reset_proto;
    return *this;
  }
  GeneratorOptions& reset(absl::string_view name, bool asynchronous,
                          bool active_low) {
    reset_proto_ = ResetProto();
    reset_proto_->set_name(ToProtoString(name));
    reset_proto_->set_asynchronous(asynchronous);
    reset_proto_->set_active_low(active_low);
    return *this;
  }
  const absl::optional<ResetProto>& reset() const { return reset_proto_; }

  // Name of the clock signal.
  GeneratorOptions& clock_name(std::string clock_name) {
    clock_name_ = std::move(clock_name);
    return *this;
  }
  absl::string_view clock_name() const { return clock_name_; }

  // Name to use for the generated module. If not given, the name of the XLS
  // function is used.
  GeneratorOptions& module_name(absl::string_view name) {
    module_name_ = name;
    return *this;
  }
  const std::string module_name() const { return module_name_; }

  // Whether to use SystemVerilog in the generated code otherwise Verilog is
  // used. The default is to use SystemVerilog.
  GeneratorOptions& use_system_verilog(bool value) {
    use_system_verilog_ = value;
    return *this;
  }
  bool use_system_verilog() const { return use_system_verilog_; }

 private:
  absl::optional<ResetProto> reset_proto_;
  std::string clock_name_;
  std::string module_name_;
  bool use_system_verilog_ = true;
};

// Generates and returns a (System)Verilog module implementing the given proc
// with the specified options. The proc must have no explicit state. That is,
// the state type must be an empty tuple. Typically, the state should have
// already been converted to a channel. Nodes in the proc (send/receive) must
// only communicate over RegisterChannels and PortChannels.
absl::StatusOr<ModuleGeneratorResult> GenerateModule(
    Proc* proc, const GeneratorOptions& options);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_PROC_GENERATOR_H_
