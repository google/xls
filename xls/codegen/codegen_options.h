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

#ifndef XLS_CODEGEN_CODEGEN_OPTIONS_H_
#define XLS_CODEGEN_CODEGEN_OPTIONS_H_

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// Options describing how codegen should be performed.
struct CodegenOptions {
  // The name of the top-level function or proc to generate a Verilog module
  // for. Required.
  // TODO(meheff): 2021/04/21 As this is required, perhaps this should be made a
  // constructor argument.
  CodegenOptions& entry(absl::string_view name);
  absl::optional<absl::string_view> entry() const { return entry_; }

  // Name to use for the generated module. If not given, the name of the XLS
  // function/proc is used.
  CodegenOptions& module_name(absl::string_view name);
  const absl::optional<std::string_view> module_name() const {
    return module_name_;
  }

  // Reset signal to use for any registers with initial values. Required if the
  // proc contains any registers with initial values.
  CodegenOptions& reset(absl::string_view name, bool asynchronous,
                        bool active_low);
  const absl::optional<ResetProto>& reset() const { return reset_proto_; }

  // Name of the clock signal. Required if the proc has any registers.
  CodegenOptions& clock_name(absl::string_view clock_name);
  absl::optional<absl::string_view> clock_name() const { return clock_name_; }

  // Whether to use SystemVerilog in the generated code otherwise Verilog is
  // used. The default is to use SystemVerilog.
  CodegenOptions& use_system_verilog(bool value);
  bool use_system_verilog() const { return use_system_verilog_; }

  // Whether to flop inputs into a register at the beginning of the pipeline. If
  // true, adds a single cycle to the latency of the pipline.
  CodegenOptions& flop_inputs(bool value);
  bool flop_inputs() const { return flop_inputs_; }

  // Whether to flop outputs into a register at the end of the pipeline. If
  // true, adds a single cycle to the latency of the pipline.
  CodegenOptions& flop_outputs(bool value);
  bool flop_outputs() const { return flop_outputs_; }

  // Format string to use when emitting assert operations in the generated
  // Verilog. Supports the following placeholders:
  //  {message}   : Message of the assert operation.
  //  {condition} : Condition of the assert.
  //  {label}     : Label of the assert operation. Returns error if the
  //                operation has no label.
  //  {clk}       : Name of the clock signal. Returns error if no clock is
  //                specified.
  //  {rst}       : Name of the reset signal. Returns error if no reset is
  //                specified.
  //
  // For example, the format string:
  //
  //    '{label}: `MY_ASSERT({condition}, "{message}")'
  //
  // Might result in the following in the emitted Verilog:
  //
  //    my_label: `MY_ASSERT(foo < 8'h42, "Oh noes!");
  CodegenOptions& assert_format(absl::string_view value);
  absl::optional<absl::string_view> assert_format() const {
    return assert_format_;
  }

 private:
  absl::optional<std::string> entry_;
  absl::optional<std::string> module_name_;
  absl::optional<ResetProto> reset_proto_;
  absl::optional<std::string> clock_name_;
  bool use_system_verilog_ = true;
  bool flop_inputs_ = false;
  bool flop_outputs_ = false;
  absl::optional<std::string> assert_format_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_OPTIONS_H_
