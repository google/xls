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

#ifndef XLS_FUZZER_SAMPLE_H_
#define XLS_FUZZER_SAMPLE_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "libs/json11/json11.hpp"
#include "xls/dslx/interp_value.h"

namespace xls {

// Returns a string representation of the args_batch.
std::string ArgsBatchToText(
    const std::vector<std::vector<dslx::InterpValue>>& args_batch);
// Returns a string representation of the ir_channel_names.
std::string IrChannelNamesToText(
    const std::vector<std::string>& ir_channel_names);
// Returns a list of ir channel names.
std::vector<std::string> ParseIrChannelNames(
    std::string_view ir_channel_names_text);

enum class TopType : int {
  kFunction = 0,
  kProc,
  // TODO(vmirian): 10-7-2022 Add block support.
};

// Options describing how to run a code sample. See member comments for details.
class SampleOptions {
 public:
  // Parses a JSON-encoded string and returns a SampleOptions object.
  //
  // Recognized field keys / expected types are those in the members of
  // SampleOptions.
  //
  // TODO(https://github.com/google/xls/issues/341): 2021-03-12 Rework to be
  // (schemaful) prototext instead.
  static absl::StatusOr<SampleOptions> FromJson(std::string_view json_text);

  // Returns a JSON-encoded string describing this object.
  //
  // TODO(https://github.com/google/xls/issues/341): 2021-03-12 Rework to be
  // (schemaful) prototext instead.
  std::string ToJsonText() const { return ToJson().dump(); }

  // As above, but returns the JSON object.
  json11::Json ToJson() const;

  bool input_is_dslx() const { return input_is_dslx_; }
  TopType top_type() const { return top_type_; }

  const std::optional<std::vector<std::string>>& ir_converter_args() const {
    return ir_converter_args_;
  }
  bool convert_to_ir() const { return convert_to_ir_; }
  bool optimize_ir() const { return optimize_ir_; }
  bool use_jit() const { return use_jit_; }
  bool codegen() const { return codegen_; }
  bool simulate() const { return simulate_; }
  const std::optional<std::string>& simulator() const { return simulator_; }
  const std::optional<std::vector<std::string>>& codegen_args() const {
    return codegen_args_;
  }
  bool use_system_verilog() const { return use_system_verilog_; }
  std::optional<int64_t> timeout_seconds() const { return timeout_seconds_; }
  int64_t calls_per_sample() const { return calls_per_sample_; }
  std::optional<int64_t> proc_ticks() const { return proc_ticks_; }

  void set_input_is_dslx(bool value) { input_is_dslx_ = value; }
  void set_top_type(TopType value) { top_type_ = value; }
  void set_ir_converter_args(const std::vector<std::string>& value) {
    ir_converter_args_ = value;
  }
  void set_codegen(bool value) { codegen_ = value; }
  void set_simulate(bool value) { simulate_ = value; }
  void set_codegen_args(const std::vector<std::string>& value) {
    codegen_args_ = value;
  }
  void set_use_system_verilog(bool value) { use_system_verilog_ = value; }
  void set_timeout_seconds(int64_t value) { timeout_seconds_ = value; }
  void set_calls_per_sample(int64_t value) { calls_per_sample_ = value; }
  void set_proc_ticks(int64_t value) { proc_ticks_ = value; }

  bool operator==(const SampleOptions& other) const {
    return ToJson() == other.ToJson();
  }
  bool operator!=(const SampleOptions& other) const {
    return !((*this) == other);
  }

 private:
  // Whether code sample is DSLX. Otherwise assumed to be XLS IR.
  bool input_is_dslx_ = true;
  // The type of the top.
  TopType top_type_ = TopType::kFunction;
  // Arguments to pass to ir_converter_main. Requires input_is_dslx_ to be true.
  std::optional<std::vector<std::string>> ir_converter_args_;
  // Convert the input code sample to XLS IR. Only meaningful if input_is_dslx
  // is true.
  bool convert_to_ir_ = true;
  // Optimize the XLS IR.
  bool optimize_ir_ = true;
  // Use LLVM JIT when evaluating the XLS IR.
  //
  // TODO(leary): 2021-03-16 Currently we run the unopt IR interpretation
  // unconditionally, and the opt IR interpretation conditionally. Should we
  // also run the opt IR interpretation unconditionally?
  bool use_jit_ = true;
  // Generate Verilog from the optimized IR. Requires optimize_ir to be true.
  bool codegen_ = false;
  // Arguments to pass to codegen_main. Requires codegen to be true.
  std::optional<std::vector<std::string>> codegen_args_;
  // Run the Verilog simulator on the generated Verilog. Requires codegen to be
  // true.
  bool simulate_ = false;
  // Verilog simulator to use; e.g. "iverilog".
  std::optional<std::string> simulator_;
  // Whether to use SystemVerilog or Verilog in codegen.
  bool use_system_verilog_ = true;
  // The timeout value in seconds when executing a subcommand (e.g.,
  // opt_main). This is a per-subcommand invocation timeout *NOT* a timeout
  // value for the entire sample run.
  std::optional<int64_t> timeout_seconds_;
  // Number of times to invoke the generated function.
  int64_t calls_per_sample_ = 1;
  // Number ticks to execute the generated proc.
  std::optional<int64_t> proc_ticks_;
};

// Abstraction describing a fuzzer code sample and how to run it.
class Sample {
 public:
  // Serializes/deserializes a sample to/from a text representation. Used for
  // pickling/unpickling for use in Python. ToCrasher includes this
  // serialization as a substring.
  // TODO(meheff): 2021-03-19 Remove this when we no longer need to
  // pickle/depickle Samples for Python. Deserialize can be replaced with a
  // method FromCrasher.
  static absl::StatusOr<Sample> Deserialize(std::string_view s);
  std::string Serialize() const;

  // Returns "crasher" text serialization.
  //
  // A crasher is a text serialization of the sample along with a copyright
  // message and the error message from the crash in the comments. As such, it
  // is a valid text serialization of the sample. Crashers enable easy
  // reproduction from a single text file. Crashers may be checked in as tests
  // in `xls/fuzzer/crashers/`.
  //
  // A crasher has the following format:
  //  // <copyright notice>
  //  // <error message>
  //  // options: <JSON-serialized SampleOptions>
  //  // args: <argument set 0>
  //  // ...
  //  // args: <argument set 1>
  //  <code sample>
  std::string ToCrasher(std::string_view error_message) const;

  Sample(
      std::string input_text, SampleOptions options,
      std::vector<std::vector<dslx::InterpValue>> args_batch,
      std::optional<std::vector<std::string>> ir_channel_names = std::nullopt)
      : input_text_(std::move(input_text)),
        options_(std::move(options)),
        args_batch_(std::move(args_batch)),
        ir_channel_names_(std::move(ir_channel_names)) {}

  const SampleOptions& options() const { return options_; }
  const std::string& input_text() const { return input_text_; }
  const std::vector<std::vector<dslx::InterpValue>>& args_batch() const {
    return args_batch_;
  }
  const std::optional<std::vector<std::string>>& ir_channel_names() const {
    return ir_channel_names_;
  }

  bool operator==(const Sample& other) const {
    return input_text_ == other.input_text_ && options_ == other.options_ &&
           ArgsBatchEqual(other) &&
           ir_channel_names_ == other.ir_channel_names_;
  }
  bool operator!=(const Sample& other) const { return !((*this) == other); }

 private:
  // Returns whether the argument batch is the same as in "other".
  bool ArgsBatchEqual(const Sample& other) const;

  std::string input_text_;  // Code sample as text.
  SampleOptions options_;   // How to run the sample.

  // Argument values to use for interpretation and simulation.
  std::vector<std::vector<dslx::InterpValue>> args_batch_;
  // Channel names as they appear in the IR.
  std::optional<std::vector<std::string>> ir_channel_names_;
};

}  // namespace xls

#endif  // XLS_FUZZER_SAMPLE_H_
