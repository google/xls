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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "libs/json11/json11.hpp"
#include "xls/dslx/interp_value.h"

namespace xls {

// Parses a semicolon-delimited list of values.
//
// Example input:
//  bits[32]:6; (bits[8]:2, bits[16]:4)
//
// Returned bits values are always unsigned.
//
// Note: these values are parsed to InterpValues, but really they are just IR
// values that we're converting into InterpValues. Things like enums or structs
// (via named tuples) can't be parsed via this mechanism, it's fairly
// specialized for the scenario we've created in our fuzzing process.
absl::StatusOr<std::vector<dslx::InterpValue>> ParseArgs(
    absl::string_view args_text);

// Parses a batch of arguments, one ParseArgs() per line.
absl::StatusOr<std::vector<std::vector<dslx::InterpValue>>> ParseArgsBatch(
    absl::string_view args_text);

// Returns a string representation of the args_batch.
std::string ArgsBatchToText(
    const std::vector<std::vector<dslx::InterpValue>>& args_batch);

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
  static absl::StatusOr<SampleOptions> FromJson(absl::string_view json_text);

  // Returns a JSON-encoded string describing this object.
  //
  // TODO(https://github.com/google/xls/issues/341): 2021-03-12 Rework to be
  // (schemaful) prototext instead.
  std::string ToJsonText() const { return ToJson().dump(); }

  // As above, but returns the JSON object.
  json11::Json ToJson() const;

  bool input_is_dslx() const { return input_is_dslx_; }
  bool convert_to_ir() const { return convert_to_ir_; }
  bool optimize_ir() const { return optimize_ir_; }
  bool use_jit() const { return use_jit_; }
  bool codegen() const { return codegen_; }
  bool simulate() const { return simulate_; }
  const absl::optional<std::string>& simulator() const { return simulator_; }
  const absl::optional<std::vector<std::string>>& codegen_args() const {
    return codegen_args_;
  }
  bool use_system_verilog() const { return use_system_verilog_; }

  void set_input_is_dslx(bool value) { input_is_dslx_ = value; }
  void set_codegen(bool value) { codegen_ = value; }
  void set_simulate(bool value) { simulate_ = value; }
  void set_codegen_args(const std::vector<std::string>& value) {
    codegen_args_ = value;
  }

  bool operator==(const SampleOptions& other) const {
    return ToJson() == other.ToJson();
  }
  bool operator!=(const SampleOptions& other) const {
    return !((*this) == other);
  }

 private:
  // Whether code sample is DSLX. Otherwise assumed to be XLS IR.
  bool input_is_dslx_ = true;
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
  absl::optional<std::vector<std::string>> codegen_args_;
  // Run the Verilog simulator on the generated Verilog. Requires codegen to be
  // true.
  bool simulate_ = false;
  // Verilog simulator to use; e.g. "iverilog".
  absl::optional<std::string> simulator_;
  // Whether to use SystemVerilog or Verilog in codegen.
  bool use_system_verilog_ = true;
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
  static absl::StatusOr<Sample> Deserialize(absl::string_view s);
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
  std::string ToCrasher(absl::string_view error_message) const;

  Sample(std::string input_text, SampleOptions options,
         std::vector<std::vector<dslx::InterpValue>> args_batch)
      : input_text_(std::move(input_text)),
        options_(std::move(options)),
        args_batch_(std::move(args_batch)) {}

  const SampleOptions& options() const { return options_; }
  const std::string& input_text() const { return input_text_; }
  const std::vector<std::vector<dslx::InterpValue>>& args_batch() const {
    return args_batch_;
  }

  bool operator==(const Sample& other) const {
    return input_text_ == other.input_text_ && options_ == other.options_ &&
           ArgsBatchEqual(other);
  }
  bool operator!=(const Sample& other) const { return !((*this) == other); }

 private:
  // Returns whether the argument batch is the same as in "other".
  bool ArgsBatchEqual(const Sample& other) const;

  std::string input_text_;  // Code sample as text.
  SampleOptions options_;   // How to run the sample.

  // Argument values to use for interpretation and simulation.
  std::vector<std::vector<dslx::InterpValue>> args_batch_;
};

}  // namespace xls

#endif  // XLS_FUZZER_SAMPLE_H_
