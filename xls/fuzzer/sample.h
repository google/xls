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
#include "xls/common/proto_adaptor_utils.h"
#include "xls/dslx/interp_value.h"
#include "xls/fuzzer/sample.pb.h"

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

  bool input_is_dslx() const { return proto_.input_is_dslx(); }
  void set_input_is_dslx(bool value) { proto_.set_input_is_dslx(value); }

  fuzzer::SampleType sample_type() const { return proto_.sample_type(); }
  void set_sample_type(fuzzer::SampleType value) {
    proto_.set_sample_type(value);
  }

  std::vector<std::string> ir_converter_args() const {
    return std::vector<std::string>(proto_.ir_converter_args().begin(),
                                    proto_.ir_converter_args().end());
  }
  void set_ir_converter_args(absl::Span<const std::string> args) {
    for (const std::string& arg : args) {
      proto_.add_ir_converter_args(arg);
    }
  }

  bool convert_to_ir() const { return proto_.convert_to_ir(); }
  void set_convert_to_ir(bool value) { proto_.set_convert_to_ir(value); }

  bool optimize_ir() const { return proto_.optimize_ir(); }
  void set_optimize_ir(bool value) { proto_.set_optimize_ir(value); }

  bool use_jit() const { return proto_.use_jit(); }
  void set_use_jit(bool value) { proto_.set_use_jit(value); }

  bool codegen() const { return proto_.codegen(); }
  void set_codegen(bool value) { proto_.set_codegen(value); }

  bool simulate() const { return proto_.simulate(); }
  void set_simulate(bool value) { proto_.set_simulate(value); }

  const std::string& simulator() const { return proto_.simulator(); }
  void set_simulator(std::string_view value) {
    proto_.set_simulator(ToProtoString(value));
  }

  std::vector<std::string> codegen_args() const {
    return std::vector<std::string>(proto_.codegen_args().begin(),
                                    proto_.codegen_args().end());
  }
  void set_codegen_args(absl::Span<const std::string> args) {
    for (const std::string& arg : args) {
      proto_.add_codegen_args(arg);
    }
  }

  bool use_system_verilog() const { return proto_.use_system_verilog(); }
  void set_use_system_verilog(bool value) {
    proto_.set_use_system_verilog(value);
  }

  std::optional<int64_t> timeout_seconds() const {
    return proto_.has_timeout_seconds()
               ? std::optional<int64_t>(proto_.timeout_seconds())
               : std::nullopt;
  }
  void set_timeout_seconds(int64_t value) { proto_.set_timeout_seconds(value); }

  int64_t calls_per_sample() const { return proto_.calls_per_sample(); }
  void set_calls_per_sample(int64_t value) {
    proto_.set_calls_per_sample(value);
  }

  int64_t proc_ticks() const { return proto_.proc_ticks(); }
  void set_proc_ticks(int64_t value) { proto_.set_proc_ticks(value); }

  bool operator==(const SampleOptions& other) const {
    return ToJson() == other.ToJson();
  }
  bool operator!=(const SampleOptions& other) const {
    return !((*this) == other);
  }

  SampleOptions ReplaceInputIsDslx(bool enabled) const {
    auto clone = *this;
    clone.set_input_is_dslx(enabled);
    return clone;
  }

  const fuzzer::SampleOptionsProto& proto() const { return proto_; }

 private:
  static fuzzer::SampleOptionsProto DefaultOptionsProto();

  fuzzer::SampleOptionsProto proto_ = DefaultOptionsProto();
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
