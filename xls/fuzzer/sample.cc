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

#include "xls/fuzzer/sample.h"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/time/civil_time.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/fuzzer/sample.pb.h"
#include "xls/fuzzer/scrub_crasher.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/tests/testvector.pb.h"

namespace xls {

using ::xls::dslx::InterpValue;

namespace {

// Markers for the start/end of the text serialization of the CrasherConfigProto
// in the crasher text.
const char kStartConfig[] = "BEGIN_CONFIG";
const char kEndConfig[] = "END_CONFIG";

// Converts an interpreter value to an argument string -- we use the
// IR-converted hex form of the value.
std::string ToArgString(const InterpValue& v) {
  return v.ConvertToIr().value().ToString(FormatPreference::kHex);
}

// Converts a list of interpreter values to a string.
std::string InterpValueListToString(
    const std::vector<InterpValue>& interpv_list) {
  return absl::StrJoin(interpv_list, "; ",
                       [](std::string* out, const InterpValue& v) {
                         absl::StrAppend(out, ToArgString(v));
                       });
}

}  // namespace

std::string ArgsBatchToText(
    const std::vector<std::vector<InterpValue>>& args_batch) {
  return absl::StrJoin(
      args_batch, "\n",
      [](std::string* out, const std::vector<InterpValue>& args) {
        absl::StrAppend(out, InterpValueListToString(args));
      });
}

std::string IrChannelNamesToText(
    const std::vector<std::string>& ir_channel_names) {
  return absl::StrJoin(ir_channel_names, ", ");
}

std::vector<std::string> ParseIrChannelNames(
    std::string_view ir_channel_names_text) {
  std::vector<std::string> ir_channel_names;
  ir_channel_names =
      absl::StrSplit(ir_channel_names_text, ',', absl::SkipWhitespace());
  for (std::string& ir_channel_name : ir_channel_names) {
    absl::StripAsciiWhitespace(&ir_channel_name);
  }
  return ir_channel_names;
}

/* static */ absl::StatusOr<SampleOptions> SampleOptions::FromPbtxt(
    std::string_view text) {
  fuzzer::SampleOptionsProto proto;
  XLS_RETURN_IF_ERROR(ParseTextProto(text,
                                     /*file_name=*/"", &proto));
  return FromProto(proto);
}

/* static */ absl::StatusOr<SampleOptions> SampleOptions::FromProto(
    fuzzer::SampleOptionsProto proto) {
  SampleOptions options;
  options.proto_ = std::move(proto);
  return options;
}

bool SampleOptions::operator==(const SampleOptions& other) const {
  google::protobuf::util::MessageDifferencer df;
  return df.Compare(proto_, other.proto_);
}

/* static */ fuzzer::SampleOptionsProto SampleOptions::DefaultOptionsProto() {
  fuzzer::SampleOptionsProto proto;
  proto.set_input_is_dslx(true);
  proto.set_sample_type(fuzzer::SAMPLE_TYPE_FUNCTION);
  proto.set_convert_to_ir(true);
  proto.set_optimize_ir(true);
  proto.set_use_jit(true);
  proto.set_codegen(false);
  proto.set_codegen_ng(false);
  proto.set_simulate(false);
  proto.set_use_system_verilog(true);
  proto.set_calls_per_sample(1);
  // TODO(https://github.com/google/xls/issues/1140): Remove when fixed.
  auto* pipeline = proto.add_known_failure();
  pipeline->set_tool(".*codegen_main");
  pipeline->set_stderr_regex(
      ".*Impossible to schedule proc .* as specified; cannot achieve the "
      "specified pipeline length.*");
  // TODO(https://github.com/google/xls/issues/1141): Remove when fixed.
  auto* throughput = proto.add_known_failure();
  throughput->set_tool(".*codegen_main");
  throughput->set_stderr_regex(
      ".*Impossible to schedule proc .* as specified; cannot achieve full "
      "throughput.*");
  return proto;
}

bool AbslParseFlag(std::string_view text, SampleOptions* sample_options,
                   std::string* error) {
  std::string unescaped_text;
  if (!absl::Base64Unescape(text, &unescaped_text)) {
    *error = "Could not parse as a SampleOptions; not a Base64 encoded string?";
    return false;
  }
  fuzzer::SampleOptionsProto proto;
  if (!proto.ParseFromString(unescaped_text)) {
    *error =
        "Could not parse as a SampleOptions; not a serialized "
        "SampleOptionsProto?";
    return false;
  }
  absl::StatusOr<SampleOptions> result =
      SampleOptions::FromProto(std::move(proto));
  if (!result.ok()) {
    *error = result.status().ToString();
    return false;
  }
  *sample_options = *std::move(result);
  return true;
}

std::string AbslUnparseFlag(const SampleOptions& sample_options) {
  return absl::Base64Escape(sample_options.proto().SerializeAsString());
}

// Legacy constructor.
Sample::Sample(std::string input_text, SampleOptions options,
               const std::vector<std::vector<dslx::InterpValue>>& args_batch,
               const std::vector<std::string>& ir_channel_names)
    : input_text_(std::move(input_text)), options_(std::move(options)) {
  if (options_.IsFunctionSample()) {
    testvector::FunctionArgsProto* args_proto =
        testvector_.mutable_function_args();
    for (const std::vector<InterpValue>& args : args_batch) {
      args_proto->add_args(InterpValueListToString(args));
    }
  } else {
    QCHECK(options_.IsProcSample());
    testvector::ChannelInputsProto* inputs_proto =
        testvector_.mutable_channel_inputs();
    for (int64_t i = 0; i < ir_channel_names.size(); ++i) {
      testvector::ChannelInputProto* input_proto = inputs_proto->add_inputs();
      input_proto->set_channel_name(ir_channel_names[i]);
      for (const std::vector<InterpValue>& args : args_batch) {
        input_proto->add_values(ToArgString(args[i]));
      }
    }
  }
}

absl::Status Sample::GetArgsAndChannels(
    std::vector<std::vector<dslx::InterpValue>>& args_batch,
    std::vector<std::string>* ir_channel_names) const {
  return ExtractArgsBatch(options_, testvector_, args_batch, ir_channel_names);
}

// Extract args batch from SampleInputsProto. If to be interpreted as
// proc_samples, also extract "ir_channel_names" (which must not be a nullptr
// then).
/* static */ absl::Status Sample::ExtractArgsBatch(
    const SampleOptions& options,
    const testvector::SampleInputsProto& testvector,
    std::vector<std::vector<InterpValue>>& args_batch,
    std::vector<std::string>* ir_channel_names) {
  // In the serialization channel inputs are grouped by channel, but the
  // fuzzer expects inputs to be grouped by input number.
  // TODO(meheff): Change the fuzzer to accept inputs grouped by channel. This
  // would enable a different number of inputs per channel.
  if (options.IsProcSample()) {
    XLS_RET_CHECK(!testvector.has_function_args());  // proc samples expected
    XLS_RET_CHECK(ir_channel_names != nullptr);
    for (const testvector::ChannelInputProto& channel_input :
         testvector.channel_inputs().inputs()) {
      ir_channel_names->push_back(channel_input.channel_name());
      for (int i = 0; i < channel_input.values().size(); ++i) {
        const std::string& value_str = channel_input.values(i);
        XLS_ASSIGN_OR_RETURN(Value value, Parser::ParseTypedValue(value_str));
        XLS_ASSIGN_OR_RETURN(InterpValue interp_value,
                             dslx::ValueToInterpValue(value));
        if (args_batch.size() <= i) {
          args_batch.resize(i + 1);
        }
        args_batch[i].push_back(interp_value);
      }
    }
    // As corner case, there is the expectation that without channels, the
    // args_batch contains the number of empty clock ticks.
    if (testvector.channel_inputs().inputs().empty()) {
      args_batch.resize(options.proc_ticks());
    }
    // TODO(hzeller): maybe XLS_RET_CHECK() if sum of args + holdoffs == ticks
    return absl::OkStatus();
  }

  // Otherwise just extract function information.
  XLS_RET_CHECK(!testvector.has_channel_inputs());  // function samples expected
  for (const std::string& arg : testvector.function_args().args()) {
    XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args, dslx::ParseArgs(arg));
    args_batch.push_back(args);
  }

  return absl::OkStatus();
}

bool Sample::TestVectorEqual(const testvector::SampleInputsProto& tv) const {
  return google::protobuf::util::MessageDifferencer::Equals(testvector_, tv);
}

/* static */ absl::StatusOr<Sample> Sample::Deserialize(std::string_view s) {
  bool in_config = false;
  std::vector<std::string_view> config_lines;
  std::vector<std::string_view> dslx_lines;
  for (std::string_view line : absl::StrSplit(s, '\n')) {
    std::string_view stripped_line = absl::StripAsciiWhitespace(line);
    if (stripped_line.empty()) {
      continue;
    }
    if (absl::StartsWith(stripped_line, "//")) {
      std::string_view contents =
          absl::StripAsciiWhitespace(absl::StripPrefix(stripped_line, "//"));
      if (contents == kStartConfig) {
        in_config = true;
      } else if (contents == kEndConfig) {
        in_config = false;
      } else if (in_config && !contents.empty() && contents[0] != '#') {
        config_lines.push_back(contents);
      }
    } else {
      dslx_lines.push_back(line);
    }
  }
  if (config_lines.empty()) {
    return absl::InvalidArgumentError(
        "Fuzz sample has a missing or empty config");
  }
  fuzzer::CrasherConfigurationProto proto;
  // Join lines with whitespace, not newline, as TextProto will have issues
  // if string-literals are separated by newline.
  // (Might happen due to file-formatting).
  XLS_RETURN_IF_ERROR(ParseTextProto(absl::StrJoin(config_lines, " "),
                                     /*file_name=*/"", &proto));
  XLS_ASSIGN_OR_RETURN(SampleOptions options,
                       SampleOptions::FromProto(proto.sample_options()));

  // Make sure we see the kind of inputs we expect.
  XLS_RET_CHECK_EQ(proto.inputs().has_function_args(),
                   options.IsFunctionSample());

  std::string dslx_code = absl::StrJoin(dslx_lines, "\n");
  return Sample(dslx_code, options, proto.inputs());
}

std::string Sample::Serialize(
    std::optional<std::string_view> error_message) const {
  std::vector<std::string> lines;
  lines.push_back(absl::StrFormat("// %s", kStartConfig));
  lines.push_back("// # proto-message: xls.fuzzer.CrasherConfigurationProto");

  fuzzer::CrasherConfigurationProto config;
  if (error_message.has_value()) {
    config.set_exception(ToProtoString(error_message.value()));
  }
  // Split the D.N.S string to avoid triggering presubmit checks.
  config.set_issue(std::string("DO NOT ") +
                   "SUBMIT Insert link to GitHub issue here.");
  *config.mutable_sample_options() = options().proto();
  *config.mutable_inputs() = testvector_;

  std::string config_text;
  CHECK(google::protobuf::TextFormat::PrintToString(config, &config_text));
  for (std::string_view line : absl::StrSplit(config_text, '\n')) {
    lines.push_back(absl::StrFormat("// %s", line));
  }
  lines.push_back(absl::StrFormat("// %s", kEndConfig));

  std::string header = absl::StrJoin(lines, "\n");
  return absl::StrCat(header, "\n", input_text_, "\n");
}

std::string Sample::ToCrasher(std::string_view error_message) const {
  absl::civil_year_t year =
      absl::ToCivilYear(absl::Now(), absl::TimeZone()).year();
  std::string license = absl::StrFormat(R"(// Copyright %d The XLS Authors
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
)",
                                        year);

  return ScrubCrasher(absl::StrCat(license, Serialize(error_message)));
}

std::ostream& operator<<(std::ostream& os, const Sample& sample) {
  os << sample.Serialize();
  return os;
}

}  // namespace xls
