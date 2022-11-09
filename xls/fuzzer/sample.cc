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

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/fuzzer/scrub_crasher.h"
#include "re2/re2.h"

namespace xls {

using dslx::InterpValue;

// Converts an interpreter value to an argument string -- we use the
// IR-converted hex form of the value.
static std::string ToArgString(const InterpValue& v) {
  return v.ConvertToIr().value().ToString(FormatPreference::kHex);
}

// Converts a list of interpreter values to a string.
static std::string InterpValueListToString(
    const std::vector<InterpValue>& interpv_list) {
  return absl::StrJoin(interpv_list, "; ",
                       [](std::string* out, const InterpValue& v) {
                         absl::StrAppend(out, ToArgString(v));
                       });
}

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

/* static */ absl::StatusOr<SampleOptions> SampleOptions::FromJson(
    std::string_view json_text) {
  std::string err;
  json11::Json parsed = json11::Json::parse(std::string(json_text), err);
  SampleOptions options;

#define HANDLE_BOOL(__name)                           \
  if (!parsed[#__name].is_null()) {                   \
    options.__name##_ = parsed[#__name].bool_value(); \
  }

  HANDLE_BOOL(input_is_dslx);
  HANDLE_BOOL(convert_to_ir);
  HANDLE_BOOL(optimize_ir);
  HANDLE_BOOL(use_jit);
  HANDLE_BOOL(codegen);
  HANDLE_BOOL(simulate);
  HANDLE_BOOL(use_system_verilog);

#undef HANDLE_BOOL

  if (!parsed["codegen_args"].is_null()) {
    std::vector<std::string> codegen_args;
    for (const json11::Json& item : parsed["codegen_args"].array_items()) {
      codegen_args.push_back(item.string_value());
    }
    options.codegen_args_ = std::move(codegen_args);
  }
  if (!parsed["ir_converter_args"].is_null()) {
    std::vector<std::string> ir_converter_args;
    for (const json11::Json& item : parsed["ir_converter_args"].array_items()) {
      ir_converter_args.push_back(item.string_value());
    }
    options.ir_converter_args_ = std::move(ir_converter_args);
  }
  if (!parsed["simulator"].is_null()) {
    options.simulator_ = parsed["simulator"].string_value();
  }
  if (!parsed["timeout_seconds"].is_null()) {
    options.timeout_seconds_ = parsed["timeout_seconds"].int_value();
  }
  if (!parsed["calls_per_sample"].is_null()) {
    options.calls_per_sample_ = parsed["calls_per_sample"].int_value();
  }
  if (!parsed["proc_ticks"].is_null()) {
    options.proc_ticks_ = parsed["proc_ticks"].int_value();
  }
  if (!parsed["top_type"].is_null()) {
    options.top_type_ = static_cast<TopType>(parsed["top_type"].int_value());
  }
  return options;
}

json11::Json SampleOptions::ToJson() const {
  absl::flat_hash_map<std::string, json11::Json> json;

#define HANDLE_BOOL(__name) json[#__name] = __name##_;

  HANDLE_BOOL(input_is_dslx);
  HANDLE_BOOL(convert_to_ir);
  HANDLE_BOOL(optimize_ir);
  HANDLE_BOOL(use_jit);
  HANDLE_BOOL(codegen);
  HANDLE_BOOL(simulate);
  HANDLE_BOOL(use_system_verilog);

#undef HANDLE_BOOL

  if (codegen_args_) {
    json["codegen_args"] = *codegen_args_;
  } else {
    json["codegen_args"] = nullptr;
  }

  if (ir_converter_args_) {
    json["ir_converter_args"] = *ir_converter_args_;
  } else {
    json["ir_converter_args"] = nullptr;
  }

  if (simulator_) {
    json["simulator"] = *simulator_;
  } else {
    json["simulator"] = nullptr;
  }

  if (timeout_seconds_) {
    json["timeout_seconds"] = static_cast<int>(*timeout_seconds_);
  } else {
    json["timeout_seconds"] = nullptr;
  }

  json["calls_per_sample"] = static_cast<int>(calls_per_sample_);

  if (proc_ticks_) {
    json["proc_ticks"] = static_cast<int>(*proc_ticks_);
  } else {
    json["proc_ticks"] = nullptr;
  }
  json["top_type"] = static_cast<int>(top_type_);
  return json11::Json(json);
}

bool Sample::ArgsBatchEqual(const Sample& other) const {
  if (args_batch_.size() != other.args_batch_.size()) {
    return false;
  }
  auto args_equal = [](const std::vector<InterpValue>& lhs,
                       const std::vector<InterpValue>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (int64_t i = 0; i < lhs.size(); ++i) {
      if (!lhs[i].Eq(rhs[i])) {
        return false;
      }
    }
    return true;
  };
  for (int64_t i = 0; i < args_batch_.size(); ++i) {
    if (!args_equal(args_batch_[i], other.args_batch_[i])) {
      return false;
    }
  }
  return true;
}

/* static */ absl::StatusOr<Sample> Sample::Deserialize(std::string_view s) {
  s = absl::StripAsciiWhitespace(s);
  std::optional<SampleOptions> options;
  std::optional<std::vector<std::string>> ir_channel_names = std::nullopt;
  std::vector<std::vector<InterpValue>> args_batch;
  std::vector<std::string_view> input_lines;
  for (std::string_view line : absl::StrSplit(s, '\n')) {
    if (RE2::FullMatch(line, "\\s*//\\s*options:(.*)", &line)) {
      XLS_ASSIGN_OR_RETURN(options, SampleOptions::FromJson(line));
    } else if (RE2::FullMatch(line, "\\s*//\\s*ir_channel_names:(.*)", &line)) {
      ir_channel_names = ParseIrChannelNames(line);
    } else if (RE2::FullMatch(line, "\\s*//\\s*args:(.*)", &line)) {
      XLS_ASSIGN_OR_RETURN(auto args, dslx::ParseArgs(line));
      args_batch.push_back(std::move(args));
    } else {
      input_lines.push_back(line);
    }
  }

  if (!options.has_value()) {
    return absl::InvalidArgumentError(
        "Crasher did not have sample 'options' comment line.");
  }

  std::string input_text = absl::StrJoin(input_lines, "\n");
  return Sample(std::move(input_text), *std::move(options),
                std::move(args_batch), std::move(ir_channel_names));
}

std::string Sample::Serialize() const {
  std::vector<std::string> lines;
  lines.push_back(absl::StrCat("// options: ", options_.ToJsonText()));
  if (ir_channel_names_.has_value()) {
    std::string ir_channel_names_str =
        IrChannelNamesToText(ir_channel_names_.value());
    lines.push_back(
        absl::StrCat("// ir_channel_names: ", ir_channel_names_str));
  }
  for (const std::vector<InterpValue>& args : args_batch_) {
    std::string args_str = InterpValueListToString(args);
    lines.push_back(absl::StrCat("// args: ", args_str));
  }
  std::string header = absl::StrJoin(lines, "\n");
  return absl::StrCat(header, "\n", input_text_, "\n");
}

std::string Sample::ToCrasher(std::string_view error_message) const {
  absl::civil_year_t year =
      absl::ToCivilYear(absl::Now(), absl::TimeZone()).year();
  std::vector<std::string> lines = {
      absl::StrFormat(R"(// Copyright %d The XLS Authors
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
                      year)};
  lines.push_back("// Exception:");
  for (std::string_view line : absl::StrSplit(error_message, '\n')) {
    lines.push_back(absl::StrCat("// ", line));
  }
  // Split the D.N.S string to avoid triggering presubmit checks.
  lines.push_back(std::string("// Issue: DO NOT ") +
                  "SUBMIT Insert link to GitHub issue here.");
  lines.push_back("//");
  std::string header = absl::StrJoin(lines, "\n");
  return ScrubCrasher(absl::StrCat(header, "\n", Serialize()));
}

}  // namespace xls
