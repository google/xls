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

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/dslx/ir_converter.h"
#include "xls/ir/ir_parser.h"
#include "re2/re2.h"

namespace xls {

using dslx::InterpValue;

static absl::StatusOr<InterpValue> InterpValueFromString(absl::string_view s) {
  XLS_ASSIGN_OR_RETURN(Value value, Parser::ParseTypedValue(s));
  return dslx::ValueToInterpValue(value);
}

absl::StatusOr<std::vector<InterpValue>> ParseArgs(
    absl::string_view args_text) {
  args_text = absl::StripAsciiWhitespace(args_text);
  std::vector<InterpValue> args;
  if (args_text.empty()) {
    return args;
  }
  for (absl::string_view piece : absl::StrSplit(args_text, ';')) {
    piece = absl::StripAsciiWhitespace(piece);
    XLS_ASSIGN_OR_RETURN(InterpValue value, InterpValueFromString(piece));
    args.push_back(value);
  }
  return args;
}

absl::StatusOr<std::vector<std::vector<InterpValue>>> ParseArgsBatch(
    absl::string_view args_text) {
  args_text = absl::StripAsciiWhitespace(args_text);
  std::vector<std::vector<InterpValue>> args_batch;
  if (args_text.empty()) {
    return args_batch;
  }
  for (absl::string_view line : absl::StrSplit(args_text, '\n')) {
    XLS_ASSIGN_OR_RETURN(auto args, ParseArgs(line));
    args_batch.push_back(std::move(args));
  }
  return args_batch;
}

std::string ArgsBatchToText(
    const std::vector<std::vector<InterpValue>>& args_batch) {
  return absl::StrJoin(
      args_batch, "\n",
      [](std::string* out, const std::vector<InterpValue>& args) {
        absl::StrAppend(
            out, absl::StrJoin(args, ";",
                               [](std::string* out, const InterpValue& v) {
                                 absl::StrAppend(out, v.ToString());
                               }));
      });
}

/* static */ absl::StatusOr<SampleOptions> SampleOptions::FromJson(
    absl::string_view json_text) {
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
  if (!parsed["simulator"].is_null()) {
    options.simulator_ = parsed["simulator"].string_value();
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

  if (simulator_) {
    json["simulator"] = *simulator_;
  } else {
    json["simulator"] = nullptr;
  }
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

/* static */ absl::StatusOr<Sample> Sample::FromCrasher(absl::string_view s) {
  s = absl::StripAsciiWhitespace(s);
  absl::optional<SampleOptions> options;
  std::vector<std::vector<InterpValue>> args_batch;
  std::vector<absl::string_view> input_lines;
  for (absl::string_view line : absl::StrSplit(s, '\n')) {
    if (RE2::FullMatch(line, "\\s*//\\s*options:(.*)", &line)) {
      XLS_ASSIGN_OR_RETURN(options, SampleOptions::FromJson(line));
    } else if (RE2::FullMatch(line, "\\s*//\\s*args:(.*)", &line)) {
      XLS_ASSIGN_OR_RETURN(auto args, ParseArgs(line));
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
                std::move(args_batch));
}

std::string Sample::ToCrasher(
    absl::optional<absl::string_view> error_message) const {
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
  if (error_message.has_value()) {
    lines.push_back("// Exception:");
    for (absl::string_view line : absl::StrSplit(*error_message, '\n')) {
      lines.push_back(absl::StrCat("// ", line));
    }
    lines.push_back("//");
  }
  // Split the D.N.S string to avoid triggering presubmit checks.
  lines.push_back(std::string("// Issue: DO NOT ") +
                  "SUBMIT Insert link to GitHub issue here.");
  lines.push_back(absl::StrCat("// options: ", options_.ToJsonText()));
  for (const std::vector<InterpValue>& args : args_batch_) {
    std::string args_str =
        absl::StrJoin(args, "; ", [](std::string* out, const InterpValue& v) {
          absl::StrAppend(out, v.ToString());
        });
    lines.push_back(absl::StrCat("// args: ", args_str));
  }
  std::string header = absl::StrJoin(lines, "\n");
  return absl::StrCat(header, "\n", input_text_, "\n");
}

}  // namespace xls
