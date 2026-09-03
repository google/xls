// Copyright 2026 The XLS Authors
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

// CLI tool to explore execution paths in an XLS IR function using symbolic
// execution and generate path summaries or testvector protos.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/cfg_symex_engine.h"
#include "xls/solvers/symex/concolic_input_spec.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/tests/testvector.pb.h"
#include "z3/src/api/z3.h"
#include "z3/src/api/z3_api.h"

static constexpr std::string_view kUsage = R"(
Explores execution paths in an XLS IR function using symbolic execution,
generating human-readable path summaries and optional testvector textprotos.

Expected invocation:
  symex_main <path/to/design.ir> [--top=<fn>] [--concrete_inputs=op=0]
)";

ABSL_FLAG(std::string, top, "",
          "Entry function to explore. If unspecified, the top function of the "
          "package will be used.");
ABSL_FLAG(std::string, concrete_inputs, "",
          "Comma-separated list of param=value pairs for concrete input "
          "assignments (e.g. 'op=0,a=10').");
ABSL_FLAG(
    std::string, output_path, "-",
    "Path to write human-readable summary output to. '-' denotes stdout.");
ABSL_FLAG(std::string, output_testvector_textproto, "",
          "Optional path to write generated test cases as a "
          "xls::testvector::SampleInputsProto textproto.");
ABSL_FLAG(int64_t, max_paths, 0,
          "Maximum number of paths to explore (0 for unlimited).");

namespace xls {
namespace {

std::string FormatPathsText(
    Function* fn, absl::Span<const solvers::symex::SymbolicPath> paths) {
  std::string output;
  absl::StrAppendFormat(&output,
                        "Explored %d feasible path(s) for function '%s':\n\n",
                        paths.size(), fn->name());
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto& path = paths[i];
    absl::StrAppendFormat(&output, "Path #%d:\n  Inputs:\n", i);
    for (const auto& assignment : path.generated_test) {
      absl::StrAppendFormat(
          &output, "    %s = %s\n", assignment.param->name(),
          assignment.value.ToString(FormatPreference::kDefault));
    }
    auto return_val =
        DropInterpreterEvents(InterpretFunction(fn, path.input_values()));
    if (return_val.ok()) {
      absl::StrAppendFormat(&output, "  Result = %s\n",
                            return_val->ToString(FormatPreference::kDefault));
    }
    if (i + 1 < paths.size()) {
      absl::StrAppend(&output, "\n");
    }
  }
  return output;
}

xls::testvector::SampleInputsProto FormatPathsProto(
    absl::Span<const solvers::symex::SymbolicPath> paths) {
  xls::testvector::SampleInputsProto sample_proto;
  xls::testvector::FunctionArgsProto* function_args =
      sample_proto.mutable_function_args();
  for (const auto& path : paths) {
    function_args->add_args(absl::StrJoin(
        path.input_values(), "; ", [](std::string* out, const Value& v) {
          absl::StrAppend(out, v.ToString(FormatPreference::kDefault));
        }));
  }
  return sample_proto;
}

absl::StatusOr<solvers::symex::ConcolicInputSpec> ParseConcreteInputs(
    std::string_view concrete_inputs_str, Function* fn) {
  solvers::symex::ConcolicInputSpec spec;
  if (concrete_inputs_str.empty()) {
    return spec;
  }
  for (std::string_view pair :
       absl::StrSplit(concrete_inputs_str, ',', absl::SkipEmpty())) {
    if (!absl::StrContains(pair, '=')) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid concrete input format: '", pair,
                       "', expected 'param_name=value' (e.g. 'op=0')"));
    }
    std::pair<std::string_view, std::string_view> kv =
        absl::StrSplit(pair, absl::MaxSplits('=', 1));
    std::string_view param_name = absl::StripAsciiWhitespace(kv.first);
    std::string_view val_str = absl::StripAsciiWhitespace(kv.second);
    if (param_name.empty() || val_str.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid concrete input format: '", pair,
                       "', parameter name and value cannot be empty"));
    }
    XLS_ASSIGN_OR_RETURN(
        Param * param, fn->GetParamByName(param_name),
        _ << "in function '" << fn->name() << "' when binding concrete inputs");
    XLS_ASSIGN_OR_RETURN(Value val,
                         Parser::ParseValue(val_str, param->GetType()),
                         _ << "for parameter '" << param_name
                           << "' in function '" << fn->name() << "'");
    spec.BindParam(param_name, val);
  }
  return spec;
}

absl::Status RealMain(absl::Span<const std::string_view> positional_args) {
  if (positional_args.empty()) {
    return absl::InvalidArgumentError(
        "Missing required input IR file.\n"
        "Expected invocation: symex_main <path/to/ir_file>");
  }
  std::string_view input_path = positional_args[0];

  int64_t max_paths = absl::GetFlag(FLAGS_max_paths);
  if (max_paths < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "--max_paths must be non-negative, got: %d", max_paths));
  }

  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(input_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text, input_path));

  std::string top = absl::GetFlag(FLAGS_top);
  Function* fn = nullptr;
  if (!top.empty()) {
    XLS_ASSIGN_OR_RETURN(fn, package->GetFunction(top));
  } else {
    XLS_ASSIGN_OR_RETURN(fn, package->GetTopAsFunction());
  }

  XLS_ASSIGN_OR_RETURN(
      solvers::symex::ConcolicInputSpec concrete_inputs,
      ParseConcreteInputs(absl::GetFlag(FLAGS_concrete_inputs), fn));

  solvers::symex::SymExOptions options;
  options.concrete_inputs = concrete_inputs;
  if (max_paths > 0) {
    options.max_paths = max_paths;
  }

  Z3_config z3_config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(z3_config);
  Z3_del_config(z3_config);

  std::vector<solvers::symex::SymbolicPath> paths;
  {
    solvers::symex::CfgSymExEngine engine(ctx);
    XLS_ASSIGN_OR_RETURN(paths, engine.ExplorePaths(fn, options));
  }
  Z3_del_context(ctx);

  std::string text_output = FormatPathsText(fn, paths);
  std::string output_path = absl::GetFlag(FLAGS_output_path);
  if (output_path.empty() || output_path == "-") {
    std::cout << text_output;
  } else {
    XLS_RETURN_IF_ERROR(SetFileContents(output_path, text_output));
  }

  std::string proto_output_path =
      absl::GetFlag(FLAGS_output_testvector_textproto);
  if (!proto_output_path.empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(proto_output_path, FormatPathsProto(paths)));
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(kUsage, argc, argv);
  return xls::ExitStatus(xls::RealMain(positional_args));
}
