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

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/annotate_types.h"
#include "xls/dslx/extract_module_name.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.pb.h"
#include "xls/dslx/warning_kind.h"

ABSL_FLAG(bool, in_place, false,
          "Whether to modify the input file(s) in-place.");
ABSL_FLAG(bool, i, false, "Alias for --in_place.");

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Annotates DSLX source code with explicit type annotations on let bindings.

Example invocation:
  annotate_types_main path/to/file.x
  annotate_types_main --in_place path/to/file.x
  annotate_types_main --output_file=out.x path/to/file.x
  cat path/to/file.x | annotate_types_main -
)";

absl::Status RealMain(absl::Span<const std::string_view> input_paths,
                      bool in_place) {
  XLS_ASSIGN_OR_RETURN(IrConverterOptionsFlagsProto ir_converter_options,
                       GetIrConverterOptionsFlagsProto());

  std::optional<std::filesystem::path> output_file =
      ir_converter_options.has_output_file()
          ? std::make_optional<std::filesystem::path>(
                ir_converter_options.output_file())
          : std::nullopt;

  if (in_place && output_file.has_value()) {
    return absl::InvalidArgumentError(
        "Cannot specify both --in_place and --output_file.");
  }

  bool has_stdin_arg =
      std::any_of(input_paths.begin(), input_paths.end(),
                  [](std::string_view path) { return path == "-"; });
  std::optional<std::vector<std::string_view>> stdin_input;
  if (has_stdin_arg) {
    if (input_paths.size() != 1) {
      return absl::InvalidArgumentError(
          "Cannot have stdin along with other file arguments.");
    }
    if (in_place) {
      return absl::InvalidArgumentError(
          "Cannot annotate stdin with in-place annotation.");
    }
    stdin_input = std::vector<std::string_view>{"/dev/stdin"};
    input_paths = absl::MakeConstSpan(stdin_input.value());
  }

  if (input_paths.size() > 1 && !in_place) {
    return absl::InvalidArgumentError(
        "Cannot have multiple input files when --in_place is not specified.");
  }

  if (output_file.has_value() && input_paths.size() > 1) {
    return absl::InvalidArgumentError(
        "Cannot specify --output_file with multiple input files.");
  }

  std::string_view dslx_stdlib_path = ir_converter_options.dslx_stdlib_path();
  std::string_view dslx_path = ir_converter_options.dslx_path();
  std::vector<std::string_view> dslx_path_strs = absl::StrSplit(dslx_path, ':');
  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& p : dslx_path_strs) {
    if (!p.empty()) {
      dslx_paths.push_back(std::filesystem::path(p));
    }
  }

  XLS_ASSIGN_OR_RETURN(
      WarningKindSet warnings,
      GetWarningsSetFromFlags(ir_converter_options.enable_warnings(),
                              ir_converter_options.disable_warnings()));

  const AnnotateTypesOptions options = {
      .dslx_stdlib_path = dslx_stdlib_path,
      .dslx_paths = dslx_paths,
      .type_inference_v2 = ir_converter_options.type_inference_v2(),
      .warnings_as_errors = ir_converter_options.warnings_as_errors(),
      .warnings = warnings,
  };

  for (std::string_view input_path : input_paths) {
    std::filesystem::path path(input_path);
    if (in_place && (input_path == "/dev/stdin" || input_path == "-")) {
      return absl::InvalidArgumentError(
          "Cannot annotate stdin with in-place annotation.");
    }

    XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(path));

    std::string module_name = "main";
    if (ir_converter_options.has_package_name()) {
      module_name = ir_converter_options.package_name();
    } else if (input_path != "/dev/stdin" && input_path != "-") {
      absl::StatusOr<std::string> extracted = ExtractModuleName(path);
      if (extracted.ok()) {
        module_name = *extracted;
      }
    }

    XLS_ASSIGN_OR_RETURN(
        std::string annotated,
        AnnotateTypes(contents, module_name, input_path, options));

    if (in_place) {
      auto status = SetFileContentsAtomically(path, annotated);
      if (!status.ok()) {
        // Couldn't do the rename based write (maybe on a tmpfs?). Just do a
        // regular write.
        LOG(WARNING) << "Unable to atomically write to " << path << " due to "
                     << status << ".  Falling back to non-atomic write.";
        XLS_RETURN_IF_ERROR(SetFileContents(path, annotated))
            << "Failed to write to " << path;
      }
    } else if (output_file.has_value()) {
      XLS_RETURN_IF_ERROR(SetFileContents(*output_file, annotated));
    } else {
      std::cout << annotated << std::flush;
    }
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got 0, want "
                << argv[0] << " <input-file>[, ...]";
  }

  bool in_place = absl::GetFlag(FLAGS_in_place) || absl::GetFlag(FLAGS_i);

  return xls::ExitStatus(xls::dslx::RealMain(args, in_place));
}
