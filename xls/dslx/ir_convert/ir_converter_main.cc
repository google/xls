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

#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.pb.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Converts a DSLX input file (and optional entry point) to XLS IR.

Successfully converted XLS IR is printed to stdout; errors are printed to
stderr.

Example invocation for a particular function:

  ir_converter_main -entry best_function path/to/frobulator.x

If no entry point is given all functions within the module are converted:

  ir_converter_main path/to/frobulator.x
)";

absl::Status RealMain(absl::Span<const std::string_view> paths) {
  XLS_ASSIGN_OR_RETURN(IrConverterOptionsFlagsProto ir_converter_options,
                       GetIrConverterOptionsFlagsProto());

  std::optional<std::filesystem::path> output_file =
      ir_converter_options.has_output_file()
          ? std::make_optional<std::filesystem::path>(
                ir_converter_options.output_file())
          : std::nullopt;

  std::string_view dslx_stdlib_path = ir_converter_options.dslx_stdlib_path();
  std::string_view dslx_path = ir_converter_options.dslx_path();
  std::vector<std::string_view> dslx_path_strs = absl::StrSplit(dslx_path, ':');

  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  std::optional<std::string_view> top;
  if (ir_converter_options.has_top()) {
    top = ir_converter_options.top();
  }

  std::optional<std::string_view> package_name;
  if (ir_converter_options.has_package_name()) {
    package_name = ir_converter_options.package_name();
  }

  bool emit_fail_as_assert = ir_converter_options.emit_fail_as_assert();
  bool verify_ir = ir_converter_options.verify();
  bool convert_tests = ir_converter_options.convert_tests();
  bool warnings_as_errors = ir_converter_options.warnings_as_errors();
  XLS_ASSIGN_OR_RETURN(WarningKindSet enabled_warnings,
                       WarningKindSetFromDisabledString(
                           ir_converter_options.disable_warnings()));
  std::optional<FifoConfig> default_fifo_config;
  if (ir_converter_options.has_default_fifo_config()) {
    XLS_ASSIGN_OR_RETURN(
        default_fifo_config,
        FifoConfig::FromProto(ir_converter_options.default_fifo_config()));
  }
  const ConvertOptions convert_options = {
      .emit_positions = true,
      .emit_fail_as_assert = emit_fail_as_assert,
      .verify_ir = verify_ir,
      .warnings_as_errors = warnings_as_errors,
      .enabled_warnings = enabled_warnings,
      .convert_tests = convert_tests,
      .default_fifo_config = default_fifo_config,
  };

  // The following checks are performed inside ConvertFilesToPackage(), but we
  // reproduce them here to give nicer error messages.
  if (!package_name.has_value()) {
    QCHECK_EQ(paths.size(), 1)
        << "-package_name *must* be given when multiple input paths are "
           "supplied";
  }
  if (paths.size() > 1) {
    QCHECK(!top.has_value())
        << "-entry cannot be supplied with multiple input paths (need a single "
           "input path to know where to resolve the entry function)";
  }

  bool printed_error = false;
  XLS_ASSIGN_OR_RETURN(
      PackageConversionData result,
      ConvertFilesToPackage(paths, dslx_stdlib_path, dslx_paths, convert_options,
                            /*top=*/top,
                            /*package_name=*/package_name, &printed_error));
  if (output_file) {
    XLS_RETURN_IF_ERROR(SetFileContents(*output_file, result.DumpIr()));
  } else {
    std::cout << result.package->DumpIr();
  }
  if (ir_converter_options.has_interface_proto_file()) {
    XLS_RETURN_IF_ERROR(
        SetFileContents(ir_converter_options.interface_proto_file(),
                        result.interface.SerializeAsString()));
  }
  if (ir_converter_options.has_interface_textproto_file()) {
    std::string res;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(result.interface, &res));
    XLS_RETURN_IF_ERROR(
        SetFileContents(ir_converter_options.interface_textproto_file(), res));
  }

  if (printed_error) {
    return absl::InternalError(
        "IR conversion failed with an earlier non-fatal error.");
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got " << args.size()
                << ": `" << absl::StrJoin(args, " ") << "`; want " << argv[0]
                << " <input-file>";
  }
  // "-" is a special path that is shorthand for /dev/stdin. Update here as
  // there isn't a better place later.
  for (auto& arg : args) {
    if (arg == "-") {
      arg = "/dev/stdin";
    }
  }

  return xls::ExitStatus(xls::dslx::RealMain(args));
}
