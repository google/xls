// Copyright 2024 The XLS Authors
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

#include "xls/dslx/ir_convert/ir_converter_options_flags.h"

#include <filesystem>  // NOLINT
#include <optional>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/ir/channel.pb.h"

// LINT.IfChange
ABSL_FLAG(std::optional<std::string>, output_file, std::nullopt,
          "Where to write the ir file. Defaults to stdout");
ABSL_FLAG(std::optional<std::string>, top, std::nullopt,
          "The name of the top entity. When provided, the function/proc is the "
          "top entity in the generated IR. When not provided, all functions "
          "and procs are converted, there is no top entity defined in the "
          "generated IR.");
ABSL_FLAG(std::string, dslx_stdlib_path, std::string(xls::kDefaultDslxStdlibPath),
          "Path to DSLX standard library files.");
ABSL_FLAG(std::optional<std::string>, dslx_path, std::nullopt,
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(
    std::optional<std::string>, package_name, std::nullopt,
    "Package name to use for output (required when multiple input .x files "
    "are given).");

ABSL_FLAG(bool, emit_fail_as_assert, true,
          "Feature flag for emitting fail!() in the DSL as an assert IR op.");
ABSL_FLAG(bool, convert_tests, false,
          "Feature flag for emitting test procs/functions to IR.");
ABSL_FLAG(bool, verify, true,
          "If true, verifies the generated IR for correctness.");

ABSL_FLAG(std::optional<std::string>, disable_warnings, std::nullopt,
          "Comma-delimited list of warnings to disable -- not generally "
          "recommended, but can be used in exceptional circumstances");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early, as an error, if warnings are detected");
ABSL_FLAG(std::optional<std::string>, interface_proto_file, std::nullopt,
          "File to store a xls.PackageInterfaceProto containing extern type "
          "information and interface specs");
ABSL_FLAG(std::optional<std::string>, interface_textproto_file, std::nullopt,
          "File to store a xls.PackageInterfaceProto containing extern type "
          "information and interface specs in textproto format");
ABSL_FLAG(std::optional<std::string>, ir_converter_options_proto, std::nullopt,
          "Path to a protobuf containing all ir converter options args.");
ABSL_FLAG(std::optional<std::string>, default_fifo_config, std::nullopt,
          "Textproto description of a default FifoConfigProto. If unspecified, "
          "no default FIFO config is specified and codegen may fail.");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)
ABSL_FLAG(std::optional<std::string>, ir_converter_options_used_textproto_file,
          std::nullopt,
          "If present, path to write a protobuf recording all ir converter "
          "args used (including those set on the cmd line).");

namespace xls {
namespace {
absl::StatusOr<bool> SetOptionsFromFlags(IrConverterOptionsFlagsProto& proto) {
  bool any_flags_set = false;

#define POPULATE_FLAG(__x)                                   \
  {                                                          \
    any_flags_set |= FLAGS_##__x.IsSpecifiedOnCommandLine(); \
    proto.set_##__x(absl::GetFlag(FLAGS_##__x));             \
  }
#define POPULATE_OPTIONAL_FLAG(__x)                          \
  {                                                          \
    any_flags_set |= FLAGS_##__x.IsSpecifiedOnCommandLine(); \
    const auto& flag = absl::GetFlag(FLAGS_##__x);           \
    if (flag.has_value()) {                                  \
      proto.set_##__x(*flag);                                \
    }                                                        \
  }
  POPULATE_OPTIONAL_FLAG(output_file);
  POPULATE_OPTIONAL_FLAG(top);
  POPULATE_FLAG(dslx_stdlib_path);
  POPULATE_OPTIONAL_FLAG(dslx_path);
  POPULATE_OPTIONAL_FLAG(package_name);
  POPULATE_FLAG(emit_fail_as_assert);
  POPULATE_FLAG(verify);
  POPULATE_FLAG(convert_tests);
  POPULATE_OPTIONAL_FLAG(disable_warnings);
  POPULATE_FLAG(warnings_as_errors);
  POPULATE_OPTIONAL_FLAG(interface_proto_file);
  POPULATE_OPTIONAL_FLAG(interface_textproto_file);

#undef POPULATE_FLAG

  // Populate default fifo config.
  {
    const std::optional<std::string>& default_fifo_config =
        absl::GetFlag(FLAGS_default_fifo_config);
    if (default_fifo_config.has_value()) {
      any_flags_set = true;

      XLS_RETURN_IF_ERROR(ParseTextProto(*default_fifo_config,
                                         std::filesystem::path("<cmdline arg>"),
                                         proto.mutable_default_fifo_config()));
    }
  }

  return any_flags_set;
}
}  // namespace

absl::StatusOr<IrConverterOptionsFlagsProto> GetIrConverterOptionsFlagsProto() {
  IrConverterOptionsFlagsProto proto;
  XLS_ASSIGN_OR_RETURN(bool any_flags_set, SetOptionsFromFlags(proto));
  if (any_flags_set) {
    if (FLAGS_ir_converter_options_proto.IsSpecifiedOnCommandLine()) {
      return absl::InvalidArgumentError(
          "ir_converter_options_proto and other flags cannot be set at the "
          "same time.");
    }
  } else if (FLAGS_ir_converter_options_proto.IsSpecifiedOnCommandLine()) {
    XLS_RETURN_IF_ERROR(ParseTextProtoFile(
        std::filesystem::path(*absl::GetFlag(FLAGS_ir_converter_options_proto)),
        &proto));
  }
  if (FLAGS_ir_converter_options_used_textproto_file
          .IsSpecifiedOnCommandLine()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        *absl::GetFlag(FLAGS_ir_converter_options_used_textproto_file), proto));
  }
  return proto;
}

}  // namespace xls
