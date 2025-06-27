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

#ifndef XLS_PUBLIC_RUNTIME_BUILD_ACTIONS_H_
#define XLS_PUBLIC_RUNTIME_BUILD_ACTIONS_H_

// Exposes XLS functionality at the level of "build actions" (e.g. the kinds of
// things we specify in Bazel BUILD rules), so they can be invoked at runtime by
// XLS consumers who want to exercise Just-in-Time capabilities, e.g. for users
// sweeping a parameterized design space at runtime.
//
// These APIs attempt to be minimal in both their surface area (number of
// routines) and their options (number of parameters), so they can remain mostly
// stable in their signatures, so long as the basic functionality exists.
//
// Note that, just like users should not depending on the precise output of a
// compiler remaining stable, users should not depend on the precise output of
// these actions remaining stable, they will evolve as the XLS system evolves.

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_result.h"
#include "xls/public/ir.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

// Returns the *default* DSLX standard library directory path. This generally
// reflects the stdlib directory path for XLS HEAD.
//
// NOTE: in an environment where XLS is released to release directories, the
// code that uses this API needs to know how to configure itself to point at XLS
// release directories instead via command line configuration; e.g. by taking a
// command line flag option that indicates what DSL stdlib path to use instead.
std::string_view GetDefaultDslxStdlibPath();

struct ConvertDslxToIrOptions {
  std::string_view dslx_stdlib_path;
  absl::Span<const std::filesystem::path> additional_search_paths;
  absl::Span<const std::string_view> enable_warnings;
  absl::Span<const std::string_view> disable_warnings;
  bool warnings_as_errors = true;
  std::vector<std::string>* warnings_out = nullptr;
};

// Converts the specified DSLX text into XLS IR text.
//
// Args:
//  dslx: DSL (module) text to convert to IR.
//  path: Path to use for source location information for the given text.
//    Since text may be generated, an empty string or a pseudo path like
//    "<generated>" is acceptable.
//  module_name: Name of the DSL module, will be used in the name of the
//    converted IR package text.
//  dslx_stdlib_path: Path to the DSLX standard library.
//  additional_search_paths: Additional filesystem paths to search for imported
//    modules.
absl::StatusOr<std::string> ConvertDslxToIr(
    std::string_view dslx, std::string_view path, std::string_view module_name,
    const ConvertDslxToIrOptions& options);

// As above, but uses a filesystem path to retrieve the DSLX module contents.
// "path" should end with ".x" suffix, the path will determine the module name.
absl::StatusOr<std::string> ConvertDslxPathToIr(
    const std::filesystem::path& path, const ConvertDslxToIrOptions& options);

// Optimizes the generated XLS IR with the given top-level entity (e.g.,
// function, proc, etc).
absl::StatusOr<std::string> OptimizeIr(std::string_view ir,
                                       std::string_view top);

// Mangles the given DSL module/function name combination so it can be resolved
// as a corresponding symbol in converted IR.
absl::StatusOr<std::string> MangleDslxName(std::string_view module_name,
                                           std::string_view function_name);

// Converts protocol buffer data into its equivalent DSLX text, as a
// module-level constant.
//
// Note: this currently only supports "standalone" protobuf schemas; i.e. this
// cannot translate schemas that import other `.proto` files. If this limitation
// affects you, please file an issue at `github.com/google/xls/issues`
//
// Args:
//  proto_def: Protobuf schema (e.g. contents of `.proto` file).
//  message_name: Name of the message type (inside the protobuf schema) to emit.
//  text_proto: Protobuf text to translate to DSLX.
//  binding_name: Name of the (const) DSLX binding (i.e. const variable name) to
//    make in the output text.
absl::StatusOr<std::string> ProtoToDslx(std::string_view proto_def,
                                        std::string_view message_name,
                                        std::string_view text_proto,
                                        std::string_view binding_name);

struct ScheduleAndCodegenResult {
  SchedulingResult scheduling_result;
  verilog::CodegenResult codegen_result;
};

// Schedules and codegen a given package.
//
// Args:
//  p: The package to schedule and codegen.
//  scheduling_options_flags_proto: The scheduling params.
//  codegen_flags_proto: The codegen params.
//  with_delay_model: Whether the delay model should be used for codegen.
absl::StatusOr<ScheduleAndCodegenResult> ScheduleAndCodegenPackage(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model);

}  // namespace xls

#endif  // XLS_PUBLIC_RUNTIME_BUILD_ACTIONS_H_
