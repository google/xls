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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xls {

// Converts the specified DSLX text into XLS IR text.
//
// Args:
//  dslx: DSL (module) text to convert to IR.
//  path: Path to use for source location information for the given text.
//    Since text may be generated, an empty string or a pseudo path like
//    "<generated>" is acceptable.
//  module_name: Name of the DSL module, will be used in the name of the
//    converted IR package text.
//  additional_search_paths: Additional filesystem paths to search for imported
//    modules.
absl::StatusOr<std::string> ConvertDslxToIr(
    absl::string_view dslx, absl::string_view path,
    absl::string_view module_name,
    absl::Span<const std::filesystem::path> additional_search_paths);

// As above, but uses a filesystem path to retrieve the DSLX module contents.
// "path" should end with ".x" suffix, the path will determine the module name.
absl::StatusOr<std::string> ConvertDslxPathToIr(
    std::filesystem::path path,
    absl::Span<const std::filesystem::path> additional_search_paths);

// Optimizes the generated XLS IR with the given entry point (which should be a
// function present inside the IR text).
absl::StatusOr<std::string> OptimizeIr(absl::string_view ir,
                                       absl::string_view entry);

// Mangles the given DSL module/function name combination so it can be resolved
// as a corresponding symbol in converted IR.
absl::StatusOr<std::string> MangleDslxName(absl::string_view module_name,
                                           absl::string_view function_name);

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
absl::StatusOr<std::string> ProtoToDslx(absl::string_view proto_def,
                                        absl::string_view message_name,
                                        absl::string_view text_proto,
                                        absl::string_view binding_name);

}  // namespace xls

#endif  // XLS_PUBLIC_RUNTIME_BUILD_ACTIONS_H_
