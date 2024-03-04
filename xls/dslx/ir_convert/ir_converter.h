// Copyright 2020 The XLS Authors
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

#ifndef XLS_DSLX_IR_CONVERT_IR_CONVERTER_H_
#define XLS_DSLX_IR_CONVERT_IR_CONVERTER_H_

#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/package.h"

namespace xls::dslx {

// Converts the contents of a module to IR form.
//
// Args:
//   module: Module to convert.
//   import_data: Contains type information used in conversion.
//
// Returns:
//   The IR package that corresponds to this module.
absl::StatusOr<std::unique_ptr<Package>> ConvertModuleToPackage(
    Module* module, ImportData* import_data, const ConvertOptions& options);

// As above, but the package is provided explicitly (instead of being created).
//
// Package must outlive this function call -- functions from "module" are placed
// inside of it, it may not be nullptr.
absl::Status ConvertModuleIntoPackage(Module* module, ImportData* import_data,
                                      const ConvertOptions& options,
                                      Package* package);

// Wrapper around ConvertModuleToPackage that converts to IR text.
absl::StatusOr<std::string> ConvertModule(Module* module,
                                          ImportData* import_data,
                                          const ConvertOptions& options);

// Converts a single function into its emitted text form.
//
// Note that there may be several IR functions in the result text due to either
// call graph dependencies within the module or constructs that turn into
// multiple functions (e.g. for-loop bodies).
//
// Args:
//   module: Module we're converting a function within.
//   entry_function_name: Entry function used as the root for conversion.
//   import_data: The import data for typechecking, etc.
//   parametric_env: Parametric bindings to use during conversion, if this
//     function is parametric.
//
// Returns an error status that indicates whether the conversion was successful.
// On success there will be a corresponding (built) function inside of
// "package".
//
// Implementation note: creates a temporary IR package based on module's name.
absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, std::string_view entry_function_name,
    ImportData* import_data, const ParametricEnv* parametric_env,
    const ConvertOptions& options);

// As above, but the package is provided explicitly.
//
// Package must outlive this function call -- functions from "module" are placed
// inside of it, it may not be nullptr.
absl::Status ConvertOneFunctionIntoPackage(Module* module,
                                           std::string_view entry_function_name,
                                           ImportData* import_data,
                                           const ParametricEnv* parametric_env,
                                           const ConvertOptions& options,
                                           Package* package);

// Converts DSLX files at paths into a package.
//
// The intent is that ir_converter_main should be a thin wrapper around this,
// and users should be able to generate IR internally much the same way they
// would by hand.
//
// Args:
//   paths: Paths to DSLX files
//   import_data: The import data for typechecking, etc.
//   options: Conversion options.
//   top: Optionally, the name of the top function/proc.
//   package_name: Optionally, the name of the package.
//   printed_error: If a non-null pointer is passes, sets the contents to a
//     boolean value indicating if an error was printed during conversion.
absl::StatusOr<std::unique_ptr<Package>> ConvertFilesToPackage(
    absl::Span<const std::string_view> paths, const std::string& stdlib_path,
    absl::Span<const std::filesystem::path> dslx_paths,
    const ConvertOptions& convert_options,
    std::optional<std::string_view> top = std::nullopt,
    std::optional<std::string_view> package_name = std::nullopt,
    bool* printed_error = nullptr);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_IR_CONVERTER_H_
