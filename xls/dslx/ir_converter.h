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

#ifndef XLS_DSLX_IR_CONVERTER_H_
#define XLS_DSLX_IR_CONVERTER_H_

#include <memory>

#include "absl/container/btree_set.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {

// Bundles together options (common among the API routines below) used in
// DSLX-to-IR conversion.
struct ConvertOptions {
  // Whether to emit positional metadata into the output IR.
  //
  // Stripping positions can be useful for less fragile string matching in
  // development, e.g. tests.
  bool emit_positions = true;

  // Whether to emit fail!() operations as predicated assertion IR nodes.
  bool emit_fail_as_assert = true;

  // Should the generated IR be verified?
  bool verify_ir = true;
};

// Converts the contents of a module to IR form.
//
// Args:
//   module: Module to convert.
//   import_data: Contains type information used in conversion.
//   traverse_tests: Whether to convert functions called in DSLX test
//   constructs.
//     Note that this does NOT convert the test constructs themselves.
//
// Returns:
//   The IR package that corresponds to this module.
absl::StatusOr<std::unique_ptr<Package>> ConvertModuleToPackage(
    Module* module, ImportData* import_data, const ConvertOptions& options,
    bool traverse_tests = false);

// As above, but the package is provided explicitly (instead of being created).
//
// Package must outlive this function call -- functions from "module" are placed
// inside of it, it may not be nullptr.
absl::Status ConvertModuleIntoPackage(Module* module, ImportData* import_data,
                                      const ConvertOptions& options,
                                      bool traverse_tests, Package* package);

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
//   symbolic_bindings: Parametric bindings to use during conversion, if this
//     function is parametric.
//
// Returns an error status that indicates whether the conversion was successful.
// On success there will be a corresponding (built) function inside of
// "package".
//
// Implementation note: creates a temporary IR package based on module's name.
absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, std::string_view entry_function_name,
    ImportData* import_data, const SymbolicBindings* symbolic_bindings,
    const ConvertOptions& options);

// As above, but the package is provided explicitly.
//
// Package must outlive this function call -- functions from "module" are placed
// inside of it, it may not be nullptr.
absl::Status ConvertOneFunctionIntoPackage(
    Module* module, std::string_view entry_function_name,
    ImportData* import_data, const SymbolicBindings* symbolic_bindings,
    const ConvertOptions& options, Package* package);

// Converts an interpreter value to an IR value.
absl::StatusOr<Value> InterpValueToValue(const InterpValue& v);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERTER_H_
