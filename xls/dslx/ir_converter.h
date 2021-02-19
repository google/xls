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
#include "xls/dslx/builtins.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace xls::dslx {

// Returns the mangled name of function with the given parametric bindings.
absl::StatusOr<std::string> MangleDslxName(
    absl::string_view function_name,
    const absl::btree_set<std::string>& free_keys, Module* module,
    const SymbolicBindings* symbolic_bindings = nullptr);

// Converts the contents of a module to IR form.
//
// Args:
//   module: Module to convert.
//   type_info: Concrete type information used in conversion.
//   emit_positions: Whether to emit positional metadata into the output IR.
//   traverse_tests: Whether to convert functions called in DSLX test
//   constructs.
//     Note that this does NOT convert the test constructs themselves.
//
// Returns:
//   The IR package that corresponds to this module.
absl::StatusOr<std::unique_ptr<xls::Package>> ConvertModuleToPackage(
    Module* module, TypeInfo* type_info, bool emit_positions = true,
    bool traverse_tests = false);

// Wrapper around ConvertModuleToPackage that converts to IR text.
absl::StatusOr<std::string> ConvertModule(Module* module, TypeInfo* type_info,
                                          bool emit_positions = true);

// Converts a single function into its emitted text form.
//
// Args:
//   module: Module we're converting a function within.
//   entry_function_name: Entry function used as the root for conversion.
//   type_info: Type information about module from the typechecking phase.
//   symbolic_bindings: Parametric bindings to use during conversion, if this
//     function is parametric.
//   emit_positions: Whether to emit position information into the IR based on
//     the AST's source positions.
//
// Returns an error status that indicates whether the conversion was successful.
// On success there will be a corresponding (built) function inside of
// "package".
//
// Implementation note: creates a temporary IR package based on module's name.
absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, absl::string_view entry_function_name, TypeInfo* type_info,
    const SymbolicBindings* symbolic_bindings, bool emit_positions);

// Converts an interpreter value to an IR value.
absl::StatusOr<Value> InterpValueToValue(const InterpValue& v);

// Converts an (IR) value to an interpreter value.
absl::StatusOr<InterpValue> ValueToInterpValue(
    const Value& v, const ConcreteType* type = nullptr);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERTER_H_
