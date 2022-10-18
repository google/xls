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

#ifndef XLS_DSLX_TYPECHECK_H_
#define XLS_DSLX_TYPECHECK_H_

#include <filesystem>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/deduce_ctx.h"

namespace xls::dslx {

using TopNode =
    std::variant<Function*, Proc*, TestFunction*, StructDef*, TypeDef*>;

// Assigns concrete types and validates such on all elements of `f`, which must
// be a non-parametric user-defined function.
absl::Status CheckFunction(Function* f, DeduceCtx* ctx);

// Assigns concrete types and validates such on all elements of the function
// "invoked" by the given invocation. The target function may be builtin and/or
// parametric.
absl::StatusOr<TypeAndBindings> CheckInvocation(
    DeduceCtx* ctx, const Invocation* invocation,
    const absl::flat_hash_map<const Param*, InterpValue>& constexpr_env);

// Validates type annotations on all functions within "module".
//
// Args:
//   module: The module to type check functions for.
//   import_cache: Import cache to use if an import is encountered in the
//      module, and that owns the type information determined for this module.
//   warnings: Object that collects warnings flagged during the typechecking
//      process.
//
// Returns type information mapping from AST nodes in the module to their
// deduced/checked type. The owner for the type info is within the import_cache.
absl::StatusOr<TypeInfo*> CheckModule(Module* module, ImportData* import_data,
                                      WarningCollector* warnings);

// Determines if the given type is best represented as a DSLX BuiltinType with
// fixed width, e.g., u7 or s64 vs bits, uN, or sN. If so, then this function
// returns that BuiltinType.
// Certain non-fixed width types, e.g., bits[17] or sN[63], while technically
// array types (ArrayTypeAnnotations), they're more easily represented
// in C++ as int64_t, as opposed to arrays of bool or "packed" int8_ts.
// These types, up to 64 bits wide, we interpret as scalar integral values.
absl::StatusOr<std::optional<BuiltinType>> GetAsBuiltinType(
    Module* module, TypeInfo* type_info, ImportData* import_data,
    const TypeAnnotation* type);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPECHECK_H_
