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

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/deduce_ctx.h"

namespace xls::dslx {

// Instantiates a builtin parametric invocation; e.g. `update()`.
absl::StatusOr<NameDef*> InstantiateBuiltinParametric(
    BuiltinNameDef* builtin_name, Invocation* invocation, DeduceCtx* ctx);

// Type-checks function f in the given module.
absl::Status CheckTopNodeInModule(
    absl::variant<Function*, TestFunction*, StructDef*, TypeDef*> f,
    DeduceCtx* ctx);

// Validates type annotations on all functions within "module".
//
// Args:
//   module: The module to type check functions for.
//   import_cache: Import cache to use if an import is encountered in the
//      module.
//  additional_search_paths: Additional paths to search for modules on import.
//
// Returns type information mapping from AST nodes in the module to their
//  deduced/checked type.
absl::StatusOr<std::shared_ptr<TypeInfo>> CheckModule(
    Module* module, ImportCache* import_cache,
    absl::Span<const std::string> additional_search_paths);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPECHECK_H_
