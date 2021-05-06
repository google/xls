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

// Type-checks function f in the given module.
absl::Status CheckTopNodeInModule(
    absl::variant<Function*, TestFunction*, StructDef*, TypeDef*> f,
    DeduceCtx* ctx);

// Validates type annotations on all functions within "module".
//
// Args:
//   module: The module to type check functions for.
//   import_cache: Import cache to use if an import is encountered in the
//      module, and that owns the type information determined for this module.
//  additional_search_paths: Additional paths to search for modules on import.
//
// Returns type information mapping from AST nodes in the module to their
// deduced/checked type. The owner for the type info is within the import_cache.
absl::StatusOr<TypeInfo*> CheckModule(
    Module* module, ImportData* import_data,
    absl::Span<const std::filesystem::path> additional_search_paths);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPECHECK_H_
