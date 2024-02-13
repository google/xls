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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPECHECK_MODULE_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPECHECK_MODULE_H_

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

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
absl::StatusOr<TypeInfo*> TypecheckModule(Module* module,
                                          ImportData* import_data,
                                          WarningCollector* warnings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPECHECK_MODULE_H_
