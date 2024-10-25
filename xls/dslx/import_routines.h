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

#ifndef XLS_DSLX_IMPORT_ROUTINES_H_
#define XLS_DSLX_IMPORT_ROUTINES_H_

#include <functional>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Type-checking callback lambda.
using TypecheckModuleFn = std::function<absl::StatusOr<TypeInfo*>(Module*)>;

// Imports the module identified (globally) by `subject`.
//
// Importing means: locating, parsing, typechecking, and caching in the import
// cache.
//
// Resolves against an existing import in `import_data` if it is present.
//
// Args:
//  ftypecheck: Function that can be used to get type information for a module.
//  subject: Tokens that globally uniquely identify the module to import; e.g.
//      something built-in like `{"std"}` for the standard library or something
//      fully qualified like `{"xls", "lib", "math"}`.
//  import_data: Cache that we resolve against so we don't waste resources
//      re-importing things in the import DAG.
//  import_span: Indicates the "importer" (i.e. the AST node, lexically) that
//      caused this attempt to import.
//
// Returns:
//  The imported module information.
absl::StatusOr<ModuleInfo*> DoImport(const TypecheckModuleFn& ftypecheck,
                                     const ImportTokens& subject,
                                     ImportData* import_data,
                                     const Span& import_span,
                                     VirtualizableFilesystem& vfs);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IMPORT_ROUTINES_H_
