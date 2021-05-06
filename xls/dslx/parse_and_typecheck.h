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

#ifndef XLS_DSLX_PARSE_AND_TYPECHECK_H_
#define XLS_DSLX_PARSE_AND_TYPECHECK_H_

#include <filesystem>

#include "xls/dslx/ast.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Note: these will be owned by the import_data used in ParseAndTypecheck().
struct TypecheckedModule {
  Module* module;
  TypeInfo* type_info;
};

// Helper that parses and typechecks the given "text" for a module.
//
// "path" is used for error reporting (`Span`s) and module_name is the name
// given to the returned `TypecheckedModule::module`. "import_data" is used to
// get-or-insert any imported modules.
//
// TODO(rspringer): 2021/03/04 Move "additional_search_paths" into ImportData
// (nee ImportCache), to avoid having to carry it everywhere.
absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
    absl::string_view text, absl::string_view path,
    absl::string_view module_name, ImportData* import_data,
    absl::Span<const std::filesystem::path> additional_search_paths);

}  // namespace xls::dslx

#endif  // XLS_DSLX_PARSE_AND_TYPECHECK_H_
