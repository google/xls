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

#include <memory>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Note: these will be owned by the import_data used in ParseAndTypecheck()
//       or TypecheckModule
struct TypecheckedModule {
  Module* module;
  TypeInfo* type_info;
  WarningCollector warnings;
};

// Helper that parses and typechecks the given "text" for a module.
//
// "path" is used for error reporting (`Span`s) and module_name is the name
// given to the returned `TypecheckedModule::module`. "import_data" is used to
// get-or-insert any imported modules.
absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
    std::string_view text, std::string_view path, std::string_view module_name,
    ImportData* import_data, std::vector<CommentData>* comments = nullptr);

// Helper that parses and creates a new module from the given "text".
//
// "path" is used for error reporting (`Span`s) and module_name is the name
// given to the returned `TypecheckedModule::module`.
absl::StatusOr<std::unique_ptr<Module>> ParseModule(
    std::string_view text, std::string_view path, std::string_view module_name,
    FileTable& file_table, std::vector<CommentData>* comments = nullptr);

// Helper that parses and created a new Module from the given DSLX file path.
//   path - path to the file to read and parse.
//   module_name - the name given to the returned Module;
absl::StatusOr<std::unique_ptr<Module>> ParseModuleFromFileAtPath(
    std::string_view path, std::string_view module_name, FileTable& file_table);

// Helper that typechecks an already parsed module, ownership of
// the module will be given to import_data.
//
// "path" is used for error reporting (`Span`s)
// "import_data" is used to get-or-insert any imported modules.
absl::StatusOr<TypecheckedModule> TypecheckModule(
    std::unique_ptr<Module> module, std::string_view path,
    ImportData* import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_PARSE_AND_TYPECHECK_H_
