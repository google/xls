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

#include "xls/dslx/parse_and_typecheck.h"

#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/scanner.h"
#include "xls/dslx/typecheck.h"

namespace xls::dslx {

absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
    absl::string_view text, absl::string_view path,
    absl::string_view module_name, ImportData* import_data) {
  XLS_RET_CHECK(import_data != nullptr);

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                       ParseModule(text, path, module_name));
  return TypecheckModule(std::move(module), path, import_data);
}

absl::StatusOr<std::unique_ptr<Module>> ParseModule(
    absl::string_view text, absl::string_view path,
    absl::string_view module_name) {
  Scanner scanner{std::string{path}, std::string{text}};
  Parser parser(std::string{module_name}, &scanner);
  return parser.ParseModule();
}

absl::StatusOr<std::unique_ptr<Module>> ParseModuleFromFileAtPath(
    absl::string_view file_path, absl::string_view module_name) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path path,
                       GetXlsRunfilePath(file_path));
  XLS_ASSIGN_OR_RETURN(std::string text_dslx, GetFileContents(path));
  return ParseModule(text_dslx, file_path, module_name);
}

absl::StatusOr<TypecheckedModule> TypecheckModule(
    std::unique_ptr<Module> module, absl::string_view path,
    ImportData* import_data) {
  XLS_RET_CHECK(module.get() != nullptr);
  XLS_RET_CHECK(import_data != nullptr);

  absl::string_view module_name = module->name();

  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       CheckModule(module.get(), import_data));
  TypecheckedModule result{module.get(), type_info};
  XLS_ASSIGN_OR_RETURN(ImportTokens subject,
                       ImportTokens::FromString(module_name));
  XLS_RETURN_IF_ERROR(import_data
                          ->Put(subject, std::make_unique<ModuleInfo>(
                                             std::move(module), type_info,
                                             std::filesystem::path(path)))
                          .status());
  return result;
}

}  // namespace xls::dslx
