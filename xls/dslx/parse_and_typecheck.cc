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

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/type_system/typecheck.h"

namespace xls::dslx {

absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
    std::string_view text, std::string_view path,
    std::string_view module_name, ImportData* import_data) {
  XLS_RET_CHECK(import_data != nullptr);

  // The outermost import doesn't have a real import statement associated with
  // it, but we need the filename to be correct to detect cycles.
  const Span fake_import_span =
      Span(Pos(std::string(path), 0, 0), Pos(std::string(path), 0, 0));
  XLS_RETURN_IF_ERROR(import_data->AddToImporterStack(fake_import_span, path));
  auto cleanup = absl::MakeCleanup([&] {
    XLS_CHECK_OK(import_data->PopFromImporterStack(fake_import_span));
  });

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                       ParseModule(text, path, module_name));
  return TypecheckModule(std::move(module), path, import_data);
}

absl::StatusOr<std::unique_ptr<Module>> ParseModule(
    std::string_view text, std::string_view path,
    std::string_view module_name) {
  Scanner scanner{std::string{path}, std::string{text}};
  Parser parser(std::string{module_name}, &scanner);
  return parser.ParseModule();
}

absl::StatusOr<std::unique_ptr<Module>> ParseModuleFromFileAtPath(
    std::string_view file_path, std::string_view module_name) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path path,
                       GetXlsRunfilePath(file_path));
  XLS_ASSIGN_OR_RETURN(std::string text_dslx, GetFileContents(path));
  return ParseModule(text_dslx, file_path, module_name);
}

absl::StatusOr<TypecheckedModule> TypecheckModule(
    std::unique_ptr<Module> module, std::string_view path,
    ImportData* import_data) {
  XLS_RET_CHECK(module.get() != nullptr);
  XLS_RET_CHECK(import_data != nullptr);

  std::string_view module_name = module->name();

  WarningCollector warnings;
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       CheckModule(module.get(), import_data, &warnings));
  TypecheckedModule result{module.get(), type_info, std::move(warnings)};
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
