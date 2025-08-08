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

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_module.h"
#include "xls/dslx/type_system_v2/typecheck_module_v2.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
    std::string_view text, std::string_view path, std::string_view module_name,
    ImportData* import_data, std::vector<CommentData>* comments,
    bool force_version2, const ConvertOptions& options) {
  XLS_RET_CHECK(import_data != nullptr);

  FileTable& file_table = import_data->file_table();
  Fileno fileno = file_table.GetOrCreate(path);

  // The outermost import doesn't have a real import statement associated with
  // it, but we need the filename to be correct to detect cycles.
  const Span fake_import_span = Span(Pos(fileno, 0, 0), Pos(fileno, 0, 0));
  XLS_RETURN_IF_ERROR(import_data->AddToImporterStack(fake_import_span, path));
  absl::Cleanup cleanup = [&] {
    CHECK_OK(import_data->PopFromImporterStack(fake_import_span));
  };

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                       ParseModule(text, path, module_name,
                                   import_data->file_table(), comments));

  XLS_RETURN_IF_ERROR(module->SetConfiguredValues(options.configured_values));
  return TypecheckModule(std::move(module), path, import_data, force_version2);
}

absl::StatusOr<std::unique_ptr<Module>> ParseModule(
    std::string_view text, std::string_view path, std::string_view module_name,
    FileTable& file_table, std::vector<CommentData>* comments) {
  Fileno fileno = file_table.GetOrCreate(path);
  Scanner scanner(file_table, fileno, std::string{text});
  Parser parser(std::string{module_name}, &scanner);
  XLS_ASSIGN_OR_RETURN(auto module, parser.ParseModule());
  if (comments != nullptr) {
    *comments = scanner.PopComments();
  }
  return module;
}

absl::StatusOr<std::unique_ptr<Module>> ParseModuleFromFileAtPath(
    std::string_view file_path, std::string_view module_name,
    ImportData* import_data) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path path,
                       GetXlsRunfilePath(file_path));
  XLS_ASSIGN_OR_RETURN(std::string text_dslx,
                       import_data->vfs().GetFileContents(path));
  return ParseModule(text_dslx, file_path, module_name,
                     import_data->file_table());
}

absl::StatusOr<TypecheckedModule> TypecheckModule(
    std::unique_ptr<Module> module, std::string_view path,
    ImportData* import_data, bool force_version2) {
  XLS_RET_CHECK(module.get() != nullptr);
  XLS_RET_CHECK(import_data != nullptr);

  std::string_view module_name = module->name();

  WarningCollector warnings(import_data->enabled_warnings());
  Module* module_ptr = module.get();
  XLS_ASSIGN_OR_RETURN(ImportTokens subject,
                       ImportTokens::FromString(module_name));

  using ExtendedTypecheckModuleFn =
      absl::StatusOr<std::unique_ptr<ModuleInfo>> (*)(
          std::unique_ptr<Module>, std::filesystem::path path, ImportData*,
          WarningCollector*);
  ExtendedTypecheckModuleFn version_entry_point =
      force_version2 || module->attributes().contains(
                            ModuleAttribute::kTypeInferenceVersion2)
          ? &TypecheckModuleV2
          : static_cast<ExtendedTypecheckModuleFn>(&TypecheckModule);
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ModuleInfo> module_info,
      version_entry_point(std::move(module), path, import_data, &warnings));

  TypeInfo* type_info = module_info->type_info();
  XLS_RETURN_IF_ERROR(
      import_data->Put(subject, std::move(module_info)).status());
  return TypecheckedModule{module_ptr, type_info, std::move(warnings)};
}

}  // namespace xls::dslx
