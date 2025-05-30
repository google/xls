// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system_v2/populate_table.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/inference_table_converter_impl.h"
#include "xls/dslx/type_system_v2/populate_table_visitor.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

absl::Status PopulateBuiltinStubs(ImportData* import_data,
                                  WarningCollector* warnings,
                                  InferenceTable* table) {
  XLS_ASSIGN_OR_RETURN(ImportTokens builtin_tokens,
                       ImportTokens::FromString(kBuiltinStubsModuleName));
  if (import_data->Contains(builtin_tokens)) {
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> builtins_module,
                       LoadBuiltinStubs());
  std::unique_ptr<PopulateTableVisitor> builtins_visitor =
      CreatePopulateTableVisitor(builtins_module.get(), table, import_data,
                                 /*typecheck_imported_module=*/nullptr);
  XLS_RETURN_IF_ERROR(
      builtins_visitor->PopulateFromModule(builtins_module.get()));

  Module* builtins_ptr = builtins_module.get();
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<InferenceTableConverter> builtins_converter,
      CreateInferenceTableConverter(
          *table, *builtins_module, *import_data, *warnings,
          import_data->file_table(),
          TypeSystemTracer::Create(/*active=*/false)));
  XLS_ASSIGN_OR_RETURN(TypeInfo * builtins_type_info,
                       import_data->GetRootTypeInfo(builtins_ptr));

  XLS_ASSIGN_OR_RETURN(std::filesystem::path builtins_path, BuiltinStubsPath());
  std::unique_ptr<ModuleInfo> builtins_module_info =
      std::make_unique<ModuleInfo>(std::move(builtins_module),
                                   builtins_type_info, builtins_path,
                                   std::move(builtins_converter));
  return import_data->Put(builtin_tokens, std::move(builtins_module_info))
      .status();
}

absl::Status PopulateTable(InferenceTable* table, Module* module,
                           ImportData* import_data, WarningCollector* warnings,
                           TypecheckModuleFn typecheck_imported_module) {
  XLS_RETURN_IF_ERROR(PopulateBuiltinStubs(import_data, warnings, table));

  std::unique_ptr<PopulateTableVisitor> visitor = CreatePopulateTableVisitor(
      module, table, import_data, std::move(typecheck_imported_module));
  return visitor->PopulateFromModule(module);
}

}  // namespace xls::dslx
