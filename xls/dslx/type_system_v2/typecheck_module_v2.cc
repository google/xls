// Copyright 2025 The XLS Authors
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

#include "xls/dslx/type_system_v2/typecheck_module_v2.h"

#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/inference_table_converter_impl.h"
#include "xls/dslx/type_system_v2/populate_table.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<ModuleInfo>> TypecheckModuleV2(
    std::unique_ptr<Module> module, std::filesystem::path path,
    ImportData* import_data, WarningCollector* warnings) {
  std::unique_ptr<InferenceTable> table = InferenceTable::Create(*module);
  std::unique_ptr<TypeSystemTracer> tracer = TypeSystemTracer::Create();
  auto& tracer_ref = *tracer;
  auto typecheck_imported_module = [import_data, warnings](
                                       std::unique_ptr<Module> module,
                                       std::filesystem::path path) {
    return TypecheckModuleV2(std::move(module), path, import_data, warnings);
  };
  XLS_RETURN_IF_ERROR(PopulateTable(table.get(), module.get(), import_data,
                                    warnings, typecheck_imported_module));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<InferenceTableConverter> converter,
                       CreateInferenceTableConverter(
                           *table, *module, *import_data, *warnings,
                           import_data->file_table(), std::move(tracer)));
  absl::Status status =
      converter->ConvertSubtree(module.get(), /*function=*/std::nullopt,
                                /*parametric_context=*/std::nullopt);
  if (!status.ok()) {
    VLOG(1) << "Inference table conversion FAILURE: " << status;
  }

  if (VLOG_IS_ON(5)) {
    std::cerr << "Inference table after conversion:\n"
              << table->ToString() << "\n"
              << "User module traces after conversion:\n"
              << tracer_ref.ConvertTracesToString() << "\n";
  }

  if (!status.ok()) {
    return status;
  }

  XLS_ASSIGN_OR_RETURN(
      TypeInfo * type_info,
      import_data->type_info_owner().GetRootTypeInfo(module.get()));

  return std::make_unique<ModuleInfo>(std::move(module), type_info,
                                      std::filesystem::path(path),
                                      std::move(table), std::move(converter));
}

}  // namespace xls::dslx
