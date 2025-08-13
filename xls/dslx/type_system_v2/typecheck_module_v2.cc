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

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/file/filesystem.h"
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
#include "xls/tools/typecheck_flags.h"
#include "xls/tools/typecheck_flags.pb.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<ModuleInfo>> TypecheckModuleV2(
    std::unique_ptr<Module> module, std::filesystem::path path,
    ImportData* import_data, WarningCollector* warnings) {
  std::string_view module_name = module->name();
  const bool top_module = !import_data->HasInferenceTable();
  if (top_module) {
    VLOG(3) << "Using type system v2 for type checking of " << path;
  }

  InferenceTable* table = import_data->GetOrCreateInferenceTable();
  XLS_ASSIGN_OR_RETURN(TypecheckFlagsProto flags, GetTypecheckFlagsProto());
  std::unique_ptr<TypeSystemTracer> tracer =
      TypeSystemTracer::Create(flags.dump_traces(), flags.time_every_action());
  auto& tracer_ref = *tracer;
  auto typecheck_imported_module = [import_data, warnings](
                                       std::unique_ptr<Module> module,
                                       std::filesystem::path path) {
    return TypecheckModuleV2(std::move(module), path, import_data, warnings);
  };
  XLS_RETURN_IF_ERROR(PopulateTable(table, module.get(), import_data, warnings,
                                    typecheck_imported_module));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<InferenceTableConverter> converter,
                       CreateInferenceTableConverter(
                           *table, *module, *import_data, *warnings,
                           import_data->file_table(), std::move(tracer)));
  import_data->SetInferenceTableConverter(module.get(), converter.get());

  absl::Status status =
      converter->ConvertSubtree(module.get(), /*function=*/std::nullopt,
                                /*parametric_context=*/std::nullopt);
  if (!status.ok()) {
    VLOG(1) << "Inference table conversion FAILURE: " << status;
  }

  if (flags.dump_inference_table() && top_module) {
    XLS_RETURN_IF_ERROR(SetFileContents(
        std::filesystem::path(flags.trace_out_dir()) /
            absl::Substitute("inference_table_$0.txt", module_name),
        table->ToString()));
  }

  if (flags.dump_traces()) {
    const std::filesystem::path out_dir =
        std::filesystem::path(flags.trace_out_dir());
    XLS_RETURN_IF_ERROR(SetFileContents(
        out_dir / absl::Substitute("traces_$0.txt", module_name),
        tracer_ref.ConvertTracesToString()));
    XLS_RETURN_IF_ERROR(SetFileContents(
        out_dir / absl::Substitute("trace_stats_$0.txt", module_name),
        tracer_ref.ConvertStatsToString(import_data->file_table())));
  }

  if (!status.ok()) {
    return status;
  }

  XLS_ASSIGN_OR_RETURN(
      TypeInfo * type_info,
      import_data->type_info_owner().GetRootTypeInfo(module.get()));

  return std::make_unique<ModuleInfo>(std::move(module), type_info,
                                      std::filesystem::path(path),
                                      std::move(converter));
}

}  // namespace xls::dslx
