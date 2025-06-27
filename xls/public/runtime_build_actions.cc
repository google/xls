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

#include "xls/public/runtime_build_actions.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_result.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/extract_module_name.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/opt.h"
#include "xls/tools/proto_to_dslx.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

std::string_view GetDefaultDslxStdlibPath() { return kDefaultDslxStdlibPath; }

absl::StatusOr<std::string> ConvertDslxToIr(
    std::string_view dslx, std::string_view path, std::string_view module_name,
    const ConvertDslxToIrOptions& options) {
  VLOG(5) << "ConvertDslxToIr; path: " << path
          << " module name: " << module_name
          << " warnings_as_errors: " << options.warnings_as_errors;

  std::string enable_warnings_str = absl::StrJoin(options.enable_warnings, ",");
  std::string disable_warnings_str =
      absl::StrJoin(options.disable_warnings, ",");
  XLS_ASSIGN_OR_RETURN(
      dslx::WarningKindSet warnings_set,
      dslx::GetWarningsSetFromFlags(enable_warnings_str, disable_warnings_str));

  dslx::ImportData import_data(dslx::CreateImportData(
      std::string(options.dslx_stdlib_path), options.additional_search_paths,
      warnings_set, std::make_unique<dslx::RealFilesystem>()));

  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule typechecked,
      dslx::ParseAndTypecheck(dslx, path, module_name, &import_data));

  const dslx::WarningCollector& warnings = typechecked.warnings;
  if (options.warnings_out != nullptr) {
    for (const dslx::WarningCollector::Entry& entry : warnings.warnings()) {
      options.warnings_out->push_back(entry.message);
    }
  }
  if (options.warnings_as_errors && !warnings.empty()) {
    return absl::InvalidArgumentError(
        "Conversion of DSLX to IR failed due to warnings during "
        "parsing/typechecking.");
  }

  return dslx::ConvertModule(
      typechecked.module, &import_data,
      dslx::ConvertOptions{
          .warnings_as_errors = options.warnings_as_errors,
      });
}

absl::StatusOr<std::string> ConvertDslxPathToIr(
    const std::filesystem::path& path, const ConvertDslxToIrOptions& options) {
  XLS_ASSIGN_OR_RETURN(std::string dslx, GetFileContents(path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, dslx::ExtractModuleName(path));
  return ConvertDslxToIr(dslx, std::string{path}, module_name, options);
}

absl::StatusOr<std::string> OptimizeIr(std::string_view ir,
                                       std::string_view top) {
  const tools::OptOptions options = {
      .opt_level = xls::kMaxOptLevel,
      .top = top,
  };
  return tools::OptimizeIrForTop(ir, options);
}

absl::StatusOr<std::string> MangleDslxName(std::string_view module_name,
                                           std::string_view function_name) {
  return dslx::MangleDslxName(module_name, function_name,
                              dslx::CallingConvention::kTypical,
                              /*free_keys=*/{},
                              /*parametric_env=*/nullptr);
}

absl::StatusOr<std::string> ProtoToDslx(std::string_view proto_def,
                                        std::string_view message_name,
                                        std::string_view text_proto,
                                        std::string_view binding_name) {
  dslx::FileTable file_table;
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::Module> module,
                       ProtoToDslxViaText(proto_def, message_name, text_proto,
                                          binding_name, file_table));
  return module->ToString();
}

absl::StatusOr<ScheduleAndCodegenResult> ScheduleAndCodegenPackage(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model) {
  std::pair<SchedulingResult, verilog::CodegenResult> result;
  XLS_ASSIGN_OR_RETURN(
      result, ScheduleAndCodegen(p, scheduling_options_flags_proto,
                                 codegen_flags_proto, with_delay_model));
  return ScheduleAndCodegenResult{.scheduling_result = result.first,
                                  .codegen_result = result.second};
}

}  // namespace xls
