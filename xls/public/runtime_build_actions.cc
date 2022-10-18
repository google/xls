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

#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/passes/passes.h"
#include "xls/tools/opt.h"
#include "xls/tools/proto_to_dslx.h"

namespace xls {

std::string_view GetDefaultDslxStdlibPath() { return kDefaultDslxStdlibPath; }

absl::StatusOr<std::string> ConvertDslxToIr(
    std::string_view dslx, std::string_view path,
    std::string_view module_name, std::string_view dslx_stdlib_path,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  XLS_VLOG(5) << "path: " << path << " module name: " << module_name
              << " stdlib_path: " << dslx_stdlib_path;
  dslx::ImportData import_data(dslx::CreateImportData(
      std::string(dslx_stdlib_path), additional_search_paths));
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule typechecked,
      dslx::ParseAndTypecheck(dslx, path, module_name, &import_data));
  return dslx::ConvertModule(typechecked.module, &import_data,
                             dslx::ConvertOptions{});
}

static absl::StatusOr<std::string> ExtractModuleName(
    std::filesystem::path path) {
  if (path.extension() != ".x") {
    return absl::InvalidArgumentError(
        absl::StrFormat("DSL module path must end with '.x', got: '%s'", path));
  }
  return path.stem();
}

absl::StatusOr<std::string> ConvertDslxPathToIr(
    std::filesystem::path path, std::string_view dslx_stdlib_path,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  XLS_ASSIGN_OR_RETURN(std::string dslx, GetFileContents(path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, ExtractModuleName(path));
  return ConvertDslxToIr(dslx, std::string(path), module_name, dslx_stdlib_path,
                         additional_search_paths);
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
                              /*symbolic_bindings=*/nullptr);
}

absl::StatusOr<std::string> ProtoToDslx(std::string_view proto_def,
                                        std::string_view message_name,
                                        std::string_view text_proto,
                                        std::string_view binding_name) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<dslx::Module> module,
      ProtoToDslxViaText(proto_def, message_name, text_proto, binding_name));
  return module->ToString();
}

}  // namespace xls
