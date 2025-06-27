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
//
// DSLX-to-SystemVerilog type and constant converter.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.pb.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/translators/dslx_to_verilog.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

ABSL_FLAG(std::string, namespace, "xls",
          "The Verilog namespace to generate the code in (e.g., `foo::bar`).");
// TODO: google/xls#922 - use a more generic way to add waivers.
ABSL_FLAG(std::vector<std::string>, lint_waivers, {},
          "Lint waivers to add to the generated code.");

namespace xls {
namespace dslx {
namespace {
bool TypeDefinitionSourceIsPublic(const TypeInfo::TypeSource& def_source) {
  return absl::visit(
      Visitor{
          [](const TypeAlias* type_alias) { return type_alias->is_public(); },
          [](const ProcDef* proc_def) { return proc_def->is_public(); },
          [](const EnumDef* enum_def) { return enum_def->is_public(); },
          [](const StructDef* struct_def) { return struct_def->is_public(); },
      },
      def_source.definition);
}

absl::Status RealMain(absl::Span<const std::string_view> paths) {
  // Reuse IR converter options as they align closely with the DSLX-to-Verilog
  // use case. The IR converter converts DSLX to IR, which involves getting
  // concrete types as a prerequisite.
  XLS_ASSIGN_OR_RETURN(IrConverterOptionsFlagsProto ir_converter_options,
                       GetIrConverterOptionsFlagsProto());

  std::optional<std::filesystem::path> output_file =
      ir_converter_options.has_output_file()
          ? std::make_optional<std::filesystem::path>(
                ir_converter_options.output_file())
          : std::nullopt;

  std::string_view dslx_stdlib_path = ir_converter_options.dslx_stdlib_path();
  std::string_view dslx_path = ir_converter_options.dslx_path();
  std::vector<std::string_view> dslx_path_strs = absl::StrSplit(dslx_path, ':');

  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  std::optional<std::string_view> top;
  if (ir_converter_options.has_top()) {
    top = ir_converter_options.top();
  }

  std::optional<std::string_view> package_name;
  if (ir_converter_options.has_package_name()) {
    package_name = ir_converter_options.package_name();
  }

  XLS_ASSIGN_OR_RETURN(WarningKindSet enabled_warnings,
                       WarningKindSetFromDisabledString(
                           ir_converter_options.disable_warnings()));

  std::string resolved_package_name;
  if (package_name.has_value()) {
    resolved_package_name = package_name.value();
  } else {
    if (paths.size() > 1) {
      return absl::InvalidArgumentError(
          "Package name must be given when multiple input paths are supplied");
    }
    // Get it from the one module name (if package name was unspecified and we
    // just have one path).
    XLS_ASSIGN_OR_RETURN(resolved_package_name, PathToName(paths[0]));
  }

  if (paths.size() > 1 && top.has_value()) {
    return absl::InvalidArgumentError(
        "Top cannot be supplied with multiple input paths (need a single input "
        "path to know where to resolve the entry function");
  }

  XLS_ASSIGN_OR_RETURN(
      DslxTypeToVerilogManager type_to_verilog,
      DslxTypeToVerilogManager::Create(absl::GetFlag(FLAGS_namespace)));

  for (std::string_view path : paths) {
    ImportData import_data(
        CreateImportData(dslx_stdlib_path, dslx_paths, enabled_warnings,
                         std::make_unique<RealFilesystem>()));
    XLS_ASSIGN_OR_RETURN(std::string text,
                         import_data.vfs().GetFileContents(path));
    XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(path));
    XLS_ASSIGN_OR_RETURN(
        TypecheckedModule tm,
        ParseAndTypecheck(text, path, module_name, &import_data, {}));

    for (const auto& def : tm.module->GetTypeDefinitions()) {
      // Ignore private type definitions.
      XLS_ASSIGN_OR_RETURN(const TypeInfo::TypeSource type_definition_source,
                           tm.type_info->ResolveTypeDefinition(def));
      if (!TypeDefinitionSourceIsPublic(type_definition_source)) {
        continue;
      }
      AstNode* def_node = TypeDefinitionToAstNode(def);
      Type* type = nullptr;
      {
        std::optional<Type*> type_from_type_info =
            tm.type_info->GetItem(def_node);
        if (!type_from_type_info.has_value()) {
          VLOG(3) << absl::StreamFormat("Skipping %s with no type info.",
                                        def_node->ToInlineString());
          continue;
        }
        type = *type_from_type_info;
      }
      if (type->HasParametricDims()) {
        VLOG(3) << absl::StreamFormat("Skipping %s with parametric type.",
                                      def_node->ToInlineString());

        continue;
      }
      VLOG(3) << absl::StreamFormat("Converting definition %s to Verilog",
                                    def_node->ToInlineString());
      XLS_RETURN_IF_ERROR(
          type_to_verilog.AddTypeForTypeDefinition(def, &import_data));
    }
  }
  std::string output;
  // Build lint waivers.
  for (std::string_view waiver : absl::GetFlag(FLAGS_lint_waivers)) {
    absl::StrAppend(&output, "// verilog_lint: waive-start ", waiver, "\n");
  }
  absl::StrAppend(&output, type_to_verilog.Emit());
  for (std::string_view waiver : absl::GetFlag(FLAGS_lint_waivers)) {
    absl::StrAppend(&output, "// verilog_lint: waive-end ", waiver, "\n");
  }
  return SetFileContents(output_file.value(), output);
}
}  // namespace
}  // namespace dslx
}  // namespace xls

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args = xls::InitXls(argv[0], argc, argv);
  if (args.empty()) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got " << args.size()
                << ": `" << absl::StrJoin(args, " ") << "`; want " << argv[0]
                << " <input-file>";
  }
  // "-" is a special path that is shorthand for /dev/stdin. Update here as
  // there isn't a better place later.
  for (auto& arg : args) {
    if (arg == "-") {
      arg = "/dev/stdin";
    }
  }

  return xls::ExitStatus(xls::dslx::RealMain(args));
}
