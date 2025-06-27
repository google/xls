// Copyright 2020 The XLS Authors
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

// Implementation note on rough usage of verbose logging levels in this file:
//
//  2: Function-scope conversion activity (things that happen a few times on a
//     per-function basis).
//  3: Conversion order.
//  5: Interesting events that may occur several times within a function
//     conversion.
//  6: Interesting events that may occur many times (and will generally be more
//     noisy) within a function conversion.

#include "xls/dslx/ir_convert/ir_converter.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/filter.hpp"
#include "cppitertools/imap.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/channel_scope.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/ir_convert/function_converter.h"
#include "xls/dslx/ir_convert/proc_config_ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/proc_conversion.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {
namespace {

constexpr WarningCollector* kNoWarningCollector = nullptr;

// Tries four heuristics as names for potential entry functions of the package.
// Returns the first found heuristic. Otherwise, returns an absl::NotFoundError.
absl::StatusOr<xls::Function*> GetEntryFunction(xls::Package* package) {
  constexpr char kMain[] = "main";
  // Try a few possibilities of names for the canonical entry function.
  const std::vector<std::string> to_try = {
      kMain,
      package->name(),
      absl::StrCat("__", package->name(), "__", kMain),
      absl::StrCat("__", package->name(), "__", package->name()),
  };

  for (const std::string& attempt : to_try) {
    if (std::optional<xls::Function*> func = package->TryGetFunction(attempt);
        func.has_value()) {
      return *func;
    }
  }

  std::vector<xls::Function*> functions;
  // Finally we use the only function if only one exists.
  for (xls::FunctionBase* fb : package->GetFunctionBases()) {
    if (fb->IsFunction()) {
      functions.emplace_back(fb->AsFunctionOrDie());
    }
  }
  if (functions.size() == 1) {
    return functions.front();
  }

  return absl::NotFoundError(absl::StrFormat(
      "Could not find an entry function for the \"%s\" package.",
      package->name()));
}

// As a postprocessing step for converting a module to a package, we check and
// see if the entry point has the "implicit token" calling convention, to see if
// it should be wrapped up.
//
// Note: we do this as a postprocessing step because we can't know what the
// module entry point is _until_ all functions have been converted.
absl::Status WrapEntryIfImplicitToken(const PackageData& package_data,
                                      ImportData* import_data,
                                      const ConvertOptions& options) {
  auto entry_point_not_found_handler =
      [](const absl::Status& status) -> absl::Status {
    XLS_RET_CHECK_EQ(status.code(), absl::StatusCode::kNotFound);
    return absl::OkStatus();
  };

  XLS_ASSIGN_OR_RETURN(
      xls::Function * entry,
      GetEntryFunction(package_data.conversion_info->package.get()),
      _.With(entry_point_not_found_handler));

  if (package_data.wrappers.contains(entry)) {
    // Already created!
    return absl::OkStatus();
  }

  dslx::Function* dslx_entry =
      dynamic_cast<Function*>(package_data.ir_to_dslx.at(entry));
  XLS_RET_CHECK(dslx_entry != nullptr);
  if (GetRequiresImplicitToken(*dslx_entry, import_data, options)) {
    // Only create implicit token wrapper.
    auto implicit_entry_proto =
        absl::c_find_if(package_data.conversion_info->interface.functions(),
                        [&](const PackageInterfaceProto::Function& f) {
                          return f.base().name() == entry->name();
                        });
    return EmitImplicitTokenEntryWrapper(
               entry, dslx_entry,
               /*is_top=*/false, &package_data.conversion_info->interface,
               *implicit_entry_proto)
        .status();
  }
  return absl::OkStatus();
}

absl::Status ConvertOneFunctionInternal(PackageData& package_data,
                                        const ConversionRecord& record,
                                        ImportData* import_data,
                                        ProcConversionData* proc_data,
                                        ChannelScope* channel_scope,
                                        const ConvertOptions& options) {
  // Validate the requested conversion looks sound in terms of provided
  // parametrics.
  XLS_RETURN_IF_ERROR(ConversionRecord::ValidateParametrics(
      record.f(), record.parametric_env()));

  FunctionConverter converter(package_data, record.module(), import_data,
                              options, proc_data, channel_scope,
                              record.IsTop());
  XLS_ASSIGN_OR_RETURN(
      auto constant_deps,
      GetConstantDepFreevars(record.f()->body(), *record.type_info()));
  for (const auto& dep : constant_deps) {
    converter.AddConstantDep(dep);
  }

  Function* f = record.f();
  if (f->tag() == FunctionTag::kProcConfig) {
    // TODO(rspringer): 2021-09-29: Probably need to pass constants in here.
    ProcConfigIrConverter config_converter(
        f, record.type_info(), import_data, proc_data, channel_scope,
        record.parametric_env(), record.proc_id().value());
    XLS_RETURN_IF_ERROR(f->Accept(&config_converter));
    XLS_RETURN_IF_ERROR(config_converter.Finalize());
    return absl::OkStatus();
  }

  if (f->tag() == FunctionTag::kProcNext) {
    if (!proc_data->id_to_initial_value.contains(record.proc_id().value())) {
      Proc* p = f->proc().value();
      // If there's no value in the map, then this should be a top-level proc.
      // Verify that there are no parametric bindings.
      XLS_RET_CHECK(record.parametric_env().empty());
      XLS_ASSIGN_OR_RETURN(
          InterpValue iv,
          ConstexprEvaluator::EvaluateToValue(
              import_data, record.type_info(), kNoWarningCollector,
              record.parametric_env(), p->init().body()));
      XLS_ASSIGN_OR_RETURN(Value ir_value, InterpValueToValue(iv));
      proc_data->id_to_initial_value[record.proc_id().value()] = ir_value;
    }

    return converter.HandleProcNextFunction(
        f, record.invocation(), record.type_info(), import_data,
        &record.parametric_env(), record.proc_id().value(), proc_data);
  }

  return converter.HandleFunction(f, record.type_info(),
                                  &record.parametric_env());
}

// Creates the recv- and send-only channels needed for the top-level proc - it
// needs to communicate with the outside world somehow.
absl::Status CreateBoundaryChannels(absl::Span<Param* const> params,
                                    const ProcId& proc_id, TypeInfo* type_info,
                                    ProcConversionData* proc_data,
                                    ChannelScope* channel_scope) {
  for (const Param* param : params) {
    TypeAnnotation* type = param->type_annotation();
    if (type->annotation_kind() == ChannelTypeAnnotation::kAnnotationKind) {
      XLS_ASSIGN_OR_RETURN(
          ChannelOrArray channel_or_array,
          channel_scope->DefineBoundaryChannelOrArray(param, type_info));
      proc_data->id_to_config_args[proc_id].push_back(
          ChannelOrArrayToProcConfigValue(channel_or_array));
    }
  }
  return absl::OkStatus();
}

// Converts the procs in the package to new-style procs. If the top is not
// a proc, or there is at least one new-style proc already in the package,
// it will not be converted.
absl::Status ConvertToNewStyleProcs(Package* package) {
  VLOG(3) << "Converting package " << package->name()
          << " to new style procs/proc scoped channels";
  if (!package->GetTop().has_value() || !package->GetTop().value()->IsProc()) {
    VLOG(5) << "Will not convert package " << package->name()
            << " to new style procs; package top is not a proc";
    return absl::OkStatus();
  }

  // Make sure all are old style
  for (const std::unique_ptr<xls::Proc>& proc : package->procs()) {
    if (proc->is_new_style_proc()) {
      VLOG(5) << "Will not convert package " << package->name()
              << " to new style procs; found a new style proc already: "
              << proc->name();
      return absl::OkStatus();
    }
  }
  XLS_RETURN_IF_ERROR(ConvertPackageToNewStyleProcs(package));
  return absl::OkStatus();
}

// Converts the functions in the call graph in a specified order.
//
// Args:
//   order: order for conversion
//   import_data: Contains type information used in conversion.
//   options: Conversion option flags.
//   package: output of function
absl::Status ConvertCallGraph(absl::Span<const ConversionRecord> order,
                              ImportData* import_data,
                              const ConvertOptions& options,
                              PackageData& package_data) {
  VLOG(3) << "Conversion order: ["
          << absl::StrJoin(
                 order, "\n\t",
                 [](std::string* out, const ConversionRecord& record) {
                   absl::StrAppend(out, record.ToString());
                 })
          << "]";
  // We need to convert Functions before procs: Channels are declared inside
  // Functions, but exist as "global" entities in the IR. By processing
  // Functions first, we can collect the declarations of these global data so
  // we can resolve them when processing Procs. GetOrder() (in
  // extract_conversion_order.h) handles evaluation ordering. In addition, for
  // procs, the config block must be evaluated before the next block, as
  // config sets channels and constants needed by next. This is handled inside
  // ConvertOneFunctionInternal().
  ProcConversionData proc_data;

  // The `ChannelScope` owns all `ChannelArray` objects in the call graph, so
  // we need one instance to span all functions. However, most uses of it need
  // to be in the context of a function, and it needs to be aware of the current
  // function context, for e.g. index expression interpretation.
  ChannelScope channel_scope(package_data.conversion_info, import_data, options,
                             options.default_fifo_config);

  // The top-level proc's input/output channels need to come from _somewhere_.
  // At conversion time, though, we won't have that info. To enable forward
  // progress, we collect any channel args to the top-level proc and declare
  // them as send- or recv-only, as appropriate.
  const ConversionRecord* first_proc_config = nullptr;
  const ConversionRecord* first_proc_next = nullptr;
  Proc* top_proc = nullptr;
  // Only "next" functions are marked as "top", but we need to treat "config"
  // functions specially too, so we need to find the associated proc.
  for (const auto& record : order) {
    if (record.IsTop() && record.f()->tag() == FunctionTag::kProcNext) {
      top_proc = record.f()->proc().value();
    }
  }

  // If there's no top proc, then we'll just do our best & pick the first one.
  if (top_proc == nullptr) {
    for (const auto& record : order) {
      if (record.f()->tag() == FunctionTag::kProcConfig) {
        top_proc = record.f()->proc().value();
        break;
      }
    }
  }

  for (const auto& record : order) {
    if (record.f()->tag() == FunctionTag::kProcConfig &&
        record.f()->proc().value() == top_proc) {
      first_proc_config = &record;
    } else if (record.IsTop() && record.f()->tag() == FunctionTag::kProcNext) {
      first_proc_next = &record;
    }

    if (first_proc_config != nullptr && first_proc_next != nullptr) {
      break;
    }
  }

  // Set first proc's initial value.
  if (first_proc_config != nullptr) {
    XLS_RET_CHECK(first_proc_config->proc_id().has_value());
    ProcId proc_id = first_proc_config->proc_id().value();
    channel_scope.EnterFunctionContext(first_proc_config->type_info(),
                                       first_proc_config->parametric_env());
    XLS_RETURN_IF_ERROR(CreateBoundaryChannels(
        first_proc_config->f()->params(), proc_id,
        first_proc_config->type_info(), &proc_data, &channel_scope));
  }

  for (const ConversionRecord& record : order) {
    VLOG(3) << "Converting to IR: " << record.ToString();
    channel_scope.EnterFunctionContext(record.type_info(),
                                       record.parametric_env());
    XLS_RETURN_IF_ERROR(ConvertOneFunctionInternal(package_data, record,
                                                   import_data, &proc_data,
                                                   &channel_scope, options));
  }

  if (options.proc_scoped_channels) {
    // TODO: https://github.com/google/xls/issues/2078 - Remove this `if` after
    // lower_to_proc_scoped_channels is turned on for all tests.

    // If all channels are already proc-scoped (because
    // lower_to_proc_scoped_channels is fully implemented), this will have no
    // effect. Until then, this will convert any stragglers.
    XLS_RETURN_IF_ERROR(
        ConvertToNewStyleProcs(package_data.conversion_info->package.get()));
  }
  if (options.verify_ir) {
    VLOG(3) << "Verifying converted package";
    XLS_RETURN_IF_ERROR(
        VerifyPackage(package_data.conversion_info->package.get()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ConvertModuleIntoPackage(Module* module, ImportData* import_data,
                                      const ConvertOptions& options,
                                      PackageConversionData* package) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * root_type_info,
                       import_data->GetRootTypeInfo(module));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrder(module, root_type_info,
                                /*include_tests=*/options.convert_tests));
  PackageData package_data{.conversion_info = package};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));

  XLS_RETURN_IF_ERROR(
      WrapEntryIfImplicitToken(package_data, import_data, options));

  return absl::OkStatus();
}

absl::StatusOr<PackageConversionData> ConvertModuleToPackage(
    Module* module, ImportData* import_data, const ConvertOptions& options) {
  PackageConversionData p{.package = std::make_unique<Package>(module->name())};
  XLS_RETURN_IF_ERROR(
      ConvertModuleIntoPackage(module, import_data, options, &p));
  return p;
}

absl::StatusOr<std::string> ConvertModule(Module* module,
                                          ImportData* import_data,
                                          const ConvertOptions& options) {
  XLS_ASSIGN_OR_RETURN(PackageConversionData conv,
                       ConvertModuleToPackage(module, import_data, options));
  return conv.DumpIr();
}

// Perform basic checks that the proc can be the top converted proc and create
// an appropriate error message if the proc is unacceptable.
//
// To be acceptable the proc must only have channels as its config arguments.
absl::Status CheckAcceptableTopProc(Proc* proc) {
  auto is_not_channel_param = [&](const Param* param) {
    return dynamic_cast<ChannelTypeAnnotation*>(param->type_annotation()) ==
           nullptr;
  };
  auto it = absl::c_find_if(proc->config().params(), is_not_channel_param);
  if (it != proc->config().params().end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot convert proc '%s' due to non-channel config arguments: %s",
        proc->identifier(),
        absl::StrJoin(
            iter::imap(
                [](const Param* nd) -> std::string {
                  return absl::StrCat("'", nd->ToInlineString(), "'");
                },
                iter::filter(is_not_channel_param, proc->config().params())),
            ", ")));
  }

  return absl::OkStatus();
}

template <typename BlockT>
absl::Status ConvertOneFunctionIntoPackageInternal(
    BlockT* block, ImportData* import_data, const ParametricEnv* parametric_env,
    const ConvertOptions& options, PackageConversionData* conv) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * func_type_info,
                       import_data->GetRootTypeInfoForNode(block));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrderForEntry(block, func_type_info));
  PackageData package_data{.conversion_info = conv};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));
  return absl::OkStatus();
}

absl::Status ConvertOneFunctionIntoPackage(Function* fn,
                                           ImportData* import_data,
                                           const ParametricEnv* parametric_env,
                                           const ConvertOptions& options,
                                           PackageConversionData* conv) {
  return ConvertOneFunctionIntoPackageInternal(fn, import_data, parametric_env,
                                               options, conv);
}

absl::Status ConvertOneFunctionIntoPackage(Module* module,
                                           std::string_view entry_function_name,
                                           ImportData* import_data,
                                           const ParametricEnv* parametric_env,
                                           const ConvertOptions& options,
                                           PackageConversionData* conv) {
  std::optional<Function*> fn_or = module->GetFunction(entry_function_name);
  if (fn_or.has_value()) {
    return ConvertOneFunctionIntoPackageInternal(fn_or.value(), import_data,
                                                 parametric_env, options, conv);
  }

  absl::StatusOr<Proc*> proc =
      module->GetMemberOrError<Proc>(entry_function_name);
  if (proc.ok()) {
    XLS_RETURN_IF_ERROR(CheckAcceptableTopProc(*proc));
    return ConvertOneFunctionIntoPackageInternal(*proc, import_data,
                                                 parametric_env, options, conv);
  }

  if (options.convert_tests) {
    absl::StatusOr<TestFunction*> test_fn =
        module->GetTest(entry_function_name);
    if (test_fn.ok()) {
      return ConvertOneFunctionIntoPackageInternal(
          &(*test_fn)->fn(), import_data, parametric_env, options, conv);
    }
    absl::StatusOr<TestProc*> test_proc =
        module->GetTestProc(entry_function_name);
    if (test_proc.ok()) {
      XLS_RETURN_IF_ERROR(CheckAcceptableTopProc((*test_proc)->proc()));
      return ConvertOneFunctionIntoPackageInternal(
          (*test_proc)->proc(), import_data, parametric_env, options, conv);
    }
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Entry \"%s\" is not present in "
                      "DSLX module %s as a Function or a Proc.",
                      entry_function_name, module->name()));
}

absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, std::string_view entry_function_name,
    ImportData* import_data, const ParametricEnv* parametric_env,
    const ConvertOptions& options) {
  PackageConversionData conv{.package =
                                 std::make_unique<Package>(module->name())};
  XLS_RETURN_IF_ERROR(ConvertOneFunctionIntoPackage(module, entry_function_name,
                                                    import_data, parametric_env,
                                                    options, &conv));
  return conv.DumpIr();
}

namespace {
absl::StatusOr<std::unique_ptr<Module>> ParseText(
    VirtualizableFilesystem& vfs, FileTable& file_table, std::string_view text,
    std::string_view module_name, bool print_on_error,
    std::string_view filename, bool* printed_error) {
  Fileno fileno = file_table.GetOrCreate(filename);
  Scanner scanner{file_table, fileno, std::string(text)};
  Parser parser(std::string(module_name), &scanner);
  absl::StatusOr<std::unique_ptr<Module>> module = parser.ParseModule();
  *printed_error = TryPrintError(module.status(), file_table, vfs);
  return module;
}

absl::Status CheckPackageName(std::string_view name) {
  XLS_ASSIGN_OR_RETURN(std::vector<xls::Token> tokens, TokenizeString(name));
  if (tokens.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "package name '%s' (len: %d) is not a valid package name and would "
        "fail to parse from text-ir. Avoid use of infix or special characters.",
        name, name.size()));
  }
  return absl::OkStatus();
}

// Adds IR-converted symbols from the module specified by "path" to the given
// "package".
//
// TODO(leary): 2021-07-21 We should be able to reuse the type checking if
// there are overlapping nodes in the module DAG between files to process. For
// now we throw it away for each file and re-derive it (we need to refactor to
// make the modules outlive any given AddPathToPackage() if we want to
// appropriately reuse things in ImportData).
absl::Status AddContentsToPackage(
    std::string_view file_contents, std::string_view module_name,
    std::optional<std::string_view> path, std::optional<std::string_view> entry,
    const ConvertOptions& convert_options, ImportData* import_data,
    PackageConversionData* conv, bool* printed_error) {
  // Parse the module text.
  const std::string_view path_value = path.value_or("<UNKNOWN>");
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                       ParseText(import_data->vfs(), import_data->file_table(),
                                 file_contents, module_name,
                                 /*print_on_error=*/true,
                                 /*filename=*/path_value, printed_error));
  absl::StatusOr<TypecheckedModule> typechecked_module =
      TypecheckModule(std::move(module), path_value, import_data,
                      convert_options.type_inference_v2);
  if (!typechecked_module.ok()) {
    *printed_error =
        TryPrintError(typechecked_module.status(), import_data->file_table(),
                      import_data->vfs());
    return typechecked_module.status();
  }

  WarningCollector& warnings = typechecked_module->warnings;
  if (convert_options.warnings_as_errors && !warnings.warnings().empty()) {
    if (printed_error != nullptr) {
      *printed_error = true;
    }
    PrintWarnings(warnings, import_data->file_table(), import_data->vfs());
    return absl::InvalidArgumentError(
        "Warnings encountered and warnings-as-errors set.");
  }

  if (entry.has_value()) {
    XLS_RETURN_IF_ERROR(ConvertOneFunctionIntoPackage(
        typechecked_module->module, entry.value(), /*import_data=*/import_data,
        /*parametric_env=*/nullptr, convert_options, conv));
  } else {
    XLS_RETURN_IF_ERROR(ConvertModuleIntoPackage(
        typechecked_module->module, import_data, convert_options, conv));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<PackageConversionData> ConvertFilesToPackage(
    absl::Span<const std::string_view> paths, std::string_view stdlib_path,
    absl::Span<const std::filesystem::path> dslx_paths,
    const ConvertOptions& convert_options, std::optional<std::string_view> top,
    std::optional<std::string_view> package_name, bool* printed_error) {
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
  XLS_RETURN_IF_ERROR(CheckPackageName(resolved_package_name));
  PackageConversionData conversion_data{
      .package = std::make_unique<xls::Package>(resolved_package_name)};
  *conversion_data.interface.mutable_name() = resolved_package_name;
  for (std::string_view p : paths) {
    *conversion_data.interface.add_files() = p;
  }

  if (paths.size() > 1 && top.has_value()) {
    return absl::InvalidArgumentError(
        "Top cannot be supplied with multiple input paths (need a single input "
        "path to know where to resolve the entry function");
  }
  for (std::string_view path : paths) {
    ImportData import_data(
        CreateImportData(stdlib_path, dslx_paths, convert_options.warnings,
                         std::make_unique<RealFilesystem>()));
    XLS_ASSIGN_OR_RETURN(std::string text,
                         import_data.vfs().GetFileContents(path));
    XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(path));
    if (NameNeedsCanonicalization(path)) {
      WarningCollector col(convert_options.warnings);
      Pos pos(import_data.file_table().GetOrCreate(path), 0, 0);
      col.Add(
          Span(pos, pos), WarningKind::kIllegalPackageName,
          absl::StrFormat(
              "Module name '%s' is not a valid identifier name and "
              "would fail to parse from text-ir. Avoid use of "
              "infix or special characters.%s",
              RawNameFromPath(path).value(),
              convert_options.warnings_as_errors
                  ? ""
                  : absl::StrFormat(" Using '%s' as fallback.", module_name)));
      if (!col.empty()) {
        PrintWarnings(col, import_data.file_table(), import_data.vfs());
        if (convert_options.warnings_as_errors) {
          return absl::InvalidArgumentError(
              "Warnings encountered and warnings-as-errors set.");
        }
      }
    }
    XLS_RETURN_IF_ERROR(AddContentsToPackage(
        text, module_name, /*path=*/path, /*entry=*/top, convert_options,
        &import_data, &conversion_data, printed_error));
  }
  return conversion_data;
}

}  // namespace xls::dslx
