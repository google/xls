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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/ir_convert/function_converter.h"
#include "xls/dslx/ir_convert/ir_conversion_utils.h"
#include "xls/dslx/ir_convert/proc_config_ir_converter.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/typecheck.h"
#include "xls/dslx/warning_collector.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"

namespace xls::dslx {
namespace {

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
    auto func_or = package->GetFunction(attempt);
    if (func_or.ok()) {
      return func_or.value();
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
  absl::StatusOr<xls::Function*> entry_or =
      GetEntryFunction(package_data.package);
  if (!entry_or.ok()) {  // Entry point not found.
    XLS_RET_CHECK_EQ(entry_or.status().code(), absl::StatusCode::kNotFound);
    return absl::OkStatus();
  }

  xls::Function* entry = entry_or.value();
  if (package_data.wrappers.contains(entry)) {
    // Already created!
    return absl::OkStatus();
  }

  dslx::Function* dslx_entry =
      dynamic_cast<Function*>(package_data.ir_to_dslx.at(entry));
  if (GetRequiresImplicitToken(dslx_entry, import_data, options)) {
    // Only create implicit token wrapper.
    return EmitImplicitTokenEntryWrapper(entry, dslx_entry, /*is_top=*/false)
        .status();
  }
  return absl::OkStatus();
}

absl::Status ConvertOneFunctionInternal(PackageData& package_data,
                                        const ConversionRecord& record,
                                        ImportData* import_data,
                                        ProcConversionData* proc_data,
                                        const ConvertOptions& options) {
  // Validate the requested conversion looks sound in terms of provided
  // parametrics.
  XLS_RETURN_IF_ERROR(ConversionRecord::ValidateParametrics(
      record.f(), record.parametric_env()));

  FunctionConverter converter(package_data, record.module(), import_data,
                              options, proc_data, record.IsTop());
  XLS_ASSIGN_OR_RETURN(auto constant_deps,
                       GetConstantDepFreevars(record.f()->body()));
  for (const auto& dep : constant_deps) {
    converter.AddConstantDep(dep);
  }

  Function* f = record.f();
  if (f->tag() == Function::Tag::kProcConfig) {
    // TODO(rspringer): 2021-09-29: Probably need to pass constants in here.
    ProcConfigIrConverter converter(
        package_data.package, f, record.type_info(), import_data, proc_data,
        record.parametric_env(), record.proc_id().value());
    XLS_RETURN_IF_ERROR(f->Accept(&converter));
    XLS_RETURN_IF_ERROR(converter.Finalize());
    return absl::OkStatus();
  }

  if (f->tag() == Function::Tag::kProcNext) {
    if (!proc_data->id_to_initial_value.contains(record.proc_id().value())) {
      Proc* p = f->proc().value();
      XLS_ASSIGN_OR_RETURN(
          ConcreteType * foo,
          record.type_info()->GetItemOrError(p->init()->body()));

      // If there's no value in the map, then this should be a top-level proc.
      // Verify that there are no parametric bindings.
      XLS_RET_CHECK(record.parametric_env().empty());
      XLS_ASSIGN_OR_RETURN(
          InterpValue iv, ConstexprEvaluator::EvaluateToValue(
                              import_data, record.type_info(),
                              record.parametric_env(), p->init()->body(), foo));
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
                                    const PackageData& package_data,
                                    ProcConversionData* proc_data) {
  for (const Param* param : params) {
    TypeAnnotation* type = param->type_annotation();
    if (auto* channel_type = dynamic_cast<ChannelTypeAnnotation*>(type);
        type != nullptr) {
      auto maybe_type = type_info->GetItem(channel_type->payload());
      XLS_RET_CHECK(maybe_type.has_value());
      ConcreteType* ct = maybe_type.value();
      XLS_ASSIGN_OR_RETURN(
          Type * ir_type, TypeToIr(package_data.package, *ct, ParametricEnv()));
      ChannelOps op = channel_type->direction() == ChannelDirection::kIn
                          ? ChannelOps::kReceiveOnly
                          : ChannelOps::kSendOnly;
      std::string channel_name =
          absl::StrCat(package_data.package->name(), "__", param->identifier());
      XLS_ASSIGN_OR_RETURN(StreamingChannel * channel,
                           package_data.package->CreateStreamingChannel(
                               channel_name, op, ir_type));
      proc_data->id_to_config_args[proc_id].push_back(channel);
    }
  }
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
  XLS_VLOG(3) << "Conversion order: ["
              << absl::StrJoin(
                     order, "\n\t",
                     [](std::string* out, const ConversionRecord& record) {
                       absl::StrAppend(out, record.ToString());
                     })
              << "]";
  // We need to convert Functions before procs: Channels are declared inside
  // Functions, but exist as "global" entities in the IR. By processing
  // Functions first, we can collect the declarations of these global data so we
  // can resolve them when processing Procs. GetOrder() (in
  // extract_conversion_order.h) handles evaluation ordering. In addition, for
  // procs, the config block must be evaluated before the next block, as config
  // sets channels and constants needed by next. This is handled inside
  // ConvertOneFunctionInternal().
  ProcConversionData proc_data;

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
    if (record.IsTop() && record.f()->tag() == Function::Tag::kProcNext) {
      top_proc = record.f()->proc().value();
    }
  }

  // If there's no top proc, then we'll just do our best & pick the first one.
  if (top_proc == nullptr) {
    for (const auto& record : order) {
      if (record.f()->tag() == Function::Tag::kProcConfig) {
        top_proc = record.f()->proc().value();
        break;
      }
    }
  }

  for (const auto& record : order) {
    if (record.f()->tag() == Function::Tag::kProcConfig &&
        record.f()->proc().value() == top_proc) {
      first_proc_config = &record;
    } else if (record.IsTop() &&
               record.f()->tag() == Function::Tag::kProcNext) {
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
    XLS_RETURN_IF_ERROR(CreateBoundaryChannels(
        first_proc_config->f()->params(), proc_id,
        first_proc_config->type_info(), package_data, &proc_data));
  }

  for (const ConversionRecord& record : order) {
    XLS_VLOG(3) << "Converting to IR: " << record.ToString();
    XLS_RETURN_IF_ERROR(ConvertOneFunctionInternal(
        package_data, record, import_data, &proc_data, options));
  }

  XLS_VLOG(3) << "Verifying converted package";
  if (options.verify_ir) {
    XLS_RETURN_IF_ERROR(VerifyPackage(package_data.package));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ConvertModuleIntoPackage(Module* module, ImportData* import_data,
                                      const ConvertOptions& options,
                                      bool traverse_tests, Package* package) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * root_type_info,
                       import_data->GetRootTypeInfo(module));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrder(module, root_type_info, traverse_tests));
  PackageData package_data{package};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));

  XLS_RETURN_IF_ERROR(
      WrapEntryIfImplicitToken(package_data, import_data, options));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Package>> ConvertModuleToPackage(
    Module* module, ImportData* import_data, const ConvertOptions& options,
    bool traverse_tests) {
  auto package = std::make_unique<Package>(module->name());
  XLS_RETURN_IF_ERROR(ConvertModuleIntoPackage(module, import_data, options,
                                               traverse_tests, package.get()));
  return package;
}

absl::StatusOr<std::string> ConvertModule(Module* module,
                                          ImportData* import_data,
                                          const ConvertOptions& options) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ConvertModuleToPackage(module, import_data, options));
  return package->DumpIr();
}

template <typename BlockT>
absl::Status ConvertOneFunctionIntoPackageInternal(
    Module* module, BlockT* block, ImportData* import_data,
    const ParametricEnv* parametric_env, const ConvertOptions& options,
    Package* package) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * func_type_info,
                       import_data->GetRootTypeInfoForNode(block));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrderForEntry(block, func_type_info));
  PackageData package_data{package};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));
  return absl::OkStatus();
}

absl::Status ConvertOneFunctionIntoPackage(Module* module,
                                           std::string_view entry_function_name,
                                           ImportData* import_data,
                                           const ParametricEnv* parametric_env,
                                           const ConvertOptions& options,
                                           Package* package) {
  std::optional<Function*> fn_or = module->GetFunction(entry_function_name);
  if (fn_or.has_value()) {
    return ConvertOneFunctionIntoPackageInternal(
        module, fn_or.value(), import_data, parametric_env, options, package);
  }

  auto proc_or = module->GetMemberOrError<Proc>(entry_function_name);
  if (proc_or.ok()) {
    return ConvertOneFunctionIntoPackageInternal(
        module, proc_or.value(), import_data, parametric_env, options, package);
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
  Package package(module->name());
  XLS_RETURN_IF_ERROR(ConvertOneFunctionIntoPackage(module, entry_function_name,
                                                    import_data, parametric_env,
                                                    options, &package));
  return package.DumpIr();
}

namespace {
absl::StatusOr<std::unique_ptr<Module>> ParseText(std::string_view text,
                                                  std::string_view module_name,
                                                  bool print_on_error,
                                                  std::string_view filename,
                                                  bool* printed_error) {
  Scanner scanner{std::string(filename), std::string(text)};
  Parser parser(std::string(module_name), &scanner);
  absl::StatusOr<std::unique_ptr<Module>> module_or = parser.ParseModule();
  *printed_error = TryPrintError(module_or.status());
  return module_or;
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
    Package* package, bool* printed_error) {
  // Parse the module text.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> module,
      ParseText(file_contents, module_name, /*print_on_error=*/true,
                /*filename=*/path.value_or("<UNKNOWN>"), printed_error));
  WarningCollector warnings;
  absl::StatusOr<TypeInfo*> type_info_or =
      CheckModule(module.get(), import_data, &warnings);
  if (!type_info_or.ok()) {
    *printed_error = TryPrintError(type_info_or.status());
    return type_info_or.status();
  }

  if (convert_options.warnings_as_errors && !warnings.warnings().empty()) {
    if (printed_error != nullptr) {
      *printed_error = true;
    }
    PrintWarnings(warnings);
    return absl::InvalidArgumentError(
        "Warnings encountered and warnings-as-errors set.");
  }

  if (entry.has_value()) {
    XLS_RETURN_IF_ERROR(ConvertOneFunctionIntoPackage(
        module.get(), entry.value(), /*import_data=*/import_data,
        /*parametric_env=*/nullptr, convert_options, package));
  } else {
    XLS_RETURN_IF_ERROR(
        ConvertModuleIntoPackage(module.get(), import_data, convert_options,
                                 /*traverse_tests=*/false, package));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<Package>> ConvertFilesToPackage(
    absl::Span<const std::string_view> paths, std::string stdlib_path,
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
  auto package =
      std::make_unique<xls::Package>(std::move(resolved_package_name));

  if (paths.size() > 1 && top.has_value()) {
    return absl::InvalidArgumentError(
        "Top cannot be supplied with multiple input paths (need a single input "
        "path to know where to resolve the entry function");
  }
  for (std::string_view path : paths) {
    ImportData import_data(CreateImportData(stdlib_path, dslx_paths));
    XLS_ASSIGN_OR_RETURN(std::string text, GetFileContents(path));
    XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(path));
    XLS_RETURN_IF_ERROR(AddContentsToPackage(
        text, module_name, /*path=*/path, /*entry=*/top, convert_options,
        &import_data, package.get(), printed_error));
  }

  return package;
}

}  // namespace xls::dslx
