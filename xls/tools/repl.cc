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

#include <strings.h>

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
// TODO(taktoa): 2021-03-10 maybe switch to https://github.com/injinj/linecook
#include "absl/log/log.h"
#include "linenoise.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_module.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass_pipeline.h"

static constexpr std::string_view kUsage = R"(
Allows interactively exploring the functionality of XLS.

Usage looks like:
repl foo.x

where `foo.x` is a DSLX file.
)";

namespace xls {
namespace {

// Not actually a trie yet, but could be reimplemented with one.
class Trie {
 public:
  Trie() = default;
  Trie(std::initializer_list<std::string_view> all) {
    for (const auto& string : all) {
      this->Insert(string);
    }
  }

  // Insert the given string into the trie.
  void Insert(std::string_view string) { strings_.emplace_back(string); }

  // Returns all strings that are suffixes of the given query string.
  std::vector<std::string> AllSuffixesOf(std::string_view query) const {
    std::vector<std::string> result;
    for (const auto& str : strings_) {
      if (absl::StartsWith(str, query)) {
        result.push_back(str);
      }
    }
    return result;
  }

 private:
  std::vector<std::string> strings_;
};

// All of the different REPL commands, like `:type`.
enum class CommandName {
  kHelp,
  kQuit,
  kReload,
  kReset,
  kIr,
  kVerilog,
  kLlvm,
  kType
};

// A parsed command, along with its arguments.
struct Command {
  CommandName command;
  std::vector<std::string_view> arguments;
};

// Parse a command, given a string like `":type identifier"`.
// Returns `std::nullopt` only if the command could not be parsed.
std::optional<Command> ParseCommand(std::string_view str) {
  std::string_view stripped_str = absl::StripAsciiWhitespace(str);
  auto munch_prefix = [&str](std::string s) {
    return absl::ConsumePrefix(&str, s);
  };
  Command result;
  if ((stripped_str == ":help") || (stripped_str == ":h")) {
    result.command = CommandName::kHelp;
  } else if ((stripped_str == ":quit") || (stripped_str == ":q")) {
    result.command = CommandName::kQuit;
  } else if ((stripped_str == ":reload") || (stripped_str == ":r")) {
    result.command = CommandName::kReload;
  } else if (stripped_str == ":reset") {
    result.command = CommandName::kReset;
  } else if (stripped_str == ":ir") {
    result.command = CommandName::kIr;
  } else if (stripped_str == ":llvm") {
    result.command = CommandName::kLlvm;
  } else if (munch_prefix(":verilog")) {
    result.command = CommandName::kVerilog;
    // Optional function name argument.
    str = absl::StripAsciiWhitespace(str);
    if (!str.empty()) {
      result.arguments.push_back(str);
    }
  } else if (munch_prefix(":type ") || munch_prefix(":t ")) {
    result.command = CommandName::kType;
    result.arguments.push_back(absl::StripAsciiWhitespace(str));
  } else {
    VLOG(1) << "Unknown command prefix: \"" << stripped_str << "\"";
    return std::nullopt;
  }
  return result;
}

// This is necessary because linenoise takes C-style function pointers to
// callbacks, without an extra `void*` parameter through which context can be
// provided.
struct DslxGlobals {
  DslxGlobals(dslx::ImportData import_data_in,
              std::unique_ptr<dslx::Module> module_in,
              dslx::TypeInfo* type_info_in)
      : import_data(std::move(import_data_in)),
        module(std::move(module_in)),
        type_info(type_info_in) {}

  dslx::ImportData import_data;
  std::unique_ptr<dslx::Module> module;
  dslx::TypeInfo* type_info;
};

struct Globals {
  std::string_view dslx_path;
  std::unique_ptr<DslxGlobals> dslx;
  std::unique_ptr<Package> ir_package;
  Trie identifier_trie;
  Trie command_trie;
};

// Returns a pointer to a global `Globals` struct.
static Globals& GetSingletonGlobals() {
  static absl::NoDestructor<Globals> globals;
  return *globals;
}

// Allows for tab completion of commands and their arguments.
void CompletionCallback(const char* buf, linenoiseCompletions* lc) {
  Globals& globals = GetSingletonGlobals();
  std::string_view so_far(buf);
  std::optional<Command> maybe_command = ParseCommand(buf);
  if (maybe_command) {
    Command command = maybe_command.value();
    switch (command.command) {
      case CommandName::kType: {
        std::string_view arg = command.arguments[0];
        for (const auto& c : globals.identifier_trie.AllSuffixesOf(arg)) {
          std::string completion_with_command = ":type " + c;
          linenoiseAddCompletion(lc, completion_with_command.c_str());
        }
        break;
      }
      default: {
        break;
      }
    }
  } else {
    for (const auto& c : globals.command_trie.AllSuffixesOf(so_far)) {
      linenoiseAddCompletion(lc, c.c_str());
    }
  }
}

// Adds fish-style "hints" to commands with arguments. These show up when you've
// typed a command name but none of its arguments.
char* HintsCallback(const char* buf, int* color, int* bold) {
  if (!strcasecmp(buf, ":t") || !strcasecmp(buf, ":type")) {
    *color = 90;
    *bold = 0;
    return (char*)" <id>";
  }
  return nullptr;
}

// Given a prefix string and a pointer to a DSLX module, this populates the
// global identifier trie with the identifiers defined in that module. If the
// prefix string is not an empty string, the identifiers are prefixed by a
// namespace, e.g.: if prefix is `"std"`, then `"std::foo"` might be inserted.
void PopulateTrieFromModule(std::string prefix, dslx::Module* module,
                            Trie* trie) {
  for (const auto& name : module->GetFunctionNames()) {
    std::string full_name;
    if (prefix.empty()) {
      full_name = name;
    } else {
      full_name = prefix + "::" + name;
    }
    trie->Insert(full_name);
  }
}

// Populate the global identifier trie with the identifiers defined in the
// current module, as well as all identifiers defined in imported modules.
Trie PopulateIdentifierTrie() {
  Globals& globals = GetSingletonGlobals();
  Trie trie;
  PopulateTrieFromModule("", globals.dslx->module.get(), &trie);
  for (const auto& import_entry : globals.dslx->module->GetImportByName()) {
    if (auto maybe_imported_info =
            globals.dslx->type_info->GetImported(import_entry.second)) {
      const dslx::ImportedInfo* imported_info = *maybe_imported_info;
      PopulateTrieFromModule(import_entry.first, imported_info->module, &trie);
    }
  }
  return trie;
}

// After this is called, the state of `GetSingletonGlobals().ir_package` is
// guaranteed to be up to date with respect to the state of
// `GetSingletonGlobals().module`.
absl::Status UpdateIr() {
  Globals& globals = GetSingletonGlobals();
  XLS_ASSIGN_OR_RETURN(dslx::PackageConversionData data,
                       ConvertModuleToPackage(globals.dslx->module.get(),
                                              &globals.dslx->import_data,
                                              dslx::ConvertOptions{}));
  globals.ir_package = std::move(data).package;
  XLS_ASSIGN_OR_RETURN(std::string mangled_name,
                       dslx::MangleDslxName(globals.dslx->module.get()->name(),
                                            "main", /*convention=*/{}));
  XLS_RETURN_IF_ERROR(globals.ir_package->SetTopByName(mangled_name));
  XLS_RETURN_IF_ERROR(
      RunOptimizationPassPipeline(globals.ir_package.get()).status());
  return absl::OkStatus();
}

// Implementation note: commands should quash top-level-recoverable errors and
// do their own printing to stderr (propagating OkStatus at the command top
// level to the REPL despite the fact it quashed an error).

// Function implementing the `:help` command, which shows a help message.
absl::Status CommandHelp() {
  std::cout << "Commands:\n\n"
            << "    :h :help         Show this help message\n"
            << "    :q :quit         Quit the REPL\n"
            << "    :r :reload       Reload the file from disk\n"
            << "    :reset           Reload file, then reset terminal\n"
            << "                     (note: this will clear terminal history)\n"
            << "    :ir              Generate and print IR for file\n"
            << "    :verilog         Generate and print Verilog for file\n"
            << "    :llvm            Generate and print LLVM for file\n"
            << "    :type <id>       Show the type of identifier <id>\n"
            << "    :t <id>          Alias for :type\n"
            << "\n";
  return absl::OkStatus();
}

// Function implementing the `:reload` command, which reloads the DSLX file from
// disk and parses/typechecks it.
absl::Status CommandReload() {
  Globals& globals = GetSingletonGlobals();

  XLS_ASSIGN_OR_RETURN(std::string dslx_contents,
                       GetFileContents(globals.dslx_path));

  dslx::FileTable& file_table = globals.dslx->import_data.file_table();
  dslx::Fileno fileno = file_table.GetOrCreate(globals.dslx_path);
  dslx::Scanner scanner(file_table, fileno, dslx_contents);
  dslx::Parser parser("main", &scanner);

  // TODO(taktoa): 2021-03-10 allow other kinds of failures to be recoverable
  absl::StatusOr<std::unique_ptr<dslx::Module>> maybe_module =
      parser.ParseModule();
  if (!maybe_module.ok()) {
    std::cout << maybe_module.status() << "\n";
    return absl::OkStatus();
  }

  std::unique_ptr<dslx::Module> module = std::move(maybe_module).value();
  dslx::ImportData import_data(dslx::CreateImportData(
      /*dslx_stdlib_path=*/"",
      /*additional_search_paths=*/{},
      /*enabled_warnings=*/dslx::kDefaultWarningsSet));
  dslx::WarningCollector warnings(import_data.enabled_warnings());
  XLS_ASSIGN_OR_RETURN(dslx::TypeInfo * type_info,
                       TypecheckModule(module.get(), &import_data, &warnings));
  globals.dslx = std::make_unique<DslxGlobals>(std::move(import_data),
                                               std::move(module), type_info);
  globals.identifier_trie = PopulateIdentifierTrie();

  std::cout << "Successfully loaded " << globals.dslx_path << "\n";

  return absl::OkStatus();
}

// Function implementing the `:reset` command, which is the same as `:reload`
// except it also resets the terminal first.
absl::Status CommandReset() {
  std::cout << "\ec";
  return CommandReload();
}

// Function implementing the `:ir` command, which generates and dumps the IR for
// the module.
absl::Status CommandIr() {
  XLS_RETURN_IF_ERROR(UpdateIr());
  std::cout << GetSingletonGlobals().ir_package->DumpIr();
  return absl::OkStatus();
}

// Attempts to find an IR function within package based on function_name --
// attempts its value and its mangled value (against "module"'s name).
//
// Returns nullptr if neither of those can be found.
absl::StatusOr<Function*> FindFunction(std::string_view function_name,
                                       dslx::Module* module, Package* package) {
  // The user may have given us a mangled or demangled name, first we see if
  // it's a mangled one, and if it's not, we mangle it and try that.
  if (std::optional<Function*> f = package->TryGetFunction(function_name);
      f.has_value()) {
    return *f;
  }
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      dslx::MangleDslxName(module->name(), function_name, /*convention=*/{}));
  if (std::optional<Function*> f = package->TryGetFunction(mangled_name)) {
    return *f;
  }

  std::cerr << absl::StreamFormat(
      "Symbol \"%s\" was not found in IR as either \"%s\" or (mangled) "
      "\"%s\" -- run :ir "
      "to see IR for this package.\n",
      function_name, function_name, mangled_name);
  return nullptr;
}

// Function implementing the `:verilog` command, which generates and dumps the
// compiled Verilog for the module.
absl::Status CommandVerilog(std::optional<std::string> function_name) {
  VLOG(1) << "Running verilog command with function name: "
          << (function_name ? *function_name : "<none>");
  XLS_RETURN_IF_ERROR(UpdateIr());
  Package* package = GetSingletonGlobals().ir_package.get();
  dslx::Module* module = GetSingletonGlobals().dslx->module.get();
  FunctionBase* main;
  if (function_name) {
    XLS_ASSIGN_OR_RETURN(main, FindFunction(*function_name, module, package));
    if (main == nullptr) {
      std::cerr << "Could not convert to verilog.\n";
      return absl::OkStatus();
    }
  } else {
    std::optional<FunctionBase*> top = package->GetTop();
    if (!top.has_value()) {
      return absl::InternalError(absl::StrFormat(
          "Top entity not set for package: %s.", package->name()));
    }
    main = top.value();
  }
  XLS_RET_CHECK(main != nullptr);
  // TODO(taktoa): 2021-03-10 add ability to generate non-combinational modules
  XLS_ASSIGN_OR_RETURN(
      verilog::ModuleGeneratorResult result,
      verilog::GenerateCombinationalModule(main, verilog::CodegenOptions()));
  std::cout << result.verilog_text;
  return absl::OkStatus();
}

// Function implementing the `:llvm` command, which generates and dumps the
// LLVM IR for the module.
absl::Status CommandLlvm() {
  // TODO(taktoa): 2021-03-10 implement LLVM command
  std::cout << "The :llvm command is not yet implemented.\n";
  return absl::OkStatus();
}

// Function implementing the `:type` command, which prints the type of the given
// identifier (defined in the current module or imported).
absl::Status CommandType(std::string_view ident) {
  Globals& globals = GetSingletonGlobals();
  std::vector<std::string> split = absl::StrSplit(ident, absl::ByString("::"));
  absl::flat_hash_map<std::string, dslx::Function*> function_map;
  dslx::TypeInfo* type_info;
  if (split.size() == 1) {
    function_map = globals.dslx->module->GetFunctionByName();
    type_info = globals.dslx->type_info;
  } else if (split.size() == 2) {
    std::string import_name = split[0];
    absl::flat_hash_map<std::string, dslx::Import*> import_map =
        globals.dslx->module->GetImportByName();
    if (!import_map.contains(import_name)) {
      std::cout << "Could not find import: " << import_name << "\n";
      return absl::OkStatus();
    }
    dslx::Import* import = import_map[import_name];
    std::optional<const dslx::ImportedInfo*> maybe_imported_info =
        globals.dslx->type_info->GetImported(import);
    if (!maybe_imported_info) {
      std::cout << "Something is wrong with TypeInfo::GetImported\n";
      return absl::OkStatus();
    }
    const dslx::ImportedInfo* imported_info = *maybe_imported_info;
    function_map = imported_info->module->GetFunctionByName();
    type_info = imported_info->type_info;
    ident = split[1];
  } else {
    std::cout << "Invalid number of :: in identifier.\n";
    return absl::OkStatus();
  }
  if (!function_map.contains(ident)) {
    std::cout << "Could not find identifier: " << ident << "\n";
    return absl::OkStatus();
  }
  std::optional<dslx::Type*> type_or = type_info->GetItem(function_map[ident]);
  if (!type_or.has_value()) {
    std::cout << "Could not find type for identifier: " << ident << "\n";
    return absl::OkStatus();
  }
  std::cout << type_or.value()->ToString() << "\n";
  return absl::OkStatus();
}

absl::Status RealMain(std::string_view dslx_path,
                      std::filesystem::path history_path) {
  Globals& globals = GetSingletonGlobals();

  std::cout << "Welcome to XLS. Type :help for help.\n";

  globals.dslx_path = dslx_path;
  globals.command_trie = Trie{{":help", ":quit", ":reload", ":reset", ":ir",
                               ":verilog", ":llvm", ":type "}};

  XLS_RETURN_IF_ERROR(CommandReload());

  linenoiseSetCompletionCallback(CompletionCallback);
  linenoiseSetHintsCallback(HintsCallback);

  {
    std::filesystem::path history_dir = history_path;
    history_dir.remove_filename();

    XLS_RETURN_IF_ERROR(RecursivelyCreateDir(history_dir));
    XLS_RETURN_IF_ERROR(AppendStringToFile(history_path, ""));
  }

  linenoiseHistoryLoad(history_path.c_str());

  while (true) {
    char* line_ptr = linenoise("xls> ");
    if (line_ptr == nullptr) {
      std::cout << "Ctrl-D received, quitting.\n";
      break;
    }
    std::string line(line_ptr);
    free(line_ptr);
    linenoiseHistoryAdd(line.c_str());
    linenoiseHistorySave(history_path.c_str());

    std::optional<Command> command = ParseCommand(line);
    if (!command) {
      continue;
    }
    switch (command->command) {
      case CommandName::kHelp: {
        XLS_RETURN_IF_ERROR(CommandHelp());
        break;
      }
      case CommandName::kQuit: {
        return absl::OkStatus();
        break;
      }
      case CommandName::kReload: {
        XLS_RETURN_IF_ERROR(CommandReload());
        break;
      }
      case CommandName::kReset: {
        XLS_RETURN_IF_ERROR(CommandReset());
        break;
      }
      case CommandName::kIr: {
        XLS_RETURN_IF_ERROR(CommandIr());
        break;
      }
      case CommandName::kVerilog: {
        std::optional<std::string> function_name;
        if (!command->arguments.empty()) {
          function_name = command->arguments[0];
        }
        XLS_RETURN_IF_ERROR(CommandVerilog(function_name));
        break;
      }
      case CommandName::kLlvm: {
        XLS_RETURN_IF_ERROR(CommandLlvm());
        break;
      }
      case CommandName::kType: {
        XLS_RETURN_IF_ERROR(CommandType(command->arguments[0]));
        break;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s DSLX_FILE",
                                      argv[0]);
  }

  std::string_view dslx_path = positional_arguments[0];
  std::filesystem::path history_path;

  if (char* xls_history_path = std::getenv("XLS_HISTORY_PATH")) {
    history_path = xls_history_path;
  } else if (char* xdg_cache_home = std::getenv("XDG_CACHE_HOME")) {
    history_path = xdg_cache_home;
    history_path += "/xls/repl-history";
  } else if (char* home = std::getenv("HOME")) {
    history_path = home;
    history_path += "/.cache/xls/repl-history";
  } else {
    LOG(QFATAL)
        << "Could not find a path to put history in;"
        << " please define XLS_HISTORY_PATH (to /dev/null if necessary).";
  }

  return xls::ExitStatus(xls::RealMain(dslx_path, history_path));
}
