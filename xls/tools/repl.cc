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

#include <stddef.h>
#include <string.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
// TODO(taktoa): 2021-03-10 maybe switch to https://github.com/injinj/linecook
#include "linenoise.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/log_message.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/logging_internal.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/builtins.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/scanner.h"
#include "xls/dslx/type_info.h"
#include "xls/dslx/typecheck.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/passes/standard_pipeline.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_pass.h"

const char kUsage[] = R"(
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
  Trie() {}
  Trie(std::vector<std::string> all) {
    for (const auto& string : all) {
      this->Insert(string);
    }
  }

  // Insert the given string into the trie.
  void Insert(absl::string_view string) {
    strings_.push_back(std::string(string));
  }

  // Returns all strings that are suffixes of the given query string.
  std::vector<std::string> AllSuffixesOf(absl::string_view query) const {
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
  std::vector<absl::string_view> arguments;
};

// Parse a command, given a string like `":type identifier"`.
// Returns `absl::nullopt` only if the command could not be parsed.
absl::optional<Command> ParseCommand(absl::string_view str) {
  absl::string_view stripped_str = absl::StripAsciiWhitespace(str);
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
    XLS_VLOG(1) << "Unknown command prefix: \"" << stripped_str << "\"";
    return absl::nullopt;
  }
  return result;
}

// This is necessary because linenoise takes C-style function pointers to
// callbacks, without an extra `void*` parameter through which context can be
// provided.
struct Globals {
  absl::string_view dslx_path;
  std::unique_ptr<dslx::Scanner> scanner;
  std::unique_ptr<dslx::Parser> parser;
  dslx::ImportData import_data;
  std::unique_ptr<dslx::Module> module;
  dslx::TypeInfo* type_info;
  std::unique_ptr<Package> ir_package;
  Trie identifier_trie;
  Trie command_trie;
};

// Returns a pointer to a global `Globals` struct.
static Globals* GetSingletonGlobals() {
  static auto* globals = new Globals;
  return globals;
}

// Allows for tab completion of commands and their arguments.
void CompletionCallback(const char* buf, linenoiseCompletions* lc) {
  Globals* globals = GetSingletonGlobals();
  absl::string_view so_far(buf);
  absl::optional<Command> maybe_command = ParseCommand(buf);
  if (maybe_command) {
    Command command = maybe_command.value();
    switch (command.command) {
      case CommandName::kType: {
        absl::string_view arg = command.arguments[0];
        for (const auto& c : globals->identifier_trie.AllSuffixesOf(arg)) {
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
    for (const auto& c : globals->command_trie.AllSuffixesOf(so_far)) {
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
  return NULL;
}

// Given a prefix string and a pointer to a DSLX module, this populates the
// global identifier trie with the identifiers defined in that module. If the
// prefix string is not an empty string, the identifiers are prefixed by a
// namespace, e.g.: if prefix is `"std"`, then `"std::foo"` might be inserted.
void PopulateIdentifierTrieFromModule(std::string prefix,
                                      dslx::Module* module) {
  Globals* globals = GetSingletonGlobals();
  for (const auto& name : module->GetFunctionNames()) {
    std::string full_name;
    if (prefix.empty()) {
      full_name = name;
    } else {
      full_name = prefix + "::" + name;
    }
    globals->identifier_trie.Insert(full_name);
  }
}

// Populate the global identifier trie with the identifiers defined in the
// current module, as well as all identifiers defined in imported modules.
void PopulateIdentifierTrie() {
  Globals* globals = GetSingletonGlobals();
  globals->identifier_trie = Trie();
  PopulateIdentifierTrieFromModule("", globals->module.get());
  for (const auto& import_entry : globals->module->GetImportByName()) {
    if (auto maybe_imported_info =
            globals->type_info->GetImported(import_entry.second)) {
      const dslx::ImportedInfo* imported_info = *maybe_imported_info;
      PopulateIdentifierTrieFromModule(import_entry.first,
                                       imported_info->module);
    }
  }
}

// After this is called, the state of `GetSingletonGlobals()->ir_package` is
// guaranteed to be up to date with respect to the state of
// `GetSingletonGlobals()->module`.
absl::Status UpdateIr() {
  Globals* globals = GetSingletonGlobals();
  XLS_ASSIGN_OR_RETURN(
      globals->ir_package,
      ConvertModuleToPackage(globals->module.get(), &globals->import_data,
                             dslx::ConvertOptions{},
                             /*traverse_tests=*/true));
  XLS_RETURN_IF_ERROR(
      RunStandardPassPipeline(globals->ir_package.get()).status());
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
  Globals* globals = GetSingletonGlobals();

  XLS_ASSIGN_OR_RETURN(std::string dslx_contents,
                       GetFileContents(globals->dslx_path));

  globals->scanner = absl::make_unique<dslx::Scanner>(
      std::string(globals->dslx_path), dslx_contents);
  globals->parser =
      absl::make_unique<dslx::Parser>("main", globals->scanner.get());

  // TODO(taktoa): 2021-03-10 allow other kinds of failures to be recoverable
  absl::StatusOr<std::unique_ptr<dslx::Module>> maybe_module =
      globals->parser->ParseModule();
  if (maybe_module.ok()) {
    globals->module = std::move(*maybe_module);
  } else {
    std::cout << maybe_module.status() << "\n";
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(globals->type_info,
                       CheckModule(globals->module.get(), &globals->import_data,
                                   /*dslx_paths=*/{}));

  PopulateIdentifierTrie();

  std::cout << "Successfully loaded " << globals->dslx_path << "\n";

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
  std::cout << GetSingletonGlobals()->ir_package->DumpIr();
  return absl::OkStatus();
}

// Attempts to find an IR function within package based on function_name --
// attempts its value and its mangled value (against "module"'s name).
//
// Returns nullptr if neither of those can be found.
absl::StatusOr<Function*> FindFunction(absl::string_view function_name,
                                       dslx::Module* module, Package* package) {
  // The user may have given us a mangled or demangled name, first we see if
  // it's a mangled one, and if it's not, we mangle it and try that.
  if (package->HasFunctionWithName(function_name)) {
    return package->GetFunction(function_name);
  }
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      dslx::MangleDslxName(module->name(), function_name, /*free_keys=*/{}));
  if (!package->HasFunctionWithName(mangled_name)) {
    std::cerr << absl::StreamFormat(
        "Symbol \"%s\" was not found in IR as either \"%s\" or (mangled) "
        "\"%s\" -- run :ir "
        "to see IR for this package.\n",
        function_name, function_name, mangled_name);
    return nullptr;
  }
  return package->GetFunction(mangled_name);
}

// Function implementing the `:verilog` command, which generates and dumps the
// compiled Verilog for the module.
absl::Status CommandVerilog(absl::optional<std::string> function_name) {
  XLS_VLOG(1) << "Running verilog command with function name: "
              << (function_name ? *function_name : "<none>");
  XLS_RETURN_IF_ERROR(UpdateIr());
  Package* package = GetSingletonGlobals()->ir_package.get();
  dslx::Module* module = GetSingletonGlobals()->module.get();
  Function* main;
  if (function_name) {
    XLS_ASSIGN_OR_RETURN(main, FindFunction(*function_name, module, package));
    if (main == nullptr) {
      std::cerr << "Could not convert to verilog.\n";
      return absl::OkStatus();
    }
  } else {
    XLS_ASSIGN_OR_RETURN(main, package->EntryFunction());
  }
  XLS_RET_CHECK(main != nullptr);
  // TODO(taktoa): 2021-03-10 add ability to generate non-combinational modules
  XLS_ASSIGN_OR_RETURN(
      verilog::ModuleGeneratorResult result,
      verilog::GenerateCombinationalModule(main, /*use_system_verilog=*/false));
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
absl::Status CommandType(absl::string_view ident) {
  Globals* globals = GetSingletonGlobals();
  std::vector<std::string> split = absl::StrSplit(ident, absl::ByString("::"));
  absl::flat_hash_map<std::string, dslx::Function*> function_map;
  dslx::TypeInfo* type_info;
  if (split.size() == 1) {
    function_map = globals->module->GetFunctionByName();
    type_info = globals->type_info;
  } else if (split.size() == 2) {
    std::string import_name = split[0];
    absl::flat_hash_map<std::string, dslx::Import*> import_map =
        globals->module->GetImportByName();
    if (!import_map.contains(import_name)) {
      std::cout << "Could not find import: " << import_name << "\n";
      return absl::OkStatus();
    }
    dslx::Import* import = import_map[import_name];
    absl::optional<const dslx::ImportedInfo*> maybe_imported_info =
        globals->type_info->GetImported(import);
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
  absl::optional<dslx::ConcreteType*> type_or =
      type_info->GetItem(function_map[ident]);
  if (!type_or.has_value()) {
    std::cout << "Could not find type for identifier: " << ident << "\n";
    return absl::OkStatus();
  }
  std::cout << type_or.value()->ToString() << "\n";
  return absl::OkStatus();
}

absl::Status RealMain(absl::string_view dslx_path,
                      std::filesystem::path history_path) {
  Globals* globals = GetSingletonGlobals();

  std::cout << "Welcome to XLS. Type :help for help.\n";

  globals->dslx_path = dslx_path;
  globals->command_trie = {{":help", ":quit", ":reload", ":reset", ":ir",
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

    absl::optional<Command> command = ParseCommand(line);
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
        absl::optional<std::string> function_name;
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
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s DSLX_FILE",
                                          argv[0]);
  }

  absl::string_view dslx_path = positional_arguments[0];
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
    XLS_LOG(QFATAL)
        << "Could not find a path to put history in;"
        << " please define XLS_HISTORY_PATH (to /dev/null if necessary).";
  }

  XLS_QCHECK_OK(xls::RealMain(dslx_path, history_path));

  return EXIT_SUCCESS;
}
