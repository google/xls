// Copyright 2023 The XLS Authors
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

#include "xls/fuzzer/sample_runner.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>  // NOLINT
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/revision.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/common/subprocess.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_kind.h"
#include "xls/fuzzer/cpp_sample_runner.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample.pb.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/public/runtime_build_actions.h"
#include "xls/simulation/check_simulator.h"
#include "xls/tools/eval_utils.h"
#include "re2/re2.h"

// These are used to forward, but also see comment below.
ABSL_DECLARE_FLAG(int32_t, v);
ABSL_DECLARE_FLAG(std::string, vmodule);

namespace xls {

namespace {

using ArgsBatch = std::vector<std::vector<dslx::InterpValue>>;

static constexpr std::string_view kCodegenMainPath = "xls/tools/codegen_main";
static constexpr std::string_view kEvalIrMainPath = "xls/tools/eval_ir_main";
static constexpr std::string_view kEvalProcMainPath =
    "xls/tools/eval_proc_main";
static constexpr std::string_view kIrConverterMainPath =
    "xls/dslx/ir_convert/ir_converter_main";
static constexpr std::string_view kIrOptMainPath = "xls/tools/opt_main";
static constexpr std::string_view kSimulateModuleMainPath =
    "xls/tools/simulate_module_main";

absl::StatusOr<ArgsBatch> ConvertFunctionKwargs(
    const dslx::Function* f, const dslx::ImportData& import_data,
    const dslx::TypecheckedModule& tm, const ArgsBatch& args_batch) {
  XLS_ASSIGN_OR_RETURN(dslx::FunctionType * fn_type,
                       tm.type_info->GetItemAs<dslx::FunctionType>(f));
  ArgsBatch converted_args;
  converted_args.reserve(args_batch.size());
  for (const std::vector<dslx::InterpValue>& unsigned_args : args_batch) {
    XLS_ASSIGN_OR_RETURN(std::vector<dslx::InterpValue> args,
                         dslx::SignConvertArgs(*fn_type, unsigned_args));
    converted_args.push_back(std::move(args));
  }
  return converted_args;
}

absl::StatusOr<std::vector<dslx::InterpValue>> RunFunctionBatched(
    const dslx::Function& f, dslx::ImportData& import_data,
    const dslx::TypecheckedModule& tm, const ArgsBatch& args_batch) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<dslx::BytecodeFunction> bf,
      dslx::BytecodeEmitter::Emit(&import_data, tm.type_info, f,
                                  /*caller_bindings=*/{}));
  std::vector<dslx::InterpValue> results;
  results.reserve(args_batch.size());
  for (const std::vector<dslx::InterpValue>& args : args_batch) {
    XLS_ASSIGN_OR_RETURN(
        dslx::InterpValue result,
        dslx::BytecodeInterpreter::Interpret(&import_data, bf.get(), args));
    results.push_back(result);
  }
  return results;
}

absl::StatusOr<std::vector<dslx::InterpValue>> InterpretDslxFunction(
    std::string_view text, std::string_view top_name,
    const ArgsBatch& args_batch, const std::filesystem::path& run_dir) {
  dslx::ImportData import_data = dslx::CreateImportData(
      GetDefaultDslxStdlibPath(),
      /*additional_search_paths=*/{}, dslx::kDefaultWarningsSet);
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule tm,
      dslx::ParseAndTypecheck(text, "sample.x", "sample", &import_data));

  std::optional<dslx::ModuleMember*> module_member =
      tm.module->FindMemberWithName(top_name);
  CHECK(module_member.has_value());
  dslx::ModuleMember* member = module_member.value();
  CHECK(std::holds_alternative<dslx::Function*>(*member));
  dslx::Function* f = std::get<dslx::Function*>(*member);
  XLS_RET_CHECK(f != nullptr);

  XLS_ASSIGN_OR_RETURN(ArgsBatch converted_args_batch,
                       ConvertFunctionKwargs(f, import_data, tm, args_batch));
  XLS_ASSIGN_OR_RETURN(
      std::vector<dslx::InterpValue> results,
      RunFunctionBatched(*f, import_data, tm, converted_args_batch));
  XLS_ASSIGN_OR_RETURN(std::vector<Value> ir_results,
                       dslx::InterpValue::ConvertValuesToIr(results));
  std::string serialized_results = absl::StrCat(
      absl::StrJoin(ir_results, "\n",
                    [](std::string* out, const Value& value) {
                      absl::StrAppend(out,
                                      value.ToString(FormatPreference::kHex));
                    }),
      "\n");
  XLS_RETURN_IF_ERROR(
      SetFileContents(run_dir / "sample.x.results", serialized_results));
  return results;
}

// Runs the given command, returning the command's stdout if successful.
absl::StatusOr<std::string> RunCommand(
    std::string_view desc, const SampleRunner::Commands::Command& command,
    std::vector<std::string> args, const std::filesystem::path& run_dir,
    const SampleOptions& options) {
  if (std::holds_alternative<SampleRunner::Commands::Callable>(command)) {
    return std::get<SampleRunner::Commands::Callable>(command)(args, run_dir,
                                                               options);
  }
  CHECK(std::holds_alternative<std::filesystem::path>(command));
  std::filesystem::path executable = std::get<std::filesystem::path>(command);
  std::string basename = executable.filename();

  std::vector<std::shared_ptr<RE2>> filters;
  for (const KnownFailure& filter : options.known_failures()) {
    if (filter.tool == nullptr || RE2::FullMatch(basename, *filter.tool)) {
      filters.emplace_back(filter.stderr_regex);
    }
  }

  std::vector<std::string> argv = {executable.string()};
  absl::c_move(std::move(args), std::back_inserter(argv));
  argv.push_back("--logtostderr");

  // TODO(epastor): We should probably inject these, rather than have them
  // grabbed from the command line inside of this library.
  if (int64_t verbosity = absl::GetFlag(FLAGS_v); verbosity > 0) {
    argv.push_back(absl::StrCat("--v=", verbosity));
  }
  if (std::string vmodule = absl::GetFlag(FLAGS_vmodule); !vmodule.empty()) {
    argv.push_back(absl::StrCat("--vmodule=", absl::GetFlag(FLAGS_vmodule)));
  }

  std::optional<absl::Duration> timeout =
      options.timeout_seconds().has_value()
          ? std::make_optional(absl::Seconds(*options.timeout_seconds()))
          : std::nullopt;
  XLS_VLOG(1) << "Running: " << desc;
  Stopwatch timer;
  XLS_ASSIGN_OR_RETURN(SubprocessResult result,
                       InvokeSubprocess(argv, run_dir, timeout));
  std::string command_string = absl::StrJoin(argv, " ");
  if (result.timeout_expired) {
    if (!options.timeout_seconds().has_value()) {
      return absl::DeadlineExceededError(
          absl::StrCat("Subprocess call timed out: ", command_string));
    }
    return absl::DeadlineExceededError(
        absl::StrCat("Subprocess call timed out after ",
                     *options.timeout_seconds(), " seconds: ", command_string));
  }
  absl::Duration elapsed = timer.GetElapsedTime();
  XLS_RETURN_IF_ERROR(SetFileContents(
      run_dir / absl::StrCat(basename, ".stderr"), result.stderr));
  if (XLS_VLOG_IS_ON(4)) {
    // stdout and stderr can be long so split them by line to avoid clipping.
    XLS_VLOG(4) << basename << " stdout:";
    XLS_VLOG_LINES(4, result.stdout);

    XLS_VLOG(4) << basename << " stderr:";
    XLS_VLOG_LINES(4, result.stderr);
  }
  XLS_VLOG(1) << desc << " complete, elapsed " << elapsed;
  if (!result.normal_termination) {
    return absl::InternalError(
        absl::StrCat("Subprocess call failed: ", command_string));
  }
  if (result.exit_status != EXIT_SUCCESS) {
    if (absl::c_any_of(filters, [&](const std::shared_ptr<RE2>& re) {
          return RE2::PartialMatch(result.stderr, *re);
        })) {
      return absl::FailedPreconditionError(
          absl::StrFormat("%s returned a non-zero exit status (%d) but failure "
                          "was suppressed due to stderr regexp",
                          executable.string(), result.exit_status));
    }
    return absl::InternalError(
        absl::StrCat(executable.string(), " returned non-zero exit status (",
                     result.exit_status, "): ", command_string));
  }
  return result.stdout;
}

// Converts the DSLX file to an IR file with a function as the top whose
// filename is returned.
absl::StatusOr<std::filesystem::path> DslxToIrFunction(
    const std::filesystem::path& input_path, const SampleOptions& options,
    const std::filesystem::path& run_dir,
    const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command =
      commands.ir_converter_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kIrConverterMainPath));
  }

  std::vector<std::string> args;
  absl::c_copy(options.ir_converter_args(), std::back_inserter(args));
  args.push_back("--warnings_as_errors=false");
  args.push_back(input_path.string());
  XLS_ASSIGN_OR_RETURN(
      std::string ir_text,
      RunCommand("Converting DSLX to IR", *command, args, run_dir, options));
  XLS_VLOG(3) << "Unoptimized IR:\n" << ir_text;

  std::filesystem::path ir_path = run_dir / "sample.ir";
  XLS_RETURN_IF_ERROR(SetFileContents(ir_path, ir_text));
  return ir_path;
}

// Parses a line-delimited sequence of text-formatted values.
//
// Example of expected input:
//   bits[32]:0x42
//   bits[32]:0x123
absl::StatusOr<std::vector<dslx::InterpValue>> ParseValues(std::string_view s) {
  std::vector<dslx::InterpValue> values;
  for (std::string_view line : absl::StrSplit(s, '\n')) {
    line = absl::StripAsciiWhitespace(line);
    if (line.empty()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Value value, xls::Parser::ParseTypedValue(line));
    XLS_ASSIGN_OR_RETURN(dslx::InterpValue interp_value,
                         dslx::ValueToInterpValue(value));
    values.push_back(std::move(interp_value));
  }
  return values;
}

// Evaluate the IR file with a function as its top and return the result Values.
absl::StatusOr<std::vector<dslx::InterpValue>> EvaluateIrFunction(
    const std::filesystem::path& ir_path,
    const std::filesystem::path& args_path, bool use_jit,
    const SampleOptions& options, const std::filesystem::path& run_dir,
    const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command =
      commands.eval_ir_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kEvalIrMainPath));
  }

  XLS_ASSIGN_OR_RETURN(
      std::string results_text,
      RunCommand(absl::StrFormat("Evaluating IR file (%s): %s",
                                 (use_jit ? "JIT" : "interpreter"), ir_path),
                 *command,
                 {
                     absl::StrCat("--input_file=", args_path.string()),
                     absl::StrFormat("--%suse_llvm_jit", use_jit ? "" : "no"),
                     ir_path,
                 },
                 run_dir, options));
  XLS_RETURN_IF_ERROR(SetFileContents(
      absl::StrCat(ir_path.string(), ".results"), results_text));
  return ParseValues(results_text);
}

absl::StatusOr<std::filesystem::path> Codegen(
    const std::filesystem::path& ir_path,
    absl::Span<const std::string> codegen_args, const SampleOptions& options,
    const std::filesystem::path& run_dir,
    const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command =
      commands.codegen_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kCodegenMainPath));
  }

  std::vector<std::string> args = {
      "--output_signature_path=module_sig.textproto",
      "--delay_model=unit",
  };
  args.insert(args.end(), codegen_args.begin(), codegen_args.end());
  args.push_back(ir_path.string());
  XLS_ASSIGN_OR_RETURN(
      std::string verilog_text,
      RunCommand("Generating Verilog", *command, args, run_dir, options));
  XLS_VLOG(3) << "Verilog:\n" << verilog_text;
  std::filesystem::path verilog_path =
      run_dir / (options.use_system_verilog() ? "sample.sv" : "sample.v");
  XLS_RETURN_IF_ERROR(SetFileContents(verilog_path, verilog_text));
  return verilog_path;
}

// Optimizes the IR file and returns the resulting filename.
absl::StatusOr<std::filesystem::path> OptimizeIr(
    const std::filesystem::path& ir_path, const SampleOptions& options,
    const std::filesystem::path& run_dir,
    const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command = commands.ir_opt_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kIrOptMainPath));
  }

  XLS_ASSIGN_OR_RETURN(
      std::string opt_ir_text,
      RunCommand("Optimizing IR", *command, {ir_path}, run_dir, options));
  XLS_VLOG(3) << "Optimized IR:\n" << opt_ir_text;
  std::filesystem::path opt_ir_path = run_dir / "sample.opt.ir";
  XLS_RETURN_IF_ERROR(SetFileContents(opt_ir_path, opt_ir_text));
  return opt_ir_path;
}

// Simulates the Verilog file representing a function and returns the results.
absl::StatusOr<std::vector<dslx::InterpValue>> SimulateFunction(
    const std::filesystem::path& verilog_path,
    const std::filesystem::path& module_sig_path,
    const std::filesystem::path& args_path, const SampleOptions& options,
    const std::filesystem::path& run_dir,
    const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command =
      commands.simulate_module_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kSimulateModuleMainPath));
  }

  std::vector<std::string> simulator_args = {
      absl::StrCat("--signature_file=", module_sig_path.string()),
      absl::StrCat("--args_file=", args_path.string()),
  };
  XLS_RETURN_IF_ERROR(CheckSimulator(options.simulator()));
  if (!options.simulator().empty()) {
    simulator_args.push_back(
        absl::StrCat("--verilog_simulator=", options.simulator()));
  }
  simulator_args.push_back(verilog_path.string());

  XLS_ASSIGN_OR_RETURN(
      std::string results_text,
      RunCommand(absl::StrCat("Simulating Verilog ", verilog_path.string()),
                 *command, simulator_args, run_dir, options));
  XLS_RETURN_IF_ERROR(SetFileContents(
      absl::StrCat(verilog_path.string(), ".results"), results_text));
  return ParseValues(results_text);
}

// Compares a set of results as for equality.
//
// Each entry in the map is sequence of Values generated from some source
// (e.g., interpreting the optimized IR). Each sequence of Values is compared
// for equality.
absl::Status CompareResultsProc(
    const absl::flat_hash_map<
        std::string, absl::flat_hash_map<std::string, std::vector<Value>>>&
        results) {
  if (results.empty()) {
    return absl::OkStatus();
  }

  std::deque<std::string> stages;
  for (const auto& [stage, _] : results) {
    stages.push_back(stage);
  }
  std::sort(stages.begin(), stages.end());

  std::string reference = stages.front();
  stages.pop_front();

  const absl::flat_hash_map<std::string, std::vector<Value>>&
      all_channel_values_ref = results.at(reference);

  for (std::string_view name : stages) {
    const absl::flat_hash_map<std::string, std::vector<Value>>&
        all_channel_values = results.at(name);
    if (all_channel_values_ref.size() != all_channel_values.size()) {
      std::vector<std::string> ref_channel_names;
      for (const auto& [channel_name, _] : all_channel_values_ref) {
        ref_channel_names.push_back(channel_name);
      }
      std::vector<std::string> channel_names;
      for (const auto& [channel_name, _] : all_channel_values) {
        channel_names.push_back(channel_name);
      }
      constexpr auto quote_formatter = [](std::string* out,
                                          const std::string& s) {
        absl::StrAppend(out, "'", s, "'");
      };
      return absl::InvalidArgumentError(absl::StrFormat(
          "Results for %s has %d channel(s), %s has %d "
          "channel(s). The IR channel names in %s are: [%s]. "
          "The IR channel names in %s are: [%s].",
          reference, all_channel_values_ref.size(), name,
          all_channel_values.size(), reference,
          absl::StrJoin(ref_channel_names, ", ", quote_formatter), name,
          absl::StrJoin(channel_names, ", ", quote_formatter)));
    }

    for (const auto& [channel_name_ref, channel_values_ref] :
         all_channel_values_ref) {
      auto it = all_channel_values.find(channel_name_ref);
      if (it == all_channel_values.end()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("A channel named %s is present in %s, but it is "
                            "not present in %s.",
                            channel_name_ref, reference, name));
      }
      const std::vector<Value>& channel_values = it->second;
      if (channel_values_ref.size() != channel_values.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "In %s, channel '%s' has %d entries. However, in %s, channel "
            "'%s' "
            "has %d entries.",
            reference, channel_name_ref, channel_values_ref.size(), name,
            channel_name_ref, channel_values.size()));
      }
      for (int i = 0; i < channel_values_ref.size(); ++i) {
        if (channel_values[i] != channel_values_ref[i]) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "In %s, at position %d channel '%s' has value "
              "%s. However, in %s, the value is %s.",
              reference, i, channel_name_ref,
              dslx::ValueToInterpValue(channel_values_ref[i])->ToString(), name,
              dslx::ValueToInterpValue(channel_values[i])->ToString()));
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<std::string, dslx::InterpValue>>
ConvertChannelValues(
    dslx::Proc* proc, const dslx::ImportData& import_data,
    const dslx::TypecheckedModule& tm,
    const std::vector<std::vector<dslx::InterpValue>>& input_channel_values) {
  XLS_ASSIGN_OR_RETURN(dslx::TypeInfo * proc_type_info,
                       tm.type_info->GetTopLevelProcTypeInfo(proc));
  std::vector<dslx::ProcInstance> proc_instances;
  absl::flat_hash_map<std::string, dslx::InterpValue> converted_channel_values;
  // Positional indexes of the input channels in the config function.
  std::vector<int64_t> in_chan_indexes;

  for (int64_t index = 0; index < proc->config().params().size(); ++index) {
    // Currently, only channels are supported as parameters to the config
    // function of a proc.
    dslx::Param* param = proc->config().params().at(index);
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(param->type_annotation());
    if (channel_type == nullptr) {
      return absl::InternalError(
          "Only channels are supported as parameters to the config function of "
          "a proc");
    }
    if (channel_type->direction() != dslx::ChannelDirection::kIn) {
      continue;
    }
    converted_channel_values.insert(
        {param->identifier(), dslx::InterpValue::MakeChannel()});
    in_chan_indexes.push_back(index);
  }

  std::vector<std::unique_ptr<dslx::Type>> channel_payload_types(
      in_chan_indexes.size());
  for (int64_t index = 0; index < in_chan_indexes.size(); ++index) {
    XLS_ASSIGN_OR_RETURN(
        dslx::Type * type,
        proc_type_info->GetItemOrError(
            proc->config().params().at(in_chan_indexes[index])));
    dslx::ChannelType* channel_type = dynamic_cast<dslx::ChannelType*>(type);
    if (channel_type == nullptr) {
      return absl::InternalError(
          "Only channels are supported as parameters to the config function of "
          "a proc");
    }
    channel_payload_types[index] = channel_type->payload_type().CloneToUnique();
  }

  for (const std::vector<dslx::InterpValue>& values : input_channel_values) {
    CHECK_EQ(in_chan_indexes.size(), values.size())
        << "The input channel count should match the args count.";
    for (int64_t index = 0; index < values.size(); ++index) {
      dslx::Param* param = proc->config().params().at(in_chan_indexes[index]);
      XLS_ASSIGN_OR_RETURN(
          dslx::InterpValue payload_value,
          SignConvertValue(*channel_payload_types[index], values[index]));
      converted_channel_values.at(param->identifier())
          .GetChannelOrDie()
          ->push_back(payload_value);
    }
  }
  return converted_channel_values;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<dslx::InterpValue>>>
RunProc(dslx::Proc* proc, dslx::ImportData& import_data,
        const dslx::TypecheckedModule& tm,
        const absl::flat_hash_map<std::string, dslx::InterpValue>&
            input_channel_values,
        int64_t proc_ticks) {
  XLS_ASSIGN_OR_RETURN(dslx::TypeInfo * proc_type_info,
                       tm.type_info->GetTopLevelProcTypeInfo(proc));
  std::vector<dslx::ProcInstance> proc_instances;
  std::vector<dslx::InterpValue> config_args;
  // Positional indexes of the output channels in the config function.
  std::vector<int64_t> out_chan_indexes;
  // The mapping of the channels in the output_channel_names follow the mapping
  // of out_chan_indexes. For example, out_channel_names[i] refers to same
  // channel at out_chan_indexes[i].
  std::vector<std::string> out_ir_channel_names;

  std::string module_name = proc->owner()->name();
  for (int64_t index = 0; index < proc->config().params().size(); ++index) {
    dslx::Param* param = proc->config().params().at(index);
    // Currently, only channels are supported as parameters to the config
    // function of a proc.
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(param->type_annotation());
    if (channel_type == nullptr) {
      return absl::InternalError(
          "Only channels are supported as parameters to the config function of "
          "a proc");
    }
    if (channel_type->direction() == dslx::ChannelDirection::kIn) {
      config_args.push_back(input_channel_values.at(param->identifier()));
    } else if (channel_type->direction() == dslx::ChannelDirection::kOut) {
      config_args.push_back(dslx::InterpValue::MakeChannel());
      out_chan_indexes.push_back(index);
      out_ir_channel_names.push_back(absl::StrCat(
          module_name, "__", proc->config().params().at(index)->identifier()));
    }
  }

  XLS_RETURN_IF_ERROR(dslx::ProcConfigBytecodeInterpreter::EvalSpawn(
      &import_data, proc_type_info, /*caller_bindings=*/std::nullopt,
      /*callee_bindings=*/std::nullopt, std::nullopt, proc, config_args,
      &proc_instances));

  // Currently a single proc is supported.
  CHECK_EQ(proc_instances.size(), 1);
  for (int i = 0; i < proc_ticks; i++) {
    XLS_RETURN_IF_ERROR(proc_instances[0].Run().status());
  }

  // TODO(vmirian): Ideally, the result should be a tuple containing two
  // tuples. The first entry is the result of the next function, the second is
  // the results of the output channels. Collect the result from the next
  // function.
  absl::flat_hash_map<std::string, std::vector<dslx::InterpValue>>
      all_channel_values;
  for (int64_t index = 0; index < out_chan_indexes.size(); ++index) {
    std::shared_ptr<dslx::InterpValue::Channel> channel =
        config_args[out_chan_indexes[index]].GetChannelOrDie();
    all_channel_values[out_ir_channel_names[index]] =
        std::vector<dslx::InterpValue>(channel->begin(), channel->end());
  }
  return all_channel_values;
}

// Interprets a DSLX module with proc as the top, returning the resulting
// Values.
absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
InterpretDslxProc(std::string_view text, std::string_view top_name,
                  const ArgsBatch& args_batch, int tick_count,
                  const std::filesystem::path& run_dir) {
  dslx::ImportData import_data = dslx::CreateImportData(
      GetDefaultDslxStdlibPath(),
      /*additional_search_paths=*/{}, dslx::kDefaultWarningsSet);
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule tm,
      dslx::ParseAndTypecheck(text, "sample.x", "sample", &import_data));

  std::optional<dslx::ModuleMember*> module_member =
      tm.module->FindMemberWithName(top_name);
  XLS_RET_CHECK(module_member.has_value());
  dslx::ModuleMember* member = module_member.value();
  XLS_RET_CHECK(std::holds_alternative<dslx::Proc*>(*member));
  dslx::Proc* proc = std::get<dslx::Proc*>(*member);

  absl::flat_hash_map<std::string, dslx::InterpValue> converted_channel_values;
  XLS_ASSIGN_OR_RETURN(converted_channel_values,
                       ConvertChannelValues(proc, import_data, tm, args_batch));
  absl::flat_hash_map<std::string, std::vector<dslx::InterpValue>> dslx_results;
  XLS_ASSIGN_OR_RETURN(
      dslx_results,
      RunProc(proc, import_data, tm, converted_channel_values, tick_count));

  absl::flat_hash_map<std::string, std::vector<Value>> ir_channel_values;
  for (const auto& [key, values] : dslx_results) {
    XLS_ASSIGN_OR_RETURN(ir_channel_values[key],
                         dslx::InterpValue::ConvertValuesToIr(values));
  }
  XLS_RETURN_IF_ERROR(SetFileContents(
      run_dir / "sample.x.results", ChannelValuesToString(ir_channel_values)));

  return ir_channel_values;
}

absl::StatusOr<std::filesystem::path> DslxToIrProc(
    const std::filesystem::path& dslx_path, const SampleOptions& options,
    const std::filesystem::path& run_dir,
    const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command =
      commands.ir_converter_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kIrConverterMainPath));
  }

  std::vector<std::string> args;
  absl::c_copy(options.ir_converter_args(), std::back_inserter(args));
  args.push_back("--warnings_as_errors=false");
  args.push_back(dslx_path);
  XLS_ASSIGN_OR_RETURN(
      std::string ir_text,
      RunCommand("Converting DSLX to IR", *command, args, run_dir, options));
  XLS_VLOG(3) << "Unoptimized IR:\n" << ir_text;
  std::filesystem::path ir_path = run_dir / "sample.ir";
  XLS_RETURN_IF_ERROR(SetFileContents(ir_path, ir_text));
  return ir_path;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
EvaluateIrProc(const std::filesystem::path& ir_path, int64_t tick_count,
               const std::filesystem::path& ir_channel_values_path,
               bool use_jit, const SampleOptions& options,
               const std::filesystem::path& run_dir,
               const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command =
      commands.eval_proc_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kEvalProcMainPath));
  }

  std::string_view evaluation_type = use_jit ? "JIT" : "interpreter";
  std::string desc =
      absl::StrFormat("Evaluating IR file (%s): %s", evaluation_type, ir_path);
  std::string_view backend_type = use_jit ? "serial_jit" : "ir_interpreter";
  std::vector<std::string> args = {
      absl::StrCat("--inputs_for_all_channels=",
                   ir_channel_values_path.string()),
      absl::StrCat("--ticks=", tick_count),
      absl::StrCat("--backend=", backend_type),
      ir_path,
  };
  XLS_ASSIGN_OR_RETURN(std::string results_text,
                       RunCommand(desc, *command, args, run_dir, options));
  XLS_RETURN_IF_ERROR(SetFileContents(
      absl::StrCat(ir_path.string(), ".results"), results_text));
  absl::flat_hash_map<std::string, std::vector<Value>> ir_channel_values;
  XLS_ASSIGN_OR_RETURN(ir_channel_values,
                       ParseChannelValues(results_text, tick_count));
  return ir_channel_values;
}

// Returns a output-channel-count map from an output-channel-values map.
absl::flat_hash_map<std::string, int64_t> GetOutputChannelCounts(
    const absl::flat_hash_map<std::string, std::vector<Value>>&
        output_channel_values) {
  absl::flat_hash_map<std::string, int64_t> output_channel_counts;
  for (const auto& [channel_name, channel_values] : output_channel_values) {
    output_channel_counts[channel_name] = channel_values.size();
  }
  return output_channel_counts;
}

// Returns a string representation of the output-channel-count map.
//
// The string format is output_channel_name=count for each entry in the map. The
// entries of the map are comma separated. For example, given an
// output-channel-count map:
//
//   {{foo, 42}, {bar,64}}
//
// the string representation is:
//
//   foo=42,bar=64
std::string GetOutputChannelToString(
    const absl::flat_hash_map<std::string, int64_t>& output_channel_counts) {
  std::vector<std::string> output_channel_counts_strings;
  for (const auto& [channel_name, count] : output_channel_counts) {
    output_channel_counts_strings.push_back(
        absl::StrCat(channel_name, "=", count));
  }
  return absl::StrJoin(output_channel_counts_strings, ",");
}

// Simulates the Verilog file representing a proc and returns the resulting
// Values.
absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
SimulateProc(const std::filesystem::path& verilog_path,
             const std::filesystem::path& module_sig_path,
             const std::filesystem::path& ir_channel_values_path,
             std::string_view output_channel_counts,
             const SampleOptions& options, const std::filesystem::path& run_dir,
             const SampleRunner::Commands& commands) {
  std::optional<SampleRunner::Commands::Command> command =
      commands.simulate_module_main;
  if (!command.has_value()) {
    XLS_ASSIGN_OR_RETURN(command, GetXlsRunfilePath(kSimulateModuleMainPath));
  }

  std::vector<std::string> simulator_args = {
      absl::StrCat("--signature_file=", module_sig_path.string()),
      absl::StrCat("--channel_values_file=", ir_channel_values_path.string()),
      absl::StrCat("--output_channel_counts=", output_channel_counts),
  };
  if (!options.simulator().empty()) {
    XLS_RETURN_IF_ERROR(CheckSimulator(options.simulator()));
    simulator_args.push_back(
        absl::StrCat("--verilog_simulator=", options.simulator()));
  }
  simulator_args.push_back(verilog_path);

  XLS_ASSIGN_OR_RETURN(
      std::string results_text,
      RunCommand(absl::StrCat("Simulating Verilog ", verilog_path.string()),
                 *command, simulator_args, run_dir, options));
  XLS_RETURN_IF_ERROR(SetFileContents(
      absl::StrCat(verilog_path.string(), ".results"), results_text));

  return ParseChannelValues(results_text);
}

}  // namespace

absl::Status SampleRunner::Run(const Sample& sample) {
  std::filesystem::path input_path = run_dir_;
  if (sample.options().input_is_dslx()) {
    input_path /= "sample.x";
  } else {
    input_path /= "sample.ir";
  }

  XLS_RETURN_IF_ERROR(SetFileContents(input_path, sample.input_text()));

  std::filesystem::path options_path = run_dir_ / "options.pbtxt";
  XLS_RETURN_IF_ERROR(
      SetFileContents(options_path, sample.options().ToPbtxt()));

  std::filesystem::path args_path = run_dir_ / "args.txt";
  XLS_RETURN_IF_ERROR(
      SetFileContents(args_path, ArgsBatchToText(sample.args_batch())));

  std::optional<std::filesystem::path> ir_channel_names_path = std::nullopt;
  if (!sample.ir_channel_names().empty()) {
    ir_channel_names_path = run_dir_ / "ir_channel_names.txt";
    XLS_RETURN_IF_ERROR(
        SetFileContents(*ir_channel_names_path,
                        IrChannelNamesToText(sample.ir_channel_names())));
  }

  return RunFromFiles(input_path, options_path, args_path,
                      ir_channel_names_path);
}

absl::Status SampleRunner::RunFromFiles(
    const std::filesystem::path& input_path,
    const std::filesystem::path& options_path,
    const std::optional<std::filesystem::path>& args_path,
    const std::optional<std::filesystem::path>& ir_channel_names_path) {
  XLS_VLOG(1) << "Running sample in directory " << run_dir_;
  XLS_VLOG(1) << "Reading sample files.";

  XLS_ASSIGN_OR_RETURN(std::string options_text, GetFileContents(options_path));
  XLS_ASSIGN_OR_RETURN(SampleOptions options,
                       SampleOptions::FromPbtxt(options_text));

  XLS_RETURN_IF_ERROR(
      SetFileContents(run_dir_ / "revision.txt", GetRevision()));

  absl::Status status;
  switch (options.sample_type()) {
    case fuzzer::SampleType::SAMPLE_TYPE_FUNCTION:
      status = RunFunction(input_path, options, args_path);
      break;
    case fuzzer::SampleType::SAMPLE_TYPE_PROC:
      status = RunProc(input_path, options, args_path, ir_channel_names_path);
      break;
    default:
      status = absl::InvalidArgumentError(
          "Unsupported sample type: " +
          fuzzer::SampleType_Name(options.sample_type()));
      break;
  }
  if (!status.ok()) {
    XLS_LOG(ERROR) << "Exception when running sample: " << status.ToString();
    XLS_RETURN_IF_ERROR(
        SetFileContents(run_dir_ / "exception.txt", status.ToString()));
  }
  if (status.code() == absl::StatusCode::kFailedPrecondition) {
    XLS_LOG(ERROR)
        << "Precondition failed, sample is not valid in the fuzz domain due to "
        << status;
    status = absl::OkStatus();
  }
  return status;
}

absl::Status SampleRunner::RunFunction(
    const std::filesystem::path& input_path, const SampleOptions& options,
    const std::optional<std::filesystem::path>& args_path) {
  XLS_ASSIGN_OR_RETURN(std::string input_text, GetFileContents(input_path));

  std::optional<ArgsBatch> args_batch = std::nullopt;
  if (args_path.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::string args_text, GetFileContents(*args_path));
    XLS_ASSIGN_OR_RETURN(args_batch, dslx::ParseArgsBatch(args_text));
  }

  absl::flat_hash_map<std::string, std::vector<dslx::InterpValue>> results;

  std::filesystem::path ir_path;
  if (options.input_is_dslx()) {
    if (args_batch.has_value()) {
      XLS_VLOG(1) << "Interpreting DSLX file.";
      Stopwatch t;
      XLS_ASSIGN_OR_RETURN(
          results["interpreted DSLX"],
          InterpretDslxFunction(input_text, "main", *args_batch, run_dir_));
      absl::Duration elapsed = t.GetElapsedTime();
      XLS_VLOG(1) << "Interpreting DSLX complete, elapsed: " << elapsed;
      timing_.set_interpret_dslx_ns(absl::ToInt64Nanoseconds(elapsed));
    }

    if (!options.convert_to_ir()) {
      return absl::OkStatus();
    }

    Stopwatch t;
    XLS_ASSIGN_OR_RETURN(
        ir_path, DslxToIrFunction(input_path, options, run_dir_, commands_));
    timing_.set_convert_ir_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));
  } else {
    ir_path = run_dir_ / "sample.ir";
    XLS_RETURN_IF_ERROR(SetFileContents(ir_path, input_text));
  }

  if (args_path.has_value()) {
    Stopwatch t;

    // Unconditionally evaluate with the interpreter even if using the JIT. This
    // exercises the interpreter and serves as a reference.
    XLS_ASSIGN_OR_RETURN(results["evaluated unopt IR (interpreter)"],
                         EvaluateIrFunction(ir_path, *args_path, false, options,
                                            run_dir_, commands_));
    timing_.set_unoptimized_interpret_ir_ns(
        absl::ToInt64Nanoseconds(t.GetElapsedTime()));

    if (options.use_jit()) {
      XLS_ASSIGN_OR_RETURN(results["evaluated unopt IR (JIT)"],
                           EvaluateIrFunction(ir_path, *args_path, true,
                                              options, run_dir_, commands_));
      timing_.set_unoptimized_jit_ns(
          absl::ToInt64Nanoseconds(t.GetElapsedTime()));
    }
  }

  if (options.optimize_ir()) {
    Stopwatch t;
    XLS_ASSIGN_OR_RETURN(std::filesystem::path opt_ir_path,
                         OptimizeIr(ir_path, options, run_dir_, commands_));
    timing_.set_optimize_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));

    if (args_path.has_value()) {
      if (options.use_jit()) {
        t.Reset();
        XLS_ASSIGN_OR_RETURN(results["evaluated opt IR (JIT)"],
                             EvaluateIrFunction(opt_ir_path, *args_path, true,
                                                options, run_dir_, commands_));
        timing_.set_optimized_jit_ns(
            absl::ToInt64Nanoseconds(t.GetElapsedTime()));
      }
      t.Reset();
      XLS_ASSIGN_OR_RETURN(results["evaluated opt IR (interpreter)"],
                           EvaluateIrFunction(opt_ir_path, *args_path, false,
                                              options, run_dir_, commands_));
      timing_.set_optimized_interpret_ir_ns(
          absl::ToInt64Nanoseconds(t.GetElapsedTime()));
    }

    if (options.codegen()) {
      t.Reset();
      XLS_ASSIGN_OR_RETURN(std::filesystem::path verilog_path,
                           Codegen(opt_ir_path, options.codegen_args(), options,
                                   run_dir_, commands_));
      timing_.set_codegen_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));

      if (options.simulate()) {
        XLS_RET_CHECK(args_path.has_value());
        t.Reset();
        XLS_ASSIGN_OR_RETURN(
            results["simulated"],
            SimulateFunction(verilog_path, "module_sig.textproto", *args_path,
                             options, run_dir_, commands_));
        timing_.set_simulate_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));
      }
    }
  }

  absl::flat_hash_map<std::string, absl::Span<const dslx::InterpValue>>
      results_spans(results.begin(), results.end());
  return CompareResultsFunction(
      results_spans, args_batch.has_value() ? &*args_batch : nullptr);
}

absl::Status SampleRunner::RunProc(
    const std::filesystem::path& input_path, const SampleOptions& options,
    const std::optional<std::filesystem::path>& args_path,
    const std::optional<std::filesystem::path>& ir_channel_names_path) {
  XLS_ASSIGN_OR_RETURN(std::string input_text, GetFileContents(input_path));

  std::optional<ArgsBatch> args_batch = std::nullopt;
  if (args_path.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::string args_text, GetFileContents(*args_path));
    XLS_ASSIGN_OR_RETURN(args_batch, dslx::ParseArgsBatch(args_text));
  }

  std::optional<std::vector<std::string>> ir_channel_names = std::nullopt;
  if (ir_channel_names_path.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::string ir_channel_names_text,
                         GetFileContents(*ir_channel_names_path));
    ir_channel_names = ParseIrChannelNames(ir_channel_names_text);
  }

  std::string ir_channel_values_file_content;
  if (args_batch.has_value() && ir_channel_names.has_value()) {
    absl::flat_hash_map<std::string, std::vector<dslx::InterpValue>>
        all_channel_values;
    for (const std::vector<dslx::InterpValue>& channel_values : *args_batch) {
      if (channel_values.size() != ir_channel_names->size()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid number of values in args_batch sample; "
                            "expected: %d, got: %d",
                            ir_channel_names->size(), channel_values.size()));
      }
      for (int i = 0; i < ir_channel_names->size(); ++i) {
        all_channel_values[ir_channel_names->at(i)].push_back(
            channel_values[i]);
      }
    }

    absl::flat_hash_map<std::string, std::vector<Value>> ir_channel_values;
    for (const auto& [key, values] : all_channel_values) {
      XLS_ASSIGN_OR_RETURN(ir_channel_values[key],
                           dslx::InterpValue::ConvertValuesToIr(values));
    }
    ir_channel_values_file_content = ChannelValuesToString(ir_channel_values);
  }
  std::filesystem::path ir_channel_values_path =
      run_dir_ / "channel_inputs.txt";
  XLS_RETURN_IF_ERROR(
      SetFileContents(ir_channel_values_path, ir_channel_values_file_content));

  // Special case: When there no inputs for a proc, typically when there are no
  // channels for a proc, tick_count results to 0. Set the tick_count to a
  // non-zero value to execute in the eval proc main (bypasses a restriction on
  // the number of ticks in eval proc main).
  int64_t tick_count =
      args_batch.has_value() ? std::max<int64_t>(args_batch->size(), 1) : 1;

  // Note the data is structure with a nested dictionary. The key of the
  // dictionary is the name of the XLS stage being evaluated. The value of the
  // dictionary is another dictionary where the key is the IR channel name. The
  // value of the nested dictionary is a sequence of values corresponding to the
  // channel.
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::vector<Value>>>
      results;
  std::optional<absl::flat_hash_map<std::string, std::vector<Value>>> reference;

  std::filesystem::path ir_path;
  if (options.input_is_dslx()) {
    if (args_batch.has_value()) {
      XLS_VLOG(1) << "Interpreting DSLX file.";
      Stopwatch t;
      XLS_ASSIGN_OR_RETURN(results["interpreted DSLX"],
                           InterpretDslxProc(input_text, "main", *args_batch,
                                             tick_count, run_dir_));
      reference = results["interpreted DSLX"];
      absl::Duration elapsed = t.GetElapsedTime();
      XLS_VLOG(1) << "Interpreting DSLX complete, elapsed: " << elapsed;
      timing_.set_interpret_dslx_ns(absl::ToInt64Nanoseconds(elapsed));
    }

    if (!options.convert_to_ir()) {
      return absl::OkStatus();
    }

    Stopwatch t;
    XLS_ASSIGN_OR_RETURN(
        ir_path, DslxToIrProc(input_path, options, run_dir_, commands_));
    timing_.set_convert_ir_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));
  } else {
    ir_path = run_dir_ / "sample.ir";
    XLS_RETURN_IF_ERROR(SetFileContents(ir_path, input_text));
  }

  if (args_batch.has_value()) {
    // Unconditionally evaluate with the interpreter even if using the JIT. This
    // exercises the interpreter and serves as a reference.
    Stopwatch t;
    XLS_ASSIGN_OR_RETURN(
        results["evaluated unopt IR (interpreter)"],
        EvaluateIrProc(ir_path, tick_count, ir_channel_values_path, false,
                       options, run_dir_, commands_));
    if (!reference.has_value()) {
      reference = results["evaluated unopt IR (interpreter)"];
    }
    timing_.set_unoptimized_interpret_ir_ns(
        absl::ToInt64Nanoseconds(t.GetElapsedTime()));

    if (options.use_jit()) {
      t.Reset();
      XLS_ASSIGN_OR_RETURN(
          results["evaluated unopt IR (JIT)"],
          EvaluateIrProc(ir_path, tick_count, ir_channel_values_path, true,
                         options, run_dir_, commands_));
      timing_.set_unoptimized_jit_ns(
          absl::ToInt64Nanoseconds(t.GetElapsedTime()));
    }
  }

  std::optional<std::filesystem::path> opt_ir_path = std::nullopt;
  if (options.optimize_ir()) {
    Stopwatch t;
    XLS_ASSIGN_OR_RETURN(opt_ir_path,
                         OptimizeIr(ir_path, options, run_dir_, commands_));
    timing_.set_optimize_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));

    if (args_batch.has_value()) {
      if (options.use_jit()) {
        t.Reset();
        XLS_ASSIGN_OR_RETURN(
            results["evaluated opt IR (JIT)"],
            EvaluateIrProc(*opt_ir_path, tick_count, ir_channel_values_path,
                           true, options, run_dir_, commands_));
        timing_.set_optimized_jit_ns(
            absl::ToInt64Nanoseconds(t.GetElapsedTime()));
      }

      t.Reset();
      XLS_ASSIGN_OR_RETURN(
          results["evaluated opt IR (interpreter)"],
          EvaluateIrProc(*opt_ir_path, tick_count, ir_channel_values_path,
                         false, options, run_dir_, commands_));
      timing_.set_optimized_interpret_ir_ns(
          absl::ToInt64Nanoseconds(t.GetElapsedTime()));

      if (options.codegen()) {
        t.Reset();
        XLS_ASSIGN_OR_RETURN(std::filesystem::path verilog_path,
                             Codegen(*opt_ir_path, options.codegen_args(),
                                     options, run_dir_, commands_));
        timing_.set_codegen_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));

        if (options.simulate()) {
          t.Reset();
          XLS_RET_CHECK(reference.has_value());
          absl::flat_hash_map<std::string, int64_t> output_channel_counts =
              GetOutputChannelCounts(*reference);
          std::string output_channel_counts_str =
              GetOutputChannelToString(output_channel_counts);
          XLS_ASSIGN_OR_RETURN(
              results["simulated"],
              SimulateProc(verilog_path, "module_sig.textproto",
                           ir_channel_values_path, output_channel_counts_str,
                           options, run_dir_, commands_));
          timing_.set_simulate_ns(absl::ToInt64Nanoseconds(t.GetElapsedTime()));
        }
      }
    }
  }

  return CompareResultsProc(results);
}

}  // namespace xls
