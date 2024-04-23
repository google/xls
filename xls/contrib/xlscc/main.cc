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

// This file contains the  main function (entry point) of the xlscc
// front-end. It accepts as input a C/C++ file and produces as textual output
// the equivalent XLS intermediate representation (IR).

#include <filesystem>  // NOLINT
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/flags.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

static constexpr std::string_view kUsage = R"(
Generates XLS IR from a given C++ file, or generates Verilog in the special
case of a combinational module.

Emit XLS IR:
xlscc foo.cc

Emit combinational Verilog module:
xlscc foo.cc --block_pb block_info.pb

)";

ABSL_FLAG(std::string, out, "",
          "Path at which to output verilog / IR. Stdout if not specified.");

ABSL_FLAG(std::string, block_pb, "",
          "HLSBlock protobuf for generating as HLS block / XLS proc");

ABSL_FLAG(bool, block_pb_text, false, "Input HLSBlock protobuf as textproto?");

ABSL_FLAG(std::string, block_from_class, "",
          "Specifies the name of a top class from which to generate the "
          "HLSBlock. This class must contain the top method. ");

ABSL_FLAG(std::string, top, "", "Top function name");

ABSL_FLAG(std::string, package, "", "Package name to generate");

ABSL_FLAG(std::string, clang_args_file, "",
          "File containing on each line one command line argument for clang");

ABSL_FLAG(std::vector<std::string>, defines, std::vector<std::string>(),
          "Comma separated list of defines to pass to clang");

ABSL_FLAG(std::vector<std::string>, include_dirs, std::vector<std::string>(),
          "Comma separated list of include directories to pass to clang");

ABSL_FLAG(std::string, meta_out, "",
          "Path at which to output metadata protobuf");

ABSL_FLAG(bool, meta_out_text, false, "Output metadata as textproto?");

ABSL_FLAG(std::string, verilog_line_map_out, "",
          "Path at which to output Verilog line map protobuf");

ABSL_FLAG(bool, error_on_uninitialized, false,
          "Generate an error when a variable is uninitialized, or has the "
          "wrong number of initializers.");

ABSL_FLAG(bool, error_on_init_interval, false,
          "Generate an error when an initiation interval is requested greater "
          "than supported");

ABSL_FLAG(
    bool, generate_fsms_for_pipelined_loops, true,
    "Generate an FSM for pipelined loops. Non-FSM mode should be considered "
    "experimental, as it can generate semantically incorrect IR.");

ABSL_FLAG(
    bool, merge_states, true,
    "Merge states in FSMs for pipelined loops. Increases throughput at some"
    "potential cost.");

ABSL_FLAG(bool, split_states_on_channel_ops, true,
          "Split FSM states so that two IO operations on the same channel "
          "cannot be in the same FSM state. This does not try to determine"
          "mutual exclusion, so it can unnecessarily reduce throughput.");

ABSL_FLAG(int, top_level_init_interval, 1,
          "Initiation interval of block top level (Run/main function)");

ABSL_FLAG(int, max_unroll_iters, 1000,
          "Maximum number of iterations to allow loops to be unrolled");

ABSL_FLAG(int, warn_unroll_iters, 100,
          "Maximum number of iterations to allow loops to be unrolled");

ABSL_FLAG(int, z3_rlimit, 100000L,
          "rlimit to set for z3 solver (eg for loop unrolling)");

ABSL_FLAG(xlscc::ChannelStrictnessMap, channel_strictness,
          xlscc::ChannelStrictnessMap(),
          "Comma separated map of channels to strictness modes");

ABSL_FLAG(xls::ChannelStrictness, default_channel_strictness,
          xls::ChannelStrictness::kProvenMutuallyExclusive,
          "Default strictness for channels not otherwise specified");

ABSL_FLAG(std::string, io_op_token_ordering, "none",
          "none (default), channel_wise, lexical");

ABSL_FLAG(bool, debug_ir_trace_loop_context, false,
          "Generate IR traces for pipelined loop context variables.");

ABSL_FLAG(bool, debug_ir_trace_loop_control, false,
          "Generate IR traces for pipelined loop control.");

ABSL_FLAG(bool, debug_print_fsm_states, false,
          "Print FSM states to XLS_LOG (try --alsologtostderr).");

namespace xlscc {

static absl::Status Run(std::string_view cpp_path) {
  // Warnings should print by default
  absl::SetFlag(&FLAGS_logtostderr, true);

  xlscc::IOOpOrdering io_op_token_ordering = IOOpOrdering::kNone;

  if (absl::GetFlag(FLAGS_io_op_token_ordering) == "none") {
    io_op_token_ordering = IOOpOrdering::kNone;
  } else if (absl::GetFlag(FLAGS_io_op_token_ordering) == "channel_wise") {
    io_op_token_ordering = IOOpOrdering::kChannelWise;
  } else if (absl::GetFlag(FLAGS_io_op_token_ordering) == "lexical") {
    io_op_token_ordering = IOOpOrdering::kLexical;
  } else {
    std::cerr << "Unknown --io_op_token_ordering: "
              << absl::GetFlag(FLAGS_io_op_token_ordering) << '\n';
  }

  xlscc::ChannelOptions channel_options = {
      .default_strictness = absl::GetFlag(FLAGS_default_channel_strictness),
      .strictness_map = absl::GetFlag(FLAGS_channel_strictness).map,
  };

  DebugIrTraceFlags ir_trace_flags = DebugIrTraceFlags_None;

  if (absl::GetFlag(FLAGS_debug_ir_trace_loop_context)) {
    ir_trace_flags = static_cast<DebugIrTraceFlags>(
        ir_trace_flags | DebugIrTraceFlags_LoopContext);
  }
  if (absl::GetFlag(FLAGS_debug_ir_trace_loop_control)) {
    ir_trace_flags = static_cast<DebugIrTraceFlags>(
        ir_trace_flags | DebugIrTraceFlags_LoopControl);
  }
  if (absl::GetFlag(FLAGS_debug_print_fsm_states)) {
    ir_trace_flags = static_cast<DebugIrTraceFlags>(
        ir_trace_flags | DebugIrTraceFlags_FSMStates);
  }

  xlscc::Translator translator(
      absl::GetFlag(FLAGS_error_on_init_interval),
      absl::GetFlag(FLAGS_error_on_uninitialized),
      absl::GetFlag(FLAGS_generate_fsms_for_pipelined_loops),
      absl::GetFlag(FLAGS_merge_states),
      absl::GetFlag(FLAGS_split_states_on_channel_ops), ir_trace_flags,
      absl::GetFlag(FLAGS_max_unroll_iters),
      absl::GetFlag(FLAGS_warn_unroll_iters), absl::GetFlag(FLAGS_z3_rlimit),
      io_op_token_ordering);

  const std::string block_pb_name = absl::GetFlag(FLAGS_block_pb);

  const std::string block_from_class_name =
      absl::GetFlag(FLAGS_block_from_class);
  const bool block_from_class = !block_from_class_name.empty();

  HLSBlock block;
  if (!block_from_class) {
    if (!block_pb_name.empty()) {
      if (!absl::GetFlag(FLAGS_block_pb_text)) {
        std::ifstream file_in(block_pb_name);
        if (!block.ParseFromIstream(&file_in)) {
          return absl::InvalidArgumentError("Couldn't parse protobuf");
        }
      } else {
        XLS_RETURN_IF_ERROR(xls::ParseTextProtoFile(block_pb_name, &block));
      }
    }
  }

  const std::string top_function_name = absl::GetFlag(FLAGS_top);

  if (!top_function_name.empty()) {
    XLS_RETURN_IF_ERROR(
        translator.SelectTop(top_function_name, block_from_class_name));
  }

  std::vector<std::string> clang_argvs;

  const std::string clang_args_file = absl::GetFlag(FLAGS_clang_args_file);

  if (!clang_args_file.empty()) {
    XLS_ASSIGN_OR_RETURN(std::string clang_args_content,
                         xls::GetFileContents(clang_args_file));
    for (auto arg :
         absl::StrSplit(clang_args_content, '\n', absl::SkipWhitespace())) {
      clang_argvs.push_back(std::string(absl::StripAsciiWhitespace(arg)));
    }
  }

  for (std::string& def : absl::GetFlag(FLAGS_defines)) {
    clang_argvs.push_back(absl::StrCat("-D", def));
  }

  for (std::string& dir : absl::GetFlag(FLAGS_include_dirs)) {
    clang_argvs.push_back(absl::StrCat("-I", dir));
  }

  std::vector<std::string_view> clang_argv;
  clang_argv.reserve(clang_argvs.size());
  for (const auto& i : clang_argvs) {
    clang_argv.push_back(i);
  }

  std::cerr << "Parsing file '" << cpp_path << "' with clang..." << '\n';
  XLS_RETURN_IF_ERROR(translator.ScanFile(
      cpp_path, clang_argv.empty()
                    ? absl::Span<std::string_view>()
                    : absl::MakeSpan(&clang_argv[0], clang_argv.size())));

  XLS_ASSIGN_OR_RETURN(std::string top_name, translator.GetEntryFunctionName());

  std::string package_name = absl::GetFlag(FLAGS_package);

  if (package_name.empty()) {
    package_name = "my_package";
  }

  std::filesystem::path output_file(absl::GetFlag(FLAGS_out));

  std::filesystem::path output_absolute = output_file;
  if (output_file.is_relative()) {
    XLS_ASSIGN_OR_RETURN(std::filesystem::path cwd, xls::GetCurrentDirectory());
    output_absolute = cwd / output_file;
  }

  auto write_to_output = [&](std::string_view output) -> absl::Status {
    if (output_file.empty()) {
      std::cout << output;
    } else {
      XLS_RETURN_IF_ERROR(xls::SetFileContents(output_file, output));
    }
    return absl::OkStatus();
  };

  std::cerr << "Generating IR..." << '\n';
  xls::Package package(package_name);
  if (block_pb_name.empty()) {
    absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>
        top_channel_injections = {};
    XLS_RETURN_IF_ERROR(
        translator.GenerateIR_Top_Function(&package, top_channel_injections)
            .status());
    // TODO(seanhaskell): Simplify IR
    XLS_RETURN_IF_ERROR(package.SetTopByName(top_name));
    translator.AddSourceInfoToPackage(package);
    XLS_RETURN_IF_ERROR(write_to_output(absl::StrCat(package.DumpIr(), "\n")));
  } else {
    xls::Proc* proc = nullptr;

    if (!block_from_class) {
      XLS_ASSIGN_OR_RETURN(
          proc,
          translator.GenerateIR_Block(
              &package, block, absl::GetFlag(FLAGS_top_level_init_interval),
              channel_options));
    } else {
      XLS_ASSIGN_OR_RETURN(
          proc,
          translator.GenerateIR_BlockFromClass(
              &package, &block, absl::GetFlag(FLAGS_top_level_init_interval),
              channel_options));

      if (!block_pb_name.empty()) {
        XLS_RETURN_IF_ERROR(xls::SetTextProtoFile(block_pb_name, block));
      }
    }

    XLS_RETURN_IF_ERROR(package.SetTop(proc));
    std::cerr << "Saving Package IR..." << '\n';
    translator.AddSourceInfoToPackage(package);
    XLS_RETURN_IF_ERROR(write_to_output(absl::StrCat(package.DumpIr(), "\n")));
  }

  const std::string metadata_out_path = absl::GetFlag(FLAGS_meta_out);
  if (!metadata_out_path.empty()) {
    XLS_ASSIGN_OR_RETURN(xlscc_metadata::MetadataOutput meta,
                         translator.GenerateMetadata());

    if (absl::GetFlag(FLAGS_meta_out_text)) {
      XLS_RETURN_IF_ERROR(xls::SetTextProtoFile(metadata_out_path, meta));
    } else {
      std::ofstream ostr(metadata_out_path);
      if (!ostr.good()) {
        return absl::NotFoundError(absl::StrFormat(
            "Couldn't open metadata output path: %s", metadata_out_path));
      }
      if (!meta.SerializeToOstream(&ostr)) {
        return absl::UnknownError("Error writing metadata proto");
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xlscc

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s CPP_FILE",
                                      argv[0]);
  }
  std::string_view cpp_path = positional_arguments[0];
  return xls::ExitStatus(xlscc::Run(cpp_path));
}
