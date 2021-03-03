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

#include <fstream>
#include <streambuf>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Stmt.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/passes/standard_pipeline.h"

const char kUsage[] = R"(
Generates XLS IR from a given C++ file, or generates Verilog in the special
case of a combinational module.

Emit XLS IR:
xlscc foo.cc

Emit combinational Verilog module:
xlscc foo.cc --block_pb block_info.pb

)";

ABSL_FLAG(std::string, module_name, "",
          "Explicit name to use for the generated module; if not provided the "
          "mangled IR function name is used");

ABSL_FLAG(std::string, block_pb, "",
          "HLSBlock protobuf for generating as HLS block / XLS proc");

ABSL_FLAG(std::string, top, "", "Top function name");

ABSL_FLAG(std::string, package, "", "Package name to generate");

ABSL_FLAG(std::string, clang_args_file, "",
          "File containing on each line one command line argument for clang");

namespace xlscc {

absl::Status Run(absl::string_view cpp_path) {
  xlscc::Translator translator;

  const std::string block_pb_name = absl::GetFlag(FLAGS_block_pb);

  HLSBlock block;
  if (!block_pb_name.empty()) {
    std::ifstream file_in(block_pb_name);
    if (!block.ParseFromIstream(&file_in)) {
      return absl::InvalidArgumentError("Couldn't parse protobuf");
    }
  }

  const std::string top_function_name = absl::GetFlag(FLAGS_top);

  if (!top_function_name.empty()) {
    XLS_RETURN_IF_ERROR(translator.SelectTop(top_function_name));
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

  std::vector<absl::string_view> clang_argv;
  for (size_t i = 0; i < clang_argvs.size(); ++i) {
    clang_argv.push_back(clang_argvs[i]);
  }

  std::cerr << "Parsing file '" << cpp_path << "' with clang..." << std::endl;
  XLS_RETURN_IF_ERROR(translator.ScanFile(
      cpp_path, clang_argv.empty()
                    ? absl::Span<absl::string_view>()
                    : absl::MakeSpan(&clang_argv[0], clang_argv.size())));

  XLS_ASSIGN_OR_RETURN(absl::StatusOr<string> top_name,
                       translator.GetEntryFunctionName());

  std::string package_name = absl::GetFlag(FLAGS_package);

  if (package_name.empty()) {
    package_name = "my_package";
  } else {
    package_name = top_name.value();
  }

  std::cerr << "Generating IR..." << std::endl;
  xls::Package package(package_name, top_name.value());
  if (block_pb_name.empty()) {
    XLS_RETURN_IF_ERROR(translator.GenerateIR_Top_Function(&package).status());
    // TODO(seanhaskell): Simplify IR
    std::cout << package.DumpIr() << std::endl;
  } else {
    XLS_ASSIGN_OR_RETURN(
        xls::Proc * proc,
        translator.GenerateIR_Block(&package, block,
                                    xlscc::XLSChannelMode::kAllSingleValue));

    XLS_RETURN_IF_ERROR(translator.InlineAllInvokes(&package));

    std::cout << package.DumpIr() << std::endl;

    absl::flat_hash_map<const xls::Channel*, xls::verilog::ProcPortType>
        channel_gen_types;

    for (const HLSChannel& channel : block.channels()) {
      XLS_ASSIGN_OR_RETURN(xls::Channel * xls_channel,
                           package.GetChannel(channel.name()));
      xls::verilog::ProcPortType port_type = xls::verilog::ProcPortType::kNull;
      switch (channel.type()) {
        case ChannelType::DIRECT_IN:
          port_type = xls::verilog::ProcPortType::kSimple;
          break;
        case ChannelType::FIFO:
          port_type = xls::verilog::ProcPortType::kReadyValid;
          break;
        default:
          return absl::UnimplementedError(
              absl::StrFormat("Channel %s has unknown type %i", channel.name(),
                              channel.type()));
      }
      channel_gen_types[xls_channel] = port_type;
    }

    std::cerr << "Generating Verilog..." << std::endl;
    XLS_ASSIGN_OR_RETURN(
        xls::verilog::ModuleGeneratorResult result,
        xls::verilog::GenerateCombinationalModuleFromProc(
            proc, channel_gen_types, false /*UseSystemVerilog*/));

    std::cout << result.verilog_text << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace xlscc

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s CPP_FILE",
                                          argv[0]);
  }
  absl::string_view cpp_path = positional_arguments[0];
  XLS_QCHECK_OK(xlscc::Run(cpp_path));
  return EXIT_SUCCESS;
}
