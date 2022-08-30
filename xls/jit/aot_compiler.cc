// Copyright 2022 The XLS Authors
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

// Driver for the XLS AOT compilation process.
// Uses the JIT to produce and object file, creates a header file and source to
// wrap (i.e., simplify) execution of the generated code, and writes the trio to
// disk.

#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"
#include "absl/strings/substitute.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/llvm_type_converter.h"

ABSL_FLAG(std::string, input, "", "Path to the IR to compile.");
ABSL_FLAG(std::string, top, "",
          "IR function to compile. "
          "If unspecified, the package top function will be used - "
          "in that case, the package-scoping mangling will be removed.");
ABSL_FLAG(std::string, namespaces, "",
          "Comma-separated list of namespaces into which to place the "
          "generated code. Earlier-specified namespaces enclose "
          "later-specified.");
ABSL_FLAG(std::string, output_object, "",
          "Path at which to write the output object file.");
ABSL_FLAG(std::string, output_header, "",
          "Path at which to write the output header file.");
ABSL_FLAG(std::string, output_source, "",
          "Path at which to write the output object-wrapping source file.");

namespace xls {

// Produces a simple header file containing a call with the same name as the
// target function (with the package name prefix removed).
absl::StatusOr<std::string> GenerateHeader(
    Package* p, Function* f, const std::vector<std::string>& namespaces) {
  // $0: Opening namespace(s)
  // $1: Closing namespace(s)
  // $2: Function name
  // $3: Function parameters
  constexpr absl::string_view kTemplate =
      R"(// AUTO-GENERATED FILE! DO NOT EDIT!
#include "absl/status/statusor.h"
#include "xls/ir/value.h"
{{open_ns}}
absl::StatusOr<xls::Value> {{wrapper_fn_name}}({{wrapper_params}});
{{close_ns}})";

  absl::flat_hash_map<std::string, std::string> substitution_map;
  std::string package_prefix = absl::StrCat("__", p->name(), "__");
  substitution_map["{{wrapper_fn_name}}"] =
      absl::StripPrefix(f->name(), package_prefix);

  std::vector<std::string> params;
  for (const Param* param : f->params()) {
    params.push_back(absl::StrCat("const ::xls::Value& ", param->name()));
  }
  substitution_map["{{wrapper_params}}"] = absl::StrJoin(params, ", ");

  if (namespaces.empty()) {
    substitution_map["{{open_ns}}"] = "";
    substitution_map["{{close_ns}}"] = "";
  } else {
    substitution_map["{{open_ns}}"] =
        absl::StrFormat("\nnamespace %s {\n", absl::StrJoin(namespaces, "::"));
    substitution_map["{{close_ns}}"] = absl::StrFormat(
        "\n}  // namespace %s\n", absl::StrJoin(namespaces, "::"));
  }

  return absl::StrReplaceAll(kTemplate, substitution_map);
}

// Generates a source file to wrap invocation of the generated function.
// This is more complicated than one might expect due to the fact that we need
// to use some LLVM internals (via the LlvmTypeConverter) to know how to convert
// an XLS Value into a bit buffer packed in the manner expected by LLVM.
// We also need to know the arguments and return types of the function for the
// same reason. This requires having those types described in this source file.
// To do that, we encode the Function's FunctionType as a text-format proto and
// decode it on first execution.
// On that note, we do as much work as we can in one-time initialization to
// reduce the tax paid during normal execution.
absl::StatusOr<std::string> GenerateWrapperSource(
    Package* p, Function* f, const std::string& header_path,
    const std::vector<std::string>& namespaces) {
  constexpr absl::string_view kTemplate =
      R"~(// AUTO-GENERATED FILE! DO NOT EDIT!
#include "{{header_path}}"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/events.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/jit/aot_runtime.h"
#include "xls/jit/jit_runtime.h"

extern "C" {
void {{extern_fn}}(const uint8_t* const*, uint8_t*, ::xls::InterpreterEvents*,
                   void*, ::xls::JitRuntime*);
}
{{open_ns}}
constexpr absl::string_view kFnTypeProto = R"({{type_textproto}})";

// We have to put "once" flags in different namespaces so their definitions
// don't collide.
namespace {{private_ns}} {
absl::once_flag once;
std::unique_ptr<xls::aot_compile::GlobalData> global_data;

void OnceInit() {
  global_data = xls::aot_compile::InitGlobalData(kFnTypeProto);
}
}  //  namespace {{private_ns}}

absl::StatusOr<::xls::Value> {{wrapper_fn_name}}({{wrapper_params}}) {
  absl::call_once({{private_ns}}::once, {{private_ns}}::OnceInit);

  ::xls::JitRuntime runtime(
      {{private_ns}}::global_data->data_layout,
      {{private_ns}}::global_data->type_converter.get());

{{arg_buffer_decls}}
  uint8_t* arg_buffers[] = {{arg_buffer_collector}};
  uint8_t result_buffer[{{result_size}}] = { 0 };
  XLS_RETURN_IF_ERROR(
      runtime.PackArgs(
          {{{param_names}}},
          {{private_ns}}::global_data->borrowed_param_types,
          absl::MakeSpan(arg_buffers)));
  ::xls::InterpreterEvents events;
  {{extern_fn}}(arg_buffers, result_buffer, &events, nullptr, &runtime);

  ::xls::Value result = runtime.UnpackBuffer(
      result_buffer, {{private_ns}}::global_data->fn_type->return_type());
  return result;
}

{{close_ns}}
)~";

  absl::flat_hash_map<std::string, std::string> substitution_map;
  substitution_map["{{header_path}}"] = header_path;
  substitution_map["{{extern_fn}}"] = f->name();

  if (namespaces.empty()) {
    substitution_map["{{open_ns}}"] = "";
    substitution_map["{{close_ns}}"] = "";
  } else {
    substitution_map["{{open_ns}}"] =
        absl::StrFormat("namespace %s {", absl::StrJoin(namespaces, "::"));
    substitution_map["{{close_ns}}"] =
        absl::StrFormat("}  // namespace %s", absl::StrJoin(namespaces, "::"));
  }

  llvm::LLVMContext ctx;
  auto error_or_target_builder =
      llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!error_or_target_builder) {
    return absl::InternalError(
        absl::StrCat("Unable to detect host: ",
                     llvm::toString(error_or_target_builder.takeError())));
  }

  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    return absl::InternalError(
        absl::StrCat("Unable to create target machine: ",
                     llvm::toString(error_or_target_machine.takeError())));
  }
  std::unique_ptr<llvm::TargetMachine> target_machine =
      std::move(error_or_target_machine.get());
  llvm::DataLayout data_layout = target_machine->createDataLayout();
  LlvmTypeConverter type_converter(&ctx, data_layout);
  int64_t return_type_bytes =
      type_converter.GetTypeByteSize(f->GetType()->return_type());

  std::vector<std::string> params;
  std::vector<std::string> param_names;
  std::vector<std::string> arg_buffer_decls;
  std::vector<std::string> arg_buffer_names;
  for (Param* param : f->params()) {
    params.push_back(absl::StrCat("const ::xls::Value& ", param->name()));
    param_names.push_back(std::string(param->name()));
    arg_buffer_decls.push_back(
        absl::StrFormat("  uint8_t %s_buffer[%d];", param->name(),
                        type_converter.GetTypeByteSize(param->GetType())));
    arg_buffer_names.push_back(absl::StrCat(param->name(), "_buffer"));
  }
  substitution_map["{{wrapper_params}}"] = absl::StrJoin(params, ", ");
  substitution_map["{{param_names}}"] = absl::StrJoin(param_names, ", ");
  substitution_map["{{arg_buffer_decls}}"] =
      absl::StrJoin(arg_buffer_decls, "\n");
  substitution_map["{{arg_buffer_collector}}"] =
      absl::StrFormat("{%s}", absl::StrJoin(arg_buffer_names, ", "));
  substitution_map["{{result_size}}"] = absl::StrCat(return_type_bytes);

  std::string type_textproto;
  google::protobuf::TextFormat::PrintToString(f->GetType()->ToProto(), &type_textproto);
  substitution_map["{{type_textproto}}"] = type_textproto;

  substitution_map["{{private_ns}}"] = absl::StrCat(f->name(), "__once_ns_");

  std::string package_prefix = absl::StrCat("__", p->name(), "__");
  substitution_map["{{wrapper_fn_name}}"] =
      absl::StripPrefix(f->name(), package_prefix);
  return absl::StrReplaceAll(kTemplate, substitution_map);
}

absl::Status RealMain(const std::string& input_ir_path, std::string top,
                      const std::string& output_object_path,
                      const std::string& output_header_path,
                      const std::string& output_source_path,
                      const std::vector<std::string>& namespaces) {
  XLS_ASSIGN_OR_RETURN(std::string input_ir, GetFileContents(input_ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(input_ir, input_ir_path));

  Function* f;
  std::string package_prefix = absl::StrCat("__", package->name(), "__");
  if (top.empty()) {
    XLS_ASSIGN_OR_RETURN(f, package->GetTopAsFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(f, package->GetFunction(top));
  }
  XLS_ASSIGN_OR_RETURN(std::vector<char> object_code,
                       FunctionJit::CreateObjectFile(f));
  XLS_RETURN_IF_ERROR(SetFileContents(
      output_object_path, std::string(object_code.begin(), object_code.end())));

  XLS_ASSIGN_OR_RETURN(std::string header_text,
                       GenerateHeader(package.get(), f, namespaces));
  XLS_RETURN_IF_ERROR(SetFileContents(output_header_path, header_text));

  XLS_ASSIGN_OR_RETURN(
      std::string source_text,
      GenerateWrapperSource(package.get(), f, output_header_path, namespaces));
  XLS_RETURN_IF_ERROR(SetFileContents(output_source_path, source_text));

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  std::string input_ir_path = absl::GetFlag(FLAGS_input);
  XLS_QCHECK(!input_ir_path.empty())
      << "--input must be specified." << std::endl;

  std::string top = absl::GetFlag(FLAGS_top);

  std::string output_object_path = absl::GetFlag(FLAGS_output_object);
  std::string output_header_path = absl::GetFlag(FLAGS_output_header);
  std::string output_source_path = absl::GetFlag(FLAGS_output_source);
  XLS_QCHECK(!output_object_path.empty() && !output_header_path.empty() &&
             !output_source_path.empty())
      << "All of --output_{object,header,source}_path must be specified.";

  std::vector<std::string> namespaces;
  std::string namespaces_string = absl::GetFlag(FLAGS_namespaces);
  if (!namespaces_string.empty()) {
    namespaces = absl::StrSplit(namespaces_string, ',');
  }
  absl::Status status =
      xls::RealMain(input_ir_path, top, output_object_path, output_header_path,
                    output_source_path, namespaces);
  if (!status.ok()) {
    std::cout << status.message();
    return 1;
  }

  return 0;
}
