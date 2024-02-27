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

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/orc_jit.h"

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
ABSL_FLAG(
    std::string, header_include_path, "",
    "The path in the source tree at which the header should be #included. This "
    "is copied verbatim into an #include directive in the generated source "
    "file (the .cc file specified with --output_source). This flag is "
    "required.");
#ifdef ABSL_HAVE_MEMORY_SANITIZER
static constexpr bool kHasMsan = true;
#else
static constexpr bool kHasMsan = false;
#endif
ABSL_FLAG(bool, include_msan, kHasMsan,
          "Whether to include msan calls in the jitted code. This *must* match "
          "the configuration of the binary the jitted code is included in.");

namespace xls {
namespace {

// Returns the text serialization of the TypeLayouts for the arguments of
// `f`. Returned string is a text proto of type TypeLayoutsProto.
std::string ArgLayoutsSerialization(Function* f,
                                    LlvmTypeConverter& type_converter) {
  TypeLayoutsProto layouts_proto;
  for (Param* param : f->params()) {
    *layouts_proto.add_layouts() =
        type_converter.CreateTypeLayout(param->GetType()).ToProto();
  }
  std::string text;
  XLS_CHECK(google::protobuf::TextFormat::PrintToString(layouts_proto, &text));
  return text;
}

// Returns the text serialization of the TypeLayout for the return value of
// `f`. Returned string is a text proto of type TypeLayoutProto.
std::string ResultLayoutSerialization(Function* f,
                                      LlvmTypeConverter& type_converter) {
  TypeLayoutProto layout_proto =
      type_converter.CreateTypeLayout(f->return_value()->GetType()).ToProto();
  std::string text;
  XLS_CHECK(google::protobuf::TextFormat::PrintToString(layout_proto, &text));
  return text;
}

// Produces a simple header file containing a call with the same name as the
// target function (with the package name prefix removed).
absl::StatusOr<std::string> GenerateHeader(
    Package* p, Function* f, const std::vector<std::string>& namespaces) {
  constexpr std::string_view kTemplate =
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
    Function* f, const JitObjectCode& object_code,
    const std::string& header_path, const std::vector<std::string>& namespaces,
    bool include_msan) {
  constexpr std::string_view kTemplate =
      R"~(// AUTO-GENERATED FILE! DO NOT EDIT!
#include "{{header_path}}"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "xls/ir/events.h"
#include "xls/jit/type_layout.h"
#include "xls/jit/aot_runtime.h"

extern "C" {
void {{extern_fn}}(const uint8_t* const* inputs,
                   uint8_t* const* outputs,
                   uint8_t* temp_buffer,
                   ::xls::InterpreterEvents* events,
                   void* unused,
                   int64_t continuation_point);
}
{{open_ns}}

namespace {

#ifdef ABSL_HAVE_MEMORY_SANITIZER
static constexpr bool kTargetHasSanitizer = true;
#else
static constexpr bool kTargetHasSanitizer = false;
#endif
static constexpr bool kExternHasSanitizer = {{extern_sanitizer}};

static_assert(kTargetHasSanitizer == kExternHasSanitizer,
              "sanitizer states do not match!");

const char* kArgLayouts = R"|({{arg_layouts_proto}})|";
const char* kResultLayout = R"|({{result_layout_proto}})|";

const xls::aot_compile::FunctionTypeLayout& GetFunctionTypeLayout() {
  static std::unique_ptr<xls::aot_compile::FunctionTypeLayout> function_layout =
    xls::aot_compile::FunctionTypeLayout::Create(kArgLayouts, kResultLayout).value();
  return *function_layout;
}

}  //  namespace

absl::StatusOr<::xls::Value> {{wrapper_fn_name}}({{wrapper_params}}) {
{{arg_buffer_decls}}
  uint8_t* arg_buffers[] = {{arg_buffer_collector}};
  alignas({{result_buffer_align}}) uint8_t result_buffer[{{result_size}}];
  GetFunctionTypeLayout().ArgValuesToNativeLayout(
    {{{param_names}}}, absl::MakeSpan(arg_buffers, {{arg_count}}));

  uint8_t* output_buffers[1] = {result_buffer};
  alignas({{temp_buffer_align}}) uint8_t temp_buffer[{{temp_buffer_size}}];
  ::xls::InterpreterEvents events;
  {{extern_fn}}(arg_buffers, output_buffers, temp_buffer,
                &events, /*unused=*/nullptr, /*continuation_point=*/0);

  return GetFunctionTypeLayout().NativeLayoutResultToValue(result_buffer);
}

{{close_ns}}
)~";
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> orc_jit,
                       OrcJit::Create(/*opt_level=*/OrcJit::kDefaultOptLevel,
                                      /*emit_object_code=*/true,
                                      /*emit_msan=*/include_msan));
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout data_layout,
                       OrcJit::CreateDataLayout(/*aot_specification=*/true));
  LlvmTypeConverter type_converter(orc_jit->GetContext(), data_layout);

  absl::flat_hash_map<std::string, std::string> substitution_map;
  substitution_map["{{extern_sanitizer}}"] = include_msan ? "true" : "false";
  substitution_map["{{header_path}}"] = header_path;
  substitution_map["{{extern_fn}}"] = object_code.function_name;
  substitution_map["{{arg_layouts_proto}}"] =
      ArgLayoutsSerialization(f, type_converter);
  substitution_map["{{result_layout_proto}}"] =
      ResultLayoutSerialization(f, type_converter);
  substitution_map["{{temp_buffer_size}}"] =
      absl::StrCat(object_code.temp_buffer_size);
  substitution_map["{{temp_buffer_align}}"] =
      absl::StrCat(object_code.temp_buffer_alignment);

  if (namespaces.empty()) {
    substitution_map["{{open_ns}}"] = "";
    substitution_map["{{close_ns}}"] = "";
  } else {
    substitution_map["{{open_ns}}"] =
        absl::StrFormat("namespace %s {", absl::StrJoin(namespaces, "::"));
    substitution_map["{{close_ns}}"] =
        absl::StrFormat("}  // namespace %s", absl::StrJoin(namespaces, "::"));
  }

  int64_t return_type_bytes = object_code.return_buffer_size;

  std::vector<std::string> params;
  std::vector<std::string> param_names;
  std::vector<std::string> arg_buffer_decls;
  std::vector<std::string> arg_buffer_names;
  for (int64_t i = 0; i < f->params().size(); ++i) {
    Param* param = f->param(i);
    params.push_back(absl::StrCat("const ::xls::Value& ", param->name()));
    param_names.push_back(std::string(param->name()));
    arg_buffer_decls.push_back(
        absl::StrFormat("  alignas(%d) uint8_t %s_buffer[%d];",
                        object_code.parameter_alignments[i], param->name(),
                        object_code.parameter_buffer_sizes[i]));
    arg_buffer_names.push_back(absl::StrCat(param->name(), "_buffer"));
  }
  substitution_map["{{wrapper_params}}"] = absl::StrJoin(params, ", ");
  substitution_map["{{param_names}}"] = absl::StrJoin(param_names, ", ");
  substitution_map["{{arg_buffer_decls}}"] =
      absl::StrJoin(arg_buffer_decls, "\n");
  substitution_map["{{arg_buffer_collector}}"] =
      absl::StrFormat("{%s}", absl::StrJoin(arg_buffer_names, ", "));
  substitution_map["{{result_size}}"] = absl::StrCat(return_type_bytes);
  substitution_map["{{result_buffer_align}}"] =
      absl::StrCat(object_code.return_buffer_alignment);
  substitution_map["{{arg_count}}"] = absl::StrCat(params.size());

  std::string type_textproto;
  google::protobuf::TextFormat::PrintToString(f->GetType()->ToProto(), &type_textproto);
  substitution_map["{{type_textproto}}"] = type_textproto;

  std::string package_prefix = absl::StrCat("__", f->package()->name(), "__");
  substitution_map["{{wrapper_fn_name}}"] =
      absl::StripPrefix(f->name(), package_prefix);
  return absl::StrReplaceAll(kTemplate, substitution_map);
}

absl::Status RealMain(const std::string& input_ir_path, const std::string& top,
                      const std::string& output_object_path,
                      const std::string& output_header_path,
                      const std::string& output_source_path,
                      const std::string& header_include_path,
                      const std::vector<std::string>& namespaces,
                      bool include_msan) {
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
  XLS_ASSIGN_OR_RETURN(JitObjectCode object_code,
                       FunctionJit::CreateObjectCode(f));
  XLS_RETURN_IF_ERROR(SetFileContents(
      output_object_path, std::string(object_code.object_code.begin(),
                                      object_code.object_code.end())));

  XLS_ASSIGN_OR_RETURN(std::string header_text,
                       GenerateHeader(package.get(), f, namespaces));
  XLS_RETURN_IF_ERROR(SetFileContents(output_header_path, header_text));

  XLS_ASSIGN_OR_RETURN(
      std::string source_text,
      GenerateWrapperSource(f, object_code, header_include_path, namespaces,
                            include_msan));
  XLS_RETURN_IF_ERROR(SetFileContents(output_source_path, source_text));

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  std::string input_ir_path = absl::GetFlag(FLAGS_input);
  QCHECK(!input_ir_path.empty()) << "--input must be specified.";

  std::string top = absl::GetFlag(FLAGS_top);

  std::string output_object_path = absl::GetFlag(FLAGS_output_object);
  std::string output_header_path = absl::GetFlag(FLAGS_output_header);
  std::string output_source_path = absl::GetFlag(FLAGS_output_source);
  QCHECK(!output_object_path.empty() && !output_header_path.empty() &&
         !output_source_path.empty())
      << "All of --output_{object,header,source}_path must be specified.";

  std::string header_include_path = absl::GetFlag(FLAGS_header_include_path);
  QCHECK(!header_include_path.empty()) << "Must specify --header_include_path.";

  std::vector<std::string> namespaces;
  std::string namespaces_string = absl::GetFlag(FLAGS_namespaces);
  if (!namespaces_string.empty()) {
    namespaces = absl::StrSplit(namespaces_string, ',');
  }
  bool include_msan = absl::GetFlag(FLAGS_include_msan);
  absl::Status status = xls::RealMain(
      input_ir_path, top, output_object_path, output_header_path,
      output_source_path, header_include_path, namespaces, include_msan);
  if (!status.ok()) {
    std::cout << status.message();
    return 1;
  }

  return 0;
}
