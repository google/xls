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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/jit/aot_compiler.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/type_layout.pb.h"

ABSL_FLAG(std::string, input, "", "Path to the IR to compile.");
ABSL_FLAG(std::string, top, "",
          "IR function to compile. "
          "If unspecified, the package top function will be used - "
          "in that case, the package-scoping mangling will be removed.");
ABSL_FLAG(std::string, output_object, "",
          "Path at which to write the output object file.");
ABSL_FLAG(std::string, output_proto, "",
          "Path at which to write the AotEntrypointProto describing the ABI of "
          "the generated object file.");
ABSL_FLAG(bool, generate_textproto, false,
          "Generate the AotEntrypointProto as a textproto");
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

// Returns the TypeLayouts for the arguments of `f`.
TypeLayoutsProto ArgLayouts(Function* f, LlvmTypeConverter& type_converter) {
  TypeLayoutsProto layouts_proto;
  for (Param* param : f->params()) {
    *layouts_proto.add_layouts() =
        type_converter.CreateTypeLayout(param->GetType()).ToProto();
  }
  return layouts_proto;
}

// Returns the TypeLayout for the return value of `f`.
TypeLayoutsProto ResultLayouts(Function* f, LlvmTypeConverter& type_converter) {
  TypeLayoutsProto layout_proto;
  *layout_proto.add_layouts() =
      type_converter.CreateTypeLayout(f->return_value()->GetType()).ToProto();
  return layout_proto;
}

absl::StatusOr<AotEntrypointProto> GenerateEntrypointProto(
    Package* package, Function* func, const JitObjectCode& object_code,
    bool include_msan) {
  AotEntrypointProto proto;
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<AotCompiler> aot_compiler,
      AotCompiler::Create(/*emit_msan=*/include_msan, /*opt_level=*/3));
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout data_layout,
                       aot_compiler->CreateDataLayout());
  LlvmTypeConverter type_converter(aot_compiler->GetContext(), data_layout);
  *proto.mutable_inputs_layout() = ArgLayouts(func, type_converter);
  *proto.mutable_outputs_layout() = ResultLayouts(func, type_converter);
  proto.add_outputs_names("result");
  proto.set_has_msan(include_msan);
  for (const Param* p : func->params()) {
    proto.add_inputs_names(p->name());
  }
  proto.set_xls_package_name(package->name());
  proto.set_xls_function_identifier(func->name());
  proto.set_function_symbol(object_code.function_base.function_name());
  absl::c_for_each(object_code.function_base.input_buffer_sizes(),
                   [&](int64_t i) { proto.add_input_buffer_sizes(i); });
  absl::c_for_each(
      object_code.function_base.input_buffer_preferred_alignments(),
      [&](int64_t i) { proto.add_input_buffer_alignments(i); });
  absl::c_for_each(
      object_code.function_base.input_buffer_abi_alignments(),
      [&](int64_t i) { proto.add_input_buffer_abi_alignments(i); });
  absl::c_for_each(object_code.function_base.output_buffer_sizes(),
                   [&](int64_t i) { proto.add_output_buffer_sizes(i); });
  absl::c_for_each(
      object_code.function_base.output_buffer_preferred_alignments(),
      [&](int64_t i) { proto.add_output_buffer_alignments(i); });
  absl::c_for_each(
      object_code.function_base.output_buffer_abi_alignments(),
      [&](int64_t i) { proto.add_output_buffer_abi_alignments(i); });
  if (object_code.function_base.HasPackedFunction()) {
    proto.set_packed_function_symbol(
        *object_code.function_base.packed_function_name());
    absl::c_for_each(
        object_code.function_base.packed_input_buffer_sizes(),
        [&](int64_t i) { proto.add_packed_input_buffer_sizes(i); });
    absl::c_for_each(
        object_code.function_base.packed_output_buffer_sizes(),
        [&](int64_t i) { proto.add_packed_output_buffer_sizes(i); });
  }

  proto.set_temp_buffer_size(object_code.function_base.temp_buffer_size());
  proto.set_temp_buffer_alignment(
      object_code.function_base.temp_buffer_alignment());
  for (const auto& [cont, node] :
       object_code.function_base.continuation_points()) {
    proto.mutable_continuation_point_node_ids()->at(cont) = node->id();
  }
  for (const auto& [chan_name, idx] :
       object_code.function_base.queue_indices()) {
    proto.mutable_channel_queue_indices()->at(chan_name) = idx;
  }
  return proto;
}

absl::Status RealMain(const std::string& input_ir_path, const std::string& top,
                      const std::string& output_object_path,
                      const std::string& output_proto_path, bool include_msan,
                      bool generate_textproto) {
  XLS_ASSIGN_OR_RETURN(std::string input_ir, GetFileContents(input_ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(input_ir, input_ir_path));

  Function* f;
  std::string package_prefix = absl::StrCat("__", package->name(), "__");
  if (top.empty()) {
    XLS_ASSIGN_OR_RETURN(f, package->GetTopAsFunction());
  } else {
    absl::StatusOr<Function*> maybe_f = package->GetFunction(top);
    if (maybe_f.ok()) {
      f = *maybe_f;
    } else {
      XLS_ASSIGN_OR_RETURN(
          f, package->GetFunction(absl::StrCat(package_prefix, top)));
    }
  }

  XLS_ASSIGN_OR_RETURN(
      JitObjectCode object_code,
      FunctionJit::CreateObjectCode(f, /*opt_level = */ 3, include_msan));
  XLS_RETURN_IF_ERROR(SetFileContents(
      output_object_path, std::string(object_code.object_code.begin(),
                                      object_code.object_code.end())));

  XLS_ASSIGN_OR_RETURN(
      AotEntrypointProto entrypoint,
      GenerateEntrypointProto(package.get(), f, object_code, include_msan));
  if (generate_textproto) {
    std::string text;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(entrypoint, &text));
    XLS_RETURN_IF_ERROR(SetFileContents(output_proto_path, text));
  } else {
    XLS_RETURN_IF_ERROR(
        SetFileContents(output_proto_path, entrypoint.SerializeAsString()));
  }

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
  std::string output_proto_path = absl::GetFlag(FLAGS_output_proto);
  QCHECK(!output_object_path.empty() && !output_proto_path.empty())
      << "All of --output_{object,proto} must be specified.";

  bool include_msan = absl::GetFlag(FLAGS_include_msan);
  absl::Status status =
      xls::RealMain(input_ir_path, top, output_object_path, output_proto_path,
                    include_msan, absl::GetFlag(FLAGS_generate_textproto));
  if (!status.ok()) {
    std::cout << status.message();
    return 1;
  }

  return 0;
}
