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
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
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
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_proc_runtime.h"
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
          "Path at which to write the AotPackageEntrypointsProto describing "
          "the ABI of the generated object files.");
ABSL_FLAG(bool, generate_textproto, false,
          "Generate the AotPackageEntrypointsProto as a textproto");
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

absl::StatusOr<AotEntrypointProto> GenerateEntrypointProto(
    Package* package, FunctionBase* func, const JittedFunctionBase& object_code,
    bool include_msan, LlvmTypeConverter& type_converter) {
  AotEntrypointProto proto;
  proto.set_has_msan(include_msan);
  if (func->IsFunction()) {
    proto.set_type(AotEntrypointProto::FUNCTION);
    proto.add_outputs_names("result");
    for (const Param* p : func->params()) {
      proto.add_inputs_names(p->name());
      *proto.mutable_inputs_layout()->add_layouts() =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
    }
    *proto.mutable_outputs_layout()->add_layouts() =
        type_converter
            .CreateTypeLayout(func->AsFunctionOrDie()->GetType()->return_type())
            .ToProto();
  } else if (func->IsProc()) {
    proto.set_type(AotEntrypointProto::PROC);
    for (const Param* p : func->params()) {
      proto.add_inputs_names(p->name());
      proto.add_outputs_names(p->name());
      auto layout_proto =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
      *proto.mutable_inputs_layout()->add_layouts() = layout_proto;
      *proto.mutable_outputs_layout()->add_layouts() = layout_proto;
    }
  } else {
    return absl::UnimplementedError("block aot dumping unsupported!");
  }
  proto.set_xls_package_name(package->name());
  proto.set_xls_function_identifier(func->name());
  proto.set_function_symbol(object_code.function_name());
  absl::c_for_each(object_code.input_buffer_sizes(),
                   [&](int64_t i) { proto.add_input_buffer_sizes(i); });
  absl::c_for_each(object_code.input_buffer_preferred_alignments(),
                   [&](int64_t i) { proto.add_input_buffer_alignments(i); });
  absl::c_for_each(object_code.input_buffer_abi_alignments(), [&](int64_t i) {
    proto.add_input_buffer_abi_alignments(i);
  });
  absl::c_for_each(object_code.output_buffer_sizes(),
                   [&](int64_t i) { proto.add_output_buffer_sizes(i); });
  absl::c_for_each(object_code.output_buffer_preferred_alignments(),
                   [&](int64_t i) { proto.add_output_buffer_alignments(i); });
  absl::c_for_each(object_code.output_buffer_abi_alignments(), [&](int64_t i) {
    proto.add_output_buffer_abi_alignments(i);
  });
  if (object_code.HasPackedFunction()) {
    proto.set_packed_function_symbol(*object_code.packed_function_name());
    absl::c_for_each(object_code.packed_input_buffer_sizes(), [&](int64_t i) {
      proto.add_packed_input_buffer_sizes(i);
    });
    absl::c_for_each(object_code.packed_output_buffer_sizes(), [&](int64_t i) {
      proto.add_packed_output_buffer_sizes(i);
    });
  }

  proto.set_temp_buffer_size(object_code.temp_buffer_size());
  proto.set_temp_buffer_alignment(object_code.temp_buffer_alignment());
  for (const auto& [cont, node] : object_code.continuation_points()) {
    proto.mutable_continuation_point_node_ids()->insert({cont, node->id()});
  }
  for (const auto& [chan_name, idx] : object_code.queue_indices()) {
    proto.mutable_channel_queue_indices()->insert({chan_name, idx});
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

  FunctionBase* f;
  std::string package_prefix = absl::StrCat("__", package->name(), "__");
  if (top.empty()) {
    XLS_RET_CHECK(package->HasTop()) << "No top given.";
    f = *package->GetTop();
  } else {
    absl::StatusOr<FunctionBase*> maybe_f = package->GetFunctionBaseByName(top);
    if (maybe_f.ok()) {
      f = *maybe_f;
    } else {
      XLS_ASSIGN_OR_RETURN(
          f, package->GetFunctionBaseByName(absl::StrCat(package_prefix, top)));
    }
  }

  std::optional<JitObjectCode> object_code;
  if (f->IsFunction()) {
    XLS_ASSIGN_OR_RETURN(object_code, FunctionJit::CreateObjectCode(
                                          f->AsFunctionOrDie(),
                                          /*opt_level = */ 3, include_msan));
  } else if (f->IsProc()) {
    if (f->AsProcOrDie()->is_new_style_proc()) {
      XLS_ASSIGN_OR_RETURN(
          object_code, CreateProcAotObjectCode(f->AsProcOrDie(), include_msan));
    } else {
      // all procs
      XLS_ASSIGN_OR_RETURN(
          object_code, CreateProcAotObjectCode(package.get(), include_msan));
    }
  } else {
    return absl::UnimplementedError(
        "Dumping block jit code is not yet supported");
  }
  AotPackageEntrypointsProto all_entrypoints;
  XLS_RETURN_IF_ERROR(SetFileContents(
      output_object_path, std::string(object_code->object_code.begin(),
                                      object_code->object_code.end())));

  *all_entrypoints.mutable_data_layout() =
      object_code->data_layout.getStringRepresentation();

  auto context = std::make_unique<llvm::LLVMContext>();
  LlvmTypeConverter type_converter(context.get(), object_code->data_layout);
  for (const FunctionEntrypoint& oc : object_code->entrypoints) {
    XLS_ASSIGN_OR_RETURN(
        *all_entrypoints.add_entrypoint(),
        GenerateEntrypointProto(package.get(), oc.function, oc.jit_info,
                                include_msan, type_converter));
  }
  if (generate_textproto) {
    std::string text;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(all_entrypoints, &text));
    XLS_RETURN_IF_ERROR(SetFileContents(output_proto_path, text));
  } else {
    XLS_RETURN_IF_ERROR(SetFileContents(output_proto_path,
                                        all_entrypoints.SerializeAsString()));
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
