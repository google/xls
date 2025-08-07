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
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/extract_interface.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/block_jit.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/observer.h"
#include "xls/jit/type_buffer_metadata.h"
#include "xls/jit/type_layout.pb.h"

ABSL_FLAG(std::string, input, "", "Path to the IR to compile.");
ABSL_FLAG(std::string, symbol_salt, "",
          "Additional text to append to symbol names to ensure no collisions.");
ABSL_FLAG(std::optional<std::string>, top, std::nullopt,
          "IR function to compile. "
          "If unspecified, the package top function will be used - "
          "in that case, the package-scoping mangling will be removed.");
ABSL_FLAG(std::optional<std::string>, output_object, std::nullopt,
          "Path at which to write the output object file.");
ABSL_FLAG(std::optional<std::string>, output_proto, std::nullopt,
          "Path at which to write the AotPackageEntrypointsProto describing "
          "the ABI of the generated object files.");
ABSL_FLAG(std::optional<std::string>, output_textproto, std::nullopt,
          "Path to write a textproto AotPackageEntrypointsProto describing the "
          "ABI of the generated object file.");
ABSL_FLAG(std::optional<std::string>, output_llvm_ir, std::nullopt,
          "Path at which to write the output llvm file.");
ABSL_FLAG(std::optional<std::string>, output_llvm_opt_ir, std::nullopt,
          "Path at which to write the output optimized llvm file.");
ABSL_FLAG(std::optional<std::string>, output_asm, std::nullopt,
          "Path at which to write the output optimized llvm file.");
ABSL_FLAG(int64_t, llvm_opt_level, xls::LlvmCompiler::kDefaultOptLevel,
          "The optimization level to use for the LLVM optimizer.");

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

class IntermediatesObserver final : public JitObserver {
 public:
  explicit IntermediatesObserver(JitObserverRequests req) : requests_(req) {}
  JitObserverRequests GetNotificationOptions() const override {
    return requests_;
  }
  // Called when a LLVM module has been created and is ready for optimization
  void UnoptimizedModule(const llvm::Module* module) override {
    llvm::raw_string_ostream ostream(unoptimized_);
    module->print(ostream, nullptr);
  }
  // Called when a LLVM module has been created and is ready for codegen
  void OptimizedModule(const llvm::Module* module) override {
    llvm::raw_string_ostream ostream(optimized_);
    module->print(ostream, nullptr);
  }
  // Called when a LLVM module has been compiled with the asm code.
  void AssemblyCodeString(const llvm::Module* module,
                          std::string_view asm_code) override {
    asm_ = asm_code;
  }

  std::string_view unoptimized_code() const { return unoptimized_; }
  std::string_view optimized_code() const { return optimized_; }
  std::string_view asm_code() const { return asm_; }

 private:
  JitObserverRequests requests_;
  std::string unoptimized_;
  std::string optimized_;
  std::string asm_;
};

absl::StatusOr<AotEntrypointProto> GenerateEntrypointProto(
    Package* package, const FunctionEntrypoint& entrypoint, bool include_msan,
    LlvmTypeConverter& type_converter) {
  FunctionBase* func = entrypoint.function;
  const JittedFunctionBase& object_code = entrypoint.jit_info;
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
    AotEntrypointProto::FunctionMetadataProto* function_metadata_proto =
        proto.mutable_function_metadata();
    *function_metadata_proto->mutable_function_interface() =
        ExtractFunctionInterface(func->AsFunctionOrDie());
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
    AotEntrypointProto::ProcMetadataProto* proc_metadata_proto =
        proto.mutable_proc_metadata();
    proc_metadata_proto->mutable_continuation_point_node_ids()->insert(
        object_code.continuation_points().begin(),
        object_code.continuation_points().end());
    proc_metadata_proto->mutable_channel_queue_indices()->insert(
        object_code.queue_indices().begin(), object_code.queue_indices().end());
    *proc_metadata_proto->mutable_proc_interface() =
        ExtractProcInterface(func->AsProcOrDie());
  } else {
    XLS_RET_CHECK(func->IsBlock());
    proto.set_type(AotEntrypointProto::BLOCK);
    for (InputPort* p : func->AsBlockOrDie()->GetInputPorts()) {
      proto.add_inputs_names(p->name());
      auto layout_proto =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
      *proto.mutable_inputs_layout()->add_layouts() = layout_proto;
    }
    for (OutputPort* p : func->AsBlockOrDie()->GetOutputPorts()) {
      proto.add_outputs_names(p->name());
      auto layout_proto =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
      *proto.mutable_outputs_layout()->add_layouts() = layout_proto;
    }
    AotEntrypointProto::BlockMetadataProto* block_metadata_proto =
        proto.mutable_block_metadata();

    for (const auto& [orig, translated] : entrypoint.register_aliases) {
      block_metadata_proto->mutable_register_aliases()->insert(
          {orig, translated});
    }
    for (const auto& [reg, ty] : entrypoint.added_registers) {
      block_metadata_proto->mutable_added_registers()->insert(
          {reg, ty->ToProto()});
    }
    *block_metadata_proto->mutable_block_interface() =
        ExtractBlockInterface(func->AsBlockOrDie());
  }
  proto.set_xls_package_name(package->name());
  proto.set_xls_function_identifier(func->name());
  proto.set_function_symbol(object_code.function_name());
  if (object_code.HasPackedFunction()) {
    proto.set_packed_function_symbol(*object_code.packed_function_name());
  }
  for (const TypeBufferMetadata& metadata :
       object_code.GetInputBufferMetadata()) {
    proto.add_input_buffer_sizes(metadata.size);
    proto.add_input_buffer_alignments(metadata.preferred_alignment);
    proto.add_input_buffer_abi_alignments(metadata.abi_alignment);
    if (object_code.HasPackedFunction()) {
      proto.add_packed_input_buffer_sizes(metadata.packed_size);
    }
  }
  for (const TypeBufferMetadata& metadata :
       object_code.GetOutputBufferMetadata()) {
    proto.add_output_buffer_sizes(metadata.size);
    proto.add_output_buffer_alignments(metadata.preferred_alignment);
    proto.add_output_buffer_abi_alignments(metadata.abi_alignment);
    if (object_code.HasPackedFunction()) {
      proto.add_packed_output_buffer_sizes(metadata.packed_size);
    }
  }

  proto.set_temp_buffer_size(object_code.temp_buffer_size());
  proto.set_temp_buffer_alignment(object_code.temp_buffer_alignment());
  return proto;
}

absl::Status RealMain(const std::string& input_ir_path,
                      const std::optional<std::string>& top,
                      const std::optional<std::string>& output_object_path,
                      const std::optional<std::string>& output_proto_path,
                      bool include_msan, int64_t llvm_opt_level,
                      const std::optional<std::string>& output_textproto_path,
                      const std::optional<std::string>& output_llvm_ir_path,
                      const std::optional<std::string>& output_llvm_opt_ir_path,
                      const std::optional<std::string>& output_asm_path,
                      std::string_view symbol_salt) {
  XLS_ASSIGN_OR_RETURN(std::string input_ir, GetFileContents(input_ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(input_ir, input_ir_path));

  IntermediatesObserver obs(
      {.unoptimized_module = output_llvm_ir_path.has_value(),
       .optimized_module = output_llvm_opt_ir_path.has_value(),
       .assembly_code_str = output_asm_path.has_value()});
  FunctionBase* f;
  std::string package_prefix = absl::StrCat("__", package->name(), "__");
  if (!top || top->empty()) {
    XLS_RET_CHECK(package->HasTop()) << "No top given.";
    f = *package->GetTop();
  } else {
    absl::StatusOr<FunctionBase*> maybe_f =
        package->GetFunctionBaseByName(*top);
    if (maybe_f.ok()) {
      f = *maybe_f;
    } else {
      XLS_ASSIGN_OR_RETURN(f, package->GetFunctionBaseByName(
                                  absl::StrCat(package_prefix, *top)));
    }
  }

  std::optional<JitObjectCode> object_code;
  if (f->IsFunction()) {
    XLS_ASSIGN_OR_RETURN(object_code, FunctionJit::CreateObjectCode(
                                          f->AsFunctionOrDie(), llvm_opt_level,
                                          include_msan, &obs, symbol_salt));
  } else if (f->IsProc()) {
    if (f->AsProcOrDie()->is_new_style_proc()) {
      XLS_ASSIGN_OR_RETURN(object_code, CreateProcAotObjectCode(
                                            f->AsProcOrDie(), llvm_opt_level,
                                            include_msan, &obs, symbol_salt));
    } else {
      // all procs
      XLS_ASSIGN_OR_RETURN(object_code, CreateProcAotObjectCode(
                                            package.get(), llvm_opt_level,
                                            include_msan, &obs, symbol_salt));
    }
  } else {
    XLS_ASSIGN_OR_RETURN(BlockElaboration elab,
                         BlockElaboration::Elaborate(f->AsBlockOrDie()));
    XLS_ASSIGN_OR_RETURN(object_code, BlockJit::CreateObjectCode(
                                          elab, llvm_opt_level, include_msan,
                                          &obs, symbol_salt));
  }
  AotPackageEntrypointsProto all_entrypoints;
  if (output_object_path) {
    XLS_RETURN_IF_ERROR(SetFileContents(
        *output_object_path, std::string(object_code->object_code.begin(),
                                         object_code->object_code.end())));
  }

  *all_entrypoints.mutable_data_layout() =
      object_code->data_layout.getStringRepresentation();

  auto context = std::make_unique<llvm::LLVMContext>();
  LlvmTypeConverter type_converter(context.get(), object_code->data_layout);
  for (const FunctionEntrypoint& oc : object_code->entrypoints) {
    XLS_ASSIGN_OR_RETURN(
        *all_entrypoints.add_entrypoint(),
        GenerateEntrypointProto(
            object_code->package ? object_code->package.get() : package.get(),
            oc, include_msan, type_converter));
  }
  if (output_textproto_path) {
    std::string text;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(all_entrypoints, &text));
    XLS_RETURN_IF_ERROR(SetFileContents(*output_textproto_path, text));
  }
  if (output_proto_path) {
    XLS_RETURN_IF_ERROR(SetFileContents(*output_proto_path,
                                        all_entrypoints.SerializeAsString()));
  }
  if (output_llvm_ir_path) {
    XLS_RETURN_IF_ERROR(
        SetFileContents(*output_llvm_ir_path, obs.unoptimized_code()));
  }
  if (output_llvm_opt_ir_path) {
    XLS_RETURN_IF_ERROR(
        SetFileContents(*output_llvm_opt_ir_path, obs.optimized_code()));
  }
  if (output_asm_path) {
    XLS_RETURN_IF_ERROR(SetFileContents(*output_asm_path, obs.asm_code()));
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  std::string input_ir_path = absl::GetFlag(FLAGS_input);
  QCHECK(!input_ir_path.empty()) << "--input must be specified.";

  std::optional<std::string> top = absl::GetFlag(FLAGS_top);

  std::optional<std::string> output_object_path =
      absl::GetFlag(FLAGS_output_object);
  std::optional<std::string> output_proto_path =
      absl::GetFlag(FLAGS_output_proto);

  bool include_msan = absl::GetFlag(FLAGS_include_msan);
  absl::Status status = xls::RealMain(
      input_ir_path, top, output_object_path, output_proto_path, include_msan,
      absl::GetFlag(FLAGS_llvm_opt_level),
      absl::GetFlag(FLAGS_output_textproto),
      absl::GetFlag(FLAGS_output_llvm_ir),
      absl::GetFlag(FLAGS_output_llvm_opt_ir), absl::GetFlag(FLAGS_output_asm),
      absl::GetFlag(FLAGS_symbol_salt));
  if (!status.ok()) {
    std::cout << status.message();
    return 1;
  }

  return 0;
}
