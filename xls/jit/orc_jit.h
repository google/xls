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

#ifndef XLS_JIT_ORC_JIT_H_
#define XLS_JIT_ORC_JIT_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/include/llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {

// A wrapper around ORC JIT which hides some of the internals of the LLVM
// interface.
class OrcJit {
 public:
  ~OrcJit();
  // Create an LLVM ORC JIT instance which compiles at the given optimization
  // level. If `emit_object_code` is true then `GetObjectCode` can be called
  // after compilation to get the object code.
  static absl::StatusOr<std::unique_ptr<OrcJit>> Create(int64_t opt_level,
                                                        bool emit_object_code);

  // Creates and returns a new LLVM module of the given name.
  std::unique_ptr<llvm::Module> NewModule(absl::string_view name);

  // Compiles the given LLVM module into the JIT's execution session.
  absl::Status CompileModule(std::unique_ptr<llvm::Module>&& module);

  // Returns the address of the given JIT'ed function.
  absl::StatusOr<llvm::JITTargetAddress> LoadSymbol(
      absl::string_view function_name);

  // Accessors to underlying LLVM guts.
  const llvm::DataLayout& GetDataLayout() const { return data_layout_; }
  LlvmTypeConverter& GetTypeConverter() { return *type_converter_; }
  llvm::LLVMContext* GetContext() { return context_.getContext(); }

  // Returns the object code which was created in the previous CompileModule
  // call (if `emit_object_code` is true).
  const std::vector<char>& GetObjectCode() { return object_code_; }

 private:
  OrcJit(int64_t opt_level, bool emit_object_code);

  absl::Status Init();

  // Method which optimizes the given module. Used within the JIT to form an IR
  // transform layer.
  llvm::Expected<llvm::orc::ThreadSafeModule> Optimizer(
      llvm::orc::ThreadSafeModule module,
      const llvm::orc::MaterializationResponsibility& responsibility);

  llvm::orc::ThreadSafeContext context_;
  llvm::orc::ExecutionSession execution_session_;
  llvm::orc::RTDyldObjectLinkingLayer object_layer_;
  llvm::orc::JITDylib& dylib_;

  int64_t opt_level_;
  bool emit_object_code_;

  std::unique_ptr<llvm::TargetMachine> target_machine_;
  llvm::DataLayout data_layout_;

  std::unique_ptr<llvm::orc::IRCompileLayer> compile_layer_;
  std::unique_ptr<llvm::orc::IRTransformLayer> transform_layer_;
  // If set, this contains the logic to emit object code.
  std::unique_ptr<llvm::orc::IRTransformLayer> object_code_layer_;

  std::unique_ptr<LlvmTypeConverter> type_converter_;

  // When `CompileModule` is called and `emit_object_code` is true, this vector
  // will be allocated and filled with the object code of the compiled module.
  std::vector<char> object_code_;
};

// Calls the dump method on the given LLVM object and returns the string.
template <typename T>
std::string DumpLlvmObjectToString(const T& llvm_object) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  llvm_object.print(ostream);
  ostream.flush();
  return buffer;
}

// Calls the dump method on the given LLVM module object and returns the string.
// DumpLlvmObjectToString cannot be used because modules are dumped slightly
// differently.
std::string DumpLlvmModuleToString(const llvm::Module& module);

}  // namespace xls

#endif  // XLS_JIT_ORC_JIT_H_
