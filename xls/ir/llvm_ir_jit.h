// Copyright 2020 Google LLC
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

#ifndef XLS_IR_LLVM_IR_JIT_H_
#define XLS_IR_LLVM_IR_JIT_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Target/TargetMachine.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/llvm_ir_runtime.h"
#include "xls/ir/llvm_type_converter.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"

namespace xls {

// This class provides a facility to execute XLS functions (on the host) by
// converting it to LLVM IR, compiling it, and finally executing it.
class LlvmIrJit {
 public:
  // Returns an object containing a host-compiled version of the specified XLS
  // function.
  static xabsl::StatusOr<std::unique_ptr<LlvmIrJit>> Create(
      Function* xls_function, int64 opt_level = 3);

  // Executes the compiled function with the specified arguments.
  xabsl::StatusOr<Value> Run(absl::Span<const Value> args);

  // As above, buth with arguments as key-value pairs.
  xabsl::StatusOr<Value> Run(
      const absl::flat_hash_map<std::string, Value>& kwargs);

  // Executes the compiled function with the specified arguments, but does not
  // unpack the result into a Value, instead populating the argument buffer with
  // the computation results. The caller can then overlay a View type
  // (from value_view.h) to consume the results.
  //
  // Argument packing and unpacking into and out of LLVM-space can consume a
  // surprisingly large amount of execution time. Deferring such transformations
  // (and applying views) can eliminate this overhead and still give access tor
  // result data. Users needing less performance can still use the
  // Value-returning methods above for code simplicity.
  absl::Status RunToBuffer(absl::Span<const Value> args,
                           absl::Span<uint8> result_buffer);

  // Returns the function that the JIT executes.
  Function* function() { return xls_function_; }

  // Gets the size of the compiled function's return type in bytes.
  int64 GetReturnTypeSize() { return return_type_bytes_; }

 private:
  explicit LlvmIrJit(Function* xls_function, int64 opt_level);

  // Performs non-trivial initialization (i.e., that which can fail).
  absl::Status Init();

  // Compiles the input function to host code.
  absl::Status CompileFunction();

  llvm::Expected<llvm::orc::ThreadSafeModule> Optimizer(
      llvm::orc::ThreadSafeModule module,
      const llvm::orc::MaterializationResponsibility& responsibility);

  llvm::orc::ThreadSafeContext context_;
  llvm::orc::ExecutionSession execution_session_;
  llvm::orc::RTDyldObjectLinkingLayer object_layer_;
  llvm::orc::JITDylib& dylib_;
  llvm::DataLayout data_layout_;

  std::unique_ptr<llvm::TargetMachine> target_machine_;
  std::unique_ptr<llvm::orc::IRCompileLayer> compile_layer_;
  std::unique_ptr<llvm::orc::IRTransformLayer> transform_layer_;

  Function* xls_function_;
  FunctionType* xls_function_type_;
  int64 opt_level_;

  // Size of the function's return type as flat bytes.
  int64 return_type_bytes_;
  // Cache for XLS type => LLVM type conversions.
  absl::flat_hash_map<const Type*, llvm::Type*> xls_to_llvm_type_;

  std::unique_ptr<LlvmTypeConverter> type_converter_;
  std::unique_ptr<LlvmIrRuntime> ir_runtime_;

  // When initialized, this points to the compiled output.
  void (*invoker_)(uint8** inputs, uint8* outputs);
};

xabsl::StatusOr<Value> CreateandRun(Function *xls_function, absl::Span<const Value> args);

}  // namespace xls

#endif  // XLS_IR_LLVM_IR_JIT_H_
