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

#ifndef XLS_FLOWS_IR_WRAPPER_H_
#define XLS_FLOWS_IR_WRAPPER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/warning_kind.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {

// This class provides a buffer and convenience functions to access a
// JitChannelQueue.
class JitChannelQueueWrapper {
 public:
  static absl::StatusOr<JitChannelQueueWrapper> Create(JitChannelQueue* queue,
                                                       JitRuntime* jit_runtime);

  // Get XLS type the queue stores.
  Type* GetType() const { return type_; }

  // Returns if the queue is empty.
  bool Empty() const { return (queue_ == nullptr) || (queue_->IsEmpty()); }

  // Write on the channel the value v.
  absl::Status Write(const Value& v);

  // Read on the channel the value v.
  absl::StatusOr<Value> Read();

  // Convenience function to write a uint64.
  absl::Status WriteWithUint64(uint64_t v);

  // Convenience function to read a uint64.
  absl::StatusOr<uint64_t> ReadWithUint64();

  // Return the buffer of the instance.
  absl::Span<uint8_t> buffer() { return absl::MakeSpan(buffer_); }

  // Write the data in the buffer to the channel.
  absl::Status Write(absl::Span<uint8_t> buffer);

  // Read the content of the channel into the buffer.
  absl::Status Read(absl::Span<uint8_t> buffer);

 private:
  // Pointer to the jit channel queue this object wraps.
  JitChannelQueue* queue_ = nullptr;

  // XLS type of the data to be sent/received from the channel.
  Type* type_ = nullptr;

  // Preallocated buffer sized to hold the data in LLVM representation.
  std::vector<uint8_t> buffer_;
};

// A class managing a dslx module and associated path.
class DslxModuleAndPath {
 public:
  // Gives up ownership the dslx module.
  std::unique_ptr<dslx::Module> GiveUpDslxModule() {
    return std::move(module_);
  }

  // Take ownership of a dslx module.
  void TakeDslxModule(std::unique_ptr<dslx::Module> module) {
    module_ = std::move(module);
    module_name_ = module_->name();
  }

  // Get Module.
  dslx::Module* GetDslxModule() { return module_.get(); }

  // Return module name.

  // Return path of the dslx module.
  std::string GetFilePath() const { return file_path_; }

  // Set new path of the dslx module.
  void SetFilePath(std::string_view path) { file_path_ = path; }

  // Take ownership of the dslx module and create a new object.
  static absl::StatusOr<DslxModuleAndPath> Create(
      std::unique_ptr<dslx::Module> module, std::string_view file_path);

  // Parse dslx file from path and create a new object.
  static absl::StatusOr<DslxModuleAndPath> Create(std::string_view module_name,
                                                  std::string_view file_path);

 private:
  DslxModuleAndPath() = default;

  std::unique_ptr<dslx::Module> module_;
  std::string module_name_;
  std::string file_path_;
};

// This class owns and is responsible for the flow to take ownership of a set
// of DSLX modules, compile/typecheck them, and convert them into an
// IR package.
//
// Additional convenience functions are available.
class IrWrapper {
 public:
  // Flags to control ir creation.
  // TODO(tedhong): 2022-10-14 - Convert this to a struct to enable greater
  // control over codegen options.
  enum class Flags {
    kDefault = 0x0,
    kSkipOpt = 0x1,
  };

  // Retrieve a specific dslx module.
  absl::StatusOr<dslx::Module*> GetDslxModule(std::string_view name) const;

  // Retrieve a specific top-level function from the compiled BOP IR.
  //
  // name is the unmangled name.
  absl::StatusOr<Function*> GetIrFunction(std::string_view name) const;

  // Retrieve a specific top-level proc from the compiled BOP IR.
  //
  // name is the unmangled name.
  absl::StatusOr<Proc*> GetIrProc(std::string_view name) const;

  // Retrieve top level package.
  absl::StatusOr<Package*> GetIrPackage() const;

  // Optimize the top-level package.

  // Retrieve and create (if needed) the JIT for the given function name.
  absl::StatusOr<FunctionJit*> GetAndMaybeCreateFunctionJit(
      std::string_view name);

  // Retrieve and create (if needed) the Proc runtime.
  absl::StatusOr<SerialProcRuntime*> GetAndMaybeCreateProcRuntime();

  // Retrieve JIT channel queue for the given channel name.
  absl::StatusOr<JitChannelQueue*> GetJitChannelQueue(
      std::string_view name) const;

  // Retrieve JIT channel queue wrapper for the given channel name and jit
  absl::StatusOr<JitChannelQueueWrapper> CreateJitChannelQueueWrapper(
      std::string_view name) const;

  // Takes ownership of a set of DSLX modules, converts to IR and creates
  // an IrWrapper object.
  static absl::StatusOr<IrWrapper> Create(
      std::string_view ir_package_name, DslxModuleAndPath top_module,
      std::vector<DslxModuleAndPath> import_modules,
      Flags flags = Flags::kDefault);

  static absl::StatusOr<IrWrapper> Create(
      std::string_view ir_package_name,
      std::unique_ptr<dslx::Module> top_module,
      std::string_view top_module_path,
      std::unique_ptr<dslx::Module> other_module = nullptr,
      std::string_view other_module_path = "", Flags flags = Flags::kDefault);

 private:
  // Construct this object with a default ImportData.
  explicit IrWrapper()
      : import_data_(dslx::CreateImportData(
            xls::kDefaultDslxStdlibPath,
            /*additional_search_paths=*/{},
            /*enabled_warnings=*/dslx::kAllWarningsSet)) {}

  // Pointers to the each of the DSLX modules explicitly given to this wrapper.
  //
  // Ownership of this and all other DSLX modules is with import_data_;
  dslx::Module* top_module_;
  std::vector<dslx::Module*> other_modules_;

  // Holds typechecked DSLX modules.
  dslx::ImportData import_data_;

  // IR Package.
  std::unique_ptr<Package> package_;

  // Holds pre-compiled IR Function Jit.
  absl::flat_hash_map<Function*, std::unique_ptr<FunctionJit>>
      pre_compiled_function_jit_;

  // Proc runtime.
  std::unique_ptr<SerialProcRuntime> proc_runtime_;
};

}  // namespace xls

#endif  // XLS_FLOWS_IR_WRAPPER_H_
