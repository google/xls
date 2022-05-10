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

// Common utilities needed by the dispatching logic for ahead-of-time compiled
// XLS designs.

#ifndef XLS_JIT_AOT_RUNTIME_H_
#define XLS_JIT_AOT_RUNTIME_H_

#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls::aot_compile {

// GlobalData holds information needed at AOT-compiled function invocation time
// for converting between XLS and LLVM argument types and executing the function
// itself. Rather than derive these statically-determinable values on every
// invocation, we one-time initialize and cache them.
struct GlobalData {
  // The overall LLVM context.
  std::unique_ptr<llvm::LLVMContext> ctx;
  // The LLVM understanding of how data is laid out in registers and memory.
  llvm::DataLayout data_layout;
  // Needed for converting between XLS and LLVM types.
  std::unique_ptr<::xls::LlvmTypeConverter> type_converter;
  // The IR package that will own fn_type below.
  ::xls::Package package;
  // The type of the [AOT-compiled] function being managed by this GlobalData.
  ::xls::FunctionType* fn_type;
  // Holds the types of the fn's args.
  std::vector<std::unique_ptr<::xls::Type>> param_types;
  // Pointers to the types above (needed to pack the arguments into a buffer).
  std::vector<::xls::Type*> borrowed_param_types;
  // The return type of the function.
  std::unique_ptr<::xls::Type> return_type;
};

// Performs [what should be] one-time initialization of a GlobalData function,
// given the specified text-format FunctionType protobuf.
std::unique_ptr<GlobalData> InitGlobalData(absl::string_view fn_type_textproto);

}  // namespace xls::aot_compile

#endif  // XLS_JIT_AOT_RUNTIME_H_
