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

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/jit/type_layout.h"

namespace xls::aot_compile {

// Data structure for converting the arguments and return value of an XLS
// function between xls::Values the native data layout used by the JIT. Each
// instance is constructed for a particular xls::Function.
class FunctionTypeLayout {
 public:
  // Creates and returns a FunctionTypeLayout based on the TypeLayout
  // serializations. `serialized_arg_layouts` is a text serialization of a
  // TypeLayoutsProto, and `serialized_result_layout` is a text serialization of
  // a TypeLayoutProto.
  static absl::StatusOr<std::unique_ptr<FunctionTypeLayout>> Create(
      std::string_view serialized_arg_layouts,
      std::string_view serialized_result_layout);

  // Converts the given argument values and writes them into `arg_buffers` in
  // the native data layout used by the JIT.
  void ArgValuesToNativeLayout(absl::Span<const Value> args,
                               absl::Span<uint8_t* const> arg_buffers) const {
    for (int64_t i = 0; i < args.size(); ++i) {
      arg_layouts_[i].ValueToNativeLayout(args[i], arg_buffers[i]);
    }
  }

  // Converts the return value in `buffer` in native data layout to an
  // xls::Value.
  Value NativeLayoutResultToValue(const uint8_t* buffer) const {
    return result_layout_.NativeLayoutToValue(buffer);
  }

 private:
  FunctionTypeLayout(std::unique_ptr<Package> package,
                     std::vector<TypeLayout> arg_layouts,
                     TypeLayout result_layout)
      : package_(std::move(package)),
        arg_layouts_(std::move(arg_layouts)),
        result_layout_(std::move(result_layout)) {}

  // Dummy package used for owning Types required by the TypeLayout data
  // structures.
  std::unique_ptr<Package> package_;
  std::vector<TypeLayout> arg_layouts_;
  TypeLayout result_layout_;
};

}  // namespace xls::aot_compile

#endif  // XLS_JIT_AOT_RUNTIME_H_
