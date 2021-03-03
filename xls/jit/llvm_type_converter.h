// Copyright 2020 The XLS Authors
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

#ifndef XLS_JIT_LLVM_TYPE_CONVERTER_H_
#define XLS_JIT_LLVM_TYPE_CONVERTER_H_

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/Constants.h"
#include "llvm/include/llvm/IR/Module.h"
#include "xls/common/integral_types.h"
#include "xls/ir/function.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// LlvmTypeConverter handles the work of translating from XLS types and values
// into the corresponding LLVM elements.
//
// This class must live as long as its constructor argument module.
class LlvmTypeConverter {
 public:
  LlvmTypeConverter(llvm::LLVMContext* context,
                    const llvm::DataLayout& data_layout);

  llvm::Type* ConvertToLlvmType(const Type* type);

  // Converts the input XLS Value to an LLVM Constant of the specified type.
  absl::StatusOr<llvm::Constant*> ToLlvmConstant(llvm::Type* type,
                                                 const Value& value);
  absl::StatusOr<llvm::Constant*> ToLlvmConstant(const Type* type,
                                                 const Value& value);

  // Returns the number of bytes that LLVM will internally use to store the
  // given element. This is not simply the flat bit count of the type (rounded
  // up to 8 bits) - a type with four 6-bit members will be held in 4 i8s,
  // instead of the three that the flat bit count would suggest. The type width
  // rules aren't necessarily immediately obvious, but fortunately the
  // DataLayout object can handle ~all of the work for us.
  int64 GetTypeByteSize(const Type* type);

  // Returns a new Value representing the LLVM form of a Token.
  llvm::Value* GetToken();

  // Gets the LLVM type used to represent a Token.
  llvm::Type* GetTokenType();

 private:
  using TypeCache = absl::flat_hash_map<const Type*, llvm::Type*>;

  // Handles the special (and base) case of converting Bits types to LLVM.
  absl::StatusOr<llvm::Constant*> ToIntegralConstant(llvm::Type* type,
                                                     const Value& value);

  llvm::LLVMContext& context_;
  llvm::DataLayout data_layout_;

  // Cache of XLS -> LLVM type conversions.
  TypeCache type_cache_;
};

}  // namespace xls

#endif  // XLS_JIT_LLVM_TYPE_CONVERTER_H_
