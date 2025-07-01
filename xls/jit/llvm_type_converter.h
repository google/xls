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

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/IR/Constant.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Type.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/type_buffer_metadata.h"
#include "xls/jit/type_layout.h"

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
  llvm::Type* ConvertToPointerToLlvmType(const Type* type) {
    return llvm::PointerType::get(ConvertToLlvmType(type), 0);
  }

  // Returns the LLVM type for the packed representation of the given XLS
  // type. A packed representation has all bits flattened into a bit vector
  // (LLVM integer type).
  llvm::Type* ConvertToPackedLlvmType(const Type* type) const;

  // Returns the width of the LLVM packed type in bits.
  int64_t PackedLlvmTypeWidth(const Type* type) const;

  int64_t GetPackedTypeByteSize(const Type* type) const;

  // Converts the input XLS Value to an LLVM Constant of the specified type.
  absl::StatusOr<llvm::Constant*> ToLlvmConstant(llvm::Type* type,
                                                 const Value& value) const;
  absl::StatusOr<llvm::Constant*> ToLlvmConstant(const Type* type,
                                                 const Value& value);

  // Returns a constant zero of the given type.
  static llvm::Constant* ZeroOfType(llvm::Type* type);

  // Returns the number of bytes that LLVM will internally use to store the
  // given element. This is not simply the flat bit count of the type (rounded
  // up to 8 bits) - a type with four 6-bit members will be held in 4 i8s,
  // instead of the three that the flat bit count would suggest. The type width
  // rules aren't necessarily immediately obvious, but fortunately the
  // DataLayout object can handle ~all of the work for us.
  int64_t GetTypeByteSize(const Type* type);

  // Returns the preferred alignment for the given type.
  int64_t GetTypePreferredAlignment(const Type* type);

  // Returns the alignment requirement for the given type.
  int64_t GetTypeAbiAlignment(const Type* type);

  TypeBufferMetadata GetTypeBufferMetadata(const Type* type);

  // Returns the next position (starting from offset) where LLVM would consider
  // an object of the given type to have ended; specifically, the next position
  // that matches the greater of the stack alignment and the type's preferred
  // alignment. As above, the rules aren't immediately obvious, but the
  // DataLayout object takes care of the details.
  int64_t AlignFor(const Type* type, int64_t offset);

  // Returns a new Value representing the LLVM form of a Token.
  llvm::Value* GetToken() const;

  // Gets the LLVM type used to represent a Token.
  llvm::Type* GetTokenType() const;

  // Return the bit count of the LLVM representation of the corresponding
  // XLS type (bit count). LLVM types are padded out to a power of two
  // which speeds up compilation and likely avoids latent bugs on dusty code
  // paths for supporting odd bit widths. For example, a three bit XLS value:
  //
  //   0bXYZ
  //
  // May be represented as an i8 in LLVM with the high bits zero-ed out:
  //
  //   0b0000_0XYZ
  int64_t GetLlvmBitCount(const BitsType* type) const {
    return GetLlvmBitCount(type->bit_count());
  }
  int64_t GetLlvmBitCount(int64_t xls_bit_count) const;

  llvm::Type* GetLlvmBitsType(int64_t xls_bit_count) const {
    return llvm::Type::getIntNTy(context_, GetLlvmBitCount(xls_bit_count));
  }

  // Zeros the padding bits of the given LLVM value representing an XLS value of
  // the given XLS type. Bits-typed XLS values are padded out to powers of two.
  llvm::Value* ClearPaddingBits(llvm::Value* value, Type* xls_type,
                                llvm::IRBuilder<>& builder);

  // Returns a mask which is 0 in padded bit positions and 1 in non-padding bit
  // positions for the LLVM representation of the given XLS type. For example,
  // if the XLS type is bits[3] represented using an i8 in LLVM, PaddingMask
  // would return:
  //
  //   i8:0b0000_0111
  llvm::Value* PaddingMask(Type* xls_type, llvm::IRBuilder<>& builder);

  // Returns the bitwise NOT of padding mask, e.g., 0b1111_1000.
  llvm::Value* InvertedPaddingMask(Type* xls_type, llvm::IRBuilder<>& builder);

  // Converts the given LLVM value representing and XLS value of the given type
  // into a signed representation. This involves extending the sign-bit of the
  // value through the padding bits. For example given a 3-bit XLS value:
  //
  //  0bXYZ
  //
  // Represented as an 8-bit LLVM value, AsSignedValue would return:
  //
  //  0bXXXX_XYZ
  //
  // If `dest_type` is given the result is sign-extended to this type before
  // returning.
  llvm::Value* AsSignedValue(
      llvm::Value* value, Type* xls_type, llvm::IRBuilder<>& builder,
      std::optional<llvm::Type*> dest_type = std::nullopt);

  // Creates a TypeLayout object describing the native layout of given xls type.
  TypeLayout CreateTypeLayout(Type* xls_type);

 private:
  using TypeCache = absl::flat_hash_map<const Type*, llvm::Type*>;

  // Handles the special (and base) case of converting Bits types to LLVM.
  absl::StatusOr<llvm::Constant*> ToIntegralConstant(llvm::Type* type,
                                                     const Value& value) const;

  // Helper method for computing the layouts of leaf elements for building a
  // TypeLayout object.
  void ComputeElementLayouts(Type* xls_type,
                             std::vector<ElementLayout>* layouts,
                             int64_t offset);

  llvm::LLVMContext& context_;
  llvm::DataLayout data_layout_;
  TypeCache type_cache_;
};

}  // namespace xls

#endif  // XLS_JIT_LLVM_TYPE_CONVERTER_H_
