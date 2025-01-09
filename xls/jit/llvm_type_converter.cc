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

#include "xls/jit/llvm_type_converter.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/IR/Constant.h"
#include "llvm/include/llvm/IR/Constants.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/Support/Alignment.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/type_layout.h"

namespace xls {

LlvmTypeConverter::LlvmTypeConverter(llvm::LLVMContext* context,
                                     const llvm::DataLayout& data_layout)
    : context_(*context), data_layout_(data_layout) {}

int64_t LlvmTypeConverter::GetLlvmBitCount(int64_t xls_bit_count) const {
  // LLVM does not accept 0-bit types and < 8 bit types often have issues, and
  // we want to be able to JIT-compile unoptimized IR, so for the time being we
  // make a dummy 8-bit value.  See https://github.com/google/xls/issues/76
  if (xls_bit_count <= 8) {
    return 8;
  }
  return int64_t{1} << CeilOfLog2(xls_bit_count);
}

llvm::Type* LlvmTypeConverter::ConvertToLlvmType(const Type* xls_type) const {
  llvm::Type* llvm_type;
  if (xls_type->IsBits()) {
    llvm_type = llvm::IntegerType::get(
        context_, GetLlvmBitCount(xls_type->AsBitsOrDie()));
  } else if (xls_type->IsTuple()) {
    std::vector<llvm::Type*> tuple_types;

    const TupleType* tuple_type = xls_type->AsTupleOrDie();
    for (Type* tuple_elem_type : tuple_type->element_types()) {
      llvm::Type* converted_llvm_type = ConvertToLlvmType(tuple_elem_type);
      tuple_types.push_back(converted_llvm_type);
    }

    llvm_type = llvm::StructType::get(context_, tuple_types);
  } else if (xls_type->IsArray()) {
    const ArrayType* array_type = xls_type->AsArrayOrDie();
    llvm::Type* element_type = ConvertToLlvmType(array_type->element_type());
    llvm_type = llvm::ArrayType::get(element_type, array_type->size());
  } else if (xls_type->IsToken()) {
    // Token types don't contain any data. A 0-element array is a convenient and
    // low-overhead way to let the rest of the llvm infrastructure treat token
    // like a normal data-type.
    llvm_type = GetTokenType();
  } else {
    LOG(FATAL) << absl::StrCat("Type not supported for LLVM conversion: %s",
                               xls_type->ToString());
  }
  return llvm_type;
}

int64_t LlvmTypeConverter::PackedLlvmTypeWidth(const Type* type) const {
  return std::max(int64_t{1},
                  RoundUpToNearest(type->GetFlatBitCount(), int64_t{8}));
}

llvm::Type* LlvmTypeConverter::ConvertToPackedLlvmType(const Type* type) const {
  return llvm::IntegerType::get(context_, PackedLlvmTypeWidth(type));
}

int64_t LlvmTypeConverter::GetPackedTypeByteSize(const Type* type) const {
  return RoundUpToNearest(PackedLlvmTypeWidth(type), int64_t{8});
}

absl::StatusOr<llvm::Constant*> LlvmTypeConverter::ToLlvmConstant(
    const Type* type, const Value& value) const {
  return ToLlvmConstant(ConvertToLlvmType(type), value);
}

absl::StatusOr<llvm::Constant*> LlvmTypeConverter::ToLlvmConstant(
    llvm::Type* type, const Value& value) const {
  if (type->isIntegerTy()) {
    return ToIntegralConstant(type, value);
  }
  if (type->isStructTy()) {
    std::vector<llvm::Constant*> llvm_elements;
    for (int i = 0; i < type->getStructNumElements(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          llvm::Constant * llvm_element,
          ToLlvmConstant(type->getStructElementType(i), value.element(i)));
      llvm_elements.push_back(llvm_element);
    }

    return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(type),
                                     llvm_elements);
  }
  if (type->isArrayTy()) {
    std::vector<llvm::Constant*> elements;
    llvm::Type* element_type = type->getArrayElementType();
    for (const Value& element : value.elements()) {
      XLS_ASSIGN_OR_RETURN(llvm::Constant * llvm_element,
                           ToLlvmConstant(element_type, element));
      elements.push_back(llvm_element);
    }

    return llvm::ConstantArray::get(
        llvm::ArrayType::get(element_type, type->getArrayNumElements()),
        elements);
  }
  LOG(FATAL) << "Unknown value kind: " << value.kind();
}

llvm::Constant* LlvmTypeConverter::ZeroOfType(llvm::Type* type) {
  if (type->isIntegerTy()) {
    return llvm::ConstantInt::get(type, 0);
  }
  if (type->isArrayTy()) {
    std::vector<llvm::Constant*> elements(
        type->getArrayNumElements(), ZeroOfType(type->getArrayElementType()));
    return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(type),
                                    elements);
  }

  // Must be a tuple/struct, then.
  std::vector<llvm::Constant*> elements(type->getStructNumElements());
  for (int i = 0; i < type->getStructNumElements(); ++i) {
    elements[i] = ZeroOfType(type->getStructElementType(i));
  }

  return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(type),
                                   elements);
}

absl::StatusOr<llvm::Constant*> LlvmTypeConverter::ToIntegralConstant(
    llvm::Type* type, const Value& value) const {
  const Bits& xls_bits = value.bits();

  if (xls_bits.bit_count() > 64) {
    std::vector<uint8_t> bytes = xls_bits.ToBytes();
    bytes.resize(xls::RoundUpToNearest(bytes.size(), 8UL), 0);

    auto array_ref =
        llvm::ArrayRef<uint64_t>(absl::bit_cast<const uint64_t*>(bytes.data()),
                                 CeilOfRatio(static_cast<int>(bytes.size()),
                                             static_cast<int>(CHAR_BIT)));
    int64_t llvm_bit_count = GetLlvmBitCount(xls_bits.bit_count());
    return llvm::ConstantInt::get(type, llvm::APInt(llvm_bit_count, array_ref));
  }
  XLS_ASSIGN_OR_RETURN(uint64_t bits, value.bits().ToUint64());
  return llvm::ConstantInt::get(type, bits);
}

int64_t LlvmTypeConverter::GetTypeByteSize(const Type* type) const {
  return data_layout_.getTypeAllocSize(ConvertToLlvmType(type)).getFixedValue();
}

int64_t LlvmTypeConverter::GetTypeAbiAlignment(const Type* type) const {
  return data_layout_.getABITypeAlign(ConvertToLlvmType(type)).value();
}
int64_t LlvmTypeConverter::GetTypePreferredAlignment(const Type* type) const {
  return data_layout_.getPrefTypeAlign(ConvertToLlvmType(type)).value();
}
int64_t LlvmTypeConverter::AlignFor(const Type* type, int64_t offset) const {
  llvm::Align alignment =
      data_layout_.getPrefTypeAlign(ConvertToLlvmType(type));
  return llvm::alignTo(offset, alignment);
}

llvm::Type* LlvmTypeConverter::GetTokenType() const {
  return llvm::ArrayType::get(llvm::IntegerType::get(context_, 1), 0);
}

llvm::Value* LlvmTypeConverter::GetToken() const {
  llvm::ArrayType* token_type =
      llvm::ArrayType::get(llvm::IntegerType::get(context_, 1), 0);
  return llvm::ConstantArray::get(token_type, {});
}

llvm::Value* LlvmTypeConverter::AsSignedValue(
    llvm::Value* value, Type* xls_type, llvm::IRBuilder<>& builder,
    std::optional<llvm::Type*> dest_type) const {
  CHECK(xls_type->IsBits());
  int64_t xls_bit_count = xls_type->AsBitsOrDie()->bit_count();
  int64_t llvm_bit_count = GetLlvmBitCount(xls_bit_count);
  llvm::Value* signed_value;
  if (llvm_bit_count == xls_bit_count || xls_bit_count == 0) {
    signed_value = value;
  } else if (xls_bit_count == 1) {
    // Just for this one case we don't need to do a shift.
    signed_value = builder.CreateICmpNE(
        value, llvm::ConstantInt::get(value->getType(), 0));
  } else {
    llvm::Value* sign_bit = builder.CreateTrunc(
        builder.CreateLShr(
            value, llvm::ConstantInt::get(value->getType(), xls_bit_count - 1)),
        builder.getIntNTy(1));
    signed_value = builder.CreateSelect(
        sign_bit,
        builder.CreateOr(InvertedPaddingMask(xls_type, builder), value), value);
  }
  return dest_type.has_value() && dest_type.value() != signed_value->getType()
             ? builder.CreateSExt(signed_value, dest_type.value())
             : signed_value;
}

llvm::Value* LlvmTypeConverter::PaddingMask(Type* xls_type,
                                            llvm::IRBuilder<>& builder) const {
  CHECK(xls_type->IsBits());
  int64_t xls_bit_count = xls_type->AsBitsOrDie()->bit_count();
  int64_t llvm_bit_count = GetLlvmBitCount(xls_type->AsBitsOrDie());
  if (xls_bit_count == 0) {
    // Special-case zero-bit types to avoid overshifting and producing poison
    // values.
    return llvm::ConstantInt::get(ConvertToLlvmType(xls_type), 0);
  }
  return builder.CreateLShr(
      llvm::ConstantInt::getSigned(ConvertToLlvmType(xls_type), -1),
      llvm_bit_count - xls_bit_count);
}

llvm::Value* LlvmTypeConverter::InvertedPaddingMask(
    Type* xls_type, llvm::IRBuilder<>& builder) const {
  return builder.CreateNot(PaddingMask(xls_type, builder));
}

llvm::Value* LlvmTypeConverter::ClearPaddingBits(
    llvm::Value* value, Type* xls_type, llvm::IRBuilder<>& builder) const {
  if (!xls_type->IsBits()) {
    // TODO(meheff): Handle non-bits types.
    return value;
  }
  return builder.CreateAnd(value, PaddingMask(xls_type, builder));
}

void LlvmTypeConverter::ComputeElementLayouts(
    Type* xls_type, std::vector<ElementLayout>* layouts, int64_t offset) {
  if (xls_type->IsToken()) {
    layouts->push_back(ElementLayout{.offset = offset,
                                     .data_size = 0,
                                     .padded_size = GetTypeByteSize(xls_type)});
    return;
  }
  if (xls_type->IsBits()) {
    layouts->push_back(ElementLayout{
        .offset = offset,
        .data_size =
            CeilOfRatio(xls_type->AsBitsOrDie()->bit_count(), int64_t{8}),
        .padded_size = GetTypeByteSize(xls_type)});
    return;
  }
  if (xls_type->IsArray()) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    Type* element_type = array_type->element_type();
    llvm::Type* llvm_element_type =
        ConvertToLlvmType(array_type->element_type());
    for (int64_t i = 0; i < array_type->size(); ++i) {
      llvm::Constant* index =
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context_), i);
      int64_t element_offset =
          data_layout_.getIndexedOffsetInType(llvm_element_type, index);
      ComputeElementLayouts(element_type, layouts, offset + element_offset);
    }
    return;
  }
  CHECK(xls_type->IsTuple());
  TupleType* tuple_type = xls_type->AsTupleOrDie();
  llvm::Type* llvm_type = ConvertToLlvmType(tuple_type);
  const llvm::StructLayout* layout =
      data_layout_.getStructLayout(llvm::cast<llvm::StructType>(llvm_type));
  for (int64_t i = 0; i < tuple_type->size(); ++i) {
    Type* element_type = tuple_type->element_type(i);
    ComputeElementLayouts(element_type, layouts,
                          offset + layout->getElementOffset(i));
  }
}

TypeLayout LlvmTypeConverter::CreateTypeLayout(Type* xls_type) {
  std::vector<ElementLayout> element_layouts;
  ComputeElementLayouts(xls_type, &element_layouts, /*offset=*/0);
  return TypeLayout(xls_type, GetTypeByteSize(xls_type), element_layouts);
}

}  // namespace xls
