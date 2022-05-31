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

#include "absl/status/statusor.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/ir_parser.h"

namespace xls {

LlvmTypeConverter::LlvmTypeConverter(llvm::LLVMContext* context,
                                     const llvm::DataLayout& data_layout)
    : context_(*context), data_layout_(data_layout) {}

int64_t LlvmTypeConverter::GetLlvmBitCount(int64_t xls_bit_count) const {
  // LLVM does not accept 0-bit types, and we want to be able to JIT-compile
  // unoptimized IR, so for the time being we make a dummy 1-bit value.
  // See https://github.com/google/xls/issues/76
  if (xls_bit_count <= 1) {
    return 1;
  }
  if (xls_bit_count <= 8) {
    return 8;
  }
  return int64_t{1} << CeilOfLog2(xls_bit_count);
}

llvm::Type* LlvmTypeConverter::ConvertToLlvmType(const Type* xls_type) const {
  auto it = type_cache_.find(xls_type);
  if (it != type_cache_.end()) {
    return it->second;
  }
  llvm::Type* llvm_type;
  if (xls_type->IsBits()) {
    llvm_type = llvm::IntegerType::get(
        context_, GetLlvmBitCount(xls_type->AsBitsOrDie()));
  } else if (xls_type->IsTuple()) {
    std::vector<llvm::Type*> tuple_types;

    const TupleType* tuple_type = xls_type->AsTupleOrDie();
    for (Type* tuple_elem_type : tuple_type->element_types()) {
      llvm::Type* llvm_type = ConvertToLlvmType(tuple_elem_type);
      tuple_types.push_back(llvm_type);
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
    XLS_LOG(FATAL) << absl::StrCat("Type not supported for LLVM conversion: %s",
                                   xls_type->ToString());
  }
  type_cache_.insert({xls_type, llvm_type});
  return llvm_type;
}

llvm::Type* LlvmTypeConverter::ConvertToPackedLlvmType(const Type* type) const {
  // LLVM doesn't support zero-width integer types so use i1 for XLS zero-bit
  // types.
  int64_t width = std::max(type->GetFlatBitCount(), int64_t{1});
  return llvm::IntegerType::get(context_, width);
}

absl::StatusOr<llvm::Constant*> LlvmTypeConverter::ToLlvmConstant(
    const Type* type, const Value& value) const {
  return ToLlvmConstant(ConvertToLlvmType(type), value);
}

absl::StatusOr<llvm::Constant*> LlvmTypeConverter::ToLlvmConstant(
    llvm::Type* type, const Value& value) const {
  if (type->isIntegerTy()) {
    return ToIntegralConstant(type, value);
  } else if (type->isStructTy()) {
    std::vector<llvm::Constant*> llvm_elements;
    for (int i = 0; i < type->getStructNumElements(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          llvm::Constant * llvm_element,
          ToLlvmConstant(type->getStructElementType(i), value.element(i)));
      llvm_elements.push_back(llvm_element);
    }

    return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(type),
                                     llvm_elements);
  } else if (type->isArrayTy()) {
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
  XLS_LOG(FATAL) << "Unknown value kind: " << value.kind();
}

absl::StatusOr<llvm::Constant*> LlvmTypeConverter::ToIntegralConstant(
    llvm::Type* type, const Value& value) const {
  Bits xls_bits = value.bits();

  if (xls_bits.bit_count() > 64) {
    std::vector<uint8_t> bytes = xls_bits.ToBytes();
    if (data_layout_.isLittleEndian()) {
      ByteSwap(absl::MakeSpan(bytes));
    }

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
  return data_layout_.getTypeAllocSize(ConvertToLlvmType(type)).getFixedSize();
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
  XLS_CHECK(xls_type->IsBits());
  int64_t xls_bit_count = xls_type->AsBitsOrDie()->bit_count();
  llvm::Value* signed_value;
  if (xls_bit_count <= 1) {
    signed_value = value;
  } else {
    llvm::Value* sign_bit = builder.CreateTrunc(
        builder.CreateLShr(
            value, llvm::ConstantInt::get(value->getType(), xls_bit_count - 1)),
        builder.getIntNTy(1));
    signed_value = builder.CreateSelect(
        sign_bit,
        builder.CreateOr(InvertedPaddingMask(xls_type, builder), value), value);
  }
  return dest_type.has_value()
             ? builder.CreateSExt(signed_value, dest_type.value())
             : signed_value;
}

llvm::Value* LlvmTypeConverter::PaddingMask(Type* xls_type,
                                            llvm::IRBuilder<>& builder) const {
  XLS_CHECK(xls_type->IsBits());
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

}  // namespace xls
