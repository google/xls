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

#include "xls/jit/jit_runtime.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/Support/Alignment.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/orc_jit.h"

namespace xls {

JitRuntime::JitRuntime(llvm::DataLayout data_layout)
    : data_layout_(data_layout),
      context_(std::make_unique<llvm::LLVMContext>()),
      type_converter_(
          std::make_unique<LlvmTypeConverter>(context_.get(), data_layout_)) {}

/* static */ absl::StatusOr<std::unique_ptr<JitRuntime>> JitRuntime::Create() {
  XLS_ASSIGN_OR_RETURN(auto orc_jit, OrcJit::Create());
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout data_layout,
                       orc_jit->CreateDataLayout());
  return std::make_unique<JitRuntime>(data_layout);
}

absl::Status JitRuntime::PackArgs(absl::Span<const Value> args,
                                  absl::Span<Type* const> arg_types,
                                  absl::Span<uint8_t* const> arg_buffers) {
  if (arg_buffers.size() < args.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input buffer is not large enough to hold all arguments: %d vs. %d",
        arg_buffers.size(), args.size()));
  }
  if (!args.empty()) {
    for (int i = 0; i < args.size(); ++i) {
      BlitValueToBuffer(
          args[i], arg_types[i],
          absl::MakeSpan(arg_buffers[i], GetTypeByteSize(arg_types[i])));
    }
  }

  return absl::OkStatus();
}

Value JitRuntime::UnpackBuffer(const uint8_t* buffer, const Type* result_type) {
  absl::MutexLock lock(&mutex_);
  return UnpackBufferInternal(buffer, result_type);
}

Value JitRuntime::UnpackBufferInternal(const uint8_t* buffer,
                                       const Type* result_type) {
  switch (result_type->kind()) {
    case TypeKind::kBits: {
      const BitsType* bits_type = result_type->AsBitsOrDie();
      int64_t bit_count = bits_type->bit_count();
      int64_t byte_count = CeilOfRatio(bit_count, kCharBit);
      return Value(
          Bits::FromBytes(absl::MakeSpan(buffer, byte_count), bit_count));
    }
    case TypeKind::kTuple: {
      // Just as with arg packing, we need the DataLayout to tell us where each
      // arg is placed in the output buffer.
      const TupleType* tuple_type = result_type->AsTupleOrDie();
      llvm::Type* llvm_type = type_converter_->ConvertToLlvmType(tuple_type);
      const llvm::StructLayout* layout =
          data_layout_.getStructLayout(llvm::cast<llvm::StructType>(llvm_type));

      std::vector<Value> values;
      values.reserve(tuple_type->size());
      for (int i = 0; i < tuple_type->size(); ++i) {
        Value value = UnpackBufferInternal(buffer + layout->getElementOffset(i),
                                           tuple_type->element_type(i));
        values.push_back(value);
      }
      return Value::TupleOwned(std::move(values));
    }
    case TypeKind::kArray: {
      const ArrayType* array_type = result_type->AsArrayOrDie();
      if (array_type->size() == 0) {
        return Value::ArrayOrDie({});
      }

      const Type* element_type = array_type->element_type();
      llvm::Type* llvm_element_type =
          type_converter_->ConvertToLlvmType(array_type->element_type());
      std::vector<Value> values;
      values.reserve(array_type->size());
      // This BitsType is only used inside the ToLlvmConstantCall() (and isn't
      // stored), so it's safe for it to live on the stack.
      BitsType bits_type(64);
      for (int i = 0; i < array_type->size(); ++i) {
        llvm::Constant* index =
            type_converter_->ToLlvmConstant(&bits_type, Value(UBits(i, 64)))
                .value();
        int64_t offset =
            data_layout_.getIndexedOffsetInType(llvm_element_type, index);
        Value value = UnpackBufferInternal(buffer + offset, element_type);
        values.push_back(value);
      }

      return Value::ArrayOrDie(values);
    }
    case TypeKind::kToken:
      return Value::Token();
    default:
      LOG(FATAL) << "Unsupported XLS Value kind: " << result_type->kind();
  }
}

void JitRuntime::BlitValueToBuffer(const Value& value, const Type* type,
                                   absl::Span<uint8_t> buffer) {
  absl::MutexLock lock(&mutex_);
  // Zero the buffer before filling in values. This ensures all padding bytes
  // are cleared.
  memset(buffer.data(), 0, type_converter_->GetTypeByteSize(type));
  BlitValueToBufferInternal(value, type, buffer);
}

absl::Span<uint8_t> JitRuntime::AsAligned(absl::Span<uint8_t> buffer,
                                          int64_t alignment) const {
  return buffer.subspan(llvm::offsetToAlignment(
      reinterpret_cast<uintptr_t>(buffer.data()), llvm::Align(alignment)));
}

void JitRuntime::BlitValueToBufferInternal(const Value& value, const Type* type,
                                           absl::Span<uint8_t> buffer) {
  if (value.IsBits()) {
    const Bits& bits = value.bits();
    int64_t byte_count = CeilOfRatio(bits.bit_count(), kCharBit);
    // Underlying Bits object relies on little-endianness.
    CHECK(data_layout_.isLittleEndian());
    bits.ToBytes(absl::MakeSpan(buffer.data(), byte_count));

    // Zero out any padding bits. Bits type are stored in the JIT as the next
    // power of two size.  LLVM requires all padding bits to be zero for safe
    // operation, e.g., for a 42 bit type (which is padded out to a 64-bit word
    // in the jit) bits 42 through 64 must be zero. The entire buffer is zeroed
    // prior to calling this function so we only need to handle the remainder
    // bits in the most-significant byte of value data here.
    int remainder_bits = bits.bit_count() % kCharBit;
    if (remainder_bits != 0) {
      buffer[byte_count - 1] &= static_cast<uint8_t>(Mask(remainder_bits));
    }
  } else if (value.IsArray()) {
    const ArrayType* array_type = type->AsArrayOrDie();
    int64_t element_size =
        type_converter_->GetTypeByteSize(array_type->element_type());
    for (int i = 0; i < value.size(); ++i) {
      BlitValueToBufferInternal(value.element(i), array_type->element_type(),
                                buffer);
      buffer = buffer.subspan(element_size);
    }
  } else if (value.IsTuple()) {
    // Due to per-target data packing (esp. as realized by the LLVM IR
    // load/store instructions), we need to make sure we blit args into LLVM
    // space as the underlying runtime expects, which means we need the
    // DataLayout to tell us where each constituent element should be placed.
    llvm::Type* llvm_type = type_converter_->ConvertToLlvmType(type);
    const llvm::StructLayout* layout =
        data_layout_.getStructLayout(llvm::cast<llvm::StructType>(llvm_type));

    const TupleType* tuple_type = type->AsTupleOrDie();
    for (int i = 0; i < value.size(); ++i) {
      BlitValueToBufferInternal(value.element(i), tuple_type->element_type(i),
                                buffer.subspan(layout->getElementOffset(i)));
    }
  } else if (value.IsToken()) {
    // Tokens contain no data.
  } else {
    LOG(FATAL) << "Unsupported XLS Value kind: " << value.kind();
  }
}

}  // namespace xls
