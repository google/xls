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

#include "absl/strings/str_format.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/Support/TargetSelect.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

JitRuntime::JitRuntime(const llvm::DataLayout& data_layout,
                       LlvmTypeConverter* type_converter)
    : data_layout_(data_layout), type_converter_(type_converter) {}

absl::Status JitRuntime::PackArgs(absl::Span<const Value> args,
                                  absl::Span<Type* const> arg_types,
                                  absl::Span<uint8_t*> arg_buffers) {
  if (arg_buffers.size() < args.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input buffer is not large enough to hold all arguments: %d vs. %d",
        arg_buffers.size(), args.size()));
  }
  if (!args.empty()) {
    for (int i = 0; i < args.size(); ++i) {
      BlitValueToBuffer(
          args[i], arg_types[i],
          absl::MakeSpan(arg_buffers[i],
                         type_converter_->GetTypeByteSize(arg_types[i])));
    }
  }

  return absl::OkStatus();
}

Value JitRuntime::UnpackBuffer(const uint8_t* buffer, const Type* result_type,
                               bool unpoison) {
  switch (result_type->kind()) {
    case TypeKind::kBits: {
      const BitsType* bits_type = result_type->AsBitsOrDie();
      int64_t bit_count = bits_type->bit_count();
      int64_t byte_count = CeilOfRatio(bit_count, kCharBit);
      absl::InlinedVector<uint8_t, 8> data;
      data.reserve(byte_count);
#ifdef ABSL_HAVE_MEMORY_SANITIZER
      if (unpoison) {
        __msan_unpoison(buffer, byte_count);
      }
#endif  // ABSL_HAVE_MEMORY_SANITIZER
      for (int i = 0; i < byte_count; ++i) {
        data.push_back(buffer[i]);
      }

      // We could copy the data out of the buffer to avoid a swap, but it's
      // probably not worth the effort.
      if (data_layout_.isLittleEndian()) {
        ByteSwap(absl::MakeSpan(data));
      }

      return Value(Bits::FromBytes(absl::MakeSpan(data), bit_count));
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
        Value value = UnpackBuffer(buffer + layout->getElementOffset(i),
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
        Value value = UnpackBuffer(buffer + offset, element_type);
        values.push_back(value);
      }

      return Value::ArrayOrDie(values);
    }
    case TypeKind::kToken:
      return Value::Token();
    default:
      XLS_LOG(FATAL) << "Unsupported XLS Value kind: " << result_type->kind();
  }
}

void JitRuntime::BlitValueToBuffer(const Value& value, const Type* type,
                                   absl::Span<uint8_t> buffer) {
  if (value.IsBits()) {
    const Bits& bits = value.bits();
    int64_t byte_count = CeilOfRatio(bits.bit_count(), kCharBit);
    bits.ToBytes(absl::MakeSpan(buffer.data(), byte_count),
                 data_layout_.isBigEndian());

    // Clear out any bits in storage above that indicated by the data type -
    // LLVM requires this for safe operation, e.g., bit 127 in the 128-bit
    // actual allocated storage for a i127 must be 0.
    // Max of 7 bits of remainder on the [little-endian] most-significant byte.
    int remainder_bits = bits.bit_count() % kCharBit;
    if (remainder_bits != 0) {
      buffer[byte_count - 1] &= static_cast<uint8_t>(Mask(remainder_bits));
    }
  } else if (value.IsArray()) {
    const ArrayType* array_type = type->AsArrayOrDie();
    int64_t element_size =
        type_converter_->GetTypeByteSize(array_type->element_type());
    for (int i = 0; i < value.size(); ++i) {
      BlitValueToBuffer(value.element(i), array_type->element_type(), buffer);
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
      BlitValueToBuffer(value.element(i), tuple_type->element_type(i),
                        buffer.subspan(layout->getElementOffset(i)));
    }
  } else if (value.IsToken()) {
    // Tokens contain no data.
  } else {
    XLS_LOG(FATAL) << "Unsupported XLS Value kind: " << value.kind();
  }
}

}  // namespace xls

extern "C" {

// One-time initialization of LLVM targets.
absl::once_flag xls_jit_llvm_once;
void XlsJitLlvmOnceInit() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
}

// Simple container struct to hold the resources needed by the below functions.
struct RuntimeState {
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  std::unique_ptr<xls::LlvmTypeConverter> type_converter;
  std::unique_ptr<xls::JitRuntime> runtime;
};

// Note: Returned pointer is owned by the caller.
//
// Implementation note: normally we'd return a unique_ptr from this, but this is
// exposed via C ABI, so LLVM warns/errors if we do. Be sure to immediately wrap
// in unique_ptr in C++ callers.
RuntimeState* GetRuntimeState() {
  absl::call_once(xls_jit_llvm_once, XlsJitLlvmOnceInit);
  auto state = std::make_unique<RuntimeState>();
  state->context = std::make_unique<llvm::LLVMContext>();
  auto error_or_target_builder =
      llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!error_or_target_builder) {
    absl::PrintF("Unable to create TargetMachineBuilder!\n");
    return nullptr;
  }

  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    absl::PrintF("Unable to create TargetMachine!\n");
    return nullptr;
  }

  state->target_machine = std::move(error_or_target_machine.get());
  llvm::DataLayout data_layout = state->target_machine->createDataLayout();
  state->type_converter = std::make_unique<xls::LlvmTypeConverter>(
      state->context.get(), data_layout);
  state->runtime = std::make_unique<xls::JitRuntime>(
      data_layout, state->type_converter.get());
  return state.release();
}

int64_t GetArgBufferSize(int arg_count, const char** input_args) {
  std::unique_ptr<RuntimeState> state(GetRuntimeState());
  if (state == nullptr) {
    return -1;
  }

  xls::Package package("get_arg_buffer_size");
  int64_t args_size = 0;
  for (int i = 1; i < arg_count; i++) {
    auto status_or_value = xls::Parser::ParseTypedValue(input_args[i]);
    if (!status_or_value.ok()) {
      return -2;
    }

    xls::Value value = status_or_value.value();
    xls::Type* type = package.GetTypeForValue(value);
    args_size += state->type_converter->GetTypeByteSize(type);
  }

  return args_size;
}

// It's a little bit wasteful to re-do all the work above in this function,
// but it's a whole lot easier to write this way. If we tried to do this all in
// one function, we'd run into weirdness with LLVM IR return types or handling
// new/free in the context of LLVM, and how those would map to allocas.
// Since LLVM "main" execution isn't a throughput case, it's really not a
// problem to be a bit wasteful, especially when it makes things that much
// simpler.
int64_t PackArgs(int arg_count, const char** input_args, uint8_t** buffer) {
  std::unique_ptr<RuntimeState> state(GetRuntimeState());
  if (state == nullptr) {
    return -1;
  }

  xls::Package package("pack_args");
  std::vector<xls::Value> values;
  std::vector<xls::Type*> types;
  std::vector<int64_t> arg_sizes;
  values.reserve(arg_count);
  types.reserve(arg_count);
  arg_sizes.reserve(arg_count);
  // Skip argv[0].
  for (int i = 1; i < arg_count; i++) {
    auto status_or_value = xls::Parser::ParseTypedValue(input_args[i]);
    if (!status_or_value.ok()) {
      return -3 - i;
    }

    values.push_back(status_or_value.value());
    types.push_back(package.GetTypeForValue(values.back()));
  }

  XLS_CHECK_OK(state->runtime->PackArgs(values, types,
                                        absl::MakeSpan(buffer, arg_count)));
  return 0;
}

int UnpackAndPrintBuffer(const char* output_type_string, int arg_count,
                         const char** input_args, const uint8_t* buffer) {
  std::unique_ptr<RuntimeState> state(GetRuntimeState());
  if (state == nullptr) {
    return -1;
  }

  xls::Package package("oink oink oink");
  xls::Type* output_type =
      xls::Parser::ParseType(output_type_string, &package).value();
  xls::Value output = state->runtime->UnpackBuffer(buffer, output_type);
  absl::PrintF("%s\n", output.ToString());

  return 0;
}

}  // extern "C"
