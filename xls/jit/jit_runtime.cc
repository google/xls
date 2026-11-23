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

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/Support/Alignment.h"
#include "llvm/include/llvm/Support/Error.h"
#include "xls/common/bits_util.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/type_buffer_metadata.h"

namespace xls {
namespace {

// Writes `bits` in little-endian byte order to `dst`
// (`CeilOfRatio(bit_count, kCharBit)` bytes) and zeroes any unused padding bits
// in the most-significant byte of value data.
//
// `Bits` types are stored in the JIT as the next power-of-two integer size, and
// LLVM requires all padding bits above `bit_count` to be zero for safe
// operation (e.g., for a 42-bit type padded out to a 64-bit word in the JIT,
// bits 42 through 63 must be zero). Callers clear any trailing container bytes
// beyond the returned `data_byte_count` (either via a tail `memset` in the
// scalar `BlitValueToBuffer` fast path or by zeroing the enclosing composite
// buffer prior to calling `BlitValueImpl`), so only the remainder bits in the
// most-significant byte of value data need to be masked here.
int64_t WriteLittleEndianBitsWithZeroedRemainder(const Bits& bits,
                                                 uint8_t* dst) {
  int64_t bit_count = bits.bit_count();
  int64_t data_byte_count = CeilOfRatio(bit_count, kCharBit);
  if (data_byte_count > 0) {
    bits.ToBytes(absl::MakeSpan(dst, data_byte_count));
    int remainder_bits = bit_count % kCharBit;
    if (remainder_bits != 0) {
      dst[data_byte_count - 1] &= static_cast<uint8_t>(Mask(remainder_bits));
    }
  }
  return data_byte_count;
}

// Upper bound matching LLVM's `Value::MaximumAlignment` (`1 << 29` bytes) so
// deserialized alignments fit `llvm::Align` and cannot wrap `size_t` /
// `int64_t` layout size calculations.
constexpr int64_t kMaxSupportedAlignment = int64_t{1} << 29;

absl::Status ValidateAlignment(int64_t align, std::string_view field_name) {
  if (align <= 0 || align > kMaxSupportedAlignment ||
      !std::has_single_bit(static_cast<uint64_t>(align))) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TargetLayoutInfo.%s must be a power of two in [1, %d]; got %d.",
        field_name, kMaxSupportedAlignment, align));
  }
  return absl::OkStatus();
}

absl::Status ValidateTargetLayoutInfo(const TargetLayoutInfo& info) {
  if (!info.is_little_endian) {
    return absl::InvalidArgumentError(
        "TargetLayoutInfo.is_little_endian must be true; big-endian targets "
        "are not supported by JitRuntime.");
  }
  XLS_RETURN_IF_ERROR(
      ValidateAlignment(info.stack_alignment, "stack_alignment"));
  XLS_RETURN_IF_ERROR(
      ValidateAlignment(info.struct_abi_alignment, "struct_abi_alignment"));
  XLS_RETURN_IF_ERROR(
      ValidateAlignment(info.struct_pref_alignment, "struct_pref_alignment"));
  XLS_RETURN_IF_ERROR(
      ValidateAlignment(info.token_abi_alignment, "token_abi_alignment"));
  XLS_RETURN_IF_ERROR(
      ValidateAlignment(info.token_pref_alignment, "token_pref_alignment"));
  for (int log2_bits = 3; log2_bits <= TargetLayoutInfo::kMaxLog2Bits;
       ++log2_bits) {
    XLS_RETURN_IF_ERROR(
        ValidateAlignment(info.int_abi_alignments[log2_bits],
                          absl::StrCat("int_abi_alignments[", log2_bits, "]")));
    XLS_RETURN_IF_ERROR(ValidateAlignment(
        info.int_pref_alignments[log2_bits],
        absl::StrCat("int_pref_alignments[", log2_bits, "]")));
  }
  return absl::OkStatus();
}

}  // namespace

/* static */ TargetLayoutInfo TargetLayoutInfo::FromDataLayout(
    const llvm::DataLayout& data_layout, llvm::LLVMContext& context) {
  // Probe a local copy of `data_layout` so that the caller's `DataLayout` never
  // populates its internal mutable `LayoutMap`.
  llvm::DataLayout probe_dl = data_layout;
  TargetLayoutInfo info;
  info.is_little_endian = probe_dl.isLittleEndian();
  info.stack_alignment = probe_dl.getStackAlignment().valueOrOne().value();

  llvm::StructType* empty_struct = llvm::StructType::get(context, {});
  info.struct_abi_alignment = probe_dl.getABITypeAlign(empty_struct).value();
  info.struct_pref_alignment = probe_dl.getPrefTypeAlign(empty_struct).value();

  llvm::Type* token_type =
      llvm::ArrayType::get(llvm::IntegerType::get(context, 1), 0);
  info.token_abi_alignment = probe_dl.getABITypeAlign(token_type).value();
  info.token_pref_alignment = probe_dl.getPrefTypeAlign(token_type).value();

  for (int log2_bits = 3; log2_bits <= kMaxLog2Bits; ++log2_bits) {
    llvm::IntegerType* int_ty =
        llvm::IntegerType::get(context, uint32_t{1} << log2_bits);
    info.int_abi_alignments[log2_bits] =
        probe_dl.getABITypeAlign(int_ty).value();
    info.int_pref_alignments[log2_bits] =
        probe_dl.getPrefTypeAlign(int_ty).value();
  }
  return info;
}

/* static */ TargetLayoutInfo TargetLayoutInfo::FromDataLayout(
    const llvm::DataLayout& data_layout) {
  llvm::LLVMContext probe_ctx;
  return FromDataLayout(data_layout, probe_ctx);
}

/* static */ absl::StatusOr<TargetLayoutInfo>
TargetLayoutInfo::FromDataLayoutString(std::string_view data_layout_str) {
  llvm::Expected<llvm::DataLayout> layout =
      llvm::DataLayout::parse(data_layout_str);
  if (!layout) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unable to parse '%s' to an llvm data-layout: %s",
                        data_layout_str, llvm::toString(layout.takeError())));
  }
  TargetLayoutInfo info = FromDataLayout(*layout);
  XLS_RETURN_IF_ERROR(ValidateTargetLayoutInfo(info));
  return info;
}

/* static */ absl::StatusOr<TargetLayoutInfo> TargetLayoutInfo::FromProto(
    const TargetLayoutProto& proto) {
  if (proto.int_abi_alignments_size() != kMaxLog2Bits + 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TargetLayoutProto.int_abi_alignments must have %d entries; got %d.",
        kMaxLog2Bits + 1, proto.int_abi_alignments_size()));
  }
  if (proto.int_pref_alignments_size() != kMaxLog2Bits + 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TargetLayoutProto.int_pref_alignments must have %d entries; got %d.",
        kMaxLog2Bits + 1, proto.int_pref_alignments_size()));
  }

  TargetLayoutInfo info;
  info.is_little_endian = proto.is_little_endian();
  info.stack_alignment = proto.stack_alignment();
  info.struct_abi_alignment = proto.struct_abi_alignment();
  info.struct_pref_alignment = proto.struct_pref_alignment();
  info.token_abi_alignment = proto.token_abi_alignment();
  info.token_pref_alignment = proto.token_pref_alignment();
  for (int log2_bits = 3; log2_bits <= kMaxLog2Bits; ++log2_bits) {
    info.int_abi_alignments[log2_bits] = proto.int_abi_alignments(log2_bits);
    info.int_pref_alignments[log2_bits] = proto.int_pref_alignments(log2_bits);
  }
  XLS_RETURN_IF_ERROR(ValidateTargetLayoutInfo(info));
  return info;
}

TargetLayoutProto TargetLayoutInfo::ToProto() const {
  TargetLayoutProto proto;
  proto.set_is_little_endian(is_little_endian);
  proto.set_stack_alignment(stack_alignment);
  proto.set_struct_abi_alignment(struct_abi_alignment);
  proto.set_struct_pref_alignment(struct_pref_alignment);
  proto.set_token_abi_alignment(token_abi_alignment);
  proto.set_token_pref_alignment(token_pref_alignment);
  proto.mutable_int_abi_alignments()->Reserve(kMaxLog2Bits + 1);
  proto.mutable_int_pref_alignments()->Reserve(kMaxLog2Bits + 1);
  for (int i = 0; i <= kMaxLog2Bits; ++i) {
    proto.add_int_abi_alignments(int_abi_alignments[i]);
    proto.add_int_pref_alignments(int_pref_alignments[i]);
  }
  return proto;
}

absl::StatusOr<TargetLayoutInfo> ResolveTargetLayoutInfo(
    const AotPackageEntrypointsProto& entrypoints) {
  if (entrypoints.has_target_layout()) {
    return TargetLayoutInfo::FromProto(entrypoints.target_layout());
  }
  if (entrypoints.has_data_layout()) {
    return TargetLayoutInfo::FromDataLayoutString(entrypoints.data_layout());
  }
  return absl::InvalidArgumentError(
      "AotPackageEntrypointsProto requires target_layout or data_layout.");
}

/* static */ void JitRuntime::MaybeInitCompositeChildMemo(
    const Type* child_type, TypeMetricsMemo* memo) {
  if (!child_type->IsBits() && !child_type->IsToken() && !memo->has_value()) {
    memo->emplace();
  }
}

JitRuntime::JitRuntime(TargetLayoutInfo layout_info)
    : layout_info_(layout_info) {
  CHECK_OK(ValidateTargetLayoutInfo(layout_info_));
}

JitRuntime::JitRuntime(const llvm::DataLayout& data_layout,
                       llvm::LLVMContext& context)
    : layout_info_(TargetLayoutInfo::FromDataLayout(data_layout, context)),
      data_layout_(data_layout) {
  CHECK_OK(ValidateTargetLayoutInfo(layout_info_));
}

JitRuntime::JitRuntime(const llvm::DataLayout& data_layout)
    : layout_info_(TargetLayoutInfo::FromDataLayout(data_layout)),
      data_layout_(data_layout) {
  CHECK_OK(ValidateTargetLayoutInfo(layout_info_));
}

const llvm::DataLayout& JitRuntime::data_layout() const {
  CHECK(data_layout_.has_value())
      << "JitRuntime::data_layout() is only available when constructed from "
         "an llvm::DataLayout.";
  return *data_layout_;
}

JitRuntime::TypeMetrics JitRuntime::GetBitsTypeMetrics(
    int64_t bit_count) const {
  int log2_bits = (bit_count <= 8) ? 3 : CeilOfLog2(bit_count);
  CHECK_LE(log2_bits, kMaxLog2Bits)
      << "Unsupported XLS bit count for LLVM IntegerType: " << bit_count;
  int64_t llvm_bytes = int64_t{1} << (log2_bits - 3);
  int64_t abi_align = layout_info_.int_abi_alignments[log2_bits];
  int64_t pref_align = layout_info_.int_pref_alignments[log2_bits];
  int64_t alloc_size = static_cast<int64_t>(
      llvm::alignTo(static_cast<uint64_t>(llvm_bytes), llvm::Align(abi_align)));
  return TypeMetrics{
      .size = alloc_size, .abi_align = abi_align, .pref_align = pref_align};
}

JitRuntime::TypeMetrics JitRuntime::ComputeTypeMetrics(
    const Type* xls_type) const {
  TypeMetricsMemo memo;
  return ComputeTypeMetricsImpl(xls_type, &memo);
}

JitRuntime::TypeMetrics JitRuntime::ComputeTypeMetricsImpl(
    const Type* xls_type, TypeMetricsMemo* memo) const {
  switch (xls_type->kind()) {
    case TypeKind::kBits:
      return GetBitsTypeMetrics(xls_type->AsBitsOrDie()->bit_count());
    case TypeKind::kToken:
      return TypeMetrics{.size = 0,
                         .abi_align = layout_info_.token_abi_alignment,
                         .pref_align = layout_info_.token_pref_alignment};
    case TypeKind::kArray: {
      if (memo->has_value()) {
        if (auto it = (*memo)->find(xls_type); it != (*memo)->end()) {
          return it->second;
        }
      }
      const ArrayType* array_type = xls_type->AsArrayOrDie();
      const Type* elem_type = array_type->element_type();
      MaybeInitCompositeChildMemo(elem_type, memo);
      TypeMetrics elem = ComputeTypeMetricsImpl(elem_type, memo);
      TypeMetrics result{.size = array_type->size() * elem.size,
                         .abi_align = elem.abi_align,
                         .pref_align = elem.pref_align};
      if (memo->has_value()) {
        (*memo)->emplace(xls_type, result);
      }
      return result;
    }
    case TypeKind::kTuple: {
      if (memo->has_value()) {
        if (auto it = (*memo)->find(xls_type); it != (*memo)->end()) {
          return it->second;
        }
      }
      const TupleType* tuple_type = xls_type->AsTupleOrDie();
      uint64_t struct_size = 0;
      int64_t max_elem_abi_align = 1;
      for (const Type* elem_type : tuple_type->element_types()) {
        MaybeInitCompositeChildMemo(elem_type, memo);
        TypeMetrics elem = ComputeTypeMetricsImpl(elem_type, memo);
        struct_size = llvm::alignTo(struct_size, llvm::Align(elem.abi_align));
        max_elem_abi_align = std::max(max_elem_abi_align, elem.abi_align);
        struct_size += static_cast<uint64_t>(elem.size);
      }
      int64_t abi_align =
          std::max(layout_info_.struct_abi_alignment, max_elem_abi_align);
      int64_t pref_align =
          std::max(layout_info_.struct_pref_alignment, max_elem_abi_align);
      struct_size = llvm::alignTo(struct_size, llvm::Align(abi_align));
      TypeMetrics result{.size = static_cast<int64_t>(struct_size),
                         .abi_align = abi_align,
                         .pref_align = pref_align};
      if (memo->has_value()) {
        (*memo)->emplace(xls_type, result);
      }
      return result;
    }
  }
  LOG(FATAL) << "Unsupported XLS Type kind: " << xls_type->kind();
}

int64_t JitRuntime::GetTypeByteSize(const Type* xls_type) const {
  return ComputeTypeMetrics(xls_type).size;
}

int64_t JitRuntime::GetTypeAlignment(const Type* xls_type) const {
  return ComputeTypeMetrics(xls_type).pref_align;
}

int64_t JitRuntime::GetTypeAbiAlignment(const Type* xls_type) const {
  return ComputeTypeMetrics(xls_type).abi_align;
}

TypeBufferMetadata JitRuntime::GetTypeBufferMetadata(
    const Type* xls_type) const {
  TypeMetrics metrics = ComputeTypeMetrics(xls_type);
  int64_t packed_llvm_width = std::max(
      int64_t{1}, RoundUpToNearest(xls_type->GetFlatBitCount(), int64_t{8}));
  return TypeBufferMetadata{
      .size = metrics.size,
      .preferred_alignment = metrics.pref_align,
      .abi_alignment = metrics.abi_align,
      .packed_size = RoundUpToNearest(packed_llvm_width, int64_t{8}),
  };
}

absl::Status JitRuntime::PackArgs(
    absl::Span<const Value> args, absl::Span<Type* const> arg_types,
    absl::Span<uint8_t* const> arg_buffers) const {
  if (arg_buffers.size() < args.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input buffer is not large enough to hold all arguments: %d vs. %d",
        arg_buffers.size(), args.size()));
  }
  for (size_t i = 0; i < args.size(); ++i) {
    const Type* type = arg_types[i];
    if (type->IsBits()) {
      int64_t bit_count = type->AsBitsOrDie()->bit_count();
      int64_t type_size = GetBitsTypeMetrics(bit_count).size;
      BlitValueToBuffer(args[i], type,
                        absl::MakeSpan(arg_buffers[i], type_size));
    } else {
      // Pass `memo` into `BlitValueImpl` so nested composite subtype metrics
      // cached during sizing are reused (`BlitValueImpl` computes any missing
      // entry on miss).
      TypeMetricsMemo memo;
      TypeMetrics metrics = ComputeTypeMetricsImpl(type, &memo);
      if (metrics.size > 0) {
        std::memset(arg_buffers[i], 0, metrics.size);
        BlitValueImpl(args[i], type, arg_buffers[i], &memo);
      }
    }
  }
  return absl::OkStatus();
}

Value JitRuntime::UnpackBuffer(const uint8_t* buffer,
                               const Type* result_type) const {
  TypeMetricsMemo memo;
  return UnpackBufferImpl(buffer, result_type, &memo);
}

Value JitRuntime::UnpackBufferImpl(const uint8_t* buffer,
                                   const Type* result_type,
                                   TypeMetricsMemo* memo) const {
  switch (result_type->kind()) {
    case TypeKind::kBits: {
      CHECK(layout_info_.is_little_endian);
      const BitsType* bits_type = result_type->AsBitsOrDie();
      int64_t bit_count = bits_type->bit_count();
      // `Bits::FromBytes` and `InlineBitmap::FromBytes` are inline in headers:
      // specializing common scalar widths passes a compile-time constant
      // `bit_count` and span size so the compiler folds `InlinedVector` to its
      // single inline word, eliminates `CHECK_GE`, lowers `memcpy` to a single
      // scalar load instead of a library call, and folds `MaskLastWord()` to a
      // constant mask (several times faster than variable `byte_count`).
      switch (bit_count) {
        case 0:
          return Value(Bits(0));
        case 1:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 1), 1));
        case 8:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 1), 8));
        case 16:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 2), 16));
        case 32:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 4), 32));
        case 64:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 8), 64));
        default: {
          int64_t byte_count = CeilOfRatio(bit_count, kCharBit);
          return Value(
              Bits::FromBytes(absl::MakeSpan(buffer, byte_count), bit_count));
        }
      }
    }
    case TypeKind::kTuple: {
      const TupleType* tuple_type = result_type->AsTupleOrDie();
      std::vector<Value> values;
      values.reserve(tuple_type->size());
      uint64_t offset = 0;
      for (int64_t i = 0; i < tuple_type->size(); ++i) {
        const Type* elem_type = tuple_type->element_type(i);
        MaybeInitCompositeChildMemo(elem_type, memo);
        TypeMetrics elem_metrics = ComputeTypeMetricsImpl(elem_type, memo);
        offset = llvm::alignTo(offset, llvm::Align(elem_metrics.abi_align));
        values.push_back(UnpackBufferImpl(buffer + offset, elem_type, memo));
        offset += static_cast<uint64_t>(elem_metrics.size);
      }
      return Value::TupleOwned(std::move(values));
    }
    case TypeKind::kArray: {
      const ArrayType* array_type = result_type->AsArrayOrDie();
      if (array_type->empty()) {
        return Value::ArrayOwned({});
      }
      const Type* element_type = array_type->element_type();
      MaybeInitCompositeChildMemo(element_type, memo);
      int64_t elem_size = ComputeTypeMetricsImpl(element_type, memo).size;
      std::vector<Value> values;
      values.reserve(array_type->size());
      for (int64_t i = 0; i < array_type->size(); ++i) {
        values.push_back(
            UnpackBufferImpl(buffer + i * elem_size, element_type, memo));
      }
      return Value::ArrayOwned(std::move(values));
    }
    case TypeKind::kToken:
      return Value::Token();
  }
  LOG(FATAL) << "Unsupported XLS Value kind: " << result_type->kind();
}

void JitRuntime::BlitValueToBuffer(const Value& value, const Type* type,
                                   absl::Span<uint8_t> buffer) const {
  if (type->IsBits()) {
    // Underlying `Bits` object relies on little-endianness.
    CHECK(layout_info_.is_little_endian);
    const Bits& bits = value.bits();
    int64_t type_byte_size = GetBitsTypeMetrics(bits.bit_count()).size;
    int64_t data_byte_count =
        WriteLittleEndianBitsWithZeroedRemainder(bits, buffer.data());
    // Clear any remaining upper bytes of the power-of-two LLVM integer
    // container (the scalar fast path skips the full-buffer `memset` below).
    if (type_byte_size > data_byte_count) {
      std::memset(buffer.data() + data_byte_count, 0,
                  type_byte_size - data_byte_count);
    }
    return;
  }

  // Pass `memo` into `BlitValueImpl` so nested composite subtype metrics
  // populated while computing `type_byte_size` are reused rather than
  // recomputed (`BlitValueImpl` does not require `memo` to be pre-populated for
  // correctness and computes any uncached entry on miss).
  TypeMetricsMemo memo;
  int64_t type_byte_size = ComputeTypeMetricsImpl(type, &memo).size;
  if (type_byte_size > 0) {
    // Zero the buffer before filling in values. This ensures all struct
    // alignment holes and padding bytes are cleared.
    std::memset(buffer.data(), 0, type_byte_size);
    BlitValueImpl(value, type, buffer.data(), &memo);
  }
}

absl::Span<uint8_t> JitRuntime::AsAligned(absl::Span<uint8_t> buffer,
                                          int64_t alignment) const {
  return buffer.subspan(llvm::offsetToAlignment(
      reinterpret_cast<uintptr_t>(buffer.data()), llvm::Align(alignment)));
}

void JitRuntime::BlitValueImpl(const Value& value, const Type* type,
                               uint8_t* buffer, TypeMetricsMemo* memo) const {
  if (value.IsBits()) {
    // Underlying `Bits` object relies on little-endianness. The enclosing
    // composite buffer is zero-initialized before `BlitValueImpl`, so only the
    // remainder bits in the most-significant byte of value data need masking.
    CHECK(layout_info_.is_little_endian);
    WriteLittleEndianBitsWithZeroedRemainder(value.bits(), buffer);
    return;
  }
  if (value.IsArray()) {
    if (value.empty()) {
      return;
    }
    const Type* elem_type = type->AsArrayOrDie()->element_type();
    MaybeInitCompositeChildMemo(elem_type, memo);
    int64_t elem_size = ComputeTypeMetricsImpl(elem_type, memo).size;
    for (int64_t i = 0; i < value.size(); ++i) {
      BlitValueImpl(value.element(i), elem_type, buffer + i * elem_size, memo);
    }
    return;
  }
  if (value.IsTuple()) {
    const TupleType* tuple_type = type->AsTupleOrDie();
    uint64_t offset = 0;
    for (int64_t i = 0; i < value.size(); ++i) {
      const Type* elem_type = tuple_type->element_type(i);
      MaybeInitCompositeChildMemo(elem_type, memo);
      TypeMetrics elem_metrics = ComputeTypeMetricsImpl(elem_type, memo);
      offset = llvm::alignTo(offset, llvm::Align(elem_metrics.abi_align));
      if (elem_metrics.size > 0) {
        BlitValueImpl(value.element(i), elem_type, buffer + offset, memo);
      }
      offset += static_cast<uint64_t>(elem_metrics.size);
    }
    return;
  }
  if (value.IsToken()) {
    return;
  }
  LOG(FATAL) << "Unsupported XLS Value kind: " << value.kind();
}

}  // namespace xls
