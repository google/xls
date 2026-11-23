// Copyright 2026 The XLS Authors
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

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "xls/common/bits_util.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/aot_entrypoint.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;

constexpr std::string_view kX86_64DataLayout =
    "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-"
    "n8:16:32:64-S128";
constexpr std::string_view kAArch64DataLayout =
    "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-"
    "i128:128-n32:64-S128-Fn32";

TEST(JitRuntimeTest, ScalarBlitZeroesPaddingBitsAndTrailingContainerBytes) {
  JitRuntime runtime((llvm::DataLayout(kX86_64DataLayout)));
  Package pkg("scalar_padding_test");

  for (int64_t bit_count :
       {0, 1, 4, 7, 8, 13, 16, 31, 32, 42, 64, 65, 128, 129}) {
    Type* type = pkg.GetBitsType(bit_count);
    int64_t byte_size = runtime.GetTypeByteSize(type);
    std::vector<uint8_t> buffer(byte_size == 0 ? 1 : byte_size, 0xff);

    Value val(Bits::AllOnes(bit_count));
    runtime.BlitValueToBuffer(val, type,
                              absl::MakeSpan(buffer.data(), byte_size));
    EXPECT_EQ(runtime.UnpackBuffer(buffer.data(), type), val)
        << "bit_count=" << bit_count;

    int64_t data_bytes = CeilOfRatio(bit_count, kCharBit);
    int remainder_bits = bit_count % kCharBit;
    if (data_bytes > 0 && remainder_bits != 0) {
      EXPECT_EQ(buffer[data_bytes - 1] & ~Mask(remainder_bits), 0)
          << "Dirty remainder bits not cleared for bit_count=" << bit_count;
    }
    for (int64_t b = data_bytes; b < byte_size; ++b) {
      EXPECT_EQ(buffer[b], 0) << "Dirty trailing container byte " << b
                              << " not cleared for bit_count=" << bit_count;
    }
  }
}

TEST(JitRuntimeTest, NestedCompositeBlitPackAndUnpackReuseMemoCleanly) {
  JitRuntime runtime((llvm::DataLayout(kX86_64DataLayout)));
  Package pkg("nested_composite_test");

  Type* u1 = pkg.GetBitsType(1);
  Type* u42 = pkg.GetBitsType(42);
  Type* tok = pkg.GetTokenType();
  Type* shared_leaf_tuple = pkg.GetTupleType({u1, u42, tok});
  Type* arr = pkg.GetArrayType(3, shared_leaf_tuple);
  Type* outer = pkg.GetTupleType({shared_leaf_tuple, arr, shared_leaf_tuple});

  Value leaf0 = Value::Tuple(
      {Value(UBits(1, 1)), Value(UBits(0x2aaabbbbccULL, 42)), Value::Token()});
  Value leaf1 = Value::Tuple(
      {Value(UBits(0, 1)), Value(UBits(0x123456789aULL, 42)), Value::Token()});
  Value arr_val = Value::ArrayOrDie({leaf0, leaf1, leaf0});
  Value outer_val = Value::Tuple({leaf1, arr_val, leaf0});

  int64_t size = runtime.GetTypeByteSize(outer);
  std::vector<uint8_t> dirty_buf(size, 0xff);
  runtime.BlitValueToBuffer(outer_val, outer, absl::MakeSpan(dirty_buf));
  EXPECT_EQ(runtime.UnpackBuffer(dirty_buf.data(), outer), outer_val);

  // Verify `PackArgs` clears alignment holes and round-trips identical bytes.
  std::vector<uint8_t> pack_buf(size, 0xaa);
  std::array<Type*, 1> arg_types = {outer};
  std::array<uint8_t*, 1> arg_ptrs = {pack_buf.data()};
  XLS_ASSERT_OK(runtime.PackArgs({outer_val}, arg_types, arg_ptrs));
  EXPECT_EQ(pack_buf, dirty_buf);
  EXPECT_EQ(runtime.UnpackBuffer(pack_buf.data(), outer), outer_val);
}

TEST(JitRuntimeTest, X86_64AndAArch64DataLayoutsShareCompositeSizesAndOffsets) {
  JitRuntime x86_rt((llvm::DataLayout(kX86_64DataLayout)));
  JitRuntime arm_rt((llvm::DataLayout(kAArch64DataLayout)));
  Package pkg("cross_arch_layout_test");

  Type* u1 = pkg.GetBitsType(1);
  Type* u5 = pkg.GetBitsType(5);
  Type* u8 = pkg.GetBitsType(8);
  Type* u12 = pkg.GetBitsType(12);
  Type* u16 = pkg.GetBitsType(16);
  Type* u32 = pkg.GetBitsType(32);
  Type* u64 = pkg.GetBitsType(64);
  Type* u128 = pkg.GetBitsType(128);
  Type* byte_tuple = pkg.GetTupleType({u8, u1, u5});
  Type* word_array = pkg.GetArrayType(4, u32);
  Type* padded_inner = pkg.GetTupleType({u32, u5});
  Type* nested_tuple =
      pkg.GetTupleType({padded_inner, pkg.GetArrayType(2, u32)});

  struct ExpectedType {
    Type* type;
    int64_t expected_size;
    int64_t expected_abi_align;
    int64_t x86_pref_align;
    int64_t arm_pref_align;
  };
  const std::array<ExpectedType, 10> kTypes = {{
      {u1, 1, 1, 1, 4},
      {u8, 1, 1, 1, 4},
      {u12, 2, 2, 2, 4},
      {u16, 2, 2, 2, 4},
      {u32, 4, 4, 4, 4},
      {u64, 8, 8, 8, 8},
      {u128, 16, 16, 16, 16},
      {byte_tuple, 3, 1, 8, 8},
      {word_array, 16, 4, 4, 4},
      {nested_tuple, 16, 4, 8, 8},
  }};

  for (const ExpectedType& entry : kTypes) {
    EXPECT_EQ(x86_rt.GetTypeByteSize(entry.type), entry.expected_size);
    EXPECT_EQ(arm_rt.GetTypeByteSize(entry.type), entry.expected_size);
    EXPECT_EQ(x86_rt.GetTypeAbiAlignment(entry.type), entry.expected_abi_align);
    EXPECT_EQ(arm_rt.GetTypeAbiAlignment(entry.type), entry.expected_abi_align);
    EXPECT_EQ(x86_rt.GetTypeAlignment(entry.type), entry.x86_pref_align);
    EXPECT_EQ(arm_rt.GetTypeAlignment(entry.type), entry.arm_pref_align);
  }

  // Verify `((u32, u5), u32[2])` places `u32` at offset 0, `u5` (masked to 5
  // bits) at offset 4, 3 zeroed struct tail-padding bytes at offsets 5..7, and
  // `u32[2]` at offsets 8..15 identically on both x86-64 and AArch64.
  Value nested_val = Value::Tuple({
      Value::Tuple({Value(UBits(0xa1b2c3d4, 32)), Value(UBits(0x1b, 5))}),
      Value::ArrayOrDie(
          {Value(UBits(0x11223344, 32)), Value(UBits(0x55667788, 32))}),
  });
  std::vector<uint8_t> x86_bytes(16, 0xff);
  std::vector<uint8_t> arm_bytes(16, 0xff);
  x86_rt.BlitValueToBuffer(nested_val, nested_tuple, absl::MakeSpan(x86_bytes));
  arm_rt.BlitValueToBuffer(nested_val, nested_tuple, absl::MakeSpan(arm_bytes));
  EXPECT_EQ(x86_bytes, arm_bytes);
  EXPECT_EQ(arm_rt.UnpackBuffer(arm_bytes.data(), nested_tuple), nested_val);

  uint32_t word = 0;
  std::memcpy(&word, arm_bytes.data() + 0, sizeof(word));
  EXPECT_EQ(word, 0xa1b2c3d4U);
  EXPECT_EQ(arm_bytes[4], 0x1bU);
  EXPECT_EQ(arm_bytes[5], 0x00U);
  EXPECT_EQ(arm_bytes[6], 0x00U);
  EXPECT_EQ(arm_bytes[7], 0x00U);
  std::memcpy(&word, arm_bytes.data() + 8, sizeof(word));
  EXPECT_EQ(word, 0x11223344U);
  std::memcpy(&word, arm_bytes.data() + 12, sizeof(word));
  EXPECT_EQ(word, 0x55667788U);
}

TEST(JitRuntimeTest, TargetLayoutProtoRoundTripsX86_64AndAArch64Layouts) {
  for (std::string_view layout_str : {kX86_64DataLayout, kAArch64DataLayout}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        TargetLayoutInfo from_str,
        TargetLayoutInfo::FromDataLayoutString(layout_str));
    EXPECT_TRUE(from_str.is_little_endian);
    EXPECT_EQ(from_str.stack_alignment, 16);

    TargetLayoutProto proto = from_str.ToProto();
    XLS_ASSERT_OK_AND_ASSIGN(TargetLayoutInfo from_proto,
                             TargetLayoutInfo::FromProto(proto));
    EXPECT_EQ(from_proto, from_str);

    JitRuntime rt_from_dl((llvm::DataLayout(layout_str)));
    JitRuntime rt_from_proto(from_proto);
    EXPECT_EQ(rt_from_proto.target_layout_info(),
              rt_from_dl.target_layout_info());
    EXPECT_EQ(rt_from_proto.ShouldAllocateForStack(24),
              rt_from_dl.ShouldAllocateForStack(24));

    AotPackageEntrypointsProto entrypoints;
    *entrypoints.mutable_target_layout() = proto;
    XLS_ASSERT_OK_AND_ASSIGN(TargetLayoutInfo resolved_from_proto,
                             ResolveTargetLayoutInfo(entrypoints));
    EXPECT_EQ(resolved_from_proto, from_str);

    AotPackageEntrypointsProto legacy_entrypoints;
    legacy_entrypoints.set_data_layout(layout_str);
    XLS_ASSERT_OK_AND_ASSIGN(TargetLayoutInfo resolved_from_str,
                             ResolveTargetLayoutInfo(legacy_entrypoints));
    EXPECT_EQ(resolved_from_str, from_str);
  }
}

TEST(JitRuntimeTest, TargetLayoutInfoRejectsInvalidProtosAndStrings) {
  EXPECT_THAT(TargetLayoutInfo::FromDataLayoutString("not-a-valid-datalayout"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  AotPackageEntrypointsProto empty_entrypoints;
  EXPECT_THAT(ResolveTargetLayoutInfo(empty_entrypoints),
              StatusIs(absl::StatusCode::kInvalidArgument));

  XLS_ASSERT_OK_AND_ASSIGN(
      TargetLayoutInfo valid,
      TargetLayoutInfo::FromDataLayoutString(kX86_64DataLayout));
  TargetLayoutProto valid_proto = valid.ToProto();

  {
    TargetLayoutProto bad = valid_proto;
    bad.set_is_little_endian(false);
    EXPECT_THAT(TargetLayoutInfo::FromProto(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  EXPECT_THAT(TargetLayoutInfo::FromDataLayoutString("E-m:e-i64:64"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  {
    TargetLayoutProto bad = valid_proto;
    bad.set_stack_alignment(0);
    EXPECT_THAT(TargetLayoutInfo::FromProto(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    TargetLayoutProto bad = valid_proto;
    bad.set_stack_alignment(int64_t{1} << 30);
    EXPECT_THAT(TargetLayoutInfo::FromProto(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    TargetLayoutProto bad = valid_proto;
    bad.set_stack_alignment(int64_t{1} << 62);
    EXPECT_THAT(TargetLayoutInfo::FromProto(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    TargetLayoutProto bad = valid_proto;
    bad.set_struct_abi_alignment(3);
    EXPECT_THAT(TargetLayoutInfo::FromProto(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    TargetLayoutProto bad = valid_proto;
    bad.mutable_int_abi_alignments()->RemoveLast();
    EXPECT_THAT(TargetLayoutInfo::FromProto(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    TargetLayoutProto bad = valid_proto;
    bad.set_int_pref_alignments(3, 6);
    EXPECT_THAT(TargetLayoutInfo::FromProto(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(JitRuntimeTest, DataLayoutAccessorMatchesConstructionLayout) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  for (std::string_view layout_str : {kX86_64DataLayout, kAArch64DataLayout}) {
    llvm::DataLayout expected(layout_str);
    JitRuntime rt_from_dl(expected);
    EXPECT_EQ(rt_from_dl.data_layout(), expected);
    EXPECT_EQ(rt_from_dl.data_layout().getStringRepresentation(), layout_str);
    const JitRuntime& const_rt = rt_from_dl;
    EXPECT_EQ(&const_rt.data_layout(), &rt_from_dl.data_layout());

    llvm::LLVMContext ctx;
    JitRuntime rt_with_ctx(expected, ctx);
    EXPECT_EQ(rt_with_ctx.data_layout(), expected);
    EXPECT_EQ(rt_with_ctx.data_layout().getStringRepresentation(), layout_str);
  }
#pragma clang diagnostic pop
}

TEST(JitRuntimeTest, DataLayoutAccessorDiesOnAotRuntime) {
  XLS_ASSERT_OK_AND_ASSIGN(
      TargetLayoutInfo info,
      TargetLayoutInfo::FromDataLayoutString(kX86_64DataLayout));
  JitRuntime rt_aot(info);
  Package pkg("aot_layout_test");
  EXPECT_EQ(rt_aot.GetTypeByteSize(pkg.GetBitsType(32)), 4);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  EXPECT_DEATH((void)rt_aot.data_layout(),
               "only available when constructed from an llvm::DataLayout");
#pragma clang diagnostic pop
}

}  // namespace
}  // namespace xls
