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
#include "xls/ir/value_view.h"

#include <strings.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "xls/common/bits_util.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {
namespace {

template <int kNumBits>
bool EvaluateMask() {
  return EvaluateMask<kNumBits - 1>() &&
         (MakeMask<kNumBits>() == Mask(kNumBits));
}

template <>
bool EvaluateMask<0>() {
  return MakeMask<0>() == Mask(0);
}

// Exhaustively tests MakeMask to verify it's equivalent to the run-time Mask()
// function.
TEST(MakeMaskTest, Exhaustive) {
  // Using a template to test a template!
  EXPECT_TRUE(EvaluateMask<63>());
}

// "Smoke"-style test: can we extract simple bytes?
TEST(PackedBitViewTest, ExtractsSimpleBytes) {
  constexpr int kElementWidth = 8;
  constexpr int kBitOffset = 0;

  uint8_t buffer = 0x13;

  auto aligned_view = PackedBitsView<kElementWidth>(&buffer, kBitOffset);
  uint8_t return_buffer;
  aligned_view.Get(&return_buffer);
  EXPECT_EQ(return_buffer, 0x13);
}

TEST(PackedBitViewTest, ExtractsSimpleBigs) {
  constexpr int kElementWidth = 64;
  constexpr int kBitOffset = 0;

  uint64_t buffer = 0xbeefbeefbeef;

  auto aligned_view = PackedBitsView<kElementWidth>(
      reinterpret_cast<uint8_t*>(&buffer), kBitOffset);
  uint64_t return_buffer;
  aligned_view.Get(reinterpret_cast<uint8_t*>(&return_buffer));
  EXPECT_EQ(return_buffer, 0xbeefbeefbeef);
}

TEST(PackedBitViewTest, ExtractsUnalignedBytes) {
  constexpr int kElementWidth = 8;
  constexpr int kBitOffset = 4;

  uint8_t buffer[2];
  buffer[0] = 0xA5;
  buffer[1] = 0x5A;

  auto view = PackedBitsView<kElementWidth>(buffer, kBitOffset);
  uint8_t return_buffer;
  view.Get(reinterpret_cast<uint8_t*>(&return_buffer));
  EXPECT_EQ(return_buffer, 0xAA);
}

TEST(PackedBitViewTest, ExtractsUnalignedBigs) {
  constexpr int kElementWidth = 64;
  constexpr int kBitOffset = 4;

  uint64_t buffer[2];
  buffer[0] = 0xBEEFBEEFF00DF00D;
  buffer[1] = 0xF00DF00DBEEFBEEF;

  auto view = PackedBitsView<kElementWidth>(reinterpret_cast<uint8_t*>(buffer),
                                            kBitOffset);
  uint64_t return_buffer;
  view.Get(reinterpret_cast<uint8_t*>(&return_buffer));
  EXPECT_EQ(return_buffer, 0xFBEEFBEEFF00DF00);
}

TEST(PackedBitViewTest, ExtractsUnalignedReallyBigs) {
  constexpr int64_t kElementWidth = 237;
  constexpr int64_t kBitOffset = 5;

  BitsRope rope(kElementWidth + kBitOffset);
  for (int i = 0; i < kBitOffset; i++) {
    rope.push_back(0);
  }
  int current_bit = 0;
  int current_byte = 0;
  for (int i = 0; i < kElementWidth; i++) {
    rope.push_back((current_byte & (1 << current_bit)) != 0);
    if (current_bit == kCharBit) {
      current_byte++;
      current_bit = 0;
    }
  }
  Bits bits = rope.Build();
  std::vector<uint8_t> buffer = bits.ToBytes();

  std::vector<uint8_t> expected =
      bits_ops::ShiftRightLogical(bits, kBitOffset).ToBytes();

  auto view = PackedBitsView<kElementWidth>(buffer.data(), kBitOffset);
  auto return_buffer = std::make_unique<uint8_t[]>(
      CeilOfRatio(kElementWidth + kBitOffset, kCharBit));
  view.Get(reinterpret_cast<uint8_t*>(return_buffer.get()));
  for (int i = 0; i < buffer.size(); i++) {
    ASSERT_EQ(return_buffer[i], expected[i]);
  }
}

template <int64_t kBitWidth>
absl::Status TestWidthAndOffset(int bit_offset) {
  BitsRope rope(kBitWidth + bit_offset);
  for (int i = 0; i < bit_offset; i++) {
    rope.push_back(false);
  }
  int current_bit = 0;
  int current_byte = 0;
  for (int i = 0; i < kBitWidth; i++) {
    rope.push_back(current_byte & (1 << current_bit));
    if (current_bit == kCharBit) {
      current_byte++;
      current_bit = 0;
    }
  }
  Bits bits = rope.Build();
  std::vector<uint8_t> buffer = bits.ToBytes();
  std::reverse(buffer.begin(), buffer.end());

  std::vector<uint8_t> expected =
      bits_ops::ShiftRightLogical(bits, bit_offset).ToBytes();
  std::reverse(expected.begin(), expected.end());

  auto view = PackedBitsView<kBitWidth>(buffer.data(), bit_offset);
  int64_t buffer_bytes = CeilOfRatio(kBitWidth + bit_offset, kCharBit);
  auto return_buffer = std::make_unique<uint8_t[]>(buffer_bytes);
  bzero(return_buffer.get(), buffer_bytes);
  view.Get(reinterpret_cast<uint8_t*>(return_buffer.get()));
  for (int i = 0; i < buffer.size(); i++) {
    XLS_RET_CHECK(return_buffer[i] == expected[i]);
  }
  return absl::OkStatus();
}

template <int kBitWidth>
absl::Status TestAllBitOffsets() {
  for (int i = 0; i < kCharBit; i++) {
    XLS_RETURN_IF_ERROR((TestWidthAndOffset<kBitWidth>(i)));
  }
  return absl::OkStatus();
}

// Just iterates through all bit widths <= the given value.
template <int kBitWidth>
absl::Status TestLowBitWidths() {
  XLS_RETURN_IF_ERROR((TestLowBitWidths<kBitWidth - 1>()));
  return TestAllBitOffsets<kBitWidth>();
}

// ...templates beget templates.
template <>
absl::Status TestLowBitWidths<0>() {
  return absl::OkStatus();
}

// Tests PackedBitView for all bit offsets (0-7) for a range of bit widths.
TEST(PackedBitViewTest, OffsetTests) {
  XLS_ASSERT_OK(TestLowBitWidths<63>());

  // Then test some awful things.
  XLS_ASSERT_OK((TestAllBitOffsets<132>()));
  XLS_ASSERT_OK((TestAllBitOffsets<300>()));
  XLS_ASSERT_OK((TestAllBitOffsets<512>()));
  XLS_ASSERT_OK((TestAllBitOffsets<750>()));
  XLS_ASSERT_OK((TestAllBitOffsets<1023>()));
  XLS_ASSERT_OK((TestAllBitOffsets<2049>()));
}

TEST(PackedArrayViewTest, ExtractsUnaligned) {
  constexpr int kBitOffset = 7;
  constexpr int kElementBits = 13;
  constexpr int kNumElements = 23;

  BitsRope rope(kElementBits * kNumElements + kBitOffset);
  for (int i = 0; i < kBitOffset; i++) {
    rope.push_back(0);
  }

  // Fill the n'th element with n (for each element).
  for (int i = 0; i < kNumElements; i++) {
    for (int j = 0; j < kElementBits; j++) {
      rope.push_back((i & (1 << j)) != 0);
    }
  }
  std::vector<uint8_t> buffer = rope.Build().ToBytes();

  PackedArrayView<PackedBitsView<kElementBits>, kNumElements> array_view(
      buffer.data(), kBitOffset);
  for (int i = 0; i < kNumElements; i++) {
    PackedBitsView<kElementBits> bits_view = array_view.Get(i);
    uint32_t result = 0;
    bits_view.Get(reinterpret_cast<uint8_t*>(&result));
    EXPECT_EQ(result, i);
  }
}

TEST(PackedTupleViewTest, ExtractsSimpleUnaligned) {
  constexpr int64_t kValueBits = 32;
  constexpr int kBitOffset = 5;

  uint32_t int_value = 0xa5a5a5a5;
  Value value = F32ToTuple(absl::bit_cast<float>(int_value));

  using SignT = PackedBitsView<1>;
  using ExpT = PackedBitsView<8>;
  using FractionT = PackedBitsView<23>;
  using TupleT = PackedTupleView<SignT, ExpT, FractionT>;
  BitsRope rope(TupleT::kBitCount + kBitOffset);

  for (int i = 0; i < kBitOffset; i++) {
    rope.push_back(0);
  }
  // Iterate low-to-high, keeping LSb-first ordering.
  absl::Span<const Value> elements = value.elements();
  for (int i = elements.size() - 1; i >= 0; i--) {
    const Value& element = elements[i];
    for (bool bit : element.bits().ToBitVector()) {
      rope.push_back(bit);
    }
  }

  std::vector<uint8_t> buffer(CeilOfRatio(kValueBits + kBitOffset, kCharBit),
                              0);
  rope.Build().ToBytes(absl::MakeSpan(buffer));

  TupleT tuple_view(buffer.data(), kBitOffset);
  uint8_t sign_data = 0;
  tuple_view.Get<0>().Get(&sign_data);
  EXPECT_EQ(sign_data, value.element(0).bits().ToUint64().value());

  uint8_t exp_data = 0;
  tuple_view.Get<1>().Get(&exp_data);
  EXPECT_EQ(exp_data, value.element(1).bits().ToUint64().value());

  uint32_t fraction_data = 0;
  tuple_view.Get<2>().Get(reinterpret_cast<uint8_t*>(&fraction_data));
  EXPECT_EQ(fraction_data, value.element(2).bits().ToUint64().value());
}

}  // namespace
}  // namespace xls
