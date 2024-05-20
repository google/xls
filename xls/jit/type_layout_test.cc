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

#include "xls/jit/type_layout.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/bits_util.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

using testing::ElementsAre;

class TypeLayoutTest : public IrTestBase {};

TypeLayout CreateTypeLayout(Type* type) {
  std::unique_ptr<OrcJit> orc_jit = OrcJit::Create().value();
  LlvmTypeConverter type_converter(
      orc_jit->GetContext(),
      orc_jit->CreateDataLayout().value());
  return type_converter.CreateTypeLayout(type);
}

std::string BytesToString(absl::Span<const uint8_t> bytes) {
  int64_t index = 0;
  std::string result;
  while (true) {
    for (int64_t i = 0; i < 2; ++i) {
      for (int64_t j = 0; j < 8; ++j) {
        if (index == bytes.size()) {
          return result;
        }
        absl::StrAppendFormat(&result, " 0x%02x", bytes[index]);
        index += 1;
      }
      absl::StrAppend(&result, "  ");
    }
    absl::StrAppend(&result, "\n");
  }
  return "";
}

std::vector<Type*> GetLeafTypes(Type* type) {
  LeafTypeTree<absl::monostate> t(type);
  std::vector<Type*> leaf_types(t.leaf_types().begin(), t.leaf_types().end());
  return leaf_types;
}

TEST_F(TypeLayoutTest, EmptyTuple) {
  auto package = CreatePackage();
  Type* empty_tuple = package->GetTupleType({});
  TypeLayout layout(empty_tuple, 0, {});
  EXPECT_EQ(layout.size(), 0);
  EXPECT_TRUE(layout.elements().empty());
  std::vector<uint8_t> buffer;
  EXPECT_EQ(layout.NativeLayoutToValue(buffer.data()), Value::Tuple({}));
  layout.ValueToNativeLayout(Value::Tuple({}), buffer.data());

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeLayout copy, TypeLayout::FromProto(layout.ToProto(), package.get()));
  EXPECT_TRUE(copy.type() == layout.type());
  EXPECT_EQ(copy.size(), layout.size());
  EXPECT_TRUE(copy.elements().empty());
}

TEST_F(TypeLayoutTest, Bits1) {
  auto package = CreatePackage();
  Type* u1 = package->GetBitsType(1);
  {
    // First test a sane layout description.
    TypeLayout layout(
        u1, 1,
        {ElementLayout({.offset = 0, .data_size = 1, .padded_size = 1})});
    EXPECT_EQ(layout.size(), 1);
    EXPECT_EQ(layout.elements().size(), 1);
    // Verify conversion to native layout. In each case, fill buffer with all
    // ones to check that padding is cleared.
    {
      uint8_t buffer[] = {0xff};
      layout.ValueToNativeLayout(Value(UBits(1, 1)), buffer);
      EXPECT_EQ(buffer[0], 0x1);
    }
    {
      uint8_t buffer[] = {0xff};
      layout.ValueToNativeLayout(Value(UBits(0, 1)), buffer);
      EXPECT_EQ(buffer[0], 0x0);
    }
    // Verify conversion to value.
    {
      uint8_t buffer[] = {0x01};
      EXPECT_EQ(layout.NativeLayoutToValue(buffer), Value(UBits(1, 1)));
    }
    {
      uint8_t buffer[] = {0x00};
      EXPECT_EQ(layout.NativeLayoutToValue(buffer), Value(UBits(0, 1)));
    }
  }

  {
    // Next test a less-sane layout description.
    TypeLayout layout(
        u1, 7,
        {ElementLayout({.offset = 0, .data_size = 1, .padded_size = 7})});
    EXPECT_EQ(layout.size(), 7);
    EXPECT_EQ(layout.elements().size(), 1);
    // Verify conversion to native layout. In each case, fill buffer with all
    // ones to check that padding is cleared.
    {
      std::vector<uint8_t> buffer(7, 0xff);
      layout.ValueToNativeLayout(Value(UBits(1, 1)), buffer.data());
      EXPECT_THAT(buffer, ElementsAre(0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0));
    }
    {
      std::vector<uint8_t> buffer(7, 0xff);
      layout.ValueToNativeLayout(Value(UBits(0, 1)), buffer.data());
      EXPECT_THAT(buffer, ElementsAre(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0));
    }
    // Verify conversion to value.
    {
      // Spurious 1's should be ignored.
      uint8_t buffer[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
      EXPECT_EQ(layout.NativeLayoutToValue(buffer), Value(UBits(1, 1)));
    }
    {
      uint8_t buffer[] = {0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
      EXPECT_EQ(layout.NativeLayoutToValue(buffer), Value(UBits(0, 1)));
    }
  }

  {
    // Next test an even less-sane layout description.
    TypeLayout layout(
        u1, 7,
        {ElementLayout({.offset = 2, .data_size = 1, .padded_size = 5})});
    EXPECT_EQ(layout.size(), 7);
    EXPECT_EQ(layout.elements().size(), 1);
    // Verify conversion to native layout. In each case, fill buffer with all
    // ones to check that padding is cleared.
    {
      std::vector<uint8_t> buffer(7, 0xff);
      layout.ValueToNativeLayout(Value(UBits(1, 1)), buffer.data());
      EXPECT_THAT(buffer, ElementsAre(0xff, 0xff, 0x1, 0x0, 0x0, 0x0, 0x0));
    }
    {
      std::vector<uint8_t> buffer(7, 0xff);
      layout.ValueToNativeLayout(Value(UBits(0, 1)), buffer.data());
      EXPECT_THAT(buffer, ElementsAre(0xff, 0xff, 0x0, 0x0, 0x0, 0x0, 0x0));
    }
    // Verify conversion to value.
    {
      // Spurious 1's should be ignored.
      uint8_t buffer[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
      EXPECT_EQ(layout.NativeLayoutToValue(buffer), Value(UBits(1, 1)));
    }
    {
      uint8_t buffer[] = {0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff};
      EXPECT_EQ(layout.NativeLayoutToValue(buffer), Value(UBits(0, 1)));
    }
  }
}

TEST_F(TypeLayoutTest, Bits24) {
  auto package = CreatePackage();
  Type* u24 = package->GetBitsType(24);
  TypeLayout layout(
      u24, 4, {ElementLayout({.offset = 0, .data_size = 3, .padded_size = 4})});
  EXPECT_EQ(layout.size(), 4);
  EXPECT_EQ(layout.elements().size(), 1);
  {
    std::vector<uint8_t> buffer = {0xff, 0xff, 0xff, 0xff};
    layout.ValueToNativeLayout(Value(UBits(0x123456, 24)), buffer.data());
    EXPECT_THAT(buffer, ElementsAre(0x56, 0x34, 0x12, 0x00));
  }
  {
    uint8_t buffer[] = {0xef, 0xcd, 0xab, 0xff};
    EXPECT_EQ(layout.NativeLayoutToValue(buffer), Value(UBits(0xabcdef, 24)));
  }
}

TEST_F(TypeLayoutTest, Bits42) {
  auto package = CreatePackage();
  Type* u42 = package->GetBitsType(42);
  TypeLayout layout(
      u42, 8, {ElementLayout({.offset = 1, .data_size = 6, .padded_size = 6})});
  EXPECT_EQ(layout.size(), 8);
  EXPECT_EQ(layout.elements().size(), 1);
  {
    std::vector<uint8_t> buffer = {0xff, 0xff, 0xff, 0xff,
                                   0xff, 0xff, 0xff, 0xff};
    layout.ValueToNativeLayout(Value(UBits(0x123456789a, 42)), buffer.data());
    EXPECT_THAT(buffer,
                ElementsAre(0xff, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x00, 0xff));
  }
  {
    uint8_t buffer[] = {0xff, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x00, 0xff};
    EXPECT_EQ(layout.NativeLayoutToValue(buffer),
              Value(UBits(0x123456789a, 42)));
  }
}

TEST_F(TypeLayoutTest, Bits100) {
  auto package = CreatePackage();
  Type* u100 = package->GetBitsType(100);
  TypeLayout layout(
      u100, 16,
      {ElementLayout({.offset = 0, .data_size = 13, .padded_size = 16})});
  EXPECT_EQ(layout.size(), 16);
  EXPECT_EQ(layout.elements().size(), 1);
  for (const Value& value :
       {Value(UBits(0, 100)), Value(Bits::AllOnes(100)),
        Value(ParseNumber("0xf_3322_1234_5678_abcd_eeff_a543").value())}) {
    std::vector<uint8_t> buffer(16, 0xff);
    layout.ValueToNativeLayout(value, buffer.data());
    EXPECT_EQ(layout.NativeLayoutToValue(buffer.data()), value);
  }
}

TEST_F(TypeLayoutTest, SimpleTuple) {
  auto package = CreatePackage();
  Type* tuple =
      package->GetTupleType({package->GetBitsType(4), package->GetBitsType(15),
                             package->GetBitsType(16)});
  TypeLayout layout(
      tuple, 6,
      {ElementLayout{.offset = 0, .data_size = 1, .padded_size = 1},
       ElementLayout{.offset = 2, .data_size = 2, .padded_size = 2},
       ElementLayout{.offset = 4, .data_size = 2, .padded_size = 2}});
  EXPECT_EQ(layout.size(), 6);
  EXPECT_EQ(layout.elements().size(), 3);
  {
    std::vector<uint8_t> buffer = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    layout.ValueToNativeLayout(
        Value(Parser::ParseValue("(0x3, 0x7bcd, 0xfedc)", tuple).value()),
        buffer.data());
    EXPECT_THAT(buffer, ElementsAre(0x3, 0xff, 0xcd, 0x7b, 0xdc, 0xfe));
  }
}

TEST_F(TypeLayoutTest, SimpleArray) {
  auto package = CreatePackage();
  Type* array = package->GetArrayType(3, package->GetBitsType(9));
  TypeLayout layout(
      array, 6,
      {ElementLayout{.offset = 0, .data_size = 2, .padded_size = 2},
       ElementLayout{.offset = 2, .data_size = 2, .padded_size = 2},
       ElementLayout{.offset = 4, .data_size = 2, .padded_size = 2}});
  EXPECT_EQ(layout.size(), 6);
  EXPECT_EQ(layout.elements().size(), 3);
  {
    std::vector<uint8_t> buffer = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    layout.ValueToNativeLayout(
        Value(Parser::ParseValue("[0x3, 0xaf, 0x1da]", array).value()),
        buffer.data());
    EXPECT_THAT(buffer, ElementsAre(0x03, 0x00, 0x0af, 0x00, 0xda, 0x01));
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      TypeLayout copy, TypeLayout::FromProto(layout.ToProto(), package.get()));
  EXPECT_TRUE(copy.type() == layout.type());
  EXPECT_EQ(copy.size(), layout.size());
  EXPECT_THAT(
      copy.elements(),
      ElementsAre(
          ElementLayout{.offset = 0, .data_size = 2, .padded_size = 2},
          ElementLayout{.offset = 2, .data_size = 2, .padded_size = 2},
          ElementLayout{.offset = 4, .data_size = 2, .padded_size = 2}));
}

TEST_F(TypeLayoutTest, JitTypes) {
  // Randomly test the layout of a bunch of types. TypeLayouts are generated by
  // the JIT and random xls::Values are round-tripped through the native layout.
  constexpr int64_t kValuesPerType = 10;
  auto package = CreatePackage();
  std::minstd_rand bitgen;
  for (const char* type_str :
       {"()", "bits[8]", "bits[64]", "bits[1024]", "bits[32][2]", "bits[64][5]",
        "bits[123][10]",
        "(bits[1], (bits[8], bits[16], bits[1][3])[2], bits[77])",
        "bits[1][100]", "(bits[3], (), bits[5], bits[7])[2][1][3]"}) {
    XLS_ASSERT_OK_AND_ASSIGN(Type * type,
                             Parser::ParseType(type_str, package.get()));
    TypeLayout layout = CreateTypeLayout(type);
    VLOG(1) << layout.ToString();

    std::vector<Type*> leaf_types = GetLeafTypes(type);
    ASSERT_EQ(leaf_types.size(), layout.elements().size());

    for (int64_t i = 0; i < kValuesPerType; ++i) {
      Value value = RandomValue(type, bitgen);
      VLOG(1) << value.ToString();

      std::vector<uint8_t> buffer(layout.size(), 0xff);
      layout.ValueToNativeLayout(value, buffer.data());

      XLS_VLOG_LINES(1, BytesToString(buffer));

      EXPECT_EQ(layout.NativeLayoutToValue(buffer.data()), value);

      // Verify padding bits and bytes are zero in the buffer for each element.
      for (int64_t leaf_index = 0; leaf_index < leaf_types.size();
           ++leaf_index) {
        Type* leaf_type = leaf_types.at(leaf_index);
        const ElementLayout& element_layout = layout.elements()[leaf_index];
        if (element_layout.data_size == 0) {
          continue;
        }

        if (leaf_type->GetFlatBitCount() % 8 != 0) {
          // Native layout has padding bits in the most-significant byte of the
          // data.
          uint8_t padding_mask =
              static_cast<uint8_t>(~Mask(leaf_type->GetFlatBitCount() % 8));
          uint8_t msb_data_byte =
              buffer.at(element_layout.offset + element_layout.data_size - 1);
          EXPECT_EQ(padding_mask & msb_data_byte, 0);
        }

        for (int64_t i = element_layout.data_size;
             i < element_layout.padded_size; ++i) {
          // Native layout has padding bytes above the actual data.
          EXPECT_EQ(buffer.at(element_layout.offset + i), 0);
        }
      }
    }
  }
}

}  // namespace
}  // namespace xls
