// Copyright 2023 The XLS Authors
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

#include "xls/simulation/generic/ir_value_access_methods.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"

namespace xls::simulation::generic {
namespace {

class BitsVectorTest : public ::testing::Test {
 protected:
  BitsVectorTest()
      : bits_types_{BitsType(1), BitsType(31), BitsType(64), BitsType(128)} {}
  void SetUp() {
    this->bits_ = std::vector<Value>{
        ZeroOfType(&bits_types_[0]), ZeroOfType(&bits_types_[1]),
        ZeroOfType(&bits_types_[2]), ZeroOfType(&bits_types_[3])};
  }

  BitsType bits_types_[4];
  std::vector<Value> bits_;
};

TEST_F(BitsVectorTest, OutOfRangeRead) {
  EXPECT_THAT(
      ValueArrayReadUInt64(this->bits_, std::string("OutOfRangeRead"), 32, 1),
      ::xls::status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Offset: 32 is outside OutOfRangeRead range"));
}

TEST_F(BitsVectorTest, SingleBitsReadLateStart) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 23; i < 31; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);
  EXPECT_THAT(ValueArrayReadUInt64(
                  this->bits_, std::string("SingleBitsReadLateStart"), 3, 1),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFF));
}

TEST_F(BitsVectorTest, SingleBitsReadEarlyEnd) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 7; i < 15; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);
  EXPECT_THAT(ValueArrayReadUInt64(this->bits_,
                                   std::string("SingleBitsReadEarlyEnd"), 1, 1),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFF));
}

TEST_F(BitsVectorTest, SingleBitsReadOverMultipleValues) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 0; i < 15; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);

  bit_representation = this->bits_[2].bits();
  bit_representation = bit_representation.UpdateWithSet(31, 1);
  this->bits_[2] = Value(bit_representation);

  EXPECT_THAT(
      ValueArrayReadUInt64(
          this->bits_, std::string("SingleBitsReadOverMultipleValues"), 0, 8),
      ::xls::status_testing::IsOkAndHolds((uint64_t)0x800000000000FFFE));
}

TEST_F(BitsVectorTest, SingleBitsReadPartial) {
  Bits bit_representation = this->bits_[3].bits();
  for (int i = 112; i < 128; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[3] = Value(bit_representation);

  EXPECT_THAT(ValueArrayReadUInt64(this->bits_,
                                   std::string("SingleBitsReadPartial"), 24, 8),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFFFF0000));
}

TEST_F(BitsVectorTest, OutOfRangeWrite) {
  EXPECT_THAT(ValueArrayWriteUInt64(this->bits_, std::string("OutOfRangeWrite"),
                                    32, 1, 0),
              ::xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  "Offset: 32 is outside OutOfRangeWrite range"));
}

TEST_F(BitsVectorTest, SingleBitsWriteLateStart) {
  Bits set_value = Bits(31);
  set_value.SetRange(23, 31, 1);
  std::vector<Value> ans{Value(Bits(1)), Value(set_value), Value(Bits(64)),
                         Value(Bits(128))};

  EXPECT_THAT(ValueArrayWriteUInt64(this->bits_,
                                    std::string("SingleBitsWriteLateStart"), 3,
                                    1, (uint64_t)0xFFFF),
              ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(BitsVectorTest, SingleBitsWriteEarlyEnd) {
  Bits set_value = Bits(31);
  set_value.SetRange(7, 15, 1);
  std::vector<Value> ans{Value(Bits(1)), Value(set_value), Value(Bits(64)),
                         Value(Bits(128))};

  EXPECT_THAT(
      ValueArrayWriteUInt64(this->bits_, std::string("SingleBitsWriteEarlyEnd"),
                            1, 1, (uint64_t)0xFFFF),
      ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(BitsVectorTest, SingleBitsWriteOverMultipleValues) {
  Bits set_value = Bits(31);
  set_value.SetRange(0, 15, 1);
  Bits set_value2 = Bits(64);
  set_value2.SetRange(31, 32, 1);
  std::vector<Value> ans{Value(Bits(1)), Value(set_value), Value(set_value2),
                         Value(Bits(128))};

  EXPECT_THAT(ValueArrayWriteUInt64(
                  this->bits_, std::string("SingleBitsWriteOverMultipleValues"),
                  0, 8, (uint64_t)0x800000000000FFFE),
              ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(BitsVectorTest, SingleBitsWritePartial) {
  Bits set_value = Bits(128);
  set_value.SetRange(112, 128, 1);
  std::vector<Value> ans{Value(Bits(1)), Value(Bits(31)), Value(Bits(64)),
                         Value(set_value)};
  EXPECT_THAT(
      ValueArrayWriteUInt64(this->bits_, std::string("SingleBitsWritePartial"),
                            26, 8, (uint64_t)0xFFFFFFFFFFFFFFFF),
      ::xls::status_testing::IsOkAndHolds(ans));
}

class TupleTest : public BitsVectorTest {
 protected:
  TupleTest()
      : tuple_type_({&bits_types_[0], &bits_types_[1], &bits_types_[2],
                     &bits_types_[3]}) {}
  void SetUp() {
    this->bits_ = std::vector<Value>{
        ZeroOfType(&bits_types_[0]), ZeroOfType(&bits_types_[1]),
        ZeroOfType(&bits_types_[2]), ZeroOfType(&bits_types_[3])};
    this->tuple_ = std::vector<Value>{ZeroOfType(&tuple_type_)};
  }

  TupleType tuple_type_;
  std::vector<Value> tuple_;
};

TEST_F(TupleTest, OutOfRangeRead) {
  EXPECT_THAT(
      ValueArrayReadUInt64(this->tuple_, std::string("OutOfRangeRead"), 32, 1),
      ::xls::status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Offset: 32 is outside OutOfRangeRead range"));
}

TEST_F(TupleTest, SingleBitsReadLateStart) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 23; i < 31; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);
  this->tuple_[0] = Value::Tuple(this->bits_);
  EXPECT_THAT(ValueArrayReadUInt64(
                  this->tuple_, std::string("SingleBitsReadLateStart"), 3, 1),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFF));
}

TEST_F(TupleTest, SingleBitsReadEarlyEnd) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 7; i < 15; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);
  this->tuple_[0] = Value::Tuple(this->bits_);
  EXPECT_THAT(ValueArrayReadUInt64(this->tuple_,
                                   std::string("SingleBitsReadEarlyEnd"), 1, 1),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFF));
}

TEST_F(TupleTest, SingleBitsReadOverMultipleValues) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 0; i < 15; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);

  bit_representation = this->bits_[2].bits();
  bit_representation = bit_representation.UpdateWithSet(31, 1);
  this->bits_[2] = Value(bit_representation);
  this->tuple_[0] = Value::Tuple(this->bits_);

  EXPECT_THAT(
      ValueArrayReadUInt64(
          this->tuple_, std::string("SingleBitsReadOverMultipleValues"), 0, 8),
      ::xls::status_testing::IsOkAndHolds((uint64_t)0x800000000000FFFE));
}

TEST_F(TupleTest, SingleBitsReadPartial) {
  Bits bit_representation = this->bits_[3].bits();
  for (int i = 112; i < 128; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[3] = Value(bit_representation);
  this->tuple_[0] = Value::Tuple(this->bits_);

  EXPECT_THAT(ValueArrayReadUInt64(this->tuple_,
                                   std::string("SingleBitsReadPartial"), 24, 8),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFFFF0000));
}

TEST_F(TupleTest, OutOfRangeWrite) {
  EXPECT_THAT(ValueArrayWriteUInt64(this->tuple_,
                                    std::string("OutOfRangeWrite"), 32, 1, 0),
              ::xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  "Offset: 32 is outside OutOfRangeWrite range"));
}

TEST_F(TupleTest, SingleBitsWriteLateStart) {
  Bits set_value = Bits(31);
  set_value.SetRange(23, 31, 1);
  std::vector<Value> ans{Value::Tuple(
      {Value(Bits(1)), Value(set_value), Value(Bits(64)), Value(Bits(128))})};

  EXPECT_THAT(ValueArrayWriteUInt64(this->tuple_,
                                    std::string("SingleBitsWriteLateStart"), 3,
                                    1, (uint64_t)0xFFFF),
              ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(TupleTest, SingleBitsWriteEarlyEnd) {
  Bits set_value = Bits(31);
  set_value.SetRange(7, 15, 1);
  std::vector<Value> ans{Value::Tuple(
      {Value(Bits(1)), Value(set_value), Value(Bits(64)), Value(Bits(128))})};

  EXPECT_THAT(ValueArrayWriteUInt64(this->tuple_,
                                    std::string("SingleBitsWriteEarlyEnd"), 1,
                                    1, (uint64_t)0xFFFF),
              ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(TupleTest, SingleBitsWriteOverMultipleValues) {
  Bits set_value = Bits(31);
  set_value.SetRange(0, 15, 1);
  Bits set_value2 = Bits(64);
  set_value2.SetRange(31, 32, 1);
  std::vector<Value> ans{Value::Tuple(
      {Value(Bits(1)), Value(set_value), Value(set_value2), Value(Bits(128))})};

  EXPECT_THAT(
      ValueArrayWriteUInt64(this->tuple_,
                            std::string("SingleBitsWriteOverMultipleValues"), 0,
                            8, (uint64_t)0x800000000000FFFE),
      ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(TupleTest, SingleBitsWritePartial) {
  Bits set_value = Bits(128);
  set_value.SetRange(112, 128, 1);
  std::vector<Value> ans{Value::Tuple(
      {Value(Bits(1)), Value(Bits(31)), Value(Bits(64)), Value(set_value)})};
  EXPECT_THAT(
      ValueArrayWriteUInt64(this->tuple_, std::string("SingleBitsWritePartial"),
                            26, 8, (uint64_t)0xFFFFFFFFFFFFFFFF),
      ::xls::status_testing::IsOkAndHolds(ans));
}

class ArrayTest : public TupleTest {
 protected:
  ArrayTest() : array_type_(1, &tuple_type_) {}

  void SetUp() {
    this->bits_ = std::vector<Value>{
        ZeroOfType(&bits_types_[0]), ZeroOfType(&bits_types_[1]),
        ZeroOfType(&bits_types_[2]), ZeroOfType(&bits_types_[3])};
    this->array_ = std::vector<Value>{ZeroOfType(&array_type_)};
  }

  ArrayType array_type_;
  std::vector<Value> array_;
};

TEST_F(ArrayTest, OutOfRangeRead) {
  EXPECT_THAT(
      ValueArrayReadUInt64(this->array_, std::string("OutOfRangeRead"), 32, 1),
      ::xls::status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Offset: 32 is outside OutOfRangeRead range"));
}

TEST_F(ArrayTest, SingleBitsReadLateStart) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 23; i < 31; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);
  this->array_[0] = Value::ArrayOrDie({Value::Tuple(this->bits_)});
  EXPECT_THAT(ValueArrayReadUInt64(
                  this->array_, std::string("SingleBitsReadLateStart"), 3, 1),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFF));
}

TEST_F(ArrayTest, SingleBitsReadEarlyEnd) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 7; i < 15; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);
  this->array_[0] = Value::ArrayOrDie({Value::Tuple(this->bits_)});
  EXPECT_THAT(ValueArrayReadUInt64(this->array_,
                                   std::string("SingleBitsReadEarlyEnd"), 1, 1),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFF));
}

TEST_F(ArrayTest, SingleBitsReadOverMultipleValues) {
  Bits bit_representation = this->bits_[1].bits();
  for (int i = 0; i < 15; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[1] = Value(bit_representation);

  bit_representation = this->bits_[2].bits();
  bit_representation = bit_representation.UpdateWithSet(31, 1);
  this->bits_[2] = Value(bit_representation);
  this->array_[0] = Value::ArrayOrDie({Value::Tuple(this->bits_)});

  EXPECT_THAT(
      ValueArrayReadUInt64(
          this->array_, std::string("SingleBitsReadOverMultipleValues"), 0, 8),
      ::xls::status_testing::IsOkAndHolds((uint64_t)0x800000000000FFFE));
}

TEST_F(ArrayTest, SingleBitsReadPartial) {
  Bits bit_representation = this->bits_[3].bits();
  for (int i = 112; i < 128; ++i) {
    bit_representation = bit_representation.UpdateWithSet(i, 1);
  }
  this->bits_[3] = Value(bit_representation);
  this->array_[0] = Value::ArrayOrDie({Value::Tuple(this->bits_)});

  EXPECT_THAT(ValueArrayReadUInt64(this->array_,
                                   std::string("SingleBitsReadPartial"), 24, 8),
              ::xls::status_testing::IsOkAndHolds((uint64_t)0xFFFF0000));
}

TEST_F(ArrayTest, OutOfRangeWrite) {
  EXPECT_THAT(ValueArrayWriteUInt64(this->array_,
                                    std::string("OutOfRangeWrite"), 32, 1, 0),
              ::xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  "Offset: 32 is outside OutOfRangeWrite range"));
}

TEST_F(ArrayTest, SingleBitsWriteLateStart) {
  Bits set_value = Bits(31);
  set_value.SetRange(23, 31, 1);
  std::vector<Value> ans{Value::ArrayOrDie({Value::Tuple(
      {Value(Bits(1)), Value(set_value), Value(Bits(64)), Value(Bits(128))})})};

  EXPECT_THAT(ValueArrayWriteUInt64(this->array_,
                                    std::string("SingleBitsWriteLateStart"), 3,
                                    1, (uint64_t)0xFFFF),
              ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(ArrayTest, SingleBitsWriteEarlyEnd) {
  Bits set_value = Bits(31);
  set_value.SetRange(7, 15, 1);
  std::vector<Value> ans{Value::ArrayOrDie({Value::Tuple(
      {Value(Bits(1)), Value(set_value), Value(Bits(64)), Value(Bits(128))})})};

  EXPECT_THAT(ValueArrayWriteUInt64(this->array_,
                                    std::string("SingleBitsWriteEarlyEnd"), 1,
                                    1, (uint64_t)0xFFFF),
              ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(ArrayTest, SingleBitsWriteOverMultipleValues) {
  Bits set_value = Bits(31);
  set_value.SetRange(0, 15, 1);
  Bits set_value2 = Bits(64);
  set_value2.SetRange(31, 32, 1);
  std::vector<Value> ans{
      Value::ArrayOrDie({Value::Tuple({Value(Bits(1)), Value(set_value),
                                       Value(set_value2), Value(Bits(128))})})};

  EXPECT_THAT(
      ValueArrayWriteUInt64(this->array_,
                            std::string("SingleBitsWriteOverMultipleValues"), 0,
                            8, (uint64_t)0x800000000000FFFE),
      ::xls::status_testing::IsOkAndHolds(ans));
}

TEST_F(ArrayTest, SingleBitsWritePartial) {
  Bits set_value = Bits(128);
  set_value.SetRange(112, 128, 1);
  std::vector<Value> ans{Value::ArrayOrDie({Value::Tuple(
      {Value(Bits(1)), Value(Bits(31)), Value(Bits(64)), Value(set_value)})})};
  EXPECT_THAT(
      ValueArrayWriteUInt64(this->array_, std::string("SingleBitsWritePartial"),
                            26, 8, (uint64_t)0xFFFFFFFFFFFFFFFF),
      ::xls::status_testing::IsOkAndHolds(ans));
}

}  // namespace
}  // namespace xls::simulation::generic
