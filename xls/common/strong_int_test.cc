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

#include "xls/common/strong_int.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>

#include "absl/container/node_hash_map.h"
#include "absl/hash/hash_testing.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using ::testing::Eq;

XLS_DEFINE_STRONG_INT_TYPE(StrongInt8, int8_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongUInt8, uint8_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongInt16, int16_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongUInt16, uint16_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongInt32, int32_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongInt64, int64_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongUInt32, uint32_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongUInt64, uint64_t);
XLS_DEFINE_STRONG_INT_TYPE(StrongLong, long);  // NOLINT

TEST(StrongIntTypeIdTest, TypeIdIsAsExpected) {
  EXPECT_EQ("StrongInt8", StrongInt8::TypeName());
  EXPECT_EQ("StrongLong", StrongLong::TypeName());
}

template <typename T>
class StrongIntTest : public ::testing::Test {
 public:
  using StrongIntTypeUnderTest = T;
};

// All tests will be executed on the following StrongInt<> types.
using SupportedStrongIntTypes =
    ::testing::Types<StrongInt8, StrongUInt8, StrongInt16, StrongUInt16,
                     StrongInt32, StrongInt64, StrongUInt64, StrongLong>;

TYPED_TEST_SUITE(StrongIntTest, SupportedStrongIntTypes);

// NOTE: On all tests, we use the accessor value() as to not invoke the
// comparison operators which must themselves be tested.

TYPED_TEST(StrongIntTest, TestTraits) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  EXPECT_TRUE(std::is_standard_layout<T>::value);
  EXPECT_TRUE(std::is_trivially_copy_constructible<T>::value);
  EXPECT_TRUE(std::is_trivially_copy_assignable<T>::value);
  EXPECT_TRUE(std::is_trivially_destructible<T>::value);
}

TYPED_TEST(StrongIntTest, TestCtors) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  {  // Test default construction.
    T x;
    EXPECT_EQ(V(), x.value());
  }

  {  // Test construction from a value.
    T x(93);
    EXPECT_EQ(V(93), x.value());
  }

  {  // Test construction from a negative value.
    T x(-1);
    EXPECT_EQ(V(-1), x.value());
  }

  {  // Test copy construction.
    T x(76);
    T y(x);
    EXPECT_EQ(V(76), y.value());
  }

  {  // Test construction from int8_t.
    constexpr int8_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());
    static_assert(T(i).value() == 93, "value() is not constexpr");

    int8_t j = -76;
    T y(j);
    EXPECT_EQ(V(-76), y.value());
  }

  {  // Test construction from uint8_t.
    uint8_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());
  }

  {  // Test construction from int16_t.
    int16_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());

    int16_t j = -76;
    T y(j);
    EXPECT_EQ(V(-76), y.value());
  }

  {  // Test construction from uint16_t.
    uint16_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());
  }

  {  // Test construction from int32_t.
    int32_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());

    int32_t j = -76;
    T y(j);
    EXPECT_EQ(V(-76), y.value());
  }

  {  // Test construction from uint32_t.
    uint32_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());
  }

  {  // Test construction from int64_t.
    int64_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());

    int64_t j = -76;
    T y(j);
    EXPECT_EQ(V(-76), y.value());
  }

  {  // Test construction from uint64_t.
    uint64_t i = 93;
    T x(i);
    EXPECT_EQ(V(93), x.value());
  }

  {  // Test construction from float.
    float i = 93.1;
    T x(i);
    EXPECT_EQ(V(93), x.value());

    // It is undefined to init an unsigned int from a negative float.
    if (std::numeric_limits<V>::is_signed) {
      float j = -76.1;
      T y(j);
      EXPECT_EQ(V(-76.1), y.value());
    }
  }

  {  // Test construction from double.
    double i = 93.1;
    T x(i);
    EXPECT_EQ(V(93), x.value());

    // It is undefined to init an unsigned int from a negative double.
    if (std::numeric_limits<V>::is_signed) {
      double j = -76.1;
      T y(j);
      EXPECT_EQ(V(-76.1), y.value());
    }
  }

  {  // Test construction from long double.
    long double i = 93.1;
    T x(i);
    EXPECT_EQ(V(93), x.value());

    // It is undefined to init an unsigned int from a negative long double.
    if (std::numeric_limits<V>::is_signed) {
      long double j = -76.1;
      T y(j);
      EXPECT_EQ(V(-76.1), y.value());
    }
  }

  {  // Test constexpr assignment
    constexpr T x(123);
    EXPECT_EQ(V(123), x.value());
  }
}

namespace {
struct PositiveValidator {
  template <class T, class U>
  static bool ValidateInit(U i) {
    if (i < 0) {
      std::cerr << "PositiveValidator" << '\n';
      abort();
    }
    return true;
  }
};
}  // namespace

TYPED_TEST(StrongIntTest, TestCtorDeath) {
  using V = typename TestFixture::StrongIntTypeUnderTest::ValueType;
  if (std::numeric_limits<V>::is_signed) {
    struct CustomTag {};
    using T = StrongInt<CustomTag, V, PositiveValidator>;
    EXPECT_DEATH(T(static_cast<V>(-123)), "PositiveValidator");
  }
}

TYPED_TEST(StrongIntTest, TestMetadata) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  T t;
  EXPECT_EQ(std::numeric_limits<V>::max(), t.Max());
  EXPECT_EQ(std::numeric_limits<V>::min(), t.Min());
}

TYPED_TEST(StrongIntTest, TestUnaryOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  {  // Test unary plus and minus of positive values.
    T x(123);
    EXPECT_EQ(V(123), (+x).value());
    EXPECT_EQ(V(-123), (-x).value());
  }
  {  // Test unary plus and minus of negative values.
    T x(-123);
    EXPECT_EQ(V(-123), (+x).value());
    EXPECT_EQ(V(123), (-x).value());
  }
  {  // Test logical not of positive values.
    T x(123);
    EXPECT_EQ(false, !x);
    EXPECT_EQ(true, !!x);
  }
  {  // Test logical not of negative values.
    T x(-123);
    EXPECT_EQ(false, !x);
    EXPECT_EQ(true, !!x);
  }
  {  // Test logical not of zero.
    T x(0);
    EXPECT_EQ(true, !x);
    EXPECT_EQ(false, !!x);
  }
  {  // Test bitwise not of positive values.
    T x(123);
    EXPECT_EQ(V(~(x.value())), (~x).value());
    EXPECT_EQ(x.value(), (~~x).value());
  }
  {  // Test bitwise not of zero.
    T x(0x00);
    EXPECT_EQ(V(~(x.value())), (~x).value());
    EXPECT_EQ(x.value(), (~~x).value());
  }
}

TYPED_TEST(StrongIntTest, TestIncrementDecrementOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  {  // Test simple increments and decrements.
    T x(0);
    EXPECT_EQ(V(0), x.value());
    EXPECT_EQ(V(0), (x++).value());
    EXPECT_EQ(V(1), x.value());
    EXPECT_EQ(V(2), (++x).value());
    EXPECT_EQ(V(2), x.value());
    EXPECT_EQ(V(2), (x--).value());
    EXPECT_EQ(V(1), x.value());
    EXPECT_EQ(V(0), (--x).value());
    EXPECT_EQ(V(0), x.value());
  }
}

TYPED_TEST(StrongIntTest, TestAssignmentOperator) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  {  // Test simple assignment from the same type.
    T x(12);
    T y(34);
    EXPECT_EQ(y.value(), (x = y).value());
    EXPECT_EQ(y.value(), x.value());
  }
#if 0  // These should fail to compile.
  {
    T x(12);
    x = 34;     // Can't assign from int.
    x = V(34);  // Can't assign from ValueType.
    x = 34.0;   // Can't assign from double.
  }
#endif
}

#define TEST_T_OP_T(xval, op, yval)                                           \
  {                                                                           \
    T x(xval);                                                                \
    T y(yval);                                                                \
    V expected = x.value() op y.value();                                      \
    EXPECT_EQ(expected, (x op y).value());                                    \
    EXPECT_EQ(expected, (x op## = y).value());                                \
    EXPECT_EQ(expected, x.value());                                           \
    constexpr T cx_x(xval);                                                   \
    constexpr T cx_y(yval);                                                   \
    constexpr V cx_expected = static_cast<V>(cx_x.value() op cx_y.value());   \
    static_assert((cx_x op cx_y) == T(cx_expected), #xval " " #op " " #yval); \
  }

TYPED_TEST(StrongIntTest, TestPlusOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test positive vs. positive addition.
  TEST_T_OP_T(9, +, 3)
  // Test negative vs. positive addition.
  TEST_T_OP_T(-9, +, 3)
  // Test positive vs. negative addition.
  TEST_T_OP_T(9, +, -3)
  // Test negative vs. negative addition.
  TEST_T_OP_T(-9, +, -3)
  // Test addition by zero.
  TEST_T_OP_T(93, +, 0);

#if 0  // These should fail to compile.
  {
    T x(9);
    x + 3;     // Can't operate on int.
    x += 3;
    x + V(3);  // Can't operate on ValueType.
    x += V(3);
    x + 3.0;   // Can't operate on double.
    x += 3.0;
  }
#endif
}

TYPED_TEST(StrongIntTest, TestMinusOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test positive vs. positive subtraction.
  TEST_T_OP_T(9, -, 3)
  // Test negative vs. positive subtraction.
  TEST_T_OP_T(-9, -, 3)
  // Test positive vs. negative subtraction.
  TEST_T_OP_T(9, -, -3)
  // Test negative vs. negative subtraction.
  TEST_T_OP_T(-9, -, -3)
  // Test positive vs. positive subtraction resulting in negative.
  TEST_T_OP_T(3, -, 9);
  // Test subtraction of zero.
  TEST_T_OP_T(93, -, 0);
  // Test subtraction from zero.
  TEST_T_OP_T(0, -, 93);

#if 0  // These should fail to compile.
  {
    T x(9);
    x - 3;     // Can't operate on int.
    x -= 3;
    x - V(3);  // Can't operate on ValueType.
    x -= V(3);
    x - 3.0;   // Can't operate on double.
    x -= 3.0;
  }
#endif
}

#define TEST_T_OP_NUM(xval, op, numtype, yval)                                \
  {                                                                           \
    T x(xval);                                                                \
    numtype y = yval;                                                         \
    V expected = x.value() op y;                                              \
    EXPECT_EQ(expected, (x op y).value());                                    \
    EXPECT_EQ(expected, (x op## = y).value());                                \
    EXPECT_EQ(expected, x.value());                                           \
    constexpr T cx_x(xval);                                                   \
    constexpr V cx_expected = static_cast<V>(cx_x.value() op yval);           \
    static_assert((cx_x op yval) == T(cx_expected), #xval " " #op " " #yval); \
  }
#define TEST_NUM_OP_T(numtype, xval, op, yval)                              \
  {                                                                         \
    numtype x = xval;                                                       \
    T y(yval);                                                              \
    V expected = x op y.value();                                            \
    EXPECT_EQ(expected, (x op y).value());                                  \
    constexpr T cx_y(yval);                                                 \
    constexpr V cx_expected = static_cast<V>(xval op cx_y.value());         \
    static_assert(xval op cx_y == T(cx_expected), #xval " " #op " " #yval); \
  }

TYPED_TEST(StrongIntTest, TestMultiplyOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test positive vs. positive multiplication.
  TEST_T_OP_NUM(9, *, V, 3);
  TEST_NUM_OP_T(V, 9, *, 3);
  if (std::is_signed<V>::value) {
    // Test negative vs. positive multiplication.
    TEST_T_OP_NUM(-9, *, V, 3);
    TEST_NUM_OP_T(V, -9, *, 3);
    // Test positive vs. negative multiplication.
    TEST_T_OP_NUM(9, *, V, -3);
    TEST_NUM_OP_T(V, 9, *, -3);
    // Test negative vs. negative multiplication.
    TEST_T_OP_NUM(-9, *, V, -3);
    TEST_NUM_OP_T(V, -9, *, -3);
  }
  // Test multiplication by one.
  TEST_T_OP_NUM(93, *, V, 1);
  TEST_NUM_OP_T(V, 93, *, 1);
  // Test multiplication by zero.
  TEST_T_OP_NUM(93, *, V, 0);
  TEST_NUM_OP_T(V, 93, *, 0);
  if (std::is_signed<V>::value) {
    // Test multiplication by a negative.
    TEST_T_OP_NUM(93, *, V, -1);
    TEST_NUM_OP_T(V, 93, *, -1);
  }
  // Test multiplication by int8_t.
  TEST_T_OP_NUM(39, *, int8_t, 2);
  TEST_NUM_OP_T(int8_t, 39, *, 2);
  // Test multiplication by uint8_t.
  TEST_T_OP_NUM(39, *, uint8_t, 2);
  TEST_NUM_OP_T(uint8_t, 39, *, 2);
  // Test multiplication by int16_t.
  TEST_T_OP_NUM(39, *, int16_t, 2);
  TEST_NUM_OP_T(int16_t, 39, *, 2);
  // Test multiplication by uint16_t.
  TEST_T_OP_NUM(39, *, uint16_t, 2);
  TEST_NUM_OP_T(uint16_t, 39, *, 2);
  // Test multiplication by int32_t.
  TEST_T_OP_NUM(39, *, int32_t, 2);
  TEST_NUM_OP_T(int32_t, 39, *, 2);
  // Test multiplication by uint32_t.
  TEST_T_OP_NUM(39, *, uint32_t, 2);
  TEST_NUM_OP_T(uint32_t, 39, *, 2);
  // Test multiplication by int64_t.
  TEST_T_OP_NUM(39, *, int64_t, 2);
  TEST_NUM_OP_T(int64_t, 39, *, 2);
  // Test multiplication by uint64_t.
  TEST_T_OP_NUM(39, *, uint64_t, 2);
  TEST_NUM_OP_T(uint64_t, 39, *, 2);
  // Test multiplication by float.
  TEST_T_OP_NUM(39, *, float, 2.1);
  TEST_NUM_OP_T(float, 39, *, 2.1);
  // Test multiplication by double.
  TEST_T_OP_NUM(39, *, double, 2.1);
  TEST_NUM_OP_T(double, 39, *, 2.1);
  // Test multiplication by long double.
  TEST_T_OP_NUM(39, *, long double, 2.1);
  TEST_NUM_OP_T(long double, 39, *, 2.1);

#if 0  // These should fail to compile.
  {
    T x(9);
    x * T(3);  // Can't operate on IntType.
    x *= T(3);
  }
#endif
}

TYPED_TEST(StrongIntTest, TestDivideOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test positive vs. positive division.
  TEST_T_OP_NUM(9, /, V, 3);
  // Test negative vs. positive division.
  TEST_T_OP_NUM(-9, /, V, 3);
  // Test positive vs. negative division.
  TEST_T_OP_NUM(9, /, V, -3);
  // Test negative vs. negative division.
  TEST_T_OP_NUM(-9, /, V, -3);
  // Test division by one.
  TEST_T_OP_NUM(93, /, V, 1);
  // Test division by a negative.
  TEST_T_OP_NUM(93, /, V, -1);
  // Test division by int8_t.
  TEST_T_OP_NUM(93, /, int8_t, 2);
  // Test division by uint8_t.
  TEST_T_OP_NUM(93, /, uint8_t, 2);
  // Test division by int16_t.
  TEST_T_OP_NUM(93, /, int16_t, 2);
  // Test division by uint16_t.
  TEST_T_OP_NUM(93, /, uint16_t, 2);
  // Test division by int32_t.
  TEST_T_OP_NUM(93, /, int32_t, 2);
  // Test division by uint32_t.
  TEST_T_OP_NUM(93, /, uint32_t, 2);
  // Test division by int64_t.
  TEST_T_OP_NUM(93, /, int64_t, 2);
  // Test division by uint64_t.
  TEST_T_OP_NUM(93, /, uint64_t, 2);
  // Test division by float.
  TEST_T_OP_NUM(93, /, float, 2.1);
  // Test division by double.
  TEST_T_OP_NUM(93, /, double, 2.1);
  // Test division by long double.
  TEST_T_OP_NUM(93, /, long double, 2.1);

#if 0  // These should fail to compile.
  {
    T x(9);
    x / T(3);  // Can't operate on IntType.
    x /= T(3);
  }
#endif
}

TYPED_TEST(StrongIntTest, TestModuloOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test positive vs. positive modulo.
  TEST_T_OP_NUM(7, %, V, 6);
  // Test negative vs. positive modulo.
  TEST_T_OP_NUM(-7, %, V, 6);
  // Test positive vs. negative modulo.
  TEST_T_OP_NUM(7, %, V, -6);
  // Test negative vs. negative modulo.
  TEST_T_OP_NUM(-7, %, V, -6);
  // Test modulo by one.
  TEST_T_OP_NUM(93, %, V, 1);
  // Test modulo by a negative.
  TEST_T_OP_NUM(93, %, V, -5);
  // Test modulo by int8_t.
  TEST_T_OP_NUM(93, %, int8_t, 5);
  // Test modulo by uint8_t.
  TEST_T_OP_NUM(93, %, uint8_t, 5);
  // Test modulo by int16_t.
  TEST_T_OP_NUM(93, %, int16_t, 5);
  // Test modulo by uint16_t.
  TEST_T_OP_NUM(93, %, uint16_t, 5);
  // Test modulo by int32_t.
  TEST_T_OP_NUM(93, %, int32_t, 5);
  // Test modulo by uint32_t.
  TEST_T_OP_NUM(93, %, uint32_t, 5);
  // Test modulo by int64_t.
  TEST_T_OP_NUM(93, %, int64_t, 5);
  // Test modulo by uint64_t.
  TEST_T_OP_NUM(93, %, uint64_t, 5);
  // Test modulo by a larger value.
  TEST_T_OP_NUM(93, %, V, 100);

#if 0  // These should fail to compile.
  {
    T x(9);
    x % T(3);          // Can't operate on IntType.
    x %= T(3);
    x % 3.0;           // Can't operate on float.
    x %= 3.0;
  }
#endif
}

TYPED_TEST(StrongIntTest, TestLeftShiftOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test basic shift.
  TEST_T_OP_NUM(0x09, <<, int, 3);
  // Test shift by zero.
  TEST_T_OP_NUM(0x09, <<, int, 0);

#if 0  // These should fail to compile.
  {
    T x(9);
    x << T(3);          // Can't operate on IntType.
    x <<= T(3);
    x << 3.0;           // Can't operate on float.
    x <<= 3.0;
  }
#endif
}

TYPED_TEST(StrongIntTest, TestRightShiftOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test basic shift.
  TEST_T_OP_NUM(0x09, >>, int, 3);
  // Test shift by zero.
  TEST_T_OP_NUM(0x09, >>, int, 0);

#if 0  // These should fail to compile.
  {
    T x(9);
    x >> T(3);          // Can't operate on IntType.
    x >>= T(3);
    x >> 3.0;           // Can't operate on float.
    x >>= 3.0;
  }
#endif
}

TYPED_TEST(StrongIntTest, TestBitAndOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test basic bit-and.
  TEST_T_OP_T(0x09, &, 0x03);
  // Test bit-and by zero.
  TEST_T_OP_T(0x09, &, 0x00);

#if 0  // These should fail to compile.
  {
    T x(9);
    x & 3;             // Can't operate on int.
    x &= 3;
    x & 3.0;           // Can't operate on float.
    x &= 3.0;
  }
#endif
}

TYPED_TEST(StrongIntTest, TestBitOrOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test basic bit-or.
  TEST_T_OP_T(0x09, |, 0x03);
  // Test bit-or by zero.
  TEST_T_OP_T(0x09, |, 0x00);

#if 0  // These should fail to compile.
  {
    T x(9);
    x | 3;             // Can't operate on int.
    x |= 3;
    x | 3.0;           // Can't operate on float.
    x |= 3.0;
  }
#endif
}

TYPED_TEST(StrongIntTest, TestBitXorOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using V = typename T::ValueType;

  // Test basic bit-xor.
  TEST_T_OP_T(0x09, ^, 0x03);
  // Test bit-xor by zero.
  TEST_T_OP_T(0x09, ^, 0x00);

#if 0  // These should fail to compile.
  {
    T x(9);
    x ^ 3;             // Can't operate on int.
    x ^= 3;
    x ^ 3.0;           // Can't operate on float.
    x ^= 3.0;
  }
#endif
}

TYPED_TEST(StrongIntTest, TestComparisonOperators) {
  using T = typename TestFixture::StrongIntTypeUnderTest;

  T x(93);

  EXPECT_TRUE(x == T(93));
  EXPECT_TRUE(T(93) == x);
  EXPECT_FALSE(x == T(76));
  EXPECT_FALSE(T(76) == x);

  EXPECT_TRUE(x != T(76));
  EXPECT_TRUE(T(76) != x);
  EXPECT_FALSE(x != T(93));
  EXPECT_FALSE(T(93) != x);

  EXPECT_TRUE(x < T(94));
  EXPECT_FALSE(T(94) < x);
  EXPECT_FALSE(x < T(76));
  EXPECT_TRUE(T(76) < x);

  EXPECT_TRUE(x <= T(94));
  EXPECT_FALSE(T(94) <= x);
  EXPECT_FALSE(x <= T(76));
  EXPECT_TRUE(T(76) <= x);
  EXPECT_TRUE(x <= T(93));
  EXPECT_TRUE(T(93) <= x);

  EXPECT_TRUE(x > T(76));
  EXPECT_FALSE(T(76) > x);
  EXPECT_FALSE(x > T(94));
  EXPECT_TRUE(T(94) > x);

  EXPECT_TRUE(x >= T(76));
  EXPECT_FALSE(T(76) >= x);
  EXPECT_FALSE(x >= T(94));
  EXPECT_TRUE(T(94) >= x);
  EXPECT_TRUE(x >= T(93));
  EXPECT_TRUE(T(93) >= x);
}

TYPED_TEST(StrongIntTest, TestStreamOutputOperator) {
  using T = typename TestFixture::StrongIntTypeUnderTest;

  T x(93);
  std::ostringstream out;
  out << x;
  EXPECT_EQ("93", out.str());
}

TYPED_TEST(StrongIntTest, TestHasher) {
  using T = typename TestFixture::StrongIntTypeUnderTest;

  typename T::Hasher hasher;
  EXPECT_EQ(hasher(T(0)), hasher(T(0)));
  EXPECT_NE(hasher(T(1)), hasher(T(2)));
}

TYPED_TEST(StrongIntTest, TestHashFunctor) {
  absl::node_hash_map<typename TestFixture::StrongIntTypeUnderTest, char,
                      typename TestFixture::StrongIntTypeUnderTest::Hasher>
      map;
  typename TestFixture::StrongIntTypeUnderTest a(0);
  map[a] = 'c';
  EXPECT_EQ('c', map[a]);
  map[++a] = 'o';
  EXPECT_EQ('o', map[a]);
}

TYPED_TEST(StrongIntTest, TestHash) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using ValueType = typename T::ValueType;
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      T(std::numeric_limits<ValueType>::min()),
      T(std::numeric_limits<ValueType>::min() + 1),
      T(std::numeric_limits<ValueType>::min() + 10),
      // Note that we use types with at least 8 bits, so adding/subtracting 100
      // gives us valid values.
      T(std::numeric_limits<ValueType>::min() + 100),
      T(std::numeric_limits<ValueType>::max() - 100),
      T(std::numeric_limits<ValueType>::max() - 10),
      T(std::numeric_limits<ValueType>::max() - 1),
      T(std::numeric_limits<ValueType>::max()),
  }));
}

TYPED_TEST(StrongIntTest, TestStrongIntRange) {
  using TypeUnderTest = typename TestFixture::StrongIntTypeUnderTest;
  const int64_t kMaxOuterIterations = 100;
  for (int64_t to = 0; to < kMaxOuterIterations; ++to) {
    int count = 0;
    int64_t sum = 0;
    for (const auto x : MakeStrongIntRange(TypeUnderTest(to))) {
      ++count;
      sum += x.value();
    }
    EXPECT_EQ(to, count);
    EXPECT_EQ(to * (to - 1) / 2, sum);
  }
  for (int64_t to = 0; to < kMaxOuterIterations; ++to) {
    for (int64_t from = 0; from <= to; ++from) {
      int count = 0;
      int64_t sum = 0;
      for (const auto x :
           MakeStrongIntRange(TypeUnderTest(from), TypeUnderTest(to))) {
        ++count;
        sum += x.value();
      }
      EXPECT_EQ(to - from, count);
      EXPECT_EQ((to * (to - 1) / 2) - (from * (from - 1) / 2), sum);
    }
  }
}

// Test Min() and Max() can be used in constexpr.
TYPED_TEST(StrongIntTest, ConstexprMinMax) {
  using T = typename TestFixture::StrongIntTypeUnderTest;
  using ValueType = typename T::ValueType;
  constexpr ValueType max = T::Max();
  constexpr ValueType min = T::Min();
  (void)max;
  (void)min;
}

template <typename Ttest, typename Tbig>
bool ExhaustiveTest() {
  using V = typename Ttest::ValueType;
  Tbig v_min = std::numeric_limits<V>::min();
  Tbig v_max = std::numeric_limits<V>::max();
  for (Tbig lhs = v_min; lhs <= v_max; ++lhs) {
    for (Tbig rhs = v_min; rhs <= v_max; ++rhs) {
      {
        Ttest t_lhs(lhs);
        Ttest t_rhs(rhs);
        EXPECT_EQ(Ttest(lhs + rhs), t_lhs + t_rhs);
      }
      {
        Ttest t_lhs(lhs);
        Ttest t_rhs(rhs);
        EXPECT_EQ(Ttest(lhs - rhs), t_lhs - t_rhs);
      }
      {
        Ttest t_lhs(lhs);
        EXPECT_EQ(Ttest(lhs * rhs), t_lhs * rhs);
      }
      {
        Ttest t_lhs(lhs);
        if (rhs != 0) {
          EXPECT_EQ(Ttest(lhs / rhs), t_lhs / rhs);
        }
      }
      {
        Ttest t_lhs(lhs);
        if (rhs != 0) {
          EXPECT_EQ(Ttest(lhs % rhs), t_lhs % rhs);
        }
      }
    }
  }
  return true;
}

TEST(StrongIntTest, Exhaustive) {
  EXPECT_TRUE((ExhaustiveTest<StrongInt8, int>()));
  EXPECT_TRUE((ExhaustiveTest<StrongUInt8, int>()));
}

TEST(StrongIntTest, ExplicitCasting) {
  StrongInt8 x(8);
  EXPECT_THAT(static_cast<int8_t>(x), Eq(x.value()));
  EXPECT_THAT(static_cast<size_t>(x), Eq(x.value()));
}

// Create some types outside the util_intops:: namespace to prove that
// conversions work.
namespace other_namespace {

XLS_DEFINE_STRONG_INT_TYPE(Inches, int64_t);
XLS_DEFINE_STRONG_INT_TYPE(Feet, int64_t);
XLS_DEFINE_STRONG_INT_TYPE(Centimeters, int32_t);

constexpr Feet StrongIntConvert(const Inches& arg, Feet* /* unused */) {
  return Feet(arg.value() / 12);
}
constexpr Centimeters StrongIntConvert(const Inches& arg,
                                       Centimeters* /* unused */) {
  return Centimeters(arg.value() * 2.54);
}

TEST(StrongIntTest, TestConversion) {
  {  // Test simple copy construction.
    Inches in1(12);
    Inches in2(in1);
    EXPECT_EQ(12, in2.value());
  }
  {  // Test conversion from Inches to Feet.
    Inches in(60);
    Feet ft(in);
    EXPECT_EQ(5, ft.value());

    constexpr Inches kIn(60);
    constexpr Feet kFt(kIn);
    EXPECT_EQ(kFt, ft);
  }
  {  // Test conversion from Inches to Centimeters.
    Inches in(10);
    Centimeters cm(in);
    EXPECT_EQ(25, cm.value());

    constexpr Inches kIn(10);
    constexpr Centimeters kCm(kIn);
    EXPECT_EQ(kCm, cm);
  }
}

// Test SFINAE on template<T> constexpr StrongInt(T init_value) constructor.
// Without it, the non-convertible case in the assertions below would become a
// hard compilation failure because of the compile-time evaluation of
// static_cast<ValueType>(init_value) in the _constexpr_ constructor body.
template <typename T>
struct StrongIntTestHelper {
  template <typename U, typename = typename std::enable_if<
                            std::is_constructible<StrongInt<void, T>, U>::value,
                            void>::type>
  StrongIntTestHelper(U x) {}  // NOLINT
};

static_assert(!std::is_convertible<void, StrongIntTestHelper<int>>::value, "");
static_assert(std::is_convertible<int, StrongIntTestHelper<int>>::value, "");

// Test the IsStrongInt type trait.
static_assert(IsStrongInt<StrongInt8>::value, "");
static_assert(IsStrongInt<StrongUInt16>::value, "");
static_assert(!IsStrongInt<int8_t>::value, "");
static_assert(!IsStrongInt<long>::value, "");  // NOLINT
static_assert(!IsStrongInt<void>::value, "");

}  // namespace other_namespace

}  // namespace
}  // namespace xls
