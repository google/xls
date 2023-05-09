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

#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/unit_test.h"

namespace xlscc {
namespace {

class XlsFixedTest : public XlsccTestBase {};

TEST_F(XlsFixedTest, Add) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax + bx).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, 8, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, AddDifferentSizes) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<5, 5, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax + bx).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, 8, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, Sub) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax - bx).to_int64();
    })";
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, -2, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, Mul) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax * bx).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, 15, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, Div) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax / bx).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 32}, {"b", 4}}, 8, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, BitwiseOr) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax | bx).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0011, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, BitwiseAnd) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax & bx).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0010, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, BitwiseXor) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return (ax ^ bx).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0001, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, GreaterThan) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return ax > bx;
    })";
  RunAcDatatypeTest({{"a", 100}, {"b", 10}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, GreaterThanOrEq) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return ax >= bx;
    })";
  RunAcDatatypeTest({{"a", 100}, {"b", 100}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, LessThan) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return ax < bx;
    })";
  RunAcDatatypeTest({{"a", 10}, {"b", 100}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, LessThanOrEq) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a, long long b) {
      XlsFixed<32, 32, true> ax = a;
      XlsFixed<32, 32, true> bx = b;
      return ax <= bx;
    })";
  RunAcDatatypeTest({{"a", 100}, {"b", 100}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, WidthAndSignMembers) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(unsigned long long a) {
      const XlsFixed<32, 32, true> ax = a;
      return ax.width + ax.sign;
    })";
  RunAcDatatypeTest({{"a", 0}}, 33, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ConstBitRef) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(unsigned long long a) {
      const XlsFixed<32, 32, true> ax = a;
      int ret = 0;
      #pragma hls_unroll yes
      for(int i=0;i<32;++i) {
        ret += ax[i];
      }
      return ret;
    })";
  RunAcDatatypeTest({{"a", 0b101011}}, 4, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, SetBitRef) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(unsigned long long a) {
      XlsFixed<32, 32, true> ax = a;
      ax[5] = 1;
      ax[0] = 0;
      return ax.to_uint64();
    })";
  RunAcDatatypeTest({{"a", 0b000001}}, 0b100000, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, PreInc) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<24, 24, true> ax = a;
      ++ax;
      return (++ax).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 102, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, PostInc) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<24, 24, true> ax = a;
      ax++;
      return (ax++).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 101, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, PreDec) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<24, 24, true> ax = a;
      --ax;
      return (--ax).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 98, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, PostDec) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<24, 24, true> ax = a;
      ax--;
      return (ax--).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 99, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ArithNeg) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<24, 20, true> ax = a;
      return (-ax).to_int64();
    })";
  RunAcDatatypeTest({{"a", 100}}, -100, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, LogicNeg) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(unsigned long long a) {
      XlsFixed<8, 8, false> ax = a;
      // operator ~ always returns signed, sign extends
      return (~ax).to_int64();
    })";
  RunAcDatatypeTest({{"a", 3}}, -4, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, LogicNegate) {
  const std::string content = R"(
    #include "xls_fixed.h"

    unsigned long long my_package(unsigned long long a) {
      XlsFixed<8, 8, false> ax = a;
      return ax.bit_complement().to_uint64();
    })";
  RunAcDatatypeTest({{"a", 3}}, 252U, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, Clz) {
  const std::string content = R"(
    #include "xls_fixed.h"

    unsigned long long my_package(unsigned long long a) {
      XlsFixed<8, 8, false> ax = a;
      return ax.clz();
    })";
  RunAcDatatypeTest({{"a", 0b1000}}, 4, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0b10000000}}, 0, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0}}, 8, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 1}}, 7, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, LeadingSign) {
  const std::string content = R"(
    #include "xls_fixed.h"

    unsigned long long my_package(int signed_mode, unsigned long long a) {
      if(signed_mode) {
        XlsFixed<16, 16, true> ax = a;
        return ax.leading_sign();
      } else {
        XlsFixed<16, 16, false> ax = a;
        return ax.leading_sign();
      }
    })";
  RunAcDatatypeTest({{"signed_mode", 0}, {"a", 0b1000}}, 12, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"signed_mode", 0}, {"a", -0b1000}}, 0, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"signed_mode", 1}, {"a", 0b1000}}, 11, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"signed_mode", 1}, {"a", -0b1000}}, 12, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, LeadingSignAllSign) {
  const std::string content = R"(
    #include "xls_fixed.h"

    bool my_package(int signed_mode, unsigned long long a) {
      bool all_sign;
      if(signed_mode) {
        XlsFixed<16, 16, true> ax = a;
        ax.leading_sign(all_sign);
      } else {
        XlsFixed<16, 16, false> ax = a;
        ax.leading_sign(all_sign);
      }
      return all_sign;
    })";
  RunAcDatatypeTest({{"signed_mode", 0}, {"a", 0b1000}},
                    static_cast<uint64_t>(false), content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"signed_mode", 0}, {"a", 0}}, static_cast<uint64_t>(true),
                    content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"signed_mode", 0}, {"a", -1}},
                    static_cast<uint64_t>(false), content,
                    xabsl::SourceLocation::current());

  RunAcDatatypeTest({{"signed_mode", 1}, {"a", 0b1000}},
                    static_cast<uint64_t>(false), content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"signed_mode", 1}, {"a", 0}}, static_cast<uint64_t>(true),
                    content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"signed_mode", 1}, {"a", -1}},
                    static_cast<uint64_t>(true), content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, Not) {
  const std::string content = R"(
    #include "xls_fixed.h"

    unsigned long long my_package(unsigned long long a) {
      XlsFixed<5, 5, false> ax = a;
      return !ax;
    })";
  RunAcDatatypeTest({{"a", 5}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, Equal) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<24, 24, false> ax = 3;
      ax = XlsFixed<30, 30, false>(a);
      return ax.to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 100, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, EqualTo) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<24, 24, false> ax = a;
      return ax == 5;
    })";
  RunAcDatatypeTest({{"a", 100}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, NotEqual) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<24, 24, false> ax = a;
      return ax != 5;
    })";
  RunAcDatatypeTest({{"a", 100}}, 1, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftRight) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      return (XlsFixed<24, 24, false>(a) >> 1).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 50, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftRightEq) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<24, 24, false> ax = a;
      ax >>= 2;
      return ax.to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 25, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftRightXlsFixed) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<4, 4, false> sa = 1;
      return (XlsFixed<24, 24, false>(a) >> sa).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 50, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftRightXlsFixedLarge) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<2, 2, false> sa = 3;
      return (XlsFixed<24, 24, false>(a) >> sa).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 80}}, 10, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftRightEqXlsFixed) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<4, 4, false> sa = 2;
      XlsFixed<24, 24, false> ax = a;
      ax >>= sa;
      return ax.to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 25, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftRightNegative) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      return (XlsFixed<24, 24, false>(a) >> -1).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 200, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftLeftNegative) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
        XlsFixed<24, 24, true> ax = -2;
      return (XlsFixed<24, 24, false>(a) << ax).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 25, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftLeft) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      return (XlsFixed<24, 24, false>(a) << 1).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 200, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftLeftEq) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<24, 24, false> ax = a;
      ax <<= 2;
      return ax.to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 400, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftLeftEqXlsFixed) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<8, 8, true> sa = 3;
      XlsFixed<24, 24, false> ax = a;
      ax <<= sa;
      return ax.to_uint64();
    })";
  RunAcDatatypeTest({{"a", 100}}, 800, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ShiftLeftXlsFixedLarge) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<2, 2, false> sa = 3;
      return (XlsFixed<24, 24, false>(a) << sa).to_uint64();
    })";
  RunAcDatatypeTest({{"a", 5}}, 40, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, SetSlc) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<24, 24, false> ax = a;
      ax.set_slc(2, XlsFixed<24, 24, false>(0b11));
      return ax.to_int();
    })";
  RunAcDatatypeTest({{"a", 0b0001}}, 0b1101, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, Slc) {
  const std::string content = R"(
    #include "xls_fixed.h"

    int my_package(int a) {
      XlsFixed<24, 24, false> ax = a;
      return ax.slc<4>(2).to_int();
    })";
  RunAcDatatypeTest({{"a", 0b10101}}, 0b101, content,
                    xabsl::SourceLocation::current());
}

// Unroll for in default function
TEST_F(XlsFixedTest, DefaultArrayInit) {
  const std::string content = R"(
    #include "xls_fixed.h"
    struct Foo {
      XlsInt<8, false> data[16];
      Foo() = default;
      Foo(const XlsInt<128, false>& vector) {
        #pragma hls_unroll yes
        for(int i=0;i<16;++i) {
          data[i] = vector.slc<8>(8*i);
        }
      }
    };
    int my_package(int a) {
      XlsFixed<4 * 128, 4 * 128, false> bigval(a);
      Foo smallval;
      smallval = bigval.slc<128>(0);
      return (1 + smallval.data[0]).to_int();
    })";
  RunAcDatatypeTest({{"a", 100}}, 101, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, SaturateUnsigned) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsInt<7, false> ax = a;
      XlsFixed<8, 7, false> bx = ax;
      return bx.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, FixedDecimalMaintainsValue) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<3, 3, false> ax = 2;
      XlsFixed<8, 7, false> bx = a;
      bx /= ax;
      bx *= ax;
      return bx.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, SameDecimalWidth) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package(long long a) {
      XlsFixed<3, 3, false> bx = a;
      return bx.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, GreaterWidthShiftLeft) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<65, 64, false> bx = a;
      return bx.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, SmallerWidthShiftLeft) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<8, 7, false> bx = a;
      return bx.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}
TEST_F(XlsFixedTest, Underflow) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package(long long a) {
      XlsFixed<3, 3, false> ax = 2;
      XlsFixed<8, 7, false> bx = a;
      bx /= ax;
      bx *= ax;
      return bx.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, MinValueUnsigned) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package() {
      XlsInt<3, false> x(MinValue<3, false>::Value());
      return x.to_int();
    })";
  RunAcDatatypeTest({}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, MinValueSigned) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package() {
      XlsInt<3, true> x(MinValue<3, true>::Value());
      return x.to_int();
    })";
  RunAcDatatypeTest({}, -4, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, MaxValueUnsigned) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package() {
      XlsInt<3, false> x(MaxValue<3, false>::Value());
      return x.to_int();
    })";
  RunAcDatatypeTest({}, 7, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, MaxValueSigned) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package() {
      XlsInt<3, true> x(MaxValue<3, true>::Value());
      return x.to_int();
    })";
  RunAcDatatypeTest({}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, QuantizationModeOtherThanAcTrnNotSupported) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package() {
      XlsFixed<10, 10, true, ac_datatypes::AC_RND, ac_datatypes::AC_WRAP> x(0);
      return x.to_int();
    })";
  // AC_RND, AC_TRN_ZERO, AC_RND_ZERO, AC_RND_INF, AC_RND_MIN_INF, AC_RND_CONV,
  // AC_RND_CONV_ODD
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<std::string> clang_args,
                           XlsccTestBase::GetClangArgForIntTest());
  std::vector<std::string_view> clang_argv(clang_args.begin(),
                                           clang_args.end());
  ASSERT_THAT(SourceToIr(content, nullptr, clang_argv).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("Unable to parse text with clang")));
}

TEST_F(XlsFixedTest, QuantizationModeAcTrnSupported) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package() {
      XlsFixed<10, 10, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP> x(0);
      return x.to_int();
    })";
  RunAcDatatypeTest({}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsFixedTest, ToAcInt) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package(int a) {
      XlsFixed<10, 6, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP> x(a);
      return x.to_ac_int();
    })";
  RunAcDatatypeTest({{"a", 9}}, 9, content, xabsl::SourceLocation::current());
}

}  // namespace

}  // namespace xlscc
