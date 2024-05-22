// Copyright 2021 The XLS Authors
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

#include <cstdint>
#include <string>

#include "gtest/gtest.h"
#include "xls/common/source_location.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"

namespace xlscc {
namespace {

class XlsIntTest : public XlsccTestBase {};

TEST_F(XlsIntTest, Add) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax + bx;
    })";
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, 8, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Sub) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax - bx;
    })";
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, -2, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Mul) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax * bx;
    })";
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, 15, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Div) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax / bx;
    })";
  RunAcDatatypeTest({{"a", 32}, {"b", 4}}, 8, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Mod) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax % bx;
    })";
  RunAcDatatypeTest({{"a", 15}, {"b", 4}}, 3, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, BitwiseOr) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax | bx;
    })";
  RunAcDatatypeTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0011, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, BitwiseAnd) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax & bx;
    })";
  RunAcDatatypeTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0010, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, BitwiseXor) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax ^ bx;
    })";
  RunAcDatatypeTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0001, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, GreaterThan) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax > bx;
    })";
  RunAcDatatypeTest({{"a", 100}, {"b", 10}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, GreaterThanOrEq) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax >= bx;
    })";
  RunAcDatatypeTest({{"a", 100}, {"b", 100}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, LessThan) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax < bx;
    })";
  RunAcDatatypeTest({{"a", 10}, {"b", 100}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, LessThanOrEq) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax <= bx;
    })";
  RunAcDatatypeTest({{"a", 100}, {"b", 100}}, 1, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, WidthAndSignMembers) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(unsigned long long a) {
      const XlsInt<32, true> ax = a;
      return ax.width + ax.sign;
    })";
  RunAcDatatypeTest({{"a", 0}}, 33, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ConstBitRef) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(unsigned long long a) {
      const XlsInt<32, true> ax = a;
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

TEST_F(XlsIntTest, SetBitRef) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(unsigned long long a) {
      XlsInt<32, true> ax = a;
      ax[5] = 1;
      ax[0] = 0;
      return ax;
    })";
  RunAcDatatypeTest({{"a", 0b000001}}, 0b100000, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, PreInc) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      ++ax;
      return ++ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 102, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, PostInc) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      ax++;
      return ax++;
    })";
  RunAcDatatypeTest({{"a", 100}}, 101, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, PreDec) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      --ax;
      return --ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 98, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, PostDec) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      ax--;
      return ax--;
    })";
  RunAcDatatypeTest({{"a", 100}}, 99, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ArithNeg) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      return -ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, -100, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, LogicNeg) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(unsigned long long a) {
      XlsInt<8, false> ax = a;
      // operator ~ always returns signed, sign extends
      return ~ax;
    })";
  RunAcDatatypeTest({{"a", 3}}, -4, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, LogicNegate) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(unsigned long long a) {
      XlsInt<8, false> ax = a;
      return ax.bit_complement();
    })";
  RunAcDatatypeTest({{"a", 3}}, 252U, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Clz) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(unsigned long long a) {
      XlsInt<8, false> ax = a;
      return ax.clz();
    })";
  RunAcDatatypeTest({{"a", 0b1000}}, 4, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0b10000000}}, 0, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0}}, 8, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 1}}, 7, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, LeadingSign) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(int signed_mode, unsigned long long a) {
      if(signed_mode) {
        XlsInt<16, true> ax = a;
        return ax.leading_sign();
      } else {
        XlsInt<16, false> ax = a;
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

TEST_F(XlsIntTest, LeadingSignAllSign) {
  const std::string content = R"(
    #include "xls_int.h"

    bool my_package(int signed_mode, unsigned long long a) {
      bool all_sign;
      if(signed_mode) {
        XlsInt<16, true> ax = a;
        ax.leading_sign(all_sign);
      } else {
        XlsInt<16, false> ax = a;
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

TEST_F(XlsIntTest, Not) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(unsigned long long a) {
      XlsInt<5, false> ax = a;
      return !ax;
    })";
  RunAcDatatypeTest({{"a", 5}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Equal) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = 3;
      ax = XlsInt<30, false>(a);
      return ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 100, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, EqualTo) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      return ax == 5;
    })";
  RunAcDatatypeTest({{"a", 100}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, NotEqual) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      return ax != 5;
    })";
  RunAcDatatypeTest({{"a", 100}}, 1, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRight) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      return XlsInt<24, false>(a) >> 1;
    })";
  RunAcDatatypeTest({{"a", 100}}, 50, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRightEq) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      ax >>= 2;
      return ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 25, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRightXlsInt) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<4, false> sa = 1;
      return XlsInt<24, false>(a) >> sa;
    })";
  RunAcDatatypeTest({{"a", 100}}, 50, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRightXlsIntLarge) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<2, false> sa = 3;
      return XlsInt<24, false>(a) >> sa;
    })";
  RunAcDatatypeTest({{"a", 80}}, 10, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRightEqXlsInt) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<4, false> sa = 2;
      XlsInt<24, false> ax = a;
      ax >>= sa;
      return ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 25, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRightNegative) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      return XlsInt<24, false>(a) >> -1;
    })";
  RunAcDatatypeTest({{"a", 100}}, 200, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftLeftNegative) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
        XlsInt<24, true> ax = -2;
      return XlsInt<24, false>(a) << ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 25, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftLeft) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      return XlsInt<24, false>(a) << 1;
    })";
  RunAcDatatypeTest({{"a", 100}}, 200, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftLeftEq) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      ax <<= 2;
      return ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 400, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftLeftEqXlsInt) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<8, true> sa = 3;
      XlsInt<24, false> ax = a;
      ax <<= sa;
      return ax;
    })";
  RunAcDatatypeTest({{"a", 100}}, 800, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftLeftXlsIntLarge) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<2, false> sa = 3;
      return XlsInt<24, false>(a) << sa;
    })";
  RunAcDatatypeTest({{"a", 5}}, 40, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, SetSlc) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      ax.set_slc(2, XlsInt<24, false>(0b11));
      return ax;
    })";
  RunAcDatatypeTest({{"a", 0b0001}}, 0b1101, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Slc) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      return ax.slc<4>(2);
    })";
  RunAcDatatypeTest({{"a", 0b10101}}, 0b101, content,
                    xabsl::SourceLocation::current());
}

// Unroll for in default function
TEST_F(XlsIntTest, DefaultArrayInit) {
  const std::string content = R"(
    #include "xls_int.h"
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
      XlsInt<4 * 128, false> bigval(a);
      Foo smallval;
      smallval = bigval.slc<128>(0);
      return 1 + smallval.data[0];
    })";
  RunAcDatatypeTest({{"a", 100}}, 101, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ToUnsigned) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned int my_package(long long a) {
      XlsInt<32, false> ax = a;
      return ax.to_uint();
    })";
  RunAcDatatypeTest({{"a", -1}}, 0xFFFFFFFF, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 500}}, 500, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, SaturateUnsigned) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<8, 8, false, ac_datatypes::AC_TRN, ac_datatypes::AC_SAT> ax = a;
      return ax.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -5}}, 0, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 500}}, 255, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, SaturateSigned) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<8, 8, true, ac_datatypes::AC_TRN, ac_datatypes::AC_SAT> ax = a;
      return ax.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -5}}, -5, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 500}}, 127, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -600}}, -128, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, SaturateXlsInt) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsInt<12, true> in = a;
      XlsFixed<8, 8, false, ac_datatypes::AC_TRN, ac_datatypes::AC_SAT> ax = in;
      return ax.to_int();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -5}}, 0, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 500}}, 255, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, SaturateSignedLarge) {
  const std::string content = R"(
    #include "xls_fixed.h"

    long long my_package(long long a) {
      XlsFixed<33, 33, true, ac_datatypes::AC_TRN, ac_datatypes::AC_SAT> ax = a;
      return ax.to_int64();
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -5}}, -5, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 17179869184L}}, 4294967295L, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -17179869184L}}, -4294967296L, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, IntTernaryAssign) {
  const std::string content = R"(
       #include "xls_int.h"

       long long my_package(long long a, long long b) {
        const XlsInt<7, false> valA = 20, valB = 50;
        XlsInt<7, false> tmp;
        tmp = b ? valA : valB;
        return tmp;
       })";
  RunAcDatatypeTest({{"a", 1}, {"b", 20}}, 20, content);
}

TEST_F(XlsIntTest, SignedBinaryOpsHaveProperResultingSign) {
  const std::string content = R"(
       #include "xls_int.h"

       long long my_package() {
        XlsInt<4, true> ax = -1;
        const XlsInt<4, false> bx = 4;
        ax = ax / bx;
        return ax;
       })";
  RunAcDatatypeTest({}, 0, content);
}

TEST_F(XlsIntTest, ACCompatibleModuloSign) {
  const std::string content = R"(
       #include "xls_int.h"

       long long my_package() {
        XlsInt<4, false> ax = 10;
        const XlsInt<4, true> bx = -2;
        auto result = ax % bx;
        return result.sign ? 1 : 0;
       })";
  RunAcDatatypeTest({}, 0, content);
}

TEST_F(XlsIntTest, ACCompatibleDivisionWidth) {
  const std::string content = R"(
       #include "xls_int.h"

       long long my_package() {
        XlsInt<4, false> ax = 10;
        const XlsInt<4, false> bx = 4;
        auto result = ax / bx;
        return result.width;
       })";
  RunAcDatatypeTest({}, 4, content);
}

TEST_F(XlsIntTest, Reverse) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package() {
      XlsInt<16, false> x = 0b1000110100001010;
      return x.reverse();
    })";
  RunAcDatatypeTest({}, 0b0101000010110001, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, DoubleConstructor) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package() {
      XlsInt<10, true> x(5.5);
      return x.to_int();
    })";
  RunAcDatatypeTest({}, 5, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, FloatConstructor) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package() {
      XlsInt<10, true> x(5.5f);
      return x.to_int();
    })";
  RunAcDatatypeTest({}, 5, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, BitElemRefCast) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a, long long b) {
      XlsInt<4, false> u4(2);
      u4[2] = (a > b) ? XlsInt<1, false>(0) : u4[1];
      return u4.to_int();
    })";
  RunAcDatatypeTest({{"a", 5}, {"b", 3}}, 2, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 3}, {"b", 5}}, 6, content,
                    xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, BitElemRefCastToXlsInt) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a) {
      XlsInt<1, false> result;
      XlsInt<4, false> u4(a);
      result = u4[1];
      return result.to_int();
    })";
  RunAcDatatypeTest({{"a", 5}}, 0, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 3}}, 1, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, XlsIntDirectAssignLongLong) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a) {
      XlsInt<11, false> result;
      result = a;
      return result.to_int();
    })";
  RunAcDatatypeTest({{"a", 5}}, 5, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, XlsIntUnaryPlus) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a) {
      XlsInt<11, true> result = a;
      return (+result).to_long();
    })";
  RunAcDatatypeTest({{"a", -5}}, -5, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ImplicitBool) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(unsigned long long a) {
      XlsInt<32, false> ai = a;
      bool aib = (bool)ai;
      if(aib) {
        return 3;
      } else {
        return 5;
      }
    })";
  RunAcDatatypeTest({{"a", 0}}, 5, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 1}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 2}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -1}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -2}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ImplicitBoolInIf) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(unsigned long long a) {
      XlsInt<32, false> ai = a;
      if(ai) {
        return 3;
      } else {
        return 5;
      }
    })";
  RunAcDatatypeTest({{"a", 0}}, 5, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 1}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 2}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -1}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -2}}, 3, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", -3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, XlsLessThanBounds) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a) {
      XlsInt<3, false> result = a;
      XlsInt<3, true> input = -1;
      return result < input;
    })";
  RunAcDatatypeTest({{"a", 2}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, XlsLessThanEqualBounds) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a) {
    XlsInt<2, false> result = 0;
    XlsInt<10, true> input = a;
    return result <= input;
    })";
  RunAcDatatypeTest({{"a", -4}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, XlsGreaterThanBounds) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a) {
      XlsInt<2, false> result = a;
      XlsInt<3, false> input =  7;
      return result >= input;
    })";
  RunAcDatatypeTest({{"a", 3}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, XlsToLongBounds) {
  const std::string content = R"(
    #include "xls_int.h"
    long long my_package(long long a) {
      XlsInt<2, false> result = a;
      return result.to_long();
  })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}
}  // namespace

}  // namespace xlscc
