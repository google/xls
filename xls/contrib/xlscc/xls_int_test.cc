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

#include <cstdio>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/unit_test.h"

namespace xlscc {
namespace {

class XlsIntTest : public XlsccTestBase {
 public:
  void RunIntTest(
      const absl::flat_hash_map<std::string, uint64_t>& args, uint64_t expected,
      absl::string_view cpp_source,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string ac_int_path,
        xls::GetXlsRunfilePath("ac_datatypes/include/ac_int.h"));
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string xls_int_path,
        xls::GetXlsRunfilePath("xls/contrib/xlscc/synth_only/xls_int.h"));

    // Get the path that includes the ac_datatypes folder, so that the
    //  ac_datatypes headers can be included with the form:
    // #include "ac_datatypes/include/foo.h"
    std::string ac_int_dir = std::filesystem::path(ac_int_path)
                                 .parent_path()
                                 .parent_path()
                                 .parent_path();
    std::string xls_int_dir = std::filesystem::path(xls_int_path).parent_path();

    std::string ac_include = std::string("-I") + ac_int_dir.data();
    std::string xls_include = std::string("-I") + xls_int_dir.data();

    std::vector<absl::string_view> argv;
    argv.push_back(xls_include);
    argv.push_back(ac_include);
    argv.push_back("-D__SYNTHESIS__");

    Run(args, expected, cpp_source, loc, argv);
  }
};

TEST_F(XlsIntTest, Add) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a, long long b) {
      XlsInt<32, true> ax = a;
      XlsInt<32, true> bx = b;
      return ax + bx;
    })";
  RunIntTest({{"a", 3}, {"b", 5}}, 8, content,
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
  RunIntTest({{"a", 3}, {"b", 5}}, -2, content,
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
  RunIntTest({{"a", 3}, {"b", 5}}, 15, content,
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
  RunIntTest({{"a", 32}, {"b", 4}}, 8, content,
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
  RunIntTest({{"a", 15}, {"b", 4}}, 3, content,
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
  RunIntTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0011, content,
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
  RunIntTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0010, content,
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
  RunIntTest({{"a", 0b0011}, {"b", 0b0010}}, 0b0001, content,
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
  RunIntTest({{"a", 100}, {"b", 10}}, 1, content,
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
  RunIntTest({{"a", 100}, {"b", 100}}, 1, content,
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
  RunIntTest({{"a", 10}, {"b", 100}}, 1, content,
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
  RunIntTest({{"a", 100}, {"b", 100}}, 1, content,
             xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, WidthAndSignMembers) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(unsigned long long a) {
      const XlsInt<32, true> ax = a;
      return ax.width + ax.sign;
    })";
  RunIntTest({{"a", 0}}, 33, content, xabsl::SourceLocation::current());
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
  RunIntTest({{"a", 0b101011}}, 4, content, xabsl::SourceLocation::current());
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
  RunIntTest({{"a", 0b000001}}, 0b100000, content,
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
  RunIntTest({{"a", 100}}, 102, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, PostInc) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      ax++;
      return ax++;
    })";
  RunIntTest({{"a", 100}}, 101, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, PreDec) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      --ax;
      return --ax;
    })";
  RunIntTest({{"a", 100}}, 98, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, PostDec) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      ax--;
      return ax--;
    })";
  RunIntTest({{"a", 100}}, 99, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ArithNeg) {
  const std::string content = R"(
    #include "xls_int.h"

    long long my_package(long long a) {
      XlsInt<24, true> ax = a;
      return -ax;
    })";
  RunIntTest({{"a", 100}}, -100, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, LogicNeg) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(unsigned long long a) {
      XlsInt<5, false> ax = a;
      // operator ~ always returns signed
      return XlsInt<8, false>(~ax);
    })";
  RunIntTest({{"a", 0b11001}}, 230, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Not) {
  const std::string content = R"(
    #include "xls_int.h"

    unsigned long long my_package(unsigned long long a) {
      XlsInt<5, false> ax = a;
      return !ax;
    })";
  RunIntTest({{"a", 5}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Equal) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = 3;
      ax = XlsInt<30, false>(a);
      return ax;
    })";
  RunIntTest({{"a", 100}}, 100, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, EqualTo) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      return ax == 5;
    })";
  RunIntTest({{"a", 100}}, 0, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, NotEqual) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      return ax != 5;
    })";
  RunIntTest({{"a", 100}}, 1, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRight) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      return XlsInt<24, false>(a) >> 1;
    })";
  RunIntTest({{"a", 100}}, 50, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftRightEq) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      ax >>= 2;
      return ax;
    })";
  RunIntTest({{"a", 100}}, 25, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftLeft) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      return XlsInt<24, false>(a) << 1;
    })";
  RunIntTest({{"a", 100}}, 200, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, ShiftLeftEq) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      ax <<= 2;
      return ax;
    })";
  RunIntTest({{"a", 100}}, 400, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, SetSlc) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      ax.set_slc(2, XlsInt<24, false>(0b11));
      return ax;
    })";
  RunIntTest({{"a", 0b0001}}, 0b1101, content,
             xabsl::SourceLocation::current());
}

TEST_F(XlsIntTest, Slc) {
  const std::string content = R"(
    #include "xls_int.h"

    int my_package(int a) {
      XlsInt<24, false> ax = a;
      return ax.slc<4>(2);
    })";
  RunIntTest({{"a", 0b10101}}, 0b101, content,
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
  RunIntTest({{"a", 100}}, 101, content, xabsl::SourceLocation::current());
}

}  // namespace

}  // namespace xlscc
