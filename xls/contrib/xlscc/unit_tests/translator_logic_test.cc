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
#include <cstdio>
#include <list>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/fileno.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

// TODO(seanhaskell): Reimplement unsequenced assignment detection
#define UNSEQUENCED_TESTS 0

namespace xlscc {
namespace {

using ::absl_testing::IsOkAndHolds;

class TranslatorLogicTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorLogicTest, IntConst) {
  std::string_view content = R"(
    int my_package(int a) {
      return 123;
    })";
  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorLogicTest, LongConst) {
  std::string_view content = R"(
      int my_package(int a) {
        return 123L;
      })";

  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorLogicTest, LongLongConst) {
  std::string_view content = R"(
      long long my_package(long long a) {
        return 123L;
      })";

  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorLogicTest, LongLongTrueConst) {
  std::string_view content = R"(
      long long my_package(long long a) {
        return 123LL;
      })";

  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorLogicTest, SyntaxError) {
  std::string_view content = R"(
      int my_package(int a) {
        return a+
      })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             testing::HasSubstr("expected expression")));
}

TEST_F(TranslatorLogicTest, Assignment) {
  {
    std::string_view content = R"(
        int my_package(int a) {
          a = 5;
          return a;
        })";

    Run({{"a", 1000}}, 5, content);
  }
  {
    std::string_view content = R"(
        int my_package(int a) {
          a = 5;
          return a = 10;
        })";

    Run({{"a", 1000}}, 10, content);
  }
}

TEST_F(TranslatorLogicTest, ChainedAssignment) {
  std::string_view content = R"(
      int my_package(int a) {
        a += 5;
        a += 10;
        return a;
      })";

  Run({{"a", 1000}}, 1015, content);
}

TEST_F(TranslatorLogicTest, UnsignedChar) {
  std::string_view content = R"(
      unsigned char my_package(unsigned char a) {
        return a+5;
      })";

  Run({{"a", 100}}, 105, content);
}

TEST_F(TranslatorLogicTest, SignedChar) {
  std::string_view content = R"(
      bool my_package(signed char a) {
        return a < 1;
      })";

  Run({{"a", 0xff}}, static_cast<int64_t>(true), content);
  Run({{"a", 2}}, static_cast<int64_t>(false), content);
}

TEST_F(TranslatorLogicTest, Bool) {
  std::string_view content = R"(
      int my_package(long long a) {
        return bool(a);
      })";

  Run({{"a", 1000}}, 1, content);
  Run({{"a", 0}}, 0, content);
  Run({{"a", -1}}, 1, content);
}

TEST_F(TranslatorLogicTest, DeclGroup) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        long long aa=a, bb=b;
        return aa+bb;
      })";

  Run({{"a", 10}, {"b", 20}}, 30, content);
}

TEST_F(TranslatorLogicTest, Short) {
  std::string_view content = R"(
      short my_package(short a, short b) {
        return a+b;
      })";

  Run({{"a", 100}, {"b", 200}}, 300, content);
}

TEST_F(TranslatorLogicTest, UShort) {
  std::string_view content = R"(
      unsigned short my_package(unsigned short a, unsigned short b) {
        return a+b;
      })";

  Run({{"a", 100}, {"b", 200}}, 300, content);
}

TEST_F(TranslatorLogicTest, Typedef) {
  std::string_view content = R"(
      typedef long long my_int;
      my_int my_package(my_int a) {
        return a*10;
      })";

  Run({{"a", 4}}, 40, content);
}

TEST_F(TranslatorLogicTest, IrAsm) {
  std::string_view content = R"(
      long long my_package(long long a) {
       int asm_out;
       asm (
           "fn (fid)(x: bits[i]) -> bits[r] { "
           "   ret op_(aid): bits[r] = bit_slice(x, start=s, width=r) }"
         : "=r" (asm_out)
         : "i" (64), "s" (1), "r" (32), "param0" (a));
       return asm_out;
      })";

  Run({{"a", 1000}}, 500, content);
}

TEST_F(TranslatorLogicTest, ArrayParam) {
  std::string_view content = R"(
       long long my_package(const long long arr[2]) {
         return arr[0]+arr[1];
       })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_src, SourceToIr(content));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> package,
                           ParsePackage(ir_src));
  XLS_ASSERT_OK(package->SetTopByName("my_package"));
  std::vector<uint64_t> in_vals = {55, 20};
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value in_arr,
                           xls::Value::UBitsArray(in_vals, 64));
  absl::flat_hash_map<std::string, xls::Value> args;
  args["arr"] = in_arr;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * entry, package->GetTopAsFunction());

  auto x = DropInterpreterEvents(xls::InterpretFunctionKwargs(
      entry, {{"arr", xls::Value::UBitsArray({55, 20}, 64).value()}}));

  ASSERT_THAT(x, IsOkAndHolds(xls::Value(xls::UBits(75, 64))));
}

TEST_F(TranslatorLogicTest, ArraySet) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         long long arr[4];
         arr[0] = a;
         arr[1] = b;
         return arr[0]+arr[1];
       })";

  Run({{"a", 11}, {"b", 50}}, 61, content);
}

TEST_F(TranslatorLogicTest, IncrementInArrayIndex1) {
  std::string_view content = R"(
       long long my_package(long long a) {
         int arr[4];
         arr[a++] = 5;
         return a;
       })";

  Run({{"a", 11}}, 12, content);
}

TEST_F(TranslatorLogicTest, IncrementInArrayIndex2) {
  std::string_view content = R"(
       struct Blah {
         int operator=(int x) {
           return 0;
         }
       };
       long long my_package(long long a) {
         Blah arr[4];
         arr[a++] = 5;
         return a;
       })";

  Run({{"a", 11}}, 12, content);
}

TEST_F(TranslatorLogicTest, IncrementInArrayIndex3) {
  std::string_view content = R"(
       long long my_package(long long a) {
         int arr[4];
         arr[a] = arr[a++];
         return a;
       })";

  Run({{"a", 11}}, 12, content);
}

TEST_F(TranslatorLogicTest, ThisExprWithCallByReference) {
  std::string_view content = R"(
    struct Blah {
      int val() {
        return 2;
      }
    };

    Blah byref(bool& a1) {
      a1 = true;
      return Blah();
    }

    int my_package(long long a) {
      bool b = false;
      int ls = byref(b).val();
      return a + (b ? ls : -1);
    })";
  Run({{"a", 100}}, 102, content);
}

TEST_F(TranslatorLogicTest, ThisExprWithCallByReference2) {
  std::string_view content = R"(
    struct Blah {
      operator int() {
        return 2;
      }
    };

    Blah byref(bool& a1) {
      a1 = true;
      return Blah();
    }

    int my_package(long long a) {
      bool b = false;
      int ls = byref(b);
      return a + (b ? ls : -1);
    })";
  Run({{"a", 100}}, 102, content);
}

TEST_F(TranslatorLogicTest, ThisExprWithCallByReference3) {
  std::string_view content = R"(
    struct Blah {
      Blah val() {
        return *this;
      }
      operator int() {
        return 2;
      }
    };

    Blah byref(bool& a1) {
      a1 = true;
      return Blah();
    }

    int my_package(long long a) {
      bool b = false;
      int ls = byref(b).val();
      return a + (b ? ls : -1);
    })";
  Run({{"a", 100}}, 102, content);
}

TEST_F(TranslatorLogicTest, ThisExprWithCallByReference4) {
  std::string_view content = R"(
    struct Blah {
      int val()const {
        return 2;
      }
    };

    Blah byref(bool& a1) {
      a1 = true;
      return Blah();
    }

    int my_package(long long a) {
      bool b = false;
      int ls = byref(b).val();
      return a + (b ? ls : -1);
    })";
  Run({{"a", 100}}, 102, content);
}

TEST_F(TranslatorLogicTest, ByReferenceSubExpr) {
  std::string_view content = R"(

    int byref(int offset, int& out) {
      out = out + offset;
      return offset;
    }

    int my_package(long long a) {
      int vals[3] = {1,2,3};
      int idx = 1;
      (void)byref(5, vals[idx++]);
      return vals[1] + idx;
    })";
  Run({{"a", 100}}, 2 + 5 + 2, content);
}

TEST_F(TranslatorLogicTest, ArrayParamAssign) {
  std::string_view content = R"(
    void addto(int v[6]) {
      v[3] += 3;
    }

    int my_package(bool configured) {
      (void)configured;
      int brr[6] = {10,20,30,40,50,60};
      addto(brr);
      return brr[3];
    })";

  Run({{"configured", 0}}, 40 + 3, content);
}

TEST_F(TranslatorLogicTest, DerefPointerToArrayAssign) {
  std::string_view content = R"(
    int my_package(bool configured) {
      (void)configured;
      int brr[6] = {10,20,30,40,50,60};
      int *pb = &brr[0];
      pb[0] = 1;
      return brr[3];
    })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("Only array subscript assignments "
                                     "directly to arrays supported")));
}

TEST_F(TranslatorLogicTest, ArrayPointerParam) {
  std::string_view content = R"(
    void addto(int* v) {
      v[3] += 3;
    }

    int my_package(bool configured) {
      (void)configured;
      int brr[6] = {10,20,30,40,50,60};
      addto(brr);
      return brr[3];
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Pointer function parameters unsupported")));
}

TEST_F(TranslatorLogicTest, Array2D) {
  std::string_view content = R"(
       int my_package(int a, int b) {
         int x[2][2] = {{b,b}, {b,b}};
         x[1][0] += a;
         return x[1][0];
       })";

  Run({{"a", 55}, {"b", 100}}, 155, content);
}

TEST_F(TranslatorLogicTest, Array2DParam) {
  std::string_view content = R"(
       int access_it(int x[2][2]) {
         return x[1][0];
       }
       int my_package(int a, int b) {
         int x[2][2] = {{b,b}, {b,b}};
         x[1][0] += a;
         return access_it(x);
       })";

  Run({{"a", 55}, {"b", 100}}, 155, content);
}

TEST_F(TranslatorLogicTest, Array2DConstParam) {
  std::string_view content = R"(
       int access_it(const int x[2][2]) {
         return x[1][0];
       }
       int my_package(int a, int b) {
         int x[2][2] = {{b,b}, {b,b}};
         x[1][0] += a;
         return access_it(x);
       })";

  Run({{"a", 55}, {"b", 100}}, 155, content);
}

TEST_F(TranslatorLogicTest, Array2DInit) {
  std::string_view content = R"(
       struct ts {
         ts(int v) : x(v) { };
         operator int () const { return x; }
         ts operator += (int v) { x += v; return (*this); }
         int x;
       };
       int my_package(int a, int b) {
         int x[2][2] = {{b,b}, {b,b}};
         x[1][0] += a;
         return x[1][0];
       })";
  Run({{"a", 55}, {"b", 100}}, 155, content);
}

TEST_F(TranslatorLogicTest, Array2DClass) {
  std::string_view content = R"(
       struct ts {
         ts(int v) : x(v) { };
         operator int () const { return x; }
         ts operator += (int v) { x += v; return (*this); }
         int x;
       };
       int my_package(int a, int b) {
         ts x[2][2] = {{b,b}, {b,b}};
         x[1][0] += a;
         return x[1][0];
       })";
  Run({{"a", 55}, {"b", 100}}, 155, content);
}

TEST_F(TranslatorLogicTest, ArrayInitList) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         long long arr[2] = {10, 20};
         arr[0] += a;
         arr[1] += b;
         return arr[0]+arr[1];
       })";
  Run({{"a", 11}, {"b", 50}}, 91, content);
}

TEST_F(TranslatorLogicTest, ArrayRefParam) {
  std::string_view content = R"(
       void asd(int b[2]) {
         b[0] += 5;
       }
       int my_package(int a) {
         int arr[2] = {a, 3*a};
         asd(arr);
         return arr[0] + arr[1];
       })";
  Run({{"a", 11}}, 11 + 5 + 3 * 11, content);
}

TEST_F(TranslatorLogicTest, ArrayInitListWrongSize) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         long long arr[4] = {10, 20};
         return a;
       })";
  ASSERT_FALSE(SourceToIr(content).ok());
}

TEST_F(TranslatorLogicTest, ArrayTooManyInitListValues) {
  std::string_view content = R"(
       long long my_package() {
         long long arr[1] = {4, 5};
         return arr[0];
       })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("excess elements")));
}

TEST_F(TranslatorLogicTest, ArrayInitListMismatchedSizeMultipleZeros) {
  std::string_view content = R"(
       long long my_package() {
         long long arr[4] = {0, 0};
         return arr[3];
       })";
  XLS_ASSERT_OK(SourceToIr(content));
}

TEST_F(TranslatorLogicTest, ArrayInitListMismatchedSizeOneNonZeros) {
  std::string_view content = R"(
       long long my_package() {
         long long arr[4] = {9};
         return arr[3];
       })";
  XLS_ASSERT_OK(SourceToIr(content));
}

TEST_F(TranslatorLogicTest, ConstInitListExpr) {
  std::string_view content = R"(
    int my_package(int a) {
      const int test_arr[][6] = {
          {  10,  0,  0,  0,  0,  0 },
          {  a,  20,  0,  0,  0,  0 }
      };
      return test_arr[0][0] + test_arr[1][1] + test_arr[1][0];
    })";
  Run({{"a", 3}}, 33, content);
}

TEST_F(TranslatorLogicTest, InitListExpr) {
  std::string_view content = R"(
    int my_package(int a) {
      int test_arr[][6] = {
          {  10,  0,  0,  0,  0,  0 },
          {  a,  20,  0,  0,  0,  0 }
      };
      return test_arr[0][0] + test_arr[1][1] + test_arr[1][0];
    })";
  Run({{"a", 3}}, 33, content);
}

TEST_F(TranslatorLogicTest, ArrayInitLoop) {
  std::string_view content = R"(
       struct tss {
         tss() : ss(15) {}
         tss(const tss &o) : ss(o.ss) {}
         int ss;
       };
       struct ts { tss vv[4]; };
       long long my_package(long long a) {
         ts x;
         x.vv[0].ss = a;
         ts y = x;
         return y.vv[0].ss;
       })";
  Run({{"a", 110}}, 110, content);
}

TEST_F(TranslatorLogicTest, StringConstantArray) {
  std::string_view content = R"(
       long long my_package(long long a) {
         const char foo[] = "A";
         return a+foo[0];
       })";
  Run({{"a", 11}}, 11 + 'A', content);
}

TEST_F(TranslatorLogicTest, GlobalInt) {
  std::string_view content = R"(
       const int off = 60;
       int foo() {
         // Reference it from another function to test context management
         return off;
       }
       long long my_package(long long a) {
         // Reference it twice to test global value re-use
         long long ret = a+foo();
         // Check context pop
         {
           ret += off;
         }
         ret += off;
         return ret;
       })";
  Run({{"a", 11}}, 11 + 60 + 60 + 60, content);
}

TEST_F(TranslatorLogicTest, GlobalEnum) {
  std::string_view content = R"(
       enum BlahE {
         A=2,B,C
       };
       long long my_package(long long a) {
         return a+B+B;
       })";
  Run({{"a", 11}}, 11 + 3 + 3, content);
}

TEST_F(TranslatorLogicTest, MaximumEnum) {
  std::string_view content = R"(
       enum BlahE {
         A=2,
         B=0xFFFFFFFF
       };
       long long my_package(long long a) {
         return a+B+B;
       })";
  Run({{"a", 11}}, 11L + 0xFFFFFFFFL + 0xFFFFFFFFL, content);
}

TEST_F(TranslatorLogicTest, CheckRepeatedGlobals) {
  std::string_view content = R"(
       enum Values {
        A=1000, B=10
       };
       long long my_package() {
        long long a = 0;
         a += A;
         a += B;
         a += A;
         return a;
       })";
  Run({}, 1000 + 10 + 1000, content);
}

TEST_F(TranslatorLogicTest, EnumVarDecl) {
  std::string_view content = R"(
       long long my_package() {
          enum BlahE {
            A=2,B,C
          } decl = A;
          return decl;
       })";
  Run({}, 2, content);
}

TEST_F(TranslatorLogicTest, SetGlobal) {
  std::string_view content = R"(
       int off = 60;
       long long my_package(long long a) {
         off = 5;
         long long ret = a;
         ret += off;
         return ret;
       })";
  ASSERT_FALSE(SourceToIr(content).ok());
}

TEST_F(TranslatorLogicTest, UnsequencedAssign) {
  std::string_view content = R"(
      int my_package(int a) {
        return (a=7)+a;
      })";
  auto ret = SourceToIr(content);

  // Clang catches this one and fails parsing
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("unsequenced modification and access")));
}

TEST_F(TranslatorLogicTest, TestXlsccCheck) {
  std::string_view content = R"(
      int my_package(int a) {
        return a+a;
      })";

  EXPECT_DEATH(
      {
        auto ret = ScanFile(
            content, {}, /*io_test_mode=*/false,
            /*error_on_init_interval=*/false,
            /*error_on_uninitialized=*/false,
            xls::SourceLocation(xls::Fileno(0), xls::Lineno(1), xls::Colno(1)),
            true);
      },
      testing::HasSubstr("0,1,1"));
}

#if UNSEQUENCED_TESTS

TEST_F(TranslatorLogicTest, UnsequencedRefParam) {
  std::string_view content = R"(
      int make7(int &a) {
        return a=7;
      }
      int my_package(int a) {
        return make7(a)+a;
      })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("unsequenced")));
}
TEST_F(TranslatorLogicTest, UnsequencedRefParam2) {
  std::string_view content = R"(
      int make7(int &a) {
        return a=7;
      }
      int my_package(int a) {
        return a+make7(a);
      })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("unsequenced")));
}

TEST_F(TranslatorLogicTest, UnsequencedRefParam3) {
  std::string_view content = R"(
      int make7(int &a) {
        return a=7;
      }
      int my_package(int a) {
        return make7(a)+a;
      })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("unsequenced")));
}

TEST_F(TranslatorLogicTest, UnsequencedRefParam4) {
  std::string_view content = R"(
      int my_package(int a) {
        return (a=7)?a:11;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("unsequenced")));
}
TEST_F(TranslatorLogicTest, UnsequencedRefParam5) {
  std::string_view content = R"(
      int my_package(int a) {
        return a?a:(a=7);
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("unsequenced")));
}

// Okay with one parameter
TEST_F(TranslatorLogicTest, AvoidUnsequencedRefParamUnary) {
  std::string_view content = R"(
      long long nop(long long a) {
        return a;
      }
      long long my_package(long long a) {
        return -nop(a=10);
      };)";

  Run({{"a", 100}}, -10, content);
}

TEST_F(TranslatorLogicTest, UnsequencedRefParamBinary) {
  std::string_view content = R"(
      int nop(int a, int b) {
        return a;
      }
      int my_package(int a) {
        return -nop(a=10, 100);
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("unsequenced")));
}

#endif  // UNSEQUENCED_TESTS

TEST_F(TranslatorLogicTest, OpAssignmentResult) {
  std::string_view content = R"(
      int my_package(int a) {
        return a+=5;
      })";

  Run({{"a", 100}}, 105, content);
}

TEST_F(TranslatorLogicTest, UndefinedConditionalAssign) {
  std::string_view content = R"(
      int my_package(int a) {
        int ret;
        if(a) {
          ret = 11;
        }
        return ret;
      })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("uninitialized")));
}

TEST_F(TranslatorLogicTest, IfStmt) {
  std::string_view content = R"(
      long long my_package(long long a) {
        if(a<-100) a = 1;
        else if(a<-10) a += 3;
        else { a *= 2; }
        return a;
      })";

  Run({{"a", 60}}, 120, content);
  Run({{"a", -50}}, -47, content);
  Run({{"a", -150}}, 1, content);
}

TEST_F(TranslatorLogicTest, IfAssignOverrideCondition) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        if(a>1000) {
          if(b)
            a=55;
          a=1234;
        }
        return a;
      })";

  Run({{"a", 60}, {"b", 0}}, 60, content);
  Run({{"a", 1001}, {"b", 0}}, 1234, content);
  Run({{"a", 1001}, {"b", 1}}, 1234, content);
}

TEST_F(TranslatorLogicTest, SwitchStmt) {
  std::string_view content = R"(
       long long my_package(long long a) {
         long long ret;
         switch(a) {
           case 1:
             ret = 100;
             break;
           case 2:
             ret = 200;
             break;
           default:
             ret = 300;
             break;
         }
         return ret;
       })";

  Run({{"a", 1}}, 100, content);
  Run({{"a", 2}}, 200, content);
  Run({{"a", 3}}, 300, content);
}

TEST_F(TranslatorLogicTest, SwitchStmtWithUnrollInCase) {
  std::string_view content = R"(
       long long my_package(long long a) {
         long long ret;
         switch(a) {
           case 1:
             #pragma hls_unroll yes
             for (int i = 0; i < 4; ++i) {
               #pragma hls_unroll yes
               for (int j = 0; j < 3; ++j) {
                 ret = 100;
               }
             }
             break;
           case 2:
             ret = 200;
             break;
           default:
             ret = 300;
             break;
         }
         return ret;
       })";

  Run({{"a", 1}}, 100, content);
  Run({{"a", 2}}, 200, content);
  Run({{"a", 3}}, 300, content);
}

TEST_F(TranslatorLogicTest, SwitchConditionalBreakReturn) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         long long ret;
         switch(a) {
           case 1:
             ret = 100;
             break;
           case 2:
             ret = 200;
             if(b) return 55;
           default:
             return 300;
         }
         return ret;
       })";

  Run({{"a", 1}, {"b", 0}}, 100, content);
  Run({{"a", 2}, {"b", 0}}, 300, content);
  Run({{"a", 2}, {"b", 1}}, 55, content);
  Run({{"a", 3}, {"b", 0}}, 300, content);
}

TEST_F(TranslatorLogicTest, SwitchConditionalBreak) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         long long ret;
         switch(a) {
           case 1:
             ret = 100;
             break;
           case 2:
             ret = 200;
             if(b) break;
           default:
             ret = 300;
             break;
         }
         return ret;
       })";

  Run({{"a", 1}, {"b", 0}}, 100, content);
  Run({{"a", 2}, {"b", 0}}, 300, content);
  Run({{"a", 2}, {"b", 1}}, 200, content);
  Run({{"a", 3}, {"b", 0}}, 300, content);
}

TEST_F(TranslatorLogicTest, SwitchStmtDefaultTop) {
  std::string_view content = R"(
       long long my_package(long long a) {
         long long ret;
         switch(a) {
           default:
             ret = 300;
             break;
           case 1: {
             ret = 100;
             break;
           } case 2:
             ret = 200;
             break;
         }
         return ret;
       })";

  Run({{"a", 1}}, 100, content);
  Run({{"a", 2}}, 200, content);
  Run({{"a", 3}}, 300, content);
}

TEST_F(TranslatorLogicTest, SwitchMultiCaseMultiLine) {
  std::string_view content = R"(
       long long my_package(long long a) {
         long long ret=0;
         switch(a) {
           case 1:
             ret += 300;
             ret += 2;
           case 2:
             ret += 5;
             ret += 100;
             break;
         }
         return ret;
       })";

  Run({{"a", 1}}, 407, content);
  Run({{"a", 2}}, 105, content);
  Run({{"a", 3}}, 0, content);
}

TEST_F(TranslatorLogicTest, SwitchMultiCaseMultiLineBrace) {
  std::string_view content = R"(
       long long my_package(long long a) {
         long long ret=0;
         switch(a) {
           case 1:
             ret += 300;
             ret += 2;
           case 2: {
             ret += 5;
             ret += 100;
             break;
           }
         }
         return ret;
       })";

  Run({{"a", 1}}, 407, content);
  Run({{"a", 2}}, 105, content);
  Run({{"a", 3}}, 0, content);
}

TEST_F(TranslatorLogicTest, SwitchDoubleBreak) {
  std::string_view content = R"(
       long long my_package(long long a) {
         long long ret=0;
         switch(a) {
           case 1:
             ret += 300;
             ret += 2;
             break;
             break;
           case 2: {
             ret += 5;
             ret += 100;
             break;
             break;
           }
         }
         return ret;
       })";

  Run({{"a", 1}}, 302, content);
  Run({{"a", 2}}, 105, content);
  Run({{"a", 3}}, 0, content);
}

TEST_F(TranslatorLogicTest, SwitchMultiCase) {
  std::string_view content = R"(
       long long my_package(long long a) {
         long long ret=0;
         switch(a) {
           case 1:
             ret += 300;
           case 2:
             ret += 100;
             break;
         }
         return ret;
       })";

  Run({{"a", 1}}, 400, content);
  Run({{"a", 2}}, 100, content);
  Run({{"a", 3}}, 0, content);
}

TEST_F(TranslatorLogicTest, SwitchReturnStmt) {
  std::string_view content = R"(
       long long my_package(long long a) {
         switch(a) {
           case 1:
             return 100;
           case 2:
             return 200;
           default:
             return 300;
         }
       })";

  Run({{"a", 1}}, 100, content);
  Run({{"a", 2}}, 200, content);
  Run({{"a", 3}}, 300, content);
}

TEST_F(TranslatorLogicTest, SwitchDeepFlatten) {
  std::string_view content = R"(
       long long my_package(long long a) {
         switch(a) {
           case 1:
           case 2:
           default:
             return 300;
         }
       })";

  Run({{"a", 1}}, 300, content);
  Run({{"a", 2}}, 300, content);
  Run({{"a", 3}}, 300, content);
}

TEST_F(TranslatorLogicTest, SwitchReturnStmt2) {
  std::string_view content = R"(
       long long my_package(long long a) {
         switch(a) {
           case 1:
             return 100;
           case 2:
             a+=10;
             break;
         }
         return a;
       })";

  Run({{"a", 1}}, 100, content);
  Run({{"a", 2}}, 12, content);
  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, SwitchDefaultPlusCase) {
  std::string_view content = R"(
       long long my_package(long long a) {
         switch(a) {
           default:
           case 1:
             return 100;
           case 2:
             a+=10;
             break;
         }
         return a;
       })";

  Run({{"a", 1}}, 100, content);
  Run({{"a", 2}}, 12, content);
  Run({{"a", 3}}, 100, content);
}

TEST_F(TranslatorLogicTest, SwitchInFor) {
  std::string_view content = R"(
       long long my_package(long long a) {
         #pragma hls_unroll yes
         for(int i=0;i<2;++i) {
           switch(i) {
             case 0:
               a += 300;
               break;
             case 1:
               a += 100;
               break;
           }
         }
         return a;
       })";

  Run({{"a", 1}}, 401, content);
}

TEST_F(TranslatorLogicTest, SwitchBreakAfterReturn) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         long long ret=0;
         switch(a) {
           case 1:
             if(b > 0) {return -1000;};
             ret += b;
             break;
         }
         return ret;
       })";

  Run({{"a", 5}, {"b", 1}}, 0, content);
  Run({{"a", 1}, {"b", 1}}, -1000, content);
  Run({{"a", 1}, {"b", -10}}, -10, content);
}

TEST_F(TranslatorLogicTest, ForInSwitch) {
  std::string_view content = R"(
       long long my_package(long long a) {
         switch(a) {
           case 0:
             #pragma hls_unroll yes
             for(int i=0;i<3;++i) {
               a+=10;
               if(a > 110) {
                 break;
               }
             }
             break;
           case 1:
             a += 100;
             break;
         }
         return a;
       })";

  Run({{"a", 0}}, 30, content);
  Run({{"a", 1}}, 101, content);
  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, SwitchDeclareInCase) {
  std::string_view content = R"(
      #pragma hls_top
      long long foo(long long in) {
        switch(in) {
          case 1: {
            int bar = in;
            in += bar;
            break;
          }
        }

        return in;
      })";

  Run({{"in", 1}}, 2, content);
}

TEST_F(TranslatorLogicTest, SwitchDeclareInCaseWithoutBraces) {
  std::string_view content = R"(
      #pragma hls_top
      long long foo(long long in) {
        switch(in) {
          case 1:
            int bar = in;
            in += bar;
            break;
        }

        return in;
      })";

  Run({{"in", 1}}, 2, content);
}

TEST_F(TranslatorLogicTest, ForUnroll) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, PragmaScoped) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        {
          #pragma hls_unroll yes
        }
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("expected statement")));
}

TEST_F(TranslatorLogicTest, ForUnrollInTemplateFunc) {
  std::string_view content = R"(
      template<typename T>
      T doit(T a, T b) {
        #pragma hls_unroll yes
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      }

      long long my_package(long long a, long long b) {
        return doit<long long>(a, b);
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, PragmaInDefineAppliesOnlyInDefine) {
  const std::string content = R"(
    #define some_macro(x) { \
          int i = 0;                 \
          _Pragma("hls_unroll yes")  \
          while (i < 2) {            \
            x[i] += 1;               \
            ++i;                     \
          }                          \
        }

    #pragma hls_top
    int bar(int (&a)[5], int b) {
      some_macro(a);
      some_macro(a);

      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("missing")));
}

TEST_F(TranslatorLogicTest, NestedLoopsNoBraces) {
  std::string_view content = R"(
      #define HLS_PRAGMA(x) _Pragma(#x)

      long long my_package(long long a, long long b) {
        HLS_PRAGMA(hls_unroll yes)
        for(int j=1;j<=2;++j)
          HLS_PRAGMA(hls_unroll yes)
          for(int i=1;i<=5;++i) a += 3*b;
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, NestedLoopsNoBraces2) {
  std::string_view content = R"(
      #define HLS_PRAGMA(x) _Pragma(#x)

      long long my_package(long long a, long long b) {
        if(a > 10)
          HLS_PRAGMA(hls_unroll yes)
          for(int i=1;i<=10;++i) a += 3*b;
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, ForUnrollLabel) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        label:
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBeforeLabelAndStatement) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        label:
        int x = 0;
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        (void)x;
        return a;
      })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("missing")));
}

TEST_F(TranslatorLogicTest, ForUnrollAssignAfterBreak) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        while(i<=10) {
          a += b;
          if(a > 60) {
            break;
          }
          ++i;
        }
        return (i*10) + a;
      })";
  Run({{"a", 11}, {"b", 20}}, 101, content);
}

TEST_F(TranslatorLogicTest, ForUnrollAssignAfterReturn) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        while(i<=10) {
          a += b;
          if(a > 60) {
            return (i*10) + a;
          }
          ++i;
        }
        return 1000;
      })";
  Run({{"a", 11}, {"b", 20}}, 101, content);
}

TEST_F(TranslatorLogicTest, ForUnrollNoIncrementOnBreak) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=0;
        #pragma hls_unroll yes
        for(;i<10;++i) {
          if(i==4) {
            break;
          }
        }
        return i;
      })";
  Run({{"a", 11}, {"b", 20}}, 4, content);
}

TEST_F(TranslatorLogicTest, WhileUnroll) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        while(i<=10) {
          a += b;
          a += 2*b;
          ++i;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, DoWhileUnroll) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        do {
          a += b;
          ++i;
        } while(i<=10 && a<10);
        return a;
      })";
  Run({{"a", 100}, {"b", 20}}, 120, content);
  Run({{"a", 3}, {"b", 2}}, 11, content);
  Run({{"a", 3}, {"b", 0}}, 3, content);
}

TEST_F(TranslatorLogicTest, WhileUnrollShortCircuit) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        while(i<=10) {
          a += b;
          ++i;
          if(a > 60) {
            break;
          }
        }
        return (i*10) + a;
      })";
  Run({{"a", 11}, {"b", 20}}, 111, content);
}

TEST_F(TranslatorLogicTest, WhileUnrollFalseCond) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        while(i<1) {
          a += b;
          a += 2*b;
          ++i;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 11, content);
}

TEST_F(TranslatorLogicTest, WhileUnrollFalseCond2) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        while(i<=10 && a<0) {
          a += b;
          a += 2*b;
          ++i;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 11, content);
}

TEST_F(TranslatorLogicTest, WhileUnrollFalseCond3) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        for(;i<=10 && a<0;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 11, content);
}

TEST_F(TranslatorLogicTest, ForUnrollShortCircuit) {
  std::string_view content = R"(

        long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<10 && a<=100;++i) {
           a += b;
          }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 111, content);
}

TEST_F(TranslatorLogicTest, ForUnrollShortCircuit2) {
  std::string_view content = R"(

        long long my_package(long long a, long long b) {
        int i=0;
         #pragma hls_unroll yes
         for(;i<10 && a<=100;++i) {
           a += b;
          }
         return a+i;
       })";
  Run({{"a", 11}, {"b", 20}}, 111 + 5, content);
}

// Check that continue doesn't skip loop increment
TEST_F(TranslatorLogicTest, ForUnrollShortCircuit2A) {
  std::string_view content = R"(
        long long my_package(long long a, long long b) {
         int i=0;
         #pragma hls_unroll yes
         for(;i<10;++i) {
           if(a>100) continue;
           a += b;
          }
         return a+i;
       })";
  Run({{"a", 11}, {"b", 20}}, 111 + 10, content);
}

TEST_F(TranslatorLogicTest, ForUnrollShortCircuit3) {
  std::string_view content = R"(
        template<int N>
        long sum(long in[N], int n) {
          long sum = 0;
          #pragma hls_unroll yes
          for (int i = 0; i < N && i < n; ++i) {
            sum += in[i];
          }
          return sum;
        }

        long long my_package(long long a, long long b) {
          long in[4] = {0,a,b,100};
          return sum<4>(in, 3);
       })";
  Run({{"a", 11}, {"b", 20}}, 31, content);
}

TEST_F(TranslatorLogicTest, ForUnrollShortCircuit4) {
  std::string_view content = R"(
         struct TestInt {
           TestInt(long long v) : x(v) { }
           int operator[](int i)const {
             return (x >> i)&1;
           }
           TestInt operator ++() {
             ++x;
             return *this;
           }
           bool operator <(int v)const {
             return x < v;
           }
           operator int() const {
             return x;
           }
           int x;
         };

        template <int W>
        long Ctz(TestInt in) {
          TestInt lz = 0;
          #pragma hls_unroll yes
          while (lz < W && in[lz] == 0) {
            ++lz;
          }
          return lz;
        }

        long long my_package(long long a) {
          return Ctz<8>(a);
       })";
  Run({{"a", 0b00001000}}, 3, content);
}

TEST_F(TranslatorLogicTest, ForUnrollShortCircuitClass) {
  std::string_view content = R"(
       struct TestInt {
         TestInt(int v) : x(v) { }
         operator int()const {
           return x;
         }
         TestInt operator ++() {
           ++x;
           return *this;
         }
         bool operator <=(int v)const {
           return x <= v;
         }
         int x;
       };
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(TestInt i=1;i<=10 && a<1000;++i) {
           a += b;
           a += 2*b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, ForUnrollShortCircuitClass2) {
  const std::string content = R"(
       struct TestInt {
         TestInt(int v) : x(v) { }
         operator int()const {
           return x;
         }
         int x;
       };
       long long my_package(long long a, long long b) {
         TestInt bounds[2] = {3,4};

         int x = 0;

         #pragma hls_unroll yes
         for(int i=0;i<bounds[a];++i) {
           x += b;
         }
         return x;
       })";
  Run({{"a", 1}, {"b", 20}}, 4L * 20L, content);
}

TEST_F(TranslatorLogicTest, ForPipelinedConstantPropagation) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      const short max_iters = 5;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        #pragma hls_unroll yes
        for(int j=0;j<max_iters;++j) {
          a += i;
        }
      }

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(80, 32)),
                  xls::Value(xls::SBits(100, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 5 * 10, 32)),
                      xls::Value(xls::SBits(100 + 5 * 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }
}

TEST_F(TranslatorLogicTest, ForUnrollMultiCondBreak) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<10;++i) {
          if(a<20) {
            break;
          }
          if(a>100) {
            break;
          }
          a += b;
        }
        return a;
      })";
  Run({{"a", 21}, {"b", 20}}, 101, content);
  Run({{"a", 11}, {"b", 20}}, 11, content);
}

TEST_F(TranslatorLogicTest, ForUnrollNestedCondBreak) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<10;++i) {
          if(a>50) {
            if(i<4) {
              break;
            }
          }
          a += b;
        }
        return a;
      })";
  Run({{"a", 51}, {"b", 20}}, 51, content);
  Run({{"a", 43}, {"b", 1}}, 52, content);
  Run({{"a", 0}, {"b", 3}}, 27, content);
}

TEST_F(TranslatorLogicTest, ForUnrollClass) {
  std::string_view content = R"(
       struct TestInt {
         TestInt(int v) : x(v) { }
         operator int()const {
           return x;
         }
         TestInt operator ++() {
           ++x;
           return *this;
         }
         bool operator <=(int v) {
           return x <= v;
         }
         int x;
       };
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(TestInt i=1;i<=10;++i) {
           a += b;
           a += 2*b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, ForUnrollConditionallyAssignLoopVar) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<10;++i) {
           a += b;
           if(a > 40) {
             ++i;
           }
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 131, content);
}

TEST_F(TranslatorLogicTest, ForUnrollNoInit) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        for(;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, ForUnrollNoInc) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<=10;) {
          a += b;
          a += 2*b;
          ++i;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, ForUnrollNoCond) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kResourceExhausted,
                                     testing::HasSubstr("maximum")));
}

TEST_F(TranslatorLogicTest, ForUnrollNoCondBreakInBody) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;;++i) {
          if(i>10) {
            break;
          }
          a += b;
          a += 2*b;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, ForUnrollNoPragma) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("missing #pragma or attribute")));
}

TEST_F(TranslatorLogicTest, ForUnrollBadNumber) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<=10;++i) {
          #pragma hls_unroll -4
          for(int j=0;j<4;++j) {
            int l = b;
            a += l;
          }
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  EXPECT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("must be an integer >= 0")));
}

TEST_F(TranslatorLogicTest, ForNestedUnroll) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<=10;++i) {
          #pragma hls_unroll yes
          for(int j=0;j<4;++j) {
            int l = b;
            a += l;
          }
        }
        return a;
      })";
  Run({{"a", 200}, {"b", 20}}, 1000, content);
}

TEST_F(TranslatorLogicTest, ForUnrollInfinite) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=1;i<=10;--i) {
           a += b;
           a += 2*b;
         }
         return a;
       })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kResourceExhausted,
                                     testing::HasSubstr("maximum")));
}

TEST_F(TranslatorLogicTest, ForUnrollBreak) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<9;++i) {
           if(a > 100) {
             break;
           }
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 111, content);
}

// Only one break condition is true, not all conditions after
TEST_F(TranslatorLogicTest, ForUnrollBreakOnEquals) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<9;++i) {
           if(a == 31) {
             break;
           }
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 31, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreakAfterAssignNested) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<4;++i) {
           a += b;
           if(a > 100) {
             if(b > 0) {
               break;
             }
           }
         }
         return a;
       })";
  Run({{"a", 101}, {"b", 20}}, 121, content);
}

TEST_F(TranslatorLogicTest, ForUnrollContinueNested) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<6;++i) {
           if(i == 1) {
             if(b > 0) {
               continue;
             }
           }
           a += b;
         }
         return a;
       })";
  Run({{"a", 101}, {"b", 20}}, 201, content);
  Run({{"a", 100}, {"b", 20}}, 200, content);
  Run({{"a", 10}, {"b", 20}}, 110, content);
  Run({{"a", 10}, {"b", -1}}, 4, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreakAfterAssign) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<3;++i) {
           a += b;
           if(a > 100) {
             break;
           }
         }
         return a;
       })";
  Run({{"a", 80}, {"b", 20}}, 120, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreakAfterAssign2) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<2;++i) {
           a += b;
           if(a > 100) {
             break;
           }
         }
         return a;
       })";
  Run({{"a", 101}, {"b", 20}}, 121, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreakTest) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<3;++i) {
           a += b;
           if(a > 100) {
             break;
           }
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 71, content);
  Run({{"a", 101}, {"b", 20}}, 121, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreak2) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<50;++i) {
           if(i==3) break;
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 71, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreak2Nested) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<50;++i) {
           if(a < 1000) {
             if(i==3) break;
           }
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 71, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreak3) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<50;++i) {
           a += b;
           if(i==3) break;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 91, content);
}

TEST_F(TranslatorLogicTest, ForUnrollBreak4) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<50;++i) {
           a += b;
           break;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 31, content);
}

TEST_F(TranslatorLogicTest, ForUnrollContinue) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           a += b;
           continue;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 231, content);
}

TEST_F(TranslatorLogicTest, ForUnrollContinue2) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           continue;
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 11, content);
}

TEST_F(TranslatorLogicTest, ForUnrollContinue3) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           break;
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 11, content);
}

TEST_F(TranslatorLogicTest, ForUnrollContinue4) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           if(a>155) {
             continue;
           }
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 171, content);
}

TEST_F(TranslatorLogicTest, ForUnrollContinue5) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           a += b;
           if(a>155) {
             continue;
           }
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 231, content);
}

TEST_F(TranslatorLogicTest, ForUnrollContinue6) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           {
             continue;
           }
           a += b;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 11, content);
}

TEST_F(TranslatorLogicTest, ReturnFromFor) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           return a;
           a += b;
         }
         return 0;
       })";
  Run({{"a", 233}, {"b", 0}}, 233, content);
}

TEST_F(TranslatorLogicTest, ReturnFromFor2) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<11;++i) {
           a += b;
           return a;
         }
         return 0;
       })";
  Run({{"a", 233}, {"b", 20}}, 253, content);
}

TEST_F(TranslatorLogicTest, ReturnFromFor3) {
  std::string_view content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<10;++i) {
           a += b;
           if(a>500) return a;
         }
         return 0;
       })";
  Run({{"a", 140}, {"b", 55}}, 525, content);
}

TEST_F(TranslatorLogicTest, MaxUnrollItersError) {
  std::string_view content = R"(
    #pragma hls_top
    int foo(int b) {
      int ret = 0;

      #pragma hls_unroll yes
      for(int i=0;i<5;++i) {
        ret += b;
      }
      return ret;
    })";
  ASSERT_THAT(SourceToIr(content, /*pfunc=*/nullptr,
                         /*clang_argv=*/{}, /*io_test_mode=*/false,
                         /*max_unroll_iters=*/4)
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kResourceExhausted,
                                     testing::HasSubstr("broke at maximum")));
}

TEST_F(TranslatorLogicTest, MaxUnrollItersEquals) {
  std::string_view content = R"(
    #pragma hls_top
    int foo(int b) {
      int ret = 0;

      #pragma hls_unroll yes
      for(int i=0;i<5;++i) {
        ret += b;
      }
      return ret;
    })";
  Run({{"b", 5}}, 25, content,
      /*loc=*/xabsl::SourceLocation::current(),
      /*clang_argv=*/{},
      /*max_unroll_iters=*/5);
}

TEST_F(TranslatorLogicTest, ReturnFromForStopsUnrolling) {
  std::string_view content = R"(
    #pragma hls_top
    int foo(int b) {
      int ret = 0;

      #pragma hls_unroll yes
      for(int i=0;i<5;++i) {
        ret += b;
        if(i==1) {
    			return 1000;
        }
      }
      return ret;
    })";
  Run({{"b", 5}}, 1000, content,
      /*loc=*/xabsl::SourceLocation::current(),
      /*clang_argv=*/{},
      /*max_unroll_iters=*/3);
}

TEST_F(TranslatorLogicTest, ReturnFromForInSwitchStopsUnrolling) {
  std::string_view content = R"(
    #pragma hls_top
    int foo(int b) {
      int ret = 0;

      #pragma hls_unroll yes
      for(int i=0;i<5;++i) {
        switch(i) {
          case 1:
      			return 1000;
          default:
            ret += b;
            break;
        }
      }
      return ret;
    })";
  Run({{"b", 5}}, 1000, content,
      /*loc=*/xabsl::SourceLocation::current(),
      /*clang_argv=*/{},
      /*max_unroll_iters=*/3);
}

TEST_F(TranslatorLogicTest, BreakFromForStopsUnrolling) {
  std::string_view content = R"(
    #pragma hls_top
    int foo(int b) {
      int ret = 0;
      #pragma hls_unroll yes
      for(int i=0;i<5;++i) {
        ret += b;
        if(i==1) {
          ret = 1000;
          break;
        }
      }
      return ret;
    })";
  Run({{"b", 5}}, 1000, content,
      /*loc=*/xabsl::SourceLocation::current(),
      /*clang_argv=*/{},
      /*max_unroll_iters=*/3);
}

TEST_F(TranslatorLogicTest, ConditionalReturnStmt) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        if(b) {
          if(a<200) return 2200;
          if(a<500) return 5500;
        }
        return a;
      })";

  Run({{"a", 505}, {"b", 1}}, 505, content);
  Run({{"a", 455}, {"b", 1}}, 5500, content);
  Run({{"a", 101}, {"b", 1}}, 2200, content);
  Run({{"a", 505}, {"b", 0}}, 505, content);
  Run({{"a", 455}, {"b", 0}}, 455, content);
  Run({{"a", 101}, {"b", 0}}, 101, content);
}

TEST_F(TranslatorLogicTest, DoubleReturn) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        if(b) {
          return b;
          return a;
        }
        return a;
        return b;
      })";

  Run({{"a", 11}, {"b", 0}}, 11, content);
  Run({{"a", 11}, {"b", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, TripleReturn) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        return 66;
        return 66;
        return a;
      })";

  Run({{"a", 11}, {"b", 0}}, 66, content);
  Run({{"a", 11}, {"b", 3}}, 66, content);
}

TEST_F(TranslatorLogicTest, VoidReturn) {
  std::string_view content = R"(
      void my_package(int &a) {
        a = 22;
      })";

  Run({{"a", 1000}}, 22, content);
  Run({{"a", 221}}, 22, content);
}

TEST_F(TranslatorLogicTest, AssignAfterReturn) {
  std::string_view content = R"(
      void my_package(int &a) {
        return;
        a = 22;
      })";

  Run({{"a", 1000}}, 1000, content);
}

TEST_F(TranslatorLogicTest, AssignAfterReturnInIf) {
  std::string_view content = R"(
      void my_package(int &a) {
        if(a == 5) {
          return;
        }
        a = 22;
      })";

  Run({{"a", 5}}, 5, content);
  Run({{"a", 10}}, 22, content);
  Run({{"a", 100}}, 22, content);
}

TEST_F(TranslatorLogicTest, AssignAfterReturn3) {
  std::string_view content = R"(
      void ff(int x[8]) {
       x[4] = x[2];
       return;
       x[3] = x[4];
      };
      #pragma hls_top
      int my_package(int a, int b,int c,int d,int e,int f,int g,int h) {
          int arr[8] = {a,b,c,d,e,f,g,h};
          ff(arr);
          return arr[4]+arr[3]+arr[5];
      })";
  Run({{"a", 3},
       {"b", 4},
       {"c", 5},
       {"d", 6},
       {"e", 7},
       {"f", 8},
       {"g", 9},
       {"h", 10}},
      19, content);
}

TEST_F(TranslatorLogicTest, CapitalizeFirstLetter) {
  std::string_view content = R"(
       class State {
        public:
           State()
            : last_was_space_(true) {
          }
           unsigned char process(unsigned char c) {
           unsigned char ret = c;
           if(last_was_space_ && (c >= 'a') && (c <= 'z'))
             ret -= ('a' - 'A');
           last_was_space_ = (c == ' ');
           return ret;
         }
        private:
          bool last_was_space_;
       };
       unsigned char my_package(State &st, unsigned char c) {
         return st.process(c);
       })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_src, SourceToIr(content));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> package,
                           ParsePackage(ir_src));
  XLS_ASSERT_OK(package->SetTopByName("my_package"));

  auto state =
      xls::Value(xls::Value::TupleOwned({xls::Value(xls::UBits(1, 1))}));

  const char* input = "hello world";
  std::string output;
  for (; *input != 0u; ++input) {
    const char inc = *input;
    XLS_ASSERT_OK_AND_ASSIGN(xls::Function * entry,
                             package->GetTopAsFunction());
    absl::flat_hash_map<std::string, xls::Value> args;
    args["st"] = state;
    args["c"] = xls::Value(xls::UBits(inc, 8));
    XLS_ASSERT_OK_AND_ASSIGN(
        xls::Value actual,
        DropInterpreterEvents(xls::InterpretFunctionKwargs(entry, args)));
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Value> returns,
                             actual.GetElements());
    ASSERT_EQ(returns.size(), 2);
    XLS_ASSERT_OK_AND_ASSIGN(char outc, returns[0].bits().ToUint64());

    state = returns[1];
    output += outc;
  }

  ASSERT_EQ(output, "Hello World");
}

TEST_F(TranslatorLogicTest, AssignmentInBlock) {
  std::string_view content = R"(
      int my_package(int a) {
        int r = a;
        {
          r = 55;
        }
        return r;
      })";

  Run({{"a", 100}}, 55, content);
}

TEST_F(TranslatorLogicTest, AssignmentInParens) {
  std::string_view content = R"(
      int my_package(int a) {
        int r = a;
        (r) = 55;
        return r;
      })";

  Run({{"a", 100}}, 55, content);
}

TEST_F(TranslatorLogicTest, ShadowAssignment) {
  std::string_view content = R"(
      int my_package(int a) {
        int r = a;
        {
          int r = 22;
          r = 55;
          (void)r;
        }
        return r;
      })";

  Run({{"a", 100}}, 100, content);
}

TEST_F(TranslatorLogicTest, CompoundStructAccess) {
  std::string_view content = R"(
       struct TestX {
         int x;
       };
       struct TestY {
         TestX tx;
       };
       int my_package(int a) {
         TestY y;
         y.tx.x = a;
         return y.tx.x;
       })";
  Run({{"a", 56}}, 56, content);
}

TEST_F(TranslatorLogicTest, SubstTemplateType) {
  constexpr const char* content = R"(
       struct TestR {
         int f()const {
           return 10;
         }
       };
       struct TestW {
         int f()const {
           return 11;
         }
       };
       template<typename T>
       int do_something(T a) {
         return a.f();
       }
       int my_package(int a) {
         %s t;
         return do_something(t);
       })";
  Run({{"a", 3}}, 10, absl::StrFormat(content, "TestR"));
  Run({{"a", 3}}, 11, absl::StrFormat(content, "TestW"));
}

TEST_F(TranslatorLogicTest, TemplateStruct) {
  std::string_view content = R"(
       template<typename T>
       struct TestX {
         T x;
       };
       int my_package(int a) {
         TestX<int> x;
         x.x = a;
         return x.x;
       })";
  Run({{"a", 56}}, 56, content);
}

TEST_F(TranslatorLogicTest, ArrayOfStructsAccess) {
  std::string_view content = R"(
       struct TestX {
         int x;
       };
       struct TestY {
         TestX tx;
       };
       int my_package(int a) {
         TestY y[3];
         y[2].tx.x = a;
         return y[2].tx.x;
       })";
  Run({{"a", 56}}, 56, content);
}

TEST_F(TranslatorLogicTest, StructWithArrayAccess) {
  std::string_view content = R"(
       struct TestX {
         int x[3];
       };
       struct TestY {
         TestX tx;
       };
       int my_package(int a) {
         TestY y;
         y.tx.x[2] = a;
         return y.tx.x[2];
       })";
  Run({{"a", 56}}, 56, content);
}

TEST_F(TranslatorLogicTest, StructInitList) {
  std::string_view content = R"(
       struct Test {
         long long x;
         long long y;
         long long z;
       };
       long long my_package(long long a, long long b) {
         Test ret = {a,5,b};
         return ret.x+ret.y+ret.z;
       })";
  Run({{"a", 11}, {"b", 50}}, 66, content);
}

TEST_F(TranslatorLogicTest, StructConditionalAssign) {
  std::string_view content = R"(
       struct Test {
         long long x;
         long long y;
         long long z;
       };
       long long my_package(long long a, long long b) {
         Test ret = {a,5,b};
         if(a>11) {
           ret.y = 100;
         }
         return ret.x+ret.y+ret.z;
       })";
  Run({{"a", 11}, {"b", 50}}, 11 + 5 + 50, content);
  Run({{"a", 12}, {"b", 50}}, 12 + 100 + 50, content);
}

TEST_F(TranslatorLogicTest, StructInitListWrongCount) {
  std::string_view content = R"(
       struct Test {
         long long x;
         long long y;
         long long z;
       };
       long long my_package(long long a, long long b) {
         Test ret = {a,b};
         return ret.x+ret.y+ret.z;
       })";
  Run({{"a", 11}, {"b", 50}}, 61, content);
}

TEST_F(TranslatorLogicTest, StructInitListWithDefaultnt) {
  std::string_view content = R"(
       struct Test {
         long long x;
         long long y;
         long long z = 100;
       };
       long long my_package(long long a, long long b) {
         Test ret = {a,b,10};
         return ret.x+ret.y+ret.z;
       })";
  Run({{"a", 11}, {"b", 50}}, 71, content);
}

TEST_F(TranslatorLogicTest, StructInitListWithDefaultWrongCount) {
  std::string_view content = R"(
       struct Test {
         long long x;
         long long y;
         long long z = 100;
       };
       long long my_package(long long a, long long b) {
         Test ret = {a,b};
         return ret.x+ret.y+ret.z;
       })";
  Run({{"a", 11}, {"b", 50}}, 161, content);
}

TEST_F(TranslatorLogicTest, NoTupleStruct) {
  std::string_view content = R"(
      struct [[hls_no_tuple]] Test {
         int x;
       };
       Test my_package(int a) {
         Test s;
         s.x=a;
         return s;
       })";
  Run({{"a", 311}}, 311, content);
}

TEST_F(TranslatorLogicTest, NoTupleMultiField) {
  std::string_view content = R"(
       struct [[hls_no_tuple]] Test {
         int x;
         int y;
       };
       Test my_package(int a) {
         Test s;
         s.x=a;
         return s;
       })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("only 1 field")));
}

TEST_F(TranslatorLogicTest, StructMemberOrder) {
  std::string_view content = R"(
       struct Test {
         int x;
         long y;
       };
       Test my_package(int a, int b) {
         Test s;
         s.x=a;
         s.y=b;
         return s;
       })";

  xls::Value tuple_values[2] = {xls::Value(xls::SBits(50, 64)),
                                xls::Value(xls::SBits(311, 32))};
  xls::Value expected = xls::Value::Tuple(tuple_values);

  absl::flat_hash_map<std::string, xls::Value> args = {
      {"a", xls::Value(xls::SBits(311, 32))},
      {"b", xls::Value(xls::SBits(50, 32))}};

  Run(args, expected, content);
}

TEST_F(TranslatorLogicTest, ImplicitConversion) {
  std::string_view content = R"(
       struct Test {
         Test(int v) : x(v) {
           this->y = 10;
         }
         operator int()const {
           return x+y;
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s(a);
         return s;
       })";
  Run({{"a", 3}}, 13, content);
}

TEST_F(TranslatorLogicTest, OperatorOverload) {
  std::string_view content = R"(
       struct Test {
         Test(int v) : x(v) {
           this->y = 10;
         }
         Test operator+=(Test const&o) {
           x *= o.y;
           return *this;
         }
         Test operator+(Test const&o) {
           return x-o.x;
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s1(a);
         Test s2(a);
         s1 += s2; // s1.x = a * 10
         return (s1 + s2).x; // Return (a*10)-a
       })";
  Run({{"a", 3}}, 27, content);
}

TEST_F(TranslatorLogicTest, OperatorOnBuiltin) {
  std::string_view content = R"(
       struct Test {
         Test(int v) : x(v) {
         }
         int x;
       };
       Test operator+(int a, Test b) {
         return Test(a+b.x);
       }
       int my_package(int a) {
         Test s1(a);
         return (10+s1).x;
       })";
  Run({{"a", 3}}, 13, content);
}

TEST_F(TranslatorLogicTest, UnaryOperatorAvoidUnsequencedError2) {
  std::string_view content = R"(
       struct Test {
         Test(int v) : x(v) {
           this->y = 10;
         }
         Test(const Test &o) : x(o.x) {
           this->y = 10;
         }
         Test operator +(Test o) const {
           return Test(x + o.x);
         }
         operator int () const {
           return x;
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s1(a);
         Test s2(0);
         s2 = s1 + Test(1);
         return s2;
       })";
  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorLogicTest, UnaryOperatorAvoidUnsequencedError3) {
  std::string_view content = R"(
       struct Test {
         Test(int v) : x(v) {
           this->y = 10;
         }
         Test(const Test &o) : x(o.x) {
           this->y = 10;
         }
         Test operator ++() {
           x = x + 1;
           return (*this);
         }
         operator int () const {
           return x;
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s1(a);
         Test s2(0);
         s2 = ++s1;
         return s2;
       })";
  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorLogicTest, TypedefStruct) {
  std::string_view content = R"(
       typedef struct {
         int x;
         int y;
       }Test;
       int my_package(int a) {
         Test s;
         s.x = a;
         s.y = a*10;
         return s.x+s.y;
       })";
  Run({{"a", 3}}, 33, content);
}

TEST_F(TranslatorLogicTest, ConvertToVoid) {
  std::string_view content = R"(
       struct ts {int x;};
       long long my_package(long long a) {
         ts t;
         (void)t;
         return a;
       })";
  Run({{"a", 10}}, 10, content);
}

TEST_F(TranslatorLogicTest, AvoidDoubleAssignmentFromBackwardsEval) {
  std::string_view content = R"(
       struct Test {
         Test(int v) : x(v) {
           this->y = 10;
         }
         Test(const Test &o) : x(o.x) {
           this->y = 10;
         }
         Test operator ++() {
           x = x + 1;
           return (*this);
         }
         operator int () const {
           return x;
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s1(a);
         Test s2(0);
         s2 = ++s1;
         return s1;
       })";
  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorLogicTest, CompoundAvoidUnsequenced) {
  std::string_view content = R"(
       struct Test {
         int x;
       };
       int my_package(int a) {
         Test s1;
         s1.x = a;
         s1.x = ++s1.x;
         return s1.x;
       })";
  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorLogicTest, CompoundAvoidUnsequenced2) {
  std::string_view content = R"(
       int my_package(int a) {
         int s1[2] = {a, a};
         s1[0] = ++s1[1];
         return s1[0];
       })";
  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorLogicTest, DefaultValues) {
  std::string_view content = R"(
       struct Test {
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         return s.x+s.y+a;
       })";
  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, StructMemberReferenceParameter) {
  std::string_view content = R"(
       struct Test {
         int p;
       };
       int do_something(Test &x, int a) {
         x.p += a;
         return x.p;
       }
       int my_package(int a) {
         Test ta;
         ta.p = a;
         do_something(ta, 5);
         return do_something(ta, 10);
       })";
  Run({{"a", 3}}, 3 + 5 + 10, content);
}

TEST_F(TranslatorLogicTest, AnonStruct) {
  std::string_view content = R"(
       int my_package(int a) {
         struct {
           int x;
           int y;
         } s;
         s.x = a;
         s.y = a*10;
         return s.x+s.y;
       })";
  Run({{"a", 3}}, 3 + 3 * 10, content);
}

TEST_F(TranslatorLogicTest, Inheritance) {
  std::string_view content = R"(
       struct Base {
         int x;
       };
       struct Derived : public Base {
         int foo()const {
           return x;
         }
       };
       int my_package(int x) {
         Derived b;
         b.x = x;
         return b.foo();
       })";
  Run({{"x", 47}}, 47, content);
}

TEST_F(TranslatorLogicTest, BaseConstructor) {
  std::string_view content = R"(
       struct Base {
         Base() : x(88) { }
          int x;
       };
       struct Derived : public Base {
       };
       int my_package(int x) {
         Derived b;
         return x + b.x;
       })";
  Run({{"x", 15}}, 103, content);
}

TEST_F(TranslatorLogicTest, BaseConstructorNoTuple) {
  std::string_view content = R"(
       struct [[hls_no_tuple]] Base {
         Base() : x(88) { }
          int x;
       };
       struct [[hls_no_tuple]] Derived : public Base {
       };
       int my_package(int x) {
         Derived b;
         return x + b.x;
       })";
  Run({{"x", 15}}, 103, content);
}

TEST_F(TranslatorLogicTest, InheritanceNoTuple) {
  std::string_view content = R"(
       struct Base {
         int x;
       };
       struct [[hls_no_tuple]] Derived : public Base {
         int foo()const {
           return x;
         }
       };
       int my_package(int x) {
         Derived b;
         b.x = x;
         return b.foo();
       })";
  Run({{"x", 47}}, 47, content);
}

TEST_F(TranslatorLogicTest, InheritanceNoTuple2) {
  std::string_view content = R"(
       struct [[hls_no_tuple]] Base {
         int x;
       };
       struct [[hls_no_tuple]] Derived : public Base {
         int foo()const {
           return x;
         }
       };
       int my_package(int x) {
         Derived b;
         b.x = x;
         return b.foo();
       })";
  Run({{"x", 47}}, 47, content);
}

TEST_F(TranslatorLogicTest, InheritanceNoTuple4) {
  std::string_view content = R"(
       struct [[hls_no_tuple]] Base {
         int x;
         void set(int v) { x=v; }
         int get()const { return x; }
       };
       struct [[hls_no_tuple]] Derived : public Base {
         void setd(int v) { x=v; }
         int getd()const { return x; }
       };
       int my_package(int x) {
         Derived d;
         d.setd(x);
         d.setd(d.getd()*3);
         d.set(d.get()*5);
         return d.x;
       })";
  Run({{"x", 10}}, 150, content);
}

TEST_F(TranslatorLogicTest, InheritanceTuple) {
  std::string_view content = R"(
       struct Base {
         int x;
         void set(int v) { x=v; }
         int get()const { return x; }
       };
       struct Derived : public Base {
         void setd(int v) { x=v; }
         int getd()const { return x; }
       };
       int my_package(int x) {
         Derived d;
         d.setd(x);
         d.setd(d.getd()*3);
         d.set(d.get()*5);
         return d.x;
       })";
  Run({{"x", 10}}, 150, content);
}

TEST_F(TranslatorLogicTest, Constructor) {
  std::string_view content = R"(
      struct Test {
        Test() : x(5) {
          y = 10;
        }
        int x;
        int y;
      };
      int my_package(int a) {
        Test s;
        return s.x+s.y;
      })";
  Run({{"a", 3}}, 15, content);
}

TEST_F(TranslatorLogicTest, Destructor) {
  std::string_view content = R"(
      struct Test {
        Test() : x(5) {
          y = 10;
        }
        ~Test() {

        }
        int x;
        int y;
      };
      int my_package(int a) {
        Test s;
        return s.x+s.y;
      })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("aren't yet called")));
}

TEST_F(TranslatorLogicTest, ConstructorWithArg) {
  std::string_view content = R"(
      struct Test {
        Test(int v) : x(v) {
          y = 10;
        }
        int x;
        int y;
      };
      int my_package(int a) {
        Test s(a);
        return s.x+s.y;
      })";
  Run({{"a", 3}}, 13, content);
}

TEST_F(TranslatorLogicTest, ConstructorWithThis) {
  std::string_view content = R"(
      struct Test {
        Test(int v) : x(v) {
          this->y = 10;
        }
        int x;
        int y;
      };
      int my_package(int a) {
        Test s(a);
        return s.x+s.y;
      })";
  Run({{"a", 3}}, 13, content);
}

TEST_F(TranslatorLogicTest, ExplicitDefaultConstructor) {
  std::string_view content = R"(
         struct TestR {
           int bb;
         };
         #pragma hls_top
         int my_package(int a) {
            TestR b = TestR();
           return b.bb + a;
         })";
  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, ConditionallyAssignThis) {
  std::string_view content = R"(
       struct ts {
         void blah() {
           return;
           v = v | 1;
         }
         int v;
       };
       #pragma hls_top
       int my_package(int a) {
         ts t;
         t.v = a;
         t.blah();
         return t.v;
       })";
  Run({{"a", 6}}, 6, content);
}

TEST_F(TranslatorLogicTest, SetMemberInnerContext) {
  std::string_view content = R"(
       struct Test {
         void set_x(int v) {
           { x = v; }
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         s.set_x(a);
         s.y = 11;
         return s.x+s.y;
       })";
  Run({{"a", 3}}, 14, content);
}

TEST_F(TranslatorLogicTest, StaticMethod) {
  std::string_view content = R"(
       struct Test {
          static int foo(int a) {
            return a+5;
          }
       };
       int my_package(int a) {
         return Test::foo(a);
       })";
  Run({{"a", 3}}, 8, content);
}

TEST_F(TranslatorLogicTest, SignExtend) {
  {
    std::string_view content = R"(
        unsigned long long my_package(long long a) {
          return long(a);
        })";

    Run({{"a", 3}}, 3, content);
  }
  {
    std::string_view content = R"(
        long long my_package(long long a) {
          return (unsigned long)a;
        })";

    Run({{"a", 3}}, 3, content);
    Run({{"a", -3}}, 18446744073709551613ull, content);
  }
}

TEST_F(TranslatorLogicTest, TopFunctionByName) {
  std::string_view content = R"(
      int my_package(int a) {
        return a + 1;
      })";

  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorLogicTest, TopFunctionPragma) {
  std::string_view content = R"(
      #pragma hls_top
      int asdf(int a) {
        return a + 1;
      })";

  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorLogicTest, TopFunctionNoPragma) {
  std::string_view content = R"(
      int asdf(int a) {
        return a + 1;
      })";
  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(absl::StatusCode::kNotFound,
                             testing::HasSubstr("No top function found")));
}

TEST_F(TranslatorLogicTest, Function) {
  std::string_view content = R"(
      int do_something(int a) {
        return a;
      }
      int my_package(int a) {
        return do_something(a);
      })";

  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, FunctionNoOutputs) {
  std::string_view content = R"(
      void do_nothing(int a) {
        (void)a;
      }
      int my_package(int a) {
        do_nothing(a);
        return a;
      })";

  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, TopFunctionNoOutputs) {
  std::string_view content = R"(
      void my_package(int a) {
        (void)a;
      })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("no outputs")));
}

TEST_F(TranslatorLogicTest, DefaultArg) {
  std::string_view content = R"(
      int do_something(int a, int b=2) {
        return a+b;
      }
      int my_package(int a) {
        return do_something(a);
      })";

  Run({{"a", 3}}, 5, content);
}

TEST_F(TranslatorLogicTest, FunctionInline) {
  std::string_view content = R"(
      inline int do_something(int a) {
        return a;
      }
      int my_package(int a) {
        return do_something(a);
      })";

  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, TemplateFunction) {
  std::string_view content = R"(
      template<int N>
      int do_something(int a) {
        return a+N;
      }
      int my_package(int a) {
        return do_something<5>(a);
      })";

  Run({{"a", 3}}, 8, content);
}

TEST_F(TranslatorLogicTest, TemplateFunctionBool) {
  constexpr const char* content = R"(
      template<bool C>
      int do_something(int a) {
        return C?a:15;
      }
      int my_package(int a) {
        return do_something<%s>(a);
      })";
  Run({{"a", 3}}, 3, absl::StrFormat(content, "true"));
  Run({{"a", 3}}, 15, absl::StrFormat(content, "false"));
}

TEST_F(TranslatorLogicTest, FunctionDeclOrder) {
  std::string_view content = R"(
      int do_something(int a);
      int my_package(int a) {
        return do_something(a);
      }
      int do_something(int a) {
        return a;
      })";

  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, FunctionDeclMissing) {
  std::string_view content = R"(
      int do_something(int a);
      int my_package(int a) {
        return do_something(a);
      })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kNotFound,
                  testing::HasSubstr("do_something used but has no body")));
}

TEST_F(TranslatorLogicTest, ReferenceParameter) {
  std::string_view content = R"(
      int do_something(int &x, int a) {
        x += a;
        return x;
      }
      int my_package(int a) {
        do_something(a, 5);
        return do_something(a, 10);
      })";

  Run({{"a", 3}}, 3 + 5 + 10, content);
}

TEST_F(TranslatorLogicTest, Namespace) {
  std::string_view content = R"(
      namespace test {
      int do_something(int a) {
        return a;
      }
      }
      int my_package(int a) {
        return test::do_something(a);
      })";

  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorLogicTest, NamespaceFailure) {
  std::string_view content = R"(
      namespace test {
      int do_something(int a) {
        return a;
      }
      }
      int my_package(int a) {
        return do_something(a);
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("undeclared")));
}

TEST_F(TranslatorLogicTest, Ternary) {
  std::string_view content = R"(
      int my_package(int a) {
        return a ? a : 11;
      })";

  Run({{"a", 3}}, 3, content);
  Run({{"a", 0}}, 11, content);
}

// This is here mainly to check for graceful exit with no memory leaks
TEST_F(TranslatorLogicTest, ParseFailure) {
  std::string_view content = "int my_package(int a) {";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("expected")));
}

std::string NativeOperatorTestIr(const std::string& op) {
  return absl::StrFormat(R"(
      long long my_package(long long a, long long b) {
        return a %s b;
      })",
                         op);
}

std::string NativeOperatorTestIrEq(const std::string& op) {
  return absl::StrFormat(R"(
      long long my_package(long long a, long long b) {
        a %s= b;
        return a;
      })",
                         op);
}

TEST_F(TranslatorLogicTest, NativeOperatorAdd) {
  const std::string op = "+";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 3}, {"b", 10}}, 13, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 11}, {"b", 22}}, 33, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorSub) {
  const std::string op = "-";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 8}, {"b", 3}}, 5, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 30}, {"b", 11}}, 19, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorMul) {
  const std::string op = "*";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 3}, {"b", 10}}, 30, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 11}, {"b", 2}}, 22, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorDiv) {
  const std::string op = "/";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 55}, {"b", 3}}, 18, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", -1800}, {"b", 18}}, -100, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorRem) {
  const std::string op = "%";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 55}, {"b", 3}}, 1, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", -1800}, {"b", 18}}, 0, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorAnd) {
  const std::string op = "&";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 0b1001}, {"b", 0b0110}}, 0b0000, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 0b1001}, {"b", 0b1110}}, 0b1000, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorOr) {
  const std::string op = "|";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 0b1001}, {"b", 0b0110}}, 0b1111, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 0b1001}, {"b", 0b1110}}, 0b1111, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 0b1000}, {"b", 0b1110}}, 0b1110, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorXor) {
  const std::string op = "^";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 0b1001}, {"b", 0b0110}}, 0b1111, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 0b1001}, {"b", 0b1110}}, 0b0111, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 0b1000}, {"b", 0b1110}}, 0b0110, content);
  }
}

TEST_F(TranslatorLogicTest, NativeOperatorNot) {
  std::string_view content = R"(
      long long my_package(unsigned long long a) {
        return (long long)(~a);
      })";

  Run({{"a", 0b000}}, ~static_cast<uint64_t>(0b000), content);
  Run({{"a", 0b111}}, ~static_cast<uint64_t>(0b111), content);
  Run({{"a", 0b101}}, ~static_cast<uint64_t>(0b101), content);
}

TEST_F(TranslatorLogicTest, NativeOperatorNeg) {
  std::string_view content = R"(
      long long my_package(long long a) {
        return (long long)(-a);
      })";

  Run({{"a", 11}}, -11, content);
  Run({{"a", 0}}, 0, content);
  Run({{"a", -1000}}, 1000, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorPlus) {
  std::string_view content = R"(
      long long my_package(long long a) {
        return (long long)(+a);
      })";

  Run({{"a", 11}}, 11, content);
  Run({{"a", 0}}, 0, content);
  Run({{"a", -1000}}, -1000, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorShrSigned) {
  const std::string op = ">>";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 10}, {"b", 1}}, 5, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", -20}, {"b", 2}}, -5, content);
  }
}
TEST_F(TranslatorLogicTest, NativeOperatorShrUnsigned) {
  std::string_view content = R"(
      unsigned long long my_package(unsigned long long a, unsigned long long b)
      {
        return a >> b;
      })";
  {
    Run({{"a", 10}, {"b", 1}}, 5, content);
  }
  {
    Run({{"a", -20}, {"b", 2}}, 4611686018427387899L, content);
  }
}
TEST_F(TranslatorLogicTest, NativeOperatorShl) {
  const std::string op = "<<";
  {
    const std::string content = NativeOperatorTestIr(op);

    Run({{"a", 16}, {"b", 1}}, 32, content);
  }
  {
    const std::string content = NativeOperatorTestIrEq(op);

    Run({{"a", 13}, {"b", 2}}, 52, content);
  }
}
TEST_F(TranslatorLogicTest, NativeOperatorPreInc) {
  {
    std::string_view content = R"(
        int my_package(int a) {
          return ++a;
        })";

    Run({{"a", 10}}, 11, content);
  }
  {
    std::string_view content = R"(
        int my_package(int a) {
          ++a;
          return a;
        })";

    Run({{"a", 50}}, 51, content);
  }
}
TEST_F(TranslatorLogicTest, NativeOperatorPostInc) {
  {
    std::string_view content = R"(
        int my_package(int a) {
          return a++;
        })";

    Run({{"a", 10}}, 10, content);
  }
  {
    std::string_view content = R"(
        int my_package(int a) {
          a++;
          return a;
        })";

    Run({{"a", 50}}, 51, content);
  }
}
TEST_F(TranslatorLogicTest, NativeOperatorPreDec) {
  {
    std::string_view content = R"(
        int my_package(int a) {
          return --a;
        })";

    Run({{"a", 10}}, 9, content);
  }
  {
    std::string_view content = R"(
        int my_package(int a) {
          --a;
          return a;
        })";

    Run({{"a", 50}}, 49, content);
  }
}
TEST_F(TranslatorLogicTest, NativeOperatorPostDec) {
  {
    std::string_view content = R"(
        int my_package(int a) {
          return a--;
        })";

    Run({{"a", 10}}, 10, content);
  }
  {
    std::string_view content = R"(
        int my_package(int a) {
          a--;
          return a;
        })";

    Run({{"a", 50}}, 49, content);
  }
}

std::string NativeBoolOperatorTestIr(const std::string& op) {
  return absl::StrFormat(R"(
      long long my_package(long long a, long long b) {
        return (long long)(a %s b);
      })",
                         op);
}

std::string NativeUnsignedBoolOperatorTestIr(const std::string& op) {
  return absl::StrFormat(R"(
      long long my_package(unsigned long long a, unsigned long long b) {
        return (long long)(a %s b);
      })",
                         op);
}

TEST_F(TranslatorLogicTest, NativeOperatorEq) {
  const std::string op = "==";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorNe) {
  const std::string op = "!=";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorGt) {
  const std::string op = ">";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorGtU) {
  const std::string op = ">";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorGte) {
  const std::string op = ">=";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorGteU) {
  const std::string op = ">=";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorLt) {
  const std::string op = "<";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorLtU) {
  const std::string op = "<";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorLte) {
  const std::string op = "<=";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorLteU) {
  const std::string op = "<=";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorLAnd) {
  const std::string op = "&&";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 0b111}, {"b", 0b111}}, 1, content);
  Run({{"a", 0b001}, {"b", 0b100}}, 1, content);
  Run({{"a", 0b111}, {"b", 0}}, 0, content);
  Run({{"a", 0}, {"b", 0}}, 0, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorLOr) {
  const std::string op = "||";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 0b111}, {"b", 0b111}}, 1, content);
  Run({{"a", 0b001}, {"b", 0b100}}, 1, content);
  Run({{"a", 0b111}, {"b", 0}}, 1, content);
  Run({{"a", 0}, {"b", 0}}, 0, content);
}

TEST_F(TranslatorLogicTest, NativeOperatorLNot) {
  std::string_view content = R"(
      long long my_package(unsigned long long a) {
        return (long long)(!a);
      })";

  Run({{"a", 0}}, 1, content);
  Run({{"a", 11}}, 0, content);
  Run({{"a", -11}}, 0, content);
}

TEST_F(TranslatorLogicTest, SizeOf) {
  std::string_view content = R"(
      int my_package() {
        short foo[3] = {2,3,4};
        return sizeof(foo) / sizeof(foo[0]);
      })";

  Run({}, 3, content);
}

TEST_F(TranslatorLogicTest, SizeOfDeep) {
  std::string_view content = R"(
      struct TestInner {
        short foo;
      };
      struct Test {
        TestInner x[5];
      };
      int my_package() {
        Test b;
        return sizeof(b) / sizeof(TestInner);
      })";

  Run({}, 5, content);
}

TEST_F(TranslatorLogicTest, ParenType) {
  std::string_view content = R"(
    int thing(const int (&arr)[2]) {
      return arr[0] + arr[1];
    }
    int my_package(int a) {
      int arr[2] = {a, 1+a};
      return thing(arr);
    })";

  Run({{"a", 10}}, 21, content);
}

TEST_F(TranslatorLogicTest, DefaultArrayInit) {
  std::string_view content = R"(
    struct Val {
      Val() : v(0) { }
      Val(const Val&o) : v(o.v) {
      }
      int v;
    };
    struct Foo {
      Val a[16];
    };

    int my_package(int a) {
      Foo x;
      x.a[1].v = a;
      Foo y;
      y = x;
      return 1+y.a[1].v;
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound,
                                     testing::HasSubstr("__builtin_memcpy")));
}

TEST_F(TranslatorLogicTest, ZeroIterationForLoop) {
  std::string_view content = R"(
    long long my_package(long long a) {
      for(;0;) {}
      return a;
    })";
  Run({{"a", 1}}, 1, content);
}

TEST_F(TranslatorLogicTest, OnlyUnrolledLoops) {
  std::string_view content = R"(
    long long my_package(long long a) {
      for(;1;) {}
      return a;
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("missing #pragma or attribute")));
}

TEST_F(TranslatorLogicTest, InvalidUnrolledLoop) {
  std::string_view content = R"(
    long long my_package(long long a) {
      #pragma hls_unroll yes
      for(;1;) {}
      return a;
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kResourceExhausted,
                                     testing::HasSubstr("maximum")));
}

TEST_F(TranslatorLogicTest, NonPragmaNestedLoop) {
  std::string_view content = R"(
    long long my_package(long long a) {
      #pragma hls_unroll yes
      for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
          a++;
        }
      }
      return a;
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("missing #pragma or attribute")));
}

TEST_F(TranslatorLogicTest, Label) {
  std::string_view content = R"(
      long long my_package(long long a, long long b) {
        this_loop:
        #pragma hls_unroll yes
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  Run({{"a", 11}, {"b", 20}}, 611, content);
}

TEST_F(TranslatorLogicTest, DisallowUsed) {
  std::string_view content = R"(
      #include "/xls_builtin.h"

      int foo(int a) {
        (void)__xlscc_unimplemented();
        return a+3;
      }

      int my_package(int a) {
        return foo(a);
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("Unimplemented")));
}

TEST_F(TranslatorLogicTest, DisallowUnused) {
  std::string_view content = R"(
      #include "/xls_builtin.h"

      int foo(int a) {
        __xlscc_unimplemented();
        return a+3;
      }

      int my_package(int a) {
        return a+5;
      })";
  Run({{"a", 11}}, 16, content);
}

// Checks that unnecessary conditions aren't applied to assignments
// Variables don't need conditions from before their declarations
// This won't cause logical incorrectness, but it overcomplicates the IR
// and can cause loop unrolling to fail.
TEST_F(TranslatorLogicTest, ConditionsPerVariable) {
  std::string_view content = R"(
    #pragma hls_top
    int my_package(int a, int b) {
      if(b) {
        int ret;
        ret = a;
        return ret;
      }
      return -1;
    })";

  xlscc::GeneratedFunction* pfunc = nullptr;
  XLS_ASSERT_OK(SourceToIr(content, &pfunc).status());
  ASSERT_NE(pfunc, nullptr);
  xls::Node* return_node = pfunc->xls_func->return_value();
  ASSERT_NE(return_node, nullptr);
  xls::Select* select_node = return_node->As<xls::Select>();
  ASSERT_NE(select_node, nullptr);
  const absl::Span<xls::Node* const>& cases = select_node->cases();
  ASSERT_EQ(cases.size(), 2);
  // Check for direct reference to parameter
  EXPECT_TRUE(cases[0]->Is<xls::Param>() || cases[1]->Is<xls::Param>());
}

TEST_F(TranslatorLogicTest, FileNumbersIncluded) {
  std::string_view content = R"(
  #include "/xls_builtin.h"

  #pragma hls_top
  void st(__xls_channel<int>& in,
           __xls_channel<int>& out) {
    const int ctrl = in.read();
    out.write(ctrl + 1);
  })";
  XLS_ASSERT_OK_AND_ASSIGN(xls::TempFile temp,
                           xls::TempFile::CreateWithContent(content, ".cc"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(temp));

  ASSERT_TRUE(absl::StrContains(ir, temp.path().string()));
}

TEST_F(TranslatorLogicTest, BooleanOrAssign) {
  std::string_view content = R"(
  #pragma hls_top
  bool st(bool b) {
    b |= false;
    return b;
  })";
  Run({{"b", false}}, static_cast<int64_t>(false), content);
  Run({{"b", true}}, static_cast<int64_t>(true), content);
}

TEST_F(TranslatorLogicTest, BooleanAndAssign) {
  std::string_view content = R"(
  #pragma hls_top
  bool st(bool b) {
    b &= true;
    return b;
  })";
  Run({{"b", true}}, static_cast<int64_t>(true), content);
  Run({{"b", false}}, static_cast<int64_t>(false), content);
}

TEST_F(TranslatorLogicTest, EnumConstant) {
  std::string_view content = R"(
  struct Values{
   enum { zero = 2 };
  };
  #pragma hls_top
  int st(int a) {
    Values v;
    const int val = a * v.zero;
    return val;
  })";
  Run({{"a", 5}}, 10, content);
}

TEST_F(TranslatorLogicTest, NonInitEnumConstant) {
  std::string_view content = R"(
  struct Values{
   enum { zero, one };
  };
  #pragma hls_top
  int st(int a) {
    Values v;
    const int val = a * v.zero;
    return val;
  })";
  Run({{"a", 5}}, 0, content);
}

TEST_F(TranslatorLogicTest, NonInitOneEnumConstant) {
  std::string_view content = R"(
  struct Values{
   enum { zero, one };
  };
  #pragma hls_top
  int st(int a) {
    Values v;
    const int val = a * v.one;
    return val;
  })";
  Run({{"a", 5}}, 5, content);
}

TEST_F(TranslatorLogicTest, TopMemberAccess) {
  std::string_view content = R"(
      #include "/xls_builtin.h"

      struct Test {
        int foo = 0;

        #pragma hls_top
        void test(__xls_channel<int>& in,
                  __xls_channel<int> &out) {
          const int ctrl = in.read() + foo;
          out.write(ctrl);
        }
      };)";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  XLS_ASSERT_OK(ScanFile(content));
  package_ = std::make_unique<xls::Package>("my_package");
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("top level methods are not supported")));
}

TEST_F(TranslatorLogicTest, ParameterPack) {
  std::string_view content = R"(
      int inner(int a, int b, int c) {
          return a+2*b+5*c;
      }

      template<typename...Ranges>
      int Test(Ranges... b) {
          return inner(b...);
      }

      #pragma hls_top
      int sum(int a, int b, int c) {
          return Test<int,int,int>(a,b,c);
      })";
  Run({{"a", 1}, {"b", 2}, {"c", 3}}, 20, content);
}

TEST_F(TranslatorLogicTest, FunctionEnum) {
  std::string_view content = R"(
    #pragma hls_top
    int sum(int a) {
        enum b { B0 = 0, B1 = 1 };
        return B1 + a;
    }
    )";
  Run({{"a", 5}}, 6, content);
}

TEST_F(TranslatorLogicTest, UnusedTemplate) {
  std::string_view content = R"(
      template<typename T, int N>
      struct StructVal { T data[N]; };

      #pragma hls_top
      int my_package(StructVal<long, 2> a) {
        return 0;
      }
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_src, SourceToIr(content));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> package,
                           ParsePackage(ir_src));
  XLS_ASSERT_OK(package->SetTopByName("my_package"));
  std::vector<uint64_t> in_vals = {55, 20};
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value in_arr,
                           xls::Value::UBitsArray(in_vals, 64));
  xls::Value in_tuple = xls::Value::Tuple({in_arr});
  absl::flat_hash_map<std::string, xls::Value> args;
  args["a"] = in_tuple;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * entry, package->GetTopAsFunction());

  auto x = DropInterpreterEvents(
      xls::InterpretFunctionKwargs(entry, {{"a", in_tuple}}));

  ASSERT_THAT(x, IsOkAndHolds(xls::Value(xls::UBits(0, 32))));
}

TEST_F(TranslatorLogicTest, BinaryOpComma) {
  std::string_view content = R"(
      #pragma hls_top
      int test(int a) {
          return (a=5,a+1);
      })";
  Run({{"a", 1}}, 6, content);
}

// Check that hls_array_allow_default_pad pragma fills with 0's
TEST_F(TranslatorLogicTest, ArrayZeroExtendFillsZeros) {
  std::string_view content = R"(
        #pragma hls_top
        int my_package(int a) {
         #pragma hls_array_allow_default_pad
         int x[4] = {1,2};
         return x[2];
       })";
  Run({{"a", 5}}, 0, content);
}

TEST_F(TranslatorLogicTest, ArrayZeroExtendFillsZerosLargeMultidimension) {
  std::string_view content = R"(
        #pragma hls_top
        int my_package(int a) {
         #pragma hls_array_allow_default_pad
         unsigned arr[2][3][4] = {{{1,0}}, {{2,2}}};
         int sum = 0;
         arr[0][0][0] = 5;
         #pragma hls_unroll yes
         for (int i = 0; i < 2; i++) {
           #pragma hls_unroll yes
           for (int j = 0; j < 3; j++) {
             #pragma hls_unroll yes
             for (int k = 0; k < 4; k++) {
               sum += arr[i][j][k];
             }
            }
          }
         return sum;
       })";
  Run({{"a", 5}}, 5 + 2 + 2, content);
}

// Check that hls_array_allow_default_pad pragma maintains supplied values
TEST_F(TranslatorLogicTest, ArrayZeroExtendMaintainsValues) {
  std::string_view content = R"(
        #pragma hls_top
        int my_package(int a) {
         #pragma hls_array_allow_default_pad
         int x[4] = {1,2};
         return x[a];
       })";
  Run({{"a", 1}}, 2, content);
  Run({{"a", 2}}, 0, content);
}

TEST_F(TranslatorLogicTest, ArrayExtendError) {
  std::string_view content = R"(
    class Foo {
        #pragma hls_top
        void my_package() {
         int x[4] = {1,2};
         (void)x;
       }
    };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false,
                         /*error_on_uninitialized=*/true));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  ASSERT_THAT(
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                             /*top_level_init_interval=*/0),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("number of initializers")));
}

// Check that hls_array_allow_default_pad pragma maintains supplied values
TEST_F(TranslatorLogicTest, ArrayZeroExtendStruct) {
  std::string_view content = R"(
        struct y {
          int z;
        };
        #pragma hls_top
        int my_package(int a) {
         #pragma hls_array_allow_default_pad
         y x[4] = {{1},{2}};
         return x[1].z;
       })";
  Run({{"a", 5}}, 2, content);
}

// Check that hls_array_allow_default_pad pragma
// works on structs with constructors
TEST_F(TranslatorLogicTest, ArrayZeroExtendStructWithConstructor) {
  std::string_view content = R"(
        struct y {
          int z;
          y() : z(7) {}
          y(int val) : z(val) {}
        };
        #pragma hls_top
        int my_package(int a) {
         #pragma hls_array_allow_default_pad
         y x[4] = {{1},{2}};
         return x[2].z;
       })";
  Run({{"a", 5}}, 7, content);
}

// Check that enum array initializers function properly
TEST_F(TranslatorLogicTest, EnumArrayInitializer) {
  std::string_view content = R"(
        typedef enum {
          VAL_1,
          VAL_2
        } enum_val;
        #pragma hls_top
        bool my_package(int a) {
         enum_val x[3] = { VAL_1, VAL_1, VAL_2 };
         return x[2] == VAL_2;
       })";
  Run({{"a", 2}}, static_cast<int64_t>(true), content);
}

TEST_F(TranslatorLogicTest, ChannelTemplateType) {
  std::string_view content = R"(
      #include "/xls_builtin.h"
      #pragma hls_top
      void test(int& in, int& out) {
        const int ctrl = in;
        out = ctrl;
      })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }
  XLS_ASSERT_OK_AND_ASSIGN(xls::TempFile temp,
                           xls::TempFile::CreateWithContent(content, ".cc"));
  XLS_ASSERT_OK(ScanFile(temp));
  package_ = std::make_unique<xls::Package>("my_package");
  ASSERT_THAT(translator_
                  ->GenerateIR_Block(package_.get(), block_spec,
                                     /*top_level_init_interval=*/1)
                  .status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr(
                      "Channel type should be a template specialization")));
}

TEST_F(TranslatorLogicTest, ClassMemberInit) {
  std::string_view content = R"(
    struct Foo {
      int x = 10;
    };
    int my_package(int a) {
      Foo foo;
      foo.x += a;
      return foo.x;
    })";
  Run({{"a", 100}}, 110, content);
}

TEST_F(TranslatorLogicTest, CXXRecordDecl) {
  std::string_view content = R"(
  int my_package(int a) {
   struct foo {
     int bar;
   } foo_struct[4];
   foo_struct[1].bar = a;
   return foo_struct[1].bar;
  })";
  Run({{"a", 100}}, 100, content);
}

std::string GenerateEnumDef(std::vector<std::optional<int64_t>> variants) {
  const std::string src_template = R"(
  enum class MyEnum {
    $0
  };
  #pragma hls_top
  MyEnum my_package() {
    return (MyEnum)0;
  }
  )";
  std::stringstream ss;
  for (size_t i = 0; i < variants.size(); ++i) {
    if (variants[i]) {
      ss << "a" << i << " = " << *variants[i] << "," << '\n';
    } else {
      ss << "a" << i << "," << '\n';
    }
  }
  auto content = absl::Substitute(src_template, ss.str());
  return content;
}

TEST_F(TranslatorLogicTest, EnumSingleZeroVariant) {
  auto content = GenerateEnumDef({0});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 1;
  bool expected_sign = false;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, EnumSinglePositiveVariant) {
  auto content = GenerateEnumDef({7});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 3;
  bool expected_sign = false;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, EnumSingleNegativeVariant) {
  auto content = GenerateEnumDef({-128});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 8;
  bool expected_sign = true;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

// TODO(comc): 32-bit unsigned enums are currently not supported
TEST_F(TranslatorLogicTest, EnumMinNegativeVariant) {
  auto content = GenerateEnumDef({0x80000000});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 32;
  bool expected_sign = true;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, EnumMaxPositiveVariant) {
  auto content = GenerateEnumDef({0x7FFFFFFF});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 31;
  bool expected_sign = false;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, EnumAutoNumbered) {
  auto content = GenerateEnumDef({std::nullopt, std::nullopt, std::nullopt});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 2;
  bool expected_sign = false;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, EnumAutoNumberedAboveZero4Bit) {
  auto content = GenerateEnumDef({0, 14, std::nullopt});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 4;
  bool expected_sign = false;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, EnumAutoNumberedAboveZero5Bit) {
  auto content = GenerateEnumDef({0, 15, std::nullopt});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 5;
  bool expected_sign = false;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, EnumAutoNumberedNegative) {
  auto content = GenerateEnumDef({-16, std::nullopt});
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto meta, translator_->GenerateMetadata());
  int expected_width = 5;
  bool expected_sign = true;
  int actual_width = meta.top_func_proto().return_type().as_enum().width();
  bool actual_sign = meta.top_func_proto().return_type().as_enum().is_signed();
  ASSERT_EQ(expected_width, actual_width);
  ASSERT_EQ(expected_sign, actual_sign);
}

TEST_F(TranslatorLogicTest, TypeAlias) {
  std::string_view content = R"(
    template<class T>
    struct Bar {
      T val;
    };
    bool my_package(long long a) {
      using Foo = Bar<int>;
      Foo v;
      v.val = a;
      return v.val < 1;
    })";

  Run({{"a", -1}}, static_cast<int64_t>(true), content);
  Run({{"a", 2}}, static_cast<int64_t>(false), content);
}

TEST_F(TranslatorLogicTest, AssignmentInInitialization) {
  std::string_view content = R"(
        struct LeafBlock {
          int a;
          int b = a;
        };

       long long my_package() {
        int ret = 1;
        LeafBlock block = {ret = 11};
        return ret + block.a;
       })";
  Run({}, 22, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitialization) {
  std::string_view content = R"(
        struct LeafBlock {
          int a;
          int b = a;
        };

       long long my_package() {
        LeafBlock block = {11};
        return block.b;
       })";
  Run({}, 11, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitialization2) {
  std::string_view content = R"(
        struct LeafBlock {
          int a = 55;
          int b = a;
        };

       long long my_package() {
        LeafBlock block = {12};
        return block.b;
       })";
  Run({}, 12, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitialization3) {
  std::string_view content = R"(
        struct LeafBlock {
          int a = 55;
          int b;
          int c = b;
        };

       long long my_package() {
        LeafBlock block = {.b = 15};
        return block.c;
       })";
  Run({}, 15, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationLValue) {
  std::string_view content = R"(
        struct LeafBlock {
          int& a;
          int& b = a;
        };

       long long my_package() {
        int aa = 10;
        LeafBlock block = {aa};
        aa = 11;
        return block.b;
       })";
  Run({}, 11, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationCtor) {
  std::string_view content = R"(
        struct LeafBlock {
          LeafBlock(int a) : a_(a) {
          }

          int a_;
          int b_ = a_;
        };

       long long my_package() {
        LeafBlock block = {11};
        return block.b_;
       })";
  Run({}, 11, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationCtorLValue) {
  std::string_view content = R"(
        struct LeafBlock {
          LeafBlock(int& a) : a_(a) {
          }

          int& a_;
          int& b_ = a_;
        };

       long long my_package() {
        int aa = 10;
        LeafBlock block = {aa};
        aa = 11;
        return block.b_;
       })";
  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("Don't know how to create")));
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationHierarchical) {
  std::string_view content = R"(
        struct LeafBlock {
          int a;
          int b;
        };

        struct HierBlock {
          LeafBlock x = {.a = 10, .b = 20};
          LeafBlock y = {.a = x.a, .b = 40};
        };

       long long my_package() {
        HierBlock block;
        return block.y.a;
       })";

  Run({}, 10, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationHierarchical2) {
  std::string_view content = R"(
        struct LeafBlock {
          int a;
          int b;
        };

        struct HierBlock1 {
          LeafBlock x = {.a = 10, .b = 20};
          LeafBlock y = {.a = x.a, .b = 40};
        };

        struct HierBlock2 {
          HierBlock1 h;
        };

       long long my_package() {
        HierBlock2 block;
        return block.h.y.a;
       })";
  Run({}, 10, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationHierarchicalLValues) {
  std::string_view content = R"(

        struct LeafBlock {
          int& a;
          int& b = a;
        };

       long long my_package() {
        int a=5;
        LeafBlock block = {a};
        a = 10;
        return block.b;
       })";
  Run({}, 10, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationAssignment) {
  std::string_view content = R"(
        struct LeafBlock {
          int a;
          int b;
        };

        struct HierBlock {
          LeafBlock x = {.a = 10, .b = 20};
          LeafBlock y = {.a = x.b=55};
        };

       long long my_package() {
        HierBlock block;
        return block.y.a;
       })";

  Run({}, 55, content);
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationHierarchicalLValue) {
  std::string_view content = R"(
        struct LeafBlock {
          int& a;
          int& b;
        };

        struct HierBlock {
          int aa = 10;
          int bb = 20;
          LeafBlock y = {.a = aa, .b = bb};
        };

       long long my_package() {
        HierBlock block;
        return block.y.a;
       })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("Tried to access 'this' in a context")));
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationBlockFromClass) {
  std::string_view content = R"(
    struct LeafBlock {
      int a;
      int b;
    };

    struct HierBlock {
      LeafBlock x = {.a = 10, .b = 20};
      LeafBlock y = {.a = x.a, .b = 40};

      __xls_channel<long, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void Run() {
        out.write(y.a);
      }
    };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 1);
  }
}

TEST_F(TranslatorLogicTest, SelfReferencingInitializationBlockFromClass2) {
  std::string_view content = R"(
    struct LeafBlock {
      int a;
      int b;
    };

    struct HierBlock {
      LeafBlock x = {.a = 10, .b = 20};
      LeafBlock y = {.a = x.a, .b = 40};

      __xls_channel<long, __xls_channel_dir_Out>& out;
      __xls_channel<long, __xls_channel_dir_Out>& out2 = out;

      #pragma hls_top
      void Run() {
        out2.write(y.a);
      }
    };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 1);
  }
}

TEST_F(TranslatorLogicTest, SelfReferencingHierarchicalChannelsGenerates) {
  std::string_view content = R"(
    struct LeafBlock {
      __xls_channel<long, __xls_channel_dir_Out> out;
    };

    struct HierBlock {
      __xls_channel<long, __xls_channel_dir_Out> out;

      LeafBlock leaf = {out};

      #pragma hls_top
      void Run() {
        leaf.out.write(10);
      }
    };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  XLS_ASSERT_OK(
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                             /*top_level_init_interval=*/0));
}

TEST_F(TranslatorLogicTest, DoubleSupportToInt) {
  std::string_view content = R"(
    long long my_package() {
      const auto x = 5.5;
      return (int)x;
    })";
  Run({}, 5, content);
}

TEST_F(TranslatorLogicTest, FloatSupportToInt) {
  std::string_view content = R"(
    long long my_package() {
      const auto x = 5.5f;
      return (int)x;
    })";
  Run({}, 5, content);
}

TEST_F(TranslatorLogicTest, FloatConvertToInt) {
  std::string_view content = R"(
    long long convert(float value) {
      return (long long)value;
    }
    long long my_package() {
      return convert(15.0f);
    })";
  Run({}, 15, content);
}

TEST_F(TranslatorLogicTest, ConstexprBinaryOperatorsForDouble) {
  std::string_view content = R"(
    long long my_package() {
      const double x = 5.5 + 1.0;
      return (int)(x * 3);
    })";
  Run({}, 19, content);
}

TEST_F(TranslatorLogicTest, ConstexprBinaryOperatorsForFloat) {
  std::string_view content = R"(
    long long my_package() {
      const float x = 5.5f + 1.0f;
      return (int)(x * 3);
    })";
  Run({}, 19, content);
}

TEST_F(TranslatorLogicTest, PragmaAllowsCommentsBetween) {
  std::string_view content = R"(
  #pragma hls_top
  // some comment
  int st() {
    return 1;
  })";
  Run({}, 1, content);
}

TEST_F(TranslatorLogicTest, PragmaAllowsBlankLines) {
  std::string_view content = R"(
  #pragma hls_top

  int st() {
    return 1;
  })";
  Run({}, 1, content);
}

TEST_F(TranslatorLogicTest, PragmaIgnoresSpaces) {
  std::string_view content = R"(
  #pragma    hls_top
  int st() {
    return 1;
  })";
  Run({}, 1, content);
}

TEST_F(TranslatorLogicTest, PragmaHlsPartialUnrollTreatedAsFullUnroll) {
  std::string_view content = R"(
  #pragma hls_top
  int st() {
    int sum = 0;
    #pragma hls_unroll  4
    for (int i = 0; i < 4; i++) {
      sum += i;
    }
    return sum;
  })";
  Run({}, 0 + 1 + 2 + 3, content);
}

TEST_F(TranslatorLogicTest, CommentedPragmaIgnored) {
  std::string_view content = R"(
  // #pragma    hls_top
  int st() {
    return 1;
  })";
  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(absl::StatusCode::kNotFound,
                             testing::HasSubstr("No top function found")));
}

TEST_F(TranslatorLogicTest, AllowEmptyInitializerList) {
  std::string_view content = R"(
  #pragma hls_top
  int st() {
    int a[20] = {};
    return a[14];
  })";
  Run({}, 0, content);
}

TEST_F(TranslatorLogicTest, UnknownPragmasIgnored) {
  std::string_view content = R"(
  #pragma unknown pragma
  #pragma top

  #pragma hls_top
  int st() {
    return 1;
  })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kOk));
}

TEST_F(TranslatorLogicTest, OnlyUnknownPragmasGiveNoWarnings) {
  std::string_view content = R"(
  #pragma unknown pragma

  #pragma once

  #pragma hls_top
  int st() {
    return 1;
  })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kOk));
  ASSERT_EQ(this->log_entries_.size(), 0);
}

TEST_F(TranslatorLogicTest, OnlyValidPragmasGiveNoWarnings) {
  std::string_view content = R"(
  #pragma hls_top
  int st() {
    #pragma hls_array_allow_default_pad
    int x[4] = {1,2};
    return x[1];
  })";
  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kOk));
  ASSERT_EQ(this->log_entries_.size(), 0);
}

TEST_F(TranslatorLogicTest, MultipleCallsInTree) {
  std::string_view content = R"(
    int xls_int(int x) {
      return x;
    }

    int xls_adjust(int x) {
      return xls_int(x);
    }

    int my_package(int a) {
      return xls_int(xls_adjust(a));
    })";
  Run({{"a", 100}}, 100, content);
}

TEST_F(TranslatorLogicTest, Recursion) {
  std::string_view content = R"(
      int do_something(int a) {
        if(a > 100) {
          return do_something(a / 2);
        }
        return a;
      }
      int my_package(int a) {
        return do_something(a);
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("ecursion")));
}

TEST_F(TranslatorLogicTest, TopRecursion) {
  std::string_view content = R"(
      int my_package(int a) {
        if(a > 100) {
          return my_package(a / 2);
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("ecursion")));
}

TEST_F(TranslatorLogicTest, ErrorOnUninitializedBasic) {
  std::string_view content = R"(
    struct HierBlock {
      __xls_channel<long, __xls_channel_dir_Out> out;

      #pragma hls_top
      void Run() {
        int y;
        (void)y;
        out.write(y);
      }
    };)";

  ASSERT_THAT(ScanFile(content, /*clang_argv=*/{},
                       /*io_test_mode=*/false,
                       /*error_on_init_interval=*/false,
                       /*error_on_uninitialized=*/true),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("uninitialized")));
}

TEST_F(TranslatorLogicTest, ErrorOnUninitializedArray) {
  std::string_view content = R"(
    struct HierBlock {
      __xls_channel<long, __xls_channel_dir_Out> out;

      #pragma hls_top
      void Run() {
        int y[2];
        out.write(y[1]);
      }
    };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false,
                         /*error_on_uninitialized=*/true));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  ASSERT_THAT(translator_
                  ->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                              /*top_level_init_interval=*/0)
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("efault init")));
}

TEST_F(TranslatorLogicTest, ErrorOnUninitializedWrongCount) {
  std::string_view content = R"(
    struct HierBlock {
      __xls_channel<long, __xls_channel_dir_Out> out;

      #pragma hls_top
      void Run() {
        int y[2] = {1};
        out.write(y[1]);
      }
    };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false,
                         /*error_on_uninitialized=*/true));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  ASSERT_THAT(translator_
                  ->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                              /*top_level_init_interval=*/0)
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("number of init")));
}

TEST_F(TranslatorLogicTest, ErrorOnUninitializedWrongCountPragma) {
  std::string_view content = R"(
    struct HierBlock {
      __xls_channel<long, __xls_channel_dir_Out> out;

      #pragma hls_top
      void Run() {
        #pragma hls_array_allow_default_pad
        int y[2] = {1};
        out.write(y[1]);
      }
    };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false,
                         /*error_on_uninitialized=*/true));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  ASSERT_THAT(translator_
                  ->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                              /*top_level_init_interval=*/0)
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kOk));
}

TEST_F(TranslatorLogicTest, ErrorOnUninitializedDefaultMember) {
  std::string_view content = R"(
    struct HierBlock {
      __xls_channel<long, __xls_channel_dir_Out> out;

      int y;

      #pragma hls_top
      void Run() {
        out.write(y);
      }
    };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false,
                         /*error_on_uninitialized=*/true));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  ASSERT_THAT(translator_
                  ->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                              /*top_level_init_interval=*/0)
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("not init")));
}

TEST_F(TranslatorLogicTest, CtorInArray) {
  std::string_view content = R"(
    struct DefaultUnrollInner {
        DefaultUnrollInner() {
            x = 10;
        }
        int x;
    };

    struct DefaultUnroll {
    public:
        DefaultUnrollInner i[4];
    };

    #pragma hls_top
    int foo() {
      DefaultUnroll d;
      return d.i[0].x;
    }
)";
  Run({}, 10, content);
}

TEST_F(TranslatorLogicTest, NoCtorInArray) {
  std::string_view content = R"(
    struct DefaultUnrollInner {
        int x;
    };

    struct DefaultUnroll {
    public:
        DefaultUnrollInner i[4];
    };

    #pragma hls_top
    int foo() {
      DefaultUnroll d;
      return d.i[0].x;
    }
)";
  Run({}, 0, content);
}

TEST_F(TranslatorLogicTest, CtorInMultidimensionalArray) {
  std::string_view content = R"(
    struct DefaultUnrollInner {
        DefaultUnrollInner() {
            x = 10;
        }
        int x;
    };

    struct DefaultUnroll {
    public:
        DefaultUnrollInner i[4][2];
    };

    #pragma hls_top
    int foo() {
      DefaultUnroll d;
      return d.i[0][0].x;
    }
)";
  Run({}, 10, content);
}

TEST_F(TranslatorLogicTest, DelegatingCtorInArray) {
  std::string_view content = R"(
    struct DefaultUnrollInner {
        DefaultUnrollInner(int value) {
          x = value; 
        }
        DefaultUnrollInner(): DefaultUnrollInner(10) {}
        int x;
    };

    struct DefaultUnroll {
    public:
        DefaultUnrollInner i[4];
    };

    #pragma hls_top
    int foo() {
      DefaultUnroll d;
      return d.i[0].x;
    }
)";
  Run({}, 10, content);
}

TEST_F(TranslatorLogicTest, DelegatingCtor) {
  std::string_view content = R"(
    struct DefaultUnrollInner {
        DefaultUnrollInner(int value) {
          x = value;
        }
        DefaultUnrollInner(int value, bool b) : DefaultUnrollInner(value) {}
        DefaultUnrollInner(): DefaultUnrollInner(10, true) {}
        int x;
    };

    #pragma hls_top
    int foo() {
      DefaultUnrollInner d;
      return d.x;
    }
)";
  Run({}, 10, content);
}

TEST_F(TranslatorLogicTest, MixedCtorSettingValueAndBase) {
  std::string_view content = R"(
  class A {
  public:
    int x = 7;
  };
  class B : public A {
  public:
    int f;
    B(A& a) : A(a), f(3) { }
  };

  #pragma hls_top
    int foo() {
      A d;
      B b(d);
      return b.x + b.f;
    }
)";
  Run({}, 10, content);
}

}  // namespace

}  // namespace xlscc
