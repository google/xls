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

#include "xls/contrib/xlscc/translator.h"

#include <cstdio>
#include <memory>
#include <vector>

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/unit_test.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"

using xls::status_testing::IsOkAndHolds;

namespace xlscc {
namespace {

using xls::status_testing::IsOkAndHolds;

class TranslatorTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorTest, IntConst) {
  const std::string content = R"(
    int my_package(int a) {
      return 123;
    })";
  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorTest, LongConst) {
  const std::string content = R"(
      int my_package(int a) {
        return 123L;
      })";

  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorTest, LongLongConst) {
  const std::string content = R"(
      long long my_package(long long a) {
        return 123L;
      })";

  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorTest, LongLongTrueConst) {
  const std::string content = R"(
      long long my_package(long long a) {
        return 123LL;
      })";

  Run({{"a", 100}}, 123, content);
}

TEST_F(TranslatorTest, SyntaxError) {
  const std::string content = R"(
      int my_package(int a) {
        return a+
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("Unable to parse text")));
}

TEST_F(TranslatorTest, Assignment) {
  {
    const std::string content = R"(
        int my_package(int a) {
          a = 5;
          return a;
        })";

    Run({{"a", 1000}}, 5, content);
  }
  {
    const std::string content = R"(
        int my_package(int a) {
          a = 5;
          return a = 10;
        })";

    Run({{"a", 1000}}, 10, content);
  }
}

TEST_F(TranslatorTest, ChainedAssignment) {
  const std::string content = R"(
      int my_package(int a) {
        a += 5;
        a += 10;
        return a;
      })";

  Run({{"a", 1000}}, 1015, content);
}

TEST_F(TranslatorTest, UnsignedChar) {
  const std::string content = R"(
      unsigned char my_package(unsigned char a) {
        return a+5;
      })";

  Run({{"a", 100}}, 105, content);
}

TEST_F(TranslatorTest, Bool) {
  const std::string content = R"(
      int my_package(long long a) {
        return bool(a);
      })";

  Run({{"a", 1000}}, 1, content);
  Run({{"a", 0}}, 0, content);
  Run({{"a", -1}}, 1, content);
}

TEST_F(TranslatorTest, DeclGroup) {
  const std::string content = R"(
      long long my_package(long long a, long long b) {
        long long aa=a, bb=b;
        return aa+bb;
      })";

  Run({{"a", 10}, {"b", 20}}, 30, content);
}

TEST_F(TranslatorTest, Short) {
  const std::string content = R"(
      short my_package(short a, short b) {
        return a+b;
      })";

  Run({{"a", 100}, {"b", 200}}, 300, content);
}

TEST_F(TranslatorTest, UShort) {
  const std::string content = R"(
      unsigned short my_package(unsigned short a, unsigned short b) {
        return a+b;
      })";

  Run({{"a", 100}, {"b", 200}}, 300, content);
}

TEST_F(TranslatorTest, Typedef) {
  const std::string content = R"(
      typedef long long my_int;
      my_int my_package(my_int a) {
        return a*10;
      })";

  Run({{"a", 4}}, 40, content);
}

TEST_F(TranslatorTest, IrAsm) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ArrayParam) {
  const std::string content = R"(
       long long my_package(const long long arr[2]) {
         return arr[0]+arr[1];
       })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_src, SourceToIr(content));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> package,
                           ParsePackage(ir_src));
  std::vector<uint64_t> in_vals = {55, 20};
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value in_arr,
                           xls::Value::UBitsArray(in_vals, 64));
  absl::flat_hash_map<std::string, xls::Value> args;
  args["arr"] = in_arr;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * entry, package->EntryFunction());

  auto x = xls::InterpretFunctionKwargs(
      entry, {{"arr", xls::Value::UBitsArray({55, 20}, 64).value()}});

  ASSERT_THAT(x, IsOkAndHolds(xls::Value(xls::UBits(75, 64))));
}

TEST_F(TranslatorTest, ArraySet) {
  const std::string content = R"(
       long long my_package(long long a, long long b) {
         long long arr[4];
         arr[0] = a;
         arr[1] = b;
         return arr[0]+arr[1];
       })";

  Run({{"a", 11}, {"b", 50}}, 61, content);
}

TEST_F(TranslatorTest, IncrementInArrayIndex1) {
  const std::string content = R"(
       long long my_package(long long a) {
         int arr[4];
         arr[a++] = 5;
         return a;
       })";

  Run({{"a", 11}}, 12, content);
}

TEST_F(TranslatorTest, IncrementInArrayIndex2) {
  const std::string content = R"(
       struct Blah {
         int operator=(int x) {
           return 0;
         }
         int operator[](int idx) {
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

TEST_F(TranslatorTest, Array2D) {
  const std::string content = R"(
       int my_package(int a, int b) {
         int x[2][2] = {{b,b}, {b,b}};
         x[1][0] += a;
         return x[1][0];
       })";

  Run({{"a", 55}, {"b", 100}}, 155, content);
}

TEST_F(TranslatorTest, Array2DInit) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, Array2DClass) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ArrayInitList) {
  const std::string content = R"(
       long long my_package(long long a, long long b) {
         long long arr[2] = {10, 20};
         arr[0] += a;
         arr[1] += b;
         return arr[0]+arr[1];
       })";
  Run({{"a", 11}, {"b", 50}}, 91, content);
}

TEST_F(TranslatorTest, ArrayRefParam) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ArrayInitListWrongSize) {
  const std::string content = R"(
       long long my_package(long long a, long long b) {
         long long arr[4] = {10, 20};
         return a;
       })";
  ASSERT_FALSE(SourceToIr(content).ok());
}

TEST_F(TranslatorTest, ArrayInitLoop) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, StringConstantArray) {
  const std::string content = R"(
       long long my_package(long long a) {
         const char foo[] = "A";
         return a+foo[0];
       })";
  Run({{"a", 11}}, 11 + 'A', content);
}

TEST_F(TranslatorTest, StaticConst) {
  const std::string content = R"(
       long long my_package(long long a) {
         static const int off = 6;
         return a+off;
       })";
  Run({{"a", 11}}, 11 + 6, content);
}

TEST_F(TranslatorTest, GlobalInt) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, GlobalEnum) {
  const std::string content = R"(
       enum BlahE {
         A=2,B,C
       };
       long long my_package(long long a) {
         return a+B+B;
       })";
  Run({{"a", 11}}, 11 + 3 + 3, content);
}

TEST_F(TranslatorTest, SetGlobal) {
  const std::string content = R"(
       int off = 60;
       long long my_package(long long a) {
         off = 5;
         long long ret = a;
         ret += off;
         return ret;
       })";
  ASSERT_FALSE(SourceToIr(content).ok());
}

TEST_F(TranslatorTest, UnsequencedAssign) {
  const std::string content = R"(
      int my_package(int a) {
        return (a=7)+a;
      })";
  auto ret = SourceToIr(content);

  // Clang catches this one and fails parsing
  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                    testing::HasSubstr("parse")));
}

TEST_F(TranslatorTest, UnsequencedRefParam) {
  const std::string content = R"(
      int make7(int &a) {
        return a=7;
      }
      int my_package(int a) {
        return make7(a)+a;
      })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("unsequenced")));
}
TEST_F(TranslatorTest, UnsequencedRefParam2) {
  const std::string content = R"(
      int make7(int &a) {
        return a=7;
      }
      int my_package(int a) {
        return a+make7(a);
      })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("unsequenced")));
}

TEST_F(TranslatorTest, UnsequencedRefParam3) {
  const std::string content = R"(
      int make7(int &a) {
        return a=7;
      }
      int my_package(int a) {
        return make7(a)+a;
      })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("unsequenced")));
}

TEST_F(TranslatorTest, UnsequencedRefParam4) {
  const std::string content = R"(
      int my_package(int a) {
        return (a=7)?a:11;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("unsequenced")));
}
TEST_F(TranslatorTest, UnsequencedRefParam5) {
  const std::string content = R"(
      int my_package(int a) {
        return a?a:(a=7);
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("unsequenced")));
}

// Okay with one parameter
TEST_F(TranslatorTest, AvoidUnsequencedRefParamUnary) {
  const std::string content = R"(
      long long nop(long long a) {
        return a;
      }
      long long my_package(long long a) {
        return -nop(a=10);
      };)";

  Run({{"a", 100}}, -10, content);
}

TEST_F(TranslatorTest, UnsequencedRefParamBinary) {
  const std::string content = R"(
      int nop(int a, int b) {
        return a;
      }
      int my_package(int a) {
        return -nop(a=10, 100);
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("unsequenced")));
}

TEST_F(TranslatorTest, OpAssignmentResult) {
  const std::string content = R"(
      int my_package(int a) {
        return a+=5;
      })";

  Run({{"a", 100}}, 105, content);
}

TEST_F(TranslatorTest, IfStmt) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, IfAssignOverrideCondition) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchStmt) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchConditionalBreak) {
  const std::string content = R"(
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

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("Conditional breaks are not supported")));
}

TEST_F(TranslatorTest, SwitchStmtDefaultTop) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchMultiCaseMultiLine) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchMultiCaseMultiLineBrace) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchDoubleBreak) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchMultiCase) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchReturnStmt) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchDeepFlatten) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchReturnStmt2) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchDefaultPlusCase) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchInFor) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SwitchBreakAfterReturn) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForInSwitch) {
  const std::string content = R"(
       long long my_package(long long a) {
         switch(a) {
           case 0:
             #pragma hls_unroll yes
             for(int i=0;i<3;++i) {
               a+=10;
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

TEST_F(TranslatorTest, ForUnroll) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollClass) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollAssignLoopVar) {
  const std::string content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
          if(a>10)
            ++i;
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("forbidden in this context")));
}

TEST_F(TranslatorTest, ForUnrollNoInit) {
  const std::string content = R"(
      long long my_package(long long a, long long b) {
        int i=1;
        #pragma hls_unroll yes
        for(;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("must have an initializer")));
}

TEST_F(TranslatorTest, ForUnrollNoInc) {
  const std::string content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;i<=10;) {
          a += b;
          a += 2*b;
          ++i;
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("must have an increment")));
}

TEST_F(TranslatorTest, ForUnrollNoCond) {
  const std::string content = R"(
      long long my_package(long long a, long long b) {
        #pragma hls_unroll yes
        for(int i=1;;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("must have a condition")));
}

TEST_F(TranslatorTest, ForUnrollNoPragma) {
  const std::string content = R"(
      long long my_package(long long a, long long b) {
        for(int i=1;i<=10;++i) {
          a += b;
          a += 2*b;
        }
        return a;
      })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                    testing::HasSubstr("Only unrolled")));
}

TEST_F(TranslatorTest, ForNestedUnroll) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollInfinite) {
  const std::string content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=1;i<=10;--i) {
           a += b;
           a += 2*b;
         }
         return a;
       })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("maximum")));
}

TEST_F(TranslatorTest, ForUnrollBreak) {
  const std::string content = R"(
       long long my_package(long long a, long long b) {
         #pragma hls_unroll yes
         for(int i=0;i<50;++i) {
           a += b;
           if(a > 100) break;
         }
         return a;
       })";
  Run({{"a", 11}, {"b", 20}}, 111, content);
}

TEST_F(TranslatorTest, ForUnrollBreak2) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollBreak3) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollBreak4) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollContinue) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollContinue2) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollContinue3) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollContinue4) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ForUnrollContinue5) {
  const std::string content = R"(
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
TEST_F(TranslatorTest, ReturnFromFor) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ReturnFromFor2) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ReturnFromFor3) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ConditionalReturnStmt) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, DoubleReturn) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, TripleReturn) {
  const std::string content = R"(
      long long my_package(long long a, long long b) {
        return 66;
        return 66;
        return a;
      })";

  Run({{"a", 11}, {"b", 0}}, 66, content);
  Run({{"a", 11}, {"b", 3}}, 66, content);
}

TEST_F(TranslatorTest, VoidReturn) {
  const std::string content = R"(
      void my_package(int &a) {
        a = 22;
      })";

  Run({{"a", 1000}}, 22, content);
  Run({{"a", 221}}, 22, content);
}

TEST_F(TranslatorTest, AssignAfterReturn) {
  const std::string content = R"(
      void my_package(int &a) {
        return;
        a = 22;
      })";

  Run({{"a", 1000}}, 1000, content);
}

TEST_F(TranslatorTest, AssignAfterReturnInIf) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, AssignAfterReturn3) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, CapitalizeFirstLetter) {
  const std::string content = R"(
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

  // std::vector<xls::Value> st_elem;
  // st_elem.push_back(xls::Value(xls::UBits(1, 1)));
  auto state =
      xls::Value(xls::Value::TupleOwned({xls::Value(xls::UBits(1, 1))}));

  const char* input = "hello world";
  std::string output = "";
  for (; *input; ++input) {
    const char inc = *input;
    XLS_ASSERT_OK_AND_ASSIGN(xls::Function * entry, package->EntryFunction());
    absl::flat_hash_map<std::string, xls::Value> args;
    args["st"] = state;
    args["c"] = xls::Value(xls::UBits(inc, 8));
    XLS_ASSERT_OK_AND_ASSIGN(xls::Value actual,
                             xls::InterpretFunctionKwargs(entry, args));
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Value> returns,
                             actual.GetElements());
    ASSERT_EQ(returns.size(), 2);
    XLS_ASSERT_OK_AND_ASSIGN(char outc, returns[0].bits().ToUint64());

    state = returns[1];
    output += outc;
  }

  ASSERT_EQ(output, "Hello World");
}

TEST_F(TranslatorTest, AssignmentInBlock) {
  const std::string content = R"(
      int my_package(int a) {
        int r = a;
        {
          r = 55;
        }
        return r;
      })";

  Run({{"a", 100}}, 55, content);
}

TEST_F(TranslatorTest, AssignmentInParens) {
  const std::string content = R"(
      int my_package(int a) {
        int r = a;
        (r) = 55;
        return r;
      })";

  Run({{"a", 100}}, 55, content);
}

TEST_F(TranslatorTest, ShadowAssigment) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, CompoundStructAccess) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, StaticStructAccess) {
  const std::string content = R"(
       struct TestX {
         static const int x = 50;
       };
       int my_package() {
         TestX y;
         return y.x;
       })";
  Run({}, 50, content);
}

TEST_F(TranslatorTest, SubstTemplateType) {
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

TEST_F(TranslatorTest, TemplateStruct) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ArrayOfStructsAccess) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, StructWithArrayAccess) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, NoTupleStruct) {
  const std::string content = R"(
       #pragma hls_no_tuple
       struct Test {
         int x;
       };
       Test my_package(int a) {
         Test s;
         s.x=a;
         return s;
       })";
  Run({{"a", 311}}, 311, content);
}

TEST_F(TranslatorTest, NoTupleMultiField) {
  const std::string content = R"(
       #pragma hls_no_tuple
       struct Test {
         int x;
         int y;
       };
       Test my_package(int a) {
         Test s;
         s.x=a;
         return s;
       })";
  auto ret = SourceToIr(content);

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                    testing::HasSubstr("only 1 field")));
}

TEST_F(TranslatorTest, NoTupleMultiFieldLineComment) {
  const std::string content = R"(
       //#pragma hls_no_tuple
       struct Test {
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         s.x=a;
         return s.x;
       })";
  Run({{"a", 311}}, 311, content);
}

TEST_F(TranslatorTest, NoTupleMultiFieldBlockComment) {
  const std::string content = R"(
       /*
       #pragma hls_no_tuple*/
       struct Test {
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         s.x=a;
         return s.x;
       })";
  Run({{"a", 311}}, 311, content);
}

TEST_F(TranslatorTest, StructMemberOrder) {
  const std::string content = R"(
       struct Test {
         int x;
         int y;
       };
       Test my_package(int a, int b) {
         Test s;
         s.x=a;
         s.y=b;
         return s;
       })";

  xls::Value tuple_values[2] = {xls::Value(xls::SBits(50, 32)),
                                xls::Value(xls::SBits(311, 32))};
  xls::Value expected = xls::Value::Tuple(tuple_values);

  absl::flat_hash_map<std::string, xls::Value> args = {
      {"a", xls::Value(xls::SBits(311, 32))},
      {"b", xls::Value(xls::SBits(50, 32))}};

  Run(args, expected, content);
}

TEST_F(TranslatorTest, ImplicitConversion) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, OperatorOverload) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, OperatorOnBuiltin) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, UnaryOperatorAvoidUnsequencedError2) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, UnaryOperatorAvoidUnsequencedError3) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, TypedefStruct) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ConvertToVoid) {
  const std::string content = R"(
       struct ts {int x;};
       long long my_package(long long a) {
         ts t;
         (void)t;
         return a;
       })";
  Run({{"a", 10}}, 10, content);
}

TEST_F(TranslatorTest, AvoidDoubleAssignmentFromBackwardsEval) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, CompoundAvoidUnsequenced) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, CompoundAvoidUnsequenced2) {
  const std::string content = R"(
       int my_package(int a) {
         int s1[2] = {a, a};
         s1[0] = ++s1[1];
         return s1[0];
       })";
  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorTest, DefaultValues) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, StructMemberReferenceParameter) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, AnonStruct) {
  const std::string content = R"(
       int my_package(int a) {
         struct {
           int x;
           int y;
         } s;
         s.x = a;
         s.y = a*10;
         return s.x+s.y;
       })";
  // Not implemented, expect graceful failure
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("DeclStmt other than Var")));
}

TEST_F(TranslatorTest, Inheritance) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, BaseConstructor) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, BaseConstructorNoTuple) {
  const std::string content = R"(
       #pragma hls_no_tuple
       struct Base {
         Base() : x(88) { }
          int x;
       };
       #pragma hls_no_tuple
       struct Derived : public Base {
       };
       int my_package(int x) {
         Derived b;
         return x + b.x;
       })";
  Run({{"x", 15}}, 103, content);
}

TEST_F(TranslatorTest, InheritanceNoTuple) {
  const std::string content = R"(
       struct Base {
         int x;
       };
       #pragma hls_no_tuple
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

TEST_F(TranslatorTest, InheritanceNoTuple2) {
  const std::string content = R"(
       #pragma hls_no_tuple
       struct Base {
         int x;
       };
       #pragma hls_no_tuple
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

TEST_F(TranslatorTest, InheritanceNoTuple4) {
  const std::string content = R"(
       #pragma hls_no_tuple
       struct Base {
         int x;
         void set(int v) { x=v; }
         int get()const { return x; }
       };
       #pragma hls_no_tuple
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

TEST_F(TranslatorTest, InheritanceTuple) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, Constructor) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ConstructorWithArg) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ConstructorWithThis) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SetThis) {
  const std::string content = R"(
       struct Test {
         void set_this(int v) {
           Test t;
           t.x = v;
           *this = t;
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         s.set_this(a);
         s.y = 12;
         return s.x+s.y;
       })";
  Run({{"a", 3}}, 15, content);
}

TEST_F(TranslatorTest, ThisCall) {
  const std::string content = R"(
       struct Test {
         void set_this(int v) {
           Test t;
           t.x = v;
           *this = t;
         }
         void set_this_b(int v) {
           set_this(v);
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         s.set_this_b(a);
         s.y = 12;
         return s.x+s.y;
       })";
  Run({{"a", 3}}, 15, content);
}

TEST_F(TranslatorTest, ExplicitDefaultConstructor) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ConditionallyAssignThis) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SetMemberInnerContext) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, StaticMethod) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, SignExtend) {
  {
    const std::string content = R"(
        unsigned long long my_package(long long a) {
          return long(a);
        })";

    Run({{"a", 3}}, 3, content);
  }
  {
    const std::string content = R"(
        long long my_package(long long a) {
          return (unsigned long)a;
        })";

    Run({{"a", 3}}, 3, content);
    Run({{"a", -3}}, 18446744073709551613ull, content);
  }
}

TEST_F(TranslatorTest, TopFunctionByName) {
  const std::string content = R"(
      int my_package(int a) {
        return a + 1;
      })";

  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorTest, TopFunctionPragma) {
  const std::string content = R"(
      #pragma hls_top
      int asdf(int a) {
        return a + 1;
      })";

  Run({{"a", 3}}, 4, content);
}

TEST_F(TranslatorTest, TopFunctionNoPragma) {
  const std::string content = R"(
      int asdf(int a) {
        return a + 1;
      })";
  auto ret = SourceToIr(content);
  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kNotFound,
                  testing::HasSubstr("No top function found")));
}

TEST_F(TranslatorTest, Function) {
  const std::string content = R"(
      int do_something(int a) {
        return a;
      }
      int my_package(int a) {
        return do_something(a);
      })";

  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorTest, DefaultArg) {
  const std::string content = R"(
      int do_something(int a, int b=2) {
        return a+b;
      }
      int my_package(int a) {
        return do_something(a);
      })";

  Run({{"a", 3}}, 5, content);
}

TEST_F(TranslatorTest, FunctionInline) {
  const std::string content = R"(
      inline int do_something(int a) {
        return a;
      }
      int my_package(int a) {
        return do_something(a);
      })";

  Run({{"a", 3}}, 3, content);
}

TEST_F(TranslatorTest, TemplateFunction) {
  const std::string content = R"(
      template<int N>
      int do_something(int a) {
        return a+N;
      }
      int my_package(int a) {
        return do_something<5>(a);
      })";

  Run({{"a", 3}}, 8, content);
}

TEST_F(TranslatorTest, TemplateFunctionBool) {
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

TEST_F(TranslatorTest, ReferenceParameter) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, ConstReferenceParameter) {
  const std::string content = R"(
      int my_package(const int &a) {
        return a + 10;
      })";

  Run({{"a", 3}}, 3 + 10, content);
}

TEST_F(TranslatorTest, Namespace) {
  const std::string content = R"(
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

TEST_F(TranslatorTest, NamespaceFailure) {
  const std::string content = R"(
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
              xls::status_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("Unable to parse text")));
}

TEST_F(TranslatorTest, Ternary) {
  const std::string content = R"(
      int my_package(int a) {
        return a ? a : 11;
      })";

  Run({{"a", 3}}, 3, content);
  Run({{"a", 0}}, 11, content);
}

// This is here mainly to check for graceful exit with no memory leaks
TEST_F(TranslatorTest, ParseFailure) {
  const std::string content = "int my_package(int a) {";
  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("Unable to parse text")));
}

TEST_F(TranslatorTest, IO) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         out.write(3*in.read());
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorTest, IOUnsequencedCheck) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         out.write(3*in.read()*2);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 30, true)});
}

TEST_F(TranslatorTest, IOMulti) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(int sel,
                       __xls_channel<int>& in,
                       __xls_channel<int>& out1,
                       __xls_channel<int>& out2) {
         const int x = in.read();
         if(sel) {
           out1.write(3*x);
         } else {
           out2.write(7*x);
         }
       })";

  {
    absl::flat_hash_map<std::string, xls::Value> args;
    args["sel"] = xls::Value(xls::UBits(1, 32));
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 5, true)},
           /*outputs=*/
           {IOOpTest("out1", 15, true), IOOpTest("out2", 0, false)}, args);
  }
  {
    absl::flat_hash_map<std::string, xls::Value> args;
    args["sel"] = xls::Value(xls::UBits(0, 32));
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 5, true)},
           /*outputs=*/
           {IOOpTest("out1", 0, false), IOOpTest("out2", 35, true)}, args);
  }
}

TEST_F(TranslatorTest, IOWriteConditional) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int x = in.read();
         if(x>10) {
           out.write(5*x);
         }
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 0, false)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 20, true)},
         /*outputs=*/{IOOpTest("out", 100, true)});
}

TEST_F(TranslatorTest, IOReadConditional) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int x = in.read();
         if(x < 8) {
           x += in.read();
         }
         out.write(x);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 10, true), IOOpTest("in", 0, false)},
         /*outputs=*/{IOOpTest("out", 10, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true), IOOpTest("in", 2, true)},
         /*outputs=*/{IOOpTest("out", 3, true)});
}

TEST_F(TranslatorTest, IOSubroutine) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int sub_recv(__xls_channel<int>& in) {
         return in.read();
       }
       void sub_send(int v, __xls_channel<int>& out) {
         out.write(v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         sub_send(7 + sub_recv(in), out);
         out.write(55);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 5 + 7, true), IOOpTest("out", 55, true)});
}

TEST_F(TranslatorTest, IOMethodSubroutine) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         int sub_recv(__xls_channel<int>& in) {
           return in.read();
         }
         void sub_send(int v, __xls_channel<int>& out) {
           out.write(v);
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Foo f;
         f.sub_send(7 + f.sub_recv(in), out);
         out.write(55);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 5 + 7, true), IOOpTest("out", 55, true)});
}

TEST_F(TranslatorTest, IOOperatorSubroutine) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         int operator+=(__xls_channel<int>& in) {
           return in.read();
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Foo f;
         out.write(f += in);
       })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("IO ops in operator calls are not supported")));
}

TEST_F(TranslatorTest, IOSaveChannel) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {

         __xls_channel<int>& out_(out);

         out_.write(in.read());
       })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("IO ops should be on channel parameters")));
}

TEST_F(TranslatorTest, IOSaveChannelStruct) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         __xls_channel<int>& out_;

         Foo(__xls_channel<int>& out) : out_(out) {
         }

         int sub_recv(__xls_channel<int>& in) {
           return in.read();
         }
         void sub_send(int v) {
           out_.write(v);
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Foo f(out);
         f.sub_send(7 + f.sub_recv(in));
       })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("IO ops should be on direct DeclRefs")));
}

TEST_F(TranslatorTest, IOUnrolled) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& out) {
         #pragma hls_unroll yes
         for(int i=0;i<4;++i) {
           out.write(i);
         }
       })";

  IOTest(content, {},
         {/*inputs=*/IOOpTest("out", 0, true), IOOpTest("out", 1, true),
          /*outputs=*/IOOpTest("out", 2, true), IOOpTest("out", 3, true)});
}

TEST_F(TranslatorTest, IOUnrolledUnsequenced) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int ret = 0;
         #pragma hls_unroll yes
         for(int i=0;i<3;++i) {
           ret += 2*in.read();
         }
         out.write(ret);
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", 10, true), IOOpTest("in", 20, true),
          IOOpTest("in", 100, true)},
         /*outputs=*/{IOOpTest("out", 260, true)});
}

TEST_F(TranslatorTest, IOProcMux) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(int& dir,
              __xls_channel<int>& in,
              __xls_channel<int>& out1,
              __xls_channel<int> &out2) {


      const int ctrl = in.read();

      if (dir == 0) {
        out1.write(ctrl);
      } else {
        out2.write(ctrl);
      }
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(DIRECT_IN);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out1");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out2");
    ch_out2->set_is_input(false);
    ch_out2->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::vector<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out1"] = {xls::Value(xls::SBits(55, 32))};
    outputs["out2"] = {};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    inputs["dir"] = {xls::Value(xls::SBits(1, 32))};

    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out1"] = {};
    outputs["out2"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorTest, IOProcMuxMethod) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    class Foo {
      #pragma hls_top
      void foo(int& dir,
                __xls_channel<int>& in,
                __xls_channel<int>& out1,
                __xls_channel<int> &out2) {


        const int ctrl = in.read();

        if (dir == 0) {
          out1.write(ctrl);
        } else {
          out2.write(ctrl);
        }
      }
    };)";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(DIRECT_IN);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out1");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out2");
    ch_out2->set_is_input(false);
    ch_out2->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::vector<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out1"] = {xls::Value(xls::SBits(55, 32))};
    outputs["out2"] = {};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorTest, IOProcMuxConstDir) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(const int dir,
              __xls_channel<int>& in,
              __xls_channel<int>& out1,
              __xls_channel<int> &out2) {


      const int ctrl = in.read();

      if (dir == 0) {
        out1.write(ctrl);
      } else {
        out2.write(ctrl);
      }
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(DIRECT_IN);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out1");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out2");
    ch_out2->set_is_input(false);
    ch_out2->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::vector<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out1"] = {xls::Value(xls::SBits(55, 32))};
    outputs["out2"] = {};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    inputs["dir"] = {xls::Value(xls::SBits(1, 32))};

    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out1"] = {};
    outputs["out2"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorTest, IOProcChainedConditionalRead) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int x = in.read();

      out.write(x);

      if(x < 50) {
        x += in.read();
        if(x > 100) {
          out.write(x);
        }
      }
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(FIFO);
  }

  {
    absl::flat_hash_map<std::string, std::vector<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(55, 32))};

    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
  {
    absl::flat_hash_map<std::string, std::vector<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(40, 32)),
                    xls::Value(xls::SBits(10, 32))};

    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(40, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
  {
    absl::flat_hash_map<std::string, std::vector<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(40, 32)),
                    xls::Value(xls::SBits(65, 32))};

    absl::flat_hash_map<std::string, std::vector<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(40, 32)),
                      xls::Value(xls::SBits(105, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

std::string NativeOperatorTestIr(std::string op) {
  return absl::StrFormat(R"(
      long long my_package(long long a, long long b) {
        return a %s b;
      })",
                         op);
}

std::string NativeOperatorTestIrEq(std::string op) {
  return absl::StrFormat(R"(
      long long my_package(long long a, long long b) {
        a %s= b;
        return a;
      })",
                         op);
}

TEST_F(TranslatorTest, NativeOperatorAdd) {
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

TEST_F(TranslatorTest, NativeOperatorSub) {
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

TEST_F(TranslatorTest, NativeOperatorMul) {
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

TEST_F(TranslatorTest, NativeOperatorDiv) {
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

TEST_F(TranslatorTest, NativeOperatorRem) {
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

TEST_F(TranslatorTest, NativeOperatorAnd) {
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

TEST_F(TranslatorTest, NativeOperatorOr) {
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

TEST_F(TranslatorTest, NativeOperatorXor) {
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

TEST_F(TranslatorTest, NativeOperatorNot) {
  const std::string content = R"(
      long long my_package(unsigned long long a) {
        return (long long)(~a);
      })";

  Run({{"a", 0b000}}, ~static_cast<uint64_t>(0b000), content);
  Run({{"a", 0b111}}, ~static_cast<uint64_t>(0b111), content);
  Run({{"a", 0b101}}, ~static_cast<uint64_t>(0b101), content);
}

TEST_F(TranslatorTest, NativeOperatorNeg) {
  const std::string content = R"(
      long long my_package(long long a) {
        return (long long)(-a);
      })";

  Run({{"a", 11}}, -11, content);
  Run({{"a", 0}}, 0, content);
  Run({{"a", -1000}}, 1000, content);
}

TEST_F(TranslatorTest, NativeOperatorShrSigned) {
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
TEST_F(TranslatorTest, NativeOperatorShrUnsigned) {
  const std::string content = R"(
      unsigned long long my_package(unsigned long long a, unsigned long long b)
      {
        return a >> b;
      })";
  { Run({{"a", 10}, {"b", 1}}, 5, content); }
  { Run({{"a", -20}, {"b", 2}}, 4611686018427387899L, content); }
}
TEST_F(TranslatorTest, NativeOperatorShl) {
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
TEST_F(TranslatorTest, NativeOperatorPreInc) {
  {
    const std::string content = R"(
        int my_package(int a) {
          return ++a;
        })";

    Run({{"a", 10}}, 11, content);
  }
  {
    const std::string content = R"(
        int my_package(int a) {
          ++a;
          return a;
        })";

    Run({{"a", 50}}, 51, content);
  }
}
TEST_F(TranslatorTest, NativeOperatorPostInc) {
  {
    const std::string content = R"(
        int my_package(int a) {
          return a++;
        })";

    Run({{"a", 10}}, 10, content);
  }
  {
    const std::string content = R"(
        int my_package(int a) {
          a++;
          return a;
        })";

    Run({{"a", 50}}, 51, content);
  }
}
TEST_F(TranslatorTest, NativeOperatorPreDec) {
  {
    const std::string content = R"(
        int my_package(int a) {
          return --a;
        })";

    Run({{"a", 10}}, 9, content);
  }
  {
    const std::string content = R"(
        int my_package(int a) {
          --a;
          return a;
        })";

    Run({{"a", 50}}, 49, content);
  }
}
TEST_F(TranslatorTest, NativeOperatorPostDec) {
  {
    const std::string content = R"(
        int my_package(int a) {
          return a--;
        })";

    Run({{"a", 10}}, 10, content);
  }
  {
    const std::string content = R"(
        int my_package(int a) {
          a--;
          return a;
        })";

    Run({{"a", 50}}, 49, content);
  }
}

std::string NativeBoolOperatorTestIr(std::string op) {
  return absl::StrFormat(R"(
      long long my_package(long long a, long long b) {
        return (long long)(a %s b);
      })",
                         op);
}

std::string NativeUnsignedBoolOperatorTestIr(std::string op) {
  return absl::StrFormat(R"(
      long long my_package(unsigned long long a, unsigned long long b) {
        return (long long)(a %s b);
      })",
                         op);
}

TEST_F(TranslatorTest, NativeOperatorEq) {
  const std::string op = "==";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorTest, NativeOperatorNe) {
  const std::string op = "!=";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorTest, NativeOperatorGt) {
  const std::string op = ">";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorTest, NativeOperatorGtU) {
  const std::string op = ">";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorTest, NativeOperatorGte) {
  const std::string op = ">=";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorTest, NativeOperatorGteU) {
  const std::string op = ">=";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 0, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 1, content);
}

TEST_F(TranslatorTest, NativeOperatorLt) {
  const std::string op = "<";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorTest, NativeOperatorLtU) {
  const std::string op = "<";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 0, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorTest, NativeOperatorLte) {
  const std::string op = "<=";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 1, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorTest, NativeOperatorLteU) {
  const std::string op = "<=";
  const std::string content = NativeUnsignedBoolOperatorTestIr(op);

  Run({{"a", -2}, {"b", 3}}, 0, content);
  Run({{"a", 2}, {"b", 3}}, 1, content);
  Run({{"a", 3}, {"b", 3}}, 1, content);
  Run({{"a", 11}, {"b", 10}}, 0, content);
}

TEST_F(TranslatorTest, NativeOperatorLAnd) {
  const std::string op = "&&";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 0b111}, {"b", 0b111}}, 1, content);
  Run({{"a", 0b001}, {"b", 0b100}}, 1, content);
  Run({{"a", 0b111}, {"b", 0}}, 0, content);
  Run({{"a", 0}, {"b", 0}}, 0, content);
}

TEST_F(TranslatorTest, NativeOperatorLOr) {
  const std::string op = "||";
  const std::string content = NativeBoolOperatorTestIr(op);

  Run({{"a", 0b111}, {"b", 0b111}}, 1, content);
  Run({{"a", 0b001}, {"b", 0b100}}, 1, content);
  Run({{"a", 0b111}, {"b", 0}}, 1, content);
  Run({{"a", 0}, {"b", 0}}, 0, content);
}

TEST_F(TranslatorTest, NativeOperatorLNot) {
  const std::string content = R"(
      long long my_package(unsigned long long a) {
        return (long long)(!a);
      })";

  Run({{"a", 0}}, 1, content);
  Run({{"a", 11}}, 0, content);
  Run({{"a", -11}}, 0, content);
}

TEST_F(TranslatorTest, MetadataNamespaceStructArray) {
  const std::string content = R"(
    namespace foo {
      struct Blah {
        int aa;
      };
      #pragma hls_top
      Blah i_am_top(short a, short b[2]) {
        Blah x;
        x.aa = a+b[1];
        return x;
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
              id: 0
            }
          }
        }
        fields {
          name: "aa"
          type {
            as_int {
              width: 32
              is_signed: true
            }
          }
        }
        no_tuple: false
      }
    }
    top_func_proto {
      name {
        name: "i_am_top"
        fully_qualified_name: "foo::i_am_top"
        id: 22180269196936
      }
      return_type {
        as_inst {
          name {
            name: "Blah"
            fully_qualified_name: "foo::Blah"
            id: 0
          }
        }
      }
      params {
        name: "a"
        type {
          as_int {
            width: 16
            is_signed: true
          }
        }
        is_reference: false
        is_const: false
      }
      params {
        name: "b"
        type {
          as_array {
            element_type {
              as_int {
                width: 16
                is_signed: true
              }
            }
            size: 2
          }
        }
        is_reference: true
        is_const: false
      }
    })";

  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);

  // id varies from run to run
  ASSERT_NE(meta.top_func_proto().name().id(), 0);
  meta.mutable_top_func_proto()->mutable_name()->set_id(22180269196936L);

  ASSERT_EQ(1, meta.structs_size());

  ASSERT_EQ(meta.top_func_proto().return_type().as_inst().name().id(),
            meta.structs(0).as_struct().name().as_inst().name().id());

  meta.mutable_top_func_proto()
      ->mutable_return_type()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(0)
      ->mutable_as_struct()
      ->mutable_name()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);

  bool did_equal = google::protobuf::util::MessageDifferencer::Equals(meta, ref_meta);

  if (!did_equal) {
    fprintf(stderr, "%s\n", meta.DebugString().c_str());
  }

  ASSERT_TRUE(google::protobuf::util::MessageDifferencer::Equals(meta, ref_meta));
}

TEST_F(TranslatorTest, MetadataNamespaceNestedStruct) {
  const std::string content = R"(
    namespace foo {
      struct Blah {
        int aa;
        struct Something {
          int bb;
        }s;
      };
      #pragma hls_top
      short i_am_top(Blah a, short b[2]) {
        Blah x;
        x.s.bb = b[0];
        x.aa = a.aa+b[1];
        x.aa += x.s.bb;
        return x.aa;
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
              id: 0
            }
          }
        }
        fields {
          name: "aa"
          type {
            as_int {
              width: 32
              is_signed: true
            }
          }
        }
        fields {
          name: "s"
          type {
            as_inst {
              name {
                name: "Something"
                fully_qualified_name: "foo::Blah::Something"
                id: 0
              }
            }
          }
        }
        no_tuple: false
      }
    }
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Something"
              fully_qualified_name: "foo::Blah::Something"
              id: 0
            }
          }
        }
        fields {
          name: "bb"
          type {
            as_int {
              width: 32
              is_signed: true
            }
          }
        }
        no_tuple: false
      }
    }
    top_func_proto {
      name {
        name: "i_am_top"
        fully_qualified_name: "foo::i_am_top"
        id: 22180269196936
      }
      return_type {
        as_int {
          width: 16
          is_signed: true
        }
      }
      params {
        name: "a"
        type {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
              id: 0
            }
          }
        }
        is_reference: false
        is_const: false
      }
      params {
        name: "b"
        type {
          as_array {
            element_type {
              as_int {
                width: 16
                is_signed: true
              }
            }
            size: 2
          }
        }
        is_reference: true
        is_const: false
      }
    })";

  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);

  // id varies from run to run
  ASSERT_NE(meta.top_func_proto().name().id(), 0);
  meta.mutable_top_func_proto()->mutable_name()->set_id(22180269196936L);

  ASSERT_EQ(2, meta.structs_size());

  const int subsidx = 1, topsidx = 0;

  // Order of structs is not deterministic, avoid protobuf equals failures
  if (meta.structs(0).as_struct().name().as_inst().name().name() ==
      "Something") {
    xlscc_metadata::Type top_struct = meta.structs(1);
    xlscc_metadata::Type sub_struct = meta.structs(0);
    *meta.mutable_structs(0) = top_struct;
    *meta.mutable_structs(1) = sub_struct;
  }

  ASSERT_EQ(1, meta.structs(subsidx).as_struct().fields_size());
  ASSERT_EQ(2, meta.structs(topsidx).as_struct().fields_size());

  ASSERT_EQ(meta.top_func_proto().params(0).type().as_inst().name().id(),
            meta.structs(topsidx).as_struct().name().as_inst().name().id());

  ASSERT_EQ(
      meta.structs(topsidx).as_struct().fields(1).type().as_inst().name().id(),
      meta.structs(subsidx).as_struct().name().as_inst().name().id());

  meta.mutable_top_func_proto()
      ->mutable_params(0)
      ->mutable_type()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(topsidx)
      ->mutable_as_struct()
      ->mutable_name()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(topsidx)
      ->mutable_as_struct()
      ->mutable_fields(1)
      ->mutable_type()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(subsidx)
      ->mutable_as_struct()
      ->mutable_name()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);

  bool did_equal = google::protobuf::util::MessageDifferencer::Equals(meta, ref_meta);

  if (!did_equal) {
    fprintf(stderr, "%s\n", meta.DebugString().c_str());
  }

  ASSERT_TRUE(google::protobuf::util::MessageDifferencer::Equals(meta, ref_meta));
}

TEST_F(TranslatorTest, MetadataRefConstParams) {
  const std::string content = R"(
    #pragma hls_top
    void i_am_top(const short &a, short &b) {
      b += a;
    })";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    top_func_proto {
      name {
        name: "i_am_top"
        fully_qualified_name: "i_am_top"
        id: 22078263808792
      }
      return_type {
        as_void {
        }
      }
      params {
        name: "a"
        type {
          as_int {
            width: 16
            is_signed: true
          }
        }
        is_reference: true
        is_const: true
      }
      params {
        name: "b"
        type {
          as_int {
            width: 16
            is_signed: true
          }
        }
        is_reference: true
        is_const: false
      }
    })";

  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);

  // id varies from run to run
  ASSERT_NE(meta.top_func_proto().name().id(), 0);
  meta.mutable_top_func_proto()->mutable_name()->set_id(22078263808792L);

  bool did_equal = google::protobuf::util::MessageDifferencer::Equals(meta, ref_meta);

  if (!did_equal) {
    fprintf(stderr, "%s\n", meta.DebugString().c_str());
  }

  ASSERT_TRUE(did_equal);
}

}  // namespace

}  // namespace xlscc
