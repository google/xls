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
#include <list>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xlscc {
namespace {

class TranslatorIOTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorIOTest, Basic) {
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

TEST_F(TranslatorIOTest, ReadToParam) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int v = 0;
         in.read(v);
         out.write(3*v);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, NonblockingRead) {
  const std::string content = R"(
    #include "/xls_builtin.h"
    #pragma hls_top
    void Run(__xls_channel<int>& in,
             __xls_channel<int>& out) {
	    int val = 1;
	    bool r = in.nb_read(val);
	    out.write(r ? val*3 : val);
    })";

  xls::Value value_in = xls::Value::Tuple(
      {xls::Value(xls::SBits(5, 32)), xls::Value::Bool(true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", value_in, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});

  xls::Value value_not_ready = xls::Value::Tuple(
      {xls::Value(xls::SBits(5, 32)), xls::Value::Bool(false)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", value_not_ready, true)},
         /*outputs=*/{IOOpTest("out", 1, true)});
}

TEST_F(TranslatorIOTest, NonblockingReadInSubroutine) {
  const std::string content = R"(
    #include "/xls_builtin.h"
    int GetNbValue(__xls_channel<int>& in) {
	    int val = 1;
	    bool r = in.nb_read(val);
      return r ? val*3 : val;
    }
    #pragma hls_top
    void Run(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int v = GetNbValue(in);
	    out.write(v);
    })";

  xls::Value value_in = xls::Value::Tuple(
      {xls::Value(xls::SBits(5, 32)), xls::Value::Bool(true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", value_in, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});

  xls::Value value_not_ready = xls::Value::Tuple(
      {xls::Value(xls::SBits(5, 32)), xls::Value::Bool(false)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", value_not_ready, true)},
         /*outputs=*/{IOOpTest("out", 1, true)});
}

TEST_F(TranslatorIOTest, ProcNonblockingRead) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int val = 1;
	    bool r = in.nb_read(val);
	    out.write(r ? val*3 : val);
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
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(5, 32))};
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(15, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {};
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(1, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorIOTest, NonblockingReadStruct) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Test {
         int x;
         int foo() const {
           return x;
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<Test>& in,
                       __xls_channel<int>& out) {
         Test v;
         const bool read = in.nb_read(v);
         if(read) {
           v.x = v.x * 5;
           out.write(v.foo());
         }
       })";
  xls::Value value_in =
      xls::Value::Tuple({xls::Value::Tuple({xls::Value(xls::SBits(5, 32))}),
                         xls::Value::Bool(true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", value_in, true)},
         /*outputs=*/{IOOpTest("out", 25, true)});

  xls::Value value_not_ready =
      xls::Value::Tuple({xls::Value::Tuple({xls::Value(xls::SBits(5, 32))}),
                         xls::Value::Bool(false)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", value_not_ready, true)},
         /*outputs=*/{IOOpTest("out", 0, false)});
}

TEST_F(TranslatorIOTest, UnsequencedCheck) {
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

TEST_F(TranslatorIOTest, Multi) {
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

TEST_F(TranslatorIOTest, WriteConditional) {
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

TEST_F(TranslatorIOTest, ReadConditional) {
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

TEST_F(TranslatorIOTest, TernaryReadConditional) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int x = in.read();
         x += (x < 8) ? in.read() : 0;
         out.write(x);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 10, true), IOOpTest("in", 0, false)},
         /*outputs=*/{IOOpTest("out", 10, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true), IOOpTest("in", 2, true)},
         /*outputs=*/{IOOpTest("out", 3, true)});
}

TEST_F(TranslatorIOTest, TernaryReadConditional2) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int x = in.read();
         x += (x >= 8) ? 0 : in.read();
         out.write(x);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 10, true), IOOpTest("in", 0, false)},
         /*outputs=*/{IOOpTest("out", 10, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true), IOOpTest("in", 2, true)},
         /*outputs=*/{IOOpTest("out", 3, true)});
}

TEST_F(TranslatorIOTest, Subroutine) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int sub_recv(__xls_channel<int>& in_sub, int &v) {
         return in_sub.read() - v;
       }
       void sub_send(int v, __xls_channel<int>& out) {
         out.write(v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int z = 1;
         sub_send(7 + sub_recv(in, z), out);
         out.write(55);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/
         {IOOpTest("out", 5 + 7 - 1, true), IOOpTest("out", 55, true)});
}

TEST_F(TranslatorIOTest, SubroutineDeclOrder) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int sub_recv(__xls_channel<int>& in, int &v);
       void sub_send(int v, __xls_channel<int>& outs) {
         outs.write(v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int z = 1;
         sub_send(7 + sub_recv(in, z), out);
         out.write(55);
       }
       int sub_recv(__xls_channel<int>& in, int &v) {
         return in.read() - v;
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/
         {IOOpTest("out", 5 + 7 - 1, true), IOOpTest("out", 55, true)});
}

TEST_F(TranslatorIOTest, SubroutineDeclMissing) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int sub_recv(__xls_channel<int>& in, int &v);
       void sub_send(int v, __xls_channel<int>& out) {
         out.write(v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int z = 1;
         sub_send(7 + sub_recv(in, z), out);
         out.write(55);
       })";

  ASSERT_THAT(SourceToIr(content, /*pfunc=*/nullptr, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kNotFound,
                  testing::HasSubstr("sub_recv used but has no body")));
}

TEST_F(TranslatorIOTest, Subroutine2) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int sub_recv(__xls_channel<int>& in, int &v) {
         return in.read() - v;
       }
       void sub_send(int v, __xls_channel<int>& out) {
         out.write(v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int z = 1;
         sub_send(7 + sub_recv(in, z), out);
         sub_send(5, out);
         out.write(55);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/
         {IOOpTest("out", 5 + 7 - 1, true), IOOpTest("out", 5, true),
          IOOpTest("out", 55, true)});
}

TEST_F(TranslatorIOTest, Subroutine3) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int sub_recv(__xls_channel<int>& in, int &v) {
         return in.read() - v;
       }
       void sub_send(int v, __xls_channel<int>& out) {
         out.write(v);
         out.write(2*v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int z = 1;
         sub_send(7 + sub_recv(in, z), out);
         out.write(55);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/
         {IOOpTest("out", 5 + 7 - 1, true),
          IOOpTest("out", 2 * (5 + 7 - 1), true), IOOpTest("out", 55, true)});
}

TEST_F(TranslatorIOTest, Subroutine4) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int sub_recvA(__xls_channel<int>& in) {
         return in.read();
       }
       int sub_recvB(__xls_channel<int>& in) {
         return in.read();
       }
       void sub_sendA(int v, __xls_channel<int>& out) {
         out.write(v);
       }
       void sub_sendB(int v, __xls_channel<int>& out) {
         out.write(v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int xx = 0;
         xx += sub_recvA(in);
         xx += sub_recvB(in);
         sub_sendA(xx, out);
         sub_sendB(xx, out);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true), IOOpTest("in", 15, true)},
         /*outputs=*/
         {IOOpTest("out", 20, true), IOOpTest("out", 20, true)});
}

TEST_F(TranslatorIOTest, Subroutine5) {
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
         int xx = 0;
         xx += sub_recv(in);
         xx += sub_recv(in);
         sub_send(xx, out);
         sub_send(xx, out);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true), IOOpTest("in", 15, true)},
         /*outputs=*/
         {IOOpTest("out", 20, true), IOOpTest("out", 20, true)});
}

TEST_F(TranslatorIOTest, MethodSubroutine) {
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

TEST_F(TranslatorIOTest, OperatorSubroutine) {
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
  ASSERT_THAT(
      SourceToIr(content, /* pfunc= */ nullptr, /* clang_argv= */ {},
                 /* io_test_mode= */ true)
          .status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("IO ops in operator calls are not supported")));
}

TEST_F(TranslatorIOTest, SubroutineConditional) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       void sub_send(int v, __xls_channel<int>& out) {
         if(v > 10) {
           out.write(v);
         }
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int v = in.read();
         sub_send(v, out);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/
         {IOOpTest("out", 5, false)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 55, true)},
         /*outputs=*/
         {IOOpTest("out", 55, true)});
}

TEST_F(TranslatorIOTest, SubroutineConditional2) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       void sub_send(int v, __xls_channel<int>& out) {
         out.write(v);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int v = in.read();
         if(v > 10) {
          sub_send(v, out);
         }
       })";
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/
         {IOOpTest("out", 5, false)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 55, true)},
         /*outputs=*/
         {IOOpTest("out", 55, true)});
}

TEST_F(TranslatorIOTest, SubroutineConditional3) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       void sub_send(int v, __xls_channel<int>& out) {
         if(v > 3) {
           out.write(v);
         }
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int v = in.read();
         if(v < 10) {
          sub_send(v, out);
         }
       })";
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true)},
         /*outputs=*/
         {IOOpTest("out", 1, false)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 8, true)},
         /*outputs=*/
         {IOOpTest("out", 8, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 55, true)},
         /*outputs=*/
         {IOOpTest("out", 55, false)});
}

TEST_F(TranslatorIOTest, SubroutineConditionalReceive) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       void sub_receive(int v, int& out, __xls_channel<int>& in) {
         if(v > 3) {
          out = in.read();
         }
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in, __xls_channel<int>& in_test,
                       __xls_channel<int>& out) {
         int v = in.read();
         int ret = 1000;
         if(v < 10) {
          sub_receive(v, ret, in_test);
         }
         out.write(ret);
       })";
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true), IOOpTest("in_test", 123, false)},
         /*outputs=*/
         {IOOpTest("out", 1000, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 8, true), IOOpTest("in_test", 123, true)},
         /*outputs=*/
         {IOOpTest("out", 123, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 55, true), IOOpTest("in_test", 123, false)},
         /*outputs=*/
         {IOOpTest("out", 1000, true)});
}

TEST_F(TranslatorIOTest, SubroutineConditionalMemoryRead) {
  const std::string content = R"(
       #include "/xls_builtin.h"

       #include "/xls_builtin.h"
       void sub_read(int v, int& out, __xls_memory<int, 32>& memory) {
         if(v > 3) {
          out = memory[5];
         }
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in, __xls_memory<int, 32>& memory,
                       __xls_channel<int>& out) {
         int v = in.read();
         int ret = 1000;
         if(v < 10) {
          sub_read(v, ret, memory);
         }
         out.write(ret);
       })";

  IOTest(
      content,
      /*inputs=*/{IOOpTest("in", 1, true), IOOpTest("memory__read", 30, false)},
      /*outputs=*/
      {IOOpTest("memory__read", xls::Value(xls::UBits(5, 5)), false),
       IOOpTest("out", 1000, true)});
  IOTest(
      content,
      /*inputs=*/{IOOpTest("in", 7, true), IOOpTest("memory__read", 30, true)},
      /*outputs=*/
      {IOOpTest("memory__read", xls::Value(xls::UBits(5, 5)), true),
       IOOpTest("out", 30, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", 55, true), IOOpTest("memory__read", 30, false)},
         /*outputs=*/
         {IOOpTest("memory__read", xls::Value(xls::UBits(5, 5)), false),
          IOOpTest("out", 1000, true)});
}

TEST_F(TranslatorIOTest, SubroutineConditionalMemoryWrite) {
  const std::string content = R"(
       #include "/xls_builtin.h"

       #include "/xls_builtin.h"
       void sub_write(int v, __xls_memory<int, 32>& memory) {
         if(v > 3) {
          memory[5] = v;
         }
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in, __xls_memory<int, 32>& memory) {
         int v = in.read();
         if(v < 10) {
          sub_write(v, memory);
         }
       })";

  {
    auto memory_write_tuple = xls::Value::Tuple(
        {xls::Value(xls::SBits(5, 5)), xls::Value(xls::SBits(7, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 1, true)},
           /*outputs=*/
           {IOOpTest("memory__write", memory_write_tuple, false)});
  }
  {
    auto memory_write_tuple = xls::Value::Tuple(
        {xls::Value(xls::SBits(5, 5)), xls::Value(xls::SBits(7, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 7, true)},
           /*outputs=*/
           {IOOpTest("memory__write", memory_write_tuple, true)});
  }
  {
    auto memory_write_tuple = xls::Value::Tuple(
        {xls::Value(xls::SBits(5, 5)), xls::Value(xls::SBits(7, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 55, true)},
           /*outputs=*/
           {IOOpTest("memory__write", memory_write_tuple, false)});
  }
}

TEST_F(TranslatorIOTest, SaveChannel) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {

         __xls_channel<int>& out_(out);

         out_.write(in.read());
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 111, true)},
         /*outputs=*/{IOOpTest("out", 111, true)});
}

TEST_F(TranslatorIOTest, MixedOps) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {

         const int x = in.read();

         in.write(x);
         out.write(x+1);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 111, true)},
         /*outputs=*/{IOOpTest("in", 111, true), IOOpTest("out", 112, true)});
}

TEST_F(TranslatorIOTest, Unrolled) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& out) {
         #pragma hls_unroll yes
         for(int i=0;i<4;++i) {
           out.write(i);
         }
       })";

  IOTest(content, /*inputs=*/{},
         /*outputs=*/
         {IOOpTest("out", 0, true), IOOpTest("out", 1, true),
          IOOpTest("out", 2, true), IOOpTest("out", 3, true)});
}

TEST_F(TranslatorIOTest, UnrolledSubroutine) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       void sub(__xls_channel<int>& in,
                int i,
                __xls_channel<int>& out) {
           out.write(i * in.read());
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         #pragma hls_unroll yes
         for(int i=0;i<4;++i) {
           sub(in , i, out);
         }
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", 2, true), IOOpTest("in", 4, true),
          IOOpTest("in", 5, true), IOOpTest("in", 10, true)},
         /*outputs=*/
         {IOOpTest("out", 0, true), IOOpTest("out", 4, true),
          IOOpTest("out", 10, true), IOOpTest("out", 30, true)});
}

TEST_F(TranslatorIOTest, UnrolledUnsequenced) {
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

TEST_F(TranslatorIOTest, InThisExpr) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Test {
         int x;
         int foo()const {
           return x;
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<Test>& in,
                       __xls_channel<int>& out) {
         out.write(3*in.read().foo());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", xls::Value::Tuple({xls::Value(xls::SBits(5, 32))}),
                   true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, InThisExprMutableTemp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Test {
         int x;
         int foo() {
           return x;
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<Test>& in,
                       __xls_channel<int>& out) {
         out.write(3*in.read().foo());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", xls::Value::Tuple({xls::Value(xls::SBits(5, 32))}),
                   true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, ShortCircuitAnd) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int zero = 0;
         int x = in.read();
         int v = 100;
         if(zero && x) {
           v = out.read();
         }
         out.write(1 + v);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1000, true)},
         /*outputs=*/{IOOpTest("out", 101, true)});
}

TEST_F(TranslatorIOTest, ShortCircuitOr) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int one = 1;
         int x = in.read();
         int v = 100;
         if(!(one || x)) {
           v = out.read();
         }
         out.write(1 + v);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1000, true)},
         /*outputs=*/{IOOpTest("out", 101, true)});
}

TEST_F(TranslatorIOTest, NoShortCircuitAnd) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int one = 1;
         int x = in.read();
         int v = 100;
         if(one && x) {
           v = in.read();
         }
         out.write(1 + v);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 0, true), IOOpTest("in", 0, false)},
         /*outputs=*/{IOOpTest("out", 101, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true), IOOpTest("in", 1000, true)},
         /*outputs=*/{IOOpTest("out", 1001, true)});
}

TEST_F(TranslatorIOTest, ConstCondition) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       template<bool direction_read>
       void read_or_write(__xls_channel<int>& ch, int& val) {
         if(direction_read) {
           val = ch.read();
         } else {
           ch.write(val);
         }
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int v = 100;
         read_or_write<true>(in, v);
         ++v;
         read_or_write<false>(out, v);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 6, true)});
}

TEST_F(TranslatorIOTest, ConstConditionShortCircuitAnd) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       template<bool direction_read>
       void read_or_write(__xls_channel<int>& ch, int& val) {
         if(direction_read) {
           val = ch.read();
         } else {
           if(val > 0) {
             ch.write(val);
           }
         }
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int v = 100;
         read_or_write<true>(in, v);
         ++v;
         read_or_write<false>(out, v);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 6, true)});
}

TEST_F(TranslatorIOTest, NonparameterIOOps) {
  const std::string content = R"(
      #pragma hls_top
      void my_package(__xls_channel<int>& in) {
         __xls_channel<int> out;
         out.write(3*in.read());
       })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("hannel declaration uninitialized")));
}

TEST_F(TranslatorIOTest, Struct) {
  const std::string content = R"(
struct Block {
  __xls_channel<int, __xls_channel_dir_In>& in;
  __xls_channel<int, __xls_channel_dir_Out>& out;
};

#pragma hls_top
void Run(__xls_channel<int, __xls_channel_dir_In>& in,
         __xls_channel<int, __xls_channel_dir_Out>& out) {

  StatementBlock* block = {.in = in, .out = out};
  block.out.write(block.in.read() * 3);
}
)";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, Class) {
  const std::string content = R"(
class Block {
 public:
  __xls_channel<int, __xls_channel_dir_In>& in;
  __xls_channel<int, __xls_channel_dir_Out>& out;

  void Run() {
    out.write(in.read() * 3);
  }
};

#pragma hls_top
void Run(__xls_channel<int, __xls_channel_dir_In>& in,
         __xls_channel<int, __xls_channel_dir_Out>& out) {
  StatementBlock* block = {.in = in, .out = out};
  block.Run();
}
)";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, SaveChannelStruct) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         __xls_channel<int>& out_;

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
         Foo f = {.out_ = out};
        f.sub_send(7 + f.sub_recv(in));


       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 7 + 5, true)});
}

TEST_F(TranslatorIOTest, SaveChannelStructConstructorChannelInit) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         __xls_channel<int>& out_;

         Foo(__xls_channel<int>& out) : out_(out) { }

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

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 7 + 5, true)});
}

TEST_F(TranslatorIOTest, SaveChannelIOInConstructor) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         __xls_channel<int>& out_;

         Foo(__xls_channel<int>& out) : out_(out) {
          out.write(5000);
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

  IOTest(
      content,
      /*inputs=*/{IOOpTest("in", 5, true)},
      /*outputs=*/{IOOpTest("out", 5000, true), IOOpTest("out", 7 + 5, true)});
}

TEST_F(TranslatorIOTest, SaveChannelStructAssignMember) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         __xls_channel<int>& out_;
         int x_ = 7;

         void Run(__xls_channel<int>& in) {
           x_ += in.read();
           out_.write(x_);
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Foo f = {.out_ = out};
         f.Run(in);
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 7 + 5, true)});
}

TEST_F(TranslatorIOTest, MuxTwoInputs) {
  const std::string content = R"(
    struct Foo {
      int x;
    };

    #pragma hls_top
    void foo(int& dir,
              __xls_channel<Foo>& in1,
              __xls_channel<Foo>& in2,
              __xls_channel<Foo>& out) {

      Foo v;

      if (dir == 0) {
        v = in1.read();
      } else {
        v = in2.read();
      }

      out.write(v);
    })";

  ASSERT_THAT(SourceToIr(content, /*pfunc=*/nullptr, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              xls::status_testing::StatusIs(absl::StatusCode::kOk));
}

TEST_F(TranslatorIOTest, AcChannelAlias) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       template <typename T>
       using Channel = __xls_channel<T>;
       #pragma hls_top
       void my_package(Channel<int>& in,
                       Channel<int>& out) {
         out.write(3*in.read());
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, ChannelAliasWithDir) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       template <typename T>
       using Consumer = __xls_channel<T, __xls_channel_dir_In>;
       template <typename T>
       using Producer = __xls_channel<T, __xls_channel_dir_Out>;
       #pragma hls_top
       void my_package(Consumer<int>& in,
                       Producer<int>& out) {
         out.write(3*in.read());
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, ChannelRef) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
          __xls_channel<int>& out_ref = out;
          out_ref.write(3*in.read());
       })";

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsRead) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& dir_in,
                       __xls_channel<int>& in1,
                       __xls_channel<int>& in2,
                       __xls_channel<int>& out) {
          const int dir = dir_in.read();
          __xls_channel<int>& ch_r = dir?in1:in2;
          out.write(3*ch_r.read());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", 5, true),
          IOOpTest("in2", 0, false)},
         /*outputs=*/{IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in1", 0, false),
          IOOpTest("in2", 2, true)},
         /*outputs=*/{IOOpTest("out", 6, true)});
}

TEST_F(TranslatorIOTest, ReturnChannelSubroutine) {
  const std::string content = R"(
       template<typename T>
       using my_ch = __xls_channel<T>;

       my_ch<int>& SelectCh(bool select_a,
                        my_ch<int>& ac,
                        my_ch<int>& bc) {
          return ac;
       }

       #pragma hls_top
       void my_package(my_ch<int>& dir_in,
                       my_ch<int>& in1,
                       my_ch<int>& in2,
                       my_ch<int>& out) {
          my_ch<int>& inOne = in1;
          my_ch<int>& inOneOne = inOne;
          const int dir = dir_in.read();
          my_ch<int>& ch_r = SelectCh(dir, inOneOne, in2);
          out.write(3*ch_r.read());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", 5, true)},
         /*outputs=*/{IOOpTest("out", 15, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsReadSubroutine) {
  const std::string content = R"(
       template<typename T>
       using my_ch = __xls_channel<T>;

       my_ch<int>& SelectCh(bool select_a,
                        my_ch<int>& ac,
                        my_ch<int>& bc) {
          return select_a ? ac : bc;
       }

       #pragma hls_top
       void my_package(my_ch<int>& dir_in,
                       my_ch<int>& in1,
                       my_ch<int>& in2,
                       my_ch<int>& out) {
          my_ch<int>& inOne = in1;
          my_ch<int>& inOneOne = inOne;
          const int dir = dir_in.read();
          my_ch<int>& ch_r = SelectCh(dir, inOneOne, in2);
          out.write(3*ch_r.read());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", 5, true),
          IOOpTest("in2", 0, false)},
         /*outputs=*/{IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in1", 0, false),
          IOOpTest("in2", 2, true)},
         /*outputs=*/{IOOpTest("out", 6, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsReadSubroutineTemplated) {
  const std::string content = R"(
       template<typename T>
       using my_ch = __xls_channel<T>;

       template<typename T>
       my_ch<T>& SelectCh(bool select_a,
                        my_ch<T>& ac,
                        my_ch<T>& bc) {
          return select_a ? ac : bc;
       }

       #pragma hls_top
       void my_package(my_ch<int>& dir_in,
                       my_ch<int>& in1,
                       my_ch<int>& in2,
                       my_ch<int>& out) {
          my_ch<int>& inOne = in1;
          my_ch<int>& inOneOne = inOne;
          const int dir = dir_in.read();
          my_ch<int>& ch_r = SelectCh(dir, inOneOne, in2);
          out.write(3*ch_r.read());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", 5, true),
          IOOpTest("in2", 0, false)},
         /*outputs=*/{IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in1", 0, false),
          IOOpTest("in2", 2, true)},
         /*outputs=*/{IOOpTest("out", 6, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsReadSubroutine2) {
  const std::string content = R"(
       template<typename T>
       using my_ch = __xls_channel<T>;

       my_ch<int>& SelectCh(bool select_a,
                            bool select_c,
                            my_ch<int>& ac,
                            my_ch<int>& bc,
                            my_ch<int>& cc) {
          return select_c ? cc : (!select_a ? ac : bc);
       }

       #pragma hls_top
       void my_package(my_ch<int>& dir_a_in,
                       my_ch<int>& dir_c_in,
                       my_ch<int>& ina,
                       my_ch<int>& inb,
                       my_ch<int>& inc,
                       my_ch<int>& out) {
          const int dir_a = dir_a_in.read();
          const int dir_c = dir_c_in.read();
          my_ch<int>& ch_r = SelectCh(dir_a, dir_c, ina, inb, inc);
          out.write(ch_r.read());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_a_in", 0, true), IOOpTest("dir_c_in", 0, true),
          IOOpTest("inc", 13, false), IOOpTest("ina", 5, true),
          IOOpTest("inb", 3, false)},
         /*outputs=*/{IOOpTest("out", 5, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_a_in", 1, true), IOOpTest("dir_c_in", 0, true),
          IOOpTest("inc", 13, false), IOOpTest("ina", 5, false),
          IOOpTest("inb", 3, true)},
         /*outputs=*/{IOOpTest("out", 3, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_a_in", 0, true), IOOpTest("dir_c_in", 1, true),
          IOOpTest("inc", 13, true), IOOpTest("ina", 5, false),
          IOOpTest("inb", 3, false)},
         /*outputs=*/{IOOpTest("out", 13, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_a_in", 1, true), IOOpTest("dir_c_in", 1, true),
          IOOpTest("inc", 13, true), IOOpTest("ina", 5, false),
          IOOpTest("inb", 3, false)},
         /*outputs=*/{IOOpTest("out", 13, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsReadSubroutineSwitch) {
  const std::string content = R"(
       template<typename T>
       using my_ch = __xls_channel<T>;

       my_ch<int>& SelectCh(int idx, my_ch<int>& a, my_ch<int>& b) {
        switch (idx) {
          default:
            __xlscc_assert("Unsupported index to SelectIf!", false);
            // Fall through to make all paths return
          case 0:
            return b;
          case 1:
            return a;
          }
        }

       #pragma hls_top
       void my_package(my_ch<int>& dir_in,
                       my_ch<int>& in1,
                       my_ch<int>& in2,
                       my_ch<int>& out) {
          const int dir = dir_in.read();
          my_ch<int>& ch_r = SelectCh(dir, in1, in2);
          out.write(3*ch_r.read());
       })";

  auto one_bit_0 = xls::Value(xls::UBits(0, 1));
  auto one_bit_1 = xls::Value(xls::UBits(1, 1));
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in2", 3, false),
          IOOpTest("in1", 5, true)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_0, "Unsupported index to SelectIf!",
                   TraceType::kAssert),
          IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in2", 3, true),
          IOOpTest("in1", 5, false)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_0, "Unsupported index to SelectIf!",
                   TraceType::kAssert),
          IOOpTest("out", 9, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 2, true), IOOpTest("in2", 3, true),
          IOOpTest("in1", 5, false)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_1, "Unsupported index to SelectIf!",
                   TraceType::kAssert),
          // Fall through outputs on b
          IOOpTest("out", 9, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsReadSubroutineSwitchTemplated) {
  const std::string content = R"(

       template<typename C>
       C& SelectCh(int idx, C& a, C& b) {
        switch (idx) {
          default:
            __xlscc_assert("Unsupported index to SelectIf!", false);
            // Fall through to make all paths return
          case 0:
            return b;
          case 1:
            return a;
          }
        }

       template<typename T>
       using my_ch = __xls_channel<T>;

       #pragma hls_top
       void my_package(my_ch<int>& dir_in,
                       my_ch<int>& in1,
                       my_ch<int>& in2,
                       my_ch<int>& out) {
          const int dir = dir_in.read();
          my_ch<int>& ch_r = SelectCh(dir, in1, in2);
          out.write(3*ch_r.read());
       })";

  auto one_bit_0 = xls::Value(xls::UBits(0, 1));
  auto one_bit_1 = xls::Value(xls::UBits(1, 1));
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in2", 3, false),
          IOOpTest("in1", 5, true)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_0, "Unsupported index to SelectIf!",
                   TraceType::kAssert),
          IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in2", 3, true),
          IOOpTest("in1", 5, false)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_0, "Unsupported index to SelectIf!",
                   TraceType::kAssert),
          IOOpTest("out", 9, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 2, true), IOOpTest("in2", 3, true),
          IOOpTest("in1", 5, false)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_1, "Unsupported index to SelectIf!",
                   TraceType::kAssert),
          // Fall through outputs on b
          IOOpTest("out", 9, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsReadAssign) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& dir_in,
                       __xls_channel<int>& in1,
                       __xls_channel<int>& in2,
                       __xls_channel<int>& out) {
          const int dir = dir_in.read();
          __xls_channel<int>& ch_r = dir?in1:in2;
          int val_read = 0;
          ch_r.read(val_read);
          out.write(3*val_read);
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", 5, true),
          IOOpTest("in2", 0, false)},
         /*outputs=*/{IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in1", 0, false),
          IOOpTest("in2", 2, true)},
         /*outputs=*/{IOOpTest("out", 6, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsWrite) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& dir_in,
                       __xls_channel<int>& in,
                       __xls_channel<int>& out1,
                       __xls_channel<int>& out2) {
          const int dir = dir_in.read();
          __xls_channel<int>& ch_w = dir?out1:out2;
          ch_w.write(5*in.read());
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("out1", 25, true), IOOpTest("out2", 0, false)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in", 3, true)},
         /*outputs=*/{IOOpTest("out1", 0, false), IOOpTest("out2", 15, true)});
}

TEST_F(TranslatorIOTest, TernaryOnChannelsNonblocking) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& dir_in,
                       __xls_channel<int>& in1,
                       __xls_channel<int>& in2,
                       __xls_channel<int>& out) {
          const int dir = dir_in.read();
          __xls_channel<int>& ch_r = dir?in1:in2;
          int val = 0;
          bool received = ch_r.nb_read(val);
          if(received) {
            out.write(3*val);
          }
       })";

  xls::Value value_in_5 = xls::Value::Tuple(
      {xls::Value(xls::SBits(5, 32)), xls::Value::Bool(true)});
  xls::Value value_in_3 = xls::Value::Tuple(
      {xls::Value(xls::SBits(3, 32)), xls::Value::Bool(true)});
  xls::Value value_not_ready = xls::Value::Tuple(
      {xls::Value(xls::SBits(0, 32)), xls::Value::Bool(false)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", value_in_5, true),
          IOOpTest("in2", value_not_ready, false)},
         /*outputs=*/{IOOpTest("out", 15, true)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in1", value_in_3, false),
          IOOpTest("in2", value_not_ready, true)},
         /*outputs=*/{IOOpTest("out", 0, false)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 1, true), IOOpTest("in1", value_not_ready, true),
          IOOpTest("in2", value_in_5, false)},
         /*outputs=*/{IOOpTest("out", 0, false)});
  IOTest(content,
         /*inputs=*/
         {IOOpTest("dir_in", 0, true), IOOpTest("in1", value_not_ready, false),
          IOOpTest("in2", value_in_3, true)},
         /*outputs=*/{IOOpTest("out", 9, true)});
}

TEST_F(TranslatorIOTest, TernaryChannelRefParam) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int do_read(__xls_channel<int>& ch) {
        return ch.read();
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& dir_in,
                       __xls_channel<int>& in1,
                       __xls_channel<int>& in2,
                       __xls_channel<int>& out) {
          const int dir = dir_in.read();
          __xls_channel<int>& ch_r = dir?in1:in2;
          out.write(3*do_read(ch_r));
       })";

  xlscc::GeneratedFunction* func;
  ASSERT_THAT(SourceToIr(content, &func, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("hannel select passed as parameter")));
}

TEST_F(TranslatorIOTest, ChannelRefInStructSubroutineRef) {
  const std::string content = R"(
       class SenderThing {
       public:
        SenderThing(__xls_channel<int>& ch, int out_init = 3)
          : ch(ch), out(out_init)
        {}

        void send(int offset) {
          ch.write(offset + out);
        }
       private:
        __xls_channel<int>& ch;
        int out;
       };

       void SubSend(SenderThing& sender, int val) {
        sender.send(val);
       }

       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
          SenderThing sender(out);
          const int val = in.read();
          SubSend(sender, val);
       })";

  xlscc::GeneratedFunction* func;
  ASSERT_THAT(SourceToIr(content, &func, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("arameters containing LValues")));
}

TEST_F(TranslatorIOTest, TopParameterStructWithChannel) {
  const std::string content = R"(
      struct Block {
        int x;
        __xls_channel<int> ch;
        int y;
      };
      #include "/xls_builtin.h"
      #pragma hls_top
      void my_package(Block in,
                       __xls_channel<int>& out) {
        out.write(3*in.ch.read());
      })";

  xlscc::GeneratedFunction* func;
  ASSERT_THAT(SourceToIr(content, &func, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("lvalues in nested structs")));
}

TEST_F(TranslatorIOTest, DebugAssert) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int r = in.read();
         if(r != 3) {
           __xlscc_assert("hello", r > 5);
         }
         out.write(3*r);
       })";

  auto one_bit_0 = xls::Value(xls::UBits(0, 1));
  auto one_bit_1 = xls::Value(xls::UBits(1, 1));
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 3, true)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_0, "hello", TraceType::kAssert),
          IOOpTest("out", 9, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 10, true)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_0, "hello", xlscc::TraceType::kAssert),
          IOOpTest("out", 30, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_1, "hello", xlscc::TraceType::kAssert),
          IOOpTest("out", 3, true)});
}

TEST_F(TranslatorIOTest, DebugAssertWithLabel) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int r = in.read();
         if(r != 3) {
           __xlscc_assert("hello", r > 5, "this one");
         }
         out.write(3*r);
       })";

  auto one_bit_0 = xls::Value(xls::UBits(0, 1));
  auto one_bit_1 = xls::Value(xls::UBits(1, 1));
  IOTest(
      content,
      /*inputs=*/{IOOpTest("in", 3, true)},
      /*outputs=*/
      {IOOpTest("__trace", one_bit_0, "hello", TraceType::kAssert, "this one"),
       IOOpTest("out", 9, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 10, true)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_0, "hello", xlscc::TraceType::kAssert,
                   "this one"),
          IOOpTest("out", 30, true)});
  IOTest(content,
         /*inputs=*/{IOOpTest("in", 1, true)},
         /*outputs=*/
         {IOOpTest("__trace", one_bit_1, "hello", xlscc::TraceType::kAssert,
                   "this one"),
          IOOpTest("out", 3, true)});
}

TEST_F(TranslatorIOTest, DebugTrace) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int r = in.read();
         if(r != 7) {
          __xlscc_trace("Value is {:d}", r);
         }
         out.write(3*r);
       })";

  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(1, 1)), xls::Value(xls::UBits(10, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 10, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 30, true)});
  }
  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(0, 1)), xls::Value(xls::UBits(7, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 7, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 21, true)});
  }
}

TEST_F(TranslatorIOTest, DebugMultiTrace) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int r = in.read();
         if(r != 7) {
          __xlscc_trace("Value is {:d}", r);
          __xlscc_trace("Second is {:d}", r*5);
         }
         out.write(3*r);
       })";

  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(1, 1)), xls::Value(xls::UBits(10, 32))});
    auto trace_tuple2 = xls::Value::Tuple(
        {xls::Value(xls::UBits(1, 1)),
         xls::Value(xls::UBits(static_cast<uint64_t>(10 * 5), 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 10, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("__trace", trace_tuple2, "Second is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 30, true)});
  }
  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(0, 1)), xls::Value(xls::UBits(7, 32))});
    auto trace_tuple2 = xls::Value::Tuple(
        {xls::Value(xls::UBits(0, 1)),
         xls::Value(xls::UBits(static_cast<uint64_t>(7 * 5), 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 7, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("__trace", trace_tuple2, "Second is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 21, true)});
  }
}

TEST_F(TranslatorIOTest, DebugTraceInSubroutine) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       void SubWrite(__xls_channel<int>& out, int r) {
         __xlscc_trace("Value is {:d}", r);
         out.write(3*r);
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         const int r = in.read();
         if(r != 20) {
           SubWrite(out, r);
         }
       })";

  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(1, 1)), xls::Value(xls::UBits(10, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 10, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 30, true)});
  }
  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(0, 1)), xls::Value(xls::UBits(20, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 20, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 60, false)});
  }
}

TEST_F(TranslatorIOTest, DebugTraceReference) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int r = in.read();
         int& r_ref = r;
         if(r != 7) {
          __xlscc_trace("Value is {:d}", r_ref);
         }
         out.write(3*r);
       })";

  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(1, 1)), xls::Value(xls::UBits(10, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 10, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 30, true)});
  }
  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(0, 1)), xls::Value(xls::UBits(7, 32))});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 7, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 21, true)});
  }
}

TEST_F(TranslatorIOTest, DebugTracePointer) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int r = in.read();
         int* r_ref = &r;
         __xlscc_trace("Value is {:d}", r_ref);
         out.write(3*r);
       })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                    testing::HasSubstr("must have R-Value")));
}

TEST_F(TranslatorIOTest, DebugTraceStruct) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int r = in.read();
         struct Foo {
           int r;
         };
         Foo f = {.r = r};
         __xlscc_trace("Value is {:d}", f);
         out.write(3*r);
       })";

  {
    auto trace_tuple = xls::Value::Tuple(
        {xls::Value(xls::UBits(1, 1)),
         xls::Value::Tuple({xls::Value(xls::UBits(10, 32))})});
    IOTest(content,
           /*inputs=*/{IOOpTest("in", 10, true)},
           /*outputs=*/
           {IOOpTest("__trace", trace_tuple, "Value is {:d}",
                     xlscc::TraceType::kTrace),
            IOOpTest("out", 30, true)});
  }
}

TEST_F(TranslatorIOTest, DebugTraceStructWithReference) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         int r = in.read();
         struct Foo {
           int r;
           int *rr;
         };
         Foo f = {.r = r, .rr = &r};
         (void)f;
         __xlscc_trace("Value is {:d}", f);
         out.write(3*r);
       })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("LValue")));
}

}  // namespace

}  // namespace xlscc
