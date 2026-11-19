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

#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"

namespace xlscc {
namespace {

class TranslatorStaticTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorStaticTest, Static) {
  const std::string content = R"(

       #pragma hls_top
       int my_package() {
         static int x = 22;
         return x++;
       })";

  const absl::flat_hash_map<std::string, xls::Value> args = {};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(22, 32)),
      xls::Value(xls::SBits(23, 32)),
      xls::Value(xls::SBits(24, 32)),
  };

  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, StaticCall) {
  const std::string content = R"(
      int inner(int input) {
        static long a = 3;
        --a;
        return a * input;
      }

       #pragma hls_top
       int my_package(int y) {
         static int x = 22;
         int f = inner(x++);
         return y+f;
       })";

  const absl::flat_hash_map<std::string, xls::Value> args = {
      {"y", xls::Value(xls::SBits(10, 32))}};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(54, 32)),
      xls::Value(xls::SBits(33, 32)),
      xls::Value(xls::SBits(10, 32)),
  };

  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, StaticCallMulti) {
  const std::string content = R"(
      int inner() {
        static int a = 10;
        a--;
        return a;
      }

       #pragma hls_top
       int my_package(int y) {
         y += inner();
         return y + inner();
      })";

  const absl::flat_hash_map<std::string, xls::Value> args = {
      {"y", xls::Value(xls::SBits(10, 32))}};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(10 + 9 + 8, 32)),
      xls::Value(xls::SBits(10 + 7 + 6, 32)),
      xls::Value(xls::SBits(10 + 5 + 4, 32)),
  };
  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, StaticConditionalAssign) {
  const std::string content = R"(

       #pragma hls_top
       int my_package(int y) {
         static int x = 22;
         if(y == 1) {
           x++;
         }
         return x;
       })";

  {
    const absl::flat_hash_map<std::string, xls::Value> args = {
        {"y", xls::Value(xls::SBits(1, 32))}};
    xls::Value expected_vals[] = {
        xls::Value(xls::SBits(23, 32)),
        xls::Value(xls::SBits(24, 32)),
        xls::Value(xls::SBits(25, 32)),
    };
    RunWithStatics(args, expected_vals, content);
  }
  {
    const absl::flat_hash_map<std::string, xls::Value> args = {
        {"y", xls::Value(xls::SBits(0, 32))}};
    xls::Value expected_vals[] = {
        xls::Value(xls::SBits(22, 32)),
        xls::Value(xls::SBits(22, 32)),
        xls::Value(xls::SBits(22, 32)),
    };
    RunWithStatics(args, expected_vals, content);
  }
}

TEST_F(TranslatorStaticTest, StaticCallConditionalAssign) {
  const std::string content = R"(
       int sub() {
         static int x = 22;
         x++;
         return x;
       }
       #pragma hls_top
       int my_package(int y) {
         int ret = 111;
         if(y == 1) {
           ret = sub();
         }
         return ret;
       })";

  {
    const absl::flat_hash_map<std::string, xls::Value> args = {
        {"y", xls::Value(xls::SBits(1, 32))}};
    xls::Value expected_vals[] = {
        xls::Value(xls::SBits(23, 32)),
        xls::Value(xls::SBits(24, 32)),
        xls::Value(xls::SBits(25, 32)),
    };
    RunWithStatics(args, expected_vals, content);
  }
  {
    const absl::flat_hash_map<std::string, xls::Value> args = {
        {"y", xls::Value(xls::SBits(0, 32))}};
    xls::Value expected_vals[] = {
        xls::Value(xls::SBits(111, 32)),
        xls::Value(xls::SBits(111, 32)),
        xls::Value(xls::SBits(111, 32)),
    };
    RunWithStatics(args, expected_vals, content);
  }
}

TEST_F(TranslatorStaticTest, StaticMethodLocal) {
  const std::string content = R"(
       struct Thing {
         int v_;

         Thing(int v) : v_(v) { }
         int doit() {
           static int l = 0;
           ++l;
           return l + v_;
         }
       };

       #pragma hls_top
       int my_package() {
         Thing thing(10);
         return thing.doit();
       })";

  const absl::flat_hash_map<std::string, xls::Value> args = {};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(11, 32)),
      xls::Value(xls::SBits(12, 32)),
      xls::Value(xls::SBits(13, 32)),
  };

  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, StaticInnerScopeName) {
  const std::string content = R"(

       #pragma hls_top
       int my_package() {
         static int x = 22;
         int inner = 0;
         {
           static int x = 100;
           inner = --x;
         }
         x += inner;
         return x;
       })";

  const absl::flat_hash_map<std::string, xls::Value> args = {};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(121, 32)),
      xls::Value(xls::SBits(219, 32)),
      xls::Value(xls::SBits(316, 32)),
      xls::Value(xls::SBits(412, 32)),
  };

  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, StaticMember) {
  const std::string content = R"(
       struct Something {
         static int foo;
       };
       #pragma hls_top
       int my_package() {
         return Something::foo++;
       })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("static")));
}

// Add inner
TEST_F(TranslatorStaticTest, StaticProc) {
  const std::string content = R"(
    #pragma hls_top
    void st(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int ctrl = in.read();
      static long count = 1;
      out.write(ctrl + count);
      count += 2;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("st");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32)),
                  xls::Value(xls::SBits(60, 32)),
                  xls::Value(xls::SBits(100, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(56, 32)),
                    xls::Value(xls::SBits(63, 32)),
                    xls::Value(xls::SBits(105, 32))};

  ProcTest(content, block_spec, inputs, outputs, /*min_ticks = */ 3);
}

TEST_F(TranslatorStaticTest, StaticInitListExpr) {
  const std::string content = R"(
    int my_package(int a) {
      static const int test_arr[][6] = {
          {  10,  0,  0,  0,  0,  0 },
          {  0,  20,  0,  0,  0,  0 }
      };
      return a + test_arr[0][0] + test_arr[1][1] + test_arr[1][0] + test_arr[0][1];
    })";
  Run({{"a", 3}}, 33, content);
}

TEST_F(TranslatorStaticTest, NonConstStaticInitListExpr) {
  const std::string content = R"(
    int my_package(int a) {
      static const int test_arr[][6] = {
          {  10,  0,  0,  0,  0,  0 },
          {  a,  20,  0,  0,  0,  0 }
      };
      return test_arr[0][0] + test_arr[1][1] + test_arr[1][0] + test_arr[0][1];
    })";
  //  Run({{"a", 3}}, 33, content);
  const absl::flat_hash_map<std::string, xls::Value> args = {
      {"a", xls::Value(xls::SBits(3, 32))}};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(33, 32)),
  };
  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, StaticConst) {
  const std::string content = R"(
       long long my_package(long long a) {
         static const int off = 6;
         return a+off;
       })";
  Run({{"a", 11}}, 11 + 6, content);
}

TEST_F(TranslatorStaticTest, StaticStructAccess) {
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

TEST_F(TranslatorStaticTest, IOProcStatic) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      static int accum = 0;
      accum += in.read();
      accum += in.read();
      out.write(accum);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {
        xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(7, 32)),
        xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(20, 32))};
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10, 32)),
                      xls::Value(xls::SBits(40, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 32);
}

TEST_F(TranslatorStaticTest, IOProcStaticNoIO) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      static int accum = 0;
      (void)accum;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  {
    ProcTest(content, block_spec, {}, {});
  }
}

TEST_F(TranslatorStaticTest, IOProcStaticStruct) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    struct Test {
      char a;
      short b;
      int c;
      long d;
    };

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      static Test accum = {1, 2, 3, 4};
      out.write(2*in.read() + accum.d);
      accum.d++;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(3, 32)),
                    xls::Value(xls::SBits(7, 32))};
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10, 32)),
                      xls::Value(xls::SBits(19, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 8 + 16 + 32 + 64);
}

TEST_F(TranslatorStaticTest, OnReset) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int> &out) {
      out.write((int)__xlscc_on_reset);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(1, 32)),
                      xls::Value(xls::SBits(0, 32)),
                      xls::Value(xls::SBits(0, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 1);
}

TEST_F(TranslatorStaticTest, OnReset2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int> &in,
             __xls_channel<int> &out) {
      static int val = 100;
      if(__xlscc_on_reset) {
        val = in.read();
      }
      out.write((int)val);
      ++val;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32)),
                      xls::Value(xls::SBits(56, 32)),
                      xls::Value(xls::SBits(57, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 1 + 32);
}

TEST_F(TranslatorStaticTest, OnResetIOInit) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int> &in,
             __xls_channel<int> &out) {
      static int val = in.read();
      out.write((int)val);
      ++val;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32)),
                      xls::Value(xls::SBits(56, 32)),
                      xls::Value(xls::SBits(57, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 1 + 32);
}

TEST_F(TranslatorStaticTest, OnResetNonConstInit) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(int &in,
             __xls_channel<int> &out) {
      static int val = in;
      out.write(val);
      ++val;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_DIRECT_IN);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32)),
                      xls::Value(xls::SBits(56, 32)),
                      xls::Value(xls::SBits(57, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 1 + 32);
}

TEST_F(TranslatorStaticTest, OnResetInPipelinedLoop) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int> &out) {
      int total = __xlscc_on_reset;
      #pragma hls_pipeline_init_interval 1
      for(int i=0;i<3;++i) {
        total += __xlscc_on_reset;
      }
      out.write(total);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {// First outer tick
                      xls::Value(xls::SBits(2, 32)),
                      // Second outer tick
                      xls::Value(xls::SBits(0, 32)),
                      // Third outer tick
                      xls::Value(xls::SBits(0, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorStaticTest, OnResetInPipelinedLoop2) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int> &out) {
      #pragma hls_pipeline_init_interval 1
      for(int i=0;i<10;++i) {
        out.write((int)__xlscc_on_reset);
      }
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        // First outer tick
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(0, 32)),
        xls::Value(xls::SBits(0, 32)),
        // Second outer tick
        xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32)),
        xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorStaticTest, OnResetScope) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    void sub(__xls_channel<int> &out) {
      out.write((int)__xlscc_on_reset + 100);
    }

    #pragma hls_top
    void foo(__xls_channel<int> &out) {
      sub(out);
      out.write((int)__xlscc_on_reset);
      sub(out);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(101, 32)), xls::Value(xls::SBits(1, 32)),
        xls::Value(xls::SBits(101, 32)), xls::Value(xls::SBits(100, 32)),
        xls::Value(xls::SBits(0, 32)),   xls::Value(xls::SBits(100, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 1);
}

TEST_F(TranslatorStaticTest, OnResetInitModify) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int> &out) {
      int y = 10;
      static int x = y++;
      (void)x;
      out.write(y);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(11, 32)),
                      xls::Value(xls::SBits(10, 32)),
                      xls::Value(xls::SBits(10, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(proc_state_bits, 1 + 32);
}

TEST_F(TranslatorStaticTest, StaticChannelRef) {
  const std::string content = R"(
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
          static __xls_channel<int>& out_ref = out;
          out_ref.write(3*in.read());
       })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(10, 32)),
                  xls::Value(xls::SBits(20, 32)),
                  xls::Value(xls::SBits(30, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(30, 32)),
                    xls::Value(xls::SBits(60, 32)),
                    xls::Value(xls::SBits(90, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticChannelRefInStruct) {
  const std::string content = R"(
       class SenderThing {
       public:
        SenderThing(__xls_channel<int>& ch, int out_init = 3)
          : ch(ch), out(out_init)
        {}

        void send(int offset) {
          ch.write(offset + out);
          ++out;
        }
       private:
        __xls_channel<int>& ch;
        int out;
       };

       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out1,
                       __xls_channel<int>& out2) {
          static SenderThing sender1(out1);
          static SenderThing sender2(out2, 10);
          const int val = in.read();
          sender1.send(val);
          sender2.send(val);
       })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out1");
    ch_out1->set_is_input(false);
    ch_out1->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out2");
    ch_out2->set_is_input(false);
    ch_out2->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(10, 32)),
                  xls::Value(xls::SBits(20, 32)),
                  xls::Value(xls::SBits(30, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out1"] = {xls::Value(xls::SBits(13, 32)),
                     xls::Value(xls::SBits(24, 32)),
                     xls::Value(xls::SBits(35, 32))};
  outputs["out2"] = {xls::Value(xls::SBits(20, 32)),
                     xls::Value(xls::SBits(31, 32)),
                     xls::Value(xls::SBits(42, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticChannelRefInSubroutine) {
  const std::string content = R"(
       class SenderThing {
       public:
        SenderThing(__xls_channel<int>& ch, int out_init = 3)
          : ch(ch), out(out_init)
        {}

        void send(int offset) {
          ch.write(offset + out);
          ++out;
        }
       private:
        __xls_channel<int>& ch;
        int out;
       };
       struct Block {
         void Run(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
          static SenderThing sender(out);
          sender.send(in.read());
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Block block;
         block.Run(in, out);
       })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(10, 32)),
                  xls::Value(xls::SBits(20, 32)),
                  xls::Value(xls::SBits(30, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(13, 32)),
                    xls::Value(xls::SBits(24, 32)),
                    xls::Value(xls::SBits(35, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticChannelRefInSubroutine2) {
  const std::string content = R"(
       struct Block {
         void Run(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
          static __xls_channel<int>& out_ref = out;
          out_ref.write(in.read());
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Block block;
         block.Run(in, out);
       })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(10, 32)),
                  xls::Value(xls::SBits(20, 32)),
                  xls::Value(xls::SBits(30, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(10, 32)),
                    xls::Value(xls::SBits(20, 32)),
                    xls::Value(xls::SBits(30, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticChannelRefInStructConst) {
  const std::string content = R"(
       class SenderThing {
       public:
        SenderThing(__xls_channel<int>& ch, int out_init = 3)
          : ch(ch), out(out_init)
        {}

        void send(int offset)const {
          ch.write(offset + out);
        }
       private:
        __xls_channel<int>& ch;
        int out;
       };

       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out1,
                       __xls_channel<int>& out2) {
          static const SenderThing sender1(out1);
          static const SenderThing sender2(out2, 10);
          const int val = in.read();
          sender1.send(val);
          sender2.send(val);
       })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out1");
    ch_out1->set_is_input(false);
    ch_out1->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out2");
    ch_out2->set_is_input(false);
    ch_out2->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(10, 32)),
                  xls::Value(xls::SBits(20, 32)),
                  xls::Value(xls::SBits(30, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out1"] = {xls::Value(xls::SBits(13, 32)),
                     xls::Value(xls::SBits(23, 32)),
                     xls::Value(xls::SBits(33, 32))};
  outputs["out2"] = {xls::Value(xls::SBits(20, 32)),
                     xls::Value(xls::SBits(30, 32)),
                     xls::Value(xls::SBits(40, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticChannelRefInStructAssign) {
  const std::string content = R"(
       class SenderThing {
       public:
        SenderThing(__xls_channel<int>& ch, int out_init = 3)
          : ch(ch), out(out_init)
        {}

        void send(int offset) {
          ch.write(offset + out);
          ++out;
        }
        SenderThing& operator=(const SenderThing& o) {
         out = o.out;
         ch = o.ch;
         return *this;
        }
       private:
        __xls_channel<int> ch;
        int out;
       };

       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out1,
                       __xls_channel<int>& out2) {
          static SenderThing sender1(out2);
          static SenderThing sender3(out1);
          static SenderThing sender2(out2, 10);
          sender2 = sender3;
          const int val = in.read();
          sender1.send(val);
          sender2.send(val);
       })";

  xlscc::GeneratedFunction* func;
  ASSERT_THAT(SourceToIr(content, &func, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("parameters containing LValues")));
}

TEST_F(TranslatorStaticTest, StaticChannelRefInStructWithOnReset) {
  const std::string content = R"(
       class SenderThing {
       public:
        SenderThing(__xls_channel<int>& ch, int out_init = 3)
          : ch(ch), out(out_init)
        {}

        void send(int offset) {
          ch.write(offset + out);
          ++out;
        }
       private:
        __xls_channel<int> ch;
        int out;
       };

       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out1,
                       __xls_channel<int>& out2) {
          SenderThing sender3(out1);
          static SenderThing sender1 = sender3;
          static SenderThing sender2(out2, 10);
          const int val = in.read();
          sender1.send(val);
          sender2.send(val);
       })";

  xlscc::GeneratedFunction* func;
  ASSERT_THAT(
      SourceToIr(content, &func, /* clang_argv= */ {},
                 /* io_test_mode= */ true)
          .status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("ompound lvalue not present")));
}

TEST_F(TranslatorStaticTest, StaticChannelRefInStructWithOnReset2) {
  const std::string content = R"(
       class SenderThing {
       public:
        SenderThing(__xls_channel<int>& ch, int out_init = 3)
          : ch(ch), out(out_init)
        {}

        void send(int offset) {
          ch.write(offset + out);
          ++out;
        }
       private:
        __xls_channel<int> ch;
        int out;
       };

       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out1,
                       __xls_channel<int>& out2) {
          static SenderThing sender1(out1, __xlscc_on_reset);
          static SenderThing sender2(out2, 10);
          const int val = in.read();
          sender1.send(val);
          sender2.send(val);
       })";

  xlscc::GeneratedFunction* func;
  ASSERT_THAT(SourceToIr(content, &func, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("using side-effects")));
}

TEST_F(TranslatorStaticTest, ReturnStaticLValue) {
  const std::string content = R"(

      int& GetStatic() {
        static int x = 22;
        return x;
      }

      #pragma hls_top
      int my_package() {
        int& sx = GetStatic();
        return sx++;
      })";

  const absl::flat_hash_map<std::string, xls::Value> args = {};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(22, 32)),
      xls::Value(xls::SBits(23, 32)),
      xls::Value(xls::SBits(24, 32)),
  };

  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, ReturnStaticLValueTemplated) {
  const std::string content = R"(

      template<int N>
      int& GetStatic() {
        static int x = 22;
        return x;
      }

      template<>
      int& GetStatic<2>() {
        static int xx = 100;
        return xx;
      }

      #pragma hls_top
      int my_package() {
        int& sx = GetStatic<1>();
        int& sy = GetStatic<2>();
        sy += 2;
        sx++;
        return sx + sy;
      })";

  const absl::flat_hash_map<std::string, xls::Value> args = {};

  xls::Value expected_vals[] = {
      xls::Value(xls::SBits(125, 32)),
      xls::Value(xls::SBits(128, 32)),
      xls::Value(xls::SBits(131, 32)),
  };

  RunWithStatics(args, expected_vals, content);
}

TEST_F(TranslatorStaticTest, StaticDecomposed) {
  const std::string content = R"(
    struct Thing {
      int x;
      long y;
    };

    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        static Thing thing;
        thing.x += in.read();
        thing.y = thing.x * 3;
        out.write(thing.y);
      }
    };
  )";

  generate_new_fsm_ = true;

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(80, 32)),
                  xls::Value(xls::SBits(100, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(3 * 80, 32)),
                      xls::Value(xls::SBits(3 * (80 + 100), 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }

  ASSERT_EQ(package_->procs().size(), 1);
  const xls::Proc* proc = package_->procs().at(0).get();

  int64_t total_32bit_elems = 0;
  int64_t total_64bit_elems = 0;

  for (const xls::StateElement* state : proc->StateElements()) {
    EXPECT_FALSE(state->type()->IsTuple() &&
                 state->type()->GetFlatBitCount() > 0);
    if (state->type()->IsBits() && state->type()->GetFlatBitCount() == 32) {
      ++total_32bit_elems;
    }
    if (state->type()->IsBits() && state->type()->GetFlatBitCount() == 64) {
      ++total_64bit_elems;
    }
  }

  EXPECT_EQ(total_32bit_elems, 1);
  EXPECT_EQ(total_64bit_elems, 1);
}

TEST_F(TranslatorStaticTest, StaticCallConditionalInterleaved) {
  const std::string content = R"(
    int leaf(int delta) {
      static int s = 10;
      s += delta;
      return s;
    }

    int mid(int en_leaf, int delta) {
      static int m = 100;
      m += 1;
      if (en_leaf != 0) {
        return m + leaf(delta);
      }
      return m;
    }

    #pragma hls_top
    void my_package(__xls_channel<int>& in, __xls_channel<int>& out) {
      int en_mid = in.read();
      int en_leaf = in.read();
      int delta = in.read();
      if (en_mid != 0) {
        out.write(mid(en_leaf, delta));
      } else {
        out.write(-1);
      }
    })";

  HLSBlock block_spec;
  block_spec.set_name("my_package");
  {
    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  const int kCalls[][3] = {
      {0, 1, 50}, {1, 0, 50}, {1, 1, 5}, {0, 1, 99}, {1, 0, 77}, {1, 1, 3},
  };

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  int s = 10;
  int m = 100;
  for (const auto& call : kCalls) {
    inputs["in"].push_back(xls::Value(xls::SBits(call[0], 32)));
    inputs["in"].push_back(xls::Value(xls::SBits(call[1], 32)));
    inputs["in"].push_back(xls::Value(xls::SBits(call[2], 32)));
    if (call[0] != 0) {
      m += 1;
      int res = m;
      if (call[1] != 0) {
        s += call[2];
        res += s;
      }
      outputs["out"].push_back(xls::Value(xls::SBits(res, 32)));
    } else {
      outputs["out"].push_back(xls::Value(xls::SBits(-1, 32)));
    }
  }

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticDeclaredInConditionalAndSwitchBranches) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in, __xls_channel<int>& out) {
      int sel = in.read();
      int inner = in.read();
      int ret = 0;
      if (sel == 1) {
        static int a = 10;
        a += 1;
        ret = a;
        if (inner != 0) {
          static int b = 200;
          b += 10;
          ret += b;
        }
      } else if (sel == 2) {
        static int c = 50;
        c += 5;
        ret = c;
      } else {
        switch (sel) {
          case 3: {
            static int d = 7;
            d *= 2;
            ret = d;
            break;
          }
          default:
            ret = -1;
            break;
        }
      }
      out.write(ret);
    })";

  HLSBlock block_spec;
  block_spec.set_name("my_package");
  {
    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  const int kSteps[][2] = {
      {0, 1}, {1, 0}, {2, 1}, {1, 1}, {3, 0},
      {0, 0}, {1, 0}, {1, 1}, {2, 0}, {3, 1},
  };

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  int a = 10, b = 200, c = 50, d = 7;
  for (const auto& step : kSteps) {
    inputs["in"].push_back(xls::Value(xls::SBits(step[0], 32)));
    inputs["in"].push_back(xls::Value(xls::SBits(step[1], 32)));
    int ret = 0;
    if (step[0] == 1) {
      a += 1;
      ret = a;
      if (step[1] != 0) {
        b += 10;
        ret += b;
      }
    } else if (step[0] == 2) {
      c += 5;
      ret = c;
    } else if (step[0] == 3) {
      d *= 2;
      ret = d;
    } else {
      ret = -1;
    }
    outputs["out"].push_back(xls::Value(xls::SBits(ret, 32)));
  }

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticCallAcrossSiblingBranchesAndUnrolledLoop) {
  const std::string content = R"(
    int bump(int delta) {
      static int acc = 5;
      acc += delta;
      return acc;
    }

    #pragma hls_top
    void my_package(__xls_channel<int>& in, __xls_channel<int>& out) {
      int mode = in.read();
      int res = 0;
      if (mode == 1) {
        res = bump(10);
      } else if (mode == 2) {
        res = bump(100);
      }
      #pragma hls_unroll yes
      for (int i = 0; i < 3; ++i) {
        if (mode == 3 && i != 1) {
          static int loop_s = 1;
          loop_s += i + 1;
          res += loop_s;
        }
      }
      out.write(res);
    })";

  HLSBlock block_spec;
  block_spec.set_name("my_package");
  {
    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  const int kModes[] = {0, 1, 0, 2, 3, 0, 1, 3};
  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  int acc = 5;
  int loop_s = 1;
  for (int mode : kModes) {
    inputs["in"].push_back(xls::Value(xls::SBits(mode, 32)));
    int res = 0;
    if (mode == 1) {
      acc += 10;
      res = acc;
    } else if (mode == 2) {
      acc += 100;
      res = acc;
    }
    for (int i = 0; i < 3; ++i) {
      if (mode == 3 && i != 1) {
        loop_s += i + 1;
        res += loop_s;
      }
    }
    outputs["out"].push_back(xls::Value(xls::SBits(res, 32)));
  }

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, StaticOnResetInitializerPreservesCalleeStatic) {
  const std::string content = R"(
    int next_seed() {
      static int seed = 40;
      seed += 2;
      return seed;
    }

    #pragma hls_top
    void my_package(__xls_channel<int>& in, __xls_channel<int>& out) {
      int call_seed = in.read();
      static int cached = next_seed();
      int res = cached;
      if (call_seed != 0) {
        res += next_seed();
      }
      out.write(res);
    })";

  HLSBlock block_spec;
  block_spec.set_name("my_package");
  {
    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32)),
                  xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(0, 32)),
                  xls::Value(xls::SBits(1, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {
      xls::Value(xls::SBits(42, 32)), xls::Value(xls::SBits(42, 32)),
      xls::Value(xls::SBits(42 + 44, 32)), xls::Value(xls::SBits(42, 32)),
      xls::Value(xls::SBits(42 + 46, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorStaticTest, ConditionalCalleeAndBlockStaticMinimalRepro) {
  const std::string content = R"(
    int callee(int x) {
      static int s = 3;
      int prev = s;
      s = x;
      return prev;
    }

    #pragma hls_top
    void top(__xls_channel<int>& in, __xls_channel<int>& out) {
      int mode = in.read();
      int x = in.read();
      if (mode == 1) {
        out.write(callee(x));
      } else if (mode == 2) {
        static int b = 10;
        out.write(b);
        b += x;
      }
    })";

  HLSBlock block_spec;
  block_spec.set_name("top");
  {
    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(CHANNEL_TYPE_FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {
      xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(99, 32)),
      xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(7, 32)),
      xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(88, 32)),
      xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(4, 32)),
      xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(1, 32)),
  };

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {
      xls::Value(xls::SBits(3, 32)),
      xls::Value(xls::SBits(10, 32)),
      xls::Value(xls::SBits(7, 32)),
      xls::Value(xls::SBits(15, 32)),
  };

  ProcTest(content, block_spec, inputs, outputs);
}

}  // namespace

}  // namespace xlscc
