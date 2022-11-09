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
#include <optional>
#include <ostream>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_test.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"

using xls::status_testing::IsOkAndHolds;

namespace xlscc {
namespace {

using xls::status_testing::IsOkAndHolds;

class TranslatorProcTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorProcTest, IOProcMux) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(const int& dir,
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

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out1"] = {xls::Value(xls::SBits(55, 32))};
    outputs["out2"] = {};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    inputs["dir"] = {xls::Value(xls::SBits(1, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out1"] = {};
    outputs["out2"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorProcTest, IOProcMux2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(int& dir,
              __xls_channel<int>& in1,
              __xls_channel<int>& in2,
              __xls_channel<int>& out) {


      int x;

      if (dir == 0) {
        x = in1.read();
      } else {
        x = in2.read();
      }

      out.write(x);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(DIRECT_IN);

    HLSChannel* ch_in1 = block_spec.add_channels();
    ch_in1->set_name("in1");
    ch_in1->set_is_input(true);
    ch_in1->set_type(FIFO);

    HLSChannel* ch_in2 = block_spec.add_channels();
    ch_in2->set_name("in2");
    ch_in2->set_is_input(true);
    ch_in2->set_type(FIFO);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in1"] = {xls::Value(xls::SBits(55, 32))};
  inputs["in2"] = {xls::Value(xls::SBits(77, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    inputs["dir"] = {xls::Value(xls::SBits(1, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(77, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorProcTest, IOProcOneOp) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(const int& dir,
             __xls_channel<int>& out) {

      out.write(dir+22);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(DIRECT_IN);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(3, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(25, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorProcTest, IOProcOneLine) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {

      out.write(2*in.read());
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
    inputs["in"] = {xls::Value(xls::SBits(11, 32))};
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(22, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(23, 32))};
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(46, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorProcTest, IOProcMuxMethod) {
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

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out1"] = {xls::Value(xls::SBits(55, 32))};
    outputs["out2"] = {};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorProcTest, IOProcMuxConstDir) {
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

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out1"] = {xls::Value(xls::SBits(55, 32))};
    outputs["out2"] = {};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    inputs["dir"] = {xls::Value(xls::SBits(1, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out1"] = {};
    outputs["out2"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorProcTest, IOProcChainedConditionalRead) {
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
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(55, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(40, 32)),
                    xls::Value(xls::SBits(10, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(40, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(40, 32)),
                    xls::Value(xls::SBits(65, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(40, 32)),
                      xls::Value(xls::SBits(105, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_F(TranslatorProcTest, IOProcStaticClassState) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    struct Test {
      int st = 5;

      int calc(const int r) {
        int a = r;
        a+=st;
        ++st;
        return a;
      }
    };

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int r = in.read();
      static Test test;
      out.write(test.calc(r));
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(33, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(85, 32)),
                      xls::Value(xls::SBits(106, 32)),
                      xls::Value(xls::SBits(40, 32))};
    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 3);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 32);
}

TEST_F(TranslatorProcTest, ForPipelined) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += i;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedII2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 2
      for(long i=1;i<=4;++i) {
        a += i;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedII2Error) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 2
      for(long i=1;i<=4;++i) {
        a += i;
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

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/true));
  package_.reset(new xls::Package("my_package"));
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("nly initiation interval 1")));
}

TEST_F(TranslatorProcTest, WhilePipelined) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      long i=1;

      #pragma hls_pipeline_init_interval 1
      while(i<=4) {
        a += i;
        ++i;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, DoWhilePipelined) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& ac,
             __xls_channel<int>& bc,
             __xls_channel<int>& out) {
      int a = ac.read();
      int b = bc.read();

      long i=1;

      #pragma hls_pipeline_init_interval 1
      do {
        a += b;
        ++i;
      } while(i<=10 && a<10);

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ac_in = block_spec.add_channels();
    ac_in->set_name("ac");
    ac_in->set_is_input(true);
    ac_in->set_type(FIFO);

    HLSChannel* bc_in = block_spec.add_channels();
    bc_in->set_name("bc");
    bc_in->set_is_input(true);
    bc_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["ac"] = {xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(3, 32))};
  inputs["bc"] = {xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(2, 32)),
                  xls::Value(xls::SBits(0, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(120, 32)),
                      xls::Value(xls::SBits(11, 32)),
                      xls::Value(xls::SBits(3, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 3);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 64 + 32 + 32);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedSerial) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += i;
      }
      #pragma hls_pipeline_init_interval 1
      for(short i=0;i<2;++i) {
        a += 10;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 10 + 20, 32)),
                      xls::Value(xls::SBits(100 + 10 + 20, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 10);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_body_proc_state_bits, 1 + 32 + 16);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedSerialIO) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += in.read();
      }
      #pragma hls_pipeline_init_interval 1
      for(short i=0;i<2;++i) {
        a += 3*in.read();
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
  inputs["in"] = {
      xls::Value(xls::SBits(2, 32)),  xls::Value(xls::SBits(6, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(2, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(2 + 6 + 10 + 2 + 3 * (10 + 20), 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 5);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_body_proc_state_bits, 1 + 32 + 16);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedReturnInBody) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += i;
        if(i == 2) {
          return;
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

  XLS_ASSERT_OK(ScanFile(content));
  package_.reset(new xls::Package("my_package"));
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("eturns in pipelined loop body unimplemented")));
}

TEST_F(TranslatorProcTest, ForPipelinedMoreVars) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<short>& in,
             __xls_channel<int>& out) {
      int a = in.read();
      static short st = 3;
      long b = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=2;i<=6;++i) {
        a += b;
      }

      out.write(st + a);
      ++st;
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
  inputs["in"] = {xls::Value(xls::SBits(100, 16)),
                  xls::Value(xls::SBits(5, 16)), xls::Value(xls::SBits(30, 16)),
                  xls::Value(xls::SBits(7, 16))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(100 + 5 * 5 + 3, 32)),
                      xls::Value(xls::SBits(30 + 5 * 7 + 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 10);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 16 + 64 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 16);
}

TEST_F(TranslatorProcTest, ForPipelinedBlank) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      long i=1;

      #pragma hls_pipeline_init_interval 1
      for(;;) {
        a += i;
        ++i;
        if(i==4) {
          break;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 6, 32)),
                      xls::Value(xls::SBits(100 + 6, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedPreCondBreak) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int r = in.read();
      int a = r;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=r;++i) {
        a += 2;
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
  inputs["in"] = {xls::Value(xls::SBits(5, 32)),
                  xls::Value(xls::SBits(-1, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(5 + 2 * 5, 32)),
                      xls::Value(xls::SBits(-1, 32))};

    // 2 ticks since messages must pass back and forth between procs
    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  // r shouldn't be in state
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedInIf) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      if(a) {
        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=4;++i) {
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
  inputs["in"] = {xls::Value(xls::SBits(0, 32)),
                  xls::Value(xls::SBits(100, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(0, 32)),
                      xls::Value(xls::SBits(100 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_F(TranslatorProcTest, ForPipelinedIfInBody) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        if(i!=4) {
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 10 - 4, 32)),
                      xls::Value(xls::SBits(100 + 10 - 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }
}

TEST_F(TranslatorProcTest, ForPipelinedContinue) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        if(i == 3) {
          continue;
        }
        a += i;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 10 - 3, 32)),
                      xls::Value(xls::SBits(100 + 10 - 3, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }
}

TEST_F(TranslatorProcTest, ForPipelinedBreak) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out,
             __xls_channel<long>& out_i) {
      int a = in.read();
      long i = 0;

      #pragma hls_pipeline_init_interval 1
      for(i=1;i<=4;++i) {
        if(i == 3) {
          break;
        }
        a += i;
      }

      out.write(a);
      out_i.write(i);
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

    HLSChannel* ch_out_i = block_spec.add_channels();
    ch_out_i->set_name("out_i");
    ch_out_i->set_is_input(false);
    ch_out_i->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(80, 32)),
                  xls::Value(xls::SBits(100, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(83, 32)),
                      xls::Value(xls::SBits(103, 32))};
    outputs["out_i"] = {xls::Value(xls::SBits(3, 64)),
                        xls::Value(xls::SBits(3, 64))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 6);
  }
}

TEST_F(TranslatorProcTest, ForPipelinedInFunction) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int calc(const int r) {
      int a = r;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += i;
      }
      return a;
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int r = in.read();
      out.write(calc(r));
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(33, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32)),
                      xls::Value(xls::SBits(33 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedInMethod) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    struct Test {
      int calc(const int r) {
        int a = r;
        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=4;++i) {
          a += i;
        }
        return a;
      }
    };

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int r = in.read();
      Test test;
      out.write(test.calc(r));
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(33, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32)),
                      xls::Value(xls::SBits(33 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedInMethodWithMember) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    struct Test {
      int st = 5;

      int calc(const int r) {
        int a = r;
        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=4;++i) {
          a += i + st;
          ++st;
        }
        ++st;
        return a;
      }
    };

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int r = in.read();
      static Test test;
      out.write(test.calc(r));
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(33, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(116, 32)),
                      xls::Value(xls::SBits(156, 32)),
                      xls::Value(xls::SBits(109, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32 + 64 + 32);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 32);
}

TEST_F(TranslatorProcTest, ForPipelinedInFunctionInIf) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int calc(const int r) {
      int a = r;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += i;
      }
      return a;
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int r = in.read();
      int a = 0;
      if(r != 22) {
        a += calc(r);
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
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(100, 32)),
      xls::Value(xls::SBits(33, 32)), xls::Value(xls::SBits(22, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32)),
                      xls::Value(xls::SBits(33 + 10, 32)),
                      xls::Value(xls::SBits(0, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedStaticInBody) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(short i=1;i<=4;++i) {
        static long st = 3;
        a += st;
        ++st;
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
  XLS_ASSERT_OK(ScanFile(content));
  package_.reset(new xls::Package("my_package"));

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(100, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(100 + 3 + 4 + 5 + 6, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 16 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedStaticOuter) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();
      static short st = 3;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += st;
      }

      out.write(a);
      ++st;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 4 * 3, 32)),
                      xls::Value(xls::SBits(100 + 4 * 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 16 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 16);
}

TEST_F(TranslatorProcTest, ForPipelinedStaticOuter2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();
      static short st = 3;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += st;
        ++st;
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
    outputs["out"] = {xls::Value(xls::SBits(98, 32)),
                      xls::Value(xls::SBits(134, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64 + 16);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 16);
}

TEST_F(TranslatorProcTest, ForPipelinedNested) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        #pragma hls_pipeline_init_interval 1
        for(long j=1;j<=4;++j) {
          ++a;
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(96, 32)),
                      xls::Value(xls::SBits(116, 32)),
                      xls::Value(xls::SBits(36, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 16);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits, 1 + 32 + 64 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedNested2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();
      int c = 1;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        #pragma hls_pipeline_init_interval 1
        for(long j=1;j<=4;++j) {
          a += c;
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(96, 32)),
                      xls::Value(xls::SBits(116, 32)),
                      xls::Value(xls::SBits(36, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 16);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits, 1 + 32 + 32 + 64 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedNestedInheritIIWithLabel) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      loop_label:
      #pragma hls_pipeline_init_interval 1
      for(long o=1;o<=1;++o) {
        for(long i=1;i<=4;++i) {
          for(long j=1;j<=4;++j) {
            ++a;
          }
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(96, 32)),
                      xls::Value(xls::SBits(116, 32)),
                      xls::Value(xls::SBits(36, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 16);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits, 1 + 32 + 64 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedNestedWithIO) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        #pragma hls_pipeline_init_interval 1
        for(long j=1;j<=2;++j) {
          a += in.read();
          out.write(a);
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

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(8, 32)),
                  xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(1, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4, 32)),
                      xls::Value(xls::SBits(4 + 8, 32)),
                      xls::Value(xls::SBits(4 + 8 + 3, 32)),
                      xls::Value(xls::SBits(4 + 8 + 3 + 1, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits, 1 + 32 + 64 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedIOInBody) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += in.read();
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
  inputs["in"] = {xls::Value(xls::SBits(6, 32)),
                  xls::Value(xls::SBits(12, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(18, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedNestedNoPragma) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        for(long j=1;j<=4;++j) {
          ++a;
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(96, 32)),
                      xls::Value(xls::SBits(116, 32)),
                      xls::Value(xls::SBits(36, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 16);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(innermost_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedIOInBodySubroutine) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_read(__xls_channel<int>& in) {
      return in.read();
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += sub_read(in);
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
  inputs["in"] = {xls::Value(xls::SBits(6, 32)),
                  xls::Value(xls::SBits(12, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(18, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedIOInBodySubroutine2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += in2.read();
      }
      return a;
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      out.write(sub_read(in));
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
  inputs["in"] = {xls::Value(xls::SBits(6, 32)),
                  xls::Value(xls::SBits(12, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(18, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedIOInBodySubroutineDeclOrder) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_read(__xls_channel<int>& in2);

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      out.write(sub_read(in));
    }
    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += in2.read();
      }
      return a;
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
  inputs["in"] = {xls::Value(xls::SBits(6, 32)),
                  xls::Value(xls::SBits(12, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(18, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedIOInBodySubroutine3) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += in2.read();
      }
      return a;
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in1,
             __xls_channel<int>& in2,
             __xls_channel<int>& out) {
      out.write(sub_read(in1) + sub_read(in2));
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in1 = block_spec.add_channels();
    ch_in1->set_name("in1");
    ch_in1->set_is_input(true);
    ch_in1->set_type(FIFO);

    HLSChannel* ch_in2 = block_spec.add_channels();
    ch_in2->set_name("in2");
    ch_in2->set_is_input(true);
    ch_in2->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  XLS_ASSERT_OK(ScanFile(content));
  package_.reset(new xls::Package("my_package"));
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("ops in pipelined loops in subroutines called "
                             "with multiple different channel arguments")));
}

TEST_F(TranslatorProcTest, ForPipelinedIOInBodySubroutine4) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += in2.read();
      }
      return a;
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int ret = sub_read(in);
      ret += sub_read(in);
      out.write(ret);
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
  inputs["in"] = {xls::Value(xls::SBits(6, 32)), xls::Value(xls::SBits(12, 32)),
                  xls::Value(xls::SBits(6, 32)),
                  xls::Value(xls::SBits(12, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(18 * 2, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, ForPipelinedNoPragma) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      for(long i=1;i<=4;++i) {
        a += i;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8,
             /*max_ticks=*/100,
             /*top_level_init_interval=*/1);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, PipelinedLoopUsingMemberChannel) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         __xls_channel<int>& out_;

         int sub_recv(__xls_channel<int>& in) {
           int ret = 0;
           #pragma hls_pipeline_init_interval 1
           for(int i=0;i<5;++i) {
            ret += in.read();
           }
           return ret;
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Foo f = {.out_ = out};
        out.write(7 + f.sub_recv(in));
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
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(9, 32)), xls::Value(xls::SBits(13, 32)),
                  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(7 + 5 + 7 + 9 + 13 + 20, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 5);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, PipelinedLoopUsingMemberChannelAndVariable) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Foo {
         __xls_channel<int>& out_;
         int accum_ = 10;

         int sub_recv(__xls_channel<int>& in) {
           #pragma hls_pipeline_init_interval 1
           for(int i=0;i<2;++i) {
            accum_ += in.read();
           }
           return accum_;
         }
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_channel<int>& out) {
         Foo f = {.out_ = out};
        out.write(7 + f.sub_recv(in));
        out.write(3 + f.sub_recv(in));
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
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(9, 32)),
                  xls::Value(xls::SBits(13, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(7 + 10 + 5 + 7, 32)),
                      xls::Value(xls::SBits(3 + 10 + 5 + 7 + 9 + 13, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits, 1 + 32 + 32);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, 0);
}

TEST_F(TranslatorProcTest, IOProcClass) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          out.write(3*in.read());
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(15, 64)),
                      xls::Value(xls::SBits(21, 64)),
                      xls::Value(xls::SBits(30, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassStructData) {
  const std::string content = R"(
      struct Thing {
        long val;

        Thing(long val) : val(val) {
        }

        operator long()const {
          return val;
        }
      };

       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<Thing, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          out.write(3*in.read());
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value::Tuple({xls::Value(xls::SBits(15, 64))}),
                      xls::Value::Tuple({xls::Value(xls::SBits(21, 64))}),
                      xls::Value::Tuple({xls::Value(xls::SBits(30, 64))})};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassLocalStatic) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          static int st = 10;

          st += 3*in.read();

          out.write(st);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10 + 15, 64)),
                      xls::Value(xls::SBits(10 + 15 + 21, 64)),
                      xls::Value(xls::SBits(10 + 15 + 21 + 30, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassState) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         int st = 10;

         #pragma hls_top
         void Run() {

          st += 3*in.read();

          out.write(st);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10 + 15, 64)),
                      xls::Value(xls::SBits(10 + 15 + 21, 64)),
                      xls::Value(xls::SBits(10 + 15 + 21 + 30, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassStateWithLocalStatic) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         int st = 10;

         #pragma hls_top
         void Run() {
          static short st_local = 7;

          st += 3*in.read();

          out.write(st + st_local);

          --st_local;
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10 + 7 + 15, 64)),
                      xls::Value(xls::SBits(10 + 6 + 15 + 21, 64)),
                      xls::Value(xls::SBits(10 + 5 + 15 + 21 + 30, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassConstTop) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run()const {
          out.write(3*in.read());
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_.reset(new xls::Package("my_package"));
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("Const top")));
}

TEST_F(TranslatorProcTest, IOProcClassConstructor) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in_;
         __xls_channel<long, __xls_channel_dir_Out>& out_;
         int x_;

         Block(
          __xls_channel<int , __xls_channel_dir_In>& in,
          __xls_channel<long, __xls_channel_dir_Out>& out,
          int x) : in_(in), out_(out), x_(x) { }

         #pragma hls_top
         void Run() {
          out_.write(3*in_.read());
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_.reset(new xls::Package("my_package"));
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("onstructors in top class")));
}

TEST_F(TranslatorProcTest, IOProcClassNonVoidReturn) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         int Run() {
          out.write(3*in.read());
          return 10;
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_.reset(new xls::Package("my_package"));
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("Non-void top method return")));
}

TEST_F(TranslatorProcTest, IOProcClassParameters) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run(int a) {
          out.write(a*in.read());
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_.reset(new xls::Package("my_package"));
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("method parameters unsupported")));
}

TEST_F(TranslatorProcTest, IOProcClassStaticMethod) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         static void Run() {
          out.write(in.read());
         }
      };)";

  auto ret = ScanFile(content, /*clang_argv=*/{},
                      /*io_test_mode=*/false,
                      /*error_on_init_interval=*/false);

  ASSERT_THAT(ret, xls::status_testing::StatusIs(
                       absl::StatusCode::kFailedPrecondition,
                       testing::HasSubstr("Unable to parse")));
}

TEST_F(TranslatorProcTest, IOProcClassWithPipelinedLoop) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int val = 1;
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<3;++i) {
            val *= in.read();
          }
          out.write(val);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(5 * 7 * 10, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassWithPipelinedLoopInSubroutine) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

        int Read3() {
          int val = 1;
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<3;++i) {
            val *= in.read();
          }
          return val;
        }

         #pragma hls_top
         void Run() {
          out.write(Read3());
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(5 * 7 * 10, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassSubClass) {
  const std::string content = R"(
       class Sub {
        private:
         int val = 1;
        public:
         __xls_channel<long, __xls_channel_dir_Out>& out;

         Sub(__xls_channel<long, __xls_channel_dir_Out>& out) : out(out) {
         }

         void app(int x) {
          val *= x;
         }

        void wr() {
          out.write(val);
         }
         
       };

       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          Sub val(out);
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<3;++i) 
          {
            int xx = in.read();
            val.app(xx);
          }
          val.wr();
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(5 * 7 * 10, 64))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 3);
  }
}

TEST_F(TranslatorProcTest, IOProcClassSubClass2) {
  const std::string content = R"(
       class Sub {
        private:
         int val = 1;
        public:
         __xls_channel<long, __xls_channel_dir_Out>& out;

         Sub(__xls_channel<long, __xls_channel_dir_Out>& out) : out(out) {
         }

         void app(int x) {
          val *= x;
         }

        void wr() {
          out.write(val);
         }
         
       };

       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          static Sub val(out);
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<3;++i) {
            int xx = in.read();
            val.app(xx);
          }
          val.wr();
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_.reset(new xls::Package("my_package"));
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(
      ret.status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Statics containing lvalues not yet supported")));
}

TEST_F(TranslatorProcTest, IOProcClassLValueMember) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         int st = 10;

         int *ptr;

         #pragma hls_top
         void Run() {

          st += 3*in.read();

          out.write(st);
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_.reset(new xls::Package("my_package"));
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr(
                      "Don't know how to create LValue for member ptr")));
}

TEST_F(TranslatorProcTest, IOProcClassDirectIn) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out1;
         __xls_channel<long, __xls_channel_dir_Out>& out2;

         const int &dir;

         #pragma hls_top
         void Run() {
          const int val = in.read();

          if(dir) {
            out1.write(val);
          } else {
            out2.write(val);
          }
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_.reset(new xls::Package("my_package"));
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("direct-ins not implemented yet")));
}

}  // namespace

}  // namespace xlscc
