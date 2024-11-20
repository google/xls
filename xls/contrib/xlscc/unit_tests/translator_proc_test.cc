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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_base.h"

namespace xlscc {
namespace {

using ::testing::AllOf;
using ::testing::Not;
using ::testing::Values;

struct TestParams {
  bool generate_fsms_for_pipelined_loops;
  bool merge_states;
  bool split_states_on_channel_ops;
};

class TranslatorProcTestWithoutFSMParam : public XlsccTestBase {
 public:
  TranslatorProcTestWithoutFSMParam() {
    generate_fsms_for_pipelined_loops_ = false;
    merge_states_ = false;
    split_states_on_channel_ops_ = false;
  }
};

class TranslatorProcTest : public TranslatorProcTestWithoutFSMParam,
                           public ::testing::WithParamInterface<TestParams> {
 protected:
  TranslatorProcTest() {
    generate_fsms_for_pipelined_loops_ =
        GetParam().generate_fsms_for_pipelined_loops;
    merge_states_ = GetParam().merge_states;
    split_states_on_channel_ops_ = GetParam().split_states_on_channel_ops;
  }
};

std::string GetTestInfo(const ::testing::TestParamInfo<TestParams>& info) {
  return absl::StrFormat(
      "With%sFSM%s%s",
      info.param.generate_fsms_for_pipelined_loops ? "" : "out",
      info.param.split_states_on_channel_ops ? "_SplitOnChannelOps" : "",
      info.param.merge_states ? "_MergeStates" : "");
}

INSTANTIATE_TEST_SUITE_P(
    TranslatorProcTest, TranslatorProcTest,
    Values(TestParams{.generate_fsms_for_pipelined_loops = true,
                      .merge_states = false,
                      .split_states_on_channel_ops = false},
           TestParams{.generate_fsms_for_pipelined_loops = true,
                      .merge_states = true,
                      .split_states_on_channel_ops = false},
           TestParams{.generate_fsms_for_pipelined_loops = true,
                      .merge_states = false,
                      .split_states_on_channel_ops = true},
           TestParams{.generate_fsms_for_pipelined_loops = true,
                      .merge_states = true,
                      .split_states_on_channel_ops = true},
           TestParams{.generate_fsms_for_pipelined_loops = false}),
    GetTestInfo);

TEST_P(TranslatorProcTest, IOProcMux) {
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

// Multiple unused channels for determinism check
TEST_P(TranslatorProcTest, IOProcUnusedChannels) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(const int& dir,
              __xls_channel<int>& in,
              __xls_channel<int>& in_unused1,
              __xls_channel<int>& in_unused2,
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

    HLSChannel* ch_in_unused1 = block_spec.add_channels();
    ch_in_unused1->set_name("in_unused1");
    ch_in_unused1->set_is_input(true);
    ch_in_unused1->set_type(FIFO);

    HLSChannel* ch_in_unused2 = block_spec.add_channels();
    ch_in_unused2->set_name("in_unused2");
    ch_in_unused2->set_is_input(true);
    ch_in_unused2->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out1");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out2");
    ch_out2->set_is_input(false);
    ch_out2->set_type(FIFO);
  }

  // Check that dummy op is present
  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  XLS_ASSERT_OK(
      translator_->GenerateIR_Block(package_.get(), block_spec).status());

  {
    XLS_ASSERT_OK_AND_ASSIGN(xls::Channel * channel,
                             package_->GetChannel("in_unused1"));
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Node*> ops,
                             GetOpsForChannelNameContains(channel->name()));
    EXPECT_FALSE(ops.empty());
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(xls::Channel * channel,
                             package_->GetChannel("in_unused2"));
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Node*> ops,
                             GetOpsForChannelNameContains(channel->name()));
    EXPECT_FALSE(ops.empty());
  }

  // Check functionality
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

TEST_P(TranslatorProcTest, IOProcUnusedLocal) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int value = in.read();

          static __xls_channel<int> unused;
          unused.write(value);

          out.write(3*value);
        }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  XLS_ASSERT_OK(
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
          .status());

  {
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Node*> ops,
                             GetOpsForChannelNameContains("unused"));
    EXPECT_EQ(ops.size(), 2);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(3L * 11L, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 1,
             /* max_ticks = */ 100,
             /* top_level_init_interval = */ 1);
  }
}

TEST_P(TranslatorProcTest, IOProcUnusedDirectIn) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(const int& dir_unused,
              __xls_channel<int>& in,
              __xls_channel<int>& out) {
      (void)dir_unused;
      const int ctrl = in.read();
      out.write(ctrl);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir_unused");
    dir_in->set_is_input(true);
    dir_in->set_type(DIRECT_IN);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  // Check that dummy op is present
  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  XLS_ASSERT_OK(
      translator_->GenerateIR_Block(package_.get(), block_spec).status());

  XLS_ASSERT_OK_AND_ASSIGN(xls::Channel * channel,
                           package_->GetChannel("dir_unused"));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Node*> ops,
                           GetOpsForChannelNameContains(channel->name()));

  EXPECT_FALSE(ops.empty());

  // Check functionality
  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir_unused"] = {xls::Value(xls::SBits(0, 32))};
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, IOProcMux2) {
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

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir"] = {xls::Value(xls::SBits(0, 32))};
    inputs["in1"] = {xls::Value(xls::SBits(55, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(55, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir"] = {xls::Value(xls::SBits(1, 32))};
    inputs["in2"] = {xls::Value(xls::SBits(77, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(77, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, IOProcOneOp) {
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

TEST_P(TranslatorProcTest, IOProcOneLine) {
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

TEST_P(TranslatorProcTest, IOProcMuxMethod) {
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

TEST_P(TranslatorProcTest, IOProcMuxConstDir) {
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

TEST_P(TranslatorProcTest, IOProcChainedConditionalRead) {
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

TEST_P(TranslatorProcTest, IOProcStaticClassState) {
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

TEST_P(TranslatorProcTest, ForPipelined) {
  const std::string content = R"(
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(15, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32)),
                      xls::Value(xls::SBits(15 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_out,
                           GetBitsForChannelNameContains("__for_1_ctx_out"));
  EXPECT_EQ(channel_bits_out,
            generate_fsms_for_pipelined_loops_ ? 0L : 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_in,
                           GetBitsForChannelNameContains("__for_1_ctx_in"));
  EXPECT_EQ(channel_bits_in, generate_fsms_for_pipelined_loops_ ? 0L : 32);
}

TEST_P(TranslatorProcTest, ForPipelined1Iter) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();
      a += in.read();
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=3;++i) {
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
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)),  xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(100, 32)), xls::Value(xls::SBits(10, 32)),
      xls::Value(xls::SBits(15, 32)),  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 5 + 1 + 1 * (2 + 3), 32)),
                      xls::Value(xls::SBits(100 + 10 + 1 + 1 * (2 + 3), 32)),
                      xls::Value(xls::SBits(15 + 20 + 1 + 1 * (2 + 3), 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 1);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedFSMInside) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        int a = in.read();
        __xlscc_trace("!! after read {:u}", a);
        for(int j=0;j<1;++j) {
          __xlscc_trace("!! inner loop {:u}", a);
          a += i;
        }
        __xlscc_trace("!! before write {:u}", a);
        out.write(a);
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
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(100, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(0, 32)),
      xls::Value(xls::SBits(3, 32)),  xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(11, 32)), xls::Value(xls::SBits(13, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(80 + 1, 32)), xls::Value(xls::SBits(100 + 2, 32)),
        xls::Value(xls::SBits(10 + 3, 32)), xls::Value(xls::SBits(0 + 4, 32)),
        xls::Value(xls::SBits(3 + 1, 32)),  xls::Value(xls::SBits(5 + 2, 32)),
        xls::Value(xls::SBits(11 + 3, 32)), xls::Value(xls::SBits(13 + 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedFSMInside2) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        int a = in.read();
        for(int j=0;j<2;++j) {
          a += i;
        }
        out.write(a);
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
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)),  xls::Value(xls::SBits(100, 32)),
      xls::Value(xls::SBits(11, 32)),  xls::Value(xls::SBits(14, 32)),
      xls::Value(xls::SBits(135, 32)), xls::Value(xls::SBits(22, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 2 * 1, 32)),
                      xls::Value(xls::SBits(100 + 2 * 2, 32)),
                      xls::Value(xls::SBits(11 + 2 * 1, 32)),
                      xls::Value(xls::SBits(14 + 2 * 2, 32)),
                      xls::Value(xls::SBits(135 + 2 * 1, 32)),
                      xls::Value(xls::SBits(22 + 2 * 2, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedFSMInside3) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        for(int j=0;j<1;++j) {
          int a = in.read();
          a += i;
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
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(100, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(0, 32)),
      xls::Value(xls::SBits(13, 32)), xls::Value(xls::SBits(17, 32)),
      xls::Value(xls::SBits(3, 32)),  xls::Value(xls::SBits(1, 32)),
      xls::Value(xls::SBits(55, 32)), xls::Value(xls::SBits(35, 32)),
      xls::Value(xls::SBits(6, 32)),  xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(80 + 1, 32)), xls::Value(xls::SBits(100 + 2, 32)),
        xls::Value(xls::SBits(10 + 3, 32)), xls::Value(xls::SBits(0 + 4, 32)),
        xls::Value(xls::SBits(13 + 1, 32)), xls::Value(xls::SBits(17 + 2, 32)),
        xls::Value(xls::SBits(3 + 3, 32)),  xls::Value(xls::SBits(1 + 4, 32)),
        xls::Value(xls::SBits(55 + 1, 32)), xls::Value(xls::SBits(35 + 2, 32)),
        xls::Value(xls::SBits(6 + 3, 32)),  xls::Value(xls::SBits(11 + 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedFSMInside4) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        for(int j=0;j<2;++j) {
          int a = in.read();
          a += i+j;
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
  inputs["in"] = {
      // First activation
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(100, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(0, 32)),
      xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(6, 32)),
      xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(1, 32)),
      // Second activation
      xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(6, 32)), xls::Value(xls::SBits(11, 32)),
      xls::Value(xls::SBits(45, 32)), xls::Value(xls::SBits(32, 32)),
      xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(3, 32)),
      // Third activation
      xls::Value(xls::SBits(40, 32)), xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(60, 32)), xls::Value(xls::SBits(11, 32)),
      xls::Value(xls::SBits(450, 32)), xls::Value(xls::SBits(32, 32)),
      xls::Value(xls::SBits(30, 32)), xls::Value(xls::SBits(3, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {// First activation
                      xls::Value(xls::SBits(80 + 1 + 0, 32)),
                      xls::Value(xls::SBits(100 + 1 + 1, 32)),
                      xls::Value(xls::SBits(10 + 2 + 0, 32)),
                      xls::Value(xls::SBits(0 + 2 + 1, 32)),
                      xls::Value(xls::SBits(2 + 3 + 0, 32)),
                      xls::Value(xls::SBits(6 + 3 + 1, 32)),
                      xls::Value(xls::SBits(5 + 4 + 0, 32)),
                      xls::Value(xls::SBits(1 + 4 + 1, 32)),
                      // Second activation
                      xls::Value(xls::SBits(4 + 1 + 0, 32)),
                      xls::Value(xls::SBits(5 + 1 + 1, 32)),
                      xls::Value(xls::SBits(6 + 2 + 0, 32)),
                      xls::Value(xls::SBits(11 + 2 + 1, 32)),
                      xls::Value(xls::SBits(45 + 3 + 0, 32)),
                      xls::Value(xls::SBits(32 + 3 + 1, 32)),
                      xls::Value(xls::SBits(3 + 4 + 0, 32)),
                      xls::Value(xls::SBits(3 + 4 + 1, 32)),
                      // Third activation
                      xls::Value(xls::SBits(40 + 1 + 0, 32)),
                      xls::Value(xls::SBits(5 + 1 + 1, 32)),
                      xls::Value(xls::SBits(60 + 2 + 0, 32)),
                      xls::Value(xls::SBits(11 + 2 + 1, 32)),
                      xls::Value(xls::SBits(450 + 3 + 0, 32)),
                      xls::Value(xls::SBits(32 + 3 + 1, 32)),
                      xls::Value(xls::SBits(30 + 4 + 0, 32)),
                      xls::Value(xls::SBits(3 + 4 + 1, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedFSMInside5) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        int a = in.read();
        for(int j=0;j<1;++j) {
          a += i * in.read();
        }
        out.write(a);
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
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)),  xls::Value(xls::SBits(1, 32)),
      xls::Value(xls::SBits(100, 32)), xls::Value(xls::SBits(2, 32)),
      xls::Value(xls::SBits(10, 32)),  xls::Value(xls::SBits(3, 32)),
      xls::Value(xls::SBits(0, 32)),   xls::Value(xls::SBits(4, 32)),
      xls::Value(xls::SBits(3, 32)),   xls::Value(xls::SBits(2, 32)),
      xls::Value(xls::SBits(5, 32)),   xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(11, 32)),  xls::Value(xls::SBits(2, 32)),
      xls::Value(xls::SBits(13, 32)),  xls::Value(xls::SBits(1, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 1 * 1, 32)),
                      xls::Value(xls::SBits(100 + 2 * 2, 32)),
                      xls::Value(xls::SBits(10 + 3 * 3, 32)),
                      xls::Value(xls::SBits(0 + 4 * 4, 32)),
                      xls::Value(xls::SBits(3 + 1 * 2, 32)),
                      xls::Value(xls::SBits(5 + 2 * 5, 32)),
                      xls::Value(xls::SBits(11 + 3 * 2, 32)),
                      xls::Value(xls::SBits(13 + 4 * 1, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedFSMInside6) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in1,
             __xls_channel<int>& in2,
             __xls_channel<int>& in3,
             __xls_channel<int>& out) {

      int outer_r = in1.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        int a = in2.read();
        for(int j=0;j<2;++j) {
          int b = in3.read();
          for(int k=0;k<2;++k) {
            a += b + i * outer_r;
          }
          __xlscc_trace("Make another state");
        }
        out.write(a);
      }

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

    HLSChannel* ch_in3 = block_spec.add_channels();
    ch_in3->set_name("in3");
    ch_in3->set_is_input(true);
    ch_in3->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in1"] = {
      xls::Value(xls::SBits(30, 32)),
      xls::Value(xls::SBits(75, 32)),
  };

  inputs["in2"] = {
      // First
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(100, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(0, 32)),

      // Second
      xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(11, 32)), xls::Value(xls::SBits(13, 32))};

  inputs["in3"] = {
      // First
      xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(3, 32)),
      // Second
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(11, 32)),
      // Third
      xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(6, 32)),
      // Fourth
      xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(1, 32)),

      // First
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(11, 32)),
      // Second
      xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(3, 32)),
      // Third
      xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(6, 32)),
      // Fourth
      xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(1, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(80 + 4 * 1 * 30 + 2 + 2 + 3 + 3, 32)),
        xls::Value(xls::SBits(100 + 4 * 2 * 30 + 10 + 10 + 11 + 11, 32)),
        xls::Value(xls::SBits(10 + 4 * 3 * 30 + 5 + 5 + 6 + 6, 32)),
        xls::Value(xls::SBits(0 + 4 * 4 * 30 + 1 + 1 + 1 + 1, 32)),

        xls::Value(xls::SBits(3 + 4 * 1 * 75 + 10 + 10 + 11 + 11, 32)),
        xls::Value(xls::SBits(5 + 4 * 2 * 75 + 2 + 2 + 3 + 3, 32)),
        xls::Value(xls::SBits(11 + 4 * 3 * 75 + 5 + 5 + 6 + 6, 32)),
        xls::Value(xls::SBits(13 + 4 * 4 * 75 + 1 + 1 + 1 + 1, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 16);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedJustAssign) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a = i;
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
                  xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4, 32)),
                      xls::Value(xls::SBits(4, 32)),
                      xls::Value(xls::SBits(4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  int64_t loop_proc_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_proc_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_proc_bits : 0L);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_out,
                           GetBitsForChannelNameContains("__for_1_ctx_out"));
  EXPECT_EQ(channel_bits_out, generate_fsms_for_pipelined_loops_ ? 0L : 1 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_in,
                           GetBitsForChannelNameContains("__for_1_ctx_in"));
  EXPECT_EQ(channel_bits_in, generate_fsms_for_pipelined_loops_ ? 0L : 32);
}

TEST_P(TranslatorProcTest, ForPipelinedUseReceivedFromOldState) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();
      int off = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        off += i;
      }

      out.write(a + off);
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
                  xls::Value(xls::SBits(12, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32)),
                      xls::Value(xls::SBits(12 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  const int64_t loop_proc_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_proc_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_proc_bits : 0L);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_out,
                           GetBitsForChannelNameContains("__for_1_ctx_out"));
  EXPECT_EQ(channel_bits_out,
            generate_fsms_for_pipelined_loops_ ? 0L : 1 + 32 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_in,
                           GetBitsForChannelNameContains("__for_1_ctx_in"));
  EXPECT_EQ(channel_bits_in, generate_fsms_for_pipelined_loops_ ? 0L : 32);
}

TEST_P(TranslatorProcTest, ForPipelinedAttribute) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        int a = in.read();

        [[xlscc::hls_pipeline_init_interval(1)]]
        for(long i=1;i<=4;++i) {
          a += i;
        }

        out.write(a);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(80, 32)),
                  xls::Value(xls::SBits(100, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedAttributeIgnoreLabel) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        int a = in.read();

        foo:
        [[xlscc::hls_pipeline_init_interval(1)]]
        for(long i=1;i<=4;++i) {
          a += i;
        }

        out.write(a);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(80, 32)),
                  xls::Value(xls::SBits(100, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedDontPropagateAttribute) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        int a = in.read();

        [[xlscc::hls_pipeline_init_interval(1)]] for(long i=1;i<=4;++i) {
          a += i;
        }

        for(long j=1;j<=4;++j) {
          a += j*3;
        }

        out.write(a);
      }
    };
  )";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("missing")));
}

TEST_P(TranslatorProcTest, ForPipelinedII2) {
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

  const int loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedII2Error) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("nly initiation interval 1")));
}

TEST_P(TranslatorProcTest, WhilePipelined) {
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(13, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10, 32)),
                      xls::Value(xls::SBits(100 + 10, 32)),
                      xls::Value(xls::SBits(13 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, DoWhilePipelined) {
  const std::string content = R"(
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
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : (1 + 64 + 32 + 32));

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (1 + 64 + 32) : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedSerial) {
  const std::string content = R"(
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(13, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10 + 20, 32)),
                      xls::Value(xls::SBits(100 + 10 + 20, 32)),
                      xls::Value(xls::SBits(13 + 10 + 20, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 10);
  }

  const int64_t loop_1_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_1_body_bits);

  const int64_t loop_2_body_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_2_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, generate_fsms_for_pipelined_loops_
                                     ? (32 + 1 + 64 + 1 + 16 + 32)
                                     : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedSerialPhasedAndNested) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      static int a = -1;
      static int phase = 0;

      switch (phase) {
        case 0: {
          a = in.read();
          break;
        }
        case 1: {
          #pragma hls_pipeline_init_interval 1
          for(long i=1;i<=5;++i) {
            #pragma hls_pipeline_init_interval 1
            for(long j=1;j<=2;++j) {
              a += 1;
            }
          }
          break;
        }
        case 2: {
          #pragma hls_pipeline_init_interval 1
          for(short i=0;i<2;++i) {
            a += 10;
          }
          break;
        }
        case 3: {
          out.write(a);
          break;
        }
      }

      if (phase == 3) {
        phase = 0;
      } else {
        ++phase;
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
  inputs["in"] = {xls::Value(xls::SBits(80, 32)),
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(13, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10 + 20, 32)),
                      xls::Value(xls::SBits(100 + 10 + 20, 32)),
                      xls::Value(xls::SBits(13 + 10 + 20, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 10);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedSerialIO) {
  const std::string content = R"(
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
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(20, 32)),
      xls::Value(xls::SBits(4, 32)),  xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(6, 32)),  xls::Value(xls::SBits(7, 32)),
      xls::Value(xls::SBits(2, 32)),  xls::Value(xls::SBits(4, 32)),
      xls::Value(xls::SBits(40, 32)), xls::Value(xls::SBits(5, 32)),
      xls::Value(xls::SBits(60, 32)), xls::Value(xls::SBits(7, 32)),
      xls::Value(xls::SBits(2, 32)),  xls::Value(xls::SBits(4, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(2 + 6 + 10 + 2 + 3 * (10 + 20), 32)),
        xls::Value(xls::SBits(4 + 5 + 6 + 7 + 3 * (2 + 4), 32)),
        xls::Value(xls::SBits(40 + 5 + 60 + 7 + 3 * (2 + 4), 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 5);
  }

  const int64_t loop_1_body_bits = 1 + 32 + 64;
  const int64_t loop_2_body_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_1_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_2_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, generate_fsms_for_pipelined_loops_
                                     ? (32 + 1 + 64 + 1 + 16 + 32)
                                     : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedReturnInBody) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("eturns in pipelined loop body unimplemented")));
}

TEST_P(TranslatorProcTest, ForPipelinedMoreVars) {
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
  inputs["in"] = {
      xls::Value(xls::SBits(100, 16)), xls::Value(xls::SBits(5, 16)),
      xls::Value(xls::SBits(30, 16)),  xls::Value(xls::SBits(7, 16)),
      xls::Value(xls::SBits(1, 16)),   xls::Value(xls::SBits(2, 16))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(100 + 5 * 5 + 3, 32)),
                      xls::Value(xls::SBits(30 + 5 * 7 + 4, 32)),
                      xls::Value(xls::SBits(1 + 5 * 2 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 10);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : (1 + 32 + 64 + 64));

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            16 + (generate_fsms_for_pipelined_loops_ ? (1 + 32 + 64) : 0L));

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_out,
                           GetBitsForChannelNameContains("__for_1_ctx_out"));
  EXPECT_EQ(channel_bits_out,
            generate_fsms_for_pipelined_loops_ ? 0L : 1 + 32 + 64 + 64);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_in,
                           GetBitsForChannelNameContains("__for_1_ctx_in"));
  EXPECT_EQ(channel_bits_in, generate_fsms_for_pipelined_loops_ ? 0L : 32);
}

TEST_P(TranslatorProcTest, ForPipelinedMoreVars2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& out) {
      static short st = 3;
      ++st;
      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=2;i<=6;++i) {
        a += st;
      }

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;

    outputs["out"] = {xls::Value(xls::SBits(5 * 4, 32)),
                      xls::Value(xls::SBits(5 * 5, 32)),
                      xls::Value(xls::SBits(5 * 6, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 10);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedMoreVarsClass) {
  const std::string content = R"(
    class Block {
    public:
      __xls_channel<int, __xls_channel_dir_Out>& out;

      short st = 3;

      #pragma hls_top
      void foo() {
        ++st;
        int a = 0;

        #pragma hls_pipeline_init_interval 1
        for(long i=2;i<=6;++i) {
          a += st;
        }

        out.write(a);
      }
    };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(5 * 4, 32)),
                      xls::Value(xls::SBits(5 * 5, 32)),
                      xls::Value(xls::SBits(5 * 6, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedBlank) {
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(13, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 6, 32)),
                      xls::Value(xls::SBits(100 + 6, 32)),
                      xls::Value(xls::SBits(13 + 6, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedPreCondBreak) {
  const std::string content = R"(
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
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(-1, 32)),
                  xls::Value(xls::SBits(1, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(5 + 2 * 5, 32)),
                      xls::Value(xls::SBits(-1, 32)),
                      xls::Value(xls::SBits(1 + 2 * 1, 32))};

    // 2 ticks since messages must pass back and forth between procs
    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  // r shouldn't be in state
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : (1 + 32 + 32 + 64));

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (1 + 32 + 64) : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedSaveAcrossLoop) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int x = in.read();
      int a = 1;
      __xlscc_trace("read {:u}", x);

      // the return from in.read() hasn't been saved because
      // there is only 1 state
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += i;
      }

      __xlscc_trace("write x {:u} a {:u}", x, a);
      out.write(x + a);
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
  inputs["in"] = {xls::Value(xls::SBits(2, 32)),
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32)),
                  xls::Value(xls::SBits(5, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(2 + 11, 32)),
                      xls::Value(xls::SBits(100 + 11, 32)),
                      xls::Value(xls::SBits(0 + 11, 32)),
                      xls::Value(xls::SBits(0 + 11, 32)),
                      xls::Value(xls::SBits(5 + 11, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedInIf) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      // a doesn't get saved in the false condition
      // and the return from in.read() hasn't been saved because
      // there is only 1 state
      if (a >= 10) {
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
  inputs["in"] = {xls::Value(xls::SBits(2, 32)),
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32)),
                  xls::Value(xls::SBits(5, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(100 + 10, 32)),
        xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32)),
        xls::Value(xls::SBits(5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedIfInBody) {
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(33, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10 - 4, 32)),
                      xls::Value(xls::SBits(100 + 10 - 4, 32)),
                      xls::Value(xls::SBits(33 + 10 - 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedContinue) {
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(33, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 10 - 3, 32)),
                      xls::Value(xls::SBits(100 + 10 - 3, 32)),
                      xls::Value(xls::SBits(33 + 10 - 3, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedBreak) {
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(83, 32)),
                      xls::Value(xls::SBits(103, 32)),
                      xls::Value(xls::SBits(23, 32))};
    outputs["out_i"] = {xls::Value(xls::SBits(3, 64)),
                        xls::Value(xls::SBits(3, 64)),
                        xls::Value(xls::SBits(3, 64))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 6);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedInFunction) {
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

  int64_t loop_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedInMethod) {
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

  const int64_t loop_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedInMethodWithMember) {
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

  const int64_t loop_bits = 1 + 32 + 64 + 32;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            32 + (generate_fsms_for_pipelined_loops_ ? loop_bits : 0L));
}

TEST_P(TranslatorProcTest, ForPipelinedInFunctionInIf) {
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

  const int64_t loop_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedStaticInBody) {
  const std::string content = R"(
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
  package_ = std::make_unique<xls::Package>("my_package");

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(200, 32)),
                  xls::Value(xls::SBits(50, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(100 + 3 + 4 + 5 + 6, 32)),
                      xls::Value(xls::SBits(200 + 7 + 8 + 9 + 10, 32)),
                      xls::Value(xls::SBits(50 + 11 + 12 + 13 + 14, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  const int64_t loop_proc_bits = 1 + 16 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_proc_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_proc_bits : 0L);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_out,
                           GetBitsForChannelNameContains("__for_1_ctx_out"));
  EXPECT_EQ(channel_bits_out,
            generate_fsms_for_pipelined_loops_ ? 0L : 1 + 32 + 16);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t channel_bits_in,
                           GetBitsForChannelNameContains("__for_1_ctx_in"));
  EXPECT_EQ(channel_bits_in, generate_fsms_for_pipelined_loops_ ? 0L : 32);
}

TEST_P(TranslatorProcTest, ForPipelinedStaticOuter) {
  const std::string content = R"(
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(30, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 4 * 3, 32)),
                      xls::Value(xls::SBits(100 + 4 * 4, 32)),
                      xls::Value(xls::SBits(30 + 4 * 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : (1 + 32 + 16 + 64));

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            16 + (generate_fsms_for_pipelined_loops_ ? (1 + 32 + 64) : 0L));
}

TEST_P(TranslatorProcTest, ForPipelinedStaticOuter2) {
  const std::string content = R"(
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(30, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 3 + 4 + 5 + 6, 32)),
                      xls::Value(xls::SBits(100 + 7 + 8 + 9 + 10, 32)),
                      xls::Value(xls::SBits(30 + 11 + 12 + 13 + 14, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 8);
  }

  const int64_t loop_body_bits = 1 + 32 + 64 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            (generate_fsms_for_pipelined_loops_ ? (32 + 16 + 1 + 64) : 16));
}

TEST_P(TranslatorProcTest, ForPipelinedStaticOuter3) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();
      static short st = 3;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=0;++i) {
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
                  xls::Value(xls::SBits(100, 32)),
                  xls::Value(xls::SBits(30, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80, 32)),
                      xls::Value(xls::SBits(100, 32)),
                      xls::Value(xls::SBits(30, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  const int64_t loop_body_bits = 1 + 32 + 64 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 16 + 1 + 64) : 16);
}

TEST_P(TranslatorProcTest, ForPipelinedOneState) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      #pragma hls_pipeline_init_interval 1
      for(long j=1;j<=4;++j) {
        int a = in.read();
        a += j;
        out.write(a);
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
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(100, 32)),
      xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(35, 32)),
      xls::Value(xls::SBits(5, 32)),  xls::Value(xls::SBits(15, 32)),
      xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(25, 32)),
      xls::Value(xls::SBits(2, 32)),  xls::Value(xls::SBits(4, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(12, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(80 + 1, 32)), xls::Value(xls::SBits(100 + 2, 32)),
        xls::Value(xls::SBits(20 + 3, 32)), xls::Value(xls::SBits(35 + 4, 32)),
        xls::Value(xls::SBits(5 + 1, 32)),  xls::Value(xls::SBits(15 + 2, 32)),
        xls::Value(xls::SBits(20 + 3, 32)), xls::Value(xls::SBits(25 + 4, 32)),
        xls::Value(xls::SBits(2 + 1, 32)),  xls::Value(xls::SBits(4 + 2, 32)),
        xls::Value(xls::SBits(10 + 3, 32)), xls::Value(xls::SBits(12 + 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedNested) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(short i=1;i<=4;++i) {
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

  const int64_t inner_loop_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : inner_loop_bits);

  const int64_t outer_loop_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : outer_loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 1 + 16 + 1 + 64) : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedNestedBreak) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& inA,
             __xls_channel<int>& inB,
             __xls_channel<int>& out) {
      int a = inA.read();
      int b = inB.read();

      #pragma hls_pipeline_init_interval 1
      for(short i=1;i<=4;++i) {
        #pragma hls_pipeline_init_interval 1
        for(long j=1;j<=4;++j) {
          ++a;
        }
        if(i == b) {
          break;
        }
      }

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* chA_in = block_spec.add_channels();
    chA_in->set_name("inA");
    chA_in->set_is_input(true);
    chA_in->set_type(FIFO);

    HLSChannel* chB_in = block_spec.add_channels();
    chB_in->set_name("inB");
    chB_in->set_is_input(true);
    chB_in->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["inA"] = {xls::Value(xls::SBits(80, 32)),
                   xls::Value(xls::SBits(100, 32)),
                   xls::Value(xls::SBits(20, 32))};

  inputs["inB"] = {xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(1, 32)),
                   xls::Value(xls::SBits(2, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(80 + 4 * 4, 32)),
                      xls::Value(xls::SBits(100 + 4 * 1, 32)),
                      xls::Value(xls::SBits(20 + 4 * 2, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 16);
  }
}

// Test that the FSM is generated
TEST_F(TranslatorProcTestWithoutFSMParam, ForPipelinedCheckFSM) {
  const std::string content = R"(
  class Block {
     public:
      __xls_channel<int , __xls_channel_dir_In>& in;
      __xls_channel<long, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void foo() {
        int a = in.read();
        a += 3*in.read();

        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=4;++i) {
          a += i;
        }

        out.write(a);
      }
    };)";

  generate_fsms_for_pipelined_loops_ = true;

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  XLS_ASSERT_OK(
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
          .status());

  absl::flat_hash_map<xls::Node*, int64_t> state_by_io_node;
  XLS_ASSERT_OK_AND_ASSIGN(state_by_io_node,
                           GetStatesByIONodeForFSMProc(/*func_name=*/"foo"));

  // Check that token network does not extend across states
  for (const auto& [node, state_index] : state_by_io_node) {
    absl::flat_hash_set<xls::Node*> deep_token_operands;
    GetTokenOperandsDeeply(node, deep_token_operands);

    for (xls::Node* token_operand : deep_token_operands) {
      if (!state_by_io_node.contains(token_operand)) {
        continue;
      }
      EXPECT_EQ(state_by_io_node.at(token_operand), state_index);
    }
  }
}

TEST_F(TranslatorProcTestWithoutFSMParam, ForPipelinedMergeCheckFSM) {
  const std::string content = R"(
  class Block {
     public:
      __xls_channel<int , __xls_channel_dir_In>& in;
      __xls_channel<long, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void foo() {
        int a = in.read();
        a += 3*in.read();

        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=4;++i) {
          a += i;
        }

        out.write(a);
      }
    };)";

  generate_fsms_for_pipelined_loops_ = true;
  merge_states_ = true;

  // Test full merging
  {
    split_states_on_channel_ops_ = false;

    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    XLS_ASSERT_OK(
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
            .status());

    absl::flat_hash_map<xls::Node*, int64_t> state_by_io_node;
    XLS_ASSERT_OK_AND_ASSIGN(state_by_io_node,
                             GetStatesByIONodeForFSMProc(/*func_name=*/"foo"));

    for (const auto& [node, state_index] : state_by_io_node) {
      EXPECT_EQ(state_index, 0);
    }
  }
  // Test split on same channel IO ops
  {
    split_states_on_channel_ops_ = true;

    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    XLS_ASSERT_OK(
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
            .status());

    absl::flat_hash_map<xls::Node*, int64_t> state_by_io_node;
    XLS_ASSERT_OK_AND_ASSIGN(state_by_io_node,
                             GetStatesByIONodeForFSMProc(/*func_name=*/"foo"));

    absl::flat_hash_set<int64_t> states_for_receives;

    for (const auto& [node, state_index] : state_by_io_node) {
      if (node->Is<xls::Receive>()) {
        states_for_receives.insert(state_index);
      } else {
        EXPECT_EQ(state_index, 1);
      }
    }

    EXPECT_EQ(states_for_receives.size(), 2);
  }
}

TEST_F(TranslatorProcTestWithoutFSMParam, ForPipelinedSplitSubProcCheckFSM) {
  const std::string content = R"(
  class Block {
     public:
      __xls_channel<int , __xls_channel_dir_In>& in;
      __xls_channel<long, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void foo() {
        int a = in.read();
        a += 3*in.read();

        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=4;++i) {
          a += i & in.read();
        }

        out.write(a);
      }
    };)";

  generate_fsms_for_pipelined_loops_ = true;
  merge_states_ = true;

  // Test full merging
  {
    split_states_on_channel_ops_ = false;

    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    XLS_ASSERT_OK(
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
            .status());

    absl::flat_hash_map<xls::Node*, int64_t> state_by_io_node;
    XLS_ASSERT_OK_AND_ASSIGN(state_by_io_node,
                             GetStatesByIONodeForFSMProc(/*func_name=*/"foo"));

    for (const auto& [node, state_index] : state_by_io_node) {
      EXPECT_EQ(state_index, 0);
    }
  }
  // Test split on same channel IO ops
  {
    split_states_on_channel_ops_ = true;

    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    XLS_ASSERT_OK(
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
            .status());

    absl::flat_hash_map<xls::Node*, int64_t> state_by_io_node;
    XLS_ASSERT_OK_AND_ASSIGN(state_by_io_node,
                             GetStatesByIONodeForFSMProc(/*func_name=*/"foo"));

    absl::flat_hash_set<int64_t> states_for_receives;

    for (const auto& [node, state_index] : state_by_io_node) {
      if (node->Is<xls::Receive>()) {
        states_for_receives.insert(state_index);
      } else {
        EXPECT_EQ(state_index, 2);
      }
    }

    EXPECT_EQ(states_for_receives.size(), 3);
  }
}

TEST_F(TranslatorProcTestWithoutFSMParam,
       ForPipelinedSplitNestedSubProcCheckFSM) {
  const std::string content = R"(
  class Block {
     public:
      __xls_channel<int , __xls_channel_dir_In>& in;
      __xls_channel<long, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void foo() {
        int a = in.read();
        a += 3*in.read();

        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=4;++i) {
          #pragma hls_pipeline_init_interval 1
          for(long j=1;j<=4;++j) {
            a += i & in.read();
          }
        }

        out.write(a);
      }
    };)";

  generate_fsms_for_pipelined_loops_ = true;
  merge_states_ = true;

  // Test full merging
  {
    split_states_on_channel_ops_ = false;

    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    XLS_ASSERT_OK(
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
            .status());

    absl::flat_hash_map<xls::Node*, int64_t> state_by_io_node;
    XLS_ASSERT_OK_AND_ASSIGN(state_by_io_node,
                             GetStatesByIONodeForFSMProc(/*func_name=*/"foo"));

    for (const auto& [node, state_index] : state_by_io_node) {
      EXPECT_EQ(state_index, 0);
    }
  }
  // Test split on same channel IO ops
  {
    split_states_on_channel_ops_ = true;

    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    XLS_ASSERT_OK(
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec)
            .status());

    absl::flat_hash_map<xls::Node*, int64_t> state_by_io_node;
    XLS_ASSERT_OK_AND_ASSIGN(state_by_io_node,
                             GetStatesByIONodeForFSMProc(/*func_name=*/"foo"));

    absl::flat_hash_set<int64_t> states_for_receives;

    for (const auto& [node, state_index] : state_by_io_node) {
      if (node->Is<xls::Receive>()) {
        states_for_receives.insert(state_index);
      } else {
        EXPECT_EQ(state_index, 2);
      }
    }

    EXPECT_EQ(states_for_receives.size(), 3);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedNested2) {
  const std::string content = R"(
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

  const int64_t loop_body_bits = 1 + 32 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 1 + 64 + 1 + 64) : 0);
}

TEST_P(TranslatorProcTest, ForPipelinedNestedInheritIIWithLabel) {
  const std::string content = R"(
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

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_3"));
  EXPECT_EQ(innermost_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t inner_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(inner_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 3 * (1 + 64)) : 0);
}

TEST_P(TranslatorProcTest, ForPipelinedNestedWithIO) {
  const std::string content = R"(
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
  inputs["in"] = {// First activation
                  xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(8, 32)),
                  xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(1, 32)),

                  // Second activation
                  xls::Value(xls::SBits(11, 32)),
                  xls::Value(xls::SBits(13, 32)), xls::Value(xls::SBits(6, 32)),
                  xls::Value(xls::SBits(7, 32)),

                  // Third activation
                  xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(2, 32)),
                  xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(6, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        // First activation
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(4 + 8, 32)),
        xls::Value(xls::SBits(4 + 8 + 3, 32)),
        xls::Value(xls::SBits(4 + 8 + 3 + 1, 32)),
        // Second activation
        xls::Value(xls::SBits(11, 32)), xls::Value(xls::SBits(11 + 13, 32)),
        xls::Value(xls::SBits(11 + 13 + 6, 32)),
        xls::Value(xls::SBits(11 + 13 + 6 + 7, 32)),
        // Third activation
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(1 + 2, 32)),
        xls::Value(xls::SBits(1 + 2 + 5, 32)),
        xls::Value(xls::SBits(1 + 2 + 5 + 6, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 1 + 64 + 1 + 64) : 0);
}

TEST_P(TranslatorProcTest, ForPipelinedNestedWithIO2) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& init,
             __xls_channel<int>& out) {
      const int n = init.read() - 5;
      __xlscc_trace("---- init.read()!");

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        __xlscc_trace("---- outer loop runs n={:u}", n);

        #pragma hls_pipeline_init_interval 1
        for(long j=1;j<=n;++j) {
          __xlscc_trace("---- extra loop runs n={:u}", n);
        }

        // using n instead of 2 breaks it
        // add i
        #pragma hls_pipeline_init_interval 1
        for(long j=1;j<=n;++j) {
          out.write(j);
          __xlscc_trace("---- out[{:u}].write()! n={:u}", j, n);
        }

      }

    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_init = block_spec.add_channels();
    ch_init->set_name("init");
    ch_init->set_is_input(true);
    ch_init->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  inputs["init"] = {xls::Value(xls::SBits(5 + 1, 32)),
                    xls::Value(xls::SBits(5 + 2, 32)),
                    xls::Value(xls::SBits(5 + 4, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        // First activation
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(1, 32)),
        // Second activation
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(2, 32)),
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(2, 32)),
        // Third activation
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(2, 32)),
        xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(4, 32)),
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(2, 32)),
        xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 1);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBody) {
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
  inputs["in"] = {
      xls::Value(xls::SBits(6, 32)),  xls::Value(xls::SBits(12, 32)),
      xls::Value(xls::SBits(1, 32)),  xls::Value(xls::SBits(3, 32)),
      xls::Value(xls::SBits(11, 32)), xls::Value(xls::SBits(15, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(6 + 12, 32)),
                      xls::Value(xls::SBits(1 + 3, 32)),
                      xls::Value(xls::SBits(11 + 15, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBody2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=1;++i) {
        a += in.read() + i;
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
  inputs["in"] = {xls::Value(xls::SBits(6, 32)), xls::Value(xls::SBits(12, 32)),
                  xls::Value(xls::SBits(15, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(6 + 1, 32)),
                      xls::Value(xls::SBits(12 + 1, 32)),
                      xls::Value(xls::SBits(15 + 1, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBody3) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=0;++i) {
        a += in.read() + i;
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
  inputs["in"] = {};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(0, 32)),
                      xls::Value(xls::SBits(0, 32)),
                      xls::Value(xls::SBits(0, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 3);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedNestedNoPragma) {
  const std::string content = R"(
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

  const int64_t loop_1_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t innermost_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(innermost_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_1_body_bits);

  const int64_t loop_2_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_2_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 1 + 64 + 1 + 64) : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubroutine) {
  const std::string content = R"(
    int sub_read(__xls_channel<int>& in_inner) {
      return in_inner.read();
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
  inputs["in"] = {xls::Value(xls::SBits(6, 32)), xls::Value(xls::SBits(12, 32)),
                  xls::Value(xls::SBits(15, 32)),
                  xls::Value(xls::SBits(3, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(6 + 12, 32)),
                      xls::Value(xls::SBits(15 + 3, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubroutine2) {
  const std::string content = R"(
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
    outputs["out"] = {xls::Value(xls::SBits(6 + 12, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }
  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubroutineMultiCall) {
  const std::string content = R"(
    bool call_b(long arg1) {
      long ret = arg1;
      #pragma hls_pipeline_init_interval 1
      for (int i=0; i < 2; i++) {
        ret = (ret << 1);
      }
      return ret > 0;
    }

    bool call_top(long arg1) {
        bool has_rem;
        if ( arg1 > 1 ) {
          has_rem = call_b(arg1);
        } else {
          has_rem = call_b(arg1);
        }
        return has_rem;
      }

    #pragma hls_top
    void run_design(__xls_channel<long> &in, __xls_channel<long> &out) {
        long dNum = in.read();

        bool tmp2 = false;
        tmp2 = call_top(dNum);

        out.write((long)tmp2);
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
  inputs["in"] = {xls::Value(xls::SBits(6, 64))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(1, 64))};

  ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubSubroutine) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_sub_read(__xls_channel<int>& in3) {
      int a = in3.read();
      return a;
    }

    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += sub_sub_read(in2);
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
    outputs["out"] = {xls::Value(xls::SBits(6 + 12, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubSubroutine2) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_sub_read(__xls_channel<int>& in3) {
      short a = 0;
      #pragma hls_pipeline_init_interval 1
      for(char i=1;i<=3;++i) {
        a += in3.read();
      }
      return a;
    }

    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += sub_sub_read(in2);
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
  inputs["in"] = {
      xls::Value(xls::SBits(6, 32)),  xls::Value(xls::SBits(12, 32)),
      xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(30, 32)),
      xls::Value(xls::SBits(1, 32)),  xls::Value(xls::SBits(2, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(6 + 12 + 20 + 30 + 1 + 2, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 6);
  }

  const int64_t innermost_loop_bits = 1 + 16 + 8;

  {
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                             GetStateBitsForProcNameContains("for_2"));
    EXPECT_EQ(body_proc_state_bits,
              generate_fsms_for_pipelined_loops_ ? 0L : innermost_loop_bits);
  }

  const int64_t loop_bits = 1 + 32 + 64;

  {
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                             GetStateBitsForProcNameContains("for_1"));
    EXPECT_EQ(body_proc_state_bits,
              generate_fsms_for_pipelined_loops_ ? 0L : loop_bits);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, generate_fsms_for_pipelined_loops_
                                     ? loop_bits + innermost_loop_bits
                                     : 0);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubSubroutine3) {
  const std::string content = R"(
    struct ChannelContainer {
      __xls_channel<int> ch;
    };

    int sub_sub_read(ChannelContainer& in3) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=3;++i) {
        a += in3.ch.read();
      }
      return a;
    }

    int sub_read(ChannelContainer& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += sub_sub_read(in2);
      }
      return a;
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      ChannelContainer container = {.ch = in};
      out.write(sub_read(container));
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
  package_ = std::make_unique<xls::Package>("my_package");
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("parameters containing LValues")));
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubSubroutine4) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    int sub_sub_sub_read(__xls_channel<int>& in4) {
      short a = 0;
      #pragma hls_pipeline_init_interval 1
      for(char i=1;i<=1;++i) {
        a += in4.read();
      }
      return a;
    }

    int sub_sub_read(__xls_channel<int>& in3) {
      short a = 0;
      #pragma hls_pipeline_init_interval 1
      for(char i=1;i<=3;++i) {
        a += sub_sub_sub_read(in3);
      }
      return a;
    }

    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += sub_sub_read(in2);
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
  inputs["in"] = {
      xls::Value(xls::SBits(6, 32)),  xls::Value(xls::SBits(12, 32)),
      xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(30, 32)),
      xls::Value(xls::SBits(1, 32)),  xls::Value(xls::SBits(2, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(6 + 12 + 20 + 30 + 1 + 2, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 6);
  }

  const int64_t sub_sub_sub_bits = 1 + 16 + 8;

  {
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                             GetStateBitsForProcNameContains("for_3"));
    EXPECT_EQ(body_proc_state_bits,
              generate_fsms_for_pipelined_loops_ ? 0L : sub_sub_sub_bits);
  }

  const int64_t sub_sub_bits = 1 + 16 + 8;

  {
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                             GetStateBitsForProcNameContains("for_2"));
    EXPECT_EQ(body_proc_state_bits,
              generate_fsms_for_pipelined_loops_ ? 0L : sub_sub_bits);
  }

  const int64_t sub_bits = 1 + 32 + 64;

  {
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                             GetStateBitsForProcNameContains("for_1"));
    EXPECT_EQ(body_proc_state_bits,
              generate_fsms_for_pipelined_loops_ ? 0L : sub_bits);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_
                ? (sub_bits + sub_sub_bits + sub_sub_sub_bits)
                : 0);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubroutineDeclOrder) {
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

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubroutine3) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  ASSERT_THAT(
      translator_->GenerateIR_Block(package_.get(), block_spec).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("ops in pipelined loops in subroutines called "
                             "with multiple different channel arguments")));
}

TEST_P(TranslatorProcTest, ForPipelinedIOInBodySubroutine4) {
  const std::string content = R"(
    int sub_read(__xls_channel<int>& in2) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=2;++i) {
        a += in2.read();
      }
      __xlscc_trace("!! sub_read ret a {:u}", a);
      return a;
    }

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int ret = sub_read(in);
      __xlscc_trace("!! foo A {:u}", ret);
      const int b = sub_read(in);
      __xlscc_trace("!! foo B b {:u} ret {:u}", b, ret);
      ret += b;
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
                  xls::Value(xls::SBits(10, 32)),
                  xls::Value(xls::SBits(20, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(6 + 12 + 10 + 20, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 2);
  }

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 1 + 64 + 1 + 32) : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedNoPragma) {
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

  const int64_t loop_body_bits = 1 + 32 + 64;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, PipelinedLoopUsingMemberChannel) {
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

  const int64_t loop_body_bits = 1 + 32 + 32;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? loop_body_bits : 0L);
}

TEST_P(TranslatorProcTest, PipelinedLoopUsingMemberChannelAndVariable) {
  const std::string content = R"(
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

  const int64_t loop_body_bits = 1 + 32 + 32;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : loop_body_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 1 + 32 + 1 + 32) : 0L);
}

TEST_P(TranslatorProcTest, PipelinedLoopWithOuterRef) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int result = 0;
          int& result_ref = result;
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<3;++i) {
            result_ref += in.read();
          }
          out.write(result);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(5 + 7 + 10, 64))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);
}

TEST_P(TranslatorProcTest, PipelinedLoopWithOuterChannelRef) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          __xls_channel<int , __xls_channel_dir_In>& in_ref = in;
          int result = 0;
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<3;++i) {
            result += in_ref.read();
          }
          out.write(result);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(5 + 7 + 10, 64))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);
}

TEST_P(TranslatorProcTest, IOProcClass) {
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

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(15, 64)),
                    xls::Value(xls::SBits(21, 64)),
                    xls::Value(xls::SBits(30, 64))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels 	 {
      name: "in"
      is_input: true
      type: FIFO
      width_in_bits: 32
    }
    channels {
      name: "out"
      is_input: false
      type: FIFO
      width_in_bits: 64
    }
    name: "Block"
  )";

  xlscc::HLSBlock ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_P(TranslatorProcTest, IOProcClassStructData) {
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

TEST_P(TranslatorProcTest, IOProcClassLocalStatic) {
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

TEST_P(TranslatorProcTest, IOProcClassState) {
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

TEST_P(TranslatorProcTest, IOProcClassNotInline) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         int st = 10;

         void Run();
      };

      #pragma hls_top
      void Block::Run() {
        st += 3*in.read();
        out.write(st);
      }
      )";

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

TEST_P(TranslatorProcTest, IOProcClassStateWithLocalStatic) {
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

TEST_P(TranslatorProcTest, IOProcClassConstTop) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("Const top")));
}

TEST_P(TranslatorProcTest, IOProcClassConstructor) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(
      ret.status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("onstructors in top class")));
}

TEST_P(TranslatorProcTest, IOProcClassNonVoidReturn) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(
      ret.status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("Non-void top method return")));
}

TEST_P(TranslatorProcTest, IOProcClassParameters) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("method parameters unsupported")));
}

TEST_P(TranslatorProcTest, IOProcClassStaticMethod) {
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

  ASSERT_THAT(ret, absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                          testing::HasSubstr("static member")));
}

TEST_P(TranslatorProcTest, IOProcClassWithPipelinedLoop) {
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

TEST_P(TranslatorProcTest, IOProcClassWithPipelinedLoopInSubroutine) {
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

TEST_P(TranslatorProcTest, IOProcClassSubClass) {
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

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(5 * 7 * 10, 64))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());
  const std::string ref_meta_str = R"(
    channels 	 {
      name: "in"
      is_input: true
      type: FIFO
      width_in_bits: 32
    }
    channels {
      name: "out"
      is_input: false
      type: FIFO
      width_in_bits: 64
    }
    name: "Block"
  )";

  xlscc::HLSBlock ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_P(TranslatorProcTest, IOProcClassSubClass2) {
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

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {
      xls::Value(xls::SBits(5, 32)),  xls::Value(xls::SBits(7, 32)),
      xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(2, 32)),
      xls::Value(xls::SBits(2, 32)),  xls::Value(xls::SBits(3, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(5 * 7 * 10, 64)),
                    xls::Value(xls::SBits(5 * 7 * 10 * 2 * 2 * 3, 64))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 6);
}

TEST_P(TranslatorProcTest, IOProcClassLValueMember) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr(
                      "Don't know how to create LValue for member ptr")));
}

TEST_P(TranslatorProcTest, IOProcClassDirectIn) {
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
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("direct-ins not implemented yet")));
}

TEST_P(TranslatorProcTest, IODefaultStrictness) {
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

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  XLS_ASSERT_OK(
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec));

  for (xls::Channel* channel : package_->channels()) {
    EXPECT_EQ(channel->kind(), xls::ChannelKind::kStreaming)
        << "Non-streaming channel: " << channel->name();
    if (channel->kind() != xls::ChannelKind::kStreaming) {
      continue;
    }
    EXPECT_EQ(xls::down_cast<xls::StreamingChannel*>(channel)->GetStrictness(),
              xls::ChannelStrictness::kProvenMutuallyExclusive)
        << "Incorrect strictness for channel: " << channel->name();
  }
}
TEST_P(TranslatorProcTest, IOWithStrictnessSpecified) {
  const std::string content = R"(
       class Block {
        public:
         [[xlscc::hls_channel_strictness(proven_mutually_exclusive)]]
         __xls_channel<int , __xls_channel_dir_In>& in;
         [[xlscc::hls_channel_strictness(arbitrary_static_order)]]
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          out.write(3*in.read());
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  XLS_ASSERT_OK(
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec));

  for (xls::Channel* channel : package_->channels()) {
    EXPECT_EQ(channel->kind(), xls::ChannelKind::kStreaming)
        << "Non-streaming channel: " << channel->name();
    if (channel->kind() != xls::ChannelKind::kStreaming) {
      continue;
    }
    EXPECT_THAT(channel->name(), testing::AnyOf("in", "out"));
    xls::ChannelStrictness expected_strictness =
        channel->name() == "in"
            ? xls::ChannelStrictness::kProvenMutuallyExclusive
            : xls::ChannelStrictness::kArbitraryStaticOrder;
    EXPECT_EQ(xls::down_cast<xls::StreamingChannel*>(channel)->GetStrictness(),
              expected_strictness)
        << "Incorrect strictness for channel: " << channel->name();
  }
}

TEST_P(TranslatorProcTest, IOWithStrictnessSpecifiedOnCommandLine) {
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

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  XLS_ASSERT_OK(translator_->GenerateIR_BlockFromClass(
      package_.get(), &block_spec, /*top_level_init_interval=*/0,
      /*channel_options=*/
      {.strictness_map = {
           {"in", xls::ChannelStrictness::kProvenMutuallyExclusive},
           {"out", xls::ChannelStrictness::kArbitraryStaticOrder}}}));

  for (xls::Channel* channel : package_->channels()) {
    EXPECT_EQ(channel->kind(), xls::ChannelKind::kStreaming)
        << "Non-streaming channel: " << channel->name();
    if (channel->kind() != xls::ChannelKind::kStreaming) {
      continue;
    }
    EXPECT_THAT(channel->name(), testing::AnyOf("in", "out"));
    xls::ChannelStrictness expected_strictness =
        channel->name() == "in"
            ? xls::ChannelStrictness::kProvenMutuallyExclusive
            : xls::ChannelStrictness::kArbitraryStaticOrder;
    EXPECT_EQ(xls::down_cast<xls::StreamingChannel*>(channel)->GetStrictness(),
              expected_strictness)
        << "Incorrect strictness for channel: " << channel->name();
  }
}

TEST_P(TranslatorProcTest, IOWithUnusedStrictnessesSpecifiedOnCommandLine) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         [[xlscc::hls_channel_strictness(arbitrary_static_order)]]
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          out.write(3*in.read());
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  ASSERT_THAT(
      translator_->GenerateIR_BlockFromClass(
          package_.get(), &block_spec, /*top_level_init_interval=*/0,
          /*channel_options=*/
          {.strictness_map =
               {{"in", xls::ChannelStrictness::kProvenMutuallyExclusive},
                {"in_unused", xls::ChannelStrictness::kProvenMutuallyExclusive},
                {"out", xls::ChannelStrictness::kArbitraryStaticOrder}}}),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(testing::HasSubstr("Unused channel strictness"),
                testing::HasSubstr("in_unused:proven_mutually_exclusive"),
                Not(testing::HasSubstr("out")))));
}

TEST_P(TranslatorProcTest, IOProcClassPropagateVars) {
  const std::string content = R"(
    class Block {
    public:
      int count = 10;
      __xls_channel<int, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void foo() {

        out.write(count);
        count++;

        if(count >= 3) {
          count = 0;
        }
      }
    };)";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(0, 32)),
        xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(2, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("Block_proc"));
  EXPECT_EQ(proc_state_bits, 32);
}

TEST_P(TranslatorProcTest, IOProcClassPropagateVars2) {
  const std::string content = R"(
    class Block {
    public:
      int count = 10;
      __xls_channel<int, __xls_channel_dir_Out>& out;

      void IncCount() {
        ++count;
      }

      #pragma hls_top
      void foo() {
        IncCount();
        out.write(count);
      }
    };)";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(11, 32)),
                      xls::Value(xls::SBits(12, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("Block_proc"));
  EXPECT_EQ(proc_state_bits, 32);
}

TEST_P(TranslatorProcTest, IOProcClassWithoutRef) {
  const std::string content = R"(
    class Block {
    public:
      int count = 10;
      __xls_channel<int, __xls_channel_dir_Out> out;

      void IncCount() {
        ++count;
      }

      #pragma hls_top
      void foo() {
        IncCount();
        out.write(count);
      }
    };)";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(11, 32)),
                      xls::Value(xls::SBits(12, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t proc_state_bits,
                           GetStateBitsForProcNameContains("Block_proc"));
  EXPECT_EQ(proc_state_bits, 32);
}

TEST_P(TranslatorProcTest, IOProcClassEnumMember) {
  const std::string content = R"(
       enum MyEnum {
         A = 3,
         B = 5,
       };

       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         MyEnum e = A;

         #pragma hls_top
         void Run() {
          out.write(3*in.read() + e);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(15 + 3, 64))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 1);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels 	 {
      name: "in"
      is_input: true
      type: FIFO
      width_in_bits: 32
    }
    channels {
      name: "out"
      is_input: false
      type: FIFO
      width_in_bits: 64
    }
    name: "Block"
  )";

  xlscc::HLSBlock ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_P(TranslatorProcTest, IOProcClassMemberSetConstruct) {
  const std::string content = R"(
      struct State {
        int val;
      };

      class Block {
      public:
        __xls_channel<int, __xls_channel_dir_In> in;
        __xls_channel<int, __xls_channel_dir_Out> out;

        State state;

        int Calculate() { return 11; }

        #pragma hls_top
        void Run() {
          state = State();
          auto x = in.read();
          int tmp = Calculate();
          out.write(tmp + x);
        }
      };
)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(11 + 5, 32))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 1);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels 	 {
      name: "in"
      is_input: true
      type: FIFO
      width_in_bits: 32
    }
    channels {
      name: "out"
      is_input: false
      type: FIFO
      width_in_bits: 32
    }
    name: "Block"
  )";

  xlscc::HLSBlock ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_P(TranslatorProcTest, IOProcClassMemberSetConstruct2) {
  const std::string content = R"(
    struct Foo {
      int x;
    };
    struct Bar {
      __xls_channel<int, __xls_channel_dir_In>& in;
      __xls_channel<int, __xls_channel_dir_Out>& out;

      Foo y;

      int Calculate() {
        return 10;
      }

      void Run() {
        y = Foo();
        y.x += Calculate();
      }
    };
    #pragma hls_top
    void my_package(__xls_channel<int, __xls_channel_dir_In>& in,
         __xls_channel<int, __xls_channel_dir_Out>& out) {
      Bar foo = {.in = in, .out = out};
      foo.y.x = in.read();
      foo.Run();
      out.write(foo.y.x);
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
  inputs["in"] = {xls::Value(xls::SBits(80, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(0 * 80 + 10, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 1);
  }
}

TEST_P(TranslatorProcTest, IOProcClassLValueInit) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int , __xls_channel_dir_In>& in;
         __xls_channel<long, __xls_channel_dir_Out>& out;
         __xls_channel<long, __xls_channel_dir_Out>& out2 = out;

         #pragma hls_top
         void Run() {
          out2.write(3*in.read());
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(7, 32)),
                  xls::Value(xls::SBits(10, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(15, 64)),
                    xls::Value(xls::SBits(21, 64)),
                    xls::Value(xls::SBits(30, 64))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels 	 {
      name: "in"
      is_input: true
      type: FIFO
      width_in_bits: 32
    }
    channels {
      name: "out"
      is_input: false
      type: FIFO
      width_in_bits: 64
    }
    name: "Block"
  )";

  xlscc::HLSBlock ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_P(TranslatorProcTest, IOProcClassHierarchicalLValue) {
  const std::string content = R"(
    struct LeafBlock {
      __xls_channel<int, __xls_channel_dir_Out> leaf_out1;
      __xls_channel<int, __xls_channel_dir_Out> leaf_out2;

      void Run() {
        leaf_out1.write(1);
        leaf_out2.write(1);
      }
    };

    struct HierBlock {
      __xls_channel<int, __xls_channel_dir_Out> out1;
      __xls_channel<int, __xls_channel_dir_Out> out2;

      LeafBlock x = {
        .leaf_out1 = out1,
        .leaf_out2 = out2,
      };

      #pragma hls_top
      void Run() {
        x.Run();
      }
    };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out1"] = {xls::Value(xls::SBits(1, 32))};
  outputs["out2"] = {xls::Value(xls::SBits(1, 32))};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 1, /*max_ticks=*/100,
           /*top_level_init_interval=*/0,
           /*top_class_name=*/"");

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  // TODO: LeafBlock should be in metadata
  const std::string ref_meta_str = R"(
    channels {
      name: "out1"
      is_input: false
      type: FIFO
      width_in_bits: 32
    }
    channels {
      name: "out2"
      is_input: false
      type: FIFO
      width_in_bits: 32
    }
    name: "HierBlock"
  )";

  xlscc::HLSBlock ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_P(TranslatorProcTest, ForPipelinedWithChannelTernary) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& dir_in,
             __xls_channel<int>& in1,
             __xls_channel<int>& in2,
             __xls_channel<int>& out) {
      const int dir = dir_in.read();

      __xls_channel<int>& in = dir ? in1 : in2;

      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += in.read();
      }

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("dir_in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

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

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir_in"] = {xls::Value(xls::SBits(1, 32))};
    inputs["in1"] = {
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(6, 32)),
        xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(5, 32))};
    inputs["in2"] = {};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4 + 6 + 5 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir_in"] = {xls::Value(xls::SBits(0, 32))};
    inputs["in1"] = {};
    inputs["in2"] = {
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(6, 32)),
        xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(5, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4 + 6 + 5 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedWithChannelTernaryNested) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& dir_in,
             __xls_channel<int>& in1,
             __xls_channel<int>& in2,
             __xls_channel<int>& in3,
             __xls_channel<int>& out) {
      const int dir = dir_in.read();

      __xls_channel<int>& in =
        (dir == 0) ? in1 : ((dir == 1) ? in2 : in3);

      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += in.read();
      }

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("dir_in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

    HLSChannel* ch_in1 = block_spec.add_channels();
    ch_in1->set_name("in1");
    ch_in1->set_is_input(true);
    ch_in1->set_type(FIFO);

    HLSChannel* ch_in2 = block_spec.add_channels();
    ch_in2->set_name("in2");
    ch_in2->set_is_input(true);
    ch_in2->set_type(FIFO);

    HLSChannel* ch_in3 = block_spec.add_channels();
    ch_in3->set_name("in3");
    ch_in3->set_is_input(true);
    ch_in3->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir_in"] = {xls::Value(xls::SBits(0, 32))};
    inputs["in1"] = {
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(6, 32)),
        xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(5, 32))};
    inputs["in2"] = {};
    inputs["in3"] = {};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4 + 6 + 5 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir_in"] = {xls::Value(xls::SBits(1, 32))};
    inputs["in1"] = {};
    inputs["in2"] = {
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(6, 32)),
        xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(5, 32))};
    inputs["in3"] = {};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4 + 6 + 5 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir_in"] = {xls::Value(xls::SBits(2, 32))};
    inputs["in1"] = {};
    inputs["in2"] = {};
    inputs["in3"] = {
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(6, 32)),
        xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(5, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4 + 6 + 5 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedWithChannelTernaryMultiple) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& dirA_in,
             __xls_channel<int>& dirB_in,
             __xls_channel<int>& in1,
             __xls_channel<int>& in2,
             __xls_channel<int>& out) {
      const int dirA = dirA_in.read();
      const int dirB = dirB_in.read();

      __xls_channel<int>& inA = dirA ? in1 : in2;
      __xls_channel<int>& inB = dirB ? in2 : in1;

      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += inA.read();
        a -= inB.read();
      }

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dirA_in = block_spec.add_channels();
    dirA_in->set_name("dirA_in");
    dirA_in->set_is_input(true);
    dirA_in->set_type(FIFO);

    HLSChannel* dirB_in = block_spec.add_channels();
    dirB_in->set_name("dirB_in");
    dirB_in->set_is_input(true);
    dirB_in->set_type(FIFO);

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
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dirA_in"] = {xls::Value(xls::SBits(1, 32))};
    inputs["dirB_in"] = {xls::Value(xls::SBits(1, 32))};
    inputs["in1"] = {
        xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(10, 32)),
        xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32))};
    inputs["in2"] = {
        xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(2, 32)),
        xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(2, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(10 * 2 - 2 * 4, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dirA_in"] = {xls::Value(xls::SBits(0, 32))};
    inputs["dirB_in"] = {xls::Value(xls::SBits(0, 32))};
    inputs["in1"] = {
        xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(10, 32)),
        xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(0, 32))};
    inputs["in2"] = {
        xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(20, 32)),
        xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(20, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(20 * 4 - 10 * 2, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, ForPipelinedWithChannelInStructTernary) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    struct ChannelContainer {
      __xls_channel<int>& in;
    };

    #pragma hls_top
    void foo(__xls_channel<int>& dir_in,
             __xls_channel<int>& in1,
             __xls_channel<int>& in2,
             __xls_channel<int>& out) {
      const int dir = dir_in.read();

      ChannelContainer container = {.in = dir ? in1 : in2};

      int a = 0;

      #pragma hls_pipeline_init_interval 1
      for(long i=1;i<=4;++i) {
        a += container.in.read();
      }

      out.write(a);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("dir_in");
    ch_in->set_is_input(true);
    ch_in->set_type(FIFO);

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

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir_in"] = {xls::Value(xls::SBits(1, 32))};
    inputs["in1"] = {
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(6, 32)),
        xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(5, 32))};
    inputs["in2"] = {};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4 + 6 + 5 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["dir_in"] = {xls::Value(xls::SBits(0, 32))};
    inputs["in1"] = {};
    inputs["in2"] = {
        xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(6, 32)),
        xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(5, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4 + 6 + 5 + 5, 32))};

    ProcTest(content, block_spec, inputs, outputs, /* min_ticks = */ 4);
  }
}

TEST_P(TranslatorProcTest, DebugAssert) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int r = in.read();
          if(r != 3) {
            __xlscc_assert("hello", r > 5, "this one");
          }
          out.write(r);
         }
      };)";

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(3, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(3, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks = */ 1);
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(1, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(1, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks=*/1,
             /*max_ticks=*/100,
             /*top_level_init_interval=*/0,
             /*top_class_name=*/"",
             /*expected_tick_status=*/absl::AbortedError("hello"));
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(9, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(9, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks = */ 1);
  }
}

TEST_P(TranslatorProcTest, DebugAssertInSubroutine) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         void Subroutine(int r) {
           __xlscc_assert("hello", r > 5);
         }

         #pragma hls_top
         void Run() {
          int r = in.read();
          if(r != 3) {
            Subroutine(r);
          }
          out.write(r);
         }
      };)";

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(3, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(3, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks = */ 1);
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(1, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(1, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks=*/1,
             /*max_ticks=*/100,
             /*top_level_init_interval=*/0,
             /*top_class_name=*/"",
             /*expected_tick_status=*/absl::AbortedError("hello"));
  }
  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(9, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(9, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks = */ 1);
  }
}

TEST_P(TranslatorProcTest, DebugTrace) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int r = in.read();
          if(r != 3) {
            __xlscc_trace("Value is {:d}", r);
          }
          out.write(r);
         }
      };)";

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(3, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(3, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks = */ 1,
             /*max_ticks=*/100,
             /*top_level_init_interval=*/0,
             /*top_class_name=*/"",
             /*expected_tick_status=*/absl::OkStatus(),
             /*expected_events_by_proc_name=*/
             {{"Block_proc", xls::InterpreterEvents()}});
  }
  {
    xls::InterpreterEvents expected_events;
    expected_events.trace_msgs.push_back(
        xls::TraceMessage{.message = "Value is 9", .verbosity = 0});

    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(9, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(9, 32))};
    ProcTest(
        content, /*block_spec=*/std::nullopt, inputs, outputs,
        /*min_ticks = */ 1,
        /*max_ticks=*/100,
        /*top_level_init_interval=*/0,
        /*top_class_name=*/"",
        /*expected_tick_status=*/absl::OkStatus(),
        /*expected_events_by_proc_name=*/{{"Block_proc", expected_events}});
  }
}

TEST_P(TranslatorProcTest, DebugTraceInPipelinedLoop) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int r = in.read();
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<=r;++i) {
            if(i != 3) {
              __xlscc_trace("Value is {:d}", i);
            }
          }
          out.write(r);
         }
      };)";

  {
    xls::InterpreterEvents expected_events;
    expected_events.trace_msgs.push_back(
        xls::TraceMessage{.message = "Value is 0", .verbosity = 0});
    expected_events.trace_msgs.push_back(
        xls::TraceMessage{.message = "Value is 1", .verbosity = 0});
    expected_events.trace_msgs.push_back(
        xls::TraceMessage{.message = "Value is 2", .verbosity = 0});
    expected_events.trace_msgs.push_back(
        xls::TraceMessage{.message = "Value is 4", .verbosity = 0});

    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(4, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(4, 32))};

    ProcTest(
        content, /*block_spec=*/std::nullopt, inputs, outputs,
        /*min_ticks = */ 1,
        /*max_ticks=*/100,
        /*top_level_init_interval=*/0,
        /*top_class_name=*/"",
        /*expected_tick_status=*/absl::OkStatus(),
        /*expected_events_by_proc_name=*/
        {{generate_fsms_for_pipelined_loops_ ? "Block_proc" : "__for_1_proc",
          expected_events}});
  }
}

TEST_P(TranslatorProcTest, NonblockingRead) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
            int value = 0;
            if(!in.nb_read(value)) {
              value = 100;
            }
            out.write(value);
         }
      };)";

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(3, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(3, 32)),
                      xls::Value(xls::SBits(100, 32)),
                      xls::Value(xls::SBits(100, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks = */ 2, /*max_ticks=*/10);
  }
}

TEST_P(TranslatorProcTest, NonblockingReadInSubroutine) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         int ReadInSubroutine(__xls_channel<int, __xls_channel_dir_In>& in) {
          int value = 0;
          if(!in.nb_read(value)) {
            value = 100;
          }
          return value;
         }

         #pragma hls_top
         void Run() {
           const int v = ReadInSubroutine(in);
           out.write(v);
         }
      };)";

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
    inputs["in"] = {xls::Value(xls::SBits(3, 32))};

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(3, 32)),
                      xls::Value(xls::SBits(100, 32)),
                      xls::Value(xls::SBits(100, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /*min_ticks = */ 2, /*max_ticks=*/10);
  }
}

TEST_P(TranslatorProcTest, LocalChannel) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        static __xls_channel<int> internal;

        const int x = in.read();
        internal.write(x*2);

        const int xr = internal.read();
        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelUsedInSubroutine) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      int Subroutine(__xls_channel<int>& ch, int x) {
        int xr = ch.read();
        return xr;
      }

      #pragma hls_top
      void foo() {
        static __xls_channel<int> internal;
        const int x = in.read();
        internal.write(x*2);
        const int xr = Subroutine(internal, x);
        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelDeclaredInSubroutine) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      int Subroutine(int x) {
        static __xls_channel<int> internal;
        internal.write(x*2);
        int xr = internal.read();
        return xr;
      }

      #pragma hls_top
      void foo() {
        const int x = in.read();
        const int xr = Subroutine(x);
        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelPipelinedLoop) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        static __xls_channel<int> internal;

        #pragma hls_pipeline_init_interval 1
        for(int i=0;i<1;++i) {
          const int x = in.read();
          internal.write(x*2);
        }

        #pragma hls_pipeline_init_interval 1
        for(int i=0;i<1;++i) {
          const int xr = internal.read();
          out.write(xr);
        }
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelPipelinedLoopDeclaredInSubroutine) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      int Subroutine(int x) {
        static __xls_channel<int> internal;

        #pragma hls_pipeline_init_interval 1
        for(int i=0;i<1;++i) {
          internal.write(x*2);
        }

        int xr = 0;

        #pragma hls_pipeline_init_interval 1
        for(int i=0;i<1;++i) {
          xr = internal.read();
        }

        return xr;
      }

      #pragma hls_top
      void foo() {
        const int x = in.read();
        const int xr = Subroutine(x);
        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelPipelinedLoopUsedInSubroutine) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      int Subroutine(__xls_channel<int>& ch, int x) {
        #pragma hls_pipeline_init_interval 1
        for(int i=0;i<1;++i) {
          ch.write(x*2);
        }

        int xr = 0;

        #pragma hls_pipeline_init_interval 1
        for(int i=0;i<1;++i) {
          xr = ch.read();
        }

        return xr;
      }

      #pragma hls_top
      void foo() {
        static __xls_channel<int> internal;

        const int x = in.read();
        const int xr = Subroutine(internal, x);
        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelSubBlock) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      void Ping(__xls_channel<int>& internal) {
        const int x = in.read();
        internal.write(x*2);
      }

      void Pong(__xls_channel<int>& internal) {
        const int xr = internal.read();
        out.write(xr);
      }

      #pragma hls_top
      void top_f() {
        static __xls_channel<int> internal_in_top;
        Ping(internal_in_top);
        Pong(internal_in_top);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelSubBlockNonStatic) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      void Ping(__xls_channel<int>& internal) {
        const int x = in.read();
        internal.write(x*2);
      }

      void Pong(__xls_channel<int>& internal) {
        const int xr = internal.read();
        out.write(xr);
      }

      #pragma hls_top
      void top_f() {
        __xls_channel<int> internal_in_top;
        Ping(internal_in_top);
        Pong(internal_in_top);
      }
    };
  )";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(
      ret.status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("declaration uninitialized")));
}

TEST_P(TranslatorProcTest, LocalChannelPassUpAndDown) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      __xls_channel<int>& SubroutineWrite(int x) {
        static __xls_channel<int> internal;
        internal.write(x*2);
        return internal;
      }

      int SubroutineRead(__xls_channel<int>& ch) {
        return ch.read();
      }

      #pragma hls_top
      void foo() {
        const int x = in.read();

        __xls_channel<int>& internal = SubroutineWrite(x);
        const int xr = SubroutineRead(internal);

        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelDeclaredInTopClassUnspecified) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;
      __xls_channel<int> internal;

      #pragma hls_top
      void foo() {
        const int x = in.read();
        internal.write(x*2);

        const int xr = internal.read();
        out.write(xr);
      }
    };
  )";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("unspecified")));
}

TEST_P(TranslatorProcTest, LocalChannelDeclaredInTopClass) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      __xls_channel<int, __xls_channel_dir_InOut> internal;

      #pragma hls_top
      void foo() {
        const int x = in.read();
        internal.write(x*2);

        const int xr = internal.read();
        out.write(xr);
      }
    };
  )";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(
      ret.status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("Internal (InOut) channels")));
}

TEST_P(TranslatorProcTest, ChannelDeclaredInClassInitialized) {
  const std::string content = R"(
    class Impl {
    private:

      __xls_channel<int> internal;

      __xls_channel<int, __xls_channel_dir_In>& in;
      __xls_channel<int, __xls_channel_dir_Out>& out;

    public:
      Impl(__xls_channel<int> internal,
           __xls_channel<int, __xls_channel_dir_In>& in,
           __xls_channel<int, __xls_channel_dir_Out>& out) :
        internal(internal), in(in), out(out) {
      }

      void Run() {
        const int x = in.read();
        internal.write(x*2);

        const int xr = internal.read();
        out.write(xr);
      }
    };

    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        static __xls_channel<int> internal;

        Impl impl(internal, in, out);

        impl.Run();
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelDeclaredInClassUninitialized) {
  const std::string content = R"(
    class Impl {
    private:
      __xls_channel<int, __xls_channel_dir_In>& in;
      __xls_channel<int, __xls_channel_dir_Out>& out;

      __xls_channel<int> internal;

    public:
      Impl(__xls_channel<int, __xls_channel_dir_In>& in,
           __xls_channel<int, __xls_channel_dir_Out>& out) :
        in(in), out(out) {
      }

      void Run() {
        const int x = in.read();
        internal.write(x*2);

        const int xr = internal.read();
        out.write(xr);
      }
    };

    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        static Impl impl(in, out);

        impl.Run();
      }
    };
  )";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("marked as InOut")));
}

TEST_P(TranslatorProcTest, LocalChannelDeclaredInClass) {
  const std::string content = R"(
    class Impl {
    private:
      __xls_channel<int, __xls_channel_dir_InOut> internal;

      __xls_channel<int, __xls_channel_dir_In>& in;
      __xls_channel<int, __xls_channel_dir_Out>& out;
    public:
      Impl(__xls_channel<int, __xls_channel_dir_In>& in,
           __xls_channel<int, __xls_channel_dir_Out>& out) :
        in(in), out(out) {
      }

      void Run() {
        const int x = in.read();
        internal.write(x*2);

        const int xr = internal.read();
        out.write(xr);
      }
    };

    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        static Impl impl(in, out);

        impl.Run();
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelDeclaredInClassNonStatic) {
  const std::string content = R"(
    class Impl {
    private:
      __xls_channel<int, __xls_channel_dir_InOut> internal;

      __xls_channel<int, __xls_channel_dir_In>& in;
      __xls_channel<int, __xls_channel_dir_Out>& out;
    public:
      Impl(__xls_channel<int, __xls_channel_dir_In>& in,
           __xls_channel<int, __xls_channel_dir_Out>& out) :
        in(in), out(out) {
      }

      void Run() {
        const int x = in.read();
        internal.write(x*2);

        const int xr = internal.read();
        out.write(xr);
      }
    };

    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        Impl impl(in, out);

        impl.Run();
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelAddCondition) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      void Ping(__xls_channel<int>& internal, int x) {
        internal.write(x*2);
      }

      #pragma hls_top
      void Run() {
        static __xls_channel<int> internal_in_top;

        const int x = in.read();

        if(x > 50) {
          Ping(internal_in_top, x);
        }

        int xr = 0;
        if(!internal_in_top.nb_read(xr)) {
          xr = 100;
        }

        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32)),
                      xls::Value(xls::SBits(100, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelAddCondition2) {
  const std::string content = R"(
    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      void Ping(__xls_channel<int>& internal, int x) {
        internal.write(x*2);
      }

      #pragma hls_top
      void Run() {
        static __xls_channel<int> internal_in_top;

        const int x = in.read();

        if(x > 50) {
          Ping(internal_in_top, x);
        }

        int xr = 0;
        if(!internal_in_top.nb_read(xr)) {
          xr = 100;
        }

        out.write(xr);
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32)),
                      xls::Value(xls::SBits(100, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, LocalChannelDeclaredInClassNonStaticAddCondition) {
  const std::string content = R"(
    class Impl {
    private:
      __xls_channel<int, __xls_channel_dir_InOut> internal;

      __xls_channel<int, __xls_channel_dir_Out>& out;
    public:
      Impl(__xls_channel<int, __xls_channel_dir_Out>& out) :
        out(out) {
      }

      void Ping(int x) {
        internal.write(x*2);
      }

      void Pong() {
        int xr = 0;
        if(!internal.nb_read(xr)) {
          xr = 100;
        }
        out.write(xr);
      }
    };

    class Block {
      __xls_channel<int, __xls_channel_dir_In> in;
      __xls_channel<int, __xls_channel_dir_Out> out;

      #pragma hls_top
      void foo() {
        static Impl impl(out);

        const int x = in.read();
        if(x > 50) {
          impl.Ping(x);
        }
        impl.Pong();
      }
    };
  )";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(55, 32)),
                  xls::Value(xls::SBits(10, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(110, 32)),
                      xls::Value(xls::SBits(100, 32))};

    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs);
  }
}

// Test serialization of loops via min ticks
TEST_P(TranslatorProcTest, PipelinedLoopSerial) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          for(int i=0;i<6;++i) {
            out.write(i);
          }
          for(int i=0;i<6;++i) {
            out.write(10+i);
          }
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(0, 32)),  xls::Value(xls::SBits(1, 32)),
        xls::Value(xls::SBits(2, 32)),  xls::Value(xls::SBits(3, 32)),
        xls::Value(xls::SBits(4, 32)),  xls::Value(xls::SBits(5, 32)),
        xls::Value(xls::SBits(10, 32)), xls::Value(xls::SBits(11, 32)),
        xls::Value(xls::SBits(12, 32)), xls::Value(xls::SBits(13, 32)),
        xls::Value(xls::SBits(14, 32)), xls::Value(xls::SBits(15, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 11,
             /* max_ticks = */ 100,
             /* top_level_init_interval = */ 1);
  }
}

TEST_F(TranslatorProcTestWithoutFSMParam, ForPipelinedASAPTrivial) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      [[xlscc::asap]]
      for(long i=1;i<=4;++i) {
        int a = in.read();
        out.write(a + i);
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

  generate_fsms_for_pipelined_loops_ = false;

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(80, 32)),
      xls::Value(xls::SBits(80, 32)), xls::Value(xls::SBits(80, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {
      xls::Value(xls::SBits(80 + 1, 32)), xls::Value(xls::SBits(80 + 2, 32)),
      xls::Value(xls::SBits(80 + 3, 32)), xls::Value(xls::SBits(80 + 4, 32))};

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));

  package_ = std::make_unique<xls::Package>("my_package");
  ASSERT_THAT(
      translator_
          ->GenerateIR_Block(package_.get(), block_spec,
                             /*top_level_init_interval=*/1)
          .status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("IO ops with schedul")));
}

TEST_F(TranslatorProcTestWithoutFSMParam, ForPipelinedASAPOutsideScopeAccess) {
  const std::string content = R"(
    class Block {
    public:
      __xls_channel<int, __xls_channel_dir_Out>& in;
      __xls_channel<int, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void foo() {
        int a = in.read();

        [[xlscc::asap]]
        for(long i=1;i<=4;++i) {
          out.write(a + i);
        }
      }
    };)";

  generate_fsms_for_pipelined_loops_ = false;

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                             /*top_level_init_interval=*/1);

  ASSERT_THAT(
      ret.status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("variable in outside scope")));
}

// TODO(seanhaskell): Turn on once b/321114633 is resolved
TEST_P(TranslatorProcTest, DISABLED_PipelinedLoopASAP) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          // Use sync channel to enforce ordering
          static __xls_channel<int, __xls_channel_dir_InOut> sync;

          __xlscc_asap();
          for(int i=0;i<6;++i) {
            out.write(i);
            sync.write(0);
          }
          __xlscc_asap();
          for(int i=0;i<6;++i) {
            (void)sync.read();
            out.write(10+i);
          }
         }
      };)";

  if (!generate_fsms_for_pipelined_loops_) {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

    {
      absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
      outputs["out"] = {
          xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(10, 32)),
          xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(11, 32)),
          xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(12, 32)),
          xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(13, 32)),
          xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(14, 32)),
          xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(15, 32))};
      ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
               /* min_ticks = */ 6,
               /* max_ticks = */ 7,
               /* top_level_init_interval = */ 1);
    }
  } else {
    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    auto ret =
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                               /*top_level_init_interval=*/1);

    ASSERT_THAT(ret.status(),
                absl_testing::StatusIs(
                    absl::StatusCode::kUnimplemented,
                    testing::HasSubstr("loops with scheduling options")));
  }
}
TEST_P(TranslatorProcTest, PipelinedLoopASAPDataDependency) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          // Use sync channel to enforce ordering
          static __xls_channel<int, __xls_channel_dir_InOut> sync;

          int value = 0;

          [[xlscc::asap]]
          for(int i=0;i<6;++i) {
            out.write(value++);
            sync.write(0);
          }

          [[xlscc::asap]]
          for(int i=0;i<6;++i) {
            (void)sync.read();
            out.write(10+value+i);
          }
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                             /*top_level_init_interval=*/1);

  ASSERT_THAT(
      ret.status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("variable in outside scope")));
}

// TODO(seanhaskell): Turn on once b/321114633 is resolved
TEST_P(TranslatorProcTest, DISABLED_PipelinedLoopASAPDataDependencyMethod) {
  const std::string content = R"(
       class Block {
        int value = 0;

        public:
         __xls_channel<int, __xls_channel_dir_Out> out;

        void write_it(int off)const {
          out.write(value + off);
        }

         #pragma hls_top
         void Run() {
          // Use sync channel to enforce ordering
          static __xls_channel<int, __xls_channel_dir_InOut> sync;

          #pragma hls_pipeline_init_interval 1
          __xlscc_asap();for(int i=0;i<6;++i) {
            write_it(i);
            sync.write(0);
          }

          #pragma hls_pipeline_init_interval 1
          __xlscc_asap();for(int i=0;i<6;++i) {
            (void)sync.read();
            out.write(10+value+i);
          }
         }
      };)";

  if (!generate_fsms_for_pipelined_loops_) {
    absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

    {
      absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
      outputs["out"] = {
          xls::Value(xls::SBits(0, 32)), xls::Value(xls::SBits(10, 32)),
          xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(11, 32)),
          xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(12, 32)),
          xls::Value(xls::SBits(3, 32)), xls::Value(xls::SBits(13, 32)),
          xls::Value(xls::SBits(4, 32)), xls::Value(xls::SBits(14, 32)),
          xls::Value(xls::SBits(5, 32)), xls::Value(xls::SBits(15, 32))};
      ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
               /* min_ticks = */ 6,
               /* max_ticks = */ 7);
    }
  } else {
    XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                           /*io_test_mode=*/false,
                           /*error_on_init_interval=*/false));
    package_ = std::make_unique<xls::Package>("my_package");
    HLSBlock block_spec;
    auto ret =
        translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec,
                                               /*top_level_init_interval=*/1);

    ASSERT_THAT(ret.status(),
                absl_testing::StatusIs(
                    absl::StatusCode::kUnimplemented,
                    testing::HasSubstr("loops with scheduling options")));
  }
}

TEST_P(TranslatorProcTest, PipelinedLoopSerialDataDependency) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int value = 0;

          for(int i=0;i<6;++i) {
            out.write(value++);
          }

          for(int i=0;i<6;++i) {
            out.write(10+value+i);
          }
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(0, 32)),  xls::Value(xls::SBits(1, 32)),
        xls::Value(xls::SBits(2, 32)),  xls::Value(xls::SBits(3, 32)),
        xls::Value(xls::SBits(4, 32)),  xls::Value(xls::SBits(5, 32)),
        xls::Value(xls::SBits(16, 32)), xls::Value(xls::SBits(17, 32)),
        xls::Value(xls::SBits(18, 32)), xls::Value(xls::SBits(19, 32)),
        xls::Value(xls::SBits(20, 32)), xls::Value(xls::SBits(21, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 11,
             /* max_ticks = */ 100,
             /* top_level_init_interval = */ 1);
  }
}

TEST_P(TranslatorProcTest, PipelinedLoopConditionFalse) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int value = in.read();

          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<0;++i) {
            out.write(55);
          }

          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<6;++i) {
            out.write(value++);
          }
        }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(11 + 0, 32)), xls::Value(xls::SBits(11 + 1, 32)),
        xls::Value(xls::SBits(11 + 2, 32)), xls::Value(xls::SBits(11 + 3, 32)),
        xls::Value(xls::SBits(11 + 4, 32)), xls::Value(xls::SBits(11 + 5, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 1,
             /* max_ticks = */ 100,
             /* top_level_init_interval = */ 1);
  }
}

TEST_P(TranslatorProcTest, PipelinedLoopConditional) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int value = in.read();

          if(0) {
            #pragma hls_pipeline_init_interval 1
            for(int i=0;i<0;++i) {
              out.write(55);
            }
          }

          if(1) {
            #pragma hls_pipeline_init_interval 1
            for(int i=0;i<6;++i) {
              out.write(value++);
            }
          }
        }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(11 + 0, 32)), xls::Value(xls::SBits(11 + 1, 32)),
        xls::Value(xls::SBits(11 + 2, 32)), xls::Value(xls::SBits(11 + 3, 32)),
        xls::Value(xls::SBits(11 + 4, 32)), xls::Value(xls::SBits(11 + 5, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 1,
             /* max_ticks = */ 100,
             /* top_level_init_interval = */ 1);
  }
}

TEST_P(TranslatorProcTest, PipelinedLoopConditional2) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int value = in.read();

          if(0) {
            #pragma hls_pipeline_init_interval 1
            for(int i=0;i<1;++i) {
              ++value;
            }
          }

          if(1) {
            #pragma hls_pipeline_init_interval 1
            for(int i=0;i<6;++i) {
              out.write(value++);
            }
          }
        }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(11 + 0, 32)), xls::Value(xls::SBits(11 + 1, 32)),
        xls::Value(xls::SBits(11 + 2, 32)), xls::Value(xls::SBits(11 + 3, 32)),
        xls::Value(xls::SBits(11 + 4, 32)), xls::Value(xls::SBits(11 + 5, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 1,
             /* max_ticks = */ 100,
             /* top_level_init_interval = */ 1);
  }
}

TEST_P(TranslatorProcTest, PipelinedLoopSerialAfterASAP) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int value = 0;

          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<6;++i) {
            out.write(10+value+i);
          }

          #pragma hls_pipeline_init_interval 1
          [[xlscc::asap]] for(int i=0;i<6;++i) {
            out.write(500);
          }
         }
      };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("IO ops with scheduling options")));
}

TEST_F(TranslatorProcTestWithoutFSMParam, OpDuplicationAcrossIO) {
  const std::string content = R"(
    class Block {
    public:
      __xls_channel<int, __xls_channel_dir_In>& in1;
      __xls_channel<int, __xls_channel_dir_In>& in2;
      __xls_channel<int, __xls_channel_dir_Out>& out1;
      __xls_channel<int, __xls_channel_dir_Out>& out2;

#pragma hls_top
      void foo() {
        int x = in1.read();

        x *= 3;

        int y = in2.read();

        x += y;

        out1.write(x);
        out2.write(x);
      }
    };)";

  XLS_ASSERT_OK(ScanFile(content, /*clang_argv=*/{},
                         /*io_test_mode=*/false,
                         /*error_on_init_interval=*/false));
  package_ = std::make_unique<xls::Package>("my_package");
  HLSBlock block_spec;
  auto ret =
      translator_->GenerateIR_BlockFromClass(package_.get(), &block_spec);
  ASSERT_THAT(ret.status(), absl_testing::IsOk());

  // Run inliner to expose the duplication
  // But don't run other passes that might eliminate it (cse etc)
  {
    // Don't do cse so that the duplication shows
    // bdd_cse pass wants a delay estimator
    std::unique_ptr<xls::OptimizationCompoundPass> pipeline =
        xls::GetOptimizationPipelineGenerator()
            .GeneratePipeline("inlining dce")
            .value();
    xls::OptimizationPassOptions options =
        xls::OptimizationPassOptions().WithOptLevel(3);
    xls::PassResults results;

    XLS_ASSERT_OK(pipeline->Run(package_.get(), options, &results).status());
  }

  int64_t multiply_op_count = 0;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Proc * proc, package_->GetProc("Block_proc"));
  for (xls::Node* node : proc->nodes()) {
    if (node->op() == xls::Op::kSMul || node->op() == xls::Op::kUMul ||
        node->op() == xls::Op::kSMulp || node->op() == xls::Op::kUMulp) {
      ++multiply_op_count;
    }
  }

  // TODO(seanhaskell): Should be =1 when continuations are finished.
  EXPECT_GT(multiply_op_count, 1);
}

TEST_P(TranslatorProcTest, ForPipelinedWithDirectIn) {
  const std::string content = R"(
    #pragma hls_top
    void foo(const long& dir,
              __xls_channel<int>& in,
              __xls_channel<int>& out) {

      const int ctrl = in.read();

      #pragma hls_pipeline_init_interval 1
      for(int i=0;i<3;++i) {
        out.write(ctrl*i + dir);
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
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(100, 64))};
  inputs["in"] = {xls::Value(xls::SBits(5, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {xls::Value(xls::SBits(100 + 0 * 5, 32)),
                      xls::Value(xls::SBits(100 + 1 * 5, 32)),
                      xls::Value(xls::SBits(100 + 2 * 5, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t body_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(body_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : (1 + 64 + 32 + 32));

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (1 + 32) : 0);
}

// Different variables names i and j as the same name may or may not get merged
// by NamedDecl* (making the test harder to follow)
TEST_P(TranslatorProcTest, ForPipelinedSerialShared) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(short i=1;i<=4;++i) {
        a+=i;
      }
      #pragma hls_pipeline_init_interval 1
      for(short j=3;j<=4;++j) {
        a-=j;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 1 + 2 + 3 + 4 - 3 - 4, 32)),
                      xls::Value(xls::SBits(100 + 1 + 2 + 3 + 4 - 3 - 4, 32)),
                      xls::Value(xls::SBits(20 + 1 + 2 + 3 + 4 - 3 - 4, 32))};

    ProcTest(content, block_spec, inputs, outputs,
             /* min_ticks = */ generate_fsms_for_pipelined_loops_ ? 6 * 3 : 15);
  }

  const int64_t first_loop_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : first_loop_bits);

  const int64_t second_loop_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : second_loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, generate_fsms_for_pipelined_loops_
                                     ? (32 + 1 + 16 + 1 + 16 + 32)
                                     : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedNestedShared) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(short i=1;i<=4;++i) {
        #pragma hls_pipeline_init_interval 1
        for(short j=1;j<=2;++j) {
          a+=j;
        }
        --a;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + (1 + 2) * 4 - 4, 32)),
                      xls::Value(xls::SBits(100 + (1 + 2) * 4 - 4, 32)),
                      xls::Value(xls::SBits(20 + (1 + 2) * 4 - 4, 32))};

    ProcTest(content, block_spec, inputs, outputs,
             /* min_ticks = */ 8);
  }

  const int64_t first_loop_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : first_loop_bits);

  const int64_t second_loop_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : second_loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? (32 + 1 + 16 + 1 + 16) : 0L);
}

TEST_P(TranslatorProcTest, ForPipelinedStaticShared) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      static int st = 10;
      st += in.read();

      #pragma hls_pipeline_init_interval 1
      for(short i=1;i<=4;++i) {
        #pragma hls_pipeline_init_interval 1
        for(short j=1;j<=2;++j) {
          st+=j;
        }
        --st;
      }
      #pragma hls_pipeline_init_interval 1
      for(short k=1;k<=2;++k) {
        st+=k;
      }

      out.write(st);
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
    int64_t accum = 10;

    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    accum += 80;
    accum += (1 + 2) * 4 - 4;
    accum += 1 + 2;
    outputs["out"].push_back(xls::Value(xls::SBits(accum, 32)));

    accum += 100;
    accum += (1 + 2) * 4 - 4;
    accum += 1 + 2;
    outputs["out"].push_back(xls::Value(xls::SBits(accum, 32)));

    accum += 20;
    accum += (1 + 2) * 4 - 4;
    accum += 1 + 2;
    outputs["out"].push_back(xls::Value(xls::SBits(accum, 32)));

    ProcTest(content, block_spec, inputs, outputs,
             /* min_ticks = */ 8);
  }

  const int64_t first_loop_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : first_loop_bits);

  const int64_t second_loop_bits = 1 + 32 + 16;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : second_loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, generate_fsms_for_pipelined_loops_
                                     ? (32 + 1 + 16 + 1 + 16 + 1 + 16)
                                     : 32);
}

TEST_P(TranslatorProcTest, ForPipelinedStaticSharedInBody) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int a = in.read();

      #pragma hls_pipeline_init_interval 1
      for(short i=1;i<=4;++i) {
        static int st = 10;
        #pragma hls_pipeline_init_interval 1
        for(short j=1;j<=2;++j) {
          st += j;
        }
        a += st;
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
    outputs["out"] = {xls::Value(xls::SBits(80 + 13 + 16 + 19 + 22, 32)),
                      xls::Value(xls::SBits(100 + 25 + 28 + 31 + 34, 32)),
                      xls::Value(xls::SBits(20 + 37 + 40 + 43 + 46, 32))};

    ProcTest(content, block_spec, inputs, outputs,
             /* min_ticks = */ 8);
  }

  const int64_t first_loop_bits = 81;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t first_proc_state_bits,
                           GetStateBitsForProcNameContains("for_1"));
  EXPECT_EQ(first_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : first_loop_bits);

  const int64_t second_loop_bits = 49;

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t second_proc_state_bits,
                           GetStateBitsForProcNameContains("for_2"));
  EXPECT_EQ(second_proc_state_bits,
            generate_fsms_for_pipelined_loops_ ? 0L : second_loop_bits);

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t top_proc_state_bits,
                           GetStateBitsForProcNameContains("foo"));
  EXPECT_EQ(top_proc_state_bits, generate_fsms_for_pipelined_loops_
                                     ? (32 + 1 + 16 + 32 + 1 + 16)
                                     : 0);
}

TEST_F(TranslatorProcTestWithoutFSMParam, MergedStatesIOFeedThrough) {
  const std::string content = R"(
  class Block {
     public:
      __xls_channel<int , __xls_channel_dir_In>& in;
      __xls_channel<int, __xls_channel_dir_Out>& out;

      #pragma hls_top
      void foo() {
        int a = in.read();
        int b = 100;
        #pragma hls_pipeline_init_interval 1
        for(long i=1;i<=3;++i) {
          out.write(a);
          b = a;
        }
        out.write(b);
      }
    };)";

  generate_fsms_for_pipelined_loops_ = true;
  merge_states_ = true;
  split_states_on_channel_ops_ = true;

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(11, 32)),
                  xls::Value(xls::SBits(33, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(11, 32)), xls::Value(xls::SBits(11, 32)),
        xls::Value(xls::SBits(11, 32)), xls::Value(xls::SBits(11, 32)),
        xls::Value(xls::SBits(33, 32)), xls::Value(xls::SBits(33, 32)),
        xls::Value(xls::SBits(33, 32)), xls::Value(xls::SBits(33, 32))};
    ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
             /* min_ticks = */ 1,
             /* max_ticks = */ 100,
             /* top_level_init_interval = */ 1);
  }
}

TEST_P(TranslatorProcTest, DirectInStoredInStatic) {
  const std::string content = R"(
    #pragma hls_top
    void foo(const int& dir,
              __xls_channel<int>& out) {
     static int x = dir;

      __xlscc_trace("foo() A: x {:d}", x);
      out.write(x*2);

      out.write(x*2 + 1);
      __xlscc_trace("foo() B: x {:d}", x);

      ++x;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(DIRECT_IN);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out");
    ch_out2->set_is_input(false);
    ch_out2->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(22, 32)), xls::Value(xls::SBits(23, 32)),
        xls::Value(xls::SBits(24, 32)), xls::Value(xls::SBits(25, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, ReadInStoredInStatic) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& dir,
              __xls_channel<int>& out) {
     static int x = dir.read();

      __xlscc_trace("foo() A: x {:d}", x);
      out.write(x*2);

      out.write(x*2 + 1);
      __xlscc_trace("foo() B: x {:d}", x);

      ++x;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out");
    ch_out2->set_is_input(false);
    ch_out2->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(22, 32)), xls::Value(xls::SBits(23, 32)),
        xls::Value(xls::SBits(24, 32)), xls::Value(xls::SBits(25, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

TEST_P(TranslatorProcTest, ReadInStoredInStaticBetweenStates) {
  const std::string content = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& dir,
              __xls_channel<int>& out) {
      static int x = 1;

      __xlscc_trace("foo() A: x {:d}", x);
      out.write(x*2);

      static int y = dir.read();

      out.write(x*2 + 1);
      __xlscc_trace("foo() B: x {:d}", x);

      x *= y;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out");
    ch_out2->set_is_input(false);
    ch_out2->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["dir"] = {xls::Value(xls::SBits(11, 32))};

  {
    absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
    outputs["out"] = {
        xls::Value(xls::SBits(2, 32)), xls::Value(xls::SBits(3, 32)),
        xls::Value(xls::SBits(22, 32)), xls::Value(xls::SBits(23, 32))};

    ProcTest(content, block_spec, inputs, outputs);
  }
}

}  // namespace

}  // namespace xlscc
