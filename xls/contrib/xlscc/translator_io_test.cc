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

class TranslatorIOTest : public XlsccTestBase {
 public:
  void IOTest(std::string content, std::list<IOOpTest> inputs,
              std::list<IOOpTest> outputs,
              absl::flat_hash_map<std::string, xls::Value> args = {}) {
    xlscc::GeneratedFunction* func;
    XLS_ASSERT_OK_AND_ASSIGN(std::string ir_src,
                             SourceToIr(content, &func, /* clang_argv= */ {},
                                        /* io_test_mode= */ true));

    XLS_ASSERT_OK_AND_ASSIGN(package_, ParsePackage(ir_src));
    XLS_ASSERT_OK_AND_ASSIGN(xls::Function * entry,
                             package_->GetTopAsFunction());

    const int total_test_ops = inputs.size() + outputs.size();
    ASSERT_EQ(func->io_ops.size(), total_test_ops);

    std::list<IOOpTest> input_ops_orig = inputs;
    for (const xlscc::IOOp& op : func->io_ops) {
      const std::string ch_name = op.channel->unique_name;

      if (op.op == xlscc::OpType::kRecv) {
        const std::string arg_name =
            absl::StrFormat("%s_op%i", ch_name, op.channel_op_index);

        const IOOpTest test_op = inputs.front();
        inputs.pop_front();

        XLS_CHECK_EQ(ch_name, test_op.name);

        const xls::Value& new_val = test_op.value;

        if (!args.contains(arg_name)) {
          args[arg_name] = new_val;
          continue;
        }

        if (args[arg_name].IsBits()) {
          args[arg_name] = xls::Value::Tuple({args[arg_name], new_val});
        } else {
          XLS_CHECK(args[arg_name].IsTuple());
          const xls::Value prev_val = args[arg_name];
          XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Value> values,
                                   prev_val.GetElements());
          values.push_back(new_val);
          args[arg_name] = xls::Value::Tuple(values);
        }
      }
    }

    XLS_ASSERT_OK_AND_ASSIGN(
        xls::Value actual,
        DropInterpreterEvents(xls::InterpretFunctionKwargs(entry, args)));

    std::vector<xls::Value> returns;

    if (total_test_ops > 1) {
      ASSERT_TRUE(actual.IsTuple());
      XLS_ASSERT_OK_AND_ASSIGN(returns, actual.GetElements());
    } else {
      returns.push_back(actual);
    }

    ASSERT_EQ(returns.size(), total_test_ops);

    inputs = input_ops_orig;

    int op_idx = 0;
    for (const xlscc::IOOp& op : func->io_ops) {
      const std::string ch_name = op.channel->unique_name;

      if (op.op == xlscc::OpType::kRecv) {
        const IOOpTest test_op = inputs.front();
        inputs.pop_front();

        XLS_CHECK(ch_name == test_op.name);

        ASSERT_TRUE(returns[op_idx].IsBits());
        XLS_ASSERT_OK_AND_ASSIGN(uint64_t val,
                                 returns[op_idx].bits().ToUint64());
        ASSERT_EQ(val, test_op.condition ? 1 : 0);

      } else if (op.op == xlscc::OpType::kSend) {
        const IOOpTest test_op = outputs.front();
        outputs.pop_front();

        XLS_CHECK(ch_name == test_op.name);

        ASSERT_TRUE(returns[op_idx].IsTuple());
        XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Value> elements,
                                 returns[op_idx].GetElements());
        ASSERT_EQ(elements.size(), 2);
        ASSERT_TRUE(elements[1].IsBits());
        XLS_ASSERT_OK_AND_ASSIGN(uint64_t val1, elements[1].bits().ToUint64());
        ASSERT_EQ(val1, test_op.condition ? 1 : 0);
        // Don't check data if it wasn't sent
        if (val1 != 0u) {
          ASSERT_EQ(elements[0], test_op.value);
        }
      } else {
        FAIL() << "IOOp was neither send nor recv: " << static_cast<int>(op.op);
      }
      ++op_idx;
    }

    ASSERT_EQ(inputs.size(), 0);
    ASSERT_EQ(outputs.size(), 0);
  }
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

TEST_F(TranslatorIOTest, Subroutine) {
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

  ASSERT_THAT(SourceToIr(content, /*func=*/nullptr, /* clang_argv= */ {},
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
         out.write(x);
       })";

  auto ret = SourceToIr(content);

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Channels should be either input or output")));
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

TEST_F(TranslatorIOTest, SelectChannel) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       int do_read(__xls_channel<int>& in) {
        return in.read();
       }
       #pragma hls_top
       void my_package(const int& sel,
                       __xls_channel<int>& inA,
                       __xls_channel<int>& inB,
                       __xls_channel<int>& out) {
         out.write(do_read(sel ? inA : inB));
       })";

  ASSERT_THAT(
      SourceToIr(content, /* pfunc= */ nullptr, /* clang_argv= */ {},
                 /* io_test_mode= */ true)
          .status(),
      xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                    testing::HasSubstr("Ternary on lvalues")));
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
                  testing::HasSubstr(
                      "Channel declaration initialized to channel value")));
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

  Block block = {.in = in, .out = out};
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
  Block block = {.in = in, .out = out};
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

  ASSERT_THAT(SourceToIr(content, /*func=*/nullptr, /* clang_argv= */ {},
                         /* io_test_mode= */ true)
                  .status(),
              xls::status_testing::StatusIs(absl::StatusCode::kOk));
}

}  // namespace

}  // namespace xlscc
