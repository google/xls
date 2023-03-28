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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/unit_test.h"

namespace xlscc {
namespace {

class TranslatorMemoryTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorMemoryTest, MemoryReadIOOp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<int, 32>& memory,
                       __xls_channel<int>& out) {
         const int addr = in.read();
         const int val = memory[addr];
         out.write(3*val);
       })";

  IOTest(
      content,
      /*inputs=*/{IOOpTest("in", 7, true), IOOpTest("memory__read", 10, true)},
      /*outputs=*/
      {IOOpTest("memory__read", xls::Value(xls::UBits(7, 5)), true),
       IOOpTest("out", 30, true)});
}

TEST_F(TranslatorMemoryTest, MemoryReadExplicitIOOp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<int, 32>& memory,
                       __xls_channel<int>& out) {
         const int addr = in.read();
         const int val = memory.read(addr);
         out.write(3*val);
       })";

  IOTest(
      content,
      /*inputs=*/{IOOpTest("in", 7, true), IOOpTest("memory__read", 10, true)},
      /*outputs=*/
      {IOOpTest("memory__read", xls::Value(xls::UBits(7, 5)), true),
       IOOpTest("out", 30, true)});
}

TEST_F(TranslatorMemoryTest, MemoryReadStructExplicitIOOp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Stuff {
         unsigned short x;
         unsigned short y;

         Stuff() : x(0), y(0) { }
         Stuff(int val) : x(val >> 16), y(val & 0b1111) { }
         operator int()const {return x + y;}
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<Stuff, 32>& memory,
                       __xls_channel<int>& out) {
         const int addr = in.read();
         const Stuff val = memory.read(addr);
         out.write(3*val.y);
       })";

  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", 7, true),
          IOOpTest("memory__read",
                   xls::Value::Tuple({xls::Value(xls::UBits(10, 16)),
                                      xls::Value(xls::UBits(5, 16))}),
                   true)},
         /*outputs=*/
         {IOOpTest("memory__read", xls::Value(xls::UBits(7, 5)), true),
          IOOpTest("out", 30, true)});
}

TEST_F(TranslatorMemoryTest, MemoryReadIOOpSubroutine) {
  const std::string content = R"(
       #include "/xls_builtin.h"

       int ReadIt(__xls_memory<short, 21>& mem, int addr) {
        return mem[addr];
       }

       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<int, 32>& memory,
                       __xls_channel<int>& out) {
         const int addr = in.read();
         const int val = memory[addr];
         out.write(3*val);
       })";

  IOTest(
      content,
      /*inputs=*/{IOOpTest("in", 7, true), IOOpTest("memory__read", 10, true)},
      /*outputs=*/
      {IOOpTest("memory__read", xls::Value(xls::UBits(7, 5)), true),
       IOOpTest("out", 30, true)});
}

TEST_F(TranslatorMemoryTest, MemoryWriteIOOp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<short, 32>& memory) {
         const int addr = in.read();
         memory[addr] = 44;
       })";

  auto memory_write_tuple = xls::Value::Tuple(
      {xls::Value(xls::SBits(5, 5)), xls::Value(xls::SBits(44, 16))});

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("memory__write", memory_write_tuple, true)});
}

TEST_F(TranslatorMemoryTest, MemoryWriteStructIOOp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       struct Stuff {
         unsigned short x;
         unsigned short y;

         Stuff() : x(0), y(0) { }
         Stuff(int val) : x(val >> 16), y(val & 0b1111) { }
         operator int()const {return x + y;}
       };
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<Stuff, 32>& memory) {
         const int addr = in.read();
         memory[addr] = Stuff(44);
       })";

  auto memory_write_tuple =
      xls::Value::Tuple({xls::Value(xls::SBits(5, 5)),
                         xls::Value::Tuple({xls::Value(xls::SBits(12, 16)),
                                            xls::Value(xls::SBits(0, 16))})});

  IOTest(content,
         /*inputs=*/{IOOpTest("in", 5, true)},
         /*outputs=*/{IOOpTest("memory__write", memory_write_tuple, true)});
}

TEST_F(TranslatorMemoryTest, MemoryReadWriteIOOp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<short, 32>& memory) {
         const int addr = in.read();

         memory[addr+1] = memory[addr]+3;
       })";

  auto memory_write_tuple = xls::Value::Tuple(
      {xls::Value(xls::SBits(6, 5)), xls::Value(xls::SBits(13, 16))});

  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", 5, true),
          IOOpTest("memory__read", xls::Value(xls::UBits(10, 16)), true)},
         /*outputs=*/
         {IOOpTest("memory__read", xls::Value(xls::UBits(5, 5)), true),
          IOOpTest("memory__write", memory_write_tuple, true)});
}

TEST_F(TranslatorMemoryTest, MemoryReadProc) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_memory<short, 21>& foo_store,
             __xls_channel<int>& out) {
      out.write(foo_store[13] + 3);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("foo_store");
    ch_in->set_type(MEMORY);
    ch_in->set_depth(21);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(50, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(12, 16))})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(103, 32)),
                    xls::Value(xls::SBits(53, 32)),
                    xls::Value(xls::SBits(15, 32))};
  outputs["foo_store__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      })};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorMemoryTest, MemoryWriteProc) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_memory<short, 21>& memory,
             __xls_channel<int>& in) {
      static unsigned addr = 13;
      memory[addr++] = in.read() + 5;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("memory");
    ch_in->set_type(MEMORY);
    ch_in->set_depth(21);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("in");
    ch_out->set_is_input(true);
    ch_out->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(122, 32)),
                  xls::Value(xls::SBits(10, 32))};
  inputs["memory__write_response"] = {xls::Value::Tuple({}),
                                      xls::Value::Tuple({})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["memory__write_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),    // addr
          xls::Value(xls::SBits(127, 16)),  // value
          xls::Value::Tuple({})             // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(14, 5)),   // addr
          xls::Value(xls::SBits(15, 16)),  // value
          xls::Value::Tuple({})            // mask
      })};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorMemoryTest, MemoryReadWriteProc) {
  const std::string content = R"(
    #include "/xls_builtin.h"
    #pragma hls_top
    void my_package(__xls_memory<short, 55>& barstore) {
      static unsigned addr = 11;
      barstore[addr-1] = barstore[addr]+5;
      addr += 3;
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("barstore");
    ch_in->set_type(MEMORY);
    ch_in->set_depth(55);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["barstore__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(50, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(12, 16))})};
  inputs["barstore__write_response"] = {xls::Value::Tuple({}),
                                        xls::Value::Tuple({})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["barstore__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(11, 6)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(14, 6)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(17, 6)),  // addr
          xls::Value::Tuple({})           // mask
      })};
  outputs["barstore__write_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(10, 6)),    // addr
          xls::Value(xls::SBits(105, 16)),  // value
          xls::Value::Tuple({})             // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 6)),   // addr
          xls::Value(xls::SBits(55, 16)),  // value
          xls::Value::Tuple({})            // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(16, 6)),   // addr
          xls::Value(xls::SBits(17, 16)),  // value
          xls::Value::Tuple({})            // mask
      })};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorMemoryTest, MemoryReadInPipelinedLoop) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_memory<short, 21>& foo_store,
             __xls_channel<int>& out) {
      short ret = 100;
      #pragma hls_pipeline_init_interval 1
      for(int i=0;i<3;++i) {
        ret += foo_store[13+i];
      }
      out.write(ret);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("foo_store");
    ch_in->set_type(MEMORY);
    ch_in->set_depth(21);

    HLSChannel* ch_out = block_spec.add_channels();
    ch_out->set_name("out");
    ch_out->set_is_input(false);
    ch_out->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(10, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(1, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(10, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(2, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(3, 16))})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(211, 32)),
                    xls::Value(xls::SBits(115, 32))};
  outputs["foo_store__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(14, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(15, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(14, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(15, 5)),  // addr
          xls::Value::Tuple({})           // mask
      })};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorMemoryTest, IOProcClassMemory) {
  const std::string content = R"(
       class Block {
        public:
         __xls_memory<short, 21>& foo_store;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          out.write(3*foo_store[13]);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(50, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(12, 16))})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(300, 64)),
                    xls::Value(xls::SBits(150, 64)),
                    xls::Value(xls::SBits(36, 64))};
  outputs["foo_store__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      })};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels {
      name: "foo_store"
      type: MEMORY
      width_in_bits: 16
      depth: 21
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

TEST_F(TranslatorMemoryTest, IOProcClassMemorySubroutine) {
  const std::string content = R"(
       class Block {
        public:
         __xls_memory<short, 21>& foo_store;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         int ReadIt(int addr) {
          return foo_store[addr];
         }

         #pragma hls_top
         void Run() {
          out.write(3*ReadIt(13));
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(50, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(12, 16))})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(300, 64)),
                    xls::Value(xls::SBits(150, 64)),
                    xls::Value(xls::SBits(36, 64))};
  outputs["foo_store__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      })};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels {
      name: "foo_store"
      type: MEMORY
      width_in_bits: 16
      depth: 21
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

TEST_F(TranslatorMemoryTest, IOProcClassMemorySubroutine2) {
  const std::string content = R"(
       class Block {
        public:
         __xls_memory<short, 21>& foo_store;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         int ReadIt(__xls_memory<short, 21>& mem, int addr) {
          return mem[addr];
         }

         #pragma hls_top
         void Run() {
          out.write(3*ReadIt(foo_store, 13));
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(50, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(12, 16))})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(300, 64)),
                    xls::Value(xls::SBits(150, 64)),
                    xls::Value(xls::SBits(36, 64))};
  outputs["foo_store__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      })};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels {
      name: "foo_store"
      type: MEMORY
      width_in_bits: 16
      depth: 21
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

TEST_F(TranslatorMemoryTest, IOProcClassMemorySubroutine3) {
  const std::string content = R"(
       int ReadIt(__xls_memory<short, 21>& mem, int addr) {
        return mem[addr];
       }

       class Block {
        public:
         __xls_memory<short, 21>& foo_store;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          const int x = ReadIt(foo_store, 13);
          out.write(3*x);
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(50, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(12, 16))})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(300, 64)),
                    xls::Value(xls::SBits(150, 64)),
                    xls::Value(xls::SBits(36, 64))};
  outputs["foo_store__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      })};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels {
      name: "foo_store"
      type: MEMORY
      width_in_bits: 16
      depth: 21
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

TEST_F(TranslatorMemoryTest, IOProcClassMemorySubroutine4) {
  const std::string content = R"(
       void WriteIt(__xls_memory<short, 21>& mem, int addr) {
        static int value = 100;
        mem[addr] = value;
        ++value;
       }

       class Block {
        public:
         __xls_memory<short, 21>& foo_store;
         __xls_channel<long, __xls_channel_dir_In>& in;

         #pragma hls_top
         void Run() {
          WriteIt(foo_store, in.read());
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__write_response"] = {xls::Value::Tuple({}),
                                         xls::Value::Tuple({})};
  inputs["in"] = {xls::Value(xls::SBits(3, 64)), xls::Value(xls::SBits(8, 64)),
                  xls::Value(xls::SBits(15, 64))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["foo_store__write_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(3, 5)),     // addr
          xls::Value(xls::SBits(100, 16)),  // value
          xls::Value::Tuple({})             // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(8, 5)),     // addr
          xls::Value(xls::SBits(101, 16)),  // value
          xls::Value::Tuple({})             // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(15, 5)),    // addr
          xls::Value(xls::SBits(102, 16)),  // value
          xls::Value::Tuple({})             // mask
      })};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels {
      name: "foo_store"
      type: MEMORY
      width_in_bits: 16
      depth: 21
    }
    channels {
      name: "in"
      is_input: true
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

TEST_F(TranslatorMemoryTest, IOProcClassMemoryPipelinedLoop) {
  const std::string content = R"(
       class Block {
        public:
         __xls_memory<short, 21>& foo_store;
         __xls_channel<long, __xls_channel_dir_Out>& out;

         int ReadIt(int addr) {
          int ret = 0;
          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<1;++i) {
            ret += foo_store[addr];
          }
          return ret;
         }

         #pragma hls_top
         void Run() {
          out.write(3*ReadIt(13));
         }
      };)";

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["foo_store__read_response"] = {
      xls::Value::Tuple({xls::Value(xls::SBits(100, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(50, 16))}),
      xls::Value::Tuple({xls::Value(xls::SBits(12, 16))})};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(300, 64)),
                    xls::Value(xls::SBits(150, 64)),
                    xls::Value(xls::SBits(36, 64))};
  outputs["foo_store__read_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value::Tuple({})           // mask
      })};
  ProcTest(content, /*block_spec=*/std::nullopt, inputs, outputs,
           /* min_ticks = */ 3);

  XLS_ASSERT_OK_AND_ASSIGN(xlscc::HLSBlock meta, GetBlockSpec());

  const std::string ref_meta_str = R"(
    channels {
      name: "foo_store"
      type: MEMORY
      width_in_bits: 16
      depth: 21
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

TEST_F(TranslatorMemoryTest, MemoryWriteReferenceParameterOutput) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       void output_value(short& output, int input) {
        output += input;
       }
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<short, 32>& memory) {
         const int addr = in.read();
         output_value(memory[addr], 44);
       })";

  auto memory_write_tuple = xls::Value::Tuple(
      {xls::Value(xls::SBits(5, 5)), xls::Value(xls::SBits(64, 16))});

  IOTest(content,
         /*inputs=*/
         {IOOpTest("in", 5, true),
          IOOpTest("memory__read", xls::Value(xls::SBits(20, 16)), true)},
         /*outputs=*/
         {
             IOOpTest("memory__read", xls::Value(xls::UBits(5, 5)), true),
             IOOpTest("memory__write", memory_write_tuple, true),
         });
}

TEST_F(TranslatorMemoryTest, ReferenceToMemoryOp) {
  const std::string content = R"(
       #include "/xls_builtin.h"
       #pragma hls_top
       void my_package(__xls_channel<int>& in,
                       __xls_memory<short, 32>& memory) {
         const int addr = in.read();
         short& val = memory[addr];
         val += 44;
       })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("eferences to side effecting operations")));
}

TEST_F(TranslatorMemoryTest, MemoryReadWriteOperatorTest) {
  constexpr std::string_view content = R"(
    #pragma hls_top
    void foo(__xls_memory<int, 1024> & mem, __xls_channel<int> addr_in,
             __xls_channel<int> out) {
      const int addr = addr_in.read();
      int value = mem[addr];
      value = value * 2;
      mem[addr] = value;
      out.write(value);
    }
  )";
  IOTest(content,
         /*inputs=*/
         {
             IOOpTest("addr_in", 5, true),
             IOOpTest("mem__read", 10, true),
         },
         /*outputs=*/
         {
             IOOpTest("mem__read", xls::Value(xls::UBits(5, 10)), true),
             IOOpTest("mem__write",
                      xls::Value::Tuple({xls::Value(xls::UBits(5, 10)),
                                         xls::Value(xls::UBits(10 * 2, 32))}),
                      true),
             IOOpTest("out", 20, true),
         });
}

TEST_F(TranslatorMemoryTest, MemoryReadWriteFnTest) {
  constexpr std::string_view content = R"(
    #pragma hls_top
    void foo(__xls_memory<int, 1024> & mem, __xls_channel<int> addr_in,
             __xls_channel<int> out) {
      const int addr = addr_in.read();
      int value = mem.read(addr);
      value = value * 2;
      mem.write(addr, value);
      out.write(value);
    }
  )";
  IOTest(content,
         /*inputs=*/
         {IOOpTest("addr_in", 5, true), IOOpTest("mem__read", 10, true)},
         /*outputs=*/
         {
             IOOpTest("mem__read", xls::Value(xls::UBits(5, 10)), true),
             IOOpTest("mem__write",
                      xls::Value::Tuple({xls::Value(xls::UBits(5, 10)),
                                         xls::Value(xls::UBits(10 * 2, 32))}),
                      true),
             IOOpTest("out", 10 * 2, true),
         });
}

}  // namespace
}  // namespace xlscc
