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
          xls::Value(xls::UBits(1, 1))    // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value(xls::UBits(1, 1))    // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 5)),  // addr
          xls::Value(xls::UBits(1, 1))    // mask
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
          xls::Value(xls::UBits(1, 1))      // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(14, 5)),   // addr
          xls::Value(xls::SBits(15, 16)),  // value
          xls::Value(xls::UBits(1, 1))     // mask
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
          xls::Value(xls::UBits(1, 1))    // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(14, 6)),  // addr
          xls::Value(xls::UBits(1, 1))    // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(17, 6)),  // addr
          xls::Value(xls::UBits(1, 1))    // mask
      })};
  outputs["barstore__write_request"] = {
      xls::Value::Tuple({
          xls::Value(xls::UBits(10, 6)),    // addr
          xls::Value(xls::SBits(105, 16)),  // value
          xls::Value(xls::UBits(1, 1))      // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(13, 6)),   // addr
          xls::Value(xls::SBits(55, 16)),  // value
          xls::Value(xls::UBits(1, 1))     // mask
      }),
      xls::Value::Tuple({
          xls::Value(xls::UBits(16, 6)),   // addr
          xls::Value(xls::SBits(17, 16)),  // value
          xls::Value(xls::UBits(1, 1))     // mask
      })};

  ProcTest(content, block_spec, inputs, outputs);
}

}  // namespace
}  // namespace xlscc
