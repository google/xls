// Copyright 2020 The XLS Authors
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

#include <bitset>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

// Simple tests of arithmetic, logical, and bit twiddling IR operations which
// operate on Bits types.
class BasicOpsTest : public IrTestBase {};

TEST_F(BasicOpsTest, LogicalNot) {
  std::string text = R"(
package LogicalNot

top fn main(x: bits[32]) -> bits[32] {
  ret result: bits[32] = not(x)
}
)";

  RunAndExpectEq({{"x", 0}}, 0xffffffffULL, text);
  RunAndExpectEq({{"x", 0xffff0000ULL}}, 0xffff, text);
  RunAndExpectEq({{"x", 0xabcd}}, 0xffff5432ULL, text);
}

TEST_F(BasicOpsTest, Add64) {
  std::string text = R"(
package Add64

top fn main(x: bits[64], y: bits[64]) -> bits[64] {
  ret result: bits[64] = add(x, y)
}
)";

  RunAndExpectEq({{"x", 42}, {"y", 123}}, 165, text);
  RunAndExpectEq({{"x", std::numeric_limits<uint64_t>::max()}, {"y", 1}}, 0,
                 text);
}

TEST_F(BasicOpsTest, BitSlice) {
  constexpr char text_template[] = R"(
package BitSlice

top fn main(x: bits[$0]) -> bits[$2] {
  ret result: bits[$2] = bit_slice(x, start=$1, width=$2)
}
)";
  auto gen_bitslice = [&](int64_t input_width, int64_t start,
                          int64_t output_width) {
    return absl::Substitute(text_template, input_width, start, output_width);
  };

  RunAndExpectEq({{"x", 1}}, 1, gen_bitslice(1, 0, 1));
  RunAndExpectEq({{"x", 0}}, 0, gen_bitslice(1, 0, 1));

  RunAndExpectEq({{"x", 0x12345678ULL}}, 0x12345678ULL,
                 gen_bitslice(32, 0, 32));
  RunAndExpectEq({{"x", 0x12345678ULL}}, 0x8, gen_bitslice(32, 0, 4));
  RunAndExpectEq({{"x", 0x12345678ULL}}, 0x78, gen_bitslice(32, 0, 8));
  RunAndExpectEq({{"x", 0x12345678ULL}}, 0x234, gen_bitslice(32, 16, 12));
  RunAndExpectEq({{"x", 0x12345678ULL}}, 0, gen_bitslice(32, 17, 1));
  RunAndExpectEq({{"x", 0x12345678ULL}}, 1, gen_bitslice(32, 10, 1));
  RunAndExpectEq({{"x", 0x12345678ULL}}, 2220, gen_bitslice(32, 7, 13));
}

TEST_F(BasicOpsTest, BitSliceUpdateOneBit) {
  constexpr std::string_view text = R"(
package BitSlice

top fn main(x: bits[1]) -> bits[1] {
  zero: bits[1] = literal(value=0)
  ret result: bits[1] = bit_slice_update(x, zero, x)
}
)";
  RunAndExpectEq({{"x", 1}}, 1, text);
  RunAndExpectEq({{"x", 0}}, 0, text);
}

TEST_F(BasicOpsTest, BitSliceUpdateOneBitWithinTwo) {
  constexpr std::string_view text = R"(
package BitSlice

top fn main(x: bits[2]) -> bits[2] {
  zero: bits[1] = literal(value=0)
  ret result: bits[2] = bit_slice_update(x, zero, zero)
}
)";
  RunAndExpectEq({{"x", 0b11}}, 2, text);
  RunAndExpectEq({{"x", 0b10}}, 2, text);
  RunAndExpectEq({{"x", 0b01}}, 0, text);
}

TEST_F(BasicOpsTest, WideAndNot) {
  std::string text = R"(
package WideAndNot

top fn main(x: bits[128], y: bits[128]) -> bits[128] {
  not_y: bits[128] = not(y)
  ret result: bits[128] = and(x, not_y)
}
)";

  Value zero(UBits(0, 128));
  Value all_ones(bits_ops::Not(zero.bits()));
  RunAndExpectEq({{"x", zero}, {"y", zero}}, zero, text);
  RunAndExpectEq({{"x", all_ones}, {"y", zero}}, all_ones, text);
  RunAndExpectEq(
      {{"x", Value(bits_ops::Concat({UBits(0xdeadbeefdeadbeefULL, 64),
                                     UBits(0x1234567812345678ULL, 64)}))},
       {"y", Value(bits_ops::Concat({UBits(0xffeeddccbbaa0099ULL, 64),
                                     UBits(0x1111111111111111ULL, 64)}))}},
      Value(bits_ops::Concat({UBits(0x000122234405be66ULL, 64),
                              UBits(0x0224466802244668ULL, 64)})),
      text);
}

TEST_F(BasicOpsTest, Nand) {
  std::string text = R"(
package test

top fn main(x: bits[2], y: bits[2]) -> bits[2] {
  ret result: bits[2] = nand(x, y)
}
)";

  Value zero(UBits(0, 2));
  Value all_ones(UBits(0b11, 2));
  RunAndExpectEq({{"x", zero}, {"y", zero}}, all_ones, text);
  RunAndExpectEq({{"x", all_ones}, {"y", zero}}, all_ones, text);
  RunAndExpectEq({{"x", all_ones}, {"y", all_ones}}, zero, text);
}

TEST_F(BasicOpsTest, Nor) {
  std::string text = R"(
package test

top fn main(x: bits[2], y: bits[2]) -> bits[2] {
  ret result: bits[2] = nor(x, y)
}
)";

  Value zero(UBits(0, 2));
  Value all_ones(UBits(0b11, 2));
  RunAndExpectEq({{"x", zero}, {"y", zero}}, all_ones, text);
  RunAndExpectEq({{"x", all_ones}, {"y", zero}}, zero, text);
  RunAndExpectEq({{"x", all_ones}, {"y", all_ones}}, zero, text);
}

TEST_F(BasicOpsTest, SignedComparisons) {
  // Function concatenates and returns the results of signed Ge, Gt, Le, and
  // Lt.
  std::string text = R"(
package SignedComparisons

top fn main(x: bits[32], y: bits[32]) -> bits[4] {
  sge.1: bits[1] = sge(x, y)
  sgt.2: bits[1] = sgt(x, y)
  sle.3: bits[1] = sle(x, y)
  slt.4: bits[1] = slt(x, y)
  ret result: bits[4] = concat(sge.1, sgt.2, sle.3, slt.4)
}
)";
  auto run_and_compare = [&](int64_t x, int64_t y, uint64_t expected) {
    RunAndExpectEq({{"x", SBits(x, 32)}, {"y", SBits(y, 32)}},
                   UBits(expected, 4), text);
  };

  run_and_compare(0, 0, 0b1010);
  run_and_compare(10, 10, 0b1010);
  run_and_compare(-10, 10, 0b0011);
  run_and_compare(42, 7, 0b1100);
  run_and_compare(-10, 100, 0b0011);
  run_and_compare(-10, -10, 0b1010);
  run_and_compare(-44, -2222, 0b1100);
  run_and_compare(10, 100, 0b0011);
}

TEST_F(BasicOpsTest, BinarySelect) {
  std::string text = R"(
package Select

top fn main(p: bits[1], x: bits[32], y: bits[32]) -> bits[32] {
  ret result: bits[32] = sel(p, cases=[x, y])
}
)";

  RunAndExpectEq({{"p", 0}, {"x", 42}, {"y", 123}}, 42, text);
  RunAndExpectEq({{"p", 1}, {"x", 42}, {"y", 123}}, 123, text);
}

TEST_F(BasicOpsTest, TwoBitSelectorWithDefault) {
  std::string text = R"(
package Select

top fn main(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret result: bits[32] = sel(p, cases=[x, y, z], default=literal.1)
}
)";

  RunAndExpectEq({{"p", 0}, {"x", 42}, {"y", 123}, {"z", 456}}, 42, text);
  RunAndExpectEq({{"p", 1}, {"x", 42}, {"y", 123}, {"z", 456}}, 123, text);
  RunAndExpectEq({{"p", 2}, {"x", 42}, {"y", 123}, {"z", 456}}, 456, text);
  RunAndExpectEq({{"p", 3}, {"x", 42}, {"y", 123}, {"z", 456}}, 0, text);
}

TEST_F(BasicOpsTest, OneHotZeroBitInput) {
  std::string text = R"(
package test

top fn main(x: bits[1]) -> bits[1] {
  sliced: bits[0] = bit_slice(x, start=1, width=0)
  ret result: bits[1] = one_hot(sliced, lsb_prio=false)
}
)";

  RunAndExpectEq({{"x", 0}}, 1, text);
}

TEST_F(BasicOpsTest, OneHotWithMsbPriority3bInput) {
  std::string text = R"(
package test

top fn main(x: bits[3]) -> bits[4] {
  ret result: bits[4] = one_hot(x, lsb_prio=false)
}
)";

  struct Example {
    uint32_t input;
    uint32_t output;
  };
  std::vector<Example> examples = {
      // Note: when MSb has priority we still tack on the "all zeros" bit as the
      // MSb in the result.
      {0b000, 0b1000},
      // One hot.
      {0b100, 0b0100},
      {0b010, 0b0010},
      {0b001, 0b0001},
      // Two hot.
      {0b011, 0b0010},
      {0b110, 0b0100},
      {0b101, 0b0100},
      // Three hot.
      {0b111, 0b0100},
  };
  for (auto example : examples) {
    LOG(INFO) << "Example: " << example.input << " = "
              << std::bitset<32>(example.input);
    RunAndExpectEq({{"x", example.input}}, example.output, text);
  }
}

TEST_F(BasicOpsTest, OneHotSelect) {
  std::string text = R"(
package Select

top fn main(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  one_hot.1: bits[3] = one_hot(p, lsb_prio=true)
  ret result: bits[32] = one_hot_sel(one_hot.1, cases=[x, y, z])
}
)";

  RunAndExpectEq({{"p", 0b00}, {"x", 42}, {"y", 123}, {"z", 456}}, 456, text);
  RunAndExpectEq({{"p", 0b01}, {"x", 42}, {"y", 123}, {"z", 456}}, 42, text);
  RunAndExpectEq({{"p", 0b10}, {"x", 42}, {"y", 123}, {"z", 456}}, 123, text);
  RunAndExpectEq({{"p", 0b11}, {"x", 42}, {"y", 123}, {"z", 456}}, 42, text);
}

TEST_F(BasicOpsTest, MultipleOneHotSelect) {
  std::string text = R"(
package Select

top fn main(p0: bits[3], p1: bits[2], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  one_hot_sel.1: bits[32] = one_hot_sel(p0, cases=[w, x, y])
  ret result: bits[32] = one_hot_sel(p1, cases=[one_hot_sel.1, z])
}
)";

  RunAndExpectEq({{"p0", 0b001},
                  {"p1", 0b01},
                  {"w", 33},
                  {"x", 42},
                  {"y", 123},
                  {"z", 456}},
                 33, text);
  RunAndExpectEq({{"p0", 0b010},
                  {"p1", 0b01},
                  {"w", 33},
                  {"x", 42},
                  {"y", 123},
                  {"z", 456}},
                 42, text);
  RunAndExpectEq({{"p0", 0b010},
                  {"p1", 0b10},
                  {"w", 33},
                  {"x", 42},
                  {"y", 123},
                  {"z", 456}},
                 456, text);
  RunAndExpectEq({{"p0", 0b100},
                  {"p1", 0b10},
                  {"w", 33},
                  {"x", 42},
                  {"y", 123},
                  {"z", 456}},
                 456, text);
}

TEST_F(BasicOpsTest, Clz) {
  std::unique_ptr<VerifiedPackage> p = CreatePackage();
  FunctionBuilder fb("main", p.get());
  auto x = fb.Param("x", p->GetBitsType(3));
  fb.Clz(x);

  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("main"));
  std::string text = p->DumpIr();

  struct Example {
    uint32_t input;
    uint32_t output;
  };
  std::vector<Example> examples = {
      {0b000, 3},
      // One hots.
      {0b001, 2},
      {0b010, 1},
      {0b100, 0},
      // Two hots.
      {0b101, 0},
      {0b110, 0},
      {0b011, 1},
  };
  for (auto example : examples) {
    LOG(INFO) << "Example: " << example.input;
    RunAndExpectEq({{"x", example.input}}, example.output, text);
  }
}

TEST_F(BasicOpsTest, Decode) {
  std::string text = R"(
package Decode

top fn main(x: bits[3]) -> bits[8] {
  ret result: bits[8] = decode(x, width=8)
}
)";

  RunAndExpectEq({{"x", 0}}, 1, text);
  RunAndExpectEq({{"x", 1}}, 2, text);
  RunAndExpectEq({{"x", 2}}, 4, text);
  RunAndExpectEq({{"x", 7}}, 128, text);
}

TEST_F(BasicOpsTest, NarrowedDecode) {
  std::string text = R"(
package NarrowedDecode

top fn main(x: bits[8]) -> bits[27] {
  ret result: bits[27] = decode(x, width=27)
}
)";

  RunAndExpectEq({{"x", 0}}, 1, text);
  RunAndExpectEq({{"x", 1}}, 2, text);
  RunAndExpectEq({{"x", 17}}, (1LL << 17), text);
  RunAndExpectEq({{"x", 26}}, (1LL << 26), text);
  RunAndExpectEq({{"x", 27}}, 0, text);
  RunAndExpectEq({{"x", 123}}, 0, text);
  RunAndExpectEq({{"x", 255}}, 0, text);
}

TEST_F(BasicOpsTest, Encode) {
  std::string text = R"(
package Encode

top fn main(x: bits[5]) -> bits[3] {
  ret result: bits[3] = encode(x)
}
)";
  RunAndExpectEq({{"x", 0}}, 0, text);
  RunAndExpectEq({{"x", 1}}, 0, text);
  RunAndExpectEq({{"x", 8}}, 3, text);
  RunAndExpectEq({{"x", 16}}, 4, text);

  // Non-one-hot values.
  RunAndExpectEq({{"x", 5}}, 2, text);
  RunAndExpectEq({{"x", 18}}, 5, text);
}

TEST_F(BasicOpsTest, Reverse) {
  std::string text = R"(
package Reverse

top fn main(x: bits[4]) -> bits[4] {
  ret result: bits[4] = reverse(x)
}
)";

  RunAndExpectEq({{"x", 0b0000}}, 0b0000, text);
  RunAndExpectEq({{"x", 0b1000}}, 0b0001, text);
  RunAndExpectEq({{"x", 0b0011}}, 0b1100, text);
}

TEST_F(BasicOpsTest, NonBitsTypedLiteral) {
  std::string text = R"(
package TupleLiteral

top fn main(x: bits[4]) -> (bits[4], bits[8]) {
  ret result: (bits[4], bits[8]) = literal(value=(1, 2))
}
)";

  RunAndExpectEq({{"x", Value(UBits(0, 4))}},
                 Value::Tuple({Value(UBits(1, 4)), Value(UBits(2, 8))}), text);
}

TEST_F(BasicOpsTest, AndReduce) {
  std::string text = R"(
package Decode

top fn main(x: bits[8]) -> bits[1] {
  ret result: bits[1] = and_reduce(x)
}
)";

  RunAndExpectEq({{"x", 0}}, 0, text);
  RunAndExpectEq({{"x", 0xff}}, 1, text);
  RunAndExpectEq({{"x", 0x7f}}, 0, text);
  RunAndExpectEq({{"x", 0xfe}}, 0, text);
}

TEST_F(BasicOpsTest, OrReduce) {
  std::string text = R"(
package Decode

top fn main(x: bits[8]) -> bits[1] {
  ret result: bits[1] = or_reduce(x)
}
)";

  RunAndExpectEq({{"x", 0}}, 0, text);
  RunAndExpectEq({{"x", 0xff}}, 1, text);
  RunAndExpectEq({{"x", 0x7f}}, 1, text);
  RunAndExpectEq({{"x", 0xfe}}, 1, text);
}

TEST_F(BasicOpsTest, XorrReduce) {
  std::string text = R"(
package Decode

top fn main(x: bits[8]) -> bits[1] {
  ret result: bits[1] = xor_reduce(x)
}
)";

  RunAndExpectEq({{"x", 0}}, 0, text);
  RunAndExpectEq({{"x", 0xff}}, 0, text);
  RunAndExpectEq({{"x", 0x7f}}, 1, text);
  RunAndExpectEq({{"x", 0xfe}}, 1, text);
}

TEST_F(BasicOpsTest, ArraySlice) {
  std::string text = R"(
package ArraySlice

top fn main(start: bits[32]) -> bits[32][4] {
   literal.1: bits[32][8] = literal(value=[5, 6, 7, 8, 9, 10, 11, 12])
   ret array_slice.3: bits[32][4] = array_slice(literal.1, start, width=4)
}
)";

  {
    XLS_ASSERT_OK_AND_ASSIGN(Value correct_result,
                             Value::UBitsArray({8, 9, 10, 11}, 32));
    RunAndExpectEq({{"start", Value(UBits(3, 32))}}, correct_result, text);
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(Value correct_result,
                             Value::UBitsArray({11, 12, 12, 12}, 32));
    RunAndExpectEq({{"start", Value(UBits(6, 32))}}, correct_result, text);
  }
}

}  // namespace
}  // namespace xls
