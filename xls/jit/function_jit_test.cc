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

#include "xls/jit/function_jit.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <ios>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fuzztest/fuzztest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "xls/common/bits_util.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/ir_evaluator_test_base.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/fuzz_type_domain.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

using status_testing::IsOk;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::HasSubstr;
using testing::TestParamInfo;
using testing::Values;

// TODO(https://github.com/google/xls/issues/506): 2021-10-12 Replace the empty
// events returned by the JIT evaluator with a entry point that includes the
// collected events (once they are supported by the JIT).
INSTANTIATE_TEST_SUITE_P(
    FunctionJitTest, IrEvaluatorTestBase,
    Values(IrEvaluatorTestParam(
        [](Function* function, absl::Span<const Value> args)
            -> absl::StatusOr<InterpreterResult<Value>> {
          XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));
          return jit->Run(args);
        },
        [](Function* function,
           const absl::flat_hash_map<std::string, Value>& kwargs)
            -> absl::StatusOr<InterpreterResult<Value>> {
          XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));
          return jit->Run(kwargs);
        })));

absl::StatusOr<Value> RunJitNoEvents(FunctionJit* jit,
                                     absl::Span<const Value> args) {
  XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> result, jit->Run(args));

  if (!result.events.trace_msgs.empty()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Unexpected traces generated during RunJitNoEvents:\n%s",
        absl::StrJoin(result.events.trace_msgs, "\n")));
  }

  return InterpreterResultToStatusOrValue(result);
}

TEST(FunctionJitTest, TraceFmtNoArgsTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn trace_no_args(tkn: token, pred: bits[1]) -> token {
    ret trace.1: token = trace(tkn, pred, format="hi I traced", data_operands=[], id=1)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  std::vector<Value> args = {Value::Token(), Value(UBits(1, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result, jit->Run(args));
  ASSERT_EQ(result.events.trace_msgs.size(), 1);
  EXPECT_EQ(result.events.trace_msgs.at(0), "hi I traced");
}

TEST(FunctionJitTest, TraceFmtOneArgTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn trace_no_args(tkn: token, pred: bits[1]) -> token {
    ret trace.1: token = trace(tkn, pred, format="hi I traced: {}", data_operands=[pred], id=1)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  std::vector<Value> args = {Value::Token(), Value(UBits(1, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result, jit->Run(args));
  ASSERT_EQ(result.events.trace_msgs.size(), 1);
  EXPECT_EQ(result.events.trace_msgs.at(0), "hi I traced: 1");
}

TEST(FunctionJitTest, TraceFmtSignedTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn trace_no_args(tkn: token, x: bits[8]) -> token {
    pred: bits[1] = literal(value=1, id=0)
    trace.1: token = trace(tkn, pred, format="signed: {:d}", data_operands=[x], id=1)
    trace.2: token = trace(trace.1, pred, format="unsigned: {:u}", data_operands=[x], id=2)
    ret trace.3: token = trace(trace.2, pred, format="default: {}", data_operands=[x], id=3)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  std::vector<Value> args = {Value::Token(), Value(UBits(0xff, 8))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result, jit->Run(args));
  ASSERT_EQ(result.events.trace_msgs.size(), 3);
  EXPECT_EQ(result.events.trace_msgs.at(0), "signed: -1");
  EXPECT_EQ(result.events.trace_msgs.at(1), "unsigned: 255");
  EXPECT_EQ(result.events.trace_msgs.at(2), "default: 255");
}

TEST(FunctionJitTest, TraceFmtTwoArgTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn trace_no_args(tkn: token, pred: bits[1]) -> token {
    literal.0: bits[64] = literal(value=42)
    ret trace.1: token = trace(tkn, pred, format="hi I traced: {} also: {:x}", data_operands=[pred, literal.0], id=1)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  std::vector<Value> args = {Value::Token(), Value(UBits(1, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result, jit->Run(args));
  ASSERT_EQ(result.events.trace_msgs.size(), 1);
  EXPECT_EQ(result.events.trace_msgs.at(0), "hi I traced: 1 also: 2a");
}

TEST(FunctionJitTest, TraceFmtBigArgTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn trace_no_args(tkn: token, pred: bits[1]) -> token {
    literal.0: bits[512] = literal(value=1)
    literal.511: bits[32] = literal(value=511)
    shll.1: bits[512] = shll(literal.0, literal.511)
    ret trace.2: token = trace(tkn, pred, format="hi I traced: {:x}", data_operands=[shll.1])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  std::vector<Value> args = {Value::Token(), Value(UBits(1, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result, jit->Run(args));
  ASSERT_EQ(result.events.trace_msgs.size(), 1);
  EXPECT_EQ(result.events.trace_msgs.at(0),
            "hi I traced: "
            "800000000000000000000000000000000000000000000000000000000000000000"
            "00000000000000000000000000000000000000000000000000000000000000");
}

// This test verifies that a compiled JIT function can be reused.
TEST(FunctionJitTest, ReuseTest) {
  Package package("my_package");
  std::string ir_text = R"(
  fn get_identity(x: bits[8]) -> bits[8] {
    ret identity.1: bits[8] = identity(x)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  EXPECT_THAT(RunJitNoEvents(jit.get(), {Value(UBits(2, 8))}),
              IsOkAndHolds(Value(UBits(2, 8))));
  EXPECT_THAT(RunJitNoEvents(jit.get(), {Value(UBits(4, 8))}),
              IsOkAndHolds(Value(UBits(4, 8))));
  EXPECT_THAT(RunJitNoEvents(jit.get(), {Value(UBits(7, 8))}),
              IsOkAndHolds(Value(UBits(7, 8))));
}

TEST(FunctionJitTest, OneHotZeroBit) {
  Package package("my_package");
  std::string ir_text = R"(
  fn get_identity(x: bits[0]) -> bits[1] {
    ret one_hot: bits[1] = one_hot(x, lsb_prio=true)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  EXPECT_THAT(RunJitNoEvents(jit.get(), {Value(UBits(0, 0))}),
              IsOkAndHolds(Value(UBits(1, 1))));
}

// Very basic smoke test for packed and unpacked types.
TEST(FunctionJitTest, PackedAndUnpackedSmoke) {
  Package package("my_package");
  std::string ir_text = R"(
  fn get_identity(x: bits[8]) -> bits[8] {
    ret identity.1: bits[8] = identity(x)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  uint8_t input_data[] = {0x5a};
  {
    uint8_t output_data = 0;
    PackedBitsView<8> input(input_data, 0);
    PackedBitsView<8> output(&output_data, 0);
    XLS_ASSERT_OK(jit->RunWithPackedViews(input, output));
    EXPECT_EQ(output_data, 0x5a);
  }

  {
    uint8_t output_data = 0;
    BitsView<8> unpacked_input(input_data);
    MutableBitsView<8> unpacked_output(&output_data);
    XLS_ASSERT_OK(jit->RunWithUnpackedViews(unpacked_input, unpacked_output));
    EXPECT_EQ(output_data, 0x5a);
  }
}

TEST(FunctionJitTest, PackedAndUnpackedSmokeWide) {
  Package package("my_package");
  std::string ir_text = R"(
  fn get_identity(x: bits[80]) -> bits[80] {
    one: bits[80] = literal(value=1)
    ret x_plus_one: bits[80] = add(x, one)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  // 80-bit data is represented as an i128 with 16 byte alignment.
  // TODO(allight): 2023-11-30: The fact this is needed is unfortunate.
  alignas(16)
      uint8_t input_data[] = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa};
  {
    uint8_t output_data[10];
    PackedBitsView<80> input(input_data, 0);
    PackedBitsView<80> output(output_data, 0);
    XLS_ASSERT_OK(jit->RunWithPackedViews(input, output));
    EXPECT_THAT(output_data,
                ElementsAre(0x2, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa));
  }

  {
    // 80-bit data is represented as an i128 with 16 byte alignment.
    // TODO(allight): 2023-11-30: The fact this is needed is unfortunate.
    alignas(16) uint8_t output_data[10];
    BitsView<80> input(input_data);
    MutableBitsView<80> output(output_data);
    XLS_ASSERT_OK(jit->RunWithUnpackedViews(input, output));
    EXPECT_THAT(output_data,
                ElementsAre(0x2, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa));
  }
}

// Tests PackedBitView<X> input/output handling.
template <int64_t kBitWidth>
absl::Status TestPackedBits(absl::BitGenRef bitgen) {
  Package package("my_package");
  std::string ir_template = R"(
  fn get_identity(x: bits[$0], y:bits[$0]) -> bits[$0] {
    ret add.1: bits[$0] = add(x, y)
  }
  )";
  std::string ir_text = absl::Substitute(ir_template, kBitWidth);
  XLS_ASSIGN_OR_RETURN(Function * function,
                       Parser::ParseFunction(ir_text, &package));
  XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));
  Value v = RandomValue(package.GetBitsType(kBitWidth), bitgen);
  Bits a(v.bits());
  v = RandomValue(package.GetBitsType(kBitWidth), bitgen);
  Bits b(v.bits());
  Bits expected = bits_ops::Add(a, b);
  int64_t byte_width = jit->GetPackedReturnTypeSize();
  auto output_data = std::make_unique<uint8_t[]>(byte_width);
  std::memset(output_data.get(), 0, byte_width);

  auto a_vector = a.ToBytes();
  auto b_vector = b.ToBytes();
  auto expected_vector = expected.ToBytes();
  PackedBitsView<kBitWidth> a_view(a_vector.data(), 0);
  PackedBitsView<kBitWidth> b_view(b_vector.data(), 0);
  PackedBitsView<kBitWidth> output(output_data.get(), 0);
  XLS_RETURN_IF_ERROR(jit->RunWithPackedViews(a_view, b_view, output));
  for (int i = 0; i < CeilOfRatio(kBitWidth, kCharBit); i++) {
    XLS_RET_CHECK(output_data[i] == expected_vector[i])
        << std::hex << ": byte " << i << ": "
        << static_cast<uint64_t>(output_data[i]) << " vs. "
        << static_cast<uint64_t>(expected_vector[i]);
  }
  return absl::OkStatus();
}

// Tests UnackedBitView<X> input/output handling.
template <int64_t kBitWidth>
absl::Status TestUnpackedBits(absl::BitGenRef bitgen) {
  Package package("my_package");
  std::string ir_template = R"(
  fn get_identity(x: bits[$0], y:bits[$0]) -> bits[$0] {
    ret add.1: bits[$0] = add(x, y)
  }
  )";
  std::string ir_text = absl::Substitute(ir_template, kBitWidth);
  XLS_ASSIGN_OR_RETURN(Function * function,
                       Parser::ParseFunction(ir_text, &package));
  XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));
  Value v = RandomValue(package.GetBitsType(kBitWidth), bitgen);
  Bits a(v.bits());
  v = RandomValue(package.GetBitsType(kBitWidth), bitgen);
  Bits b(v.bits());
  Bits expected = bits_ops::Add(a, b);
  int64_t byte_width = jit->GetPackedReturnTypeSize();
  auto output_data = std::make_unique<uint8_t[]>(byte_width);
  std::memset(output_data.get(), 0, byte_width);

  auto a_vector = a.ToBytes();
  auto b_vector = b.ToBytes();
  auto expected_vector = expected.ToBytes();
  BitsView<kBitWidth> a_view(a_vector.data());
  BitsView<kBitWidth> b_view(b_vector.data());
  MutableBitsView<kBitWidth> output(output_data.get());
  XLS_RETURN_IF_ERROR(jit->RunWithPackedViews(a_view, b_view, output));
  for (int i = 0; i < CeilOfRatio(kBitWidth, kCharBit); i++) {
    XLS_RET_CHECK(output_data[i] == expected_vector[i])
        << std::hex << ": byte " << i << ": "
        << static_cast<uint64_t>(output_data[i]) << " vs. "
        << static_cast<uint64_t>(expected_vector[i]);
  }
  return absl::OkStatus();
}

// Smoke test of PackedBitsViews in the JIT.
TEST(FunctionJitTest, PackedBits) {
  std::minstd_rand bitgen;

  // The usual suspects:
  XLS_ASSERT_OK(TestPackedBits<1>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<2>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<4>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<8>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<16>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<32>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<64>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<128>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<256>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<512>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<1024>(bitgen));

  // Now some weirdos:
  XLS_ASSERT_OK(TestPackedBits<7>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<15>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<44>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<543>(bitgen));
  XLS_ASSERT_OK(TestPackedBits<1000>(bitgen));
}

// Smoke test of UnpackedBitsViews in the JIT.
TEST(FunctionJitTest, UnpackedBits) {
  std::minstd_rand bitgen;

  // The usual suspects:
  XLS_ASSERT_OK(TestUnpackedBits<1>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<2>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<4>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<8>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<16>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<32>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<64>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<128>(bitgen));

  // Now some weirdos:
  XLS_ASSERT_OK(TestUnpackedBits<7>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<15>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<44>(bitgen));
  XLS_ASSERT_OK(TestUnpackedBits<127>(bitgen));
}

// Concatenates the contents of several Bits objects into a single one.
// Operates differently than bits_ops::Concat, as input[0] remains the LSbits.
Bits VectorToPackedBits(const std::vector<Bits>& input) {
  BitsRope rope(input[0].bit_count() * input.size());
  for (const Bits& bits : input) {
    for (int i = 0; i < bits.bit_count(); i++) {
      rope.push_back(bits.Get(i));
    }
  }

  return rope.Build();
}

void FlattenValue(const Value& value, BitsRope& rope) {
  if (value.IsBits()) {
    rope.push_back(value.bits());
  } else if (value.IsArray()) {
    for (const Value& element : value.elements()) {
      FlattenValue(element, rope);
    }
  } else if (value.IsTuple()) {
    // Tuple elements are declared MSelement to LSelement, so we need to pack
    // them in "reverse" order, so the LSelement is at the LSb.
    for (int64_t i = value.elements().size() - 1; i >= 0; i--) {
      FlattenValue(value.elements()[i], rope);
    }
  }
}

std::vector<uint8_t> FlattenValue(const Value& value) {
  BitsRope rope(value.GetFlatBitCount());
  FlattenValue(value, rope);
  std::vector<uint8_t> bytes = rope.Build().ToBytes();
  return bytes;
}

// Utility struct to hold different representations of the same data together.
template <typename ViewT>
struct TestData {
  explicit TestData(const Value& value_in)
      : value(value_in), bytes(FlattenValue(value)), view(bytes.data(), 0) {}
  Value value;
  std::vector<uint8_t> bytes;
  ViewT view;
};

// Tests PackedArrayView input/output from the JIT. Takes in an array, an index,
// and a replacement value, and does an array_update(). We then verify that the
// output array looks like expected.
template <int64_t kBitWidth, int64_t kNumElements>
absl::Status TestSimpleArray(absl::BitGenRef bitgen) {
  using ArrayT = PackedArrayView<PackedBitsView<kBitWidth>, kNumElements>;

  Package package("my_package");
  std::string ir_template = R"(
  fn array_update(array: bits[$0][$1], idx: bits[$0], new_value: bits[$0]) -> bits[$0][$1] {
    ret array_update.4: bits[$0][$1] = array_update(array, new_value, indices=[idx])
  }
  )";
  std::string ir_text = absl::Substitute(ir_template, kBitWidth, kNumElements);
  XLS_ASSIGN_OR_RETURN(Function * function,
                       Parser::ParseFunction(ir_text, &package));
  XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));

  std::vector<Bits> bits_vector;
  for (int i = 0; i < kNumElements; i++) {
    Value value = RandomValue(package.GetBitsType(kBitWidth), bitgen);
    bits_vector.push_back(value.bits());
  }
  TestData<ArrayT> array_data(Value(VectorToPackedBits(bits_vector)));

  int index = absl::Uniform(bitgen, 0, kNumElements);
  TestData<PackedBitsView<kBitWidth>> index_data(
      Value(UBits(index, kBitWidth)));

  Value value = RandomValue(package.GetBitsType(kBitWidth), bitgen);
  TestData<PackedBitsView<kBitWidth>> replacement_data(value);
  bits_vector[index] = replacement_data.value.bits();

  TestData<ArrayT> expected_data(Value(VectorToPackedBits(bits_vector)));

  TestData<ArrayT> output_data(Value(Bits(kBitWidth * kNumElements)));

  XLS_RETURN_IF_ERROR(jit->RunWithPackedViews(array_data.view, index_data.view,
                                              replacement_data.view,
                                              output_data.view));

  for (int i = 0; i < CeilOfRatio(kBitWidth * kNumElements, kCharBit); i++) {
    XLS_RET_CHECK(output_data.bytes[i] == expected_data.bytes[i])
        << std::hex << ": byte " << i << ": "
        << "0x" << static_cast<int>(output_data.bytes[i]) << " vs. "
        << "0x" << static_cast<int>(expected_data.bytes[i]);
  }
  return absl::OkStatus();
}

TEST(FunctionJitTest, PackedArrays) {
  std::minstd_rand bitgen;
  XLS_ASSERT_OK((TestSimpleArray<4, 4>(bitgen)));
  XLS_ASSERT_OK((TestSimpleArray<4, 15>(bitgen)));
  XLS_ASSERT_OK((TestSimpleArray<113, 33>(bitgen)));
}

// Creates a simple function to perform a tuple update.
absl::StatusOr<Function*> CreateTupleFunction(Package* p, TupleType* tuple_type,
                                              int64_t replacement_index) {
  FunctionBuilder builder("tuple_update", p);
  BValue input_tuple = builder.Param("input_tuple", tuple_type);
  BValue new_element =
      builder.Param("new_element", tuple_type->element_type(replacement_index));
  std::vector<BValue> elements;
  elements.reserve(tuple_type->size());
  for (int i = 0; i < tuple_type->size(); i++) {
    elements.push_back(builder.TupleIndex(input_tuple, i));
  }
  elements[replacement_index] = new_element;
  BValue result_tuple = builder.Tuple(elements);
  return builder.BuildWithReturnValue(result_tuple);
}

// With some template acrobatics, we could eliminate the need for either
// ReplacementT or kReplacementIndex...but I don't think it'd be worth the
// effort.
template <typename TupleT, typename ReplacementT, int kReplacementIndex>
absl::Status TestTuples(absl::BitGenRef bitgen) {
  Package package("my_package");
  TupleType* tuple_type = TupleT::GetFullType(&package);

  Type* replacement_type = tuple_type->element_type(kReplacementIndex);
  XLS_ASSIGN_OR_RETURN(
      Function * function,
      CreateTupleFunction(&package, tuple_type, kReplacementIndex));
  XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));

  Value input_tuple = RandomValue(tuple_type, bitgen);
  TestData<TupleT> input_tuple_data(input_tuple);
  Value replacement = RandomValue(replacement_type, bitgen);
  TestData<ReplacementT> replacement_data(replacement);

  XLS_ASSIGN_OR_RETURN(std::vector<Value> elements, input_tuple.GetElements());
  elements[kReplacementIndex] = replacement;
  TestData<TupleT> expected_data(Value::Tuple(elements));

  // Braces used to fight the most vexing parse.
  TestData<TupleT> output_data(Value{Bits(TupleT::kBitCount)});
  XLS_RETURN_IF_ERROR(jit->RunWithPackedViews(
      input_tuple_data.view, replacement_data.view, output_data.view));

  for (int i = 0; i < CeilOfRatio(TupleT::kBitCount, kCharBit); i++) {
    XLS_RET_CHECK(output_data.bytes[i] == expected_data.bytes[i])
        << std::hex << ": byte " << i << ": "
        << "0x" << static_cast<int>(output_data.bytes[i]) << " vs. "
        << "0x" << static_cast<int>(expected_data.bytes[i]);
  }

  return absl::OkStatus();
}

TEST(FunctionJitTest, PackedTuples) {
  using PackedFloat32T =
      PackedTupleView<PackedBitsView<1>, PackedBitsView<8>, PackedBitsView<23>>;

  std::minstd_rand bitgen;
  {
    using TupleT = PackedTupleView<PackedBitsView<3>, PackedBitsView<7>>;
    using ReplacementT = PackedBitsView<3>;
    XLS_ASSERT_OK((TestTuples<TupleT, ReplacementT, 0>(bitgen)));
  }

  {
    using TupleT = PackedTupleView<PackedBitsView<3>, PackedBitsView<7>>;
    using ReplacementT = PackedBitsView<7>;
    XLS_ASSERT_OK((TestTuples<TupleT, ReplacementT, 1>(bitgen)));
  }

  {
    using TupleT = PackedTupleView<PackedArrayView<PackedFloat32T, 15>,
                                   PackedFloat32T, PackedFloat32T>;
    using ReplacementT = PackedArrayView<PackedFloat32T, 15>;
    XLS_ASSERT_OK((TestTuples<TupleT, ReplacementT, 0>(bitgen)));
  }

  {
    using TupleT = PackedTupleView<PackedArrayView<PackedFloat32T, 15>,
                                   PackedFloat32T, PackedFloat32T>;
    using ReplacementT = PackedFloat32T;
    XLS_ASSERT_OK((TestTuples<TupleT, ReplacementT, 1>(bitgen)));
  }
}

TEST(FunctionJitTest, ArrayConcatArrayOfBits) {
  Package package("my_package");

  std::string ir_text = R"(
  fn f(a0: bits[32][2], a1: bits[32][3]) -> bits[32][7] {
    array_concat.3: bits[32][5] = array_concat(a0, a1)
    ret array_concat.4: bits[32][7] = array_concat(array_concat.3, a0)
  }
  )";

  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::UBitsArray({1, 2, 3, 4, 5, 1, 2}, 32));

  std::vector args{a0, a1};
  EXPECT_THAT(RunJitNoEvents(jit.get(), args), IsOkAndHolds(ret));
}

TEST(FunctionJitTest, ArrayConcatArrayOfBitsMixedOperands) {
  Package package("my_package");

  std::string ir_text = R"(
  fn f(a0: bits[32][2], a1: bits[32][3], a2: bits[32][1]) -> bits[32][7] {
    array_concat.4: bits[32][1] = array_concat(a2)
    array_concat.5: bits[32][2] = array_concat(array_concat.4, array_concat.4)
    array_concat.6: bits[32][7] = array_concat(a0, array_concat.5, a1)
    ret array_concat.7: bits[32][7] = array_concat(array_concat.6)
  }
  )";

  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a2, Value::SBitsArray({-1}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value ret,
      Value::UBitsArray({1, 2, 0xffffffff, 0xffffffff, 3, 4, 5}, 32));

  std::vector args{a0, a1, a2};
  EXPECT_THAT(RunJitNoEvents(jit.get(), args), IsOkAndHolds(ret));
}

TEST(FunctionJitTest, ArrayConcatArrayOfArrays) {
  Package package("my_package");

  std::string ir_text = R"(
  fn f() -> bits[32][2][3] {
    literal.1: bits[32][2][2] = literal(value=[[1, 2], [3, 4]])
    literal.2: bits[32][2][1] = literal(value=[[5, 6]])

    ret array_concat.3: bits[32][2][3] = array_concat(literal.2, literal.1)
  }
  )";

  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::SBits2DArray({{5, 6}, {1, 2}, {3, 4}}, 32));

  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  std::vector<Value> args;
  EXPECT_THAT(RunJitNoEvents(jit.get(), args), IsOkAndHolds(ret));
}

// The assert tests below are duplicates of the ones in
// xls/interpereter/ir_evaluator_test_base.cc because those recompile
// the test function each time they run it. These tests check that
// reusing the test function also works.
TEST(FunctionJitTest, Assert) {
  Package p("assert_test");
  FunctionBuilder b("fun", &p);
  auto p0 = b.Param("tkn", p.GetTokenType());
  auto p1 = b.Param("cond", p.GetBitsType(1));
  b.Assert(p0, p1, "the assertion error message");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(f));

  std::vector<Value> ok_args = {Value::Token(), Value(UBits(1, 1))};
  EXPECT_THAT(RunJitNoEvents(jit.get(), ok_args), IsOkAndHolds(Value::Token()));

  std::vector<Value> fail_args = {Value::Token(), Value(UBits(0, 1))};
  EXPECT_THAT(RunJitNoEvents(jit.get(), fail_args),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("the assertion error message")));
}

TEST(FunctionJitTest, FunAssert) {
  Package p("fun_assert_test");

  FunctionBuilder fun_builder("fun", &p);
  auto x = fun_builder.Param("x", p.GetBitsType(5));

  auto seven = fun_builder.Literal(Value(UBits(7, 5)));
  auto test = fun_builder.ULe(x, seven);

  auto token = fun_builder.Literal(Value::Token());
  fun_builder.Assert(token, test, "x is more than 7");

  auto one = fun_builder.Literal(Value(UBits(1, 5)));
  fun_builder.Add(x, one);

  XLS_ASSERT_OK_AND_ASSIGN(Function * fun, fun_builder.Build());

  FunctionBuilder top_builder("top_f", &p);
  auto y = top_builder.Param("y", p.GetBitsType(5));

  std::vector<BValue> args = {y};
  top_builder.Invoke(args, fun);

  XLS_ASSERT_OK_AND_ASSIGN(Function * top, top_builder.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(top));

  std::vector<Value> ok_args = {Value(UBits(6, 5))};
  EXPECT_THAT(RunJitNoEvents(jit.get(), ok_args),
              IsOkAndHolds(Value(UBits(7, 5))));

  std::vector<Value> fail_args = {Value(UBits(8, 5))};
  EXPECT_THAT(
      RunJitNoEvents(jit.get(), fail_args),
      StatusIs(absl::StatusCode::kAborted, HasSubstr("x is more than 7")));
}

TEST(FunctionJitTest, TwoAssert) {
  Package p("assert_test");
  FunctionBuilder b("fun", &p);
  auto p0 = b.Param("tkn", p.GetTokenType());
  auto p1 = b.Param("cond1", p.GetBitsType(1));
  auto p2 = b.Param("cond2", p.GetBitsType(1));

  BValue token1 = b.Assert(p0, p1, "first assertion error message");
  b.Assert(token1, p2, "second assertion error message");

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(f));

  std::vector<Value> ok_args = {Value::Token(), Value(UBits(1, 1)),
                                Value(UBits(1, 1))};

  EXPECT_THAT(RunJitNoEvents(jit.get(), ok_args), IsOkAndHolds(Value::Token()));

  std::vector<Value> fail1_args = {Value::Token(), Value(UBits(0, 1)),
                                   Value(UBits(1, 1))};

  EXPECT_THAT(RunJitNoEvents(jit.get(), fail1_args),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("first assertion error message")));

  std::vector<Value> fail2_args = {Value::Token(), Value(UBits(1, 1)),
                                   Value(UBits(0, 1))};

  EXPECT_THAT(RunJitNoEvents(jit.get(), fail2_args),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("second assertion error message")));

  std::vector<Value> failboth_args = {Value::Token(), Value(UBits(0, 1)),
                                      Value(UBits(0, 1))};

  // The token-plumbing ensures that the first assertion is checked first,
  // so test that it is reported properly.
  EXPECT_THAT(RunJitNoEvents(jit.get(), failboth_args),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("first assertion error message")));
}

TEST(FunctionJitTest, TokenCompareError) {
  Package p("token_eq");
  FunctionBuilder b("fun", &p);
  auto p0 = b.Param("tkn", p.GetTokenType());

  b.Eq(p0, p0);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(FunctionJit::Create(f),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Tokens are incomparable")));
}

// Make sure the token comparison error is still reported when the token is
// inside a larger structure.
TEST(FunctionJitTest, CompoundTokenCompareError) {
  Package p("compound_token_eq");
  FunctionBuilder b("fun", &p);

  auto p0 = b.Param("tkn", p.GetTokenType());
  BValue two = b.Literal(Value(UBits(2, 32)));

  BValue tup = b.Tuple({p0, two});

  b.Eq(tup, tup);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(FunctionJit::Create(f),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Tokens are incomparable")));
}

TEST(FunctionJitTest, BigFunctionInputsOutputs) {
  Package package("my_package");

  FunctionBuilder fb("test", &package);
  fb.Add(fb.Param("x", package.GetBitsType(256)),
         fb.Param("y", package.GetBitsType(256)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  Bits ret_bits =
      bits_ops::Concat({UBits(0, 256 - 65), UBits(1, 1), UBits(0, 64)});

  // Test using values
  {
    Value x(bits_ops::ZeroExtend(UBits(-1, 64), 256));
    Value y(bits_ops::ZeroExtend(UBits(1, 64), 256));
    Value ret = Value(ret_bits);

    EXPECT_THAT(RunJitNoEvents(jit.get(), {x, y}), IsOkAndHolds(ret));
  }

  // Test using views.
  {
    // TODO(allight): 2023-12-08: The fact that we need alignas is unfortunate.
    alignas(16) std::array<uint8_t, 256 / 8> x_view{
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    alignas(16) std::array<uint8_t, 256 / 8> y_view{
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    alignas(16) std::array<uint8_t, 256 / 8> ret_view{};

    InterpreterEvents events;
    EXPECT_THAT(jit->RunWithViews({x_view.data(), y_view.data()},
                                  absl::MakeSpan(ret_view), &events),
                IsOk());
    EXPECT_EQ(Bits::FromBytes(ret_view, 256), ret_bits);
  }
}

TEST(FunctionJitTest, TupleViewSmokeTest2) {
  Package package("my_package");

  std::string ir_text = R"(
fn f(x: bits[1], y: bits[21]) -> (bits[1], bits[21]) {
  ret tuple.4: (bits[1], bits[21]) = tuple(x, y)
})";

  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  // Test using values.
  {
    Value x(UBits(0x1, 1));
    Value y(UBits(0xabcd, 21));
    Value ret = Value::Tuple(std::vector<Value>{x, y});

    std::vector args{x, y};
    EXPECT_THAT(RunJitNoEvents(jit.get(), args), IsOkAndHolds(ret));
  }

  // Test using views.
  {
    int64_t x = 0x1;
    int64_t y = 0xabcd;
    uint8_t result[8] = {0};

    std::vector<uint8_t*> args{reinterpret_cast<uint8_t*>(&x),
                               reinterpret_cast<uint8_t*>(&y)};
    absl::Span<uint8_t> result_buffer(result);
    InterpreterEvents events;
    XLS_ASSERT_OK(jit->RunWithViews(args, result_buffer, &events));

    xls::TupleView<xls::BitsView<1>, xls::BitsView<21>> result_view(result);
    EXPECT_EQ(result_view.Get<0>().GetValue(), 0x1);
    EXPECT_EQ(result_view.Get<1>().GetValue(), 0xabcd);
    EXPECT_THAT(result, ElementsAreArray(
                            {0x1, 0x00, 0x00, 0x00, 0xcd, 0xab, 0x00, 0x00}));
  }
}

TEST(FunctionJitTest, TupleViewSmokeTest) {
  Package package("my_package");

  std::string ir_text = R"(
fn f(x: bits[1], y: bits[8]) -> (bits[1], bits[8], bits[16]) {
  literal.3: bits[16] = literal(value=43981)
  ret tuple.4: (bits[1], bits[8], bits[16]) = tuple(x, y, literal.3)
})";

  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  // Test using values.
  {
    Value x(UBits(0x1, 1));
    Value y(UBits(0x34, 8));
    Value ret =
        Value::Tuple(std::vector<Value>{x, y, Value(UBits(0xabcd, 16))});

    std::vector args{x, y};
    EXPECT_THAT(RunJitNoEvents(jit.get(), args), IsOkAndHolds(ret));
  }

  // Test using views.
  {
    int64_t x = 0x1;
    int64_t y = 0x34;
    uint8_t result[4] = {0};

    std::vector<uint8_t*> args{reinterpret_cast<uint8_t*>(&x),
                               reinterpret_cast<uint8_t*>(&y)};
    absl::Span<uint8_t> result_buffer(result);
    InterpreterEvents events;
    XLS_ASSERT_OK(jit->RunWithViews(args, result_buffer, &events));

    xls::TupleView<xls::BitsView<1>, xls::BitsView<8>, xls::BitsView<16>>
        result_view(result);
    EXPECT_EQ(result_view.Get<0>().GetValue(), 0x1);
    EXPECT_EQ(result_view.Get<1>().GetValue(), 0x34);
    EXPECT_EQ(result_view.Get<2>().GetValue(), 0xabcd);
    EXPECT_THAT(result, ElementsAreArray({0x1, 0x34, 0xcd, 0xab}));
  }
}

TEST(FunctionJitTest, MisalignedPointerCopied) {
  Package package("my_package");

  FunctionBuilder fb("test", &package);
  fb.Add(fb.Param("x", package.GetBitsType(256)),
         fb.Param("y", package.GetBitsType(256)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));
  Bits ret_bits =
      bits_ops::Concat({UBits(0, 256 - 65), UBits(1, 1), UBits(0, 64)});

  alignas(16) std::array<uint8_t, 1 + (256 / 8)> x_view{
      0xAB, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  alignas(16) std::array<uint8_t, 1 + (256 / 8)> y_view{
      0xAB, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  alignas(16) std::array<uint8_t, 1 + (256 / 8)> ret_view{};
  {
    InterpreterEvents events;
    EXPECT_THAT(jit->RunWithViews</*kForceZeroCopy=*/false>(
                    {x_view.data() + 1, y_view.data() + 1},
                    absl::MakeSpan(ret_view).subspan(1), &events),
                IsOk());
    EXPECT_EQ(Bits::FromBytes(absl::MakeSpan(ret_view).subspan(1), 256),
              ret_bits);
  }
}

TEST(FunctionJitDeathTest, MisalignedPointerCaught) {
#ifndef NDEBUG
  Package package("my_package");

  FunctionBuilder fb("test", &package);
  fb.Add(fb.Param("x", package.GetBitsType(256)),
         fb.Param("y", package.GetBitsType(256)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(function));

  alignas(16) std::array<uint8_t, 1 + (256 / 8)> x_view{
      0xAB, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  alignas(16) std::array<uint8_t, 1 + (256 / 8)> y_view{
      0xAB, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  alignas(16) std::array<uint8_t, 1 + (256 / 8)> ret_view{};
  ASSERT_DEATH(
      {
        InterpreterEvents events;
        auto unused = jit->RunWithViews</*kForceZeroCopy=*/true>(
            {x_view.data() + 1, y_view.data() + 1},
            absl::MakeSpan(ret_view).subspan(1), &events);
      },
      ".*does not have alignment of [0-9]+.*");
#else
  GTEST_SKIP() << "Checking only performed in dbg mode.";
#endif
}

// Check that expected_data matched output_data.
// Log values of expected_data, output_data, and whatever entries are in
// extra_data.
void CheckOutput(absl::Span<uint8_t const> expected_data,
                 absl::Span<uint8_t const> output_data,
                 std::initializer_list<
                     std::tuple<std::string_view, absl::Span<uint8_t const>>>
                     extra_data) {
  CHECK_EQ(expected_data.size(), output_data.size());
  if (XLS_VLOG_IS_ON(3)) {
    auto print_byte_vec = [](absl::Span<uint8_t const> bytes,
                             std::string_view name) {
      std::vector<std::string> strs;
      for (uint8_t byte : bytes) {
        strs.push_back(absl::StrFormat("%d", byte));
      }
      XLS_VLOG(3) << absl::StreamFormat("%s: %s", name,
                                        absl::StrJoin(strs, ", "));
    };
    for (const auto& [name, data] : extra_data) {
      print_byte_vec(data, name);
    }
    print_byte_vec(output_data, "output_data");
    print_byte_vec(expected_data, "expected_data");
  }

  for (int i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], expected_data[i])
        << std::hex << ": byte " << i << ": "
        << "0x" << static_cast<int>(output_data[i]) << " vs. "
        << "0x" << static_cast<int>(expected_data[i]);
  }
}

// Tests the packed jit interface with bits inputs/outputs.
// The dut performs min(x, y).
//
// type_proto must be a bits type.
void TestPackedBitsWithType(const TypeProto& type_proto) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Type * converted_type_proto,
                           package.GetTypeFromProto(type_proto));
  XLS_ASSERT_OK_AND_ASSIGN(BitsType * tpe, converted_type_proto->AsBits());

  // f(x, y) = min(x, y)
  FunctionBuilder b("f", &package);
  BValue x = b.Param("x", tpe);
  BValue y = b.Param("y", tpe);
  BValue x_lt_y = b.ULt(x, y);
  b.Select(x_lt_y, {y, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      llvm::DataLayout data_layout,
      OrcJit::CreateDataLayout(/*aot_specification=*/false));
  constexpr int64_t opt_level = 3;
  XLS_ASSERT_OK_AND_ASSIGN(auto orc_jit,
                           OrcJit::Create(opt_level, /*emit_object_code=*/false,
                                          /*observer=*/nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(JittedFunctionBase jit,
                           JittedFunctionBase::Build(function, *orc_jit));

  std::minstd_rand bitgen;
  Value lhs_value = RandomValue(tpe, bitgen);
  Value rhs_value = RandomValue(tpe, bitgen);
  std::vector<uint8_t> lhs = FlattenValue(lhs_value);
  std::vector<uint8_t> rhs = FlattenValue(rhs_value);
  std::vector<uint8_t> output(lhs.size());

  bool lhs_lt_rhs = bits_ops::ULessThan(lhs_value.bits(), rhs_value.bits());
  absl::Span<uint8_t const> expected = lhs_lt_rhs ? lhs : rhs;

  std::array<uint8_t*, 2> inputs = {lhs.data(), rhs.data()};
  std::array<uint8_t*, 1> outputs = {output.data()};

  InterpreterEvents events;
  JitRuntime runtime(data_layout);
  JitTempBuffer temp_buffer = jit.CreateTempBuffer();
  std::optional<int64_t> ret = jit.RunPackedJittedFunction(
      inputs.data(), outputs.data(), &temp_buffer, &events,
      /*user_data=*/nullptr, /*jit_runtime=*/&runtime,
      /*continuation_point=*/0);
  ASSERT_TRUE(ret.has_value());
  CheckOutput(/*expected_data=*/expected, /*output_data=*/output,
              /*extra_data=*/{{"lhs", lhs}, {"rhs", rhs}});
}

// Tests the packed jit interface.
// The DUT takes in two tuples x and y of the same type and produces an output
// tuple (x0, y1, x2, ...).
//
// type_proto must be a tuple type.
void TestPackedTupleWithType(const TypeProto& type_proto) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Type * converted_type_proto,
                           package.GetTypeFromProto(type_proto));
  XLS_ASSERT_OK_AND_ASSIGN(TupleType * tuple_type,
                           converted_type_proto->AsTuple());

  // Build DUT
  FunctionBuilder b("f", &package);
  BValue x = b.Param("x", tuple_type);
  BValue y = b.Param("y", tuple_type);
  std::vector<BValue> output_elements;
  output_elements.reserve(tuple_type->size());
  for (int64_t i = 0; i < tuple_type->size(); ++i) {
    if (i % 2 == 0) {
      // pull from x
      output_elements.push_back(b.TupleIndex(x, i));
    } else {
      // pull from y
      output_elements.push_back(b.TupleIndex(y, i));
    }
  }
  b.Tuple(output_elements);

  XLS_ASSERT_OK_AND_ASSIGN(Function * function, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      llvm::DataLayout data_layout,
      OrcJit::CreateDataLayout(/*aot_specification=*/false));
  constexpr int64_t opt_level = 3;
  XLS_ASSERT_OK_AND_ASSIGN(auto orc_jit,
                           OrcJit::Create(opt_level, /*emit_object_code=*/false,
                                          /*observer=*/nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(JittedFunctionBase jit,
                           JittedFunctionBase::Build(function, *orc_jit));

  std::minstd_rand bitgen;
  Value lhs_value = RandomValue(tuple_type, bitgen);
  Value rhs_value = RandomValue(tuple_type, bitgen);
  std::vector<uint8_t> lhs = FlattenValue(lhs_value);
  std::vector<uint8_t> rhs = FlattenValue(rhs_value);
  std::vector<uint8_t> output(lhs.size());
  std::vector<Value> expected_elements;
  expected_elements.reserve(lhs.size());
  // Build expected tuple value.
  for (int64_t i = 0; i < lhs_value.size(); ++i) {
    if (i % 2 == 0) {
      // pull from x
      expected_elements.push_back(lhs_value.element(i));
    } else {
      // pull from y
      expected_elements.push_back(rhs_value.element(i));
    }
  }
  std::vector<uint8_t> expected = FlattenValue(Value::Tuple(expected_elements));
  CHECK_EQ(expected.size(), output.size());

  // Run jit and check output.
  std::array<uint8_t*, 2> inputs = {lhs.data(), rhs.data()};
  std::array<uint8_t*, 1> outputs = {output.data()};

  InterpreterEvents events;
  JitRuntime runtime(data_layout);
  JitTempBuffer temp_buffer = jit.CreateTempBuffer();
  std::optional<int64_t> ret = jit.RunPackedJittedFunction(
      inputs.data(), outputs.data(), &temp_buffer, &events,
      /*user_data=*/nullptr, /*jit_runtime=*/&runtime,
      /*continuation_point=*/0);
  ASSERT_TRUE(ret.has_value());
  CheckOutput(/*expected_data=*/expected, /*output_data=*/output,
              /*extra_data=*/{{"lhs", lhs}, {"rhs", rhs}});
}

// Tests the packed jit interface.
// The DUT takes in an array, an index, and a replacement value, and does an
// array_update(). We then verify that the output array looks as expected.
//
// type_proto must be an array type.
void TestPackedArrayWithType(const TypeProto& type_proto) {
  std::minstd_rand bitgen;
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Type * converted_type_proto,
                           package.GetTypeFromProto(type_proto));
  XLS_ASSERT_OK_AND_ASSIGN(ArrayType * array_type,
                           converted_type_proto->AsArray());
  Type* element_type = array_type->element_type();
  int64_t num_elements = array_type->size();
  int64_t kIndexBitWidth = CeilOfLog2(num_elements);

  FunctionBuilder b("f", &package);
  BValue array =
      b.Param("array", package.GetArrayType(num_elements, element_type));
  BValue idx = b.Param("idx", package.GetBitsType(kIndexBitWidth));
  BValue new_value = b.Param("new_value", element_type);
  b.ArrayUpdate(array, new_value, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      llvm::DataLayout data_layout,
      OrcJit::CreateDataLayout(/*aot_specification=*/false));
  constexpr int64_t opt_level = 3;
  XLS_ASSERT_OK_AND_ASSIGN(auto orc_jit,
                           OrcJit::Create(opt_level, /*emit_object_code=*/false,
                                          /*observer=*/nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(JittedFunctionBase jit,
                           JittedFunctionBase::Build(function, *orc_jit));

  std::vector<Bits> bits_vector;
  for (int i = 0; i < num_elements; i++) {
    Value value = RandomValue(
        package.GetBitsType(element_type->GetFlatBitCount()), bitgen);
    bits_vector.push_back(value.bits());
  }
  std::vector<uint8_t> array_data =
      FlattenValue(Value(VectorToPackedBits(bits_vector)));
  std::vector<uint8_t> index_data;
  std::vector<uint8_t> output_data(array_data.size());

  Value replacement =
      RandomValue(package.GetBitsType(element_type->GetFlatBitCount()), bitgen);
  std::vector<uint8_t> replacement_data = FlattenValue(replacement);

  std::array<uint8_t*, 3> inputs = {array_data.data(),
                                    nullptr,  // fill in index_data later
                                    replacement_data.data()};
  std::array<uint8_t*, 1> outputs = {output_data.data()};
  for (int64_t index = 0; index < num_elements; ++index) {
    index_data = FlattenValue(Value(UBits(index, kIndexBitWidth)));
    inputs[1] = index_data.data();
    Bits old_bits_value = bits_vector[index];
    bits_vector[index] = replacement.bits();
    std::vector<uint8_t> expected_data =
        FlattenValue(Value(VectorToPackedBits(bits_vector)));
    InterpreterEvents events;
    JitRuntime runtime(data_layout);
    JitTempBuffer temp_buffer = jit.CreateTempBuffer();
    std::optional<int64_t> ret = jit.RunPackedJittedFunction(
        inputs.data(), outputs.data(), &temp_buffer, &events,
        /*user_data=*/nullptr, /*jit_runtime=*/&runtime,
        /*continuation_point=*/0);
    ASSERT_TRUE(ret.has_value());
    bits_vector[index] = old_bits_value;

    CheckOutput(/*expected_data=*/expected_data, /*output_data=*/output_data,
                /*extra_data=*/{{"array_data", array_data}});
  }
}

FUZZ_TEST(FunctionJitTest, TestPackedBitsWithType)
    .WithDomains(BitsTypeDomain(/*max_bit_count=*/5000));

FUZZ_TEST(FunctionJitTest, TestPackedTupleWithType)
    .WithDomains(TypeDomainWithSizeInRange(
        /*min_size=*/1,
        /*max_size=*/65536,
        TupleTypeDomain(TypeDomain(/*max_bit_count=*/132,
                                   /*max_elements=*/68),
                        /*max_elements=*/1030)));

FUZZ_TEST(FunctionJitTest, TestPackedArrayWithType)
    .WithDomains(TypeDomainWithSizeInRange(
        /*min_size=*/1,
        /*max_size=*/65536,
        ArrayTypeDomain(TypeDomain(/*max_bit_count=*/132,
                                   /*max_elements=*/68),
                        /*max_elements=*/1030)));

}  // namespace
}  // namespace xls
