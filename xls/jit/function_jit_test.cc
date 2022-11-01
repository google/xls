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

#include <cstdio>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/ir_evaluator_test_base.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/function_builder.h"
#include "re2/re2.h"

namespace xls {
namespace {

using status_testing::IsOk;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;

// TODO(https://github.com/google/xls/issues/506): 2021-10-12 Replace the empty
// events returned by the JIT evaluator with a entry point that includes the
// collected events (once they are supported by the JIT).
INSTANTIATE_TEST_SUITE_P(
    FunctionJitTest, IrEvaluatorTestBase,
    testing::Values(IrEvaluatorTestParam(
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

// This test verifies that a compiled JIT function can be re-used.
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

// Very basic smoke test for packed types.
TEST(FunctionJitTest, PackedSmoke) {
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
  uint8_t output_data;
  PackedBitsView<8> input(input_data, 0);
  PackedBitsView<8> output(&output_data, 0);
  XLS_ASSERT_OK(jit->RunWithPackedViews(input, output));
  EXPECT_EQ(output_data, 0x5a);
}

TEST(FunctionJitTest, PackedSmokeWide) {
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
  uint8_t input_data[] = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa};
  uint8_t output_data[10];
  PackedBitsView<80> input(input_data, 0);
  PackedBitsView<80> output(output_data, 0);
  XLS_ASSERT_OK(jit->RunWithPackedViews(input, output));
  EXPECT_THAT(output_data, testing::ElementsAre(0x2, 0x2, 0x3, 0x4, 0x5, 0x6,
                                                0x7, 0x8, 0x9, 0xa));
}

// Tests PackedBitView<X> input/output handling.
template <int64_t kBitWidth>
absl::Status TestPackedBits(std::minstd_rand& bitgen) {
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
  Value v = RandomValue(package.GetBitsType(kBitWidth), &bitgen);
  Bits a(v.bits());
  v = RandomValue(package.GetBitsType(kBitWidth), &bitgen);
  Bits b(v.bits());
  Bits expected = bits_ops::Add(a, b);
  int64_t byte_width = jit->GetPackedReturnTypeSize();
  auto output_data = std::make_unique<uint8_t[]>(byte_width);
  bzero(output_data.get(), byte_width);

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

// Utility struct to hold different representations of the same data together.
template <typename ViewT>
struct TestData {
  explicit TestData(const Value& value_in)
      : value(value_in), bytes(FlattenValue(value)), view(bytes.data(), 0) {}
  Value value;
  std::vector<uint8_t> bytes;
  ViewT view;

  static std::vector<uint8_t> FlattenValue(const Value& value) {
    BitsRope rope(value.GetFlatBitCount());
    FlattenValue(value, rope);
    std::vector<uint8_t> bytes = rope.Build().ToBytes();
    return bytes;
  }

  static void FlattenValue(const Value& value, BitsRope& rope) {
    if (value.IsBits()) {
      rope.push_back(value.bits());
    } else if (value.IsArray()) {
      for (const Value& element : value.elements()) {
        FlattenValue(element, rope);
      }
    } else if (value.IsTuple()) {
      // Tuple elements are declared MSelement to LSelement, so we need to pack
      // them in "reverse" order, so the LSelement is at the LSb.
      for (int i = value.elements().size() - 1; i >= 0; i--) {
        FlattenValue(value.elements()[i], rope);
      }
    }
  }
};

// Tests PackedArrayView input/output from the JIT. Takes in an array, an index,
// and a replacement value, and does an array_update(). We then verify that the
// output array looks like expected.
template <int64_t kBitWidth, int64_t kNumElements>
absl::Status TestSimpleArray(std::minstd_rand& bitgen) {
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
    Value value = RandomValue(package.GetBitsType(kBitWidth), &bitgen);
    bits_vector.push_back(value.bits());
  }
  TestData<ArrayT> array_data(Value(VectorToPackedBits(bits_vector)));

  int index = absl::Uniform(bitgen, 0, kNumElements);
  TestData<PackedBitsView<kBitWidth>> index_data(
      Value(UBits(index, kBitWidth)));

  Value value = RandomValue(package.GetBitsType(kBitWidth), &bitgen);
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
absl::Status TestTuples(std::minstd_rand& bitgen) {
  Package package("my_package");
  TupleType* tuple_type = TupleT::GetFullType(&package);

  Type* replacement_type = tuple_type->element_type(kReplacementIndex);
  XLS_ASSIGN_OR_RETURN(
      Function * function,
      CreateTupleFunction(&package, tuple_type, kReplacementIndex));
  XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));

  Value input_tuple = RandomValue(tuple_type, &bitgen);
  TestData<TupleT> input_tuple_data(input_tuple);
  Value replacement = RandomValue(replacement_type, &bitgen);
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
                       testing::HasSubstr("the assertion error message")));
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

  FunctionBuilder top_builder("top", &p);
  auto y = top_builder.Param("y", p.GetBitsType(5));

  std::vector<BValue> args = {y};
  top_builder.Invoke(args, fun);

  XLS_ASSERT_OK_AND_ASSIGN(Function * top, top_builder.Build());

  XLS_ASSERT_OK_AND_ASSIGN(auto jit, FunctionJit::Create(top));

  std::vector<Value> ok_args = {Value(UBits(6, 5))};
  EXPECT_THAT(RunJitNoEvents(jit.get(), ok_args),
              IsOkAndHolds(Value(UBits(7, 5))));

  std::vector<Value> fail_args = {Value(UBits(8, 5))};
  EXPECT_THAT(RunJitNoEvents(jit.get(), fail_args),
              StatusIs(absl::StatusCode::kAborted,
                       testing::HasSubstr("x is more than 7")));
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
                       testing::HasSubstr("first assertion error message")));

  std::vector<Value> fail2_args = {Value::Token(), Value(UBits(1, 1)),
                                   Value(UBits(0, 1))};

  EXPECT_THAT(RunJitNoEvents(jit.get(), fail2_args),
              StatusIs(absl::StatusCode::kAborted,
                       testing::HasSubstr("second assertion error message")));

  std::vector<Value> failboth_args = {Value::Token(), Value(UBits(0, 1)),
                                      Value(UBits(0, 1))};

  // The token-plumbing ensures that the first assertion is checked first,
  // so test that it is reported properly.
  EXPECT_THAT(RunJitNoEvents(jit.get(), failboth_args),
              StatusIs(absl::StatusCode::kAborted,
                       testing::HasSubstr("first assertion error message")));
}

TEST(FunctionJitTest, TokenCompareError) {
  Package p("token_eq");
  FunctionBuilder b("fun", &p);
  auto p0 = b.Param("tkn", p.GetTokenType());

  b.Eq(p0, p0);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(FunctionJit::Create(f),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("Tokens are incomparable")));
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
                       testing::HasSubstr("Tokens are incomparable")));
}
}  // namespace
}  // namespace xls
