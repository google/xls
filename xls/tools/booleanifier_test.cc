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
#include "xls/tools/booleanifier.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/jit/function_jit.h"

namespace xls {
namespace {

using VB = ValueBuilder;

// Handles parsing/initialization boilerplate.
struct FunctionData {
  std::unique_ptr<Package> package;
  Function* source;
  Function* boolified;
};

class BooleanifierTest : public IrTestBase {
 protected:
  absl::StatusOr<FunctionData> GetFunctionData(const std::string& ir_text,
                                               const std::string& fn_name) {
    XLS_ASSIGN_OR_RETURN(auto package, ParsePackage(ir_text));
    XLS_ASSIGN_OR_RETURN(Function * fancy, package->GetFunction(fn_name));
    XLS_ASSIGN_OR_RETURN(Function * basic, Booleanifier::Booleanify(fancy));

    return FunctionData{
        .package = std::move(package), .source = fancy, .boolified = basic};
  }

  absl::StatusOr<FunctionData> GetFunctionDataFromFile(
      const std::string& ir_path, const std::string& fn_name) {
    XLS_ASSIGN_OR_RETURN(std::string runfile_path, GetXlsRunfilePath(ir_path));
    XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(runfile_path));
    return GetFunctionData(ir_text, fn_name);
  }
};

// This test verifies that the CRC32 example can be correctly boolified, as a
// decently-broad example.
TEST_F(BooleanifierTest, Crc32) {
  const std::string kIrPath = "xls/examples/crc32/crc32.opt.ir";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd,
                           GetFunctionDataFromFile(kIrPath, "__crc32__main"));
  // CRC32 main takes an 8b message.
  for (int i = 0; i < 256; i++) {
    std::vector<Value> inputs({Value(UBits(i, 8))});
    XLS_ASSERT_OK_AND_ASSIGN(
        Value fancy_value,
        DropInterpreterEvents(InterpretFunction(fd.source, inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(
        Value basic_value,
        DropInterpreterEvents(InterpretFunction(fd.boolified, inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, Crc32_Jit) {
  const std::string kIrPath = "xls/examples/crc32/crc32.opt.ir";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd,
                           GetFunctionDataFromFile(kIrPath, "__crc32__main"));
  // CRC32 main takes an 8b message.
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  for (int i = 0; i < 256; i++) {
    std::vector<Value> inputs({Value(UBits(i, 8))});
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, ShuffleMarshalsTuples) {
  const std::string kIrText = R"(
package p

fn main(a: (bits[2], bits[3], bits[4])) -> (bits[4], bits[3], bits[2]) {
  a_0: bits[2] = tuple_index(a, index=0)
  a_1: bits[3] = tuple_index(a, index=1)
  a_2: bits[4] = tuple_index(a, index=2)
  c: (bits[4], bits[2], bits[3]) = tuple(a_2, a_0, a_1)
  x_0: bits[3] = tuple_index(c, index=2)
  x_1: bits[2] = tuple_index(c, index=1)
  x_2: bits[4] = tuple_index(c, index=0)
  ret x: (bits[4], bits[3], bits[2]) = tuple(x_2, x_0, x_1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));
  // Don't cover all 4B samples in the space; just enough to see _some_ values
  // in all elements.
  for (int i = 0; i < 512; i++) {
    std::vector<Value> inputs(1);
    Value a_0(UBits(i & 0x3, 2));
    Value a_1(UBits((i >> 2) & 0x7, 3));
    Value a_2(UBits((i >> 5) & 0x1F, 4));
    inputs[0] = Value::Tuple({a_0, a_1, a_2});

    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value) << testing::PrintToString(inputs);
  }
}
// This test verifies that the Boolifier can properly handle extracting from
// and packing into tuples.
TEST_F(BooleanifierTest, MarshalsTuples) {
  const std::string kIrText = R"(
package p

fn main(a: (bits[2], bits[3], bits[4]), b: (bits[2], bits[3], bits[4])) -> (bits[4], bits[3], bits[2]) {
  a_0: bits[2] = tuple_index(a, index=0)
  a_1: bits[3] = tuple_index(a, index=1)
  a_2: bits[4] = tuple_index(a, index=2)
  b_0: bits[2] = tuple_index(b, index=0)
  b_1: bits[3] = tuple_index(b, index=1)
  b_2: bits[4] = tuple_index(b, index=2)
  c_0: bits[2] = add(a_0, b_0)
  c_1: bits[3] = add(a_1, b_1)
  c_2: bits[4] = add(a_2, b_2)
  c: (bits[2], bits[3], bits[4]) = tuple(c_0, c_1, c_2)
  x_0: bits[4] = tuple_index(c, index=2)
  x_1: bits[3] = tuple_index(c, index=1)
  x_2: bits[2] = tuple_index(c, index=0)
  ret x: (bits[4], bits[3], bits[2]) = tuple(x_0, x_1, x_2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));
  // Don't cover all 4B samples in the space; just enough to see _some_ values
  // in all elements.
  for (int i = 0; i < 512; i++) {
    std::vector<Value> inputs(2);
    Value a_0(UBits(i & 0x3, 2));
    Value a_1(UBits((i >> 2) & 0x7, 3));
    Value a_2(UBits((i >> 5) & 0x1F, 4));
    inputs[0] = Value::Tuple({a_0, a_1, a_2});

    for (int j = 0; j < 512; j++) {
      Value b_0(UBits(j & 0x3, 2));
      Value b_1(UBits((j >> 2) & 0x7, 3));
      Value b_2(UBits((j >> 5) & 0x1F, 4));
      inputs[1] = Value::Tuple({b_0, b_1, b_2});

      XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                               DropInterpreterEvents(fancy_jit->Run(inputs)));
      XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                               DropInterpreterEvents(basic_jit->Run(inputs)));
      ASSERT_EQ(fancy_value, basic_value) << testing::PrintToString(inputs);
    }
  }
}

// This test verifies that the Boolifier can properly handle extracting from and
// packing into arrays.
TEST_F(BooleanifierTest, MarshalsArrays) {
  const std::string kIrText = R"(
package p

fn main(a: bits[4][4], i: bits[2], j: bits[2]) -> bits[4][2] {
  a_i: bits[4] = array_index(a, indices=[i])
  a_j: bits[4] = array_index(a, indices=[j])
  ret x: bits[4][2] = array(a_i, a_j)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBitsArray({4, 7, 2, 1}, 4));

  std::vector<Value> inputs(3);
  inputs[0] = a;
  for (int i = 0; i < 4; ++i) {
    inputs[1] = Value(UBits(i, 2));
    for (int j = 0; j < 4; ++j) {
      inputs[2] = Value(UBits(j, 2));

      XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                               DropInterpreterEvents(fancy_jit->Run(inputs)));
      XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                               DropInterpreterEvents(basic_jit->Run(inputs)));
      ASSERT_EQ(fancy_value, basic_value);
    }
  }
}

// This test verifies that the Boolifier can properly handle extracting from and
// packing into arrays.
TEST_F(BooleanifierTest, ArrayUpdate) {
  const std::string kIrText = R"(
package p

fn main(a: bits[4][4], i: bits[2], value: bits[4]) -> bits[4][4] {
  ret x: bits[4][4] = array_update(a, value, indices=[i])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBitsArray({4, 7, 2, 1}, 4));

  std::vector<Value> inputs(3);
  inputs[0] = a;
  for (int i = 0; i < 4; ++i) {
    inputs[1] = Value(UBits(i, 2));
    for (int value = 0; value < 16; ++value) {
      inputs[2] = Value(UBits(value, 4));

      XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                               DropInterpreterEvents(fancy_jit->Run(inputs)));
      XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                               DropInterpreterEvents(basic_jit->Run(inputs)));
      ASSERT_EQ(fancy_value, basic_value);
    }
  }
}

TEST_F(BooleanifierTest, MultidimArrayUpdate) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], i: bits[2][2], value: bits[8]) -> bits[8][4][4] {
  literal.1: bits[1] = literal(value=0)
  literal.2: bits[1] = literal(value=1)
  array_index.3: bits[2] = array_index(i, indices=[literal.1])
  array_index.4: bits[2] = array_index(i, indices=[literal.2])
  ret x: bits[8][4][4] = array_update(a, value, indices=[array_index.3, array_index.4])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(3);
  inputs[0] = a;
  Value replacement(UBits(0xff, 8));
  inputs[2] = replacement;
  for (uint64_t i = 0; i < 16; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(inputs[1], Value::UBitsArray({i / 4, i % 4}, 2));
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, MultidimArrayUpdateUsingLiteralIndices) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], value: bits[8]) -> bits[8][4][4] {
  literal.1: bits[2] = literal(value=2)
  literal.2: bits[2] = literal(value=3)
  ret x: bits[8][4][4] = array_update(a, value, indices=[literal.1, literal.2])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(2);
  inputs[0] = a;
  Value replacement(UBits(0xff, 8));
  inputs[1] = replacement;
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

TEST_F(BooleanifierTest, MultidimArrayPartialUpdateUsingLiteralIndex) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], value: bits[8][4]) -> bits[8][4][4] {
  literal.1: bits[2] = literal(value=2)
  ret x: bits[8][4][4] = array_update(a, value, indices=[literal.1])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(2);
  inputs[0] = a;
  XLS_ASSERT_OK_AND_ASSIGN(inputs[1],
                           Value::UBitsArray({0xc0, 0xd0, 0xe0, 0xf0}, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

TEST_F(BooleanifierTest, MultidimArrayUpdateUsingVariableAndLiteralIndices) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], i: bits[2], value: bits[8]) -> bits[8][4][4] {
  literal.1: bits[2] = literal(value=2)
  ret x: bits[8][4][4] = array_update(a, value, indices=[i, literal.1])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(3);
  inputs[0] = a;
  Value replacement(UBits(0xff, 8));
  inputs[2] = replacement;
  for (uint64_t i = 0; i < 4; ++i) {
    Value index(UBits(i, 2));
    inputs[1] = index;
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, MultidimArrayUpdateUsingLiteralAndVariableIndices) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], i: bits[2], value: bits[8]) -> bits[8][4][4] {
  literal.1: bits[2] = literal(value=2)
  ret x: bits[8][4][4] = array_update(a, value, indices=[literal.1, i])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(3);
  inputs[0] = a;
  Value replacement(UBits(0xff, 8));
  inputs[2] = replacement;
  for (uint64_t i = 0; i < 4; ++i) {
    Value index(UBits(i, 2));
    inputs[1] = index;
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, MultidimArrayPartialIndexUpdate) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], i: bits[2], value: bits[8][4]) -> bits[8][4][4] {
  ret x: bits[8][4][4] = array_update(a, value, indices=[i])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(3);
  inputs[0] = a;
  XLS_ASSERT_OK_AND_ASSIGN(inputs[2],
                           Value::UBitsArray({0xc0, 0xd0, 0xe0, 0xf0}, 8));
  for (uint64_t i = 0; i < 4; ++i) {
    Value index(UBits(i, 2));
    inputs[1] = index;
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, MultidimArrayNullIndexUpdate) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], value: bits[8][4][4]) -> bits[8][4][4] {
  ret x: bits[8][4][4] = array_update(a, value, indices=[])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  XLS_ASSERT_OK_AND_ASSIGN(Value b,
                           Value::UBits2DArray({{0xf0, 0xf1, 0xf2, 0xf3},
                                                {0xf4, 0xf5, 0xf6, 0xf7},
                                                {0xf8, 0xf9, 0xfa, 0xfb},
                                                {0xfc, 0xfd, 0xfe, 0xff}},
                                               8));

  std::vector<Value> inputs(2);
  inputs[0] = a;
  inputs[1] = b;
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

TEST_F(BooleanifierTest, MultidimArrayOOBIndexUpdateUsingLiteralIndices) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], value: bits[8]) -> bits[8][4][4] {
  literal.1: bits[3] = literal(value=5)
  literal.2: bits[3] = literal(value=0)
  ret x: bits[8][4][4] = array_update(a, value, indices=[literal.1, literal.2])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(2);
  inputs[0] = a;
  inputs[1] = Value(UBits(0xff, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

TEST_F(BooleanifierTest, MultidimArrayOOBIndexUpdateUsingLiteralIndices2) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], value: bits[8]) -> bits[8][4][4] {
  literal.1: bits[3] = literal(value=0)
  literal.2: bits[3] = literal(value=5)
  ret x: bits[8][4][4] = array_update(a, value, indices=[literal.1, literal.2])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(2);
  inputs[0] = a;
  inputs[1] = Value(UBits(0xff, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

TEST_F(BooleanifierTest, MultidimArrayOOBIndexUpdateUsingVariableIndices) {
  const std::string kIrText = R"(
package p

fn main(a: bits[8][4][4], i: bits[3][2], value: bits[8]) -> bits[8][4][4] {
  literal.1: bits[1] = literal(value=0)
  literal.2: bits[1] = literal(value=1)
  array_index.3: bits[3] = array_index(i, indices=[literal.1])
  array_index.4: bits[3] = array_index(i, indices=[literal.2])
  ret x: bits[8][4][4] = array_update(a, value, indices=[array_index.3, array_index.4])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        8));

  std::vector<Value> inputs(3);
  inputs[0] = a;
  Value replacement(UBits(0xff, 8));
  inputs[2] = replacement;

  // [5, 0]
  {
    XLS_ASSERT_OK_AND_ASSIGN(inputs[1], Value::UBitsArray({5, 0}, 3));
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
  // [0, 5]
  {
    XLS_ASSERT_OK_AND_ASSIGN(inputs[1], Value::UBitsArray({0, 5}, 3));
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, BooleanFunctionName) {
  auto p = CreatePackage();
  FunctionBuilder fb("foo", p.get());
  fb.Param("x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Function * default_name_f,
                           Booleanifier::Booleanify(f));
  EXPECT_EQ(default_name_f->name(), "foo_boolean");

  XLS_ASSERT_OK_AND_ASSIGN(Function * given_name_f,
                           Booleanifier::Booleanify(f, "baz"));
  EXPECT_EQ(given_name_f->name(), "baz");
}

TEST_F(BooleanifierTest, HandlesArrayEmptyIndex) {
  const std::string kIrText = R"(
package p

fn main(a: bits[4][4][4]) -> bits[4][4][4] {
  ret x: bits[4][4][4] = array_index(a, indices=[])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        4));

  std::vector<Value> inputs = {a};
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

TEST_F(BooleanifierTest, HandlesMultidimArrayIndex) {
  const std::string kIrText = R"(
package p

fn main(a: bits[4][4][4], i: bits[2][2]) -> bits[4] {
  literal.1: bits[1] = literal(value=0)
  literal.2: bits[1] = literal(value=1)
  array_index.3: bits[2] = array_index(i, indices=[literal.1])
  array_index.4: bits[2] = array_index(i, indices=[literal.2])
  ret x: bits[4] = array_index(a, indices=[array_index.3, array_index.4])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        4));

  std::vector<Value> inputs(2);
  inputs[0] = a;
  for (uint64_t i = 0; i < 16; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(inputs[1], Value::UBitsArray({i / 4, i % 4}, 2));
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, HandlesMultidimArrayPartialIndex) {
  const std::string kIrText = R"(
package p

fn main(a: bits[4][4][4], i: bits[2]) -> bits[4][4] {
  ret x: bits[4][4] = array_index(a, indices=[i])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        4));

  std::vector<Value> inputs(2);
  inputs[0] = a;
  for (uint64_t i = 0; i < 4; ++i) {
    inputs[1] = Value(UBits(i, 2));
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
}

TEST_F(BooleanifierTest, HandleArraySlice) {
  static constexpr int64_t kArraySize = 32;
  static constexpr int64_t kStartBits = 6;
  static constexpr int64_t kResultWidth = 3;
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArraySlice(
      fb.Param("input", p->GetArrayType(kArraySize, p->GetBitsType(8))),
      fb.Param("start", p->GetBitsType(kStartBits)), kResultWidth);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fancy, fb.Build());
  auto start = absl::Now();
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * basic,
      Booleanifier::Booleanify(fancy, absl::StrCat(TestName(), "_bool")));
  auto post_boolify = absl::Now();
  FunctionData fd{.package = std::move(p), .source = fancy, .boolified = basic};

  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  std::vector<Value> array_elements;
  array_elements.reserve(kArraySize);
  for (int64_t i = 0; i < kArraySize; i++) {
    array_elements.push_back(Value(UBits(i, 8)));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           VB::ArrayV(array_elements).Build());
  for (int64_t i = 0; i < (1 << kStartBits); ++i) {
    std::array<Value, 2> inputs{array, Value(UBits(i, kStartBits))};
    XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                             DropInterpreterEvents(fancy_jit->Run(inputs)));
    XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                             DropInterpreterEvents(basic_jit->Run(inputs)));
    ASSERT_EQ(fancy_value, basic_value);
  }
  auto finish = absl::Now();
  RecordProperty("booleanify_time",
                 testing::PrintToString(post_boolify - start));
  RecordProperty(
      "execute_time_avg",
      testing::PrintToString((finish - post_boolify) / (1 << kStartBits)));
}

TEST_F(BooleanifierTest, HandlesMultidimArrayLiteralIndex) {
  const std::string kIrText = R"(
package p

fn main(a: bits[4][4][4]) -> bits[4] {
  literal.1: bits[2] = literal(value=2)
  literal.2: bits[2] = literal(value=3)
  ret x: bits[4] = array_index(a, indices=[literal.1, literal.2])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        4));

  std::vector<Value> inputs = {a};
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

TEST_F(BooleanifierTest, HandlesMultidimArrayPartialLiteralIndex) {
  const std::string kIrText = R"(
package p

fn main(a: bits[4][4][4]) -> bits[4][4] {
  literal.1: bits[2] = literal(value=2)
  ret x: bits[4][4] = array_index(a, indices=[literal.1])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(FunctionData fd, GetFunctionData(kIrText, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(auto fancy_jit, FunctionJit::Create(fd.source));
  XLS_ASSERT_OK_AND_ASSIGN(auto basic_jit, FunctionJit::Create(fd.boolified));

  XLS_ASSERT_OK_AND_ASSIGN(Value a, Value::UBits2DArray({{0x0, 0x1, 0x2, 0x3},
                                                         {0x4, 0x5, 0x6, 0x7},
                                                         {0x8, 0x9, 0xa, 0xb},
                                                         {0xc, 0xd, 0xe, 0xf}},
                                                        4));

  std::vector<Value> inputs = {a};
  XLS_ASSERT_OK_AND_ASSIGN(Value fancy_value,
                           DropInterpreterEvents(fancy_jit->Run(inputs)));
  XLS_ASSERT_OK_AND_ASSIGN(Value basic_value,
                           DropInterpreterEvents(basic_jit->Run(inputs)));
  ASSERT_EQ(fancy_value, basic_value);
}

}  // namespace
}  // namespace xls
