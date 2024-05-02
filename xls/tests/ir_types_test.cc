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

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

// Tests of non-trivial IR types.
class IrTypeTests : public IrTestBase {
 protected:
  // Creates a one-dimensional array of Bits values of the given width.
  Value Array1D(absl::Span<const int64_t> values, int64_t width) {
    std::vector<Value> elements;
    for (int64_t value : values) {
      elements.push_back(Value(UBits(value, width)));
    }
    return Value::ArrayOrDie(elements);
  }

  // Creates a two-dimensional array of Bits values of the given width.
  Value Array2D(absl::Span<const absl::Span<const int64_t>> values,
                int64_t width) {
    std::vector<Value> rows;
    for (auto row : values) {
      rows.push_back(Array1D(row, width));
    }
    return Value::ArrayOrDie(rows);
  }
};

TEST_F(IrTypeTests, TupleConstant) {
  // Index and return an element from a tuple constant.
  std::string text = R"(
package TupleConstant

top fn main() -> bits[32] {
  literal.1: (bits[4], bits[32], bits[42]) = literal(value=(1, 123, 7))
  ret tuple_index.2: bits[32] = tuple_index(literal.1, index=1)
}
)";

  RunAndExpectEq({}, 123, text);
}

TEST_F(IrTypeTests, NestedTupleConstant) {
  // Index two elements from a nested tuple constant, add and return them.
  std::string text = R"(
package NestedTupleConstant

top fn main() -> bits[16] {
  literal.1: (bits[4], (bits[16], bits[1]), (bits[42], bits[16])) = literal(value=(1, (123,0), (7, 33)))
  tuple_index.2: (bits[16], bits[1]) = tuple_index(literal.1, index=1)
  tuple_index.3: (bits[42], bits[16]) = tuple_index(literal.1, index=2)
  tuple_index.4: bits[16] = tuple_index(tuple_index.2, index=0)
  tuple_index.5: bits[16] = tuple_index(tuple_index.3, index=1)
  ret add.6: bits[16] = add(tuple_index.4, tuple_index.5)
}
)";

  RunAndExpectEq({}, 156, text);
}

TEST_F(IrTypeTests, TuplesAsInputsAndOutputs) {
  // Take a nested tuple as input, add two elements from it, and return a newly
  // constructed tuple.
  std::string text = R"(
package TuplesAsInputsAndOutputs

top fn main(x: (bits[16], (bits[16], bits[1]))) -> (bits[16], bits[1]) {
  // Extract the two bits[16] values from the input 'x'.
  tuple_index.2: bits[16] = tuple_index(x, index=0)
  tuple_index.3: (bits[16], bits[1]) = tuple_index(x, index=1)
  tuple_index.4: bits[16] = tuple_index(tuple_index.3, index=0)
  add.5: bits[16] = add(tuple_index.2, tuple_index.4)

  // Return the sum along with the bits[1] from 'x'.
  tuple_index.6: bits[1] = tuple_index(tuple_index.3, index=1)
  ret tuple.7: (bits[16], bits[1]) = tuple(add.5, tuple_index.6)
}
)";

  Value input =
      Value::Tuple({Value(UBits(33, 16)),
                    Value::Tuple({Value(UBits(44, 16)), Value(UBits(1, 1))})});
  Value expected = Value::Tuple({Value(UBits(77, 16)), Value(UBits(1, 1))});
  RunAndExpectEq({{"x", input}}, expected, text);
}

TEST_F(IrTypeTests, ArrayConstant) {
  // Return an element from a constant array indexed by an argument.
  std::string text = R"(
package ArrayConstant

top fn main(x: bits[32]) -> bits[13] {
  literal.1: bits[13][3] = literal(value=[42, 44, 77])
  ret array_index.2: bits[13] = array_index(literal.1, indices=[x])
}
)";

  RunAndExpectEq({{"x", 0}}, 42, text);
  RunAndExpectEq({{"x", 1}}, 44, text);
  RunAndExpectEq({{"x", 2}}, 77, text);
}

TEST_F(IrTypeTests, NestedArrayConstant) {
  // Return an element from a 2d constant array indexed by arguments.
  std::string text = R"(
package NestedArrayConstant

top fn main(x: bits[32], y: bits[32]) -> bits[13] {
  literal.1: bits[13][3][2] = literal(value=[[42, 44, 77], [11, 22, 33]])
  array_index.2: bits[13][3] = array_index(literal.1, indices=[x])
  ret array_index.3: bits[13] = array_index(array_index.2, indices=[y])
}
)";

  RunAndExpectEq({{"x", 0}, {"y", 0}}, 42, text);
  RunAndExpectEq({{"x", 1}, {"y", 0}}, 11, text);
  RunAndExpectEq({{"x", 1}, {"y", 2}}, 33, text);
}

TEST_F(IrTypeTests, ArraysAsInputsAndOutputs) {
  // Pass in a 2d array and indices, return an array constructed from the
  // indexed elements.
  std::string text = R"(
package ArraysAsInputsAndOutputs

top fn main(in: bits[16][3][2], x: bits[16], y0: bits[16], y1:bits[16]) -> bits[16][2] {
  array_index.2: bits[16][3] = array_index(in, indices=[x])
  array_index.3: bits[16] = array_index(array_index.2, indices=[y0])
  array_index.4: bits[16] = array_index(array_index.2, indices=[y1])
  ret array.5: bits[16][2] = array(array_index.3, array_index.4)
}
)";

  auto v16 = [](int64_t v) { return Value(UBits(v, 16)); };

  Value input = Array2D({{101, 102, 103}, {201, 202, 203}}, 16);
  RunAndExpectEq({{"in", input}, {"x", v16(0)}, {"y0", v16(1)}, {"y1", v16(2)}},
                 Array1D({102, 103}, 16), text);

  RunAndExpectEq({{"in", input}, {"x", v16(1)}, {"y0", v16(2)}, {"y1", v16(0)}},
                 Array1D({203, 201}, 16), text);
}

TEST_F(IrTypeTests, PassThrough2DArray) {
  std::string text = R"(
package PassThrough2DArray

top fn main(in: bits[16][3][2]) -> bits[16][3][2] {
  ret param.1: bits[16][3][2] = param(name=in)
}
)";

  Value input = Array2D({{101, 102, 103}, {201, 202, 203}}, 16);
  RunAndExpectEq({{"in", input}}, input, text);
}

TEST_F(IrTypeTests, ArrayOfTuplesConstant) {
  // Index from an constant array of tuple and return an array of tuple
  // constructed from the elements.
  std::string text = R"(
package ArrayOfTuplesConstant

top fn main(x: bits[32]) -> (bits[17], bits[13])[3] {
  literal.1: (bits[13], bits[17])[2] = literal(value=[(12, 34), (56, 78)])
  array_index.2: (bits[13], bits[17]) = array_index(literal.1, indices=[x])
  tuple_index.3: bits[13] = tuple_index(array_index.2, index=0)
  tuple_index.4: bits[17] = tuple_index(array_index.2, index=1)
  tuple.5: (bits[17], bits[13]) = tuple(tuple_index.4, tuple_index.3)
  ret array.6: (bits[17], bits[13])[3] = array(tuple.5, tuple.5, tuple.5)
}
)";

  {
    Value element = Value::Tuple({Value(UBits(34, 17)), Value(UBits(12, 13))});
    Value expected = Value::ArrayOrDie({element, element, element});
    RunAndExpectEq({{"x", Value(UBits(0, 32))}}, expected, text);
  }

  {
    Value element = Value::Tuple({Value(UBits(78, 17)), Value(UBits(56, 13))});
    Value expected = Value::ArrayOrDie({element, element, element});
    RunAndExpectEq({{"x", Value(UBits(1, 32))}}, expected, text);
  }
}

TEST_F(IrTypeTests, TuplesOfArraysConstant) {
  // Index from an constant tuple of arrays and return the sum of two elements.
  std::string text = R"(
package TuplesOfArraysConstant

top fn main(x: bits[32], y: bits[32]) -> bits[13] {
  literal.1: (bits[13][3], bits[13][2]) = literal(value=([1, 2, 3], [55, 66]))
  tuple_index.2: bits[13][3] = tuple_index(literal.1, index=0)
  tuple_index.3: bits[13][2] = tuple_index(literal.1, index=1)
  array_index.4: bits[13] = array_index(tuple_index.2, indices=[x])
  array_index.5: bits[13] = array_index(tuple_index.3, indices=[y])
  ret add.6: bits[13] = add(array_index.4, array_index.5)
}
)";

  RunAndExpectEq({{"x", 0}, {"y", 1}}, 67, text);
  RunAndExpectEq({{"x", 2}, {"y", 0}}, 58, text);
}

}  // namespace
}  // namespace xls
