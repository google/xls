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

#include "xls/ir/random_value.h"

namespace xls {

Value RandomValue(Type* type, std::minstd_rand* engine) {
  if (type->IsTuple()) {
    TupleType* tuple_type = type->AsTupleOrDie();
    std::vector<Value> elements;
    for (int64_t i = 0; i < tuple_type->size(); ++i) {
      elements.push_back(RandomValue(tuple_type->element_type(i), engine));
    }
    return Value::Tuple(elements);
  }
  if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    std::vector<Value> elements;
    for (int64_t i = 0; i < array_type->size(); ++i) {
      elements.push_back(RandomValue(array_type->element_type(), engine));
    }
    return Value::Array(elements).value();
  }
  if (type->IsToken()) {
    return Value::Token();
  }
  int64_t bit_count = type->AsBitsOrDie()->bit_count();
  std::vector<uint8_t> bytes;
  std::uniform_int_distribution<uint8_t> generator(0, 255);
  for (int64_t i = 0; i < bit_count; i += 8) {
    bytes.push_back(generator(*engine));
  }
  return Value(Bits::FromBytes(bytes, bit_count));
}

std::vector<Value> RandomFunctionArguments(Function* f,
                                           std::minstd_rand* engine) {
  std::vector<Value> inputs;
  for (Param* param : f->params()) {
    inputs.push_back(RandomValue(param->GetType(), engine));
  }
  return inputs;
}

}  // namespace xls
