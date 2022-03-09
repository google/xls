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

#include "xls/interpreter/random_value.h"

#include "absl/status/status.h"
#include "xls/interpreter/function_interpreter.h"

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
  bytes.reserve(bit_count);
  std::uniform_int_distribution<int32_t> generator(0, 255);
  for (int64_t i = 0; i < bit_count; i += 8) {
    bytes.push_back(static_cast<uint8_t>(generator(*engine)));
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

absl::StatusOr<std::vector<Value>> RandomFunctionArguments(
    Function* f, std::minstd_rand* engine, Function* validator,
    int64_t max_attempts) {
  if (validator == nullptr) {
    return absl::InvalidArgumentError(
        "Function argument validator can not be null.");
  }

  Type* validator_return = validator->GetType()->return_type();
  if (!validator_return->IsBits() || validator_return->GetFlatBitCount() != 1) {
    XLS_LOG(INFO) << "VR: " << validator_return->ToString();
    return absl::InvalidArgumentError(
        "Function argument validator must return a single bit value.");
  }

  // Accept or reject candidates based on the result of evaluating against the
  // validator function.
  int64_t num_attempts = 0;
  while (num_attempts++ < max_attempts) {
    std::vector<Value> inputs;
    for (Param* param : f->params()) {
      inputs.push_back(RandomValue(param->GetType(), engine));
    }

    XLS_ASSIGN_OR_RETURN(Value result, DropInterpreterEvents(InterpretFunction(
                                           validator, inputs)));
    if (result.bits().IsOne()) {
      return inputs;
    }
  }

  return absl::ResourceExhaustedError(absl::StrCat(
      "Unable to generate valid input after ", max_attempts,
      "attempts. The validator may be difficult/impossible to satisfy "
      "or the limit should be increased."));
}

}  // namespace xls
