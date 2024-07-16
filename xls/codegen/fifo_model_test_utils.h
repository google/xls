// Copyright 2024 The XLS Authors
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

#ifndef XLS_CODEGEN_FIFO_MODEL_TEST_UTILS_H_
#define XLS_CODEGEN_FIFO_MODEL_TEST_UTILS_H_

#include <cstdint>
#include <string>
#include <variant>

#include "gmock/gmock.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/value.h"
namespace xls::verilog {

MATCHER_P2(
    UsableOutputsMatch, input_set, output_set,
    absl::StrFormat(
        "All ready/valid signals to match and the data signal to match if "
        "ready & valid is asserted for output: %s, input %s",
        testing::PrintToString(output_set),
        testing::PrintToString(input_set))) {
  const absl::flat_hash_map<std::string, Value>& inputs = input_set;
  const absl::flat_hash_map<std::string, Value>& outputs = output_set;
  if (outputs.at(FifoInstantiation::kPopValidPortName).bits().IsOne() &&
      inputs.at(FifoInstantiation::kPopReadyPortName).bits().IsOne()) {
    return testing::ExplainMatchResult(
        testing::UnorderedElementsAreArray(outputs), arg, result_listener);
  }
  // It would be nice to check the pop_data value is the same too but it
  // doesn't seem useful if the data can't be read.
  return testing::ExplainMatchResult(
      testing::AllOf(testing::Contains(testing::Pair(
                         FifoInstantiation::kPopValidPortName,
                         outputs.at(FifoInstantiation::kPopValidPortName))),
                     testing::Contains(testing::Pair(
                         FifoInstantiation::kPushReadyPortName,
                         outputs.at(FifoInstantiation::kPushReadyPortName)))),
      arg, result_listener);
}

class BaseOperation {
 public:
  virtual ~BaseOperation() = default;
  virtual absl::flat_hash_map<std::string, Value> InputSet() const = 0;
};
struct Push : public BaseOperation {
  explicit Push(int32_t v) : v(v) {}
  int32_t v;

  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(v, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
struct Pop : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(0xf0f0f0f0, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct PushAndPop : public BaseOperation {
  explicit PushAndPop(int32_t v) : v(v) {}
  uint32_t v;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(v, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
struct NotReady : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(987654321, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct ResetOp : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct ResetPop : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct ResetPush : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
struct ResetPushPop : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
class Operation : public BaseOperation,
                  public std::variant<Push, Pop, PushAndPop, NotReady, ResetOp,
                                      ResetPush, ResetPop, ResetPushPop> {
 public:
  using std::variant<Push, Pop, PushAndPop, NotReady, ResetOp, ResetPush,
                     ResetPop, ResetPushPop>::variant;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return std::visit([&](auto v) { return v.InputSet(); }, *this);
  }
};

inline auto FifoConfigDomain() {
  return fuzztest::ConstructorOf<FifoConfig>(
      /*depth=*/fuzztest::InRange(1, 10),
      /*bypass=*/fuzztest::Arbitrary<bool>(),
      /*register_push_outputs=*/fuzztest::Arbitrary<bool>(),
      /*register_pop_outputs=*/fuzztest::Arbitrary<bool>());
}

inline auto OperationDomain() {
  return fuzztest::OneOf(
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<Pop>()),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<NotReady>()),
      fuzztest::ConstructorOf<Operation>(
          fuzztest::ConstructorOf<Push>(fuzztest::InRange(1, 1000))),
      fuzztest::ConstructorOf<Operation>(
          fuzztest::ConstructorOf<PushAndPop>(fuzztest::InRange(1, 1000))),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<ResetOp>()),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<ResetPush>()),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<ResetPop>()),
      fuzztest::ConstructorOf<Operation>(
          fuzztest::ConstructorOf<ResetPushPop>()));
}

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_FIFO_MODEL_TEST_UTILS_H_
