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
#include <memory>
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
        "ready & valid is asserted for\noutput:\n %s\ninput:\n %s",
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
  // Coerce operation into an operation of the given bitwidth but otherwise
  // equivalent semantics.
  virtual std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const = 0;
};
struct Push : public BaseOperation {
  explicit Push(int32_t v) : v(UBits(v, 32)) {}
  explicit Push(Bits v) : v(v) {}
  Bits v;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
    if (v.bit_count() > 0) {
      result[std::string(FifoInstantiation::kPushDataPortName)] = Value(v);
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<Push>(
        Push(Bits::FromBitmap(v.bitmap().WithSize(count))));
  }
};
struct Pop : public BaseOperation {
  explicit Pop() : bit_count(32) {};
  explicit Pop(int64_t bit_count) : bit_count(bit_count) {};
  int64_t bit_count;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
    if (bit_count > 0) {
      result[std::string(FifoInstantiation::kPushDataPortName)] =
          Value(UBits(0xf0f0f0f0, 32));
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<Pop>(Pop(count));
  }
};
struct PushAndPop : public BaseOperation {
  explicit PushAndPop(int32_t v) : v(UBits(v, 32)) {}
  explicit PushAndPop(Bits v) : v(v) {}
  Bits v;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
    if (v.bit_count() > 0) {
      result[std::string(FifoInstantiation::kPushDataPortName)] = Value(v);
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<PushAndPop>(
        PushAndPop(Bits::FromBitmap(v.bitmap().WithSize(count))));
  }
};
struct NotReady : public BaseOperation {
  explicit NotReady() : bit_count(32) {};
  explicit NotReady(int64_t bit_count) : bit_count(bit_count) {};
  int64_t bit_count;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
    if (bit_count > 0) {
      result[std::string(FifoInstantiation::kPushDataPortName)] =
          Value(UBits(987654321, bit_count));
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<NotReady>(NotReady(count));
  }
};
struct ResetOp : public BaseOperation {
  explicit ResetOp() : bit_count(32) {};
  explicit ResetOp(int64_t bit_count) : bit_count(bit_count) {};
  int64_t bit_count;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
    if (bit_count > 0) {
      result[std::string(FifoInstantiation::kPushDataPortName)] =
          Value(UBits(123456789, bit_count));
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<ResetOp>(ResetOp(count));
  }
};
struct ResetPop : public BaseOperation {
  explicit ResetPop() : bit_count(32) {};
  explicit ResetPop(int64_t bit_count) : bit_count(bit_count) {};
  int64_t bit_count;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
    if (bit_count) {
      result[std::string(FifoInstantiation::kPushDataPortName)] =
          Value(UBits(123456789, bit_count));
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<ResetPop>(ResetPop(count));
  }
};
struct ResetPush : public BaseOperation {
  explicit ResetPush() : bit_count(32) {};
  explicit ResetPush(int64_t bit_count) : bit_count(bit_count) {};
  int64_t bit_count;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
    if (bit_count > 0) {
      result[std::string(FifoInstantiation::kPushDataPortName)] =
          Value(UBits(123456789, bit_count));
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<ResetPush>(ResetPush(count));
  }
};
struct ResetPushPop : public BaseOperation {
  explicit ResetPushPop() : bit_count(32) {};
  explicit ResetPushPop(int64_t bit_count) : bit_count(bit_count) {};
  int64_t bit_count;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    absl::flat_hash_map<std::string, Value> result = {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
    if (bit_count > 0) {
      result[std::string(FifoInstantiation::kPushDataPortName)] =
          Value(UBits(123456789, bit_count));
    }
    return result;
  }
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::make_unique<ResetPushPop>(ResetPushPop(count));
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
  std::unique_ptr<BaseOperation> WithBitWidth(int64_t count) const override {
    return std::visit([&](auto v) { return v.WithBitWidth(count); }, *this);
  }
};

struct FifoTestParam {
  int64_t data_bit_count;
  FifoConfig config;
};

template <typename Sink>
void AbslStringify(Sink& sink, const FifoTestParam& value) {
  absl::Format(&sink,
               "kBitCount%iDepth%i%sBypass%sRegPushOutputs%sRegPopOutputs",
               value.data_bit_count, value.config.depth(),
               value.config.bypass() ? "" : "No",
               value.config.register_push_outputs() ? "" : "No",
               value.config.register_pop_outputs() ? "" : "No");
}

inline auto FifoTestParamDomain() {
  return fuzztest::Filter(
      [](const FifoTestParam& params) { return params.config.Validate().ok(); },
      fuzztest::ConstructorOf<FifoTestParam>(
          /*data_bit_count=*/fuzztest::OneOf(fuzztest::Just(0),
                                             fuzztest::Just(32)),
          fuzztest::ConstructorOf<FifoConfig>(
              /*depth=*/fuzztest::InRange(0, 10),
              /*bypass=*/fuzztest::Arbitrary<bool>(),
              /*register_push_outputs=*/fuzztest::Arbitrary<bool>(),
              /*register_pop_outputs=*/fuzztest::Arbitrary<bool>())));
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
