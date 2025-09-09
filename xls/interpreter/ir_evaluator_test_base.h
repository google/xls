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

#ifndef XLS_INTERPRETER_IR_EVALUATOR_TEST_BASE_H_
#define XLS_INTERPRETER_IR_EVALUATOR_TEST_BASE_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"

namespace xls {

// Simple holder struct to contain the per-evaluator data needed to run these
// tests.
struct IrEvaluatorTestParam {
  // Function to perform evaluation of the specified program with the given
  // [positional] args.
  using EvaluatorFnT = std::function<absl::StatusOr<InterpreterResult<Value>>(
      Function* function, absl::Span<const Value> args,
      const EvaluatorOptions& options,
      std::optional<EvaluationObserver*> observer)>;

  // Function to perform evaluation of the specified program with the given
  // keyword args.
  using KwargsEvaluatorFnT =
      std::function<absl::StatusOr<InterpreterResult<Value>>(
          Function* function,
          const absl::flat_hash_map<std::string, Value>& kwargs,
          const EvaluatorOptions& options,
          std::optional<EvaluationObserver*> observer)>;

  IrEvaluatorTestParam(EvaluatorFnT evaluator_in,
                       KwargsEvaluatorFnT kwargs_evaluator_in,
                       bool supports_observer, std::string name)
      : evaluator(std::move(evaluator_in)),
        kwargs_evaluator(std::move(kwargs_evaluator_in)),
        supports_observer(supports_observer),
        name(name) {}

  // Function to execute a function and return a Value.
  EvaluatorFnT evaluator;

  // Function to execute a function w/keyword args and return a Value.
  KwargsEvaluatorFnT kwargs_evaluator;

  bool supports_observer;

  std::string name;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const IrEvaluatorTestParam& p) {
    absl::Format(&sink, "%s", p.name);
  }
};

// Public face of the suite of tests to run against IR evaluators
// (IrInterpreter, FunctionJit). Users should instantiate with an
// INSTANTIATE_TEST_SUITE_P macro; see llvm_ir_jit_test.cc for an example.
class IrEvaluatorTestBase
    : public IrTestBase,
      public testing::WithParamInterface<IrEvaluatorTestParam> {
 protected:
  Value AsValue(std::string_view input_string) {
    return Parser::ParseTypedValue(input_string).value();
  }

  absl::StatusOr<Function*> ParseAndGetFunction(Package* package,
                                                std::string_view program) {
    XLS_ASSIGN_OR_RETURN(Function * function,
                         Parser::ParseFunction(program, package));
    VLOG(1) << "Dumped:\n" << function->DumpIr();
    return function;
  }

  absl::StatusOr<Function*> get_neg_function(Package* package) {
    return ParseAndGetFunction(package, R"(
  fn negate_value(a: bits[4]) -> bits[4] {
    ret neg.1: bits[4] = neg(a)
  }
  )");
  }

  // Run the given function with Values as input, returning the result and any
  // events generated.
  absl::StatusOr<InterpreterResult<Value>> RunWithEvents(
      Function* f, absl::Span<const Value> args,
      const EvaluatorOptions& options = EvaluatorOptions(),
      std::optional<EvaluationObserver*> observer = std::nullopt) {
    return GetParam().evaluator(f, args, options, observer);
  }

  // Runs the given function with Values as input, checking that no traces or
  // assertion failures are recorded.
  absl::StatusOr<Value> RunWithNoEvents(
      Function* f, absl::Span<const Value> args,
      const EvaluatorOptions& options = EvaluatorOptions(),
      std::optional<EvaluationObserver*> observer = std::nullopt) {
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> result,
                         GetParam().evaluator(f, args, options, observer));

    if (!result.events.trace_msgs.empty()) {
      std::vector<std::string_view> trace_messages;
      trace_messages.reserve(result.events.trace_msgs.size());
      for (const TraceMessage& trace : result.events.trace_msgs) {
        trace_messages.push_back(trace.message);
      }
      return absl::FailedPreconditionError(
          absl::StrFormat("Unexpected traces during RunWithNoEvents:\n%s",
                          absl::StrJoin(trace_messages, "\n")));
    }

    return InterpreterResultToStatusOrValue(result);
  }

  // Runs the given function with uint64s as input, checking that no events are
  // generated. Converts to/from Values under the hood. All arguments and result
  // must be bits-typed.
  absl::StatusOr<uint64_t> RunWithUint64sNoEvents(
      Function* f, absl::Span<const uint64_t> args,
      std::optional<EvaluationObserver*> observer = std::nullopt) {
    std::vector<Value> value_args;
    for (int64_t i = 0; i < args.size(); ++i) {
      XLS_RET_CHECK(f->param(i)->GetType()->IsBits());
      value_args.push_back(Value(UBits(args[i], f->param(i)->BitCountOrDie())));
    }
    XLS_ASSIGN_OR_RETURN(
        Value value_result,
        RunWithNoEvents(f, value_args, EvaluatorOptions(), observer));
    XLS_RET_CHECK(value_result.IsBits());
    return value_result.bits().ToUint64();
  }

  // Runs the given function with Bits as input, checking that no events are
  // generated. Converts to/from Values under the hood. All arguments and result
  // must be bits-typed.
  absl::StatusOr<Bits> RunWithBitsNoEvents(Function* f,
                                           absl::Span<const Bits> args) {
    std::vector<Value> value_args;
    value_args.reserve(args.size());
    for (int64_t i = 0; i < args.size(); ++i) {
      value_args.push_back(Value(args[i]));
    }
    XLS_ASSIGN_OR_RETURN(Value value_result, RunWithNoEvents(f, value_args));
    XLS_RET_CHECK(value_result.IsBits());
    return value_result.bits();
  }

  // Runs the given function with keyword arguments as input, checking that no
  // events are generated.
  absl::StatusOr<Value> RunWithKwargsNoEvents(
      Function* function, const absl::flat_hash_map<std::string, Value>& kwargs,
      const EvaluatorOptions& options = EvaluatorOptions()) {
    XLS_ASSIGN_OR_RETURN(
        InterpreterResult<Value> result,
        GetParam().kwargs_evaluator(function, kwargs, options, std::nullopt));
    XLS_RET_CHECK(result.events.trace_msgs.empty());
    XLS_RET_CHECK(result.events.assert_msgs.empty());
    return result.value;
  }
};

}  // namespace xls

#endif  // XLS_INTERPRETER_IR_EVALUATOR_TEST_BASE_H_
