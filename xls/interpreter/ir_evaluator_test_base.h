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

#ifndef XLS_IR_IR_EVALUATOR_TEST_H_
#define XLS_IR_IR_EVALUATOR_TEST_H_

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {

// Simple holder struct to contain the per-evaluator data needed to run these
// tests.
struct IrEvaluatorTestParam {
  // Function to perform evaluation of the specified program with the given
  // [positional] args.
  using EvaluatorFnT = std::function<absl::StatusOr<Value>(
      Function* function, const std::vector<Value>& args)>;

  // Function to perform evaluation of the specified program with the given
  // keyword args.
  using KwargsEvaluatorFnT = std::function<absl::StatusOr<Value>(
      Function* function,
      const absl::flat_hash_map<std::string, Value>& kwargs)>;

  IrEvaluatorTestParam(EvaluatorFnT evaluator_in,
                       KwargsEvaluatorFnT kwargs_evaluator_in)
      : evaluator(std::move(evaluator_in)),
        kwargs_evaluator(std::move(kwargs_evaluator_in)) {}

  // Function to execute a function and return a Value.
  EvaluatorFnT evaluator;

  // Function to execute a function w/keyword args and return a Value.
  KwargsEvaluatorFnT kwargs_evaluator;
};

// Public face of the suite of tests to run against IR evaluators
// (IrInterpreter, LlvmIrJit). Users should instantiate with an
// INSTANTIATE_TEST_SUITE_P macro; see llvm_ir_jit_test.cc for an example.
class IrEvaluatorTestBase
    : public IrTestBase,
      public testing::WithParamInterface<IrEvaluatorTestParam> {
 protected:
  Value AsValue(absl::string_view input_string) {
    return Parser::ParseTypedValue(input_string).value();
  }

  absl::StatusOr<Function*> ParseAndGetFunction(Package* package,
                                                absl::string_view program) {
    XLS_ASSIGN_OR_RETURN(Function * function,
                         Parser::ParseFunction(program, package));
    XLS_VLOG(1) << "Dumped:\n" << function->DumpIr();
    return function;
  }

  absl::StatusOr<Function*> get_neg_function(Package* package) {
    return ParseAndGetFunction(package, R"(
  fn negate_value(a: bits[4]) -> bits[4] {
    ret neg.1: bits[4] = neg(a)
  }
  )");
  }

  // Runs the given function with uint64s as input. Converts to/from Values
  // under the hood. All arguments and result must be bits-typed.
  absl::StatusOr<uint64_t> Run(Function* f, absl::Span<const uint64_t> args) {
    std::vector<Value> value_args;
    for (int64_t i = 0; i < args.size(); ++i) {
      XLS_RET_CHECK(f->param(i)->GetType()->IsBits());
      value_args.push_back(Value(UBits(args[i], f->param(i)->BitCountOrDie())));
    }
    XLS_ASSIGN_OR_RETURN(Value value_result,
                         GetParam().evaluator(f, value_args));
    XLS_RET_CHECK(value_result.IsBits());
    return value_result.bits().ToUint64();
  }

  // Runs the given function with Bits as input. Converts to/from Values under
  // the hood. All arguments and result must be bits-typed.
  absl::StatusOr<Bits> RunBits(Function* f, absl::Span<const Bits> args) {
    std::vector<Value> value_args;
    for (int64_t i = 0; i < args.size(); ++i) {
      value_args.push_back(Value(args[i]));
    }
    XLS_ASSIGN_OR_RETURN(Value value_result,
                         GetParam().evaluator(f, value_args));
    XLS_RET_CHECK(value_result.IsBits());
    return value_result.bits();
  }
};

}  // namespace xls

#endif  // XLS_IR_IR_EVALUATOR_TEST_H_
