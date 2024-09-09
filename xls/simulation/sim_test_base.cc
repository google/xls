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

#include "xls/simulation/sim_test_base.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_test_util.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/simulation/default_verilog_simulator.h"
#include "xls/simulation/module_simulator.h"

namespace xls {

void SimTestBase::RunAndExpectEq(
    const absl::flat_hash_map<std::string, uint64_t>& args, uint64_t expected,
    std::string_view package_text, bool run_optimized, bool simulate,
    xabsl::SourceLocation loc) {
  // Emit the filename/line of the test code in any failure message. The
  // location is captured as a default argument to RunAndExpectEq.
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "RunAndExpectEq failed");
  VLOG(3) << "Package text:\n" << package_text;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(package_text));
  absl::flat_hash_map<std::string, Value> arg_values;
  XLS_ASSERT_OK_AND_ASSIGN(arg_values, UInt64ArgsToValues(args, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_value,
                           UInt64ResultToValue(expected, package.get()));

  RunAndExpectEq(arg_values, expected_value, std::move(package), run_optimized,
                 simulate);
}

void SimTestBase::RunAndExpectEq(
    const absl::flat_hash_map<std::string, Bits>& args, Bits expected,
    std::string_view package_text, bool run_optimized, bool simulate,
    xabsl::SourceLocation loc) {
  // Emit the filename/line of the test code in any failure message. The
  // location is captured as a default argument to RunAndExpectEq.
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "RunAndExpectEq failed");
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(package_text));
  absl::flat_hash_map<std::string, Value> args_as_values;
  for (const auto& pair : args) {
    args_as_values[pair.first] = Value(pair.second);
  }
  RunAndExpectEq(args_as_values, Value(std::move(expected)), std::move(package),
                 run_optimized, simulate);
}

void SimTestBase::RunAndExpectEq(
    const absl::flat_hash_map<std::string, Value>& args, Value expected,
    std::string_view package_text, bool run_optimized, bool simulate,
    xabsl::SourceLocation loc) {
  // Emit the filename/line of the test code in any failure message. The
  // location is captured as a default argument to RunAndExpectEq.
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "RunAndExpectEq failed");
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(package_text));
  RunAndExpectEq(args, expected, std::move(package), run_optimized, simulate);
}

void SimTestBase::RunAndExpectEq(
    const absl::flat_hash_map<std::string, Value>& args, const Value& expected,
    std::unique_ptr<Package>&& package, bool run_optimized, bool simulate) {
  InterpreterEvents unopt_events;

  // Run interpreter on unoptimized IR.
  {
    XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->GetTopAsFunction());
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                             InterpretFunctionKwargs(entry, args));
    XLS_ASSERT_OK(InterpreterEventsToStatus(result.events));
    unopt_events = result.events;
    ASSERT_TRUE(ValuesEqual(expected, result.value))
        << "(interpreted unoptimized IR)";
  }

  if (run_optimized) {
    // Run main pipeline.
    XLS_ASSERT_OK(RunOptimizationPassPipeline(package.get()));

    // Run interpreter on optimized IR.
    {
      XLS_ASSERT_OK_AND_ASSIGN(Function * main, package->GetTopAsFunction());
      XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                               InterpretFunctionKwargs(main, args));
      XLS_ASSERT_OK(InterpreterEventsToStatus(result.events));
      ASSERT_EQ(unopt_events, result.events);
      ASSERT_TRUE(ValuesEqual(expected, result.value))
          << "(interpreted optimized IR)";
    }
  }

  // Emit Verilog with combinational generator and run with ModuleSimulator.
  if (simulate) {
    ASSERT_EQ(package->functions().size(), 1);
    std::optional<FunctionBase*> top = package->GetTop();
    EXPECT_TRUE(top.has_value());
    EXPECT_TRUE(top.value()->IsFunction());

    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::ModuleGeneratorResult result,
        verilog::GenerateCombinationalModule(
            top.value(), verilog::CodegenOptions().use_system_verilog(false)));

    absl::flat_hash_map<std::string, Value> arg_set;
    for (const auto& pair : args) {
      arg_set.insert(pair);
    }
    VLOG(3) << "Verilog text:\n" << result.verilog_text;
    verilog::ModuleSimulator simulator(result.signature, result.verilog_text,
                                       verilog::FileType::kVerilog,
                                       &verilog::GetDefaultVerilogSimulator());
    XLS_ASSERT_OK_AND_ASSIGN(Value actual, simulator.RunFunction(arg_set));
    ASSERT_TRUE(ValuesEqual(expected, actual)) << "(Verilog simulation)";
  }
}

}  // namespace xls
