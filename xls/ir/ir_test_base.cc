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

#include "xls/ir/ir_test_base.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value_test_util.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_simulators.h"

namespace xls {

VerifiedPackage::~VerifiedPackage() {
  absl::Status status = VerifyPackage(this);
  if (!status.ok()) {
    ADD_FAILURE() << absl::StrFormat(
        "IR verifier failed on package %s during destruction: %s", name(),
        status.message());
  }
}

absl::StatusOr<std::unique_ptr<VerifiedPackage>> IrTestBase::ParsePackage(
    std::string_view text) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedPackage> package,
                       Parser::ParseDerivedPackageNoVerify<VerifiedPackage>(
                           text, std::nullopt));
  XLS_RETURN_IF_ERROR(VerifyPackage(package.get()));
  return std::move(package);
}

absl::StatusOr<std::unique_ptr<Package>> IrTestBase::ParsePackageNoVerify(
    std::string_view text) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackageNoVerify(text));
  return std::move(package);
}

absl::StatusOr<Function*> IrTestBase::ParseFunction(std::string_view text,
                                                    Package* package) {
  return Parser::ParseFunction(text, package);
}

absl::StatusOr<Proc*> IrTestBase::ParseProc(std::string_view text,
                                            Package* package) {
  return Parser::ParseProc(text, package);
}

Node* IrTestBase::FindNode(std::string_view name, Package* package) {
  for (FunctionBase* function : package->GetFunctionBases()) {
    for (Node* node : function->nodes()) {
      if (node->GetName() == name) {
        return node;
      }
    }
  }
  LOG(FATAL) << "No node named " << name << " in package:\n" << *package;
}

Node* IrTestBase::FindNode(std::string_view name, FunctionBase* function) {
  for (Node* node : function->nodes()) {
    if (node->GetName() == name) {
      return node;
    }
  }
  LOG(FATAL) << "No node named " << name << " in function:\n" << *function;
}

bool IrTestBase::HasNode(std::string_view name, Package* package) {
  for (FunctionBase* function : package->GetFunctionBases()) {
    for (Node* node : function->nodes()) {
      if (node->GetName() == name) {
        return true;
      }
    }
  }
  return false;
}

bool IrTestBase::HasNode(std::string_view name, FunctionBase* function) {
  for (Node* node : function->nodes()) {
    if (node->GetName() == name) {
      return true;
    }
  }
  return false;
}

Function* IrTestBase::FindFunction(std::string_view name, Package* package) {
  for (auto& function : package->functions()) {
    if (function->name() == name) {
      return function.get();
    }
  }
  LOG(FATAL) << "No function named " << name << " in package:\n" << *package;
}

Proc* IrTestBase::FindProc(std::string_view name, Package* package) {
  for (auto& proc : package->procs()) {
    if (proc->name() == name) {
      return proc.get();
    }
  }
  LOG(FATAL) << "No proc named " << name << " in package:\n" << *package;
}

Block* IrTestBase::FindBlock(std::string_view name, Package* package) {
  for (auto& block : package->blocks()) {
    if (block->name() == name) {
      return block.get();
    }
  }
  LOG(FATAL) << "No block named " << name << " in package:\n" << *package;
}

void IrTestBase::RunAndExpectEq(
    const absl::flat_hash_map<std::string, uint64_t>& args, uint64_t expected,
    std::string_view package_text, bool run_optimized, bool simulate,
    xabsl::SourceLocation loc) {
  // Emit the filename/line of the test code in any failure message. The
  // location is captured as a default argument to RunAndExpectEq.
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "RunAndExpectEq failed");
  XLS_VLOG(3) << "Package text:\n" << package_text;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           ParsePackage(package_text));
  absl::flat_hash_map<std::string, Value> arg_values;
  XLS_ASSERT_OK_AND_ASSIGN(arg_values, UInt64ArgsToValues(args, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_value,
                           UInt64ResultToValue(expected, package.get()));

  RunAndExpectEq(arg_values, expected_value, std::move(package), run_optimized,
                 simulate);
}

void IrTestBase::RunAndExpectEq(
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
  RunAndExpectEq(args_as_values, Value(expected), std::move(package),
                 run_optimized, simulate);
}

void IrTestBase::RunAndExpectEq(
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

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
IrTestBase::UInt64ArgsToValues(
    const absl::flat_hash_map<std::string, uint64_t>& args, Package* package) {
  absl::flat_hash_map<std::string, Value> value_args;
  for (const auto& pair : args) {
    const std::string& param_name = pair.first;
    const uint64_t arg_value = pair.second;
    std::optional<FunctionBase*> top = package->GetTop();
    if (!top.has_value()) {
      return absl::InternalError(absl::StrFormat(
          "Top entity not set for package: %s.", package->name()));
    }
    XLS_ASSIGN_OR_RETURN(Param * param,
                         top.value()->GetParamByName(pair.first));
    Type* type = param->GetType();
    XLS_RET_CHECK(type->IsBits())
        << absl::StrFormat("Parameter '%s' is not a bits type: %s",
                           param->name(), type->ToString());
    XLS_RET_CHECK_GE(type->AsBitsOrDie()->bit_count(),
                     Bits::MinBitCountUnsigned(arg_value))
        << absl::StrFormat(
               "Argument value %d for parameter '%s' does not fit in type %s",
               arg_value, param->name(), type->ToString());
    value_args[param_name] =
        Value(UBits(arg_value, type->AsBitsOrDie()->bit_count()));
  }

  return value_args;
}

absl::StatusOr<Value> IrTestBase::UInt64ResultToValue(uint64_t value,
                                                      Package* package) {
  std::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Top entity not set for package: %s.", package->name()));
  }
  if (!top.value()->IsFunction()) {
    return absl::InternalError(absl::StrFormat(
        "Top entity is not a function for package: %s.", package->name()));
  }
  Function* entry = top.value()->AsFunctionOrDie();
  Type* return_type = entry->return_value()->GetType();
  XLS_RET_CHECK(return_type->IsBits()) << absl::StrFormat(
      "Return value of function not a bits type: %s", return_type->ToString());
  XLS_RET_CHECK_GE(return_type->AsBitsOrDie()->bit_count(),
                   Bits::MinBitCountUnsigned(value))
      << absl::StrFormat("Value %d does not fit in return type %s", value,
                         return_type->ToString());
  return Value(UBits(value, return_type->AsBitsOrDie()->bit_count()));
}

void IrTestBase::RunAndExpectEq(
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
    XLS_VLOG(3) << "Verilog text:\n" << result.verilog_text;
    verilog::ModuleSimulator simulator(result.signature, result.verilog_text,
                                       verilog::FileType::kVerilog,
                                       &verilog::GetDefaultVerilogSimulator());
    XLS_ASSERT_OK_AND_ASSIGN(Value actual, simulator.RunFunction(arg_set));
    ASSERT_TRUE(ValuesEqual(expected, actual)) << "(Verilog simulation)";
  }
}

}  // namespace xls
