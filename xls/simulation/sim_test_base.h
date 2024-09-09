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

#ifndef XLS_SIMULATION_SIM_TEST_BASE_H_
#define XLS_SIMULATION_SIM_TEST_BASE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "xls/common/source_location.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {

class SimTestBase : public IrTestBase {
 public:
  // Reexport common statics.
  using IrTestBase::CreatePackage;
  using IrTestBase::FindBlock;
  using IrTestBase::FindFunction;
  using IrTestBase::FindNode;
  using IrTestBase::FindProc;
  using IrTestBase::HasNode;
  using IrTestBase::ParsePackage;
  using IrTestBase::ParsePackageNoVerify;
  using IrTestBase::TestName;

  // Runs the given package (passed as IR text) and EXPECTs the result to equal
  // 'expected'. Runs the package in several ways:
  // (1) unoptimized IR through the interpreter.
  // (2) optimized IR through the interpreter. (enabled with run_optimized)
  // (3) pipeline generator emitted Verilog through a Verilog simulator.
  //          (enabled with simulate)
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, uint64_t>& args, uint64_t expected,
      std::string_view package_text, bool run_optimized = true,
      bool simulate = true,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  // Overload which takes Bits as arguments and the expected result.
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, Bits>& args, Bits expected,
      std::string_view package_text, bool run_optimized = true,
      bool simulate = true,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  // Overload which takes Values as arguments and the expected result.
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, Value>& args, Value expected,
      std::string_view package_text, bool run_optimized = true,
      bool simulate = true,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

 private:
  // Helper for RunAndExpectEq which accepts arguments and expectation as Values
  // and takes a std::unique_ptr<Package>.
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, Value>& args,
      const Value& expected, std::unique_ptr<Package>&& package,
      bool run_optimized = true, bool simulate = true);
};

}  // namespace xls

#endif  // XLS_SIMULATION_SIM_TEST_BASE_H_
