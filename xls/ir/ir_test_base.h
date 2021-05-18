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

#ifndef XLS_IR_IR_TEST_BASE_H_
#define XLS_IR_IR_TEST_BASE_H_

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/source_location.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/verifier.h"

namespace xls {

// A package which verifies itself upon destruction. This is useful to verify
// that any transformations done during the test were valid without having to
// explicitly call the verifier.
class VerifiedPackage : public Package {
 public:
  explicit VerifiedPackage(absl::string_view name,
                           absl::optional<absl::string_view> entry)
      : Package(name, entry) {}
  ~VerifiedPackage() override;
};

// A test base class with convenience functions for IR tests.
class IrTestBase : public ::testing::Test {
 protected:
  IrTestBase() {}

  static std::string TestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  // Creates an empty package with a name equal to TestName().
  static std::unique_ptr<VerifiedPackage> CreatePackage() {
    return absl::make_unique<VerifiedPackage>(TestName(), absl::nullopt);
  }

  // Parses the given text as a package and replaces the owned packaged with the
  // result. Package can be accessed by calling "package()".
  static absl::StatusOr<std::unique_ptr<VerifiedPackage>> ParsePackage(
      absl::string_view text);

  // As above but skips IR verification and returns an ordinary Package..
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackageNoVerify(
      absl::string_view text);

  // Parse the input_string as a function into the given package.
  absl::StatusOr<Function*> ParseFunction(absl::string_view text,
                                          Package* package);

  // Parse the input_string as a proc into the given package.
  absl::StatusOr<Proc*> ParseProc(absl::string_view text, Package* package);

  // Finds and returns the node in the given package (function) with the given
  // name. Dies if no such node exists.
  static Node* FindNode(absl::string_view name, Package* package);
  static Node* FindNode(absl::string_view name, FunctionBase* function);

  // Finds and returns the function, proc, or block in the given package with
  // the given name. Dies if no such function or proc exists.
  static Function* FindFunction(absl::string_view name, Package* package);
  static Proc* FindProc(absl::string_view name, Package* package);
  static Block* FindBlock(absl::string_view name, Package* package);

  // Runs the given package (passed as IR text) and EXPECTs the result to equal
  // 'expected'. Runs the package in several ways:
  // (1) unoptimized IR through the interpreter.
  // (2) optimized IR through the interpreter. (enabled with run_optimized)
  // (3) pipeline generator emitted Verilog through a Verilog simulator.
  //          (enabled with simulate)
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, uint64_t>& args, uint64_t expected,
      absl::string_view package_text, bool run_optimized = true,
      bool simulate = true,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  // Overload which takes Bits as arguments and the expected result.
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, Bits>& args, Bits expected,
      absl::string_view package_text, bool run_optimized = true,
      bool simulate = true,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  // Overload which takes Values as arguments and the expected result.
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, Value>& args, Value expected,
      absl::string_view package_text, bool run_optimized = true,
      bool simulate = true,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

 private:
  // Helper for RunAndExpectEq which accepts arguments and expectation as Values
  // and takes a std::unique_ptr<Package>.
  static void RunAndExpectEq(
      const absl::flat_hash_map<std::string, Value>& args,
      const Value& expected, std::unique_ptr<Package>&& package,
      bool run_optimized = true, bool simulate = true);

  // Converts the given map of uint64_t arguments into a map of Value argument
  // with the appropriate bit widths as determined by the package.
  static absl::StatusOr<absl::flat_hash_map<std::string, Value>>
  UInt64ArgsToValues(const absl::flat_hash_map<std::string, uint64_t>& args,
                     Package* package);

  // Converts the uint64_t result to a Value with the appropriate bit widths as
  // determined by the package return value.
  static absl::StatusOr<Value> UInt64ResultToValue(uint64_t value,
                                                   Package* package);
};

}  // namespace xls

#endif  // XLS_IR_IR_TEST_BASE_H_
