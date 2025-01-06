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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"

namespace xls {

// A package which verifies itself upon destruction. This is useful to verify
// that any transformations done during the test were valid without having to
// explicitly call the verifier.
class VerifiedPackage : public Package {
 public:
  explicit VerifiedPackage(std::string_view name) : Package(name) {}
  ~VerifiedPackage() override;

  void AcceptInvalid() { verify_ = false; }

 private:
  bool verify_ = true;
};

// A test base class with convenience functions for IR tests.
class IrTestBase : public ::testing::Test {
 protected:
  IrTestBase() = default;

  static std::string TestName() {
    // If we try to run the program it can't have the '/' in its name. Remove
    // them so this pattern works.
    std::string name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    absl::StrReplaceAll(std::vector{std::pair{"/", "_"}}, &name);
    return name;
  }

  // Creates an empty package with a name equal to TestName().
  static std::unique_ptr<VerifiedPackage> CreatePackage() {
    return std::make_unique<VerifiedPackage>(TestName());
  }

  // Parses the given text as a package and replaces the owned packaged with the
  // result. Package can be accessed by calling "package()".
  static absl::StatusOr<std::unique_ptr<VerifiedPackage>> ParsePackage(
      std::string_view text);

  // As above but skips IR verification and returns an ordinary Package..
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackageNoVerify(
      std::string_view text);

  // Parse the input_string as a function into the given package.
  absl::StatusOr<Function*> ParseFunction(std::string_view text,
                                          Package* package);

  // Parse the input_string as a proc into the given package.
  absl::StatusOr<Proc*> ParseProc(std::string_view text, Package* package);

  // Finds and returns the node in the given package (function) with the given
  // name. Dies if no such node exists.
  static Node* FindNode(std::string_view name, Package* package);
  static Node* FindNode(std::string_view name, FunctionBase* function);

  // Finds returns True if the node in the given package (function)
  // with the given name.
  static bool HasNode(std::string_view name, Package* package);
  static bool HasNode(std::string_view name, FunctionBase* function);

  // Finds and returns the function, proc, or block in the given package with
  // the given name. Dies if no such function or proc exists.
  static Function* FindFunction(std::string_view name, Package* package);
  static Proc* FindProc(std::string_view name, Package* package);
  static Block* FindBlock(std::string_view name, Package* package);

 protected:
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

class TestDelayEstimator : public DelayEstimator {
 public:
  explicit TestDelayEstimator(int64_t base_delay = 1)
      : DelayEstimator("test"), base_delay_(base_delay) {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kAfterAll:
      case Op::kMinDelay:
      case Op::kBitSlice:
      case Op::kConcat:
      case Op::kLiteral:
      case Op::kParam:
      case Op::kStateRead:
      case Op::kNext:
      case Op::kReceive:
      case Op::kSend:
      case Op::kTupleIndex:
        return 0;
      case Op::kUDiv:
      case Op::kSDiv:
        return 2 * base_delay_;
      default:
        return base_delay_;
    }
  }

 private:
  int64_t base_delay_;
};

// Helper to record IR before and after some test event which changes it.
struct ScopedRecordIr {
 public:
  explicit ScopedRecordIr(Package* p, std::string_view name = "",
                          bool with_initial = true)
      : p_(p), name_(name) {
    if (with_initial) {
      testing::Test::RecordProperty(
          absl::StrFormat("initial%s%s", name_.empty() ? "" : "_", name_),
          p_->DumpIr());
    }
  }
  ~ScopedRecordIr() {
    testing::Test::RecordProperty(
        absl::StrFormat("final%s%s", name_.empty() ? "" : "_", name_),
        p_->DumpIr());
  }

 private:
  Package* p_;
  std::string_view name_;
};

// Helper to record something on failure.
template <typename T>
struct ScopedMaybeRecord {
 public:
  ScopedMaybeRecord(std::string_view title, T t) : title_(title), t_(t) {}
  ~ScopedMaybeRecord() {
    if (testing::Test::HasFailure()) {
      if constexpr (std::is_convertible_v<T, std::string_view>) {
        testing::Test::RecordProperty(title_, t_);
      } else {
        testing::Test::RecordProperty(title_, testing::PrintToString(t_));
      }
    }
  }

 private:
  std::string title_;
  T t_;
};

template <typename T>
ScopedMaybeRecord(std::string_view, T) -> ScopedMaybeRecord<T>;

}  // namespace xls

#endif  // XLS_IR_IR_TEST_BASE_H_
