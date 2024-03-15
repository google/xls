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

#include "xls/common/status/status_macros.h"

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_builder.h"

namespace {

using ::testing::AllOf;
using ::testing::ContainsRegex;
using ::testing::Eq;
using ::testing::HasSubstr;

absl::Status ReturnOk() { return absl::OkStatus(); }

xabsl::StatusBuilder ReturnOkBuilder() {
  return xabsl::StatusBuilder(absl::OkStatus());
}

absl::Status ReturnError(std::string_view msg) {
  return absl::UnknownError(msg);
}

xabsl::StatusBuilder ReturnErrorBuilder(std::string_view msg) {
  return xabsl::StatusBuilder(absl::UnknownError(msg));
}

absl::StatusOr<int> ReturnStatusOrValue(int v) { return v; }

absl::StatusOr<int> ReturnStatusOrError(std::string_view msg) {
  return absl::UnknownError(msg);
}

template <class... Args>
absl::StatusOr<std::tuple<Args...>> ReturnStatusOrTupleValue(Args&&... v) {
  return std::tuple<Args...>(std::forward<Args>(v)...);
}

template <class... Args>
absl::StatusOr<std::tuple<Args...>> ReturnStatusOrTupleError(
    std::string_view msg) {
  return absl::UnknownError(msg);
}

absl::StatusOr<std::unique_ptr<int>> ReturnStatusOrPtrValue(int v) {
  return std::make_unique<int>(v);
}

TEST(AssignOrReturn, Works) {
  auto func = []() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(int value1, ReturnStatusOrValue(1));
    EXPECT_EQ(1, value1);
    XLS_ASSIGN_OR_RETURN(const int value2, ReturnStatusOrValue(2));
    EXPECT_EQ(2, value2);
    XLS_ASSIGN_OR_RETURN(const int& value3, ReturnStatusOrValue(3));
    EXPECT_EQ(3, value3);
    XLS_ASSIGN_OR_RETURN(int value4, ReturnStatusOrError("EXPECTED"));
    (void)value4;  // Avoid unused error.
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, WorksWithCommasInType) {
  auto func = []() -> absl::Status {
    XLS_ASSIGN_OR_RETURN((std::tuple<int, int> t1),
                         ReturnStatusOrTupleValue(1, 1));
    EXPECT_EQ((std::tuple{1, 1}), t1);
    XLS_ASSIGN_OR_RETURN((const std::tuple<int, std::tuple<int, int>, int> t2),
                         ReturnStatusOrTupleValue(1, std::tuple{1, 1}, 1));
    EXPECT_EQ((std::tuple{1, std::tuple{1, 1}, 1}), t2);
    XLS_ASSIGN_OR_RETURN(
        (std::tuple<int, std::tuple<int, int>, int> t3),
        (ReturnStatusOrTupleError<int, std::tuple<int, int>, int>("EXPECTED")));
    t3 = {};  // fix unused error
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, WorksWithStructureBindings) {
  auto func = []() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(
        (const auto& [t1, t2, t3, t4, t5]),
        ReturnStatusOrTupleValue(std::tuple{1, 1}, 1, 2, 3, 4));
    EXPECT_EQ((std::tuple{1, 1}), t1);
    EXPECT_EQ(1, t2);
    EXPECT_EQ(2, t3);
    EXPECT_EQ(3, t4);
    EXPECT_EQ(4, t5);
    XLS_ASSIGN_OR_RETURN(int t6, ReturnStatusOrError("EXPECTED"));
    t6 = 0;
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, WorksWithParenthesesAndDereference) {
  auto func = []() -> absl::Status {
    int integer;
    int* pointer_to_integer = &integer;
    XLS_ASSIGN_OR_RETURN((*pointer_to_integer), ReturnStatusOrValue(1));
    EXPECT_EQ(1, integer);
    XLS_ASSIGN_OR_RETURN(*pointer_to_integer, ReturnStatusOrValue(2));
    EXPECT_EQ(2, integer);
    // Make the test where the order of dereference matters and treat the
    // parentheses.
    pointer_to_integer--;
    int** pointer_to_pointer_to_integer = &pointer_to_integer;
    XLS_ASSIGN_OR_RETURN((*pointer_to_pointer_to_integer)[1],
                         ReturnStatusOrValue(3));
    EXPECT_EQ(3, integer);
    XLS_ASSIGN_OR_RETURN(int t1, ReturnStatusOrError("EXPECTED"));
    (void)t1;
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, WorksWithAppend) {
  auto fail_test_if_called = []() -> std::string {
    ADD_FAILURE();
    return "FAILURE";
  };
  auto func = [&]() -> absl::Status {
    int value;
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrValue(1),
                         _ << fail_test_if_called());
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrError("EXPECTED A"),
                         _ << "EXPECTED B");
    (void)value;  // Avoid set-but-not-used error.
    return ReturnOk();
  };

  EXPECT_THAT(func().message(),
              AllOf(HasSubstr("EXPECTED A"), HasSubstr("EXPECTED B")));
}

TEST(AssignOrReturn, WorksWithAdaptorFunc) {
  auto fail_test_if_called = [](xabsl::StatusBuilder builder) {
    ADD_FAILURE();
    return builder;
  };
  auto adaptor = [](xabsl::StatusBuilder builder) {
    return builder << "EXPECTED B";
  };
  auto func = [&]() -> absl::Status {
    int value;
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrValue(1), fail_test_if_called(_));
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrError("EXPECTED A"), adaptor(_));
    (void)value;  // Avoid set-but-not-used error.
    return ReturnOk();
  };

  EXPECT_THAT(func().message(),
              AllOf(HasSubstr("EXPECTED A"), HasSubstr("EXPECTED B")));
}

TEST(AssignOrReturn, WorksWithThirdArgumentAndCommas) {
  auto fail_test_if_called = [](xabsl::StatusBuilder builder) {
    ADD_FAILURE();
    return builder;
  };
  auto adaptor = [](xabsl::StatusBuilder builder) {
    return builder << "EXPECTED B";
  };
  auto func = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN((const auto& [t1, t2, t3]),
                         ReturnStatusOrTupleValue(1, 2, 3),
                         fail_test_if_called(_));
    EXPECT_EQ(t1, 1);
    EXPECT_EQ(t2, 2);
    EXPECT_EQ(t3, 3);
    XLS_ASSIGN_OR_RETURN(
        (const auto& [t4, t5, t6]),
        (ReturnStatusOrTupleError<int, int, int>("EXPECTED A")), adaptor(_));
    // Silence errors about the unused values.
    static_cast<void>(t4);
    static_cast<void>(t5);
    static_cast<void>(t6);
    return ReturnOk();
  };

  EXPECT_THAT(func().message(),
              AllOf(HasSubstr("EXPECTED A"), HasSubstr("EXPECTED B")));
}

TEST(AssignOrReturn, WorksWithAppendIncludingLocals) {
  auto func = [&](const std::string& str) -> absl::Status {
    int value;
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrError("EXPECTED A"), _ << str);
    (void)value;
    return ReturnOk();
  };

  EXPECT_THAT(func("EXPECTED B").message(),
              AllOf(HasSubstr("EXPECTED A"), HasSubstr("EXPECTED B")));
}

TEST(AssignOrReturn, WorksForExistingVariable) {
  auto func = []() -> absl::Status {
    int value = 1;
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrValue(2));
    EXPECT_EQ(2, value);
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrValue(3));
    EXPECT_EQ(3, value);
    XLS_ASSIGN_OR_RETURN(value, ReturnStatusOrError("EXPECTED"));
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, UniquePtrWorks) {
  auto func = []() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<int> ptr, ReturnStatusOrPtrValue(1));
    EXPECT_EQ(*ptr, 1);
    return ReturnError("EXPECTED");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, UniquePtrWorksForExistingVariable) {
  auto func = []() -> absl::Status {
    std::unique_ptr<int> ptr;
    XLS_ASSIGN_OR_RETURN(ptr, ReturnStatusOrPtrValue(1));
    EXPECT_EQ(*ptr, 1);

    XLS_ASSIGN_OR_RETURN(ptr, ReturnStatusOrPtrValue(2));
    EXPECT_EQ(*ptr, 2);
    return ReturnError("EXPECTED");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

namespace assign_or_return {
absl::Status CallThree() { return absl::InternalError("foobar"); }
absl::StatusOr<int> CallTwo() {
  XLS_RETURN_IF_ERROR(CallThree());
  return 1;
}
absl::Status CallOne() {
  XLS_ASSIGN_OR_RETURN(auto abc, CallTwo());
  (void)abc;
  return absl::OkStatus();
}
}  // namespace assign_or_return
TEST(AssignOrReturn, KeepsBackTrace) {
#ifndef XLS_USE_ABSL_SOURCE_LOCATION
  GTEST_SKIP() << "Back trace not recorded";
#endif
  auto result = assign_or_return::CallOne();
  RecordProperty("result", testing::PrintToString(result));
  EXPECT_THAT(result.message(), Eq("foobar"));
  // Expect 3 lines in the stack trace with status_macros_test.cc in them for
  // the three deep call stack.
  EXPECT_THAT(
      result.ToString(absl::StatusToStringMode::kWithEverything),
      ContainsRegex(
          "(.*xls/common/status/status_macros_test.cc:[0-9]+.*\n?){3}"));
}

namespace assign_or_return_with_message {
absl::Status CallThree() { return absl::InternalError("Clap"); }
absl::StatusOr<int> CallTwo() {
  XLS_RETURN_IF_ERROR(CallThree()) << "Your";
  return 1;
}
absl::Status CallOne() {
  XLS_ASSIGN_OR_RETURN(auto abc, CallTwo(), _ << "Hands");
  (void)abc;
  return absl::OkStatus();
}
}  // namespace assign_or_return_with_message
TEST(AssignOrReturn, KeepsBackTraceWithMessage) {
#ifndef XLS_USE_ABSL_SOURCE_LOCATION
  GTEST_SKIP() << "Back trace not recorded";
#endif
  auto result = assign_or_return_with_message::CallOne();
  RecordProperty("result", testing::PrintToString(result));
  EXPECT_THAT(result.message(), Eq("Clap; Your; Hands"));
  // Expect 3 lines in the stack trace with status_macros_test.cc in them for
  // the three deep call stack.
  EXPECT_THAT(
      result.ToString(absl::StatusToStringMode::kWithEverything),
      ContainsRegex(
          "(.*xls/common/status/status_macros_test.cc:[0-9]+.*\n?){3}"));
}

TEST(ReturnIfError, Works) {
  auto func = []() -> absl::Status {
    XLS_RETURN_IF_ERROR(ReturnOk());
    XLS_RETURN_IF_ERROR(ReturnOk());
    XLS_RETURN_IF_ERROR(ReturnError("EXPECTED"));
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(ReturnIfError, WorksWithBuilder) {
  auto func = []() -> absl::Status {
    XLS_RETURN_IF_ERROR(ReturnOkBuilder());
    XLS_RETURN_IF_ERROR(ReturnOkBuilder());
    XLS_RETURN_IF_ERROR(ReturnErrorBuilder("EXPECTED"));
    return ReturnErrorBuilder("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(ReturnIfError, WorksWithLambda) {
  auto func = []() -> absl::Status {
    XLS_RETURN_IF_ERROR([] { return ReturnOk(); }());
    XLS_RETURN_IF_ERROR([] { return ReturnError("EXPECTED"); }());
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(ReturnIfError, WorksWithAppend) {
  auto fail_test_if_called = []() -> std::string {
    ADD_FAILURE();
    return "FAILURE";
  };
  auto func = [&]() -> absl::Status {
    XLS_RETURN_IF_ERROR(ReturnOk()) << fail_test_if_called();
    XLS_RETURN_IF_ERROR(ReturnError("EXPECTED A")) << "EXPECTED B";
    return absl::OkStatus();
  };

  EXPECT_THAT(func().message(),
              AllOf(HasSubstr("EXPECTED A"), HasSubstr("EXPECTED B")));
}

TEST(ReturnIfError, WorksWithVoidReturnAdaptor) {
  int code = 0;
  int phase = 0;
  auto adaptor = [&](absl::Status status) -> void { code = phase; };
  auto func = [&]() -> void {
    phase = 1;
    XLS_RETURN_IF_ERROR(ReturnOk()).With(adaptor);
    phase = 2;
    XLS_RETURN_IF_ERROR(ReturnError("EXPECTED A")).With(adaptor);
    phase = 3;
  };

  func();
  EXPECT_EQ(phase, 2);
  EXPECT_EQ(code, 2);
}

namespace return_if_error {
absl::Status CallThree() { return absl::InternalError("foobar"); }
absl::Status CallTwo() {
  XLS_RETURN_IF_ERROR(CallThree());
  return absl::OkStatus();
}
absl::Status CallOne() {
  XLS_RETURN_IF_ERROR(CallTwo());
  return absl::OkStatus();
}
}  // namespace return_if_error

TEST(ReturnIfError, KeepsBackTrace) {
#ifndef XLS_USE_ABSL_SOURCE_LOCATION
  GTEST_SKIP() << "Back trace not recorded";
#endif
  auto result = return_if_error::CallOne();
  RecordProperty("result", testing::PrintToString(result));
  EXPECT_THAT(result.message(), Eq("foobar"));
  // Expect 3 lines in the stack trace with status_macros_test.cc in them for
  // the three deep call stack.
  EXPECT_THAT(
      result.ToString(absl::StatusToStringMode::kWithEverything),
      ContainsRegex(
          "(.*xls/common/status/status_macros_test.cc:[0-9]+.*\n?){3}"));
}

namespace return_if_error_with_message {
absl::Status CallThree() { return absl::InternalError("Clap"); }
absl::Status CallTwo() {
  XLS_RETURN_IF_ERROR(CallThree()) << "Your";
  return absl::OkStatus();
}
absl::Status CallOne() {
  XLS_RETURN_IF_ERROR(CallTwo()) << "Hands";
  return absl::OkStatus();
}
}  // namespace return_if_error_with_message

TEST(ReturnIfError, KeepsBackTraceWithMessage) {
#ifndef XLS_USE_ABSL_SOURCE_LOCATION
  GTEST_SKIP() << "Back trace not recorded";
#endif
  auto result = return_if_error_with_message::CallOne();
  RecordProperty("result", testing::PrintToString(result));
  EXPECT_THAT(result.message(), Eq("Clap; Your; Hands"));
  // Expect 3 lines in the stack trace with status_macros_test.cc in them for
  // the three deep call stack.
  EXPECT_THAT(
      result.ToString(absl::StatusToStringMode::kWithEverything),
      ContainsRegex(
          "(.*xls/common/status/status_macros_test.cc:[0-9]+.*\n?){3}"));
}

// Basis for XLS_RETURN_IF_ERROR and XLS_ASSIGN_OR_RETURN benchmarks.  Derived
// classes override LoopAgain() with the macro invocation(s).
template <class T>
class ReturnLoop {
 public:
  using ReturnType = T;

  explicit ReturnLoop(ReturnType return_value)
      : value_(std::move(return_value)) {}
  virtual ~ReturnLoop() = default;

  ReturnType Loop(size_t* ops) {
    if (*ops == 0) {
      return value_;
    }
    // LoopAgain is virtual, with the intent that this defeats tail
    // recursion optimization.
    return LoopAgain(ops);
  }

  ReturnType return_value() { return value_; }

 private:
  virtual ReturnType LoopAgain(size_t* ops) = 0;

  const ReturnType value_;
};

class ReturnIfErrorLoop : public ReturnLoop<absl::Status> {
 public:
  explicit ReturnIfErrorLoop(absl::Status return_value)
      : ReturnLoop(std::move(return_value)) {}

 private:
  absl::Status LoopAgain(size_t* ops) override {
    --*ops;
    XLS_RETURN_IF_ERROR(Loop(ops));
    return absl::OkStatus();
  }
};

class ReturnIfErrorWithAnnotateLoop : public ReturnLoop<absl::Status> {
 public:
  explicit ReturnIfErrorWithAnnotateLoop(absl::Status return_value)
      : ReturnLoop(std::move(return_value)) {}

 private:
  absl::Status LoopAgain(size_t* ops) override {
    --*ops;
    XLS_RETURN_IF_ERROR(Loop(ops))
        << "The quick brown fox jumped over the lazy dog.";
    return absl::OkStatus();
  }
};

class AssignOrReturnLoop : public ReturnLoop<absl::StatusOr<int>> {
 public:
  explicit AssignOrReturnLoop(ReturnType return_value)
      : ReturnLoop(std::move(return_value)) {}

 private:
  ReturnType LoopAgain(size_t* ops) override {
    --*ops;
    XLS_ASSIGN_OR_RETURN(int result, Loop(ops));
    return result;
  }

  ReturnType result_;
};

class AssignOrReturnAnnotateLoop : public ReturnLoop<absl::StatusOr<int>> {
 public:
  explicit AssignOrReturnAnnotateLoop(ReturnType return_value)
      : ReturnLoop(std::move(return_value)) {}

 private:
  ReturnType LoopAgain(size_t* ops) override {
    --*ops;
    XLS_ASSIGN_OR_RETURN(int result, Loop(ops),
                         _ << "The quick brown fox jumped over the lazy dog.");
    return result;
  }

  ReturnType result_;
};

}  // namespace
