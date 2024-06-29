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

#ifndef XLS_DSLX_TEST_XML_H_
#define XLS_DSLX_TEST_XML_H_

// Simple layer for building XML-serializable objects that test-reporting
// infrastructure wants.
//
// NOTE: The C++ objects in this file correspond as closely as possible to the
// schema we want to emit in XML, so sometimes fields will be less idiomatic
// than usual (e.g.  naming a field "time" where a duration is held).

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace xls::dslx::test_xml {

class XmlNode {
 public:
  explicit XmlNode(std::string tag) : tag_(std::move(tag)) {}

  std::string_view tag() const { return tag_; }
  absl::btree_map<std::string, std::string>& attrs() { return attrs_; }
  const absl::btree_map<std::string, std::string>& attrs() const {
    return attrs_;
  }
  std::vector<std::unique_ptr<XmlNode>>& children() { return children_; }
  absl::Span<const std::unique_ptr<XmlNode>> children() const {
    return absl::MakeConstSpan(children_);
  }

 private:
  const std::string tag_;
  // btree map as a simple way to get stable ordering on output
  absl::btree_map<std::string, std::string> attrs_;
  std::vector<std::unique_ptr<XmlNode>> children_;
};

// Schema recommends using the 'result' attribute to provide more details.
enum class RunStatus : uint8_t {
  kNotRun,
  kRun,
};

std::string_view ToXmlString(RunStatus rs);

enum class RunResult : uint8_t {
  // Test that either a) passed or b) failed due to the test indicating a
  // failure.
  kCompleted,

  // Started by not completed because a signal was received and runner decided
  // to stop running tests.
  kInterrupted,

  // Not started because test harness run interrupted by a signal or timed out.
  kCancelled,

  // User or process running the test specified that it should be filtered out
  // of the test run.
  //
  // Harness may choose to instead not emit testcase elements for tests that do
  // not match a user-specified filter.
  kFiltered,

  // The test itself decided it should not be run (e.g. for tests that return a
  // skipped status, like `GTEST_SKIP()` equivalent.)
  //
  // Harness may choose to use "completed" and mark the test
  // case as passed, or may choose to not emit testcase elements for skipped
  // tests.
  kSkipped,

  // Test framework did not run any part of the test case (including
  // setup/teardown) because test was albeled in the code as being suppressed.
  //
  // (leary@ aside: is this like GTEST's DISABLED_ prefix? I think it's more
  // like a "known failure" annotation so things could
  // try-to-run-and-enable-when-not-broken as mentioned below.)
  //
  // Test harnesses may choose to instead not emit testcase elements for tests
  // that are suppressed.
  kSuppressed,

  // The test framework ran this test case but with the assumption that a
  // failure does not cause the suite to fail.
  //
  // Example: a test which will pass in the future is "silenced". Once it is
  // observed to be passing an automated system can remove the annotation,
  // effectively promoting the case to being in use.
  //
  // Harnesses may choose to not support silenced tests, and instead mark all
  // failing tests as completed.
  kSilenced,
};

std::string_view ToXmlString(RunResult rr);

struct Values {
  std::vector<std::string> value;
};

// Represents a violated assertion.
struct Failure {
  // Typically one line.
  std::string message;

  // Pairs of (expected, actual).
  std::vector<std::pair<Values, Values>> expected_actuals;
};

// A single test /case/ that was run (in a test suite).
struct TestCase {
  // Name of the test case, e.g. method.
  std::string name;

  // Starting position where the test case is located.
  std::string file;
  int64_t line;

  // Note: use the "result" attribute to provide more details.
  RunStatus status;

  // What happened when the runner tried to execute the given testcase.
  RunResult result;

  // Elapsed time for the test case to run.
  absl::Duration time;

  // Timestamp the test case started.
  absl::Time timestamp;

  // Failure reporting, if this test case failed.
  std::optional<Failure> failure;
};

std::unique_ptr<XmlNode> ToXml(const TestCase& test_case, absl::TimeZone tz);

struct TestCounts {
  int64_t tests;
  int64_t failures;
  int64_t disabled;
  int64_t skipped;
  int64_t errors;
};

// A single test /suite/ that was run.
struct TestSuite {
  std::string name;
  TestCounts counts;

  // How long the test suite took.
  absl::Duration time;

  // When the test suite started.
  absl::Time timestamp;

  // Test cases that were part of this test suite.
  std::vector<TestCase> test_cases;
};

std::unique_ptr<XmlNode> ToXml(const TestSuite& suite, absl::TimeZone tz);

// Top-level object that contains all the test suites for a given test running
// session.
struct TestSuites {
  TestCounts counts;

  // How long the test suites took.
  absl::Duration time;

  // When the test suites started.
  absl::Time timestamp;

  // Individual information on various suites that were run.
  std::vector<TestSuite> test_suites;
};

std::unique_ptr<XmlNode> ToXml(const TestSuites& suites, absl::TimeZone tz);

std::string XmlNodeToString(const XmlNode& root);

// As above, but puts a DTD as the first line.
std::string XmlRootToString(const XmlNode& root);

}  // namespace xls::dslx::test_xml

#endif  // XLS_DSLX_TEST_XML_H_
