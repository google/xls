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

#include "xls/dslx/test_xml.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/civil_time.h"
#include "absl/time/time.h"

namespace xls::dslx::test_xml {
namespace {

TEST(TestResultSerdesTest, OneTestCase) {
  const absl::TimeZone la = absl::UTCTimeZone();

  // For some reason clang-format has trouble keeping this in 80 column lines.
  // clang-format off
  const TestSuites suites = {
    .counts = TestCounts{
      .tests = 1,
      .failures = 0,
      .disabled = 0,
      .skipped = 0,
      .errors = 0,
    },
    .time = absl::Milliseconds(42),
    .timestamp = absl::FromCivil(
        absl::CivilSecond(2024, 1, 1, 0, 0, 0), la),
    .test_suites = {
      TestSuite{
        .name = "MyTestSuite",
        .counts = TestCounts{
          .tests = 1,
          .failures = 0,
          .disabled = 0,
          .skipped = 0,
          .errors = 0,
        },
        .time = absl::Milliseconds(32),
        .timestamp = absl::FromCivil(
            absl::CivilSecond(2024, 1, 1, 0, 0, 1), la),
        .test_cases = {
            TestCase{
              .name = "MyTestCase",
              .file = "/path/to/foo.x",
              .line = 64,
              .status = RunStatus::kRun,
              .result = RunResult::kCompleted,
              .time = absl::Milliseconds(24),
              .timestamp = absl::FromCivil(
                  absl::CivilSecond(2024, 1, 1, 0, 0, 2), la),
            },
            TestCase{
              .name = "MyFailingTestCase",
              .file = "/path/to/foo.x",
              .line = 128,
              .status = RunStatus::kRun,
              .result = RunResult::kCompleted,
              .time = absl::Milliseconds(24),
              .timestamp = absl::FromCivil(
                  absl::CivilSecond(2024, 1, 1, 0, 0, 3), la),
              .failure = Failure{
                  "this is a one liner describing the failure with angle "
                  "brackets in it for fun <> they should be escaped",
               },
            },
         },
       }
    },
  };
  // clang-format on

  std::unique_ptr<XmlNode> node = ToXml(suites, la);
  std::string got = XmlNodeToString(*node);
  EXPECT_EQ(
      got,
      R"(<testsuites disabled="0" errors="0" failures="0" name="all tests" skipped="0" tests="1" time="0.042" timestamp="2024-01-01T00:00:00+00:00">
    <testsuite disabled="0" errors="0" failures="0" name="MyTestSuite" skipped="0" tests="1" time="0.032" timestamp="2024-01-01T00:00:01+00:00">
      <testcase file="/path/to/foo.x" line="64" name="MyTestCase" result="completed" status="run" time="0.024" timestamp="2024-01-01T00:00:02+00:00" />
      <testcase file="/path/to/foo.x" line="128" name="MyFailingTestCase" result="completed" status="run" time="0.024" timestamp="2024-01-01T00:00:03+00:00">
        <failure message="this is a one liner describing the failure with angle brackets in it for fun &lt;&gt; they should be escaped" />
    </testcase>
  </testsuite>
</testsuites>)");

  EXPECT_THAT(XmlRootToString(*node),
              testing::AllOf(testing::StartsWith(
                                 "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"),
                             testing::HasSubstr(got)));
}

}  // namespace
}  // namespace xls::dslx::test_xml
