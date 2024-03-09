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

#include "xls/dslx/run_routines/test_xml.h"

#include <memory>
#include <string>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/time/time.h"
#include "xls/common/indent.h"

namespace xls::dslx::test_xml {

std::string_view ToXmlString(RunStatus rs) {
  switch (rs) {
    case RunStatus::kNotRun:
      return "notrun";
    case RunStatus::kRun:
      return "run";
  }
}

std::string_view ToXmlString(RunResult rr) {
  switch (rr) {
    case RunResult::kCompleted:
      return "completed";
    case RunResult::kInterrupted:
      return "interrupted";
    case RunResult::kCancelled:
      return "cancelled";
    case RunResult::kFiltered:
      return "filtered";
    case RunResult::kSkipped:
      return "skipped";
    case RunResult::kSuppressed:
      return "suppressed";
    case RunResult::kSilenced:
      return "silenced";
  }
}

template <typename T>
static void TimesToAttrs(XmlNode& node, const T& item, absl::TimeZone tz) {
  node.attrs["time"] =
      absl::StrFormat("%.03f", absl::ToDoubleSeconds(item.time));
  node.attrs["timestamp"] = absl::FormatTime(item.timestamp, tz);
}

static void CountsToAttrs(XmlNode& node, const TestCounts& counts) {
  node.attrs["tests"] = absl::StrCat(counts.tests);
  node.attrs["failures"] = absl::StrCat(counts.failures);
  node.attrs["disabled"] = absl::StrCat(counts.disabled);
  node.attrs["skipped"] = absl::StrCat(counts.skipped);
  node.attrs["errors"] = absl::StrCat(counts.errors);
}

static std::unique_ptr<XmlNode> ToXml(const Failure& failure) {
  auto node = std::make_unique<XmlNode>("failure");
  node->attrs["message"] = failure.message;

  // TODO(leary): 2024-02-08 Handle expected/actual value reporting.

  return node;
}

std::unique_ptr<XmlNode> ToXml(const TestCase& test_case, absl::TimeZone tz) {
  auto node = std::make_unique<XmlNode>("testcase");
  node->attrs["name"] = test_case.name;
  node->attrs["file"] = test_case.file;
  node->attrs["line"] = absl::StrCat(test_case.line);
  node->attrs["status"] = std::string(ToXmlString(test_case.status));
  node->attrs["result"] = std::string(ToXmlString(test_case.result));

  if (test_case.failure.has_value()) {
    node->children.push_back(ToXml(test_case.failure.value()));
  }

  TimesToAttrs(*node, test_case, tz);
  return node;
}

std::unique_ptr<XmlNode> ToXml(const TestSuite& suite, absl::TimeZone tz) {
  auto node = std::make_unique<XmlNode>("testsuite");
  node->attrs["name"] = suite.name;
  CountsToAttrs(*node, suite.counts);
  TimesToAttrs(*node, suite, tz);
  for (const TestCase& test_case : suite.test_cases) {
    node->children.push_back(ToXml(test_case, tz));
  }
  return node;
}

std::unique_ptr<XmlNode> ToXml(const TestSuites& suites, absl::TimeZone tz) {
  auto node = std::make_unique<XmlNode>("testsuites");

  node->attrs["name"] = "all tests";

  CountsToAttrs(*node, suites.counts);
  TimesToAttrs(*node, suites, tz);

  for (const TestSuite& suite : suites.test_suites) {
    node->children.push_back(ToXml(suite, tz));
  }
  return node;
}

static std::string SimpleXmlEscape(std::string_view value) {
  return absl::StrReplaceAll(value, {
                                        {"<", "&lt;"},
                                        {">", "&gt;"},
                                        {"\"", "&quot;"},
                                        {"'", "&apos;"},
                                        {"&", "&amp;"},
                                        {"\n", "&#10;"},
                                        {"\t", "&#9;"},
                                    });
}

std::string XmlNodeToString(const XmlNode& root) {
  std::string attrs =
      absl::StrJoin(root.attrs, " ", [](std::string* out, const auto& kv) {
        const auto& [key, value] = kv;
        absl::StrAppendFormat(out, "%s=\"%s\"", key, SimpleXmlEscape(value));
      });
  if (root.children.empty()) {
    return absl::StrFormat("<%s %s />", root.tag, attrs);
  }
  std::string children = absl::StrJoin(
      root.children, "\n  ", [](std::string* out, const auto& child_node) {
        absl::StrAppend(out, xls::Indent(XmlNodeToString(*child_node)));
      });
  return absl::StrFormat("<%s %s>\n  %s\n</%s>", root.tag, attrs, children,
                         root.tag);
}

std::string XmlRootToString(const XmlNode& root) {
  return absl::StrCat("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
                      XmlNodeToString(root));
}

}  // namespace xls::dslx::test_xml
