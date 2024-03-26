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

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/solvers/z3_ir_translator.h"

ABSL_FLAG(std::string, subject, "",
          "Node that is subject of the proof; default: return value");
ABSL_FLAG(std::string, kind, "eq_zero",
          "Predicate to attempt to prove; choices: eq_zero, ne_zero, eq_node");
ABSL_FLAG(std::string, other, "",
          "Node for comparison; e.g. when kind is eq_node");
ABSL_FLAG(int64_t, timeout_ms, 60000,
          "Timeout for proof attempt, in milliseconds");

const char kUsage[] = R"(
Attempts to prove a property of a node in an XLS IR entry function within a
user-specified timeout (given by -timeout_ms).

Example invocations:

Prove that node and.1234 in the entry function is always equal to zero:

  solver /tmp/my.ir -subject and.1234 -kind eq_zero

Prove that node and.1234 is equivalent to and.2345:

  solver /tmp/my.ir -subject and.1234 -kind eq_node -other and.2345
)";

namespace xls {
namespace {

using solvers::z3::Predicate;

absl::Status RealMain(std::string_view ir_path,
                      std::string_view subject_node_name,
                      std::string_view predicate_kind,
                      std::string_view other_node_name, int64_t timeout_ms) {
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(contents, ir_path));
  XLS_ASSIGN_OR_RETURN(Function * f, package->GetTopAsFunction());
  XLS_ASSIGN_OR_RETURN(Node * subject, f->GetNode(subject_node_name));
  absl::Duration timeout = absl::Milliseconds(timeout_ms);

  std::optional<Predicate> predicate;
  if (predicate_kind == "eq_zero") {
    predicate = Predicate::EqualToZero();
  } else if (predicate_kind == "ne_zero") {
    predicate = Predicate::NotEqualToZero();
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid predicate kind: \"%s\"", predicate_kind));
  }

  XLS_ASSIGN_OR_RETURN(
      bool proved,
      solvers::z3::TryProve(f, subject, predicate.value(), timeout));
  std::cout << "Proved " << subject_node_name << " " << predicate->ToString()
            << " holds for all input?"
            << ": " << (proved ? "true" : "false") << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1 || positional_arguments[0].empty() ||
      absl::GetFlag(FLAGS_subject).empty()) {
    LOG(QFATAL) << absl::StreamFormat(
        "Expected invocation:\n  %s <path> -subject <node>\n", argv[0]);
  }

  return xls::ExitStatus(
      xls::RealMain(positional_arguments[0], absl::GetFlag(FLAGS_subject),
                    absl::GetFlag(FLAGS_kind), absl::GetFlag(FLAGS_other),
                    absl::GetFlag(FLAGS_timeout_ms)));
}
