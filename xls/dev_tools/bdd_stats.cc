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

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/examples/sample_packages.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/passes/bdd_query_engine.h"

static constexpr std::string_view kUsage = R"(
Builds a BDD from XLS IR and prints various metrics about the BDD. Usage:

To gather BDD stats of an IR file:
   bdd_stats <ir_file>

To gather BDD stats of a set of benchmarks:
   bdd_stats --benchmarks=sha256,crc32
   bdd_stats --benchmarks=all
)";

ABSL_FLAG(int64_t, bdd_path_limit, 0,
          "Maximum number of paths before truncating the BDD subgraph "
          "and declaring a new variable. If zero, then no limit.");
ABSL_FLAG(std::vector<std::string>, benchmarks, {},
          "Comma-separated list of benchmarks gather BDD stats about.");

namespace xls {
namespace {

// Return list of pairs of {name, Package} for the specified bechmarks.
absl::StatusOr<std::vector<std::pair<std::string, std::unique_ptr<Package>>>>
GetBenchmarks(absl::Span<const std::string> benchmark_names) {
  std::vector<std::pair<std::string, std::unique_ptr<Package>>> packages;
  std::vector<std::string> names;
  if (benchmark_names.size() == 1 && benchmark_names.front() == "all") {
    XLS_ASSIGN_OR_RETURN(names, sample_packages::GetBenchmarkNames());
  } else {
    names = std::vector<std::string>(benchmark_names.begin(),
                                     benchmark_names.end());
  }
  for (const std::string& name : names) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<Package> package,
        sample_packages::GetBenchmark(name, /*optimized=*/true));
    packages.push_back({name, std::move(package)});
  }
  return packages;
}

absl::Status RealMain(std::string_view input_path) {
  std::vector<std::pair<std::string, std::unique_ptr<Package>>> packages;
  if (absl::GetFlag(FLAGS_benchmarks).empty()) {
    QCHECK(!input_path.empty());
    std::string path;
    if (input_path == "-") {
      path = "/dev/stdin";
    } else {
      path = std::string(input_path);
    }
    XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(path));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                         Parser::ParsePackage(contents, path));
    packages.push_back({path, std::move(package)});
  } else {
    XLS_ASSIGN_OR_RETURN(packages,
                         GetBenchmarks(absl::GetFlag(FLAGS_benchmarks)));
  }

  absl::Duration total_time;
  for (const auto& pair : packages) {
    const std::string& name = pair.first;
    const auto& package = pair.second;
    if (packages.size() > 1) {
      // Use endl to flush cout so the banner appears before starting work on
      // the BDD.
      std::cout << "================== " << name << '\n';
    }
    std::optional<FunctionBase*> top = package->GetTop();
    if (!top.has_value()) {
      return absl::InternalError(absl::StrFormat(
          "Top entity not set for package: %s.", package->name()));
    }
    BddQueryEngine query_engine(absl::GetFlag(FLAGS_bdd_path_limit));
    Stopwatch bdd_stopwatch;
    XLS_RETURN_IF_ERROR(query_engine.EagerlyPopulate(top.value()));
    absl::Duration bdd_time = bdd_stopwatch.GetElapsedTime();
    total_time += bdd_time;
    std::cout << "BDD construction time: " << bdd_time << "\n";
    std::cout << "BDD node count: " << query_engine.bdd().size() << "\n";
    std::cout << "BDD variable count: " << query_engine.bdd().variable_count()
              << "\n";

    int64_t number_bits = 0;
    for (Node* node : top.value()->nodes()) {
      number_bits += node->GetType()->GetFlatBitCount();
    }
    std::cout << "Bits in graph: " << number_bits << "\n";

    int64_t max_paths = 0;
    for (int64_t i = 0; i < query_engine.bdd().size(); ++i) {
      max_paths =
          std::max(max_paths, query_engine.bdd().path_count(BddNodeIndex(i)));
    }
    if (max_paths == std::numeric_limits<int32_t>::max()) {
      std::cout << "Maximum paths of any expression: INT32_MAX\n";
    } else {
      std::cout << "Maximum paths of any expression: " << max_paths << "\n";
    }
  }

  if (packages.size() > 1) {
    std::cout << "\nTotal construction time: " << total_time << "\n";
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.empty() && absl::GetFlag(FLAGS_benchmarks).empty()) {
    LOG(QFATAL) << absl::StreamFormat(
        "Expected invocation:\n  %s <path>\n  %s "
        "--benchmarks=<benchmark-names>",
        argv[0], argv[0]);
  }

  return xls::ExitStatus(xls::RealMain(positional_arguments[0]));
}
