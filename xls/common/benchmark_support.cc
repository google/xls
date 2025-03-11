// Copyright 2023 The XLS Authors
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

#include "xls/common/benchmark_support.h"

#include <optional>
#include <string>

#include "benchmark/benchmark.h"
#include "absl/flags/flag.h"

// Workaround absl and benchmark:: fighting for flags.
// TODO(https://github.com/google/xls/issues/1150): 2023-11-10 We should try to
// remove this workaround at some point.
ABSL_FLAG(std::optional<std::string>, benchmark_filter, std::nullopt,
          "regex to select benchmarks to run");

namespace xls {

void RunSpecifiedBenchmarks(std::optional<std::string> default_spec) {
  // Match BENCHMARK_MAIN macro definition of the benchmark MAIN function.
  // We don't want to run benchmarks unless some are actually requested and we
  // want to always run gtests so we can't use the standard BENCHMARK_MAIN
  // macro. This replicates the parts that configure benchmarks and run them
  // with our filters.
  int fake_argc = 1;
  char fake_argv[] = "benchmark";
  char* fake_argv_ptr = &fake_argv[0];
  benchmark::Initialize(&fake_argc, &fake_argv_ptr);

  // Only run benchmarks if requested.
  std::optional<std::string> filter = absl::GetFlag(FLAGS_benchmark_filter);
  if (!filter.has_value()) {
    filter = default_spec;
  }
  if (filter.has_value()) {
    benchmark::SetBenchmarkFilter(*filter);
    benchmark::RunSpecifiedBenchmarks();
  }
  benchmark::Shutdown();
}

}  // namespace xls
