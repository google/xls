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

#ifndef XLS_COMMON_BENCHMARK_SUPPORT_H_
#define XLS_COMMON_BENCHMARK_SUPPORT_H_

#include <optional>
#include <string>

namespace xls {

// Helper to perform setup and execute benchmarks. Not a public API.
// This should be called after parsing all flags and setting up googletest.
void RunSpecifiedBenchmarks(
    std::optional<std::string> default_spec = std::nullopt);

}  // namespace xls

#endif  // XLS_COMMON_BENCHMARK_SUPPORT_H_
