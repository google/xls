// Copyright 2025 The XLS Authors
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

#include "xls/fuzzer/ir_fuzzer/reproducer_repo_files.h"

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls {

// Does this path look like a reference to a well-known fuzztest reproduction
// repository.
bool IsFuzztestReproPath(std::string_view path) { return false; }
// Get the actual file-path a fuzztest reproduction repository path refers to.
absl::StatusOr<std::string> FuzztestRepoToFilePath(std::string_view path) {
  return absl::UnimplementedError("Fuzztest repo not supported.");
}
}  // namespace xls
