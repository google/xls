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

#ifndef XLS_FUZZER_IR_FUZZER_REPRODUCER_REPO_FILES_H_
#define XLS_FUZZER_IR_FUZZER_REPRODUCER_REPO_FILES_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
namespace xls {

// Does this path look like a reference to a well-known fuzztest reproduction
// repository.
bool IsFuzztestReproPath(std::string_view path);
// Get the actual file-path a fuzztest reproduction repository path refers to.
absl::StatusOr<std::string> FuzztestRepoToFilePath(std::string_view path);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_REPRODUCER_REPO_FILES_H_
