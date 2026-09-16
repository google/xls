// Copyright 2026 The XLS Authors
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

#ifndef XLS_COMMON_SUBPROCESS_FOR_OS_H_
#define XLS_COMMON_SUBPROCESS_FOR_OS_H_

#include <spawn.h>
#include <sys/types.h>

#include <filesystem>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xls::internal {

// Starts a child with the given arguments, environment, and optional working
// directory. argv must contain a non-null command followed by a null-terminated
// argument list; envp must also be null-terminated. Bare command names are
// looked up using PATH in envp, relative to the child's working directory.
//
// file_actions must be initialized by the caller, which retains ownership. The
// implementation may append actions. On success, the caller must reap the PID.
// Uses posix_spawn rather than fork/exec so the parent can safely be
// multithreaded.
absl::StatusOr<pid_t> SpawnSubprocess(
    absl::Span<const char* const> argv,
    const std::optional<std::filesystem::path>& cwd,
    posix_spawn_file_actions_t* file_actions, char* const* envp);

}  // namespace xls::internal

#endif  // XLS_COMMON_SUBPROCESS_FOR_OS_H_
