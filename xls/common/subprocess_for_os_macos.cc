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

#include <paths.h>
#include <spawn.h>
#include <sys/types.h>

#include <cerrno>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xls/common/strerror.h"
#include "xls/common/subprocess_for_os.h"

namespace xls::internal {
namespace {

// macOS 26 introduced the standard addchdir spelling and deprecated _np, which
// has been available since macOS 10.15. SDK availability and the deployment
// target are separate: a new SDK can build a binary for an older runtime.
// Both APIs return zero on success or an error number directly.
int AddChdirFileAction(posix_spawn_file_actions_t* file_actions,
                       const char* cwd) {
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_26_0
  // The minimum deployment target guarantees that the new API exists.
  return posix_spawn_file_actions_addchdir(file_actions, cwd);
#else
// An older SDK lacks the new declaration, even inside a runtime check.
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  // __builtin_available is Clang's C/C++ runtime availability check. For older
  // deployment targets, the SDK annotation makes the new function a weak
  // import; this check prevents calling it on an OS that lacks it. The required
  // '*' covers unlisted platforms; Bazel selects this file only for macOS.
  // https://clang.llvm.org/docs/LanguageExtensions.html#objective-c-available
  if (__builtin_available(macOS 26.0, *)) {
    return posix_spawn_file_actions_addchdir(file_actions, cwd);
  }
#endif
  // Use the macOS 10.15 API with older SDKs or on pre-26 runtimes.
  return posix_spawn_file_actions_addchdir_np(file_actions, cwd);
#endif
}

// Returns zero with *pid set on success, or a POSIX error number on failure.
int SpawnExecutable(pid_t* pid, const char* executable,
                    absl::Span<const char* const> argv,
                    const posix_spawn_file_actions_t* file_actions,
                    char* const* envp) {
  int err;
  // posix_spawn reports the error number in its return value; errno need not
  // match. Defensively retry EINTR (an interrupted operation).
  do {
    err = posix_spawn(pid, executable, file_actions, nullptr,
                      const_cast<char* const*>(argv.data()), envp);
  } while (err == EINTR);
  // ENOEXEC means the executable format was not recognized. Only that error
  // selects the shell fallback; preserve success and all other errors for the
  // caller, which may try another PATH candidate.
  if (err != ENOEXEC) {
    return err;
  }

  // Match execvp in the Linux helper: executable text without a shebang is
  // interpreted by the shell. posix_spawn does not provide this fallback.
  std::vector<const char*> shell_argv = {"/bin/sh", executable};
  shell_argv.insert(shell_argv.end(), argv.begin() + 1, argv.end());
  // Apply the same interrupted-operation retry when spawning the shell.
  do {
    err = posix_spawn(pid, "/bin/sh", file_actions, nullptr,
                      const_cast<char* const*>(shell_argv.data()), envp);
  } while (err == EINTR);
  return err;
}

int SpawnWithPathSearch(pid_t* pid, absl::Span<const char* const> argv,
                        const posix_spawn_file_actions_t* file_actions,
                        char* const* envp) {
  std::string_view executable = argv.front();
  if (executable.empty()) {
    // An empty command cannot name an executable; report "not found" without
    // constructing PATH candidates that would name directories instead.
    return ENOENT;
  }
  if (absl::StrContains(executable, '/')) {
    return SpawnExecutable(pid, argv.front(), argv, file_actions, envp);
  }

  std::string_view path = _PATH_DEFPATH;
  for (char* const* entry = envp; *entry != nullptr; ++entry) {
    std::string_view variable = *entry;
    if (absl::ConsumePrefix(&variable, "PATH=")) {
      path = variable;
      break;
    }
  }

  // Darwin's posix_spawnp searches the parent's PATH and can return ENOENT even
  // after launching a child when a chdir action and relative PATH are combined.
  // Search the child's PATH explicitly with posix_spawn instead. Relative and
  // empty entries must resolve after the child's chdir, not in the parent.
  bool saw_eacces = false;
  // StrSplit preserves empty entries, including an entirely empty PATH.
  for (std::string_view directory : absl::StrSplit(path, ':')) {
    std::string candidate = directory.empty()
                                ? absl::StrCat("./", executable)
                                : absl::StrCat(directory, "/", executable);
    int err = SpawnExecutable(pid, candidate.c_str(), argv, file_actions, envp);
    if (err == EACCES) {
      // Permission was denied for this candidate. A later PATH entry may still
      // work; remember this error so it takes precedence if the search fails.
      saw_eacces = true;
    } else if (err != ENOENT && err != ENOTDIR) {
      // Zero means a child was launched; other errors here end the search.
      // ENOENT (a missing file or path component) and ENOTDIR (a non-directory
      // path component) instead allow us to try the next PATH entry.
      return err;
    }
  }
  // Match execvp: report permission denied if any candidate was inaccessible,
  // otherwise report that no executable was found.
  return saw_eacces ? EACCES : ENOENT;
}

}  // namespace

absl::StatusOr<pid_t> SpawnSubprocess(
    absl::Span<const char* const> argv,
    const std::optional<std::filesystem::path>& cwd,
    posix_spawn_file_actions_t* file_actions, char* const* envp) {
  // Darwin can change the child directory without a helper executable.
  if (cwd.has_value()) {
    // This records the action; the directory change happens when spawning.
    // A nonzero result here means the action could not be added.
    if (int err = AddChdirFileAction(file_actions, cwd->c_str()); err != 0) {
      return absl::InternalError(absl::StrCat(
          "Cannot add child working directory action: ", Strerror(err)));
    }
  }
  pid_t pid;
  if (int err = SpawnWithPathSearch(&pid, argv, file_actions, envp); err != 0) {
    return absl::InternalError(
        absl::StrCat("Cannot spawn child process: ", Strerror(err)));
  }
  return pid;
}

}  // namespace xls::internal
