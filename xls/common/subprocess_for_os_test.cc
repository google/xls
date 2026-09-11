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

#include "xls/common/subprocess_for_os.h"

#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <cerrno>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

// Creates a command with controlled permissions. A non-executable script lets
// the PATH-search tests check that an inaccessible candidate is skipped.
absl::Status WriteScript(const std::filesystem::path& path,
                         std::string_view contents, bool executable = true) {
  absl::Status status = SetFileContents(path, contents);
  if (!status.ok()) {
    return status;
  }
  std::error_code error;
  std::filesystem::permissions(path,
                               executable
                                   ? std::filesystem::perms::owner_all
                                   : std::filesystem::perms::owner_read |
                                         std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::replace, error);
  if (error) {
    return absl::InternalError(error.message());
  }
  return absl::OkStatus();
}

// Exercises the OS launcher directly with a chosen child PATH and returns its
// normal exit code, including nonzero codes. Scripts use distinctive exit codes
// so assertions can identify which PATH candidate ran.
absl::StatusOr<int> SpawnAndWait(std::initializer_list<const char*> arguments,
                                 const std::filesystem::path& cwd,
                                 std::string path) {
  // posix_spawn requires null-terminated argv and envp arrays. Keep their
  // backing storage alive until SpawnSubprocess returns.
  std::vector<const char*> argv(arguments);
  argv.push_back(nullptr);

  // Build a minimal child environment with the requested PATH, independent of
  // the test runner's PATH. Keeping it in envp also avoids changing the
  // parent's process-wide environment, which other threads may use.
  std::string path_variable = absl::StrCat("PATH=", path);
  char* envp[] = {path_variable.data(), nullptr};

  // These tests observe the exit code, so file actions start empty. The
  // launcher may append a child chdir action. The caller owns and destroys the
  // actions even if spawning fails.
  posix_spawn_file_actions_t file_actions;
  if (posix_spawn_file_actions_init(&file_actions) != 0) {
    return absl::InternalError("Failed to initialize spawn file actions");
  }
  auto destroy_actions = absl::MakeCleanup(
      [&] { EXPECT_EQ(posix_spawn_file_actions_destroy(&file_actions), 0); });

  absl::StatusOr<pid_t> pid = internal::SpawnSubprocess(
      absl::MakeConstSpan(argv), cwd, &file_actions, envp);
  if (!pid.ok()) {
    return pid.status();
  }

  // A successful spawn transfers responsibility for reaping the child to us.
  // Retry interrupted waits so a signal cannot leave the child unreaped.
  int wait_status;
  pid_t waited;
  do {
    waited = waitpid(*pid, &wait_status, 0);
  } while (waited == -1 && errno == EINTR);
  if (waited != *pid) {
    return absl::InternalError("Failed to wait for spawned process");
  }
  if (!WIFEXITED(wait_status)) {
    return absl::InternalError("Spawned process did not exit normally");
  }
  return WEXITSTATUS(wait_status);
}

TEST(SubprocessForOsTest, DotAndEmptyPathEntriesUseChildWorkingDirectory) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory directory, TempDirectory::Create());
  XLS_ASSERT_OK(
      WriteScript(directory.path() / "command", "#!/bin/sh\nexit 23\n"));

  // Both "." and an empty PATH component mean the child's working directory.
  // This exercises the relative-PATH case that breaks Darwin's posix_spawnp
  // when combined with a chdir action.
  for (const std::string& path : {".", ":missing"}) {
    SCOPED_TRACE(path);
    XLS_ASSERT_OK_AND_ASSIGN(int exit_status,
                             SpawnAndWait({"command"}, directory.path(), path));
    EXPECT_EQ(exit_status, 23);
  }
}

TEST(SubprocessForOsTest, SearchesRelativePathEntriesFromChildDirectory) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory directory, TempDirectory::Create());
  XLS_ASSERT_OK(RecursivelyCreateDir(directory.path() / "tools"));
  XLS_ASSERT_OK(WriteScript(directory.path() / "tools" / "command",
                            "#!/bin/sh\nexit 29\n"));

  XLS_ASSERT_OK_AND_ASSIGN(
      int exit_status,
      SpawnAndWait({"command"}, directory.path(), "missing:tools"));
  EXPECT_EQ(exit_status, 29);
}

TEST(SubprocessForOsTest, SearchesPastInaccessiblePathEntry) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory directory, TempDirectory::Create());
  XLS_ASSERT_OK(RecursivelyCreateDir(directory.path() / "blocked"));
  XLS_ASSERT_OK(RecursivelyCreateDir(directory.path() / "tools"));
  XLS_ASSERT_OK(WriteScript(directory.path() / "blocked" / "command",
                            "#!/bin/sh\nexit 31\n", /*executable=*/false));
  XLS_ASSERT_OK(WriteScript(directory.path() / "tools" / "command",
                            "#!/bin/sh\nexit 37\n"));

  XLS_ASSERT_OK_AND_ASSIGN(
      int exit_status,
      SpawnAndWait({"command"}, directory.path(), "blocked:tools"));
  EXPECT_EQ(exit_status, 37);
}

TEST(SubprocessForOsTest, SlashInCommandBypassesPathSearch) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory directory, TempDirectory::Create());
  XLS_ASSERT_OK(RecursivelyCreateDir(directory.path() / "tools"));
  XLS_ASSERT_OK(RecursivelyCreateDir(directory.path() / "on_path"));
  XLS_ASSERT_OK(WriteScript(directory.path() / "tools" / "command",
                            "#!/bin/sh\nexit 41\n"));
  XLS_ASSERT_OK(WriteScript(directory.path() / "on_path" / "command",
                            "#!/bin/sh\nexit 43\n"));

  XLS_ASSERT_OK_AND_ASSIGN(
      int exit_status,
      SpawnAndWait({"./tools/command"}, directory.path(), "on_path"));
  EXPECT_EQ(exit_status, 41);
}

TEST(SubprocessForOsTest, ExecutableTextWithoutShebangUsesShell) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory directory, TempDirectory::Create());
  // execvp in the wrapper falls back to /bin/sh for executable text
  // without a shebang. The POSIX launcher must provide that fallback and
  // preserve args.
  XLS_ASSERT_OK(WriteScript(directory.path() / "command",
                            "test \"$1\" = expected || exit 47\nexit 53\n"));

  XLS_ASSERT_OK_AND_ASSIGN(
      int exit_status,
      SpawnAndWait({"command", "expected"}, directory.path(), "."));
  EXPECT_EQ(exit_status, 53);
}

}  // namespace
}  // namespace xls
