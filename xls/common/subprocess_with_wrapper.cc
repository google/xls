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

#include <linux/memfd.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/file/file_descriptor.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strerror.h"
#include "xls/common/subprocess_for_os.h"
#include "xls/common/subprocess_helper_embedded_embedded.h"

namespace xls::internal {
namespace {

absl::StatusOr<FileDescriptor> GetSubprocessHelperFd() {
  FileDescriptor fd(memfd_create("subprocess_helper", MFD_CLOEXEC));
  if (fd.get() == -1) {
    return absl::InternalError(absl::StrCat(
        "Failed to create memfd for subprocess helper: ", Strerror(errno)));
  }
  if (write(fd.get(), get_subprocess_helper_embedded().data(),
            get_subprocess_helper_embedded().size()) !=
      get_subprocess_helper_embedded().size()) {
    return absl::InternalError(absl::StrCat(
        "Failed to write subprocess helper to memfd: ", Strerror(errno)));
  }
  if (lseek(fd.get(), 0, SEEK_SET) != 0) {
    return absl::InternalError(absl::StrCat(
        "Failed to seek subprocess helper in memfd: ", Strerror(errno)));
  }
  return std::move(fd);
}

}  // namespace

absl::StatusOr<pid_t> SpawnSubprocess(
    absl::Span<const char* const> argv,
    const std::optional<std::filesystem::path>& cwd,
    posix_spawn_file_actions_t* file_actions, char* const* envp) {
  // The helper changes directory and calls execvp, supporting Linux toolchains
  // whose libc lacks a posix_spawn chdir action. Run it out of a memfd so
  // subprocesses do not depend on Bazel build artifacts remaining on disk.
  static const absl::StatusOr<FileDescriptor> subprocess_helper_fd =
      GetSubprocessHelperFd();
  XLS_RETURN_IF_ERROR(subprocess_helper_fd.status());
  std::string subprocess_helper =
      absl::StrCat("/proc/self/fd/", subprocess_helper_fd->get());
  std::vector<const char*> helper_argv;
  helper_argv.reserve(argv.size() + 2);
  helper_argv.push_back(subprocess_helper.c_str());
  helper_argv.push_back(cwd.has_value() ? cwd->c_str() : "");
  helper_argv.insert(helper_argv.end(), argv.begin(), argv.end());

  pid_t pid;
  if (int err =
          posix_spawn(&pid, subprocess_helper.c_str(), file_actions, nullptr,
                      const_cast<char* const*>(helper_argv.data()), envp);
      err != 0) {
    return absl::InternalError(
        absl::StrCat("Cannot spawn child process: ", Strerror(err)));
  }
  return pid;
}

}  // namespace xls::internal
