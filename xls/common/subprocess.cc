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

#include "xls/common/subprocess.h"

#include <fcntl.h>
#include <signal.h>  // NOLINT
#include <spawn.h>
#include <stdlib.h>  // NOLINT for WIFEXITED, WEXITSTATUS; not in <cstdlib>
#include <sys/poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/file/file_descriptor.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strerror.h"
#include "xls/common/thread.h"

#if defined(__APPLE__)
extern char** environ;
#endif

namespace xls {
namespace {

// Note: this path is runfiles-relative.
constexpr std::string_view kSubprocessHelperPath =
    "xls/common/subprocess_helper";

struct Pipe {
  Pipe(FileDescriptor exit, FileDescriptor&& entrance)
      : exit(std::move(exit)), entrance(std::move(entrance)) {}

  // Opens a Unix pipe with a C++ friendly interface.
  static absl::StatusOr<Pipe> Open() {
    int descriptors[2];
    if (pipe(descriptors) != 0 ||
        fcntl(descriptors[0], F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(descriptors[1], F_SETFD, FD_CLOEXEC) != 0) {
      return absl::InternalError(
          absl::StrCat("Failed to initialize pipe:", Strerror(errno)));
    }
    return Pipe(FileDescriptor(descriptors[0]), FileDescriptor(descriptors[1]));
  }

  FileDescriptor exit;
  FileDescriptor entrance;
};

absl::Status ReplaceFdWithPipe(posix_spawn_file_actions_t& actions, int fd,
                               Pipe& pipe, std::string_view pipe_name) {
  if (int err = posix_spawn_file_actions_addclose(&actions, pipe.exit.get());
      err != 0) {
    return absl::InternalError(absl::StrFormat(
        "Cannot add close() action for %s exit: %s", pipe_name, Strerror(err)));
  }

  if (int err =
          posix_spawn_file_actions_adddup2(&actions, pipe.entrance.get(), fd);
      err != 0) {
    return absl::InternalError(
        absl::StrFormat("Cannot add dup2() action for %s entrance: %s",
                        pipe_name, Strerror(err)));
  }
  if (int err =
          posix_spawn_file_actions_addclose(&actions, pipe.entrance.get());
      err != 0) {
    return absl::InternalError(absl::StrFormat(
        "Cannot clean up for %s entrance: %s", pipe_name, Strerror(err)));
  }
  return absl::OkStatus();
}

absl::StatusOr<posix_spawn_file_actions_t> CreateChildFileActions(
    Pipe& stdout_pipe, Pipe& stderr_pipe) {
  posix_spawn_file_actions_t actions;

  if (int err = posix_spawn_file_actions_init(&actions); err != 0) {
    return absl::InternalError(
        absl::StrCat("Cannot initialize file actions: ", Strerror(err)));
  }
  if (int err = posix_spawn_file_actions_addclose(&actions, STDIN_FILENO);
      err != 0) {
    return absl::InternalError(
        absl::StrCat("Cannot add close() action (stdin): ", Strerror(err)));
  }

  XLS_RETURN_IF_ERROR(
      ReplaceFdWithPipe(actions, STDOUT_FILENO, stdout_pipe, "stdout"));
  XLS_RETURN_IF_ERROR(
      ReplaceFdWithPipe(actions, STDERR_FILENO, stderr_pipe, "stderr"));

  return actions;
}

absl::StatusOr<pid_t> ExecInChildProcess(
    const std::vector<const char*>& argv_pointers,
    const std::optional<std::filesystem::path>& cwd, Pipe& stdout_pipe,
    Pipe& stderr_pipe,
    absl::Span<const EnvironmentVariable> environment_variables) {
  // We previously used fork() & exec() here, but that's prone to many subtle
  // problems (e.g., allocating between fork() and exec() can cause arbitrary
  // problems)... and it's also slow. vfork() might have made the performance
  // better, but it's not fully clear what's safe between vfork() and exec()
  // either, so we just use posix_spawn for safety and convenience.

  // Since we may need the child to have a different working directory (per
  // `cwd`), and posix_spawn does not (yet) have support for a chdir action, we
  // use a helper binary that chdir's to its first argument, then invokes
  // "execvp" with the remaining arguments to replace itself with the command we
  // actually wanted to run.
  XLS_ASSIGN_OR_RETURN(std::filesystem::path subprocess_helper,
                       GetXlsRunfilePath(kSubprocessHelperPath));
  std::vector<const char*> helper_argv_pointers;
  helper_argv_pointers.reserve(argv_pointers.size() + 2);
  helper_argv_pointers.push_back(subprocess_helper.c_str());
  helper_argv_pointers.push_back(cwd.has_value() ? cwd->c_str() : "");
  helper_argv_pointers.insert(helper_argv_pointers.end(), argv_pointers.begin(),
                              argv_pointers.end());

  XLS_ASSIGN_OR_RETURN(posix_spawn_file_actions_t file_actions,
                       CreateChildFileActions(stdout_pipe, stderr_pipe));

  // posix_spawnp takes a null-terminate array of char* for environment
  // variables. Each element has the form "NAME=VALUE".
  std::vector<std::string> env_vars;
  std::vector<char*> env_var_ptrs;
  char** child_env;
  if (environment_variables.empty()) {
    // No extra environment variables are specified. Use the existing
    // environment.
    child_env = environ;
  } else {
    // Append environment variables to the existing environment.
    for (int i = 0; environ[i] != nullptr; ++i) {
      env_vars.push_back(environ[i]);
    }
    for (const EnvironmentVariable& var : environment_variables) {
      env_vars.push_back(absl::StrCat(var.name, "=", var.value));
    }
    for (const std::string& s : env_vars) {
      env_var_ptrs.push_back(const_cast<char*>(s.data()));
    }
    env_var_ptrs.push_back(nullptr);
    child_env = env_var_ptrs.data();
  }
  pid_t pid;
  if (int err = posix_spawnp(
          &pid, subprocess_helper.c_str(), &file_actions, nullptr,
          const_cast<char* const*>(helper_argv_pointers.data()), child_env);
      err != 0) {
    return absl::InternalError(
        absl::StrCat("Cannot spawn child process: ", Strerror(err)));
  }

  if (int err = posix_spawn_file_actions_destroy(&file_actions); err != 0) {
    return absl::InternalError(
        absl::StrCat("Cannot destroy file actions: ", Strerror(err)));
  }
  stdout_pipe.entrance.Close();
  stderr_pipe.entrance.Close();
  return pid;
}

// Takes a list of file descriptor data streams and reads them into a list of
// strings, one for each provided file descriptor.
//
// The 'result' vector is populated with all the data that was read in from the
// fd's regardless to the status that is returned. Non-OK status will still
// populate what it can into 'result'.
absl::Status ReadFileDescriptors(absl::Span<FileDescriptor*> fds,
                                 std::vector<std::string>& result) {
  absl::FixedArray<char> buffer(4096);
  result.resize(fds.size());
  std::vector<pollfd> poll_list;
  poll_list.resize(fds.size());
  for (int i = 0; i < fds.size(); i++) {
    poll_list[i].fd = fds[i]->get();
    poll_list[i].events = POLLIN;
  }
  size_t descriptors_left = fds.size();

  auto close_fd_by_index = [&](int idx) {
    poll_list[idx].fd = -1;
    descriptors_left--;
    fds[idx]->Close();
  };

  while (descriptors_left > 0) {
    int data_count = poll(poll_list.data(), poll_list.size(), -1);
    if (data_count <= 0) {
      if (errno == EINTR) {
        continue;
      }
      return absl::InternalError(absl::StrCat("poll failed:", Strerror(errno)));
    }

    for (int i = 0; i < fds.size(); i++) {
      if (poll_list[i].revents & POLLERR) {
        // Unspecified error.
        return absl::InternalError("Subprocess poll failed.");
      }

      // This should "never" happen. If it does, someone has e.g. closed our fd.
      CHECK(!(poll_list[i].revents & POLLNVAL));

      // If poll_list[i].revents & POLLHUP, the remote side closed its
      // connection, but there may be data waiting to be read. read() will
      // return 0 bytes when we consume all the data, so just ignore that error.
      if ((poll_list[i].revents & (POLLHUP | POLLIN)) != 0) {
        size_t bytes = read(poll_list[i].fd, buffer.data(), buffer.size());
        if (bytes == 0) {
          // All data is read.
          close_fd_by_index(i);
        } else if (bytes > 0) {
          result[i].append(buffer.data(), bytes);
        } else if (errno != EINTR) {
          close_fd_by_index(i);
        }
      }
    }
  }

  return absl::OkStatus();
}

// Waits for a process to finish. Returns the wait_status.
absl::StatusOr<int> WaitForPid(pid_t pid) {
  int wait_status;
  while (waitpid(pid, &wait_status, 0) == -1) {
    if (errno == EINTR) {
      continue;
    }
    return absl::InternalError(
        absl::StrCat("waitpid failed: ", Strerror(errno)));
  }
  return wait_status;
}

}  // namespace

absl::StatusOr<SubprocessResult> InvokeSubprocess(
    absl::Span<const std::string> argv,
    std::optional<std::filesystem::path> cwd,
    std::optional<absl::Duration> optional_timeout,
    absl::Span<const EnvironmentVariable> environment_variables) {
  if (argv.empty()) {
    return absl::InvalidArgumentError("Cannot invoke empty argv list.");
  }
  std::string bin_name = std::filesystem::path(argv[0]).filename();

  VLOG(1) << absl::StreamFormat(
      "Running %s; argv: [ %s ], cwd: %s", bin_name, absl::StrJoin(argv, " "),
      cwd.has_value() ? cwd->string()
                      : std::filesystem::current_path().string());

  std::vector<const char*> argv_pointers;
  argv_pointers.reserve(argv.size() + 1);
  for (const auto& arg : argv) {
    argv_pointers.push_back(arg.c_str());
  }
  argv_pointers.push_back(nullptr);

  XLS_ASSIGN_OR_RETURN(auto stdout_pipe, Pipe::Open());
  XLS_ASSIGN_OR_RETURN(auto stderr_pipe, Pipe::Open());

  XLS_ASSIGN_OR_RETURN(pid_t pid,
                       ExecInChildProcess(argv_pointers, cwd, stdout_pipe,
                                          stderr_pipe, environment_variables));

  // Order is important here. The optional<Thread> must appear after the mutex
  // because the thread's destructor calls Join() and because the thread has
  // references to the mutex. We are depending on destructor invocation being
  // the reverse order of construction.
  //
  // Note that release_watchdog is the Condition trigger protected by the mutex
  // and signaling that the process has finished and the watchdog should exit.
  std::atomic<bool> timeout_expired = false;
  bool release_watchdog = false;
  absl::Mutex watchdog_mutex;
  std::optional<xls::Thread> watchdog_thread;
  if (optional_timeout.has_value() &&
      *optional_timeout > absl::ZeroDuration()) {
    auto watchdog = [pid, timeout = optional_timeout.value(), &watchdog_mutex,
                     &release_watchdog, &timeout_expired]() {
      absl::MutexLock lock(&watchdog_mutex);
      auto condition_lambda = [](void* release_val) {
        return *static_cast<bool*>(release_val);
      };
      if (!watchdog_mutex.AwaitWithTimeout(
              absl::Condition(condition_lambda, &release_watchdog), timeout)) {
        // Timeout has lapsed, try to kill the subprocess.
        timeout_expired.store(true);
        if (kill(pid, SIGKILL) == 0) {
          VLOG(1) << "Watchdog killed " << pid;
        }
      }
    };
    watchdog_thread.emplace(watchdog);
  }

  // Read from the output streams of the subprocess.
  FileDescriptor* fds[] = {&stdout_pipe.exit, &stderr_pipe.exit};
  std::vector<std::string> output_strings;
  absl::Status read_status = ReadFileDescriptors(fds, output_strings);
  if (!read_status.ok()) {
    VLOG(1) << "ReadFileDescriptors non-ok status: " << read_status;
  }
  const std::string& stdout_output = output_strings[0];
  const std::string& stderr_output = output_strings[1];

  XLS_VLOG_LINES(2, absl::StrCat(bin_name, " stdout:\n ", stdout_output));
  XLS_VLOG_LINES(2, absl::StrCat(bin_name, " stderr:\n ", stderr_output));

  XLS_ASSIGN_OR_RETURN(int wait_status, WaitForPid(pid));

  if (watchdog_thread != std::nullopt) {
    absl::MutexLock lock(&watchdog_mutex);
    release_watchdog = true;
  }

  return SubprocessResult{.stdout_content = stdout_output,
                          .stderr_content = stderr_output,
                          .exit_status = WEXITSTATUS(wait_status),
                          .normal_termination = WIFEXITED(wait_status),
                          .timeout_expired = timeout_expired.load()};
}

absl::StatusOr<std::pair<std::string, std::string>> SubprocessResultToStrings(
    absl::StatusOr<SubprocessResult> result) {
  if (result.ok()) {
    return std::make_pair(result->stdout_content, result->stderr_content);
  }
  return result.status();
}

absl::StatusOr<SubprocessResult> SubprocessErrorAsStatus(
    absl::StatusOr<SubprocessResult> result_or_status) {
  if (!result_or_status.ok() || (result_or_status->exit_status == 0 &&
                                 result_or_status->normal_termination)) {
    return result_or_status;
  }

  return absl::InternalError(absl::StrFormat(
      "Subprocess exit_code: %d normal_termination: %d stdout: %s stderr: %s",
      result_or_status->exit_status, result_or_status->normal_termination,
      result_or_status->stdout_content, result_or_status->stderr_content));
}

std::ostream& operator<<(std::ostream& os, const SubprocessResult& other) {
  os << "exit_status:" << other.exit_status
     << " normal_termination:" << other.normal_termination
     << "\nstdout:" << other.stdout_content
     << "\nstderr:" << other.stderr_content << "\n";
  return os;
}

}  // namespace xls
