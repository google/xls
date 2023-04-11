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
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>  // NOLINT
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/file/file_descriptor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strerror.h"
#include "xls/common/thread.h"

namespace xls {
namespace {

struct Pipe {
  Pipe(FileDescriptor exit, FileDescriptor&& entrance)
      : exit(std::move(exit)), entrance(std::move(entrance)) {}

  // Opens a Unix pipe with a C++ friendly interface.
  static absl::StatusOr<Pipe> Open() {
    int descriptors[2];
    if (pipe(descriptors) != 0 ||
        fcntl(descriptors[0], F_SETFD, FD_CLOEXEC) != 0||
        fcntl(descriptors[1], F_SETFD, FD_CLOEXEC) != 0) {
      return absl::InternalError(
          absl::StrCat("Failed to initialize pipe:", Strerror(errno)));
    }
    return Pipe(FileDescriptor(descriptors[0]), FileDescriptor(descriptors[1]));
  }

  FileDescriptor exit;
  FileDescriptor entrance;
};

void PrepareAndExecInChildProcess(const std::vector<const char*>& argv_pointers,
                                  std::optional<std::filesystem::path> cwd,
                                  const Pipe& stdout_pipe,
                                  const Pipe& stderr_pipe) {
  if (cwd.has_value()) {
    if (chdir(cwd->c_str()) != 0) {
      XLS_LOG(ERROR) << "chdir failed: " << Strerror(errno);
      _exit(127);
    }
  }

  while ((dup2(stdout_pipe.entrance.get(), STDOUT_FILENO) == -1) &&
         (errno == EINTR)) {
  }
  while ((dup2(stderr_pipe.entrance.get(), STDERR_FILENO) == -1) &&
         (errno == EINTR)) {
  }

  execv(argv_pointers[0], const_cast<char* const*>(argv_pointers.data()));
  XLS_LOG(ERROR) << "Execv syscall failed: " << Strerror(errno);
  _exit(127);
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
      XLS_CHECK(!(poll_list[i].revents & POLLNVAL));

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
    std::optional<absl::Duration> optional_timeout) {
  if (argv.empty()) {
    return absl::InvalidArgumentError("Cannot invoke empty argv list.");
  }
  std::string bin_name = std::filesystem::path(argv[0]).filename();

  XLS_VLOG(1) << absl::StreamFormat(
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

  pid_t pid = fork();
  if (pid == -1) {
    return absl::InternalError(
        absl::StrCat("Failed to fork: ", Strerror(errno)));
  }
  if (pid == 0) {
    PrepareAndExecInChildProcess(argv_pointers, cwd, stdout_pipe, stderr_pipe);
  }
  // This is the parent process.
  stdout_pipe.entrance.Close();
  stderr_pipe.entrance.Close();

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
              absl::Condition(condition_lambda, &release_watchdog),
              timeout)) {
        // Timeout has lapsed, try to kill the subprocess.
        timeout_expired.store(true);
        if (kill(pid, SIGKILL) == 0) {
          XLS_VLOG(1) << "Watchdog killed " << pid;
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
    XLS_VLOG(1) << "ReadFileDescriptors non-ok status: " << read_status;
  }
  const auto& stdout_output = output_strings[0];
  const auto& stderr_output = output_strings[1];

  XLS_VLOG_LINES(2, absl::StrCat(bin_name, " stdout:\n ", stdout_output));
  XLS_VLOG_LINES(2, absl::StrCat(bin_name, " stderr:\n ", stderr_output));

  XLS_ASSIGN_OR_RETURN(int wait_status, WaitForPid(pid));

  if (watchdog_thread != std::nullopt) {
    absl::MutexLock lock(&watchdog_mutex);
    release_watchdog = true;
  }

  return SubprocessResult{.stdout = stdout_output,
                          .stderr = stderr_output,
                          .exit_status = WEXITSTATUS(wait_status),
                          .normal_termination = WIFEXITED(wait_status),
                          .timeout_expired = timeout_expired.load()};
}

absl::StatusOr<std::pair<std::string, std::string>> SubprocessResultToStrings(
    absl::StatusOr<SubprocessResult> result) {
  if (result.ok()) {
    return std::make_pair(result->stdout, result->stderr);
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
      result_or_status->stdout, result_or_status->stderr));
}

std::ostream& operator<<(std::ostream& os, const SubprocessResult& other) {
  os << "exit_status:" << other.exit_status
     << " normal_termination:" << other.normal_termination
     << "\nstdout:" << other.stdout << "\nstderr:" << other.stderr << std::endl;
  return os;
}

}  // namespace xls
