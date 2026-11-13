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

// Usage: subprocess_helper [--[no]close_extra_fds] [--] <desired working
// directory> <executable> [args...]
//
// Exec's the given executable in the specified working directory. If the
// working directory argument is an empty string, stays in the current working
// directory.

#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <system_error>  // NOLINT
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"

ABSL_FLAG(bool, close_extra_fds, true,
          "Close inherited file descriptors >= 3 before execing the child.");

static void CloseExtraFileDescriptors() {
  std::error_code ec;
  // Collect descriptors before closing so we do not close the
  // directory_iterator's own underlying descriptor while iterating /dev/fd.
  std::vector<int> fds_to_close;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator("/dev/fd", ec)) {
    int fd = 0;
    if (absl::SimpleAtoi(entry.path().filename().c_str(), &fd) && fd >= 3) {
      fds_to_close.push_back(fd);
    }
  }
  for (int fd : fds_to_close) {
    close(fd);
  }
}

static void PrintInvocation(int argc, char** argv) {
  std::cerr << "Subprocess helper was invoked as:\n\t";
  for (int i = 0; i < argc; ++i) {
    if (i > 0) {
      std::cerr << ' ';
    }
    std::cerr << '"' << argv[i] << '"';
  }
  std::cerr << "\n";
}

int main(int argc, char** argv) {
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);
  if (positional_args.size() < 3) {
    std::cerr << "Usage:\n\tsubprocess_helper [--[no]close_extra_fds] [--] "
                 "<desired working directory> <executable> [args...]\n";
    return EXIT_FAILURE;
  }

  const char* cwd = positional_args[1];
  std::vector<char*> exec_argv(positional_args.begin() + 2,
                               positional_args.end());
  exec_argv.push_back(nullptr);
  if (!std::string_view(cwd).empty()) {
    if (chdir(cwd) != 0) {
      int err = errno;
      std::cerr << "Subprocess helper failed to chdir to target \"" << cwd
                << "\": " << strerror(err) << "\n";
      PrintInvocation(argc, argv);
      return err;
    }
  }
  if (absl::GetFlag(FLAGS_close_extra_fds)) {
    CloseExtraFileDescriptors();
  }
  execvp(exec_argv[0], exec_argv.data());

  int err = errno;
  std::cerr << "Subprocess helper failed to exec: " << strerror(err) << "\n";
  PrintInvocation(argc, argv);
  return err;
}
