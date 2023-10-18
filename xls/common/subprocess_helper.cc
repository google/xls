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

// Usage: subprocess_helper <desired working directory> <executable> [args...]
//
// Exec's the given executable in the specified working directory. If the first
// argument is an empty string, stays in the current working directory.

#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

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
  if (argc < 3) {
    std::cerr << "Usage:\n\tsubprocess_helper <desired working directory> "
                 "<executable> [args...]\n";
    return EXIT_FAILURE;
  }

  if (!std::string_view(argv[1]).empty()) {
    if (chdir(argv[1]) != 0) {
      int err = errno;
      std::cerr << "Subprocess helper failed to chdir to target \"" << argv[1]
                << "\": " << strerror(err) << "\n";
      PrintInvocation(argc, argv);
      return err;
    }
  }
  execvp(argv[2], &argv[2]);

  int err = errno;
  std::cerr << "Subprocess helper failed to exec: " << strerror(err) << "\n";
  PrintInvocation(argc, argv);
  return err;
}
