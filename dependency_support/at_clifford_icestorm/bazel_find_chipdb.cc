// Copyright 2020 Google LLC
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

// This file has a replacement for find_chipdb in iceutil.cc that works with
// Bazel's paths.

#include <unistd.h>

#include <string>

#include "absl/strings/str_cat.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;

std::string GetExePath() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  return std::string(result, (count > 0) ? count : 0);
}

}  // namespace

std::string find_chipdb(std::string config_device) {
  std::string error;
  auto runfiles =
      std::unique_ptr<Runfiles>(Runfiles::Create(GetExePath(), &error));
  if (runfiles == nullptr) {
    fprintf(stderr, "Failed to create Runfiles object: %s\n", error.c_str());
    abort();
  }
  return runfiles->Rlocation(
      absl::StrCat("at_clifford_icestorm/chipdb-", config_device, ".txt"));
}
