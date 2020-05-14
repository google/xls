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

#include "xls/common/file/get_runfile_path.h"

#include <cstdlib>

#include "xls/common/logging/logging.h"
#include "xls/common/module_initializer.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "absl/status/status.h"

namespace xls {
namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;

static Runfiles* runfiles;

// Returns true if the environment variables indicate that the program is run by
// the Bazel test runner. See
// https://docs.bazel.build/versions/master/test-encyclopedia.html
bool IsTest() {
  return std::getenv("TEST_SRCDIR") != nullptr &&
         std::getenv("TEST_TMPDIR") != nullptr;
}

XLS_REGISTER_MODULE_INITIALIZER(xls_runfiles_initializer, {
  // Because Runfiles::CreateForTest does not need the executable path (unlike
  // Runfiles::Create), it's possible to initialize runfiles statically in
  // tests. Doing that is good because it makes tests work even though they
  // don't call InitXls().
  if (IsTest()) {
    std::string error;
    runfiles = Runfiles::CreateForTest(&error);
    XLS_CHECK(runfiles != nullptr)
        << "Failed to initialize Runfiles: " << error;
  }
});

}  // namespace

std::filesystem::path GetXlsRunfilePath(const std::filesystem::path& path) {
  XLS_CHECK(runfiles != nullptr)
      << "GetXlsRunfilePath called before InitRunfilesDir()";
  return runfiles->Rlocation("com_google_xls" / path);
}

absl::Status InitRunfilesDir(const std::string& argv0) {
  if (runfiles != nullptr && !IsTest()) {
    // No need to initialize runfiles if it's already initialized or for tests
    // (that is handled by a static module initializer above).
    return absl::OkStatus();
  }

  std::string error;
  runfiles = Runfiles::Create(argv0, &error);
  if (runfiles == nullptr) {
    return absl::UnknownError(error);
  } else {
    return absl::OkStatus();
  }
}

}  // namespace xls
