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

#include "absl/base/const_init.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/logging/logging.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/ret_check.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace xls {
namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;

static absl::Mutex mutex(absl::kConstInit);
static Runfiles* runfiles;

xabsl::StatusOr<Runfiles*> GetRunfiles(
    const std::string& argv0 = "/proc/self/exe") {
  absl::MutexLock lock(&mutex);
  if (runfiles == nullptr) {
    // Need to dereference the path, in case it's a link (as with the default).
    XLS_ASSIGN_OR_RETURN(auto path, GetRealPath(argv0));

    std::string error;
    runfiles = Runfiles::Create(path.string(), &error);
    XLS_RET_CHECK(runfiles != nullptr)
        << "Failed to initialize Runfiles: " << error;
  }

  return runfiles;
}

}  // namespace

std::filesystem::path GetXlsRunfilePath(const std::filesystem::path& path) {
  Runfiles* runfiles = GetRunfiles().value();
  return runfiles->Rlocation("com_google_xls" / path);
}

absl::Status InitRunfilesDir(const std::string& argv0) {
  return GetRunfiles(argv0).status();
}

}  // namespace xls
